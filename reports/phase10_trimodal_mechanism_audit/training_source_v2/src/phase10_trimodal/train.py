"""Full-partition risk sets and split-local model selection/refit."""
import hashlib,json,time,copy
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from .features import prepare,derangement,ahash
from .math import fit_ridge,log_weights,torch_cox,comparable_pairs
from .reference import components
from .models import MechanismModel
from .data import write_json,digest


def inner_split(d,indices,seed):
    ix=np.asarray(indices);strata=d['DSS_event'][ix].astype(int).astype(str)+d['cancer_type'][ix].astype(str)
    a,b=next(StratifiedShuffleSplit(1,test_size=.2,random_state=seed).split(ix,strata))
    it,iv=ix[a],ix[b]
    if d['DSS_event'][it].sum()==0 or d['DSS_event'][iv].sum()==0:raise ValueError('zero-event inner split')
    return it,iv

def ridge_x(f,arm):
    mapping={'A0':['C'],'A0_min':['C_min'],'A1':['R'],'A3':['C','S'],'A4':['C','T'],'A7':['C','S','R'],'A8':['C','S','R','CNA']}
    return np.concatenate([f[k] for k in mapping[arm]],axis=1).astype(float)

def choose_ridge(x,t,e,it,iv,grid):
    rows=[];best=None
    for lam in grid:
        w,info=fit_ridge(x[it],t[it],e[it],lam);n,den=components(t[iv],e[iv],x[iv]@w)
        score=n/den;rows.append({'lambda':lam,'inner_Harrell':score,**info})
        if best is None or score>best[0]+1e-12:best=(score,lam)
    return best[1],rows

def teacher_crossfit(d,fit,groups,cfg):
    fit=np.asarray(fit);t=d['DSS_time'];e=d['DSS_event'];out=np.full(len(t),np.nan);audit=[]
    for sub,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=cfg['split_seed']).split(fit,e[fit])):
        train,hold=fit[a],fit[b];it,iv=inner_split(d,train,cfg['split_seed']+sub+99)
        f_i,ai=prepare(d,it,groups);x_i=np.c_[f_i['C'],f_i['S']]
        lam,selection=choose_ridge(x_i,t,e,it,iv,cfg['ridge_grid'])
        f_o,ao=prepare(d,train,groups);x_o=np.c_[f_o['C'],f_o['S']];w,info=fit_ridge(x_o[train],t[train],e[train],lam);out[hold]=x_o[hold]@w
        audit.append({'fit_indices':train.tolist(),'selection_fit':it.tolist(),'selection_validation':iv.tolist(),'heldout_indices':hold.tolist(),'lambda':lam,'preprocessing':ao,'selection':selection})
        if set(hold)&(set(it)|set(iv)|set(train)):raise ValueError('teacher heldout leak')
    if not np.isfinite(out[fit]).all():raise ValueError('teacher coverage')
    return out,audit

def torch_inputs(f,patch,mask,device):
    # tensor() copies read-only memmap storage; training cannot mutate raw cache.
    return ({k:torch.as_tensor(v,device=device) for k,v in f.items()},torch.tensor(patch,device=device),torch.as_tensor(mask,device=device))

def rkey(arm):return 'G_random' if arm=='C5' else ('G' if arm.startswith('C') else 'R_tokens')

def inputs(f,p,pm,idx,arm,d,seed):
    idx=np.asarray(idx);c=f['C'][idx]
    if arm=='A2':c=torch.zeros_like(c)
    ri=derangement(d,idx,seed) if arm in ['B5','C3'] else idx
    return c,f[rkey(arm)][ri],p[idx],pm[idx]

def grad_summary(model):
    out={}
    for key,part in [('C',model.clinical_projection),('R',model.rna_projection),('P',model.patch_projection),('head',model.head),('gate',model.ifa)]:
        if part is None:continue
        vs=[p.grad.detach().flatten() for p in part.parameters() if p.grad is not None];out[key]=float(torch.cat(vs).norm()) if vs else 0.
    axis={}
    for k,layer in [('C',model.clinical_projection),('R',model.rna_projection[0][0]),('P',model.patch_projection[1])]:
        if layer.weight.grad is not None:
            v=layer.weight.grad.mean(1)
            if k=='C':v=v.reshape(6,-1).mean(0)
            axis[k]=v
    for a,b in [('C','R'),('C','P'),('R','P')]:
        if a in axis and b in axis:out[a+'_'+b+'_projection_axis_cosine']=float(torch.nn.functional.cosine_similarity(axis[a],axis[b],dim=0))
    return out

def train_neural(d,f,patch,mask,fit,val,arm,seed,cfg,clinical_lam,teacher=None,epochs=None,checkpoint=None):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(seed);np.random.seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True,warn_only=False)
    model=MechanismModel('B0' if arm.startswith('D') else arm,f['C'].shape[1],d=cfg['hidden_dim']).to(device)
    if arm!='A2':
        w,ci=fit_ridge(f['C'][fit].astype(float),d['DSS_time'][fit],d['DSS_event'][fit],clinical_lam)
        with torch.no_grad():model.clinical_skip.weight.copy_(torch.as_tensor(w,device=device,dtype=torch.float32)[None,:])
    tensors,p,pm=torch_inputs(f,patch,mask,device);e=torch.tensor(d['DSS_event'][fit],device=device);loga=log_weights(d['DSS_time'][fit],d['DSS_event'][fit],device)
    vin=inputs(tensors,p,pm,val,arm,d,cfg['split_seed']+202) if len(val) else None
    tin=inputs(tensors,p,pm,fit,arm,d,cfg['split_seed']+101)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);logs=[];best=None;ema=None;start=time.monotonic()
    if arm in ['D2','D3']:
        pi,pj=comparable_pairs(d['DSS_time'][fit],d['DSS_event'][fit]);pi=torch.tensor(pi,device=device);pj=torch.tensor(pj,device=device)
        target=torch.sigmoid(torch.as_tensor(teacher[fit],device=device)[pi]-torch.as_tensor(teacher[fit],device=device)[pj]).float()
    n_epochs=epochs or cfg['epochs']
    model.eval()
    with torch.no_grad():initial_reps={k:v.cpu().numpy() for k,v in model(*tin,return_details=True)[1].items()}
    for epoch in range(1,n_epochs+1):
        model.train();opt.zero_grad(set_to_none=True);risk,reps,aux,gate=model(*tin,return_details=True);main=torch_cox(risk,loga,e);loss=main
        auxvals={k:torch_cox(v,loga,e) for k,v in aux.items()}
        sel=pal=torch.zeros((),device=device)
        if arm in ['D1','D3']:
            av=torch.stack(list(auxvals.values()));bal=torch.softmax(av.detach(),0);sel=(bal*av).sum();loss=loss+.1*sel
        if arm in ['D2','D3']:
            logits=aux['P'][pi]-aux['P'][pj];pal=nn_bce(logits,target);loss=loss+.1*pal
        loss.backward();gs=grad_summary(model)
        participating=sum(p.numel() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);opt.step();model.eval()
        with torch.no_grad():
            train_risk,_,_,gate=model(*tin,return_details=True)
            trainloss=float(torch_cox(train_risk,loga,e));vscore=vloss=None
            if len(val):
                vr=model(*vin).cpu().numpy();vloss=float(torch_cox(torch.tensor(vr,device=device),log_weights(d['DSS_time'][val],d['DSS_event'][val],device),torch.tensor(d['DSS_event'][val],device=device)))
                num,den=components(d['DSS_time'][val],d['DSS_event'][val],vr);vscore=num/den
                ema=vloss if ema is None else .3*vloss+.7*ema
                if epoch>=cfg['minimum_selection_epoch'] and (best is None or ema<best[0]):best=(ema,epoch,vscore)
        with torch.no_grad():correction=train_risk-(model.clinical_skip(tin[0]).squeeze(1) if arm!='A2' else 0)
        row={'epoch':epoch,'train_cox_before_step':float(main.detach()),'train_cox_after_step':trainloss,'inner_val_loss':vloss,'inner_val_loss_EMA':ema,'inner_val_Harrell':vscore,'SEL':float(sel.detach()),'PAL':float(pal.detach()),'gradients':gs,'risk_sd':float(train_risk.std()),'correction_sd':float(correction.std()),'correction_max_abs':float(correction.abs().max()),'gate_mean':gate.mean(0).tolist(),'gate_sd':gate.std(0).tolist(),'participating_parameters':participating};logs.append(row)
        if not np.isfinite(trainloss) or not all(np.isfinite(v) for v in gs.values()):raise RuntimeError('nonfinite training')
    selected=best[1] if len(val) else n_epochs
    model.eval()
    with torch.no_grad():
        # Callers use final refit only for final test prediction. Inner model
        # needs no public OOF; selected metrics are from the selected epoch log.
        final_reps={k:v.cpu().numpy() for k,v in model(*tin,return_details=True)[1].items()}
    if checkpoint:torch.save({'state':model.state_dict(),'arm':arm,'seed':seed,'epochs':n_epochs,'fit_hash':ahash(np.asarray(fit)),'protocol':cfg},checkpoint)
    info={'total_parameters':sum(p.numel() for p in model.parameters()),'selected_epoch':selected,'actual_epochs':n_epochs,'selected_inner_Harrell':best[2] if best else None,'selected_inner_EMA':best[0] if best else None,'training_logs':logs,'seconds':time.monotonic()-start,'peak_cuda_MB':torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0.,'boundary_selected':selected==n_epochs,'device':device}
    return model,info,(initial_reps,final_reps)

def nn_bce(logits,target):return torch.nn.functional.binary_cross_entropy_with_logits(logits,target)

def predict_neural(model,d,f,patch,mask,test,arm,cfg):
    device=next(model.parameters()).device;tensors,p,pm=torch_inputs(f,patch,mask,device)
    model.eval()
    with torch.no_grad():risk,reps,aux,gate=model(*inputs(tensors,p,pm,test,arm,d,cfg['split_seed']+303),return_details=True)
    return risk.cpu().numpy(),{k:v.cpu().numpy() for k,v in reps.items()}, {k:v.cpu().numpy() for k,v in aux.items()}

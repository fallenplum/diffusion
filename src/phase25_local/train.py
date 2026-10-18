"""Complete-risk residual fitting on strict crossfitted R0 offsets only."""
from __future__ import annotations
import random
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from src.phase17_ssd.losses import full_risk_efron
from .model import LocalRouting
from .runtime import cfg, NEURAL_ARMS


def seed_all(seed,fold):
    value=int(seed)*1009+int(fold)
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(value)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def _forward(model,ds,patients,*,training=False,intervention=None,details=False):
    patients=np.asarray(patients,int);risks=[];units=[];collected=[]
    for start in range(0,len(patients),32):
        current=patients[start:start+32]
        slides=np.flatnonzero(np.isin(ds.owner,current));owners=np.asarray(ds.owner)[slides]
        kwargs={}
        if intervention:
            if not np.isin(owners,ds.valid).all():raise ValueError('frozen interventions are heldout-only')
            if intervention=='RNA_MEAN':kwargs['query_rna']=ds.R[np.asarray(ds.fit)].mean(0).expand(len(owners),-1)
            elif intervention=='WRONG_RNA':kwargs['query_rna']=ds.R[np.asarray(ds.wrong_rna)[owners]]
            elif intervention=='C_MEAN':kwargs['query_clinical']=ds.C[np.asarray(ds.fit)].mean(0).expand(len(owners),-1)
            elif intervention=='WRONG_C':kwargs['query_clinical']=ds.C[np.asarray(ds.wrong_c)[owners]]
            elif intervention!='OFF':raise ValueError('unknown intervention')
        if intervention=='OFF':
            wsi=ds.base.new_zeros(len(owners))
        else:
            args=(ds.visual[slides],ds.mask[slides],ds.R[owners],ds.C[owners])
            def call(*x, options=dict(kwargs)):
                return model(*x,**options)['unit']
            if training:wsi=checkpoint(call,*args,use_reentrant=False,preserve_rng_state=True)
            else:
                out=model(*args,**kwargs,return_details=details);wsi=out['unit']
                if details:collected.append({'output':out,'owner':owners.copy(),'mask':ds.mask[slides]})
        unit=torch.stack([wsi[torch.as_tensor(owners==p,device=wsi.device)].double().mean() for p in current])
        base=ds.base[current].detach().double()
        units.append(unit);risks.append(base+float(ds.scale)*unit)
    return torch.cat(risks),torch.cat(units),collected


def gradient_groups(model):
    result={k:0. for k in ('keys','values','head','q0','rna','clinical')}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        g=p.grad.detach().double()
        if not torch.isfinite(g).all():raise ValueError('nonfinite gradient')
        key=name.split('.')[0];result[key]+=float(g.square().sum())
    return {k:v**.5 for k,v in result.items()}


def train_arm(ds,arm,seed,fold,epochs=80,step_callback=None):
    if arm not in NEURAL_ARMS or not 1<=int(epochs)<=80 or int(epochs)!=epochs:raise ValueError('invalid arm/budget')
    seed_all(seed,fold)
    model=LocalRouting(ds.C.shape[1],arm,16).to(ds.visual.device)
    fit=np.array(ds.fit,copy=True);t,e=ds.fit_labels()
    if len(t)!=len(fit) or not np.isfinite(t).all() or not np.isin(e,[0,1]).all() or not np.sum(e):raise ValueError('invalid fit-only supervision')
    # The inherited complete-risk loss constructs Efron weights with NumPy
    # before moving them to risk.device. Keep these fit-only labels on CPU.
    t=np.asarray(t,dtype=np.float64);e=np.asarray(e,dtype=np.float64)
    base_snapshot=ds.base.detach().clone()
    if not np.isfinite(ds.scale) or ds.scale<=0:raise ValueError('invalid fit-only scale')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    history=[]
    for epoch in range(1,int(epochs)+1):
        started=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
        risk,u,_=_forward(model,ds,fit,training=True)
        cox=full_risk_efron(risk,t,e);penalty=.5*u.square().mean();loss=cox+penalty
        if not torch.isfinite(loss):raise ValueError('nonfinite complete objective')
        loss.backward();groups=gradient_groups(model)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        opt.step();model.eval()
        if any(not torch.isfinite(v).all() for v in model.parameters()):raise ValueError('nonfinite updated parameters')
        if not torch.equal(ds.base,base_snapshot):raise ValueError('frozen main offset changed')
        with torch.no_grad():
            risk,u,_=_forward(model,ds,fit)
            fitted=full_risk_efron(risk,t,e);objective=fitted+.5*u.square().mean()
        history.append({'epoch':epoch,'training_loss':float(loss.detach()),'fit_loss':float(objective),
            'fit_cox':float(fitted),'unit_residual_RMS':float(u.square().mean().sqrt()),
            'grad_norm':float(norm),'gradients':groups,'epoch_seconds':time.monotonic()-started})
        if step_callback:step_callback(arm,epoch)
    model.eval()
    audit={'actual_epochs':int(epochs),'optimizer_steps':int(epochs),'heldout_endpoint_reads':0,
        'fixed_final_epoch':True,'main_offset_unchanged':True,'fit_offset_kind':'strict_threefold_crossfit_R0',
        'heldout_offset_kind':'full_fit_R0','conditional_heads_no_RNA_Clinical_direct_risk_bypass':True,
        'teacher_error_pairs_used':False,'global_convergence_established':False}
    return model,initial,history,audit


def _moments(a):
    a=np.asarray(a,float).reshape(-1)
    if not len(a) or not np.isfinite(a).all():raise ValueError('invalid aggregate')
    return {'count':len(a),'mean':float(a.mean()),'population_SD':float(a.std()),
            'RMS':float(np.sqrt(np.mean(a*a))),'minimum':float(a.min()),'maximum':float(a.max())}


def attention_stats(details):
    rows=[]
    for item in details:
        a=item['output']['attention'].detach().cpu().numpy();m=item['mask'].cpu().numpy()
        for w,keep in zip(a,m):
            w=w[keep].astype(float);n=len(w);h=-np.sum(w*np.log(np.maximum(w,1e-30)))
            rows.append((h/np.log(n) if n>1 else 1.,np.exp(h)/n,.5*np.abs(w-1/n).sum()))
    x=np.asarray(rows)
    return {k:_moments(x[:,i]) for i,k in enumerate(('normalized_entropy','effective_patch_fraction','TV_from_uniform'))}


def attention_difference(matched,other):
    l1=[];overlap=[]
    if len(matched)!=len(other):raise ValueError('intervention WSI chunks differ')
    for a,b in zip(matched,other):
        if not np.array_equal(a['owner'],b['owner']) or not torch.equal(a['mask'],b['mask']):raise ValueError('same-query control changed patch input')
        x=a['output']['attention'].detach().cpu().numpy();y=b['output']['attention'].detach().cpu().numpy();mask=a['mask'].cpu().numpy()
        for aa,bb,keep in zip(x,y,mask):
            aa=aa[keep];bb=bb[keep];k=min(32,len(aa))
            ai=set(np.argsort(-aa,kind='stable')[:k]);bi=set(np.argsort(-bb,kind='stable')[:k])
            l1.append(float(.5*np.abs(aa-bb).sum()));overlap.append(len(ai&bi)/len(ai|bi))
    return {'attention_TV':_moments(l1),'top32_Jaccard':_moments(overlap),
            'scope':'WSI_weighted_same_valid_patches_not_histological_ground_truth'}


@torch.no_grad()
def predict_arm(model,ds):
    model.eval();fit,uf,_=_forward(model,ds,ds.fit)
    risk,u,detail=_forward(model,ds,ds.valid,details=True)
    record={'fit_risk':fit.cpu().numpy(),'risk':risk.cpu().numpy(),'sensitivities':{},
        'mechanism':{'fit_unit_residual':_moments(uf.cpu().numpy()),'valid_unit_residual':_moments(u.cpu().numpy()),
                     'attention':attention_stats(detail),'frozen_query_changes':{}}}
    names=['OFF']
    if model.arm in ('R2','R3'):names+=['RNA_MEAN','WRONG_RNA']
    if model.arm=='R3':names+=['C_MEAN','WRONG_C']
    for name in names:
        r,_,d=_forward(model,ds,ds.valid,intervention=name,details=name!='OFF')
        record['sensitivities'][name]=r.cpu().numpy()
        change={'patient_risk_difference_over_fit_offset_SD':_moments((risk-r).cpu().numpy()/ds.scale)}
        if d:change.update(attention_difference(detail,d))
        record['mechanism']['frozen_query_changes'][name]=change
    if not np.array_equal(record['sensitivities']['OFF'],ds.R0_valid):raise ValueError('local-OFF changed own main risk')
    return record


def run_cell(ds,identity,seed,fold,stage,epochs=80,step_callback=None):
    result={'source':identity,'stage':stage,'seed':int(seed),'fold':int(fold),
        'fit':np.array(ds.fit),'valid':np.array(ds.valid),'R0_valid':np.array(ds.R0_valid),
        'CP_valid':np.array(ds.CP_valid),'data_audit':ds.audit,'arms':{}}
    for arm in NEURAL_ARMS:
        model,initial,history,audit=train_arm(ds,arm,seed,fold,epochs,step_callback)
        record=predict_arm(model,ds)
        record.update(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},initial_state=initial,
            clinical_dim=ds.C.shape[1],history=history,training_audit=audit,
            parameter_counts={'trainable_local':sum(p.numel() for p in model.parameters()),
                              'common_frozen_main':int(ds.audit['fullfit_main_parameters'])})
        result['arms'][arm]=record;del model
    return result

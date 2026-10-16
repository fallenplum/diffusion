import time,random
import numpy as np
import torch
from .data import Data,cfg,source
from .model import Model
from src.phase10_trimodal.math import fit_ridge
from src.phase17_ssd.losses import full_risk_efron
from src.phase13_rna.math import inner_score


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def arguments(data,indices,wrong=False):
    indices=data.check_indices(indices);f=data.ft;ix=torch.as_tensor(indices,dtype=torch.long)
    ri=torch.as_tensor(data.wrong[np.asarray(indices)],dtype=torch.long) if wrong else ix
    return tuple(x.to('cuda') for x in [f['X'][ix],f['C'][ix],f['R'][ri],f['RM'][ri],f['P'][ix],f['PM'][ix]])


def forward(model,data,indices,chunk,wrong=False):
    return torch.cat([model(*arguments(data,indices[j:j+chunk],wrong)) for j in range(0,len(indices),chunk)])


def train_cell(arm,seed,fold):
    seed_all(seed+100*fold);config=cfg();tc=config['training'];data=Data(fold,'inner',seed)
    fit,valid=data.fit,data.valid;ep=data.ep;t,e=ep['time'],ep['event']
    w,diag=fit_ridge(data.f['X'][fit],t[fit],e[fit],tc['main_effect_l2'])
    baseline=data.f['X']@w
    model=Model(data,arm,w,config).cuda();wrong=arm=='A6';chunk=tc['microbatch_patients']
    opt=torch.optim.AdamW([{'params':[model.main],'weight_decay':0.0},{'params':list(model.interaction.parameters()) if model.interaction else [],'weight_decay':tc['weight_decay']}],lr=tc['learning_rate'])
    history=[];best=float('inf');best_epoch=0;best_state=None;ema=None;begin=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    if arm=='A0':
        best_epoch=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        for epoch in range(1,tc['max_epochs']+1):
            model.train();opt.zero_grad(set_to_none=True)
            # Exact full-risk-set gradient with bounded activation memory. Parameters
            # stay fixed throughout both passes; all stochastic dropout is disabled.
            with torch.no_grad():rr=forward(model,data,fit,chunk,wrong)
            leaf=rr.detach().requires_grad_();loss=full_risk_efron(leaf,t[fit],e[fit]);dr=torch.autograd.grad(loss,leaf)[0]
            for j in range(0,len(fit),chunk):
                r=model(*arguments(data,fit[j:j+chunk],wrong));(r*dr[j:j+chunk]).sum().backward()
            penalty=.5*tc['main_effect_l2']*model.main.square().sum();penalty.backward()
            grad={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None}
            gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip']));opt.step();model.eval()
            with torch.no_grad():vr=forward(model,data,valid,chunk,wrong);vl=float(full_risk_efron(vr,t[valid],e[valid]));hc=inner_score(t[valid],e[valid],vr.cpu().numpy())
            ema=vl if ema is None else .2*vl+.8*ema
            row={'epoch':epoch,'train_loss':float(loss.detach()),'penalty':float(penalty.detach()),'validation_loss':vl,'validation_loss_EMA':ema,'validation_Harrell':hc,'gradient_norm':gn,'query_gradient':sum(v*v for k,v in grad.items() if '.q.' in k or '.rna.' in k or '.c.' in k)**.5,'gate_gradient':grad.get('interaction.gate.weight',0.0),'seconds':time.monotonic()-begin}
            history.append(row)
            if epoch>=tc['minimum_epoch'] and ema<best-1e-8:
                best=ema;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch==1 or epoch%10==0:print('TRAIN',arm,seed,fold,epoch,round(vl,6),round(hc,6),round(time.monotonic()-begin,1),flush=True)
            if epoch-best_epoch>=tc['early_stopping_patience']:break
    model.load_state_dict(best_state);model.eval();risks=[];mech=[];wrong_risks=[];nulls=[]
    with torch.no_grad():
        for j in range(0,len(valid),chunk):
            ii=valid[j:j+chunk];aa=arguments(data,ii,wrong);r,md=model(*aa,diagnostic=True);risks.extend(r.cpu().numpy());mech.append({k:v.cpu().numpy() for k,v in md.items()})
            wrong_risks.extend(model(*arguments(data,ii,not wrong)).cpu().numpy())
            if model.interaction is not None and arm in ['A4','A5','A6','A7']:
                args=list(aa);args[1]=model.interaction.c0.expand_as(args[1]);nulls.extend((model(*args)-aa[0]@model.main).abs().cpu().numpy())
    risks=np.asarray(risks);own_main=data.f['X'][valid]@model.main.detach().cpu().numpy();res=risks-own_main
    mechanism={}
    for k in mech[0]:
        values=np.concatenate([q[k].reshape(-1) for q in mech]);mechanism[k+'_mean']=float(values.mean());mechanism[k+'_max']=float(values.max())
    mechanism.update(residual_SD=float(res.std()),matched_wrong_risk_MAE=float(np.mean(abs(risks-np.asarray(wrong_risks)))),clinical_null_max=float(max(nulls,default=0)),query_gradient_last=history[-1]['query_gradient'] if history else 0.0)
    summary={'arm':arm,'seed':seed,'fold':fold,'role':'inner_only','fit_patients':len(fit),'fit_events':int(e[fit].sum()),'validation_patients':len(valid),'validation_events':int(e[valid].sum()),'Harrell':inner_score(t[valid],e[valid],risks),'baseline_Harrell':inner_score(t[valid],e[valid],baseline[valid]),'validation_loss':float(full_risk_efron(torch.tensor(risks),t[valid],e[valid])),'selected_epoch':best_epoch,'actual_epochs':len(history),'hit_maximum':len(history)==tc['max_epochs'],'parameters_registered':sum(p.numel() for p in model.parameters()),'parameters_gradient_active':sum(p.numel() for k,p in model.named_parameters() if arm=='A0' and k=='main' or arm!='A0' and k in grad),'seconds':time.monotonic()-begin,'peak_GPU_MB':torch.cuda.max_memory_allocated()/2**20,**mechanism}
    return {'summary':summary,'history':history,'source':source(),'fit':fit,'valid':valid,'risk':risks,'baseline_risk':baseline[valid],'wrong_risk':np.asarray(wrong_risks),'state':best_state,'preprocessing':data.audit,'ridge':diag}

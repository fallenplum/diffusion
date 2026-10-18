"""Fit only the new multi-query branch on strict crossfit expert offsets."""
from __future__ import annotations
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from src.phase17_ssd.losses import full_risk_efron
from src.phase25_local.train import seed_all, _moments
from .model import MultiQuery
from .runtime import NEURAL_ARMS, cfg

CONTROLS=('OFF','VALUE_OFF','WRONG_RNA_QUERY','RNA_MEAN_QUERY','WRONG_C_QUERY','C_MEAN_QUERY')


def _forward(model, ds, patients, *, training=False, intervention=None, details=False):
    patients=np.asarray(patients,int); risks=[]; units=[]; collected=[]
    for start in range(0,len(patients),32):
        current=patients[start:start+32]
        slides=np.flatnonzero(np.isin(ds.owner,current)); owners=np.asarray(ds.owner)[slides]
        options={}
        if intervention is not None:
            if not np.isin(owners,ds.valid).all():raise ValueError('frozen controls are valid-only')
            if intervention not in CONTROLS:raise ValueError('unknown frozen control')
            if intervention=='VALUE_OFF':options['value_off']=True
            elif intervention=='WRONG_RNA_QUERY':options['query_rna']=ds.R[np.asarray(ds.wrong_rna)[owners]]
            elif intervention=='RNA_MEAN_QUERY':options['query_rna']=ds.R[np.asarray(ds.fit)].mean(0).expand(len(owners),-1)
            elif intervention=='WRONG_C_QUERY':options['query_clinical']=ds.C[np.asarray(ds.wrong_c)[owners]]
            elif intervention=='C_MEAN_QUERY':options['query_clinical']=ds.C[np.asarray(ds.fit)].mean(0).expand(len(owners),-1)
        if intervention=='OFF':wsi=ds.base.new_zeros(len(owners))
        else:
            args=(ds.visual[slides],ds.mask[slides],ds.R[owners],ds.C[owners])
            def call(*x, frozen_options=dict(options)):
                return model(*x,**frozen_options)['unit']
            if training:wsi=checkpoint(call,*args,use_reentrant=False,preserve_rng_state=True)
            else:
                output=model(*args,**options,return_details=details);wsi=output['unit']
                if details:collected.append({'output':output,'owner':owners.copy(),'mask':ds.mask[slides]})
        unit=torch.stack([wsi[torch.as_tensor(owners==p,device=wsi.device)].double().mean() for p in current])
        if not torch.isfinite(unit).all() or unit.abs().max()>1.+1e-7:raise ValueError('bounded unit contract failed')
        units.append(unit);risks.append(ds.base[current].detach().double()+.1*unit)
    return torch.cat(risks),torch.cat(units),collected


def gradient_groups(model):
    norms={};active=0
    for name,p in model.named_parameters():
        group=name.split('.')[0]; norms.setdefault(group,0.)
        if p.grad is None:continue
        g=p.grad.detach().double()
        if not torch.isfinite(g).all():raise ValueError('nonfinite gradient')
        norms[group]+=float(g.square().sum())
        if g.any():active+=p.numel()
    return {k:v**.5 for k,v in norms.items()},active


def train_arm(ds,arm,seed,fold,epochs=80,step_callback=None):
    if arm not in NEURAL_ARMS or not 1<=int(epochs)<=80:raise ValueError('invalid prototype arm/budget')
    seed_all(seed,fold)
    model=MultiQuery(ds.C.shape[1],arm).to(ds.visual.device)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    t,e=ds.fit_labels();t=np.asarray(t,np.float64);e=np.asarray(e,np.float64)
    if t.shape!=e.shape or len(t)!=len(ds.fit) or e.sum()==0:raise ValueError('fit-only labels invalid')
    base=ds.base.detach().clone()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    history=[];active=0
    for epoch in range(1,int(epochs)+1):
        started=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
        risk,unit,_=_forward(model,ds,ds.fit,training=True)
        cox=full_risk_efron(risk,t,e);loss=cox+.005*unit.square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite objective')
        loss.backward();groups,active=gradient_groups(model)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        opt.step();model.eval()
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite parameter')
        with torch.no_grad():
            r,u,_=_forward(model,ds,ds.fit)
            c=full_risk_efron(r,t,e);objective=c+.005*u.square().mean()
        if not torch.equal(ds.base,base):raise ValueError('strong main offset changed')
        if ds.visual.is_cuda:torch.cuda.synchronize(ds.visual.device)
        history.append({'epoch':epoch,'training_loss':float(loss.detach()),'fit_loss':float(objective),
            'fit_cox':float(c),'unit_residual_RMS':float(u.square().mean().sqrt()),
            'correction_RMS':float(.1*u.square().mean().sqrt()),'saturation_fraction':float((u.abs()>.95).double().mean()),
            'grad_norm':float(norm),'gradients':groups,'epoch_seconds':time.monotonic()-started})
        if step_callback:step_callback(arm,epoch)
    audit={'actual_epochs':int(epochs),'optimizer_steps':int(epochs),'heldout_endpoint_reads':0,
           'fixed_final_epoch':True,'main_offset_unchanged':True,'fit_offset_kind':'strict_threefold_crossfit_M0',
           'heldout_offset_kind':'full_fit_M0','global_convergence_established':False,
           'bound':.1,'query_residual_preserved':True}
    counts={'allocated_trainable':sum(p.numel() for p in model.parameters()),
            'last_epoch_parameters_with_any_nonzero_gradient':active}
    return model,initial,history,audit,counts


def _attention_rows(details):
    result=[]
    for item in details:
        a=item['output']['attention'].detach().cpu().numpy();mask=item['mask'].cpu().numpy()
        for rows,keep in zip(a,mask):
            for row in rows:
                w=row[keep].astype(float);n=len(w);h=-np.sum(w*np.log(np.maximum(w,1e-30)))
                result.append((h/np.log(n) if n>1 else 1.,np.exp(h)/n,.5*np.abs(w-1/n).sum()))
    return np.asarray(result)


def _slot_rows(details):
    cosine=[];ranks=[];ratios=[];mass=[]
    for item in details:
        out=item['output'];slots=out['slots'].detach().cpu().double().numpy()
        query=out['query'].detach().cpu().double().numpy();message=out['message'].detach().cpu().double().numpy()
        for h,q,m in zip(slots,query,message):
            normalized=h/np.maximum(np.linalg.norm(h,axis=-1,keepdims=True),1e-12)
            cosine.extend((normalized@normalized.T)[~np.eye(8,dtype=bool)].tolist())
            singular=np.linalg.svd(h,compute_uv=False);energy=singular**2
            probability=energy/energy.sum() if energy.sum() else np.full(8,1/8)
            ranks.append(float(np.exp(-np.sum(probability*np.log(np.maximum(probability,1e-30))))))
            ratios.extend((np.linalg.norm(m,axis=-1)/np.maximum(np.linalg.norm(q,axis=-1),1e-12)).tolist())
        n=item['mask'].sum(-1).detach().cpu().double().numpy()
        mass.extend((out['slot_mass'].detach().cpu().double().numpy()/n[:,None]).reshape(-1).tolist())
    return {'pairwise_cosine':_moments(cosine),'effective_rank':_moments(ranks),
            'message_over_query_norm':_moments(ratios),'assignment_fraction':_moments(mass)}


def _query_difference(matched,other):
    tv=[];overlap=[]
    if len(matched)!=len(other):raise ValueError('different control WSI count')
    for left,right in zip(matched,other):
        if not np.array_equal(left['owner'],right['owner']) or not torch.equal(left['mask'],right['mask']):
            raise ValueError('query intervention changed patch scope')
        x=left['output']['attention'].detach().cpu().numpy();y=right['output']['attention'].detach().cpu().numpy()
        for aa,bb,mask in zip(x,y,left['mask'].cpu().numpy()):
            for a,b in zip(aa,bb):
                a,b=a[mask],b[mask];k=min(32,len(a))
                ai=set(np.argsort(-a,kind='stable')[:k]);bi=set(np.argsort(-b,kind='stable')[:k])
                tv.append(.5*float(np.abs(a-b).sum()));overlap.append(len(ai&bi)/len(ai|bi))
    return {'attention_TV':_moments(tv),'top32_Jaccard':_moments(overlap)}


@torch.no_grad()
def predict_arm(model,ds):
    model.eval();fit,uf,_=_forward(model,ds,ds.fit)
    risk,uv,details=_forward(model,ds,ds.valid,details=True)
    attention=_attention_rows(details)
    record={'fit_risk':fit.cpu().numpy(),'risk':risk.cpu().numpy(),'sensitivities':{},
        'mechanism':{'fit_unit_residual':_moments(uf.cpu().numpy()),'valid_unit_residual':_moments(uv.cpu().numpy()),
            'attention':{key:_moments(attention[:,i]) for i,key in
                enumerate(('normalized_entropy','effective_patch_fraction','TV_from_uniform'))},
            'slot':_slot_rows(details),'frozen_query_changes':{}}}
    scale=float(ds.scale)  # Saved fit-CF population SD, diagnostic only.
    for control in CONTROLS:
        r,_,changed=_forward(model,ds,ds.valid,intervention=control,details=control not in ('OFF','VALUE_OFF'))
        record['sensitivities'][control]=r.cpu().numpy()
        value={'patient_risk_difference_over_fit_offset_SD':_moments((risk-r).cpu().numpy()/scale)}
        if changed:value.update(_query_difference(details,changed))
        record['mechanism']['frozen_query_changes'][control]=value
    if not np.array_equal(record['sensitivities']['OFF'],ds.R0_valid):raise ValueError('OFF not exact M0')
    if model.arm=='QUERY' and not np.array_equal(record['risk'],record['sensitivities']['VALUE_OFF']):
        raise ValueError('QUERY/ValueOFF mismatch')
    return record


def run_cell(ds,identity,seed,fold,epochs=80,step_callback=None):
    result={'source':identity,'stage':'prototype','seed':int(seed),'fold':int(fold),
        'fit':np.array(ds.fit),'valid':np.array(ds.valid),'M0_valid':np.array(ds.R0_valid),
        'CP_valid':np.array(ds.CP_valid),'RNA_valid':np.array(ds.RNA_valid),'data_audit':ds.audit,'arms':{}}
    for arm in NEURAL_ARMS:
        model,initial,history,audit,counts=train_arm(ds,arm,seed,fold,epochs,step_callback)
        record=predict_arm(model,ds)
        record.update(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},initial_state=initial,
            history=history,training_audit=audit,parameter_counts=counts,clinical_dim=ds.C.shape[1])
        result['arms'][arm]=record;del model
    return result

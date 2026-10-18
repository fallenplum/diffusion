"""Joint fit-only kernel/readout training; no validation epoch selection."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from src.phase17_ssd.losses import full_risk_efron
from src.phase23_cdei.data import Data
from src.phase23_cdei.warmup import _seed, _fit_contract
from src.phase23_cdei.model import module_fingerprint
from .model import G1Model
from .runtime import ARMS

EPOCHS=80
MICROBATCH=32


def reference_raw(ds):
    ix=np.array(ds.fit,copy=True)
    return {**{k:ds.ft[k][ix] for k in ('R','G','L','RM','CM')},'indices':ix}


def _forward(model,ds,patients,ref,*,training=False,intervention=None,details=False):
    """Keep complete patient risks; checkpoint recomputes CURRENT references."""
    values=[]; collected=[]
    for start in range(0,len(patients),MICROBATCH):
        current=np.asarray(patients[start:start+MICROBATCH])
        slides=np.flatnonzero(np.isin(ds.owner,current));owners=np.asarray(ds.owner)[slides]
        ft=ds.ft
        other_o=np.asarray(ds.wrong_o)[owners];other_c=np.asarray(ds.wrong_c)[owners]
        kw={}
        heldout=bool(np.isin(owners,np.asarray(ds.valid)).all())
        if heldout and model.arm!='REL':
            # Every frozen heldout contrast uses one support/reference plan.
            # Both queried identities are outside fit, so union exclusions do
            # not remove donors. This is not used in the training forward.
            # CUDA GEMM does not implement int64 addmm. Binary incidence counts
            # <=512 are represented exactly in float32 on CPU and CUDA.
            member=ft['membership'].to(torch.float32)
            def support(ix):
                r=(ft['RM'][ix].to(torch.float32)@member.T)>0
                c=(ft['CM'][ix].to(torch.float32)@member.T)>0
                return torch.cat((r,c,c),-1)
            kw['absolute_token_mask']=support(owners)&support(other_o)
            kw['reference_exclusions']=np.stack((owners,other_o),-1)
        if intervention:
            # A common reference exclusion is immaterial for heldout sources,
            # but is explicit for future FIT diagnostics. No labels are used.
            if not heldout:raise ValueError('frozen patient interventions are heldout-only')
            if intervention=='OFF':kw['absolute_off']=True
            elif intervention=='NO_C':kw['absolute_clinical']=ft['C'][np.array(ds.fit)].mean(0).expand(len(owners),-1)
            elif intervention=='WRONG_C':kw['absolute_clinical']=ft['C'][other_c]
            elif intervention=='WRONG_O':
                kw['absolute_omics']=tuple(ft[k][other_o] for k in ('R','G','L','RM','CM'))
                kw['omics_source_indices']=other_o
            else:raise ValueError('unknown intervention')
        args=(ds.visual[slides],ds.mask[slides],ft['C'][owners],
              *(ft[k][owners] for k in ('R','G','L','RM','CM')))
        # Bind loop variables: backward checkpoint must never capture the last
        # chunk's owner rows or reference plan through a mutable closure.
        def call(*data, owner_rows=owners.copy(), options=dict(kw)):
            return model(*data,owner_rows,ref,**options)['risk_delta_unit']
        if training:
            wsi=checkpoint(call,*args,use_reentrant=False,preserve_rng_state=True)
        else:
            result=model(*args,owners,ref,**kw,return_details=details)
            wsi=result['risk_delta_unit']
            if details:collected.append(result)
        values.append(torch.stack([wsi[torch.as_tensor(owners==p,device=wsi.device)].double().mean() for p in current]))
    correction=ds.anchor_sd_fit*torch.cat(values)
    anchor=ds.ft['ANCHOR'][np.array(patients,copy=True)].detach().double()
    return anchor+correction,correction,collected


def gradient_groups(model):
    groups={'kernel':[],'clinical_nonintercept':[],'RNA':[],'CNA_gain':[],
            'CNA_loss':[],'relative_head':[],'extra_head':[],'output_head':[]}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        g=p.grad.detach().double()
        if not torch.isfinite(g).all():raise ValueError('nonfinite gradient')
        norm=float(g.norm())
        if name.startswith('backbone.'):
            groups['kernel'].append(norm**2)
            if name=='backbone.compatibility.c.weight':groups['clinical_nonintercept'].append(float(g[:,1:].norm())**2)
            if name.startswith('backbone.molecular.filters.'):
                i=int(name.split('.')[3]);groups[('RNA','CNA_gain','CNA_loss')[i//8]].append(norm**2)
        elif name.startswith('extra'):groups['extra_head'].append(norm**2)
        elif name.startswith('relative'):groups['relative_head'].append(norm**2)
        else:groups['output_head'].append(norm**2)
    return {k:float(np.sqrt(sum(v))) for k,v in groups.items()}


def train_arm(ds,arm,seed,fold,epochs=EPOCHS,step_callback=None):
    if (arm not in ARMS or isinstance(epochs,bool) or int(epochs)!=epochs
            or not 1<=int(epochs)<=80):raise ValueError('invalid arm/budget')
    _seed(seed,fold)
    fit,owner,anchor,anchor_snapshot,scale,t,e,_=_fit_contract(ds)
    model=G1Model(int(ds.ft['C'].shape[1]),ds.ft['membership'].detach().cpu(),arm=arm,tau=1.).to(ds.visual.device)
    ref=reference_raw(ds)
    initial=module_fingerprint(model.backbone)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    history=[]
    with torch.no_grad():anchor_loss=float(full_risk_efron(anchor,t,e))
    for epoch in range(1,int(epochs)+1):
        started=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
        risk,delta,_=_forward(model,ds,fit,ref,training=True)
        loss=full_risk_efron(risk,t,e)
        if not torch.isfinite(loss):raise ValueError('nonfinite complete Efron objective')
        loss.backward();grad=gradient_groups(model)
        total=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        opt.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite updated parameters')
        if not torch.equal(ds.ft['ANCHOR'][fit],anchor_snapshot):raise ValueError('frozen main effect changed')
        model.eval()
        with torch.no_grad():
            er,ed,_=_forward(model,ds,fit,ref)
            eval_loss=float(full_risk_efron(er,t,e))
        history.append({'epoch':epoch,'training_loss':float(loss.detach()),'fit_loss':eval_loss,
                        'grad_norm':float(total),'gradients':grad,'fit_correction_SD':float(ed.std(unbiased=False)),
                        'epoch_seconds':time.monotonic()-started})
        if step_callback:step_callback(arm,epoch)
    model.eval()
    return model,{'actual_epochs':int(epochs),'optimizer_steps':int(epochs),'history':history,
                  'initial_backbone_sha256':initial,'final_backbone_sha256':module_fingerprint(model.backbone),
                  'anchor_only_fit_loss':anchor_loss,'fixed_final_epoch_no_validation_selection':True,
                  'heldout_endpoint_reads':0,'anchor_unchanged':True,
                  'reference_reencoded_each_forward':True,'reference_gradient_detached':False}


@torch.no_grad()
def predict_arm(model,ds):
    model.eval();ref=reference_raw(ds)
    fit_risk,_,_=_forward(model,ds,np.array(ds.fit),ref)
    risk,_,_=_forward(model,ds,np.array(ds.valid),ref)
    result={'fit_risk':fit_risk.cpu().numpy(),'risk':risk.cpu().numpy()}
    if model.arm=='LOCAL':
        result['sensitivities']={key:_forward(model,ds,np.array(ds.valid),ref,intervention=key)[0].cpu().numpy()
                                 for key in ('OFF','NO_C','WRONG_C','WRONG_O')}
    return result


def run_cell(ds,source,output_dir,epochs=80,step_callback=None):
    result={'source':source,'role':ds.role,'seed':ds.seed,'fold':ds.fold,
            'fit':np.array(ds.fit),'valid':np.array(ds.valid),
            'anchor_fit':ds.ft['ANCHOR'][np.array(ds.fit)].double().cpu().numpy(),
            'anchor_valid':ds.ft['ANCHOR'][np.array(ds.valid)].double().cpu().numpy(),
            'J_CS_valid':np.array(ds.inherited_simple['models']['J_CS']['risk']),
            'data_audit':ds.audit,'arms':{}}
    for arm in ARMS:
        model,audit=train_arm(ds,arm,ds.seed,ds.fold,epochs=epochs,step_callback=step_callback)
        record=predict_arm(model,ds)
        record.update(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                      clinical_dim=int(ds.ft['C'].shape[1]),membership=ds.ft['membership'].cpu(),
                      history=audit['history'],training_audit={k:v for k,v in audit.items() if k!='history'},
                      fit_only_reference_audit={'raw_fit_patient_count':len(ds.fit),'patient_equal':True,
                        'fit_query_leave_self_out':True,'current_encoder_current_kernel':True,'no_heldout_reference':True},
                      parameter_counts={'total':sum(p.numel() for p in model.parameters()),
                        'backbone':sum(p.numel() for p in model.backbone.parameters())})
        result['arms'][arm]=record
        del model
    if output_dir is not None:
        path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
        torch.save(result,path/'checkpoint.partial');(path/'checkpoint.partial').replace(path/'checkpoint.pt')
    return result

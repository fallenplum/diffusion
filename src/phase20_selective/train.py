"""Joint inner selection, fresh refits, exact frozen offsets, full-risk Cox."""
import copy
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .data import Data, cfg, source
from src.phase19_attention.train import seed_all
from src.phase17_ssd.losses import full_risk_efron
from src.phase13_rna.math import inner_score

MODES = ['matched','wrongC','wrongO','constantC','off']
EXTRA_MODES = ['fullOmicsWrong','uniformPair']


def simple_models(ds):
    return ds.inherited_simple


def make_model(ds, arm, w=None):
    from .model import SelectiveModel
    return SelectiveModel(ds.f['C'].shape[1], ds.f['membership'],
        ds.anchor_w if w is None else w, ds.f['C'][ds.fit].mean(0),
        ds.anchor_sd_fit, arm, cfg()).to(ds.visual.device)


def assigned_mode(arm):
    return {'WC16':'wrongC','WO16':'wrongO'}.get(arm,'matched')


def _call_batch(model, ds, patients, mode='matched', training=False,
                diagnostic=False, context_overrides=None):
    f = ds.ft
    slides = np.flatnonzero(np.isin(ds.owner, patients))
    owners = ds.owner[slides]
    if set(owners) != set(patients):
        raise ValueError('patient without qualified WSI')
    oi = ds.wrong_o[owners] if mode == 'wrongO' else owners
    ci = ds.wrong_c[owners] if mode == 'wrongC' else owners
    ct = f['C'][ci]
    if mode == 'constantC':
        ct = model.clinical_mean.expand(len(owners), -1)
    own_i = ds.wrong_o[owners] if mode == 'fullOmicsWrong' else owners
    own = tuple(f[k][own_i] for k in ['R','G','L','RM','CM'])
    ctx = [f[k][oi] for k in ['R','G','L','RM','CM']]
    if mode == 'fullOmicsWrong':
        ctx = list(own)
    if context_overrides:
        if 'clinical' in context_overrides:
            ct = context_overrides['clinical'][ci]
        for j, key in enumerate(['R','G','L','RM','CM']):
            if key in context_overrides:
                ctx[j] = context_overrides[key][oi]
    args = (ds.visual[slides],ds.mask[slides],f['C'][owners],*own,
            f['X'][owners],ct,*ctx,f['ANCHOR'][owners])
    def call(vv,mm,cc,rr,gg,ll,rmm,cmm,xx,ctxc,ctxr,ctxg,ctxl,ctxrm,ctxcm,aa):
        return model(vv,mm,cc,rr,gg,ll,rmm,cmm,xx,
            context_clinical=ctxc,context_omics=(ctxr,ctxg,ctxl,ctxrm,ctxcm),
            diagnostic=diagnostic,interaction_off=mode=='off',main_only=mode=='anchor',
            pair_uniform=mode=='uniformPair',anchor_override=aa)
    result = checkpoint(call,*args,use_reentrant=False,preserve_rng_state=True) if training else call(*args)
    return result, owners


def forward(model, ds, indices, mode='matched', training=False, diagnostic=False,
            context_overrides=None, **unused):
    if mode not in MODES+EXTRA_MODES+['anchor']:
        raise ValueError('invalid absolute intervention mode')
    indices = ds.check_indices(indices)
    risks, probes = [], []
    n = cfg()['training']['microbatch_patients']
    for start in range(0,len(indices),n):
        patients = indices[start:start+n]
        result, owners = _call_batch(model,ds,patients,mode,training,diagnostic,context_overrides)
        if diagnostic:
            result, dg = result
            probes.append({key:value.detach().cpu() if isinstance(value,torch.Tensor) else value
                           for key,value in dg.items()})
        if mode=='anchor':
            # Repeated identical float32 offsets can lose one ULP in CUDA
            # mean reduction for three WSIs.  Verify every model output first,
            # then select one identical value; do not change neural aggregation.
            expected_anchor = ds.ft['ANCHOR'][owners]
            if (result.shape != expected_anchor.shape or result.dtype != expected_anchor.dtype
                    or not torch.equal(result,expected_anchor)):
                raise ValueError('per-WSI frozen anchor output identity failed')
            risks.append(torch.stack([result[torch.as_tensor(owners==pi,device=result.device)][0]
                                      for pi in patients]))
        else:
            risks.append(torch.stack([result[torch.as_tensor(owners==pi,device=result.device)].mean()
                                      for pi in patients]))
    risk = torch.cat(risks)
    return (risk,probes) if diagnostic else risk


def _gradients(model):
    groups = {}
    active = 0
    for name,p in model.named_parameters():
        if p.grad is None:
            continue
        active += p.numel()
        group = name.split('.')[0]
        groups[group] = groups.get(group,0.) + float(p.grad.detach().norm())**2
    return {k+'_gradient':float(v**.5) for k,v in groups.items()}, active


def train_cell(arm,seed,fold,role,epochs=None,dataset=None,progress_callback=None):
    if role=='outer' and epochs is None:
        raise ValueError('fresh outer requires locked inner epoch')
    seed_all(seed+100*fold)
    ds = Data(fold,role,seed) if dataset is None else dataset
    config, tc = cfg(), cfg()['training']
    model = make_model(ds,arm)
    if 'main' in dict(model.named_parameters()):
        raise ValueError('frozen main was registered as a trainable parameter')
    optimizer = torch.optim.AdamW(model.parameters(),lr=tc['learning_rate'],weight_decay=tc['weight_decay'])
    mode = assigned_mode(arm)
    history,best_state,best_epoch,best,ema = [],None,0,float('inf'),None
    length = int(epochs if epochs is not None else tc['max_epochs'])
    begin = time.monotonic()
    if ds.visual.is_cuda:
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,length+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        risk = forward(model,ds,ds.fit,mode,training=True)
        objective = full_risk_efron(risk.double(),ds.ep['time'][ds.fit],ds.ep['event'][ds.fit])
        if not torch.isfinite(objective):
            raise ValueError('nonfinite complete-risk objective')
        objective.backward()
        gradients, active = _gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(),tc['gradient_clip'])
        optimizer.step()
        if progress_callback is not None:
            progress_callback('generator',epoch)
        if not torch.equal(model.main.detach().cpu(),torch.as_tensor(ds.anchor_w,dtype=torch.float64)):
            raise ValueError('frozen main weights changed')
        model.eval()
        with torch.no_grad():
            fitted = forward(model,ds,ds.fit,mode)
            fit_loss = float(full_risk_efron(fitted.double(),ds.ep['time'][ds.fit],ds.ep['event'][ds.fit]))
            row = {'epoch':epoch,'training_loss':float(objective.detach()),'train_loss':float(objective.detach()),
                   'penalty':0.,'auxiliary_loss':0.,'post_update_fit_loss':fit_loss,
                   **gradients,'stored_gradient_parameters':active}
            if role=='inner':
                valid = forward(model,ds,ds.valid,mode)
                valid_loss = float(full_risk_efron(valid.double(),ds.ep['time'][ds.valid],ds.ep['event'][ds.valid]))
                ema = valid_loss if ema is None else tc['validation_ema_alpha']*valid_loss+(1-tc['validation_ema_alpha'])*ema
                row.update(validation_loss=valid_loss,validation_EMA=ema,validation_loss_EMA=ema,
                    validation_Harrell=float(inner_score(ds.ep['time'][ds.valid],ds.ep['event'][ds.valid],valid.cpu().numpy())))
                if epoch>=min(tc['minimum_epoch'],length) and ema<best-1e-8:
                    best,best_epoch,best_state=ema,epoch,copy.deepcopy(model.state_dict())
            else:
                # Outer labels never decide weights, epochs or configuration.
                best_epoch=epoch
        history.append(row)
        if epoch==1 or epoch%20==0:
            print('TRAIN_PHASE20',arm,seed,fold,role,epoch,round(row['training_loss'],6),
                  row.get('validation_Harrell','NA'),round(time.monotonic()-begin,1),flush=True)
        if role=='inner' and epoch>=tc['minimum_epoch'] and epoch-best_epoch>=tc['early_stopping_patience']:
            break
    if role=='inner':
        if best_state is None:
            raise ValueError('no qualified inner checkpoint')
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        values = {md:forward(model,ds,ds.valid,md).cpu().numpy() for md in MODES}
        anchor = forward(model,ds,ds.valid,'anchor').cpu().numpy()
        extras = {md:forward(model,ds,ds.valid,md).cpu().numpy() for md in EXTRA_MODES} if arm in ['P1','P16'] else {}
    if not np.array_equal(anchor,ds.inherited_simple['models']['J0']['risk']):
        raise ValueError('exact frozen offset identity failed')
    if not np.array_equal(values['constantC'],values['off']):
        raise ValueError('reference Clinical did not exactly disable correction')
    from .diagnostics import inspect_model
    mechanism = inspect_model(model,ds)
    summary = {'registered_parameters':sum(p.numel() for p in model.parameters()),
        'last_step_gradient_parameters':active,'discriminator_parameters':0,'discriminator_optimizer_steps':0,
        'actual_epochs':len(history),'selected_epoch':best_epoch,'hit_maximum':role=='inner' and len(history)==tc['max_epochs'],
        'seconds':time.monotonic()-begin,
        'peak_GPU_MiB':float(torch.cuda.max_memory_allocated()/1024**2) if ds.visual.is_cuda else 0.,
        'last_step_gradients':gradients,'main_trainable':False}
    ck = {'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'source':source(),'arm':arm,'seed':seed,'fold':fold,'role':role,
        'fit':ds.fit.copy(),'validation_or_test':ds.valid.copy(),
        'main_initial_w':ds.anchor_w.copy(),'anchor_w':ds.anchor_w.copy(),
        'anchor_sd_fit':ds.anchor_sd_fit,'anchor_risk':anchor,'anchor_reference':ds.anchor_reference,
        'selected_epoch':best_epoch,'history':history,'preprocess_audit':ds.audit,
        'risk':values[mode],'assigned_mode':mode,'matched_risk':values['matched'],
        'wrongC_risk':values['wrongC'],'wrongO_risk':values['wrongO'],
        'constantC_risk':values['constantC'],'interaction_off_risk':values['off'],
        'summary':summary,'mechanism':mechanism,
        'probe_indices':{'risk_attention':ds.fit.copy(),'isolated_gradient':ds.gradient_probe_indices.copy()}}
    if extras:
        ck.update(full_omics_wrong_risk=extras['fullOmicsWrong'],uniform_pair_risk=extras['uniformPair'])
    return ck

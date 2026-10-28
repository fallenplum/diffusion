"""Locked R5 full-risk-set optimization and patient-private predictions.

The extra cross-amplitude penalty is common to every formal R5 readout. This
is a new matched objective, not an assertion that an old optimizer trajectory
has been reproduced. Mean-query contexts are detached train-only references,
refreshed once before each deterministic two-pass update.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from src.phase10_trimodal.math import torch_cox,log_weights
from .survival import evaluate


def state_hash(state, *, backbone_only=False):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        if backbone_only and name.startswith('readout.'):continue
        value=tensor.detach().cpu().contiguous()
        digest.update(name.encode()+str(value.dtype).encode()+str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def saved_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def refresh_reference(model,batch_size=16):
    # Also needed for frozen mean-query probes, independent of training mode.
    return model.refresh_training_query_mean(batch_size)


def full_risk_step(model,batches,time,event,optimizer,*,total_weight=.1,cross_weight=.1):
    model.train();optimizer.zero_grad(set_to_none=True)
    if model.intervention in ('mean_query','query_mean'):refresh_reference(model)
    with torch.no_grad():
        cached_risk=[];cached_cross=[];cached_delta=[]
        for idx in batches:
            row=model(idx);cached_risk.append(row['risk']);cached_cross.append(row['cross_delta']);cached_delta.append(row['delta'])
        risk=torch.cat(cached_risk);cross=torch.cat(cached_cross);delta=torch.cat(cached_delta)
        del row,cached_risk,cached_cross,cached_delta
    if not torch.isfinite(risk).all() or not torch.isfinite(cross).all():
        raise RuntimeError('NONFINITE_CACHED_RISK')
    leaf=risk.detach().requires_grad_(True)
    cox=torch_cox(leaf,log_weights(time,event,model.device),torch.as_tensor(event,device=model.device))
    if not torch.isfinite(cox):raise RuntimeError('NONFINITE_FULL_RISK_COX_LOSS')
    gradient=torch.autograd.grad(cox,leaf)[0].detach()
    position=0;maximum=0.;cross_maximum=0.
    for idx in batches:
        row=model(idx);n=len(idx)
        maximum=max(maximum,float((row['risk'].detach()-risk[position:position+n]).abs().max()))
        cross_maximum=max(cross_maximum,float((row['cross_delta'].detach()-cross[position:position+n]).abs().max()))
        if maximum>2e-5 or cross_maximum>2e-5:raise RuntimeError('DETERMINISTIC_CACHE_REPLAY_FAILED')
        penalty=(total_weight*row['delta'].square().sum()+cross_weight*row['cross_delta'].square().sum())/len(risk)
        (torch.dot(row['risk'],gradient[position:position+n])+penalty).backward();position+=n
    regularizer=model.regularization()
    if not torch.isfinite(regularizer):raise RuntimeError('NONFINITE_PARAMETER_REGULARIZER')
    regularizer.backward()
    try:
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
    except RuntimeError as exc:
        raise RuntimeError('NONFINITE_FULL_RISK_GRADIENT_NORM') from exc
    if not np.isfinite(norm):raise RuntimeError('NONFINITE_FULL_GRADIENT_NORM')
    optimizer.step()
    return dict(cox_loss=float(cox.detach()),parameter_regularizer=float(regularizer.detach()),
        total_risk_penalty=float(total_weight*delta.square().mean()),
        cross_risk_penalty=float(cross_weight*cross.square().mean()),gradient_norm=norm,
        risk_cache_replay_max=maximum,cross_cache_replay_max=cross_maximum,
        full_riskset_patients=len(risk),optimizer_updates=1)


SCALARS=('risk','delta','main_delta','cross_delta','sham_delta','pool_p','pool_o','z_o','z_p','z')


def predict(model,indices,batch_size=8,control=None,*,attention=False):
    model.eval();parts={k:[] for k in SCALARS};aux={}
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            idx=torch.as_tensor(indices[start:start+batch_size],device=model.device)
            row=model(idx,control=control)
            for name in SCALARS:parts[name].append(row[name].detach().cpu().numpy())
            inactive=row['interaction_details']['diagnostics']['interaction_off']
            aux.setdefault('cross_route_active',[]).append(np.full(len(idx),not inactive,dtype=float))
            if row['scalar_derivative'] is not None:
                aux.setdefault('readout_derivative_z',[]).append(row['scalar_derivative'].cpu().numpy())
            if model.readout_mode=='directional':
                do,dp=model.readout.directional_derivatives(row['z'])
                aux.setdefault('readout_derivative_zo',[]).append(np.zeros(len(idx)) if inactive else do.cpu().numpy())
                aux.setdefault('readout_derivative_zp',[]).append(np.zeros(len(idx)) if inactive else dp.cpu().numpy())
            if model.readout_mode=='rank4':
                joined=torch.cat((row['summary_omics_update'],row['summary_path_update']),-1)
                jac=model.readout.summary_jacobian(joined)
                aux.setdefault('summary_jacobian_L2',[]).append(np.zeros(len(idx)) if inactive else jac.norm(dim=-1).cpu().numpy())
            if attention:
                trace=row['interaction_details']['trace']
                for direction in ('omics_to_patch','path_to_omics'):
                    if not trace or direction not in trace[-1]:continue
                    prob=trace[-1][direction]['attention'];valid=prob.sum(-1)>0
                    entropy=-(prob*prob.clamp_min(torch.finfo(prob.dtype).tiny).log()).sum(-1)
                    flat_valid=valid.flatten(1);counts=flat_valid.sum(1).clamp_min(1)
                    value=(entropy.flatten(1)*flat_valid).sum(1)/counts
                    aux.setdefault(direction+'_row_entropy',[]).append(value.cpu().numpy())
                    k=min(32,prob.shape[-1]);top=prob.topk(k,dim=-1).indices
                    selected=torch.zeros_like(prob,dtype=torch.bool).scatter_(-1,top,True)&(prob>0)
                    # Count within each WSI before summing: index n in two
                    # different slides is not the same pathology patch.
                    union=selected.any(-2).flatten(1).sum(-1)
                    possible=selected.flatten(1).sum(-1).clamp_min(1)
                    aux.setdefault(direction+'_per_WSI_top32_union_ratio',[]).append((union/possible).cpu().numpy())
    result={name:np.concatenate(value).astype(np.float64) for name,value in parts.items()}
    result.update({name:np.concatenate(value).astype(np.float64) for name,value in aux.items()})
    if any(not np.isfinite(value).all() for value in result.values()):raise RuntimeError('NONFINITE_PREDICTION')
    return result


def describe(values):
    values=np.asarray(values,float)
    return dict(n=int(values.size),mean=float(values.mean()),sd=float(values.std()),
        min=float(values.min()),q10=float(np.quantile(values,.1)),median=float(np.median(values)),
        q90=float(np.quantile(values,.9)),max=float(values.max()))


def risk_diagnostics(prediction,offset):
    denom=max(float(np.std(offset)),1e-12)
    result={name:describe(value) for name,value in prediction.items() if name!='risk'}
    result['SD_total_delta_over_offset']=float(np.std(prediction['delta'])/denom)
    result['SD_cross_delta_over_offset']=float(np.std(prediction['cross_delta'])/denom)
    if 'readout_derivative_z' in prediction:
        active=prediction['cross_route_active']>0;d=prediction['readout_derivative_z']
        result['derivative_below_one_percent_of_initial_max']=float(np.mean(np.abs(d[active])<.0005)) if active.any() else None
        result['derivative_saturation_not_applicable_when_route_disabled']=not bool(active.any())
    result['scalar_z_derivative_not_defined_for_rank4']=('summary_jacobian_L2' in prediction)
    return result


def metrics_for(raw,prepared,indices,prediction,*,include_pfi=True):
    tr=prepared['train'];risk=prediction['risk']
    refs={'CP_STRONG':prepared['CP'][indices],'CP_SCALAR':prepared['base'][indices]}
    result={name:evaluate(raw['t'][tr],raw['e'][tr],raw['t'][indices],raw['e'][indices],risk,ref)
            for name,ref in refs.items()}
    if include_pfi:
        usable=raw['pfi_valid'][indices];pfitrain=tr[raw['pfi_valid'][tr]]
        result['PFI_frozen_DSS_risk']=evaluate(raw['pfi_t'][pfitrain],raw['pfi_e'][pfitrain],
            raw['pfi_t'][indices][usable],raw['pfi_e'][indices][usable],risk[usable])
    return result


def frozen_attention_probe(model,indices,*,batch_size=8):
    """Eight predeclared held-out patients; no fitting or medical truth claim."""
    selected=np.asarray(indices)[:8];answers=[];model.eval()
    refresh_reference(model)
    with torch.no_grad():
        for start in range(0,len(selected),batch_size):
            idx=torch.as_tensor(selected[start:start+batch_size],device=model.device)
            correct=model(idx,control='matched_query');wrong=model(idx,control='wrong_query')
            trace=correct['interaction_details']['trace'];other=wrong['interaction_details']['trace']
            for direction in ('omics_to_patch','path_to_omics'):
                if not trace or not other or direction not in trace[-1] or direction not in other[-1]:continue
                a,b=trace[-1][direction]['attention'],other[-1][direction]['attention']
                valid=(a.sum(-1)>0)&(b.sum(-1)>0)
                if not valid.any():continue
                tv=.5*(a-b).abs().sum(-1)
                k=min(32,a.shape[-1]);va,ia=a.topk(k,dim=-1);vb,ib=b.topk(k,dim=-1)
                equal=(ia[..., :,None]==ib[...,None,:])&(va[..., :,None]>0)&(vb[...,None,:]>0)
                intersection=equal.any(-1).sum(-1);union=(va>0).sum(-1)+(vb>0).sum(-1)-intersection
                overlap=intersection/union.clamp_min(1)
                answers.append(dict(direction=direction,valid_rows=int(valid.sum()),
                    attention_TV_sum=float(tv[valid].sum()),top32_overlap_sum=float(overlap[valid].sum())))
    output={}
    for name in {r['direction'] for r in answers}:
        rows=[r for r in answers if r['direction']==name];n=sum(r['valid_rows'] for r in rows)
        output[name]=dict(rows=n,attention_TV=sum(r['attention_TV_sum'] for r in rows)/n,
            top32_positive_support_Jaccard=sum(r['top32_overlap_sum'] for r in rows)/n)
    return dict(probe_patients=len(selected),scope='fixed_first_eight_validation_patients_not_histology_truth',directions=output)


def pure_survival_gradient_probe(model,prepared,raw,batch_size=8):
    model.eval();train=np.asarray(prepared['train']);refresh_reference(model)
    batches=[torch.as_tensor(train[j:j+batch_size],device=model.device) for j in range(0,len(train),batch_size)]
    with torch.no_grad():risk=torch.cat([model(idx)['risk'] for idx in batches])
    leaf=risk.detach().requires_grad_(True)
    loss=torch_cox(leaf,log_weights(raw['t'][train],raw['e'][train],model.device),torch.as_tensor(raw['e'][train],device=model.device))
    full_gradient=torch.autograd.grad(loss,leaf)[0].detach()
    # Only the contribution of a fixed first microbatch is inspected below.
    row=model(batches[0])
    names=[];parameters=[]
    for name,p in model.named_parameters():
        if p.requires_grad and name.startswith(('interaction.','readout.')):names.append(name);parameters.append(p)
    paths={}
    for path in ('cross_delta','risk'):
        target=torch.dot(row[path],full_gradient[:len(batches[0])])
        gradients=torch.autograd.grad(target,parameters,allow_unused=True,retain_graph=True) if target.requires_grad else [None]*len(parameters)
        groups={}
        for name,g in zip(names,gradients):
            group='.'.join(name.split('.')[:2]);cell=groups.setdefault(group,dict(tensors=0,nonzero=0,L2_squared=0.))
            cell['tensors']+=1
            if g is not None:
                cell['nonzero']+=int(bool(torch.any(g!=0)));cell['L2_squared']+=float(g.square().sum())
        paths[path]=groups
    return dict(scope='first_train_microbatch_contribution_to_full_riskset_Cox_no_regularizers',
        patients=len(batches[0]),groups_by_path=paths,new_optimizer_updates=0)

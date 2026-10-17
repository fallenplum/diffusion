"""Read-only, outcome-blind diagnostics of a frozen relative-update branch.

No optimizer, endpoint access, probe fitting, persistent hooks, raw maps, raw
patient predictions or patient identifiers are used/returned.  Population
moments retain query -> equal-WSI -> equal-patient aggregation explicitly.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from src.phase20_selective.operators import conservative_update
from src.phase20_selective.train import _call_batch


class _Moments:
    def __init__(self):
        self.data = {}

    def add(self, key, values):
        x = (values.detach().cpu().double().numpy() if isinstance(values,torch.Tensor)
             else np.asarray(values,dtype=np.float64)).reshape(-1)
        if not np.isfinite(x).all():
            raise ValueError('nonfinite diagnostic: '+key)
        q = self.data.setdefault(key,{'count':0,'sum':0.,'sum_squares':0.,'min':None,'max':None})
        if len(x):
            q['count'] += len(x); q['sum'] += float(x.sum());q['sum_squares'] += float(x@x)
            q['min'] = float(x.min()) if q['min'] is None else min(q['min'],float(x.min()))
            q['max'] = float(x.max()) if q['max'] is None else max(q['max'],float(x.max()))

    def merge(self, other):
        for key, value in other.items():
            q = self.data.setdefault(key,{'count':0,'sum':0.,'sum_squares':0.,'min':None,'max':None})
            q['count'] += int(value['count']);q['sum'] += float(value['sum'])
            q['sum_squares'] += float(value['sum_squares'])
            if value['count']:
                q['min'] = value['min'] if q['min'] is None else min(q['min'],value['min'])
                q['max'] = value['max'] if q['max'] is None else max(q['max'],value['max'])

    def summary(self):
        result = {}
        for key,q in self.data.items():
            n = q['count'];mean = q['sum']/n if n else None
            result[key] = {'count':n,'mean':mean,
                'SD':math.sqrt(max(q['sum_squares']/n-mean**2,0.)) if n else None,
                'min':q['min'],'max':q['max']}
        return result


def _hierarchy(store, key, values, defined, owners, patients):
    """Aggregate one scalar per query, excluding only explicitly undefined rows."""
    x = values.detach().cpu().double().numpy()
    valid = defined.detach().cpu().numpy().astype(bool)
    if x.shape != valid.shape or len(x) != len(owners):
        raise ValueError('query ownership/shape differs')
    store.add(key+'_query',x[valid])
    wsi = np.asarray([x[j,valid[j]].mean() if valid[j].any() else np.nan for j in range(len(x))])
    store.add(key+'_WSI',wsi[np.isfinite(wsi)])
    patient = []
    for p in patients:
        chosen = wsi[owners==p];chosen = chosen[np.isfinite(chosen)]
        if len(chosen):patient.append(float(chosen.mean()))
    store.add(key+'_patient',patient)


def _wsi_hierarchy(store, key, values, owners, patients):
    x = values.detach().cpu().double().numpy().reshape(-1)
    if len(x) != len(owners):raise ValueError('WSI ownership differs')
    store.add(key+'_WSI',x)
    store.add(key+'_patient',[x[owners==p].mean() for p in patients])


def _ratio(numerator, denominator):
    valid = denominator > 0.
    value = numerator/torch.where(valid,denominator,torch.ones_like(denominator))
    return torch.where(valid,value,torch.zeros_like(value)),valid


def _direction_metrics(base, updated, gate, delta, eligible, values, query,
                       rho=.5, update_operator=conservative_update):
    """Private transient per-query tensors; callers expose aggregate moments only."""
    if base.shape != updated.shape or base.shape != gate.shape or base.shape != delta.shape:
        raise ValueError('directional map shapes differ')
    if eligible.shape != base.shape or query.shape != base.shape[:-1]:
        raise ValueError('directional masks differ')
    if values.shape[:2] != (base.shape[0],base.shape[-1]):
        raise ValueError('directional values differ')
    n = eligible.sum(-1);raw_defined=query & (n>0)
    safe = torch.where(eligible,delta,torch.zeros_like(delta)).double()
    mean = safe.sum(-1)/n.clamp_min(1)
    centered_raw = torch.where(eligible,safe-mean[...,None],torch.zeros_like(safe))
    raw_var = centered_raw.square().sum(-1)/n.clamp_min(1)
    # Match the original operator's product rounding before double accumulation.
    weight = torch.where(eligible,base*gate,torch.zeros_like(base)).double()
    mass = weight.sum(-1);positive=query & (mass>0.)
    safe_mass = torch.where(mass>0.,mass,torch.ones_like(mass))
    mu = (weight*safe).sum(-1)/safe_mass
    centered = safe-mu[...,None]
    variance = (weight*centered.square()).sum(-1)/safe_mass
    energy = (weight*safe.square()).sum(-1)/safe_mass
    common_fraction,energy_defined = _ratio(mu.square(),energy)
    centered_fraction,_ = _ratio(variance,energy)
    exp_delta = safe.exp();Z = (weight*exp_delta).sum(-1)/safe_mass
    change = updated-base
    change64 = updated.double()-base.double()
    tv = .5*change64.abs().sum(-1)
    taylor = rho*weight*centered
    error = (change64-taylor).abs().sum(-1)
    actual_l1 = change64.abs().sum(-1)
    relative_error,nonzero_change = _ratio(error,actual_l1)
    weighted_rms = variance.clamp_min(0.).sqrt()
    tv_gain,nonzero_variation = _ratio(tv,rho*mass*weighted_rms)
    recomputed = update_operator(base,gate,delta,eligible,rho=rho)
    # These are operator-only identities, not extra model forward evaluations.
    constant_delta = mu.to(delta.dtype)[...,None].expand_as(delta)
    constant = update_operator(base,gate,constant_delta,eligible,rho=rho)
    centered_delta = delta-mu.to(delta.dtype)[...,None]
    centered_updated = update_operator(base,gate,centered_delta,eligible,rho=rho)
    eps = torch.finfo(base.dtype).eps
    checks = {
        'recomputed_map_max_error':float((recomputed-updated).abs().max().detach().cpu()),
        'constant_only_map_max_error':float((constant-base).abs().max().detach().cpu()),
        'centered_delta_invariance_max_error':float((centered_updated-recomputed).abs().max().detach().cpu()),
        'map_probability_sum_max_error':float((updated.double().sum(-1)-base.double().sum(-1)).abs().max().detach().cpu()),
        'numeric_absolute_tolerance':float(32*eps)}
    if max(checks[k] for k in ['recomputed_map_max_error','constant_only_map_max_error',
                               'centered_delta_invariance_max_error']) > checks['numeric_absolute_tolerance']:
        raise ValueError('operator identity failed: '+str(checks))
    delta_message = change@values
    taylor_message = taylor@values.double()
    message_norm = delta_message.double().norm(dim=-1)
    triangle_bound = change64.abs()@values.double().norm(dim=-1)[...,None]
    triangle_bound = triangle_bound.squeeze(-1)
    message_efficiency,message_defined = _ratio(message_norm,triangle_bound)
    value_mean = (weight@values.double())/safe_mass[...,None]
    value_var = (weight[...,None]*(values.double()[:,None]-value_mean[:,:,None]).square()).sum((-1,-2))/safe_mass
    taylor_bound = rho*mass*weighted_rms*value_var.clamp_min(0.).sqrt()
    covariance_efficiency,covariance_defined = _ratio(taylor_message.norm(dim=-1),taylor_bound)
    rows = {
        'eligible_count':(n.double(),query),'zero_selected_mass':((mass<=0.).double(),query),
        'raw_delta_row_mean':(mean,raw_defined),'raw_delta_row_SD':(raw_var.clamp_min(0.).sqrt(),raw_defined),
        'selected_mass':(mass,query),'weighted_delta_mean':(mu,positive),
        'weighted_centered_variance':(variance,positive),'weighted_centered_RMS':(weighted_rms,positive),
        'weighted_raw_second_moment':(energy,positive),
        'zero_weighted_delta_energy':((energy<=0.).double(),positive),
        'common_energy_fraction':(common_fraction,positive&energy_defined),
        'centered_energy_fraction':(centered_fraction,positive&energy_defined),
        'conservative_reference_Z':(Z,positive),'attention_TV':(tv,query),
        'actual_change_L1':(actual_l1,query),'taylor_change_L1':(taylor.abs().sum(-1),query),
        'taylor_absolute_error_L1':(error,query),'taylor_relative_error_L1':(relative_error,query&nonzero_change),
        'TV_over_rho_mass_centered_RMS':(tv_gain,positive&nonzero_variation),
        'delta_message_L2_before_query_average':(message_norm,query),
        'delta_message_triangle_bound':(triangle_bound,query),
        'message_over_triangle_bound':(message_efficiency,query&message_defined),
        'weighted_value_centered_RMS':(value_var.clamp_min(0.).sqrt(),positive),
        'taylor_message_L2':(taylor_message.norm(dim=-1),query),
        'taylor_message_Cauchy_bound':(taylor_bound,query),
        'taylor_covariance_efficiency':(covariance_efficiency,positive&covariance_defined),
        'taylor_message_error_L2':((delta_message.double()-taylor_message).norm(dim=-1),query)}
    return {'rows':rows,'message':delta_message,'query':query,'checks':checks}


def diagnose_batch(model, ds, patients, mode='matched', update_operator=conservative_update):
    """One fit-only model call plus six operator calls; aggregate-only result."""
    patients=np.asarray(ds.check_indices(patients),dtype=int)
    if (not len(patients) or len(patients)>32 or len(np.unique(patients))!=len(patients)
            or not np.isin(patients,ds.fit).all()):
        raise ValueError('diagnostic batch must contain 1..32 unique fit patients')
    if mode not in ['matched','wrongC','wrongO','constantC']:
        raise ValueError('a correction-bearing context mode is required')
    if not getattr(model,'has_correction',False):
        raise ValueError('a correction-bearing frozen model is required')
    captured={};handles=[];store=_Moments();was_training=model.training
    def capture(name):
        def hook(module,inputs,output):
            if name in captured:raise ValueError('unexpected repeated module call: '+name)
            captured[name]=(inputs[0].detach(),output.detach())
        return hook
    for name in ['p_value','o_value','correction_head']:
        handles.append(getattr(model,name).register_forward_hook(capture(name)))
    try:
        model.eval()
        with torch.no_grad():
            (risk,diag),owners=_call_batch(model,ds,patients,mode,False,True)
            if set(captured)!={'p_value','o_value','correction_head'}:
                raise ValueError('required message/head hooks did not execute')
            pv,ov=captured['p_value'][1],captured['o_value'][1]
            pm,om=diag['patch_mask'].bool(),diag['omics_mask'].bool()
            eligible=diag['correction_eligible_mask'].bool();delta=diag['delta']
            directions={};checks={};pooled=[]
            for name,transpose,a0key,a1key,gkey,value,qmask in [
                ('po',False,'base_p_to_o','p_to_o','gate_p_to_o',ov,pm&om.any(1)[:,None]),
                ('op',True,'base_o_to_p','o_to_p','gate_o_to_p',pv,om&pm.any(1)[:,None])]:
                allow=eligible.transpose(1,2) if transpose else eligible
                d=delta.transpose(1,2) if transpose else delta
                part=_direction_metrics(diag[a0key],diag[a1key],diag[gkey],d,allow,value,qmask,
                                        update_operator=update_operator)
                for key,(values,defined) in part['rows'].items():
                    _hierarchy(store,name+'_'+key,values,defined,owners,patients)
                message=part['message'];query=part['query']
                count=query.sum(-1).clamp_min(1)
                pool=torch.where(query[...,None],message,torch.zeros_like(message)).sum(1)/count[:,None]
                pooled.append(pool)
                mean_norm=(message.double().norm(dim=-1)*query).sum(-1)/count
                cancellation,_=_ratio(pool.double().norm(dim=-1),mean_norm)
                _wsi_hierarchy(store,name+'_delta_message_L2_after_query_average',pool.double().norm(dim=-1),owners,patients)
                _wsi_hierarchy(store,name+'_query_average_survival_ratio',cancellation,owners,patients)
                # Project each direction before pooling; the same bias-free head
                # is linear, so query-projection and query-averaging commute.
                weight=model.correction_head.weight[0,(slice(32,None) if transpose else slice(None,32))]
                projection=message@weight
                _hierarchy(store,name+'_head_projection_before_query_average',projection.double(),query,owners,patients)
                _hierarchy(store,name+'_head_projection_abs_before_query_average',projection.double().abs(),query,owners,patients)
                checks.update({name+'_'+key:value for key,value in part['checks'].items()})
                directions[name]=part
            message=torch.cat(pooled,-1)
            head_input,head_output=captured['correction_head']
            head_raw=head_output.squeeze(-1)
            checks['head_input_reconstruction_max_error']=float((message-head_input).abs().max().cpu())
            projected=(message@model.correction_head.weight.T).squeeze(-1)
            checks['head_projection_reconstruction_max_error']=float((projected-head_raw).abs().max().cpu())
            bounded=model.correction_cap*torch.tanh(head_raw)
            checks['risk_branch_reconstruction_max_error']=float((bounded-diag['correction_risk']).abs().max().cpu())
            tolerance=32*torch.finfo(message.dtype).eps
            if max(checks[k] for k in ['head_input_reconstruction_max_error',
                'head_projection_reconstruction_max_error','risk_branch_reconstruction_max_error'])>tolerance:
                raise ValueError('message/risk chain reconstruction failed')
            msg_norm=message.double().norm(dim=-1)
            head_bound=model.correction_head.weight.double().norm()*msg_norm
            head_efficiency,_=_ratio(head_raw.double().abs(),head_bound)
            tanh_efficiency,tanh_defined=_ratio(torch.tanh(head_raw.double()).abs(),head_raw.double().abs())
            tanh_efficiency=torch.where(tanh_defined,tanh_efficiency,torch.ones_like(tanh_efficiency))
            for key,value in {'message_L2_after_both_query_averages':msg_norm,
                'head_projection':head_raw.double(),'head_projection_absolute':head_raw.double().abs(),
                'head_projection_Cauchy_bound':head_bound,'head_projection_efficiency':head_efficiency,
                'tanh_survival_ratio':tanh_efficiency,'bounded_correction_risk':bounded.double(),
                'bounded_correction_risk_absolute':bounded.double().abs()}.items():
                _wsi_hierarchy(store,'chain_'+key,value,owners,patients)
            # Vector cancellation across WSIs is distinct from averaging norms.
            patient_message=torch.stack([message[torch.as_tensor(owners==p,device=message.device)].double().mean(0) for p in patients])
            patient_mean_norm=torch.stack([msg_norm[torch.as_tensor(owners==p,device=message.device)].mean() for p in patients])
            wsi_survival,_=_ratio(patient_message.norm(dim=-1),patient_mean_norm)
            store.add('chain_message_L2_after_patient_WSI_average',patient_message.norm(dim=-1))
            store.add('chain_patient_WSI_average_survival_ratio',wsi_survival)
            patient_risk=torch.stack([bounded[torch.as_tensor(owners==p,device=message.device)].double().mean() for p in patients])
            store.add('chain_actual_patient_correction_risk',patient_risk)
            store.add('chain_actual_patient_correction_risk_absolute',patient_risk.abs())
            checks['status']='PASS'
        return {'status':'PASS','patients':len(patients),'WSIs':len(owners),'mode':mode,
            'model_forward_calls':1,'operator_calls':6,'gradient_calls':0,'optimizer_steps':0,
            'moments':store.data,'metrics':store.summary(),'checks':checks}
    finally:
        for handle in handles:handle.remove()
        model.train(was_training)


def diagnose_checkpoint(model,ds,chunk_size=32,update_operator=conservative_update):
    """All fit patients, no endpoint/label access, no saved individual outputs."""
    if not 1<=int(chunk_size)<=32:raise ValueError('chunk size must be within 1..32')
    fit=np.asarray(ds.fit,dtype=int)
    if not len(fit) or len(np.unique(fit))!=len(fit):raise ValueError('invalid fit partition')
    store=_Moments();checks={};batches=0;wsi_count=0
    for start in range(0,len(fit),int(chunk_size)):
        part=diagnose_batch(model,ds,fit[start:start+int(chunk_size)],update_operator=update_operator)
        store.merge(part['moments']);batches+=1;wsi_count+=part['WSIs']
        for key,value in part['checks'].items():
            if key!='status':checks[key]=max(checks.get(key,0.),float(value))
    return {'status':'PASS','arm':str(model.arm),'fold':int(ds.fold),'role':str(ds.role),'seed':int(ds.seed),
        'fit_patients':len(fit),'fit_WSIs':wsi_count,'batches':batches,
        'additional_model_forward_calls':batches,'additional_operator_calls':6*batches,
        'additional_gradient_calls':0,'optimizer_steps':0,'metrics':store.summary(),'checks':checks,
        'scope':'frozen fit-only label-blind diagnostic; query/WSI/patient moments separately named; ratio undefined cases omitted with explicit counts',
        'constant_delta_scope':'original update evaluated with its weighted row mean only; recorded numerical residual, never relabeled literal zero',
        'centered_delta_scope':'subtract selected a0*gate-weighted row mean without clipping; test invariance of original update',
        'Taylor_scope':'rho*a0*gate*(delta-weighted_mean); first-order approximation, not assumed exact',
        'privacy':{'raw_maps_returned':False,'raw_patient_risks_returned':False,'patient_indices_returned':False,
                   'endpoint_arrays_read':False,'new_probe_fits':0,'persistent_hooks':False}}

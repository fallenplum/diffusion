"""Deterministic R7 full-risk training and full-train diagnostic gradients."""
from __future__ import annotations
import numpy as np
import torch
from src.bgb_en.r6_training import full_risk_step, saved_state, state_hash, metrics_for, describe, risk_diagnostics
from src.phase10_trimodal.math import torch_cox,log_weights

SCALARS=('risk','delta','main_delta','cross_delta','path_main_delta','gene_main_delta','hallmark_main_delta',
         'path_cross_delta','gene_cross_delta')
SLOT_KEYS=('slot_path_contributions','slot_gene_contributions')


def predict(model,indices,batch_size=8,control=None):
    model.eval()
    if control=='mean_query':model.refresh_training_query_mean()
    pieces={key:[] for key in SCALARS}
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            idx=torch.as_tensor(indices[start:start+batch_size],device=model.device)
            row=model(idx,control=control)
            for key in SCALARS:pieces[key].append(row[key].detach().cpu().numpy())
            for key in ('delta_E','delta_G'):
                pieces.setdefault(key+'_L2',[]).append(row[key].detach().flatten(1).norm(dim=-1).cpu().numpy())
            for key in SLOT_KEYS:
                value=row.get(key)
                if value is not None:
                    if value.shape!=(len(idx),4):raise ValueError('R7_FOUR_SLOT_CONTRIBUTION_SHAPE_REQUIRED')
                    pieces.setdefault(key,[]).append(value.detach().cpu().numpy())
            for key,value in row.get('diagnostic_patient_scalars',{}).items():
                if key not in ('gate_path_mean','gate_gene_mean','gate_path_sd','gate_gene_sd',
                               'gate_path_round2_mean','gate_gene_round2_mean',
                               'gate_path_valid_programs','gate_gene_valid_programs'):
                    raise ValueError('R7_UNEXPECTED_DIAGNOSTIC_SCALAR')
                if value.shape!=(len(idx),):raise ValueError('R7_DIAGNOSTIC_SCALAR_NOT_PATIENT_LEVEL')
                pieces.setdefault(key,[]).append(value.detach().cpu().numpy())
            del row
    result={key:np.concatenate(value).astype(np.float64) for key,value in pieces.items()}
    if any(not np.isfinite(value).all() or len(value)!=len(indices) for value in result.values()):
        raise RuntimeError('R7_NONFINITE_OR_INCOMPLETE_PREDICTION')
    return result


def branch_diagnostics(prediction,offset):
    result=risk_diagnostics(prediction,offset)
    result['quantities']={key:describe(value) for key,value in prediction.items() if key!='risk' and value.ndim==1}
    slots={}
    for key in SLOT_KEYS:
        value=prediction.get(key)
        if value is None:continue
        slots[key]=dict(per_program=[dict(program=k,**describe(value[:,k])) for k in range(4)],
            sum=describe(value.sum(-1)),sum_absolute=describe(np.abs(value).sum(-1)),
            nonzero_absolute_sum_patients=int((np.abs(value).sum(-1)>0).sum()),
            absolute_sum_over_sum_absolute_mean=(float(np.mean(np.abs(value.sum(-1))[np.abs(value).sum(-1)>0]
                /np.abs(value).sum(-1)[np.abs(value).sum(-1)>0])) if np.any(np.abs(value).sum(-1)>0) else None),
            cancellation_is_descriptive_not_performance=True)
    result['slot_contributions']=slots
    return result


def full_train_gradient_probe(model,prepared,raw,batch_size=8):
    """Accumulate vector gradients across ALL train microbatches, no optimizer."""
    previous=model.training;model.eval();train=np.asarray(prepared['train'])
    if not np.array_equal(train,model.fit_indices):raise ValueError('R7_GRADIENT_PROBE_TRAIN_SCOPE_CHANGED')
    batches=[torch.as_tensor(train[j:j+batch_size],device=model.device) for j in range(0,len(train),batch_size)]
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad];parameters=[p for _,p in named]
    with torch.no_grad():
        values=[]
        for idx in batches:
            row=model(idx);values.append(row['risk'].detach());del row
        risk=torch.cat(values);del values
    leaf=risk.detach().requires_grad_(True)
    loss=torch_cox(leaf,log_weights(raw['t'][train],raw['e'][train],model.device),
        torch.as_tensor(raw['e'][train],device=model.device))
    gradient=torch.autograd.grad(loss,leaf)[0].detach()
    paths_to_accumulate=('cross_delta','risk','path_cross_delta','gene_cross_delta','total_delta_L2','cross_delta_L2')
    accum={path:[torch.zeros_like(p) for p in parameters] for path in paths_to_accumulate}
    position=0
    for idx in batches:
        row=model(idx);n=len(idx)
        for path in paths_to_accumulate:
            if path=='total_delta_L2':target=.1*row['delta'].square().sum()/len(train)
            elif path=='cross_delta_L2':target=.1*row['cross_delta'].square().sum()/len(train)
            else:target=torch.dot(row[path],gradient[position:position+n])
            grads=torch.autograd.grad(target,parameters,allow_unused=True,retain_graph=path!=paths_to_accumulate[-1]) if target.requires_grad else [None]*len(parameters)
            for total,value in zip(accum[path],grads):
                if value is not None:total.add_(value.detach())
        position+=n;del row
    regularizer=model.regularization()
    rg=torch.autograd.grad(regularizer,parameters,allow_unused=True) if regularizer.requires_grad else [None]*len(parameters)
    accum['parameter_L2']=[torch.zeros_like(p) if value is None else value.detach() for p,value in zip(parameters,rg)]
    paths={}
    for path,grads in accum.items():
        groups={}
        for (name,p),value in zip(named,grads):
            group='.'.join(name.split('.')[:2]);cell=groups.setdefault(group,dict(tensors=0,numel=0,nonzero=0,L2_squared=0.))
            cell['tensors']+=1;cell['numel']+=p.numel();cell['nonzero']+=int(bool(torch.any(value!=0)))
            cell['L2_squared']+=float(value.double().square().sum())
        paths[path]=groups
    model.train(previous)
    return dict(scope='complete_train_riskset_Cox_vector_accumulated_all_microbatches',patients=len(train),
        microbatches=len(batches),groups_by_path=paths,cox_loss=float(loss.detach()),
        parameter_regularizer=float(regularizer.detach()),parameter_L2_uses_actual_slot_one_quarter_geometry=True,
        risk_penalties_excluded_from_pure_Cox_paths=True,new_optimizer_updates=0,
        pure_Cox_paths=['cross_delta','risk','path_cross_delta','gene_cross_delta'],
        residual_penalty_paths=['total_delta_L2','cross_delta_L2'],
        diagnostic_forward_calls=2*len(batches),diagnostic_gradient_calls_per_microbatch=6,
        independent_parameter_L2_gradient_calls=1,
        norm_differences_are_not_vector_gradient_differences=True)


def training_query_probe(model,prepared,max_patients=8):
    """Fixed first TRAIN rows only, including epoch60; anonymous summaries."""
    indices=np.asarray(prepared['train'])[:max_patients]
    if not np.array_equal(np.asarray(prepared['train']),model.fit_indices):raise ValueError('R7_QUERY_PROBE_TRAIN_SCOPE_CHANGED')
    previous=model.training;model.eval();idx=torch.as_tensor(indices,device=model.device)
    controls=('matched_query','wrong_query')+(() if model.arm in ('RAW','ADD') else ('wrong_path',))
    def masked_summary(value,mask):
        value,mask=value.detach(),mask.bool()
        if value.shape!=mask.shape:raise ValueError('R7_DIAGNOSTIC_VALID_SHAPE_CHANGED')
        per_patient=[]
        for i in range(len(value)):
            if mask[i].any():per_patient.append(float(value[i][mask[i]].mean()))
        return dict(valid_items=int(mask.sum()),patients_with_valid_items=len(per_patient),
            mean=float(np.mean(per_patient)) if per_patient else None,sd=float(np.std(per_patient)) if per_patient else None)
    def norm_summary(value,mask):return masked_summary(value.detach().norm(dim=-1),mask)
    outputs={};summaries={};tv={}
    with torch.no_grad():
        for tag in controls:
            row=model(idx,control=tag);outputs[tag]=row
            own=row['own_program_mask'];source=row['query_source_mask']
            pm=row['patch_base_read']['program_mask']&row['patch_cond_read']['program_mask']
            values=dict(raw_G0=norm_summary(row['G0'],own),RNA_context=norm_summary(row['query_context_used'],source),
                base_query=norm_summary(row['base_queries'],source),condition_query=norm_summary(row['conditioned_queries'],source),
                query_delta=norm_summary(row['conditioned_queries']-row['base_queries'],source),
                P0=norm_summary(row['Ebase'],pm),P_update=norm_summary(row['delta_E'],pm))
            reverse=row['reverse_cond_read']
            if reverse is not None:
                gm=row['reverse_base_read']['program_mask']&reverse['program_mask']
                values.update(reverse_query=norm_summary(reverse['query'],gm),G_reverse_base=norm_summary(row['reverse_base_read']['tokens'],gm),
                    G_update=norm_summary(row['delta_G'],gm))
            patch,_=model.patch_bank.select(idx);patch_mask=patch['patch_mask']
            patch_norm=row['patch_values'].norm(dim=-1);patient=[]
            for b in range(len(idx)):
                wsi=[float(patch_norm[b,s,patch_mask[b,s]].mean()) for s in range(patch_mask.shape[1]) if patch_mask[b,s].any()]
                if wsi:patient.append(float(np.mean(wsi)))
            values['raw_patch_mean_L2']=dict(patients=len(patient),mean=float(np.mean(patient)) if patient else None,
                sd=float(np.std(patient)) if patient else None)
            reader=row['patch_cond_read'];values['null']=masked_summary(reader['null'],reader['valid_WSI'])
            rounds=[]
            for step in row.get('state_trace',[]):
                mpath=row['patch_base_read']['program_mask'];mgene=own
                rounds.append(dict(round=step['round'],P_before=norm_summary(step['P_before'],mpath),P_after=norm_summary(step['P_after'],mpath),
                    G_before=norm_summary(step['G_before'],mgene),G_after=norm_summary(step['G_after'],mgene),
                    raw_delta_P=norm_summary(step['raw_delta_P'],mpath),raw_delta_G=norm_summary(step['raw_delta_G'],mgene),
                    RNA_route=norm_summary(step['RNA_route'],step['RNA_route_mask']),PATH_route=norm_summary(step['PATH_route'],step['PATH_route_mask']),
                    path_gate=masked_summary(step['path_gate'].squeeze(-1),mpath),gene_gate=masked_summary(step['gene_gate'].squeeze(-1),mgene)))
            summaries[tag]=dict(interfaces=values,rounds=rounds,query_source_valid_programs=int(source.sum()),own_valid_programs=int(own.sum()))
        matched=outputs['matched_query']
        for tag in controls[1:]:
            other=outputs[tag]
            invariant=all(torch.equal(matched[k],other[k]) for k in ('patch_values','gene_values','gene_measured_mask','main_delta','offset'))
            if not invariant:raise RuntimeError('R7_QUERY_PROBE_CHANGED_OWN_VALUES_MAIN_OR_OFFSET')
            a,b=matched['patch_cond_read'],other['patch_cond_read']
            change=.5*((a['weights']-b['weights']).abs().sum(-1)+(a['null']-b['null']).abs())
            tv[tag]=dict(patch_plus_null_TV=masked_summary(change,a['valid_WSI']&b['valid_WSI']),
                risk_difference_SD=float((matched['risk']-other['risk']).std(unbiased=False)),own_values_main_offset_exact=True,
                frozen_not_retrained=True,RNA_query_wrong_is_not_reverse_path_wrong=True)
    model.train(previous)
    return dict(scope='fixed_first_eight_current_train_rows_no_validation_no_outcome_selection',patients=len(indices),
        controls=summaries,matched_vs_controls=tv,diagnostic_forward_calls=len(controls),new_backward_calls=0,new_optimizer_updates=0,
        fixed_six_RMS={name:float(getattr(model.core,name)) for name in
            ('context_rms','patch_rms','query_rms','gene_key_rms','evidence_rms','reverse_query_rms')},
        fixed_initial_full_train_calibration=getattr(model,'calibration',None),
        no_patient_identifiers_or_raw_attention_exported=True)


def checkpoint_diagnostics(model,prepared,raw,epoch,*,include_validation,train_prediction=None,valid_prediction=None):
    model.refresh_training_query_mean()
    train=train_prediction if train_prediction is not None else predict(model,prepared['train'])
    result=dict(epoch=epoch,optimizer_updates_completed=epoch,
        train=branch_diagnostics(train,prepared['base'][prepared['train']]),
        gradient_probe=full_train_gradient_probe(model,prepared,raw),
        query_probe=training_query_probe(model,prepared),parameter_budget=model.parameter_budget(),validation_ranking_used_for_selection=False)
    if include_validation:
        valid=valid_prediction if valid_prediction is not None else predict(model,prepared['valid'])
        result['validation']=branch_diagnostics(valid,prepared['base'][prepared['valid']])
    else:result['validation']='NOT_EVALUATED_AT_DIAGNOSTIC_ONLY_EPOCH'
    return result

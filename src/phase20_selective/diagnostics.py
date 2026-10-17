"""Label-free sampling and aggregate-only frozen Phase20 mechanism inspection.

No endpoint arrays, raw predictions/maps or patient indices are returned. Model
and inherited anchor weights were supervised previously; these diagnostics do
not create a new survival target, fit a probe, or read survival labels.
"""
import math

import numpy as np
import torch


def gradient_probe_indices(fit):
    fit = np.asarray(fit, int)
    if not len(fit) or len(np.unique(fit)) != len(fit):
        raise ValueError('nonempty unique fit patients required')
    return fit[np.unique(np.linspace(0, len(fit)-1, min(8,len(fit))).astype(int))]


def _array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class _Stats:
    def __init__(self):
        self.values = {}

    def add(self, name, values):
        x = np.asarray(_array(values), np.float64).ravel()
        if not len(x):
            return
        if not np.isfinite(x).all():
            raise ValueError('nonfinite diagnostic: '+name)
        old = self.values.setdefault(name, [0,0.,0.,math.inf,-math.inf])
        old[0] += len(x); old[1] += float(x.sum()); old[2] += float(x@x)
        old[3] = min(old[3],float(x.min())); old[4] = max(old[4],float(x.max()))

    def output(self):
        out = {}
        for key, (n,total,square,low,high) in self.values.items():
            mean = total/n
            out.update({key+'_count':int(n),key+'_mean':float(mean),
                        key+'_SD':float(max(square/n-mean*mean,0.)**.5),
                        key+'_min':float(low),key+'_max':float(high)})
        return out


def _map_summary(stats, diag, has_correction, prefix=''):
    pm, om = _array(diag['patch_mask']).astype(bool), _array(diag['omics_mask']).astype(bool)
    eligible = _array(diag['correction_eligible_mask']).astype(bool)
    for label, transpose, a0key, a1key, gkey in [
        ('po',False,'base_p_to_o','p_to_o','gate_p_to_o'),
        ('op',True,'base_o_to_p','o_to_p','gate_o_to_p')]:
        a0, a1, gate = [np.asarray(_array(diag[k]),np.float64) for k in [a0key,a1key,gkey]]
        allow = eligible.transpose(0,2,1) if transpose else eligible
        query = (om & pm.any(1)[:,None]) if transpose else (pm & om.any(1)[:,None])
        n = allow.sum(-1); active = query & (n > 0)
        change = a1-a0; selected = (a0*gate).sum(-1)
        tv = .5*np.abs(change).sum(-1)
        key = prefix+label+'_'
        stats.add(key+'eligible_candidates',n[query])
        stats.add(key+'attention_TV',tv[query])
        stats.add(key+'TV_bound',.5*selected[query])
        stats.add(key+'TV_bound_violation',np.maximum(tv-.5*selected,0.)[query])
        stats.add(key+'probability_row_sum_error',np.abs(a1.sum(-1)-a0.sum(-1))[query])
        stats.add(key+'normalized_row_error',np.abs(a1.sum(-1)-1.)[query])
        stats.add(key+'minimum_probability',a1.min(-1)[query])
        stats.add(key+'unselected_probability_abs_error',np.abs(change)[(gate==0) | ~allow])
        target = .25*n if has_correction else np.zeros_like(n)
        stats.add(key+'gate_mass_error',np.abs(gate.sum(-1)-target)[query])
        stats.add(key+'gate_support_fraction',((gate > 0)&allow).sum(-1)[active]/n[active])
        stats.add(key+'gate_mass_fraction',gate.sum(-1)[active]/n[active])
        stats.add(key+'selected_base_probability_mass',selected[query])
        positive_bound = query & (selected > 0)
        stats.add(key+'TV_over_rho_M',tv[positive_bound]/(.5*selected[positive_bound]))
        stats.add(key+'zero_selected_mass_row',selected[query] == 0)
        raw = np.asarray(_array(diag['gate_score']),np.float64)
        if transpose:
            raw = raw.transpose(0,2,1)
        stats.add(key+'raw_gate_score',raw[allow])
        score_key = 'normalized_gate_score_o' if transpose else 'normalized_gate_score_p'
        if score_key in diag:
            normalized = np.asarray(_array(diag[score_key]),np.float64)
            stats.add(key+'normalized_gate_score',normalized[allow])
            for kind,score in [('raw',raw),('normalized',normalized)]:
                mean = (score*allow).sum(-1)/np.maximum(n,1)
                variance = (((score-mean[...,None])**2)*allow).sum(-1)/np.maximum(n,1)
                stats.add(key+kind+'_gate_score_row_population_SD',np.sqrt(variance)[active])
    stats.add(prefix+'delta_absolute',np.abs(_array(diag['delta']))[eligible])


def _selection_summary(stats, diag, owners, patients, prefix=''):
    """Locked WSI-query -> equal-WSI patient -> all-patient selection summary.

    The primary support summary includes every eligible>=2 row, including
    constant scores. All-base-query summaries also retain low/empty support.
    Rows without >=2 candidates are counted, never silently selected away.
    """
    pm = _array(diag['patch_mask']).astype(bool)
    om = _array(diag['omics_mask']).astype(bool)
    eligible = _array(diag['correction_eligible_mask']).astype(bool)
    raw = np.asarray(_array(diag['gate_score']),np.float64)
    for label,transpose,gkey in [('po',False,'gate_p_to_o'),('op',True,'gate_o_to_p')]:
        allow = eligible.transpose(0,2,1) if transpose else eligible
        score = raw.transpose(0,2,1) if transpose else raw
        gate = np.asarray(_array(diag[gkey]),np.float64)
        query = (om & pm.any(1)[:,None]) if transpose else (pm & om.any(1)[:,None])
        n = allow.sum(-1)
        support = ((gate>1e-6)&allow).sum(-1)/np.maximum(n,1)
        mean = np.where(allow,score,0.).sum(-1)/np.maximum(n,1)
        sd = np.sqrt((np.where(allow,score-mean[...,None],0.)**2).sum(-1)/np.maximum(n,1))
        entries = [
            ('support',support,query&(n>=2)),
            ('all_rows_support',support,query),
            ('eligible_lt2',(n<2).astype(float),query),
            ('raw_score_degenerate',(sd<=1e-6).astype(float),query)]
        for name,value,mask in entries:
            key = prefix+label+'_selectivity_'+name
            stats.add(key+'_query',value[mask])
            wsi_means = np.array([value[i][mask[i]].mean() if mask[i].any() else np.nan
                                  for i in range(len(owners))])
            stats.add(key+'_WSI',wsi_means[np.isfinite(wsi_means)])
            patient_means = []
            for patient in patients:
                selected = wsi_means[owners==patient]
                selected = selected[np.isfinite(selected)]
                if len(selected):
                    patient_means.append(float(selected.mean()))
            stats.add(key+'_patient',patient_means)


def _compare_maps(stats, matched, changed, label):
    pm, om = _array(matched['patch_mask']).astype(bool), _array(matched['omics_mask']).astype(bool)
    ea = _array(matched['correction_eligible_mask']).astype(bool)
    eb = _array(changed['correction_eligible_mask']).astype(bool)
    common = ea & eb
    for name, qmask, akey, gkey, transpose in [
        ('po',pm&om.any(1)[:,None],'p_to_o','gate_p_to_o',False),
        ('op',om&pm.any(1)[:,None],'o_to_p','gate_o_to_p',True)]:
        a,b = [np.asarray(_array(d[akey]),np.float64) for d in [matched,changed]]
        stats.add(label+'_'+name+'_attention_L1',np.abs(a-b).sum(-1)[qmask])
        ga,gb = [np.asarray(_array(d[gkey]),np.float64) for d in [matched,changed]]
        overlap = common.transpose(0,2,1) if transpose else common
        stats.add(label+'_'+name+'_common_eligible_gate_abs_difference',np.abs(ga-gb)[overlap])
    stats.add(label+'_eligible_pair_loss_fraction',
              (ea.sum((1,2))-eb.sum((1,2)))/np.maximum(ea.sum((1,2)),1))


def _patient_rows(diag, key, owners, patients):
    x = np.asarray(_array(diag[key]),np.float64)
    if len(x) != len(owners):
        raise ValueError('WSI diagnostic ownership mismatch')
    return np.stack([x[owners==patient].mean(0) for patient in patients])


def _covariance_stats(values):
    x = np.array(values,dtype=np.float64,copy=True).reshape(len(values),-1)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite token representation')
    x -= x.mean(0,keepdims=True)
    gram = x@x.T if len(x) <= x.shape[1] else x.T@x
    gram = (gram+gram.T)*.5
    eigen = np.maximum(np.linalg.eigvalsh(gram),0.)
    total = float(eigen.sum())
    if total <= 1e-25:
        return {'effective_rank':0.,'participation_rank':0.,'first_PC_fraction':0.,
                'total_population_variance':0.,'mean_feature_variance':0.,'covariance_numerical_rank':0}
    q = eigen[eigen > 0]/total
    return {'effective_rank':float(np.exp(-(q*np.log(q)).sum())),
            'participation_rank':float(1/(q*q).sum()),'first_PC_fraction':float(q[-1]),
            'total_population_variance':total/len(x),'mean_feature_variance':total/(len(x)*x.shape[1]),
            'covariance_numerical_rank':int((eigen > eigen.max()*1e-10).sum())}


def _rank(matrix):
    singular = np.linalg.svd(np.asarray(matrix,np.float64),compute_uv=False)
    if not len(singular) or singular[0] <= 1e-14:
        return 0
    return int((singular > singular[0]*1e-8).sum())


def _filter_diagnostics(model, ds, fit):
    member = np.asarray(_array(model.molecular.membership),bool)
    union = member.any(0)
    result = {'molecular_input_genes':member.shape[1], 'RNA_program_union_genes':int(union.sum()),
              'RNA_genes_outside_program_union':int((~union).sum()),
              'programs_with_fewer_than16_genes':int((member.sum(1)<16).sum()),
              'signed_filters_available':hasattr(model.molecular,'filters')}
    masks = [np.asarray(_array(ds.ft[k])[fit],bool) for k in ['RM','CM','CM']]
    nominal_rna = []
    stats = _Stats()
    for channel,name in enumerate(['RNA','CNA_gain','CNA_loss']):
        for program in range(8):
            ix = np.flatnonzero(member[program])
            patterns, counts = np.unique(masks[channel][:,ix],axis=0,return_counts=True)
            if hasattr(model.molecular,'filters'):
                raw = np.asarray(_array(model.molecular.filters[channel*8+program]),np.float64)
                for pattern,count in zip(patterns,counts):
                    measured = raw*pattern[None]
                    norm = np.sqrt((measured*measured).sum(1,keepdims=True))
                    before = measured/np.where(norm>0,norm,1.)
                    after = np.broadcast_to(4*before.mean(0,keepdims=True),before.shape) if model.rank_one else before
                    stats.add(name+'_measured_filter_rank_before',np.full(count,_rank(before)))
                    stats.add(name+'_measured_response_map_rank_after',np.full(count,_rank(after)))
                norm = np.sqrt((raw*raw).sum(1,keepdims=True))
                nominal = raw/np.where(norm>0,norm,1.)
                if model.rank_one:
                    nominal = 4*nominal.mean(0,keepdims=True)
            else:
                logits = np.asarray(_array(model.molecular.logits[channel,program]),np.float64)[ix]
                weights = np.exp(logits-logits.max()); nominal = (weights/weights.sum())[None]
            if channel==0:
                embedded = np.zeros((len(nominal),member.shape[1]),np.float64)
                embedded[:,ix] = nominal; nominal_rna.append(embedded)
    result.update(stats.output())
    blocks = getattr(ds,'blocks',getattr(ds,'audit',{}).get('X_blocks',{}))
    if 'R' not in blocks or not hasattr(ds,'anchor_w'):
        return {**result,'RNA_anchor_beta_projection_available':False,
                'RNA_anchor_beta_projection_scope':'R block or inherited J0 coefficients unavailable; no inference'}
    start,end = blocks['R']
    beta = np.asarray(ds.anchor_w,np.float64)[start:end]
    if beta.shape != (member.shape[1],):
        raise ValueError('RNA anchor coefficient block differs from tokenizer gene ordering')
    directions = np.concatenate(nominal_rna)
    _,s,v = np.linalg.svd(directions,full_matrices=False)
    rank = int((s>s.max()*1e-8).sum()) if len(s) and s.max()>1e-14 else 0
    basis = v[:rank]; total = float(beta@beta)
    result.update(RNA_anchor_beta_projection_available=True,
        RNA_anchor_beta_squared_norm=total,
        RNA_anchor_beta_program_union_energy_fraction=float(beta[union]@beta[union]/total) if total else 0.,
        RNA_anchor_beta_nominal_filter_span_rank=rank,
        RNA_anchor_beta_nominal_filter_span_energy_fraction=float(np.linalg.norm(basis@beta)**2/total) if total else 0.,
        RNA_anchor_beta_projection_scope='Euclidean coefficient geometry only; nominal all-genes-measured filter responses; not explained DSS information, a patient-specific bound, or a distilled target')
    return result


def inspect_model(model, ds):
    """Inspect all fit queries in chunks and eight true context-input gradients."""
    from .train import forward
    fit = np.asarray(ds.fit,int)
    gradient_ix = gradient_probe_indices(fit)
    if hasattr(ds,'gradient_probe_indices') and not np.array_equal(gradient_ix,ds.gradient_probe_indices):
        raise ValueError('gradient query selection differs from locked data metadata')
    was_training = model.training
    model.eval()
    stats, risks, representations = _Stats(), {key:[] for key in ['matched','off','anchor','wrongC','wrongO','constantC','branch_pair','branch_correction']}, {}
    try:
        with torch.no_grad():
            for begin in range(0,len(fit),32):
                patients = fit[begin:begin+32]
                slides = np.flatnonzero(np.isin(ds.owner,patients)); owners = ds.owner[slides]
                matched,dg = forward(model,ds,patients,'matched',diagnostic=True)
                if len(dg)!=1:
                    raise ValueError('diagnostic outer chunk exceeded fixed training microbatch32')
                matched_diag = dg[0]
                risks['matched'].append(_array(matched))
                risks['branch_pair'].append(_patient_rows(matched_diag,'pair_risk',owners,patients))
                risks['branch_correction'].append(_patient_rows(matched_diag,'correction_risk',owners,patients))
                _map_summary(stats,matched_diag,model.has_correction)
                _selection_summary(stats,matched_diag,owners,patients)
                for key in ['tokens','summary16_before','summary16_after','token_coverage']:
                    if key in matched_diag:
                        representations.setdefault(key,[]).append(_patient_rows(matched_diag,key,owners,patients))
                if 'filter_measured_norms' in matched_diag:
                    stats.add('filter_measured_norm',matched_diag['filter_measured_norms'])
                for mode in ['wrongC','wrongO']:
                    risk,other = forward(model,ds,patients,mode,diagnostic=True)
                    risks[mode].append(_array(risk))
                    _map_summary(stats,other[0],model.has_correction,prefix=mode+'_')
                    _selection_summary(stats,other[0],owners,patients,prefix=mode+'_')
                    _compare_maps(stats,matched_diag,other[0],mode)
                    del other
                for mode in ['off','anchor','constantC']:
                    risks[mode].append(_array(forward(model,ds,patients,mode)))
                if model.arm in ['P1','P16']:
                    for mode in ['fullOmicsWrong','uniformPair']:
                        changed = _array(forward(model,ds,patients,mode))
                        stats.add('B_'+mode+'_risk_absolute_change',np.abs(changed-_array(matched)))
                del matched_diag,dg
        values = {key:np.concatenate(parts).astype(np.float64) for key,parts in risks.items()}
        pair,correction = values['off']-values['anchor'],values['matched']-values['off']
        scale = float(model.risk_scale.detach().cpu())
        out = {'diagnostic_mode':'matched absolute context, including models trained under wrong context',
            'fit_probe_patients':len(fit),'fit_probe_WSIs':int(np.isin(ds.owner,fit).sum()),
            'anchor_risk_SD':float(values['anchor'].std()),'pair_risk_SD':float(pair.std()),
            'correction_risk_SD':float(correction.std()),
            'correction_SD_over_anchor_SD':float(correction.std()/values['anchor'].std()) if values['anchor'].std()>0 else 0.,
            'correction_SD_ratio_anchor_nonzero':bool(values['anchor'].std()>0),
            'matched_wrongC_risk_MAE':float(np.abs(values['matched']-values['wrongC']).mean()),
            'matched_wrongO_risk_MAE':float(np.abs(values['matched']-values['wrongO']).mean()),
            'constantC_off_max_absolute_error':float(np.abs(values['constantC']-values['off']).max()),
            'pair_risk_absolute_max':float(np.abs(pair).max()),
            'correction_risk_absolute_max':float(np.abs(correction).max()),
            'pair_final_cap_violation':float(max(np.abs(pair).max()-scale,0.)),
            'correction_final_cap_violation':float(max(np.abs(correction).max()-.25*scale,0.)),
            'pair_branch_cap_violation':float(max(np.abs(values['branch_pair']).max()-scale,0.)),
            'correction_branch_cap_violation':float(max(np.abs(values['branch_correction']).max()-.25*scale,0.)),
            'final_subtraction_branch_roundoff_max':float(max(np.abs(pair-values['branch_pair']).max(),
                                                            np.abs(correction-values['branch_correction']).max())),
            'gate_budget_applicable':bool(model.has_correction),
            'map_summary_unit':'pooled directional query rows across fit WSIs; risk/Token summaries equal WSI mean within patient',
            'selectivity_summary_unit':'query mean within WSI, equal WSI mean within patient, equal mean over all eligible fit patients; separate pooled-query summaries are not the behavior gate',
            'selectivity_support_threshold':1e-6,
            'selectivity_minimum_eligible_candidates':2,
            'selectivity_raw_score_degenerate_SD_floor':1e-6,
            'selectivity_missing_scope':'primary support excludes rows with <2 candidates by prelocked definition; all-row support, low-candidate and constant-score rates retain all base-valid queries; absent-query WSIs/patients omitted only from undefined averages and counts disclosed',
            'survival_labels_read':False,'new_probe_or_teacher_fits':0,'raw_maps_persisted':False,
            'covariance_rank_scope':'population centered feature covariance; nonlinear post-MLP covariance rank is not input Jacobian rank; mask-pattern variation retained',
            **stats.output()}
        for prefix in ['', 'wrongC_', 'wrongO_']:
            for direction in ['po','op']:
                key = prefix+direction+'_selectivity_'
                out[key+'eligible_ge2_query_count'] = int(out.get(key+'support_query_count',0))
        for key,parts in representations.items():
            x = np.concatenate(parts)
            for channel,name in [(None,'all'),(0,'RNA'),(1,'CNA_gain'),(2,'CNA_loss')]:
                subset = x if channel is None else x[:,channel*8:(channel+1)*8]
                out.update({name+'_'+key+'_'+k:v for k,v in _covariance_stats(subset).items()})
        if 'summary16_before' in representations:
            before,after = [np.concatenate(representations[key]) for key in ['summary16_before','summary16_after']]
            denom = float((before*before).sum())
            out['summary16_empirical_after_before_energy_ratio'] = float((after*after).sum()/denom) if denom else 0.
            out['summary16_energy_match_scope'] = 'actual fit-response energy ratio; isotropic nominal matching does not force this value to one'
        out.update(_filter_diagnostics(model,ds,fit))
        # Override leaves are distinct from own main/base inputs. Autograd on the
        # total risk therefore isolates the only context-connected new correction.
        clinical = ds.ft['C'].detach().clone().requires_grad_(True)
        rna = ds.ft['R'].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            target = forward(model,ds,gradient_ix,'matched',
                     context_overrides={'clinical':clinical,'R':rna})
            gc,gr = torch.autograd.grad(target.sum(),(clinical,rna),allow_unused=True)
        cgrad = clinical.new_zeros((len(gradient_ix),clinical.shape[1])) if gc is None else gc[gradient_ix]
        rgrad = rna.new_zeros((len(gradient_ix),rna.shape[1])) if gr is None else gr[gradient_ix]
        if not torch.isfinite(cgrad).all() or not torch.isfinite(rgrad).all():
            raise ValueError('nonfinite isolated context-input gradient')
        out.update(isolated_correction_input_gradient_available=bool(model.has_correction),
            isolated_gradient_query_patients=len(gradient_ix),
            isolated_gradient_query_WSIs=int(np.isin(ds.owner,gradient_ix).sum()),
            isolated_correction_clinical_gradient_L2=float(cgrad.norm().detach().cpu()),
            isolated_correction_RNA_gradient_L2=float(rgrad.norm().detach().cpu()),
            isolated_correction_clinical_per_patient_gradient_L2_mean=float(cgrad.norm(dim=1).mean().detach().cpu()),
            isolated_correction_RNA_per_patient_gradient_L2_mean=float(rgrad.norm(dim=1).mean().detach().cpu()),
            isolated_gradient_target='sum of eight matched patient risks; only new correction reads overridden C/R leaves; no raw target output',
            isolated_gradient_no_correction_scope='unused leaves yield zero with unavailable flag; no claim of broken gradients in no-correction arms')
        return out
    finally:
        model.train(was_training)

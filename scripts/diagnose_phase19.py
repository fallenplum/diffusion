"""Private, label-blind frozen MMP token inspection. No model fit or mutation."""
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(2)
P = Path('/root/autodl-tmp/phase19_attention_private')
R = Path('/root/phase16_routing_repo/reports/phase19_clinical_trilinear_attention')
V = Path('/root/autodl-tmp/phase17_ssd_private/visual')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def ah(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def gram(value):
    x = np.array(value, dtype=np.float64, copy=True).reshape(len(value), -1)
    x -= x.mean(0, keepdims=True)
    return x @ x.T


def spectrum(k, positions, dimensions):
    g = np.asarray(k[np.ix_(positions, positions)], np.float64)
    g = g-g.mean(0, keepdims=True)-g.mean(1, keepdims=True)+g.mean()
    g = (g+g.T)/2
    eig = np.linalg.eigvalsh(g)
    total = max(float(np.trace(g)), 0.)
    eig = np.maximum(eig, 0.)
    if total <= 1e-25:
        return {'total_feature_variance': 0., 'mean_feature_variance': 0.,
                'effective_rank_entropy_covariance': 0., 'effective_rank_participation': 0.,
                'first_PC_variance_fraction': 0., 'numerical_rank_relative1e12': 0}
    q = eig[eig > 0]/eig.sum()
    return {'total_feature_variance': total/len(positions),
            'mean_feature_variance': total/(len(positions)*dimensions),
            'effective_rank_entropy_covariance': float(np.exp(-(q*np.log(q)).sum())),
            'effective_rank_participation': float(1/(q*q).sum()),
            'first_PC_variance_fraction': float(q[-1]),
            'numerical_rank_relative1e12': int((eig > eig.max()*1e-12).sum())}


def qstats(values, prefix):
    a = np.asarray(values, np.float64).ravel()
    return {prefix+'_mean':float(a.mean()), prefix+'_min':float(a.min()),
            prefix+'_p05':float(np.quantile(a,.05)), prefix+'_median':float(np.median(a)),
            prefix+'_p95':float(np.quantile(a,.95)), prefix+'_max':float(a.max())}


def pair_differences(values, prefix):
    x = np.asarray(values, np.float64)
    norms = np.sqrt((x*x).sum(-1))
    k = x @ x.transpose(0, 2, 1)
    cos = k/np.maximum(norms[:,:,None]*norms[:,None,:],1e-30)
    dist = np.sqrt(np.maximum(norms[:,:,None]**2+norms[:,None,:]**2-2*k,0.))
    tri = np.triu_indices(x.shape[1],1)
    cos, dist = cos[:,tri[0],tri[1]], dist[:,tri[0],tri[1]]
    relative = np.sqrt(((x-x.mean(1,keepdims=True))**2).sum((1,2))) / np.maximum(
               np.sqrt((x*x).sum((1,2))),1e-30)
    return {**qstats(cos,prefix+'_pairwise_cosine'), **qstats(dist,prefix+'_pairwise_L2'),
            **qstats(relative,prefix+'_between_prototype_relative_Frobenius')}


def main():
    previous_path = R/'MMP_TOKEN_DIAGNOSTICS.json'
    previous = read(previous_path) if previous_path.exists() else None
    lock = read(R/'PROTOCOL_LOCK.json'); source = lock['source']
    ready = read(V/'READY.json')
    assert sha(V/'owner.npy') == ready['hashes']['owner']
    assert sha(V/'mask.npy') == ready['hashes']['mask']
    owner = np.load(V/'owner.npy'); masks = np.load(V/'mask.npy')
    visual = np.load(V/'u.npy', mmap_mode='r')
    snapshots = sorted((P/'baseline_features').glob('MMP_*.json'))
    rows, audits, ignored = [], [], []
    for receipt_path in snapshots:
        receipt = read(receipt_path); provenance = receipt['provenance']
        if provenance['preprocess_audit'].get('source') != source:
            ignored.append(receipt['sha256']); continue
        path = receipt_path.with_suffix('.pt')
        assert sha(path) == receipt['sha256']
        bank = torch.load(path, map_location='cpu', weights_only=False)
        assert bank['provenance'] == provenance
        role, fold = provenance['role'], int(provenance['fold'])
        prep_path = P/'preprocessing'/f'fold{fold}_{role}.pt'
        # This bank contains preprocessing states/features and partition indices,
        # not survival endpoints. Do not open training checkpoints or OOF files.
        prep = torch.load(prep_path, map_location='cpu', weights_only=False)
        assert prep['provenance']['source'] == source
        assert sha(prep_path) == provenance['preprocess_audit']['cache_sha256']
        fit, held, active = [np.asarray(prep[key],int) for key in ['fit','valid','active']]
        wsis = np.asarray(bank['global_wsi_indices'],int)
        assert np.array_equal(wsis,np.flatnonzero(np.isin(owner,active)))
        owned, mask = owner[wsis], masks[wsis]
        assert ah(owned) == provenance['owner_hash'] and ah(mask) == provenance['mask_hash']
        assert ah(fit) == provenance['fit_hash']
        tokens, centers = np.asarray(bank['tokens']), np.asarray(bank['centroids'])
        d = visual.shape[-1]; assert d == 1536
        assert tokens.shape == (len(wsis),16,2*d+1) and centers.shape == (16,d)
        assert np.isfinite(tokens).all() and np.isfinite(centers).all()
        prob, mean, cov = tokens[:,:,0], tokens[:,:,1:d+1], tokens[:,:,d+1:]
        assert (prob > 0).all() and (cov > 0).all()
        assert np.abs(prob.sum(1)-1).max() < 1e-5
        fit_local = np.flatnonzero(np.isin(owned,fit))
        eligible = np.argwhere(mask[fit_local])
        mc = provenance['parameters']
        choice = np.random.default_rng(mc['kmeans_seed']).choice(len(eligible),
                   min(mc['initial_pool_max'],len(eligible)),replace=False)
        pairs = eligible[choice]
        selected_wsis = wsis[fit_local[pairs[:,0]]]
        sample = np.asarray(visual[selected_wsis,pairs[:,1]],np.float32)
        assert ah(sample) == provenance['sample_hash']
        feature_var = sample.astype(np.float64).var(0)
        prior_var = float(mc['prior_covariance_scale'])
        scale = {**qstats(np.linalg.norm(sample,axis=1),'fit_sample_UNI2_L2norm'),
                 **qstats(feature_var,'fit_sample_per_dimension_variance'),
                 'prior_covariance_scale':prior_var,
                 'prior_covariance_over_mean_feature_variance':prior_var/float(feature_var.mean()),
                 'prior_covariance_over_median_feature_variance':prior_var/float(np.median(feature_var)),
                 'fit_sample_patches':len(sample), 'sample_hash_reconstructed_exactly':True}
        patient_ids = np.unique(owned)
        pt = np.stack([tokens[owned == pi].mean(0,dtype=np.float64) for pi in patient_ids])
        pp, pm, pc = pt[:,:,0], pt[:,:,1:d+1], pt[:,:,d+1:]
        grams = {'mixture_probability':gram(pp),'prototype_mean':gram(pm),'prototype_covariance':gram(pc)}
        grams['combined_tokens'] = sum(grams.values())
        dimensions = {'mixture_probability':16,'prototype_mean':16*d,
                      'prototype_covariance':16*d,'combined_tokens':16*(2*d+1)}
        for scope, ids in [('fit',fit),('held_out',held),('active',active)]:
            pos = np.flatnonzero(np.isin(patient_ids,ids))
            assert len(pos) == len(ids)
            pp_s, pm_s, pc_s = pp[pos], pm[pos], pc[pos]
            prob_stats = {**qstats(-(pp_s*np.log(pp_s)).sum(1)/np.log(16),'probability_entropy_normalized'),
                          **qstats(pp_s.max(1),'maximum_prototype_probability'),
                          **qstats(np.abs(pp_s-1/16).sum(1),'probability_L1_from_uniform'),
                          'probability_sum_max_abs_error':float(np.abs(pp_s.sum(1)-1).max())}
            winners = np.bincount(pp_s.argmax(1), minlength=16)
            wf = winners[winners > 0]/len(pos)
            prob_stats['winning_prototypes_observed'] = int((winners > 0).sum())
            prob_stats['winning_prototype_frequency_entropy_normalized'] = float(-(wf*np.log(wf)).sum()/np.log(16))
            differences = {**pair_differences(pm_s,'mean'), **pair_differences(pc_s,'covariance')}
            common = {'role':role,'fold':fold,'scope':scope,'patients':len(pos),
                      'WSIs':int(np.isin(owned,ids).sum()),'source':source,'cache_sha256':receipt['sha256'],
                      'NaN_count':0,'Inf_count':0, **prob_stats, **differences, **scale}
            for component,k in grams.items():
                rows.append({**common,'component':component,'feature_dimensions':dimensions[component],
                             **spectrum(k,pos,dimensions[component])})
        audits.append({'role':role,'fold':fold,'source':source,'cache_sha256':receipt['sha256'],
                'preprocessing_cache_sha256':sha(prep_path),'fit_patients':len(fit),
                'held_out_patients':len(held),'active_patients':len(active),'active_WSIs':len(wsis),
                'tokens_shape':list(tokens.shape),'centroids_shape':list(centers.shape),
                'centroid_norm_min':float(np.linalg.norm(centers,axis=1).min()),
                'centroid_norm_max':float(np.linalg.norm(centers,axis=1).max()),
                **pair_differences(centers[None],'prior_centroid'), **scale})
        print('MMP_LABEL_BLIND_DIAGNOSTICS',role,fold,len(active),len(wsis),flush=True)
        del bank,prep,tokens,centers,prob,mean,cov,sample,pt,pp,pm,pc,grams
        gc.collect()
    assert rows and audits
    frame = pd.DataFrame(rows)
    csv_path = R/'MMP_TOKEN_DIAGNOSTICS.csv'
    tmp = csv_path.with_suffix('.tmp'); frame.to_csv(tmp,index=False,lineterminator='\n');tmp.replace(csv_path)
    active_rows = frame[(frame.scope=='active') & (frame.component=='combined_tokens')]
    active_means = frame[(frame.scope=='active') & (frame.component=='prototype_mean')]
    active_covariances = frame[(frame.scope=='active') & (frame.component=='prototype_covariance')]
    def ranges(values):
        return [float(np.min(values)),float(np.max(values))]
    observations = {
       'fit_sample_UNI2_mean_norm_range':ranges(active_rows.fit_sample_UNI2_L2norm_mean),
       'fit_sample_mean_per_dimension_variance_range':ranges(active_rows.fit_sample_per_dimension_variance_mean),
       'prior_covariance_to_mean_feature_variance_ratio_range':ranges(active_rows.prior_covariance_over_mean_feature_variance),
       'mean_normalized_probability_entropy_range':ranges(active_rows.probability_entropy_normalized_mean),
       'mean_maximum_allocation_probability_range':ranges(active_rows.maximum_prototype_probability_mean),
       'mean_prototype_pairwise_cosine_range':ranges(active_rows.mean_pairwise_cosine_mean),
       'covariance_prototype_pairwise_cosine_range':ranges(active_rows.covariance_pairwise_cosine_mean),
       'prototype_mean_patient_effective_rank_range':ranges(active_means.effective_rank_entropy_covariance),
       'prototype_covariance_patient_effective_rank_range':ranges(active_covariances.effective_rank_entropy_covariance),
       'covariance_fraction_of_combined_feature_variance_range':ranges(
            active_covariances.total_feature_variance.to_numpy()/active_rows.total_feature_variance.to_numpy()),
       'winning_prototypes_observed_range':ranges(active_rows.winning_prototypes_observed),
       'interpretation':'The observed mixture probabilities are far from uniform and prototype mean vectors differ; covariance vectors are nearly parallel, with covariance-scale variation dominating unstandardized concatenated token variance. A large prior/feature variance ratio alone does not establish collapse. This is a descriptive snapshot, not semantic or prognostic validation.'}
    result = {'status':'PASS','source':source,'diagnostic_script_sha256':sha(Path(__file__)),
      'current_source_successful_private_caches_audited':len(audits),
      'private_cache_receipts_in_snapshot':len(snapshots),'other_source_caches_excluded':len(ignored),
      'all_ten_role_fold_caches_present':len(audits)==10,
      'role_fold_pairs':[{'role':q['role'],'fold':q['fold']} for q in audits],
      'scope':'label-blind descriptive snapshot of already frozen private PANTHER distributions; no outcome or checkpoint-risk access',
      'partition_definition':{'inner':'fit=inner-train; held_out=inner-val; active excludes outer-test',
                              'outer':'fit=outer-train; held_out=outer-test'},
      'aggregation':'equal WSI arithmetic mean within patient; no concatenation across independently fit fold prototype coordinate systems',
      'effective_rank_definition':'covariance spectrum: exp(-sum q log q), participation=1/sum(q^2), q=eigenvalues/trace; variance population denominator n; centered Gram float64; no per-coordinate rescaling',
      'probability_reference':'16 components; uniform entropy/log16=1, maximum=0.0625, L1 distance=0',
      'prior_scale_comparison':'native initial diagonal covariance=eps*softplus(log(exp(1)-1))=.1; compare actual reconstructed fixed fit-only initialization sample per-dimension variance, not outcome-tuned scale',
      'read_files':['MMP successful cache/receipt','preprocessing cache partition indices/audit','frozen UNI2 u/mask/owner and READY receipt','PROTOCOL_LOCK'],
      'not_read':['endpoints.npz','OOF_PRIVATE.npz','neural checkpoints','survival time/event labels'],
      'outcomes_read':False,'parameters_or_training_source_changed':False,'models_retrained':0,
      'NaN_count':0,'Inf_count':0,'native_eps_unchanged':0.1,
      'active_combined_token_effective_rank_range':[float(active_rows.effective_rank_entropy_covariance.min()),float(active_rows.effective_rank_entropy_covariance.max())],
      'active_normalized_probability_entropy_range':[float(active_rows.probability_entropy_normalized_min.min()),float(active_rows.probability_entropy_normalized_max.max())],
      'CSV_sha256':sha(csv_path),'caches':audits,
      'observed_current_snapshot':observations,
      'interpretation_boundary':'A near-uniform or redundant distribution would limit this fixed-scale UNI2 adaptation; it does not establish failure of native MMP, justify post-outcome eps tuning, or prove no prognostic information. Numeric rank/variation alone is not clinical utility.'}
    history = list(previous.get('previous_snapshots', [])) if previous else []
    if previous:
        history.append({'snapshot_number':int(previous.get('snapshot_count',1)),
            'successful_caches':previous['current_source_successful_private_caches_audited'],
            'role_fold_pairs':previous['role_fold_pairs'], 'CSV_sha256':previous['CSV_sha256'],
            'diagnostic_JSON_sha256':sha(previous_path),
            'scope':'earlier label-blind cache-availability snapshot, not a new training result'})
    result['snapshot_count'] = len(history)+1
    result['previous_snapshots'] = history
    result['prepublication_auxiliary_diagnostic_repair'] = {
         'failed_analysis_attempts':1,'scope':'auxiliary label-blind in-memory analysis only; not a model/training failure',
         'cause':'centered Gram helper initially mutated a float64 patient-array view; JSON rejected NaN',
         'repair':'center a copy; successful outputs recomputed from SHA-verified immutable cache bytes',
         'private_cache_changed':False,'optimizer_steps':0,'training_source_changed':False}
    target=R/'MMP_TOKEN_DIAGNOSTICS.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(target)
    print(json.dumps({k:result[k] for k in ['status','current_source_successful_private_caches_audited',
       'active_combined_token_effective_rank_range','active_normalized_probability_entropy_range','NaN_count','Inf_count']}),flush=True)


if __name__=='__main__':main()

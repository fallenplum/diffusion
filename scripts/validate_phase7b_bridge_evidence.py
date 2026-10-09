"""Read-only independent checks of actual OOF, coefficients and alignment nulls."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from phase7_p0_biology_prompt_audit import digest,write_json,STATES
from phase7a_foundation_survival_headroom import features
from phase7b_bridge_evidence_audit import locked,nuisance,DEFAULT


def manual_c(t,e,r,w=None,fold=None):
    w=np.ones(len(t)) if w is None else w;num=0.;den=0.
    for i in np.flatnonzero((e==1)&(w>0)):
        js=(t>t[i])|((t==t[i])&(e==0))
        if fold is not None:js &= fold==fold[i]
        den+=w[i]*js.sum();num+=w[i]*((r[i]>r[js]).sum()+.5*(r[i]==r[js]).sum())
    return num/den


def manual_efron_gradient(x,time,event,w,lam):
    r=x@w;exp=np.exp(r-r.max());gr=-event.copy()
    for tt in np.unique(time[event==1]):
        die=(time==tt)&(event==1);risk=time>=tt;n=int(die.sum())
        for l in range(n):
            fraction=l/n;den=exp[risk].sum()-fraction*exp[die].sum()
            gr[risk]+=exp[risk]/den;gr[die]-=fraction*exp[die]/den
    return x.T@gr/event.sum()+lam*w


def manual_auc(y,r):
    pos=r[y==1];neg=r[y==0]
    return float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])))


def main(out):
    p,m,pm=locked(out);assert (out/'EXITCODE').read_text().strip()=='0'
    assert json.loads((out/'A_decision.json').read_text())['status']=='PASS'
    assert json.loads((out/'B_DONE.json').read_text())['status']=='PASS'
    parent=Path(p['parent_run']);d=dict(np.load(parent/'private_dataset.npz'));pp=json.loads((parent/'protocol.json').read_text())
    table=pd.read_csv(out/'A_seed_oof_metrics.csv');metricerr=0.;preerr=0.;pred_error=0.;gradient_error=0.;files=0
    for seed in pp['seeds']:
        for fold in range(1,6):
            tag=f'DSS_seed{seed}_fold{fold}';pre=np.load(out/'preprocessing'/f'{tag}_outer.npz');tr=pre['tr'];te=pre['te'];it=pre['it'];iv=pre['iv']
            assert not set(tr)&set(te) and not set(it)&set(iv) and set(it)|set(iv)==set(tr)
            for scope in ['inner','outer']:
                a=np.load(out/'preprocessing'/f'{tag}_{scope}.npz');b=np.load(parent/'preprocessing'/f'{tag}_{scope}.npz')
                for k in b.files:
                    assert k in a.files
                    if a[k].dtype.kind=='f':
                        error=float(np.max(np.abs(a[k]-b[k])));preerr=max(preerr,error);assert np.allclose(a[k],b[k],rtol=1e-12,atol=1e-12)
                    else:assert np.array_equal(a[k],b[k])
            f,_,_=features(d,tr,pm['groups']);f['CNA8']=f['MC'][:,8:]
            for method,blocks in p['A']['new_methods'].items():
                name=tag+'_'+method;c=json.loads((out/'cells'/f'{name}.json').read_text());op=out/'oof'/f'{name}.npz';oo=np.load(op);coef=np.load(out/'coefficients'/f'{name}.npz')
                assert digest(op)==c['oof_sha256'] and not c['outer_test_used_for_selection'];assert np.array_equal(oo['patient_ids'],d['patient_ids'][te])
                x=np.concatenate([f[k] for k in blocks],1);r=x@coef['coefficients'];actual=(r[te]-r[tr].mean())/r[tr].std()
                pred_error=max(pred_error,float(np.max(np.abs(actual-oo['risk']))));assert np.allclose(actual,oo['risk'],rtol=0,atol=1e-10)
                grad=manual_efron_gradient(x[tr],d['DSS_time'][tr],d['DSS_event'][tr],coef['coefficients'],c['selected_ridge'])
                gradient_error=max(gradient_error,float(np.max(np.abs(grad))));assert np.max(np.abs(grad))<=1e-5
                assert c['selected_ridge']==max(c['trials'],key=lambda t:(t['inner_harrell'],t['ridge']))['ridge']
                assert all(t['gradient_max']<=1e-5 and t['iterations']<5000 for t in c['trials']);files+=1
    for seed in pp['seeds']:
        for method in p['A']['reuse_methods']+list(p['A']['new_methods']):
            root=parent if method in p['A']['reuse_methods'] else out;pred=np.full(512,np.nan);raw=pred.copy();weight=pred.copy();count=np.zeros(512,int)
            for fold in range(1,6):
                tag=f'DSS_seed{seed}_fold{fold}_{method}';op=root/'oof'/f'{tag}.npz';a=np.load(op);c=json.loads((root/'cells'/f'{tag}.json').read_text());assert digest(op)==c['oof_sha256']
                te=a['test_indices'];assert np.array_equal(te,np.flatnonzero(d['folds']==fold));assert np.array_equal(a['event'],d['DSS_event'][te]) and np.array_equal(a['time'],d['DSS_time'][te])
                pred[te]=a['risk'];raw[te]=a['raw_risk'];weight[te]=a['uno_weight'];count[te]+=1
            assert (count==1).all();row=table[(table.seed==seed)&(table.method==method)].iloc[0]
            for col,r,w,folds in [('oof_harrell',pred,None,None),('oof_uno60',pred,weight,None),('raw_oof_harrell',raw,None,None),
                ('within_fold_pair_weighted_harrell',pred,None,d['folds']),('within_fold_pair_weighted_uno60',pred,weight,d['folds'])]:
                value=manual_c(d['DSS_time'],d['DSS_event'],r,w,folds);metricerr=max(metricerr,abs(value-row[col]));assert abs(value-row[col])<1e-12
    pairs=pd.read_csv(out/'A_paired_increments.csv')
    for row in pairs.itertuples(index=False):
        a,b=row.pair.split('-')
        for metric in ['oof_harrell','oof_uno60','raw_oof_harrell','within_fold_pair_weighted_harrell']:
            aa=table[table.method==a].sort_values('seed')[metric].to_numpy();bb=table[table.method==b].sort_values('seed')[metric].to_numpy()
            assert abs((aa-bb).mean()-getattr(row,'delta_'+metric))<1e-12

    # B computations can be reproduced without loading ANY survival array.
    archive=np.load(parent/'private_dataset.npz');blind={k:archive[k] for k in archive.files if not k.startswith(('DSS','PFS','y_'))}
    cache=np.load(out/'private_alignment_cache.npz');xr=np.empty((512,16));yr=np.empty((512,32));residualerr=0.
    for fold in range(1,6):
        a=np.load(out/'preprocessing'/f'alignment_fold{fold}.npz');tr=a['tr'];te=a['te'];assert not set(tr)&set(te)
        z,f=nuisance(blind,tr,cache['platform']);pinv=np.linalg.pinv(z[tr],rcond=max(z[tr].shape)*np.finfo(float).eps)
        rx=f['MC']-z@(pinv@f['MC'][tr]);ry=f['S']-z@(pinv@f['S'][tr]);xr[te]=rx[te]/rx[tr].std(0);yr[te]=ry[te]/ry[tr].std(0)
    residualerr=max(float(np.max(np.abs(xr-cache['molecular_residual']))),float(np.max(np.abs(yr-cache['concept_residual']))));assert residualerr<1e-9
    correlation=np.corrcoef(np.concatenate([xr,yr],axis=1),rowvar=False)[:16,16:];assert np.allclose(correlation,cache['correlation'],rtol=0,atol=1e-10)
    mask=np.asarray(m['RNA_prior_mask'],bool).T;maps=cache['random_mapping_masks'];assert maps.shape==(999,8,32)
    assert all(np.array_equal(a.sum(0),mask.sum(0)) and np.array_equal(a.sum(1),mask.sum(1)) for a in maps)
    assert len({a.tobytes() for a in maps})==999
    perm=cache['permutation_correlations'];rng=np.random.default_rng(88411);groups=[np.flatnonzero(cache['block_keys']==k) for k in np.unique(cache['block_keys'])]
    for b in range(1999):
        index=np.arange(512)
        for g in groups:index[g]=rng.permutation(g)
        if b in [0,1,997,1998]:
            alt=np.corrcoef(np.concatenate([xr[index],yr],1),rowvar=False)[:16,16:];assert np.allclose(alt,perm[b],rtol=0,atol=1e-10)
    statistic=np.square(correlation[:8][mask]).mean();null=np.square(perm[:,:8][:,mask]).mean(1);mapnull=np.asarray([np.square(correlation[:8][a]).mean() for a in maps])
    decision=json.loads((out/'B_decision.json').read_text());assert abs(statistic-decision['global_prior_mean_r_squared'])<1e-12
    assert (1+int((null>=statistic).sum()))/2000==decision['patient_pairing_benchmark_p']
    assert (1+int((mapnull>=statistic).sum()))/1000==decision['degree_matched_mapping_benchmark_p']
    ctab=pd.read_csv(out/'B_alignment_correlations.csv');primarymax=np.abs(perm[:,:8][:,mask]).max(1);checked=0
    for row in ctab.itertuples(index=False):
        j=STATES.index(row.state)+(0 if row.modality=='RNA' else 8);k=int(row.concept_id[1:])-1
        assert abs(correlation[j,k]-row.adjusted_oof_correlation)<1e-10
        if row.prior_edge:
            assert (1+int((primarymax>=abs(correlation[j,k])).sum()))/2000==row.primary_15_maxT_p;checked+=1
    assert checked==15
    morphology=json.loads((out/'B_morphology_label_audit.json').read_text());morph=np.load(out/'private_morphology_oof.npz');v=morph['valid'];y=morph['diagnosis'];r=morph['C10_score']
    assert abs(manual_auc(y[v],r[v])-morphology['zero_shot_C10_auc'])<1e-12
    for k,col in [(0,'nuisance_oof_auc'),(1,'nuisance_plus_C10_oof_auc')]:assert abs(manual_auc(y[v],morph['probabilities'][v,k])-morphology[col])<1e-12
    result={'status':'PASS','new_outer_OOF_and_coefficients_verified':files,'parent_OOF_files_reused':45,'complete_seed_OOF_models':21,
        'independent_metric_max_error':metricerr,'parent_float_normalizer_max_difference':preerr,'prediction_reconstruction_max_error':pred_error,
        'actual_coefficients_max_Efron_gradient':gradient_error,'independent_nuisance_residual_error':residualerr,
        'survival_array_free_B_reproduction':True,'all_999_maps_unique_and_degree_preserving':True,'blocked_patient_permutation_reproductions':4,
        'all_512_adjusted_correlations_verified':True,'all_15_primary_maxT_values_verified':True,'independent_morphology_AUC_recomputed':True}
    write_json(out/'independent_validation.json',result);write_json(out/'DONE.json',{'status':'PASS','protocol_sha256':m['protocol_sha256'],'independent_validation':'PASS'})
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,default=DEFAULT);main(a.parse_args().out)

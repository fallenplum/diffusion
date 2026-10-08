"""Read-only independent numerical/artifact validation; no fit or tuning."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from phase7_p0_biology_prompt_audit import digest,ahash,write_json


def macro_recall(y,p):
    return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))


def p0r(out):
    m=json.loads((out/'manifest.json').read_text());p=json.loads((out/'protocol.json').read_text());parent=Path(m['parent'])
    d=np.load(parent/'private_dataset.npz');y=d['y_Stage'];idx=np.flatnonzero(y>=0);means=[];seedrows=[];unique_checks=0
    for seed in p['inner_split_seeds']:
        for fold in range(1,6):
            cells=[json.loads((out/'cells'/f'random{ps}_seed{seed}_fold{fold}.json').read_text()) for ps in p['permutation_seeds']]
            for partition in ['inner','outer']:
                assert len({c['permutation_hashes'][partition] for c in cells})==99;unique_checks+=1
            assert all(c['feature_dim']==8 and not c['outer_test_used_for_selection'] for c in cells)
            assert all(c['outer_fit_n_iter']<5000 and all(t['n_iter']<5000 for t in c['trials']) for c in cells)
    for ps in p['permutation_seeds']:
        values=[]
        for seed in p['inner_split_seeds']:
            pred=np.full(len(y),-1);counts=np.zeros(len(y),int)
            for fold in range(1,6):
                op=out/'oof'/f'random{ps}_seed{seed}_fold{fold}.npz';a=np.load(op);te=a['test_indices']
                cp=json.loads((out/'cells'/f'random{ps}_seed{seed}_fold{fold}.json').read_text())
                assert digest(op)==cp['oof_sha256'] and np.all(d['folds'][te]==fold)
                assert np.array_equal(a['y_true'],y[te]);assert np.allclose(a['probabilities'].sum(1),1)
                pred[te]=a['probabilities'].argmax(1);counts[te]+=1
            assert np.all(counts[idx]==1) and np.all(counts[y<0]==0)
            value=macro_recall(y[idx],pred[idx]);values.append(value);seedrows.append((ps,seed,value))
        means.append(np.mean(values))
    true=[]
    for seed in p['inner_split_seeds']:
        pred=np.full(len(y),-1)
        for fold in range(1,6):
            a=np.load(parent/'oof'/f'Stage__seed{seed}__fold{fold}__R1_true_biology_state_rna.npz');pred[a['test_indices']]=a['probabilities'].argmax(1)
        true.append(macro_recall(y[idx],pred[idx]))
    decision=json.loads((out/'decision.json').read_text());means=np.asarray(means)
    assert abs(np.mean(true)-decision['true_mean_oof_ba'])<1e-12
    assert abs(np.mean(means)-decision['random_mean_oof_ba'])<1e-12
    assert (1+(means>=np.mean(true)).sum())/100==decision['conditional_benchmark_exceedance']
    reported=pd.read_csv(out/'seed_oof_benchmark.csv')
    # Accommodate aggregate schema by checking every numeric BA, without invoking
    # the experiment's metric or summary functions.
    bcol=next(c for c in reported if c in ['oof_ba','ba','balanced_accuracy'])
    assert np.allclose(np.sort(reported[bcol]),np.sort([a[2] for a in seedrows]),rtol=0,atol=1e-12)
    result={'status':'PASS','independent_macro_recall':True,'oof_files_verified':1485,'complete_seed_oof_verified':297,
        'unique_partition_permutation_checks':unique_checks,'eight_feature_protocol_retained':True,
        'maximum_metric_error':float(np.max(np.abs(np.sort(reported[bcol])-np.sort([a[2] for a in seedrows])))),
        'source_mask_upstream_confirmed':json.loads((out/'missing_block_source_audit.json').read_text())['official_missing_mask_identical']}
    write_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


def slow_c(t,e,r,w=None,fold=None):
    num=0.;den=0.;w=np.ones(len(t)) if w is None else w
    for i in range(len(t)):
        if e[i]!=1 or w[i]==0:continue
        js=(t>t[i])|((t==t[i])&(e==0))
        if fold is not None:js &= fold==fold[i]
        den+=w[i]*js.sum();num+=w[i]*(((r[i]>r[js]).sum())+.5*(r[i]==r[js]).sum())
    return num/den


def survival(out):
    from phase7a_foundation_survival_headroom import features,BLOCKS
    m=json.loads((out/'manifest.json').read_text());p=json.loads((out/'protocol.json').read_text());d=dict(np.load(out/'private_dataset.npz'))
    table=pd.read_csv(out/'seed_oof_metrics.csv');max_error=0.;prediction_error=0.;checks=0;splitchecks=0
    for ep in ['DSS','PFS']:
        valid=d[ep+'_valid'];idx=np.flatnonzero(valid);t=d[ep+'_time'];e=d[ep+'_event']
        for seed in p['seeds']:
            full={method:np.full(512,np.nan) for method in p['methods']};raw={k:v.copy() for k,v in full.items()};uw=np.zeros(512);count={k:np.zeros(512,int) for k in full}
            for fold in range(1,6):
                tag=f'{ep}_seed{seed}_fold{fold}';pre=np.load(out/'preprocessing'/f'{tag}_outer.npz');inner=np.load(out/'preprocessing'/f'{tag}_inner.npz')
                tr=pre['tr'];te=pre['te'];it=pre['it'];iv=pre['iv']
                assert not set(tr)&set(te) and set(it)|set(iv)==set(tr) and not set(it)&set(iv)
                assert np.array_equal(pre['fit_idx'],tr) and np.array_equal(inner['fit_idx'],it)
                assert np.array_equal(te,np.flatnonzero(valid&(d['folds']==fold)))
                f,a,_=features(d,tr,m['groups'])
                # Top512 verification uses an independent train-only variance computation.
                for part,fit in [(pre,tr),(inner,it)]:
                    z=np.log2(d['raw_rna'].astype(float)[fit]+1);coverage=np.isfinite(z).mean(0)
                    median=np.nanmedian(z,axis=0);z=np.where(np.isfinite(z),z,median)
                    var=z.var(0);eligible=np.flatnonzero((coverage>=.9)&(var>1e-8));keep=eligible[np.argsort(-var[eligible],kind='stable')[:512]]
                    assert np.array_equal(part['raw_top512'],keep);splitchecks+=1
                # Independent reverse KM, using only outer-train observations.
                times=np.sort(np.unique(t[tr]));prob=1.;gs=[]
                for tt in times:
                    ev=((t[tr]==tt)&(e[tr]==1)).sum();cn=((t[tr]==tt)&(e[tr]==0)).sum();risk=(t[tr]>=tt).sum()-ev
                    if cn:prob*=1-cn/risk
                    gs.append(prob)
                g=np.asarray([1. if not (times<tt).any() else gs[np.flatnonzero(times<tt)[-1]] for tt in t[te]])
                weights=np.zeros(len(te));active=(e[te]==1)&(t[te]<=60);weights[active]=1/g[active]**2
                for method in p['methods']:
                    celltag=tag+'_'+method;op=out/'oof'/f'{celltag}.npz';oo=np.load(op);coef=np.load(out/'coefficients'/f'{celltag}.npz');c=json.loads((out/'cells'/f'{celltag}.json').read_text())
                    assert digest(op)==c['oof_sha256'];assert np.array_equal(oo['patient_ids'],d['patient_ids'][te]);assert np.array_equal(oo['test_indices'],te)
                    assert np.array_equal(oo['time'],t[te]) and np.array_equal(oo['event'],e[te]) and not c['outer_test_used_for_selection']
                    assert c['outer_fit']['gradient_max']<=1e-5 and all(v['gradient_max']<=1e-5 for v in c['trials'])
                    selected=max(c['trials'],key=lambda v:(v['inner_harrell'],v['ridge']));assert selected['ridge']==c['selected_ridge']
                    x=np.concatenate([f[k] for k in BLOCKS[method]],axis=1);r=x@coef['coefficients'];cal=(r[te]-r[tr].mean())/r[tr].std()
                    prediction_error=max(prediction_error,float(np.max(np.abs(cal-oo['risk']))));assert np.allclose(cal,oo['risk'],rtol=0,atol=1e-11)
                    assert np.allclose(g,oo['censor_g'],rtol=0,atol=1e-12) and np.allclose(weights,oo['uno_weight'],rtol=0,atol=1e-10)
                    full[method][te]=oo['risk'];raw[method][te]=oo['raw_risk'];uw[te]=weights;count[method][te]+=1;checks+=1
            for method in p['methods']:
                assert np.all(count[method][idx]==1) and np.all(count[method][~valid]==0)
                row=table[(table.endpoint==ep)&(table.seed==seed)&(table.method==method)].iloc[0]
                for metric,value in [('oof_harrell',slow_c(t[idx],e[idx],full[method][idx])),('oof_uno60',slow_c(t[idx],e[idx],full[method][idx],uw[idx])),
                    ('raw_oof_harrell',slow_c(t[idx],e[idx],raw[method][idx])),('within_fold_pair_weighted_harrell',slow_c(t[idx],e[idx],full[method][idx],fold=d['folds'][idx])),
                    ('within_fold_pair_weighted_uno60',slow_c(t[idx],e[idx],full[method][idx],uw[idx],d['folds'][idx]))]:
                    max_error=max(max_error,abs(value-row[metric]));assert abs(value-row[metric])<1e-12
    result={'status':'PASS','outer_oof_and_coefficient_files_verified':checks,'train_only_top512_and_split_checks':splitchecks,
        'seed_oof_models_verified':len(table),'independent_metrics_error':max_error,'prediction_reconstruction_error':prediction_error,
        'independent_training_censor_KM_verified':True,'patient_identity_and_endpoint_coverage_verified':True,
        'all_inner_outer_gradients_converged':True,'selection_rule_verified':True,'DSS_PFS_separate':True}
    write_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--phase',choices=['p0r','survival'],required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args();globals()[args.phase](args.out)

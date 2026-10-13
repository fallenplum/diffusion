"""Supplementary ALL nonduplicate arms versus A1; no winner-only testing.

This separate exploratory family leaves the locked eight primary contrasts
and their 16-test correction untouched. Plan stored before outer inspection.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binom
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from validate_phase11_independent import pairs,ipcw
R=ROOT/'reports/phase11_semantic_anchored_trimodal';P=Path('/root/phase11_semantic_anchor_private/formal')


def main():
    plan=json.loads((R/'SYSTEM_ANCHOR_PLAN.json').read_text())
    cfg=json.loads((ROOT/'configs/phase11_semantic_anchored.json').read_text(encoding='utf-8-sig'))
    d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));t=d['DSS_time'];e=d['DSS_event'];folds=d['folds']
    pi,pj=pairs(t,e);keep=folds[pi]==folds[pj];pi,pj=pi[keep],pj[keep];weights=np.zeros(len(t))
    for fold in cfg['folds']:weights[folds==fold]=ipcw(t[folds!=fold],e[folds!=fold],t[folds==fold],e[folds==fold])
    risks={}
    for arm in plan['arms']+['A1']:
        for seed in cfg['seeds']:
            risk=np.empty(len(t))
            for fold in cfg['folds']:
                z=np.load(P/'oof'/f'{arm}_seed{seed}_fold{fold}.npz');risk[z['test_indices']]=z['risk']
            delta=risk[pi]-risk[pj];risks[arm,seed]=np.where(abs(delta)<=1e-8,.5,delta>0)
    credit=np.stack([np.mean([risks[arm,seed]-risks['A1',seed] for seed in cfg['seeds']],axis=0) for arm in plan['arms']],axis=1)
    rng=np.random.default_rng(cfg['bootstrap']['seed']);drawhash=hashlib.sha256();draws={m:[] for m in ['Harrell','Uno60']}
    for start in range(0,cfg['bootstrap']['draws'],250):
        count=rng.multinomial(len(t),np.ones(len(t))/len(t),size=min(250,cfg['bootstrap']['draws']-start));drawhash.update(count.tobytes())
        q=(count[:,pi]*count[:,pj]).astype(float);qu=q*weights[pi]
        draws['Harrell'].append((q@credit)/q.sum(1)[:,None]);draws['Uno60'].append((qu@credit)/qu.sum(1)[:,None])
    primary=json.loads((R/'BOOTSTRAP_AUDIT.json').read_text())
    if drawhash.hexdigest()!=primary['counts_sha256']:raise RuntimeError('bootstrap patients differ from primary family')
    metrics=pd.read_csv(R/'SEED_OOF_METRICS.csv');rows=[];family_tests=2*len(plan['arms']);alpha=.05/family_tests
    for metric,parts in draws.items():
        values=np.concatenate(parts)
        for col,arm in enumerate(plan['arms']):
            delta=[float(metrics[(metrics.arm==arm)&(metrics.seed==seed)][metric].iloc[0]-metrics[(metrics.arm=='A1')&(metrics.seed==seed)][metric].iloc[0]) for seed in cfg['seeds']]
            ordinary=np.quantile(values[:,col],[.025,.975]);sim=np.quantile(values[:,col],[alpha/2,1-alpha/2]);ordered=np.sort(values[:,col]);mc={}
            for label,prob in [('low',alpha/2),('high',1-alpha/2)]:
                ranks=np.clip(binom.ppf([.025,.975],len(ordered),prob).astype(int),0,len(ordered)-1)
                mc['sim_'+label+'_MC95_low']=ordered[ranks[0]];mc['sim_'+label+'_MC95_high']=ordered[ranks[1]]
            rows.append({'comparison':arm+'-A1','metric':metric,'mean_delta':np.mean(delta),'positive_seeds':sum(v>0 for v in delta),**{'seed'+str(seed)+'_delta':v for seed,v in zip(cfg['seeds'],delta)},'CI95_low':ordinary[0],'CI95_high':ordinary[1],'simultaneous_low':sim[0],'simultaneous_high':sim[1],'family_tests':family_tests,'draws':len(values),'scope':'SUPPLEMENTARY_EXPLORATORY_SYSTEM_ANCHOR',**mc})
    pd.DataFrame(rows).to_csv(R/'SYSTEM_ANCHOR_COMPARISONS.csv',index=False)
    (R/'SYSTEM_ANCHOR_AUDIT.json').write_text(json.dumps({'status':'PASS','counts_sha256':drawhash.hexdigest(),'primary_draws_identical':True,'tests':family_tests,'draws':cfg['bootstrap']['draws'],'expected_extreme_tail_draws':cfg['bootstrap']['draws']*alpha/2,'tail_probability_MC_SE':float(np.sqrt(alpha/2*(1-alpha/2)/cfg['bootstrap']['draws'])),'does_not_change_primary_family':True,'refit_variance_included':False},indent=2))
    print('SUPPLEMENTARY_SYSTEM_ANCHOR_ALL_ARMS_COMPLETE',family_tests)


if __name__=='__main__':main()

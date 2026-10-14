"""Absolute intervals for every model; descriptive, no primary upgrade."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd
from validate_phase13_independent import expected_models,raw_endpoints
from validate_phase11_independent import pairs,ipcw
P=Path('/root/phase13_rna_private/formal');R=ROOT/'reports/phase13_rna_information_audit'
cfg=json.loads((ROOT/'configs/phase13_rna_information_audit.json').read_text());d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));endpoints=raw_endpoints(d,cfg);models=expected_models(cfg);rows=[]
for ep in cfg['endpoints']:
    t,e,valid=[endpoints[ep][k] for k in ['time','event','valid']];indices=np.flatnonzero(valid);weight=np.zeros(512)
    for f in cfg['folds']:
        tr=valid&(d['folds']!=f);te=valid&(d['folds']==f);weight[te]=ipcw(t[tr],e[tr],t[te],e[te])
    pi,pj=pairs(t,e);keep=valid[pi]&valid[pj]&(d['folds'][pi]==d['folds'][pj]);pi,pj=pi[keep],pj[keep];matrix=[]
    for model in models:
        credits=[]
        for seed in cfg['seeds']:
            r=np.full(512,np.nan)
            for f in cfg['folds']:
                z=np.load(P/'oof'/f'{ep}_{model}_seed{seed}_fold{f}.npz');r[z['test_indices']]=z['risk']
            diff=r[pi]-r[pj];credits.append(np.where(abs(diff)<=1e-8,.5,diff>0))
        matrix.append(np.mean(credits,axis=0))
    matrix=np.stack(matrix,1);rng=np.random.default_rng(cfg['bootstrap']['seed']+(0 if ep=='DSS' else 1));samples={m:[] for m in ['Harrell','Uno60']}
    for start in range(0,cfg['bootstrap']['draws'],250):
        n=min(250,cfg['bootstrap']['draws']-start);counts=np.zeros((n,512),int);counts[:,indices]=rng.multinomial(len(indices),np.ones(len(indices))/len(indices),size=n);q=(counts[:,pi]*counts[:,pj]).astype(float);qu=q*weight[pi]
        samples['Harrell'].append(q@matrix/q.sum(1)[:,None]);samples['Uno60'].append(qu@matrix/qu.sum(1)[:,None])
    for metric,blocks in samples.items():
        boot=np.concatenate(blocks);weights=np.ones(len(pi)) if metric=='Harrell' else weight[pi];point=weights@matrix/weights.sum()
        for j,model in enumerate(models):
            lo,hi=np.quantile(boot[:,j],[.025,.975]);rows.append({'endpoint':ep,'model':model,'metric':metric,'point':point[j],'CI95_low':lo,'CI95_high':hi,'draws':len(boot),'scope':'descriptive absolute interval,no multiplicity/retraining variance'})
pd.DataFrame(rows).to_csv(R/'MODEL_ABSOLUTE_CI.csv',index=False);print('ALL_MODELS_ABSOLUTE_CI_COMPLETE',len(rows))

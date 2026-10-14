"""Check all12 primary intervals against literal repeated-patient metrics."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from validate_phase13_independent import contrasts,raw_endpoints
from validate_phase11_independent import pairs,components,ipcw
R=ROOT/'reports/phase13_rna_information_audit';P=Path('/root/phase13_rna_private/formal')
cfg=json.loads((ROOT/'configs/phase13_rna_information_audit.json').read_text());d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));ep=raw_endpoints(d,cfg);all_contrasts,primary=contrasts(cfg);contrasts={name:terms for name,terms in all_contrasts.items() if name in primary};error=0.;checks=0
for endpoint in cfg['endpoints']:
    t,e,valid=[ep[endpoint][k] for k in ['time','event','valid']];indices=np.flatnonzero(valid);weight=np.zeros(512)
    for fold in cfg['folds']:
        tr=valid&(d['folds']!=fold);te=valid&(d['folds']==fold);weight[te]=ipcw(t[tr],e[tr],t[te],e[te])
    risks={}
    for model in set(a for terms in contrasts.values() for a,c in terms):
        for seed in cfg['seeds']:
            r=np.full(512,np.nan)
            for fold in cfg['folds']:
                z=np.load(P/'oof'/f'{endpoint}_{model}_seed{seed}_fold{fold}.npz');r[z['test_indices']]=z['risk']
            risks[model,seed]=r
    pi,pj=pairs(t,e);keep=valid[pi]&valid[pj]&(d['folds'][pi]==d['folds'][pj]);pi,pj=pi[keep],pj[keep];rng=np.random.default_rng(cfg['bootstrap']['seed']+(0 if endpoint=='DSS' else 1))
    for draw in range(25):
        count=np.zeros(512,int);count[indices]=rng.multinomial(len(indices),np.ones(len(indices))/len(indices));q=count[pi]*count[pj];qu=q*weight[pi];literal={};credit={}
        for key,r in risks.items():
            sums=np.zeros(4)
            for fold in cfg['folds']:
                ix=np.flatnonzero(valid&(d['folds']==fold));expanded=np.repeat(ix,count[ix]);hn,hd=components(t[expanded],e[expanded],r[expanded]);un,ud=components(t[expanded],e[expanded],r[expanded],weight[expanded]);sums+=np.array([hn,hd,un,ud])
            literal[key]=np.array([sums[0]/sums[1],sums[2]/sums[3]]);diff=r[pi]-r[pj];credit[key]=np.where(abs(diff)<=1e-8,.5,diff>0)
        for name,terms in contrasts.items():
            for seed in cfg['seeds']:
                exact=sum(c*literal[a,seed] for a,c in terms);cr=sum(c*credit[a,seed] for a,c in terms);optimized=np.array([q@cr/q.sum(),qu@cr/qu.sum()]);error=max(error,float(abs(exact-optimized).max()));checks+=2
if error>1e-12:raise RuntimeError('literal patient bootstrap mismatch')
(R/'LITERAL_BOOTSTRAP_AUDIT.json').write_text(json.dumps({'status':'PASS','endpoints':['DSS','PFI'],'draws_per_endpoint':25,'scalar_checks':checks,'max_error':error,'same_patientcounts_all_seeds':True,'no_risk_averaging':True,'IPCW_fixed_training':True},indent=2));print('LITERAL_BOOTSTRAP_PASS',checks,error)

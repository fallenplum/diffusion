"""Literal duplicated-patient bootstrap, including DID and group averaging."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from validate_phase11_independent import pairs,ipcw,components
R=ROOT/'reports/phase12_c4_confirmation';P=Path('/root/phase12_c4_private/formal');cfg=json.loads((ROOT/'configs/phase12_c4_confirmation.json').read_text());d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True))
t=d['DSS_time'];e=d['DSS_event'];f=d['folds'];weights=np.zeros(512)
for fold in cfg['folds']:weights[f==fold]=ipcw(t[f!=fold],e[f!=fold],t[f==fold],e[f==fold])
contrasts={**cfg['primary'],'Mean8Groups-Null':cfg['secondary']['Mean8Groups-Null']};risks={}
for arm in set(a for terms in contrasts.values() for a,c in terms):
    for seed in cfg['seeds']:
        r=np.empty(512)
        for fold in cfg['folds']:
            z=np.load(P/'oof'/f'{arm}_seed{seed}_fold{fold}.npz');r[z['test_indices']]=z['risk']
        risks[arm,seed]=r
pi,pj=pairs(t,e);keep=f[pi]==f[pj];pi,pj=pi[keep],pj[keep];rng=np.random.default_rng(cfg['bootstrap']['seed']);err=0.;checks=0
for draw in range(25):
    count=rng.multinomial(512,np.ones(512)/512);q=count[pi]*count[pj];qu=q*weights[pi];literal={};credit={}
    for (arm,seed),r in risks.items():
        sums=np.zeros(4)
        for fold in cfg['folds']:
            ix=np.flatnonzero(f==fold);expanded=np.repeat(ix,count[ix]);hn,hd=components(t[expanded],e[expanded],r[expanded]);un,ud=components(t[expanded],e[expanded],r[expanded],weights[expanded]);sums+=np.array([hn,hd,un,ud])
        literal[arm,seed]=np.array([sums[0]/sums[1],sums[2]/sums[3]]);delta=r[pi]-r[pj];credit[arm,seed]=np.where(abs(delta)<=1e-8,.5,delta>0)
    for name,terms in contrasts.items():
        for seed in cfg['seeds']:
            exact=sum(c*literal[a,seed] for a,c in terms);cr=sum(c*credit[a,seed] for a,c in terms);optimized=np.array([q@cr/q.sum(),qu@cr/qu.sum()]);err=max(err,float(abs(exact-optimized).max()));checks+=2
if err>1e-12:raise RuntimeError('DID/group literal patient bootstrap mismatch')
(R/'LITERAL_BOOTSTRAP_AUDIT.json').write_text(json.dumps({'status':'PASS','draws_rechecked':25,'contrasts':list(contrasts),'scalar_metric_checks':checks,'max_error':err,'patients_explicitly_duplicated':True,'DID_verified':True,'groups_not_independent_patients':True},indent=2));print('LITERAL_BOOTSTRAP_PASS',err,checks)

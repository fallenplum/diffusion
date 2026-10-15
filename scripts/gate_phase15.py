"""Fail-closed label/teacher/pair partition gates; no smoke predictions persisted."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.phase15_complement.core import *


def main():
    torch.set_num_threads(2);d=parent.load();ev=parent.endpoints(d);part=parent.splits(d,'DSS',1);fit=part['inner_train'];held=part['inner_val'];t,e=ev['DSS']['time'],ev['DSS']['event'];checks={}
    x,st=cs_features(d,fit);lam,scores=select_cs(d,t,e,fit,31,2);w,_=fit_ridge(x[fit],t[fit],e[fit],lam)
    tt=t.copy();ee=e.copy();ee[held]=1-ee[held];tt[held]+=123;ee[part['outer_test']]=1-ee[part['outer_test']];tt[part['outer_test']]+=321
    lam2,scores2=select_cs(d,tt,ee,fit,31,2);w2,_=fit_ridge(x[fit],tt[fit],ee[fit],lam2)
    if scores!=scores2 or not np.array_equal(w,w2):raise RuntimeError('validation/test labels reached anchor')
    checks['inner_validation_and_outer_test_label_perturbation']='same anchor lambda,candidate scores andcoefficients'
    dd={k:v.copy() for k,v in d.items()};dd['static'][held]+=100;dd['static'][part['outer_test']]-=100
    xx,_=cs_features(dd,fit)
    if not np.array_equal(xx[fit],x[fit]):raise RuntimeError('held-out features reached anchor fit')
    checks['heldout_S32_feature_perturbation']='train coding/scaling bitwiseunchanged'
    fake={'fit':np.arange(6),'group':np.array([0,0,1,1,2,2]),'oof':np.array([0.,1,0,1,0,1]),'teachers':[{'train':np.setdiff1d(np.arange(6),[j,j+1]),'held':np.array([j,j+1])} for j in [0,2,4]]}
    ii,jj,margin=honest_pairs(fake,np.arange(1,7),np.ones(6,int))
    if len(ii)!=3:raise RuntimeError('cross-teacher pairs included')
    bad={**fake,'teachers':[dict(v) for v in fake['teachers']]};bad['teachers'][0]['train']=np.arange(6)
    try:honest_pairs(bad,np.arange(1,7),np.ones(6,int))
    except RuntimeError:checks['injected_pair_teacher_leakage_rejected']=True
    else:raise RuntimeError('teacher leakage not rejected')
    rng=np.random.default_rng(31);r=rng.normal(size=(20,4));ii,jj=patient_pairs(np.arange(20),np.ones(20));margin=rng.normal(size=len(ii));a=weights('SOFT',margin,ii,1);b=weights('RANDOM',margin,ii,1)
    for i in np.unique(ii):
        if abs(a[ii==i].sum()-b[ii==i].sum())>1e-12 or not np.array_equal(np.sort(a[ii==i]),np.sort(b[ii==i])):raise RuntimeError('randomweight eventmass mismatch')
    w,info=pair_fit(r,ii,jj,margin,a,1.);D=r[ii]-r[jj];manual=D.T@(-expit(-(margin+D@w))*a/a.sum())+w
    if abs(manual).max()>2e-5:raise RuntimeError('independent pair gradient mismatch')
    checks['pair_stationarity_and_event_mass']='PASS'
    for ep in cfg()['endpoints']:
        for fold in cfg()['folds']:
            p=parent.splits(d,ep,fold)
            if set(p['inner_train'])&set(p['inner_val']) or set(p['outer_train'])&set(p['outer_test']):raise RuntimeError('patient split leak')
        checks[ep]={'patients':int(ev[ep]['valid'].sum()),'events':int(ev[ep]['event'][ev[ep]['valid']].sum())}
    result={'status':'PASS','source_hash':source(),'protocol_sha256':sha(CONFIG),'checks':checks,'smoke_predictions_saved':False};write(REPORT/'HARD_GATES.json',result);write(PRIVATE/'PROTOCOL_LOCK.json',result);print(json.dumps(result))


if __name__=='__main__':main()

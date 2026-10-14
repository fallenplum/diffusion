"""Fixed-lambda probe; each fit uses one producer's latent coordinate system."""
import json
import numpy as np
from .math import block_fit,linear_map
from .data import standardize,ahash,PRIVATE,write_json


def fixed_probe(train,test,t,e,lam=1,base_train=None,base_test=None,base_lambda=None):
    x,xt,mean,sd=standardize(train.astype(float),test.astype(float))
    if base_train is None:penalty=np.full(x.shape[1],lam);a=x;b=xt
    else:a=np.c_[base_train,x];b=np.c_[base_test,xt];penalty=np.r_[np.full(base_train.shape[1],base_lambda),np.full(x.shape[1],lam)]
    w,diag=block_fit(a,t,e,penalty)
    return b@w,w,{'mean':mean.tolist(),'sd':sd.tolist(),'penalty':penalty.tolist(),'fit_features_hash':ahash(a),'test_features_hash':ahash(b),'optimizer':diag}


def reference_risk_map(train,test,rtrain,rtest,lam=1):
    a,b,mean,sd=standardize(train.astype(float),test.astype(float));ym=rtrain.mean();ys=rtrain.std()
    w=linear_map(a,(rtrain-ym)/max(ys,1e-8),lam);prediction=(b@w)*ys+ym
    denom=np.sum((rtest-rtest.mean())**2);r2=1-np.sum((rtest-prediction)**2)/denom if denom>0 else np.nan
    if np.std(prediction)>1e-8 and np.std(rtest)>1e-8:correlation=float(np.corrcoef(prediction,rtest)[0,1])
    else:correlation=None
    return prediction,{'heldout_reference_risk_R2':float(r2),'heldout_reference_risk_correlation':correlation,'notice':'target isTRAIN-fitted RNA Cox score,not patient survival truth;fixedlambda1'},w,mean,sd,ym,ys

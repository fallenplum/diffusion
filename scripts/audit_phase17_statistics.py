import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT))
import numpy as np
from sksurv.metrics import concordance_index_censored,concordance_index_ipcw
from sksurv.util import Surv
from validate_phase17_independent import pairs,weights,components
from src.phase17_ssd.data import R,write

def main():
    tt=np.array([.5,1.,2.,2.,3.,4.,6.,9.,12.]);ee=np.array([1,0,1,0,1,0,1,0,1]);t=np.array([1.,2.,2.,3.,5.,7.]);e=np.array([1,1,0,1,0,1]);r=np.array([2.,1.,1.,.5,.4,-.1]);w=weights(tt,ee,t,e,60)
    hn,hd,*_=components(t,e,r,np.ones(len(t)));un,ud,*_=components(t,e,r,w)
    h=concordance_index_censored(e.astype(bool),t,r)[0];u=concordance_index_ipcw(Surv.from_arrays(ee.astype(bool),tt),Surv.from_arrays(e.astype(bool),t),r,tau=60)[0]
    assert abs(h-hn/hd)<1e-12 and abs(u-un/ud)<1e-12
    i,j=pairs(t,e);delta=r[i]-r[j];credit=np.where(abs(delta)<=1e-8,.5,delta>0);rng=np.random.default_rng(1707);maximum=0.
    for _ in range(100):
        counts=rng.multinomial(len(t),np.ones(len(t))/len(t));repeat=np.repeat(np.arange(len(t)),counts)
        for ww in [np.ones(len(t)),w]:
            products=counts[i]*counts[j]*ww[i];den=products.sum()
            if den==0:continue
            expected=float((products*credit).sum()/den);a,b,*_=components(t[repeat],e[repeat],r[repeat],ww[repeat]);maximum=max(maximum,abs(expected-a/b));assert maximum<1e-12
    write(R/'STATISTICAL_OPERATOR_AUDIT.json',{'status':'PASS','toy_Harrell_error':abs(h-hn/hd),'toy_Uno_error':abs(u-un/ud),'bootstrap_duplicate_expansion_checks':200,'maximum_count_weight_vs_expanded_sample_error':maximum,'risk_direction':'higher risk means earlier event','KM':'reverse censoring Kaplan-Meier with training event tie adjustment'})
    print('STATISTICAL_OPERATOR_PASS',flush=True)

if __name__=='__main__':main()

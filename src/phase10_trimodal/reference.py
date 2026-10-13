"""Independent evaluation: no imports from training, Phase8, or model code."""
import numpy as np


def censor_probability(train_time,train_event,query):
    """Reverse Kaplan-Meier at the observed time, matching sksurv IPCW.

    Events are removed before censor jumps on tied times. Event times use the
    post-time value; tied censor/event support is tested against sksurv.
    """
    t=np.asarray(train_time,float);e=np.asarray(train_event,int)
    times=np.unique(t);g=[];surv=1.
    for u in times:
        cens=int(np.count_nonzero((t==u)&(e==0)))
        at_risk=int(np.count_nonzero(t>=u))-int(np.count_nonzero((t==u)&(e==1)))
        if cens: surv*=1.-cens/at_risk
        g.append(surv)
    pos=np.searchsorted(times,np.asarray(query),side='right')-1
    return np.where(pos<0,1.,np.asarray(g)[np.maximum(pos,0)])


def pairs(t,e):
    """Enumerate once per pair independently of the training loss risk sets."""
    t=np.asarray(t);e=np.asarray(e);left=[];right=[]
    for i in range(len(t)):
        if not e[i]:continue
        js=np.flatnonzero((t>t[i])|((t==t[i])&(e==0)))
        left.extend([i]*len(js));right.extend(js.tolist())
    return np.array(left,int),np.array(right,int)


def components(t,e,risk,weights=None,tied_tol=1e-8):
    r=np.asarray(risk,float)
    if not np.isfinite(r).all():raise ValueError('nonfinite risk')
    i,j=pairs(t,e); w=np.ones(len(r)) if weights is None else np.asarray(weights,float)
    delta=r[i]-r[j]; credit=np.where(np.abs(delta)<=tied_tol,.5,delta>0)
    return float((w[i]*credit).sum()),float(w[i].sum())


def uno_weights(tt,te,t,e,tau=60.):
    eligible=(np.asarray(e)==1)&(np.asarray(t)<tau)
    g=censor_probability(tt,te,np.asarray(t)[eligible])
    if np.any(g<=0):raise ValueError('IPCW censor support failure')
    w=np.zeros(len(t));w[eligible]=1./g**2
    return w


def metrics(t,e,r,folds,weights,counts=None):
    """Fold-local micro, plus scale-sensitive pooled diagnostic."""
    t,e,r,f,w=map(np.asarray,(t,e,r,folds,weights));hn=hd=un=ud=0.
    count=np.ones(len(t)) if counts is None else np.asarray(counts)
    for fold in np.unique(f):
        ix=np.flatnonzero((f==fold)&(count>0)); ii,jj=pairs(t[ix],e[ix]); q=count[ix[ii]]*count[ix[jj]]
        diff=r[ix[ii]]-r[ix[jj]]; credit=np.where(np.abs(diff)<=1e-8,.5,diff>0)
        hn+=float((q*credit).sum());hd+=float(q.sum());un+=float((q*w[ix[ii]]*credit).sum());ud+=float((q*w[ix[ii]]).sum())
    return {'Harrell':hn/hd if hd else np.nan,'Uno60':un/ud if ud else np.nan,'Harrell_num':hn,'Harrell_den':hd,'Uno_num':un,'Uno_den':ud}


def reference_efron(t,e,r):
    """Independent scalar/event-loop oracle, stable even for extreme logits."""
    t=np.asarray(t);e=np.asarray(e);r=np.asarray(r,float)
    if not np.isin(e,[0,1]).all() or e.sum()==0:raise ValueError('invalid event vector')
    total=0.
    for u in sorted(set(t[e==1])):
        death=np.flatnonzero((t==u)&(e==1));d=len(death)
        for j in range(d):
            entries=[]
            for i in np.flatnonzero(t>=u):
                coefficient=1.-j/d if i in death else 1.
                if coefficient>0:entries.append(r[i]+np.log(coefficient))
            high=max(entries);total+=high+np.log(sum(np.exp(v-high) for v in entries))
        total-=sum(r[death])
    return float(total/e.sum())

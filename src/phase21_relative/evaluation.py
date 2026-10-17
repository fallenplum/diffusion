"""Anonymous, within-fold pair-credit accounting and shared patient bootstrap.

Higher risk predicts an earlier event. Risks retain their original dtype when
pair differences are formed, matching Phase20's 1e-8 risk-tie convention.
No patient indices, risks, pair tables or resampled patient counts are returned.
"""
from __future__ import annotations
import hashlib
import numpy as np

TIE_TOLERANCE=1e-8
PAIR_STATES=('wrong','tied','correct')
TRANSITIONS=tuple(a+'_to_'+b for a in PAIR_STATES for b in PAIR_STATES)
STATISTICS=('C0','C1','repair','damage','net',*TRANSITIONS)


def _inputs(time,event,folds,valid,ipcw):
    time=np.asarray(time);event=np.asarray(event);folds=np.asarray(folds)
    if time.ndim!=1 or event.shape!=time.shape or folds.shape!=time.shape:
        raise ValueError('time/event/folds must be matching one-dimensional arrays')
    valid=np.ones(len(time),bool) if valid is None else np.asarray(valid)
    if valid.dtype!=np.dtype(bool) or valid.shape!=time.shape or not valid.any():
        raise ValueError('valid must be a nonempty Boolean patient mask')
    if np.iscomplexobj(time) or np.iscomplexobj(event) or any(q is None or q!=q for q in folds[valid]):
        raise ValueError('endpoint and fold labels must be real and present')
    if not np.isfinite(time[valid]).all() or (time[valid]<0).any() or not np.isin(event[valid],[0,1]).all():
        raise ValueError('valid endpoint time/event values are invalid')
    metrics=['Harrell'];weights=np.ones((1,len(time)),float)
    if ipcw is not None:
        ipcw=np.asarray(ipcw)
        if ipcw.shape!=time.shape or not np.isfinite(ipcw[valid]).all() or (ipcw[valid]<0).any():
            raise ValueError('frozen IPCW must be finite, nonnegative and patient-aligned')
        metrics.append('Uno60');weights=np.vstack([weights,ipcw])
    return time,event.astype(bool),folds,valid,metrics,weights


def _risk(risk,n,valid):
    risk=np.asarray(risk)
    if risk.shape!=(n,) or not np.issubdtype(risk.dtype,np.number) or np.issubdtype(risk.dtype,np.unsignedinteger) or np.iscomplexobj(risk):
        raise ValueError('risk must be a matching signed numeric patient vector')
    if not np.isfinite(risk[valid]).all():raise ValueError('nonfinite risk among valid patients')
    return risk


def _folds(time,event,folds,valid,weights):
    for label in np.unique(folds[valid]):
        patients=np.flatnonzero(valid&(folds==label));t=time[patients];e=event[patients]
        i,j=np.nonzero(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
        yield label,patients,i,j,weights[:,patients[i]]


def _features(r0,r1,patients,i,j):
    a=r0[patients];b=r1[patients]
    d0=a[i]-a[j];d1=b[i]-b[j]
    q0=np.where(abs(d0)<=TIE_TOLERANCE,.5,d0>0)
    q1=np.where(abs(d1)<=TIE_TOLERANCE,.5,d1>0)
    delta=q1-q0
    codes=(2*q0).astype(int)*3+(2*q1).astype(int)
    transitions=np.eye(9,dtype=float)[codes]
    return np.column_stack([q0,q1,np.maximum(delta,0),np.maximum(-delta,0),delta,transitions])


def _checks(values):
    if not np.allclose(values[...,2]-values[...,3],values[...,4],rtol=0,atol=1e-12):
        raise ValueError('repair minus damage does not equal net credit')
    if not np.allclose(values[...,1]-values[...,0],values[...,4],rtol=0,atol=1e-12):
        raise ValueError('net credit does not equal C1 minus C0')
    if not np.allclose(values[...,5:].sum(-1),1.,rtol=0,atol=1e-12):
        raise ValueError('transition fractions do not sum to one')


def _row(values,denominator):
    result={name:float(value) for name,value in zip(STATISTICS,values)}
    result.update(denominator=float(denominator),C0_numerator=float(values[0]*denominator),
        C1_numerator=float(values[1]*denominator),repair_credit_mass=float(values[2]*denominator),
        damage_credit_mass=float(values[3]*denominator),
        baseline_strict_wrong_fraction=float(values[5:8].sum()),
        baseline_tied_fraction=float(values[8:11].sum()),
        baseline_strict_correct_fraction=float(values[11:14].sum()))
    for name,value in zip(TRANSITIONS,values[5:]):result[name+'_mass']=float(value*denominator)
    return result


def pair_transition_summary(time,event,r0,r1,*,folds,valid=None,ipcw=None):
    """Compare one baseline/new risk vector, pooling within-fold num/den.

    Pass the actual outer-fold vector for complete OOF inputs, or one constant
    fold label for an already isolated held-out fold. IPCW must come from the
    corresponding frozen outer-train reverse KM; it is never fitted here.
    repair/damage use the all-comparable-pair denominator, including half-credit
    transitions involving risk ties. They are not error-group conditional rates.
    """
    t,e,f,v,metrics,w=_inputs(time,event,folds,valid,ipcw)
    r0=_risk(r0,len(t),v);r1=_risk(r1,len(t),v)
    num=np.zeros((len(metrics),len(STATISTICS)));den=np.zeros(len(metrics));pairs=0
    for _,patients,i,j,weight in _folds(t,e,f,v,w):
        features=_features(r0,r1,patients,i,j);num+=weight@features;den+=weight.sum(1);pairs+=len(i)
    if (den<=0).any():raise ValueError('zero comparable-pair denominator')
    values=num/den[:,None];_checks(values)
    return {'status':'PASS','rows':[{'metric':metric,**_row(value,d)} for metric,value,d in zip(metrics,values,den)],
        'comparable_pairs':pairs,'valid_patients':int(v.sum()),'fold_count':int(len(np.unique(f[v]))),
        'risk_tie_tolerance':TIE_TOLERANCE,'within_fold_only':True,'patient_or_pair_details_returned':False}


def patient_shared_transition_bootstrap(time,event,comparisons,*,folds,valid=None,ipcw=None,
        draws=20000,seed=2107911,batch_size=256,forced_counts=None,family_tests=None,
        minimum_valid_fraction=.99):
    """Bootstrap {comparison: {algorithm_seed: (baseline_risk,new_risk)}}.

    One patient multinomial count matrix per fixed fold is shared by every
    comparison and algorithm seed. The pair weight is count_i*count_j, with no
    invented self-pairs. Each complete seed statistic is formed after summing
    all fold num/den; only then are seeds averaged. Returned intervals cover
    fixed fitted models, not refitting, model selection or researcher choices.
    forced_counts is a mathematical-oracle hook, not a formal resampling plan.
    """
    t,e,f,v,metrics,w=_inputs(time,event,folds,valid,ipcw)
    if not isinstance(draws,int) or draws<1 or batch_size<1 or not 0<minimum_valid_fraction<=1:
        raise ValueError('invalid bootstrap budget')
    if family_tests is not None and (not isinstance(family_tests,int) or family_tests<1):
        raise ValueError('family_tests must be a predeclared positive integer')
    if not comparisons:raise ValueError('no paired comparisons')
    names=sorted(comparisons)
    seeds=sorted(comparisons[names[0]],key=str)
    if not seeds or any(set(comparisons[name])!=set(seeds) for name in names):
        raise ValueError('complete, identical algorithm-seed coverage is required')
    keys=[(name,s) for name in names for s in seeds];risks=[]
    for name,s in keys:
        pair=comparisons[name][s]
        if len(pair)!=2:raise ValueError('each comparison seed needs baseline and new risk')
        risks.append(tuple(_risk(q,len(t),v) for q in pair))
    k,m,l=len(keys),len(metrics),len(STATISTICS)
    point_num=np.zeros((k,m,l));den=np.zeros(m)
    sampled_num=np.zeros((draws,k,m,l));sampled_den=np.zeros((draws,m))
    rng=np.random.default_rng(seed);audit=[];pair_count=0
    for label,patients,i,j,weight in _folds(t,e,f,v,w):
        n=len(patients);pair_count+=len(i)
        features=np.stack([_features(a,b,patients,i,j) for a,b in risks],axis=1)
        contributions=(features[:,:,None,:]*weight.T[:,None,:,None]).reshape(len(i),k*m*l)
        point_num+=contributions.sum(0).reshape(k,m,l);den+=weight.sum(1)
        counts=(rng.multinomial(n,np.full(n,1/n),size=draws) if forced_counts is None else np.asarray(forced_counts[label]))
        if counts.shape!=(draws,n) or not np.issubdtype(counts.dtype,np.integer) or (counts<0).any() or not np.all(counts.sum(1)==n):
            raise ValueError('bootstrap counts violate fixed fold patient sizes')
        audit.append({'fold':int(label) if isinstance(label,(int,np.integer)) else str(label),
            'patients':n,'draws':draws,'counts_sha256':hashlib.sha256(np.ascontiguousarray(counts,dtype=np.int64).tobytes()).hexdigest(),
            'shared_comparison_seed_vectors':k})
        for start in range(0,draws,batch_size):
            cc=counts[start:start+batch_size];products=(cc[:,i]*cc[:,j]).astype(float)
            sampled_num[start:start+len(cc)]+=(products@contributions).reshape(len(cc),k,m,l)
            sampled_den[start:start+len(cc)]+=products@weight.T
    if (den<=0).any():raise ValueError('zero point comparable-pair denominator')
    good=(sampled_den>0).all(1)
    if good.sum()<minimum_valid_fraction*draws:raise ValueError('too many zero-denominator patient draws')
    points=point_num/den[None,:,None]
    samples=sampled_num[good]/sampled_den[good,None,:,None]
    _checks(points);_checks(samples)
    point_rows=[];seed_rows=[];interval_rows=[]
    for name in names:
        selected=[keys.index((name,s)) for s in seeds]
        means=points[selected].mean(0);boot=samples[:,selected].mean(1);_checks(means);_checks(boot)
        for mi,metric in enumerate(metrics):
            point_rows.append({'comparison':name,'metric':metric,**_row(means[mi],den[mi])})
            seed_rows.extend({'comparison':name,'seed':s,'metric':metric,**_row(points[index,mi],den[mi])} for s,index in zip(seeds,selected))
            for si,statistic in enumerate(STATISTICS):
                low,high=np.quantile(boot[:,mi,si],[.025,.975])
                row={'comparison':name,'metric':metric,'statistic':statistic,'point':float(means[mi,si]),
                    'CI_low':float(low),'CI_high':float(high)}
                if family_tests is not None:
                    bounds=np.quantile(boot[:,mi,si],[.025/family_tests,1-.025/family_tests])
                    row.update(family_tests=family_tests,corrected_low=float(bounds[0]),corrected_high=float(bounds[1]))
                interval_rows.append(row)
    return {'status':'PASS','point_rows':point_rows,'seed_rows':seed_rows,'interval_rows':interval_rows,
        'audit':{'draws_requested':draws,'draws_valid':int(good.sum()),'zero_denominator_draws':int((~good).sum()),
            'comparable_pairs':pair_count,'risk_tie_tolerance':TIE_TOLERANCE,'shared_patient_count_matrices':audit,
            'within_fold_only':True,'risk_ensembling':False,'patient_or_pair_details_returned':False,
            'IPCW':'provided frozen outer-train weights, event-owner i only; never refitted',
            'uncertainty_scope':'fixed fitted models; no refitting or model-selection uncertainty',
            'identity':'repair-damage = C1-C0 in every complete seed and patient draw'}}

"""Independent literal patient-expansion oracles for pair repair accounting."""
import numpy as np
import pytest
from src.phase21_relative.evaluation import pair_transition_summary,patient_shared_transition_bootstrap,STATISTICS


def literal(t,e,a,b,folds,weights):
    num=np.zeros(len(STATISTICS));den=0.
    for i in range(len(t)):
        for j in range(len(t)):
            if folds[i]!=folds[j] or not e[i] or not (t[i]<t[j] or (t[i]==t[j] and not e[j])):continue
            credits=[]
            for risk in [a,b]:
                delta=risk[i]-risk[j]
                credits.append(.5 if abs(delta)<=1e-8 else float(delta>0))
            q0,q1=credits;delta=q1-q0;row=np.zeros(len(STATISTICS))
            row[:5]=q0,q1,max(delta,0),max(-delta,0),delta
            row[5+int(2*q0)*3+int(2*q1)]=1
            num+=weights[i]*row;den+=weights[i]
    return num/den,den


def test_half_credit_transitions_close_repair_minus_damage():
    t=np.arange(1.,5.);e=np.array([1,1,1,0]);a=np.array([0.,0.,2.,1.]);b=np.array([3.,1.,1.,0.]);f=np.ones(4,int)
    result=pair_transition_summary(t,e,a,b,folds=f)['rows'][0]
    expected,den=literal(t,e,a,b,f,np.ones(4))
    assert result['denominator']==den
    for key,value in zip(STATISTICS,expected):assert result[key]==pytest.approx(value)
    assert result['repair']-result['damage']==pytest.approx(result['C1']-result['C0'])
    assert result['tied_to_correct']>0 and result['wrong_to_tied']>0


def test_same_time_event_censor_is_comparable_and_event_event_is_not():
    t=np.array([1.,1.,1.,2.]);e=np.array([1,1,0,0]);a=np.array([3.,2.,1.,0.]);f=np.ones(4,int)
    result=pair_transition_summary(t,e,a,a,folds=f)['rows'][0]
    assert result['denominator']==4 and result['C0']==1 and result['net']==0


def test_cross_fold_risk_pairs_are_never_included():
    t=np.array([1.,2.,1.,2.]);e=np.array([1,0,1,0]);a=np.array([100.,90.,1.,0.]);b=a.copy();f=np.array([1,1,2,2])
    result=pair_transition_summary(t,e,a,b,folds=f)
    assert result['comparable_pairs']==2 and result['rows'][0]['C0']==1
    with pytest.raises(TypeError):pair_transition_summary(t,e,a,b)


def test_uno_uses_only_frozen_event_owner_weights_and_preserves_valid_mask():
    t=np.array([1.,2.,3.,np.nan]);e=np.array([1,1,0,0]);a=np.array([3.,1.,2.,np.nan]);b=np.array([1.,3.,2.,np.nan]);valid=np.array([1,1,1,0],bool)
    weights=np.array([4.,9.,999.,np.nan]);f=np.ones(4,int)
    result=pair_transition_summary(t,e,a,b,folds=f,valid=valid,ipcw=weights)
    uno=result['rows'][1];expected,den=literal(t[:3],e[:3],a[:3],b[:3],f[:3],weights[:3])
    assert uno['denominator']==17
    for key,value in zip(STATISTICS,expected):assert uno[key]==pytest.approx(value)


def test_float32_margin_arithmetic_and_tie_boundary_match_existing_protocol():
    t=np.array([1.,2.]);e=np.array([1,0]);a=np.array([0.,0.],np.float32)
    b=np.array([1e-8,0.],np.float32)
    result=pair_transition_summary(t,e,a,b,folds=np.ones(2,int))['rows'][0]
    delta=b[0]-b[1];expected=.5 if abs(delta)<=1e-8 else float(delta>0)
    # NumPy array comparison casts the scalar threshold to the margin dtype.
    array_expected=np.where(abs(b[:1]-b[1:])<=1e-8,.5,(b[:1]-b[1:])>0)[0]
    assert result['C1']==array_expected


def test_shared_patient_bootstrap_matches_literal_expansion_for_all_statistics():
    t=np.tile([1.,2.,3.,4.],2);e=np.tile([1,1,0,0],2);f=np.repeat([1,2],4);w=np.tile([2.,3.,0.,0.],2)
    base=np.tile([0.,0.,2.,1.],2);new=np.tile([3.,1.,1.,0.],2)
    comparisons={'main':{31:(base,new),47:(base,-new)},'reverse':{31:(new,base),47:(-new,base)}}
    forced={1:np.array([[1,1,1,1],[2,1,0,1],[0,2,1,1]]),2:np.array([[1,1,1,1],[1,2,1,0],[2,0,1,1]])}
    result=patient_shared_transition_bootstrap(t,e,comparisons,folds=f,ipcw=w,draws=3,forced_counts=forced,family_tests=4)
    assert result['audit']['draws_valid']==3 and all(q['shared_comparison_seed_vectors']==4 for q in result['audit']['shared_patient_count_matrices'])
    for name,risks in comparisons.items():
        for metric,weights in [('Harrell',np.ones(8)),('Uno60',w)]:
            values=[]
            for draw in range(3):
                expanded=np.concatenate([np.repeat(np.flatnonzero(f==fold),forced[fold][draw]) for fold in [1,2]])
                per_seed=[literal(t[expanded],e[expanded],a[expanded],b[expanded],f[expanded],weights[expanded])[0] for a,b in risks.values()]
                values.append(np.mean(per_seed,axis=0))
            values=np.asarray(values)
            for si,statistic in enumerate(STATISTICS):
                row=next(q for q in result['interval_rows'] if (q['comparison'],q['metric'],q['statistic'])==(name,metric,statistic))
                assert [row['CI_low'],row['CI_high']]==pytest.approx(np.quantile(values[:,si],[.025,.975]))
            for row in result['point_rows']:
                if (row['comparison'],row['metric'])==(name,metric):assert row['repair']-row['damage']==pytest.approx(row['net'])


def test_complete_seed_credit_mean_is_not_concordance_of_mean_risk():
    t=np.arange(1.,5.);e=np.array([1,1,1,0]);f=np.ones(4,int);base=np.zeros(4);a=np.array([3.,2.,1.,0.]);b=-2*a
    result=patient_shared_transition_bootstrap(t,e,{'x':{31:(base,a),47:(base,b)}},folds=f,draws=2,forced_counts={1:np.ones((2,4),int)})
    assert result['point_rows'][0]['C1']==.5
    wrong=pair_transition_summary(t,e,base,(a+b)/2,folds=f)['rows'][0]['C1']
    assert wrong==0


def test_invalid_inputs_missing_seeds_and_zero_denominators_fail_closed():
    t=np.array([1.,2.]);e=np.array([1,0]);f=np.ones(2,int);a=np.array([1.,0.])
    with pytest.raises(ValueError,match='nonfinite'):pair_transition_summary(t,e,a,np.array([np.nan,0.]),folds=f)
    with pytest.raises(ValueError,match='denominator'):pair_transition_summary(t,np.zeros(2),a,a,folds=f)
    with pytest.raises(ValueError,match='IPCW'):pair_transition_summary(t,e,a,a,folds=f,ipcw=np.array([-1.,0.]))
    with pytest.raises(ValueError,match='fold'):pair_transition_summary(t,e,a,a,folds=np.array([1.,np.nan]))
    with pytest.raises(ValueError,match='signed numeric'):pair_transition_summary(t,e,a.astype(complex),a,folds=f)
    with pytest.raises(ValueError,match='seed'):patient_shared_transition_bootstrap(t,e,{'x':{31:(a,a)},'y':{47:(a,a)}},folds=f,draws=2)
    with pytest.raises(ValueError,match='zero-denominator'):patient_shared_transition_bootstrap(t,e,{'x':{31:(a,a)}},folds=f,draws=2,forced_counts={1:np.array([[2,0],[2,0]])})

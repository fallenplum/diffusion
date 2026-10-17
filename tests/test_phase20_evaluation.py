"""Independent factorial statistic, patient-resampling and decision-axis checks."""
import numpy as np
import pandas as pd
import pytest
from scripts.validate_phase20 import (PRIMARY,PRIMARY_FAMILY_TESTS,ARMS,DID_NAME,definitions,
    point_tensor,linear_c_contrast,resampled_c_indices,components,flatten_diagnostics,selectivity_behavior_audit)
from scripts.report_phase20 import scientific_decision

SEEDS=[31,47,83]

def test_seven_factorial_contrasts_reserve_fourteen_tests_and_require_coverage():
    risks={(arm,seed):np.arange(6.) for arm in ARMS for seed in SEEDS}
    assert len(PRIMARY)*2==PRIMARY_FAMILY_TESTS==14
    assert [name for name,_,family in definitions(risks,SEEDS) if family=='primary']==list(PRIMARY)
    augmented={**risks,**{(arm+'__'+suffix,seed):np.arange(6.) for arm in ['P1','P16']
                        for suffix in ['FULL_OMICS_WRONG','UNIFORM_PAIR'] for seed in SEEDS}}
    extra=[q for q in definitions(augmented,SEEDS) if q[2]=='exploratory_representation']
    assert len(extra)==5 and extra[-1][0]=='DID_uniform'
    assert extra[-1][1]=={'P16':1.,'P1':-1.,'P16__UNIFORM_PAIR':-1.,'P1__UNIFORM_PAIR':1.}
    risks.pop(('S1',83))
    with pytest.raises(ValueError,match='seeds'): definitions(risks,SEEDS)

def test_patient_bootstrap_four_complete_c_did_matches_explicit_replication():
    t=np.array([1.,2.,4.,1.,2.,4.]);e=np.array([1,1,0,1,0,0],bool)
    fold=np.repeat([1,2],3);valid=np.ones(6,bool);weights=np.array([1.,2.,0.,1.5,0.,0.])
    base={'S16':np.array([3,2,1,1,3,2.]),'P16':np.array([1,2,3,3,1,2.]),
          'S1':np.array([2,1,3,2,3,1.]),'P1':np.array([2,3,1,3,2,1.])}
    risks={(arm,seed):risk.copy() for arm,risk in base.items() for seed in SEEDS}
    risks['S1',47]=-risks['S1',47];risks['P1',83]=-risks['P1',83]
    counts={1:np.array([[1,1,1],[2,0,1]]),2:np.array([[1,1,1],[2,0,1]])};audit=[]
    keys,values,den,good=resampled_c_indices(t,e,valid,fold,risks,weights,draws=2,seed=9,forced_counts=counts,count_audit=audit)
    assert good.all() and len(audit)==2
    assert all(q['one_shared_count_matrix_for_model_seed_vectors']==12 for q in audit)
    oracle=np.zeros_like(values)
    for draw in range(2):
        for j,key in enumerate(keys):
            numerator=np.zeros(2);denominator=np.zeros(2)
            for f in [1,2]:
                ix=np.flatnonzero(fold==f);rep=np.repeat(ix,counts[f][draw])
                nn,dd=components(t[rep],e[rep],risks[key][rep],weights[rep]);numerator+=nn;denominator+=dd
            oracle[draw,j]=numerator/denominator
    np.testing.assert_allclose(values,oracle,atol=1e-14)
    actual=linear_c_contrast(keys,values,PRIMARY[DID_NAME],SEEDS)
    direct=np.array([[sum(coefficient*oracle[d,[keys.index((arm,s)) for s in SEEDS],m].mean()
        for arm,coefficient in PRIMARY[DID_NAME].items()) for m in [0,1]] for d in range(2)])
    np.testing.assert_allclose(actual,direct,atol=1e-14)
    assert np.ptp(actual[:,0])>0

def test_mean_seed_c_is_not_c_of_mean_patient_risk():
    t=np.array([1.,2.,3.]);e=np.array([1,1,0],bool);w=np.ones(3)
    a=np.array([100.,2.,1.]);b=np.array([-1.,0.,100.])
    first=[components(t,e,r,w) for r in [a,b]]
    expected=np.mean([n/d for n,d in first],axis=0)
    n,d=components(t,e,(a+b)/2,w)
    np.testing.assert_allclose(expected,[.5,.5]);np.testing.assert_allclose(n/d,[1/3,1/3])

def test_point_tensor_rejects_missing_or_repeated_models():
    row={'arm':'S16','seed':31,'Harrell':.8,'Uno60':.75}
    assert point_tensor([row],[('S16',31)]).shape==(1,1,2)
    with pytest.raises(ValueError,match='same models'): point_tensor([row],[('S16',31),('P16',31)])
    with pytest.raises(ValueError,match='repeated'): point_tensor([row,row],[('S16',31)])

def fixture_results():
    values={'L0':.61,'P1':.60,'P16':.65,'S1':.62,'S16':.72,'D16':.68,'F16':.66,'WC16':.64,'WO16':.63,'J0':.80,'J_CS':.81}
    seed=pd.DataFrame([{'arm':arm,'seed':seed,'Harrell':value,'Uno60':value-.03} for arm,value in values.items() for seed in SEEDS])
    pairs=[]
    for name,coefficient in PRIMARY.items():
        delta=sum(weight*values[arm] for arm,weight in coefficient.items())
        for metric in ['Harrell','Uno60']:
            pairs.append({'comparison':name,'metric':metric,'family':'primary','family_tests':14,
                'delta':delta,'positive_seeds':3,'Bonferroni_low':delta-.0001})
    budget={'rows':[{'arm':arm,'budget_gate_pass':True} for arm in ARMS]}
    return seed,pd.DataFrame(pairs),budget

def test_mechanism_support_is_separate_from_absolute_predictive_rank():
    seed,paired,budget=fixture_results();behavior={'status':'PASS','behavior_gate_pass':True}
    decision=scientific_decision(seed,paired,budget,behavior)
    assert decision['axes']['selective_clinical_correction']['status']=='WITHIN_COHORT_CORRECTED_SUPPORT'
    assert decision['axes']['representation_by_correction_interaction']['status']=='WITHIN_COHORT_CORRECTED_SUPPORT'
    assert decision['practical_prediction_value']['all_J0_J_CS_mean_no_harm'] is False
    assert decision['status']=='WITHIN_COHORT_CORRECTED_SUPPORT'
    behavior['behavior_gate_pass']=False
    assert scientific_decision(seed,paired,budget,behavior)['status']=='CONDITIONED_MODULATION_CANDIDATE_WITHOUT_SPARSE_SELECTION'
    records=[]
    for seed in SEEDS:
        for fold in range(1,6):
            row={'arm':'S16','role':'inner','seed':seed,'fold':fold,'selected_epoch':8}
            for direction in ['po','op']:
                row.update({direction+'_selectivity_'+key:value for key,value in {
                    'support_patient_mean':.7,'all_rows_support_patient_mean':.72,'eligible_lt2_patient_mean':.02,
                    'raw_score_degenerate_patient_mean':.03,'eligible_ge2_query_count':5000,
                    'support_query_count':5000,'support_WSI_count':200,'support_patient_count':190}.items()})
            records.append(row)
    records[0]['op_selectivity_support_patient_mean']=.95
    assert selectivity_behavior_audit(records,'synthetic')['behavior_gate_pass']
    for row in records: row['op_selectivity_support_patient_mean']=.75
    assert not selectivity_behavior_audit(records,'synthetic')['behavior_gate_pass']

def test_budget_censoring_and_reported_point_mismatch_fail_closed():
    seed,paired,budget=fixture_results();budget['rows'][0]['budget_gate_pass']=False
    behavior={'status':'PASS','behavior_gate_pass':True}
    # L0 is not in the selective-primary optimization contrast: no invented censoring.
    assert scientific_decision(seed,paired,budget,behavior)['axes']['selective_clinical_correction']['budget_sufficient']
    budget['rows'][ARMS.index('S16')]['budget_gate_pass']=False
    assert scientific_decision(seed,paired,budget,behavior)['status']=='BUDGET_UNRESOLVED'
    paired.loc[0,'delta']+=.01
    with pytest.raises(ValueError,match='point differs'): scientific_decision(seed,paired,budget,behavior)

def test_single_patient_targets_and_private_probe_indices_are_not_public():
    row={'arm':'S16','anchor_risk':8.,'target_value':9.,'risk_raw':7.,'attention_entropy_mean':.6,
         'probe_indices':{'isolated_gradient':np.arange(8)},
         'position_gradient_probes':[{'target':'final_risk','target_value':3.,'entropy':.5}]}
    result=flatten_diagnostics([row])[0]
    assert not any(key in result for key in ['anchor_risk','target_value','risk_raw','probe_indices','position_probe_target_value'])
    assert result['attention_entropy_mean']==.6
    assert row['target_value']==9.

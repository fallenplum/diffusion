"""Scheduler arithmetic/qualification counterexamples; no real fitting."""
import pytest
from scripts import semantic_b0b7_p1r_matrix as m


def metric(huber,gap):
    return {'program':{'patient_equal_masked_huber':huber},'retrieval_hard':{
        d:{'paired_gap_mean':gap} for d in ('image_to_gene','gene_to_image')}}


def rows(a,huber,gap,tol=.001):
    tag='A050' if a==.5 else 'A075'
    return [{'variant':'T3_'+arm+'_DUAL_'+tag,'seed':seed,
             'selected_validation':metric(huber,gap),'program_huber_tolerance':tol}
            for arm in ('B3','B4') for seed in m.SEEDS]


def test_uniform_checkpoint_selection_uses_training_tolerance_then_gap():
    history=[{'epoch':50,'validation':metric(.04,.1)},
             {'epoch':100,'validation':metric(.0405,.3)},
             {'epoch':200,'validation':metric(.05,.9)}]
    assert m.select_epoch(history,.001)['epoch']==100
    assert m.select_epoch(history,.0001)['epoch']==50


def test_uniform_checkpoint_gap_tie_prefers_earlier_epoch():
    history=[{'epoch':100,'validation':metric(.04,.1)},
             {'epoch':50,'validation':metric(.0401,.1)}]
    assert m.select_epoch(history,.001)['epoch']==50


def test_dual_selection_uses_both_query_families_all_seeds():
    result=m.select_fraction(rows(.5,.05,.2)+rows(.75,.0495,.1))
    assert result['selected_fraction']==.5
    assert result['replication_scores_used'] is False


def test_dual_selection_outside_tolerance_uses_lower_huber():
    result=m.select_fraction(rows(.5,.05,.9)+rows(.75,.045,.1))
    assert result['selected_fraction']==.75


def test_dual_exact_tie_fixed_half_not_later_results():
    assert m.select_fraction(rows(.5,.05,.2)+rows(.75,.05,.2))['selected_fraction']==.5


def test_missing_paired_seed_cannot_select_fraction():
    with pytest.raises(ValueError,match='INCOMPLETE_PAIRED_DUAL_SELECTION_GROUP'):
        m.select_fraction(rows(.5,.05,.2)+rows(.75,.05,.2)[:-1])


def test_replication_needs_exact_prelocked_wrong_aux_definition():
    with pytest.raises(ValueError,match='EXACT_ONE_PRELOCKED_DUAL_WRONG_AUX_REQUIRED'):
        m.replication_names(.5,[])
    names=m.replication_names(.5,[{'name':'WRONG050','wrong_aux':True,'relation_mode':'dual','known_fraction':.5}])
    assert len(names)==8 and 'WRONG050' in names and 'T4_B3_LAMBDA003' in names


def test_parallel_admission_uses_container_not_host_free_RAM():
    resource={'filesystem_free_bytes':3*1024**3,'cgroup_memory':{'usage_bytes':85*1024**3,'limit_bytes':90*1024**3}}
    assert m.admission(resource) is False
    resource['cgroup_memory']['usage_bytes']=70*1024**3
    assert m.admission(resource) is True
    resource['filesystem_free_bytes']=1024**3
    assert m.admission(resource) is False


def test_job_identity_cartesian_product_does_not_collapse_partitions():
    parts={1:{'file':'one.pt','sha256':'a'*64},2:{'file':'two.pt','sha256':'b'*64}}
    jobs=m.make_jobs('replication',['A','B'],parts,(1,2))
    assert len(jobs)==12 and len({x['key'] for x in jobs})==12

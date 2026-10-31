import numpy as np
from src.semantic_m123.diagnostics import effective_rank,summarize_attention


def test_uniform_identical_slots_have_rank_one_and_full_effective_coverage():
    a=np.ones((1,2,8,16))/16;z=np.ones((1,8,64))
    d=summarize_attention(a,z,np.ones((1,16),bool))
    assert abs(d['effective_patch_fraction_mean']-1)<1e-12
    assert abs(d['attention_pair_cosine_mean']-1)<1e-12
    assert abs(d['slot_effective_rank_mean']-1)<1e-12
    assert d['centered_slot_effective_rank_mean']==0


def test_orthogonal_distinct_slots_have_expected_rank_and_disjoint_attention():
    a=np.stack([np.eye(4),np.eye(4)])[None]
    z=np.eye(4,64)[None]
    d=summarize_attention(a,z,np.ones((1,4),bool))
    assert abs(d['effective_patch_fraction_mean']-.25)<1e-12
    assert abs(d['attention_pair_cosine_mean'])<1e-12
    assert abs(d['slot_effective_rank_mean']-4)<1e-12


def test_unobserved_queries_do_not_fabricate_coverage_or_rank():
    d=summarize_attention(np.ones((1,2,3,4))/4,np.ones((1,3,64)),np.ones((1,4),bool),np.zeros((1,3),bool))
    assert d['patients_with_observed_queries']==0
    assert d['slot_effective_rank_mean'] is None


def test_padding_and_single_query_pair_statistic():
    a=np.array([[[[.5,.5,0.]]]])
    d=summarize_attention(a,np.ones((1,1,64)),np.array([[True,True,False]]))
    assert d['effective_patch_fraction_mean']==1
    assert d['attention_pair_cosine_mean'] is None


def test_zero_slots_rank_zero_not_one():assert effective_rank(np.zeros((8,64)))==0


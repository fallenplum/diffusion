"""Finite exact-chance/mask counterexamples, no checkpoint or data reads."""
import numpy as np
import pytest
from scripts import semantic_b0b7_p1r_chance as c


def test_exact_anchor_average_not_reciprocal_mean_count():
    valid=np.ones((4,4),bool)
    negative=np.array([[0,1,0,0],[1,0,1,1],[1,0,0,0],[1,1,1,0]],bool)
    row=c.exact_direction(valid,negative)
    assert row['exact_chance_mean']==pytest.approx((.5+.25+.5+.25)/4)
    assert row['exact_chance_mean']!=pytest.approx(1/(row['negative_count_mean']+1))
    assert sum(v['anchors'] for v in row['negative_count_histogram'])==4


def test_no_eligible_support_is_null_not_zero():
    row=c.exact_direction(np.ones((3,3),bool),np.zeros((3,3),bool))
    assert row['anchors_covered']==0 and row['exact_chance_mean'] is None


def test_diagonal_invalid_positive_anchor_is_excluded():
    valid=np.ones((3,3),bool);valid[0,0]=False
    row=c.exact_direction(valid,~np.eye(3,dtype=bool))
    assert row['anchors_covered']==2


def test_non_boolean_or_non_square_support_rejected():
    with pytest.raises(ValueError,match='PAIR_SUPPORT_MUST_BE_SQUARE_BOOLEAN'):
        c.exact_direction(np.ones((2,3),bool),np.ones((2,3),bool))


def test_singleton_stratum_has_explicit_unsupported_chance():
    meta={'patient':np.arange(3),'stage':np.array(['I','II','II']),
          'cohort':np.array(['A','A','A']),'missing_pattern':np.zeros(3,int)}
    rows=c.compute_scopes(meta,np.ones((3,3),bool))
    singleton=[r for r in rows if r['scope']=='stage' and r['group']=='I']
    assert len(singleton)==4 and all(r['exact_chance_mean'] is None for r in singleton)


def test_appended_chance_does_not_modify_source_metrics():
    row={'scope':'global','group':None,'mode':'hard','direction':'image_to_gene',
         'anchors_covered':2,'negative_count_mean':1.,'exact_chance_mean':.5}
    metric={'retrieval_hard':{'image_to_gene':{'anchors_covered':2,'mean_negative_count':1.,'top1_tie_averaged':.75}}}
    out=c.append_reference(metric,[row])
    assert out['retrieval_hard']['image_to_gene']['top1_minus_exact_chance']==.25
    assert 'exact_chance_mean' not in metric['retrieval_hard']['image_to_gene']

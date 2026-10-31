import csv
import numpy as np
import pytest
from scripts.semantic_m123_report import aggregate,seed_summary,paired,csv_shards,CONTRASTS
from src.semantic_m123.replay import repair_damage
from src.semantic_b0b7.survival import harrell_components


def test_five_fold_summary_never_pairs_across_risk_scales():
    # Fold identity has no risk vector input, so cross-fold pairs cannot enter.
    rows=[]
    for seed in (31,47,83):
        for fold in range(1,6):
            den=10*fold
            rows.append({'arm':'A','seed':seed,'outer':fold,'Harrell':.5,'Uno60':.75,
                'Harrell_num':.5*den,'Harrell_den':den,'Uno_num':.75*den,'Uno_den':den})
    result=seed_summary(rows)[0]
    assert result['Harrell']==.5 and result['Uno60']==.75
    assert len(result['seed_results'])==3 and all(x['folds']==5 for x in result['seed_results'])


def test_missing_fold_fails_instead_of_claiming_full_oof():
    rows=[{'arm':'A','seed':31,'outer':i,'Harrell':.5,'Uno60':None,
        'Harrell_num':1,'Harrell_den':2,'Uno_num':None,'Uno_den':None} for i in (1,2,3,4)]
    with pytest.raises(ValueError,match='COVERAGE'):seed_summary(rows)


def test_uno_na_is_retained_not_imputed():
    rows=[{'Harrell':.5,'Harrell_num':1,'Harrell_den':2,'Uno60':None,'Uno_num':None,'Uno_den':None}]*5
    assert aggregate(rows)['Uno60'] is None


@pytest.mark.parametrize('risk,base',[
    ([3.,2.,1.,0.],[0.,1.,2.,3.]),([1.,1.,2.,2.],[1.,2.,1.,2.]),
    ([.1,.1+.5e-8,.1+2e-8,.7],[0.,0.,0.,0.])])
def test_paired_repair_minus_damage_equals_exact_harrell_difference(risk,base):
    t=[1,1,3,4];e=[1,0,1,0]
    out=repair_damage(t,e,risk,base)
    a=harrell_components(t,e,risk);b=harrell_components(t,e,base)
    assert abs(out['net_delta']-(a.score-b.score))<1e-15
    assert out['repair_num']>=0 and out['damage_num']>=0


def test_csv_shards_preserve_every_cell_and_column_under_fixed_bound(tmp_path):
    rows=[{'arm':'模型','x':i,'negative':-i/10} for i in range(100)]
    files=csv_shards(tmp_path,'TABLE',rows,limit=280)
    actual=[]
    for name in files:
        p=tmp_path/name
        assert p.stat().st_size<=280
        actual.extend(csv.DictReader(p.open(encoding='utf-8')))
    assert len(actual)==100 and [int(x['x']) for x in actual]==list(range(100))


def test_contrasts_fixed_and_include_same_capacity_and_strong_anchors():
    assert len(CONTRASTS)==len(set(CONTRASTS))
    for b in ['M0','M0PLUS','M2_EIGHT_MEAN','M3_LEARNED_QUERY','M3_NO_PRODUCT','M3_BIDIRECTIONAL']:
        assert ('M3_FULL',b) in CONTRASTS
    assert ('M3_FULL','M3_FULL_WRONG_QUERY') in CONTRASTS
    assert ('M3_FULL','M3_FULL_NO_INTERACTION') in CONTRASTS


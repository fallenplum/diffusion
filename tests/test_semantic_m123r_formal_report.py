import json
from pathlib import Path
import numpy as np
import pytest
from scripts.semantic_m123r_report_formal import complete_coverage,final_barrier
from src.semantic_m123.replay import repair_damage
from src.semantic_b0b7.survival import uno_weights,pair_components


def test_oof_coverage_rejects_duplicates_and_missing_rows():
    complete_coverage(np.arange(512)[::-1])
    for x in (np.arange(511),np.r_[np.arange(511),510]):
        with pytest.raises(ValueError,match='COVERAGE'):complete_coverage(x)


def test_active_training_cannot_access_final_label_phase(tmp_path):
    (tmp_path/'OWNER.private.json').write_text('{}')
    with pytest.raises(ValueError,match='TRAINING_STILL_ACTIVE'):
        final_barrier(tmp_path,tmp_path/'unavailable.json','frozen')


def test_uno_pair_repairs_equal_exact_metric_delta_with_tied_times():
    tt=np.array([1.,2.,3.,4.,6.,8.,12.,70.]);ee=np.array([1,0,1,0,1,1,0,1])
    t=np.array([1.,3.,3.,5.,8.,65.]);e=np.array([1,1,0,1,0,1])
    r=np.array([.1,.4,.4,.3,.2,0]);r0=np.array([.5,.3,.2,.4,.6,0])
    w=uno_weights(tt,ee,t,e,tau=60.)
    a=pair_components(t,e,r,w);b=pair_components(t,e,r0,w)
    d=repair_damage(t,e,r,r0,w)
    assert d['net_delta']==pytest.approx(a.score-b.score,abs=1e-12)

"""Synthetic arithmetic checks only; no endpoint file, scoring or estimator fit."""
import copy

import numpy as np
import pytest

from scripts import semantic_b0b7_p2_r4_precision_audit as audit


def test_shape_dtype_value_and_signed_zero_are_distinguished_without_raw_export():
    old=np.asarray([[0.,1.],[2.,3.]],np.float64);new=old.copy()
    assert audit.difference(old,new)["exact_normalized_value_bytes"]
    new[0,0]=-0.
    result=audit.difference(old,new)
    assert not result["exact_normalized_value_bytes"] and result["max_abs_difference"]==0.
    result=audit.difference(old,old.astype(np.float32))
    assert not result["dtype_equal"] and result["max_abs_difference"]==0.
    result=audit.difference(old,old[:1])
    assert not result["shape_equal"] and "max_abs_difference" not in result


def test_recursive_first_difference_is_saved_order_not_a_causal_inference():
    old={"clinical":{"levels":np.asarray(["A","B"])},
         "visual":{"before":np.asarray([.1]),"components":np.asarray([[.2,.3]])}}
    new=copy.deepcopy(old);new["visual"]["components"][0,0]+=1e-14
    result=audit.leaf_comparisons(old,new,"preprocessing")
    assert result["leaf_count"]==3 and result["different_leaf_count"]==1
    assert result["first_different_saved_order_leaf"]=="preprocessing.visual.components"
    assert result["different_leaves"][0]["max_abs_difference"]==pytest.approx(1e-14,rel=.01,abs=1e-16)
    assert "values" not in result["different_leaves"][0]


def test_F_to_C_copy_hypothesis_is_actually_measured_after_row_selection():
    values=np.asfortranarray(np.random.default_rng(7).normal(size=(18,5)))
    rows=np.asarray([0,1,3,4,7,9,11,14]);selected=np.arange(len(values))
    copied=values[selected].copy()
    assert values.flags.f_contiguous and copied.flags.c_contiguous
    before=audit.direct_scale(values,rows);after=audit.direct_scale(copied,rows)
    result=audit.compare_reductions(values,copied,rows,rows,before,after)
    assert result["original_whole"]["F_contiguous"] and result["copy_whole"]["C_contiguous"]
    # Advanced row indexing here yields C inputs even from an F whole array.
    assert result["original_training_view"]["C_contiguous"]
    assert all(v["original_vs_copy_reduction"]["exact_normalized_value_bytes"] for v in result["statistics"].values())
    assert all(v["original_vs_stored_old"]["exact_normalized_value_bytes"] for v in result["statistics"].values())


def test_stored_reduction_mismatch_is_not_hidden_by_identical_recomputed_arrays():
    x=np.arange(30.,dtype=float).reshape(10,3);rows=np.arange(6)
    before=audit.direct_scale(x,rows);after=copy.deepcopy(before);after["mean"][0]+=1e-12
    result=audit.compare_reductions(x,x.copy(),rows,rows,before,after)
    assert result["statistics"]["mean"]["original_vs_copy_reduction"]["exact_normalized_value_bytes"]
    assert not result["statistics"]["mean"]["copy_vs_stored_current"]["exact_normalized_value_bytes"]


def test_no_authority_prevents_all_input_reads(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError("Input accessed")
    monkeypatch.setattr(audit,"sha",forbidden)
    with pytest.raises(ValueError,match="BEFORE_INPUT_READ"):
        audit.run_audit("unread","unread",assets_path="unread",legacy_root="unread",historical_root="unread",
                        formal_private="unread",original_audit_path="unread",output="unread")

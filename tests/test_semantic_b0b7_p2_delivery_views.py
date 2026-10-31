"""Synthetic aggregation/copy boundaries; no new models or patient data."""
import copy
import csv
import io

import pytest

from scripts import semantic_b0b7_p2_delivery_views as view


def aggregate(arm,values):
    return {"arm":arm,"required_seeds":[31] if arm=="B0" else list(view.SEEDS),
        "Harrell":sum(values)/len(values),"Uno60":sum(values)/len(values),
        "seed_results":[{"seed":s,"Harrell":v,"Uno60":v,"Harrell_num":10*v,"Harrell_den":10,
            "Uno_num":20*v,"Uno_den":20} for s,v in zip((31,) if arm=="B0" else view.SEEDS,values)]}


def comparison():
    return {"comparison":"B5-B0","left":"B5","reference":"B0","Harrell_delta":.1,"Uno60_delta":.1,
        **{m+suffix:value for m in ("Harrell","Uno60") for suffix,value in
           (("_positive_fold_seed_pairs",9),("_negative_fold_seed_pairs",6),("_zero_fold_seed_pairs",0))}}


def test_seed_micro_three_rows_share_one_b0_without_fake_reference_fits():
    rows=view.seed_contrasts([comparison()],[aggregate("B0",[.5]),aggregate("B5",[.5,.6,.7])])
    assert len(rows)==3 and {r["reference_seed"] for r in rows}=={31}
    assert all(r["reference_B0_is_one_shared_deterministic_fit"] and r["new_fits"]==0 for r in rows)
    assert [r["Harrell_seed_micro_delta"] for r in rows]==pytest.approx([0.,.1,.2])
    assert all(r["Harrell_final_equal_seed_delta"]==pytest.approx(.1) for r in rows)
    bad=comparison();bad["Harrell_delta"]=.11
    with pytest.raises(ValueError,match="ACTUAL_FINAL_CONTRAST"):
        view.seed_contrasts([bad],[aggregate("B0",[.5]),aggregate("B5",[.5,.6,.7])])


def test_repair_uses_fold_denominators_and_preserves_uno_na():
    rows=[{"comparison":"B5-B0","seed":31,"fold":fold,"repair_credit":1.,"damage_credit":0.,
        "comparable_pairs":1 if fold==1 else 100,"Uno_repair_credit":2.,"Uno_damage_credit":0.,
        "Uno_pair_weight_denominator":20.} for fold in range(1,6)]
    actual=view.repair_micro(rows)[("B5-B0",31)]
    assert actual["Harrell_delta"]==pytest.approx(5/401)
    assert actual["Uno60_delta"]==pytest.approx(.1)
    rows[0]["Uno_pair_weight_denominator"]=0.
    assert view.repair_micro(rows)[("B5-B0",31)]["Uno60_delta"] is None
    with pytest.raises(ValueError,match="FIVE_FOLDS"):
        view.repair_micro(rows[:-1])


def test_compact_csv_blank_means_na_and_rejects_nested_or_nonfinite():
    rows=list(csv.DictReader(io.StringIO(view.csv_bytes([{"x":1.2345678901234567,"absent":None}]).decode())))
    assert rows[0]["absent"]=="" and float(rows[0]["x"])==1.2345678901234567
    with pytest.raises(ValueError,match="NONFINITE"):
        view.csv_bytes([{"bad":float("nan")}])
    with pytest.raises(ValueError,match="NONSCALAR"):
        view.csv_bytes([{"bad":[1,2]}])


def test_full_curve_index_links_every_actual_update_without_copying_rows():
    phase=[];coverage=[]
    for seed,start in ((31,0),(47,400)):
        p={"family":"MAIN","split":"outer1_inner0","seed":seed,"receipt_sha256":str(seed)*32,
           "optimizer_steps":400,"evaluate_valid":True,"selected_epoch":400,"actual_neural_fits":1,
           "actual_checkpoint_replays":4,"elapsed_seconds":1.,"peak_cuda_allocated_bytes":100,
           "parameters":{"pretraining_trainable":50},"source_checkpoint_sha256":"a"*64}
        phase.append(p);coverage.append({"family":"MAIN","split":p["split"],"seed":seed,
            "source_receipt_sha256":p["receipt_sha256"],"curve_start_offset":start,"curve_records":400})
    index={"actual_epoch_records":800,"source_coverage":coverage,
        "curve_shards":[{"file":"first.csv","data_records":500},{"file":"last.csv","data_records":300}]}
    actual=view.curve_links(index,phase)
    assert len(actual)==2 and sum(r["curve_epoch_records"] for r in actual)==800
    assert actual[1]["curve_CSV_shards"]=="first.csv;last.csv"
    assert all(r["shared_teacher_or_risk_head_fits_assigned_to_this_source"]==0 for r in actual)
    changed=copy.deepcopy(index);changed["source_coverage"][0]["source_receipt_sha256"]="b"*64
    with pytest.raises(ValueError,match="RECEIPT_BINDING"):
        view.curve_links(changed,phase)


def test_append_only_guard_precedes_actual_source_reads(tmp_path):
    (tmp_path/view.OUTPUTS[0]).write_text("keep original")
    with pytest.raises(ValueError,match="APPEND_ONLY"):
        view.generate(tmp_path)
    assert (tmp_path/view.OUTPUTS[0]).read_text()=="keep original"

"""Budget accounting on synthetic receipts only; no real model fit or curve read."""
import copy
import csv
import io
import json
import math
from pathlib import Path

import pytest

from scripts import semantic_b0b7_p2_training_budget_audit as audit


def direction(gap,covered=6):
    return {"anchors":8,"anchors_covered":covered,"coverage":covered/8,"mean_negative_count":3.,
        "mean_reciprocal_midrank":.5,"paired_gap_mean":gap,"top1_tie_averaged":.25}


def validation(huber,gap):
    return {"auxiliary_loss":huber+.2,"program_loss":huber,"pair_hard_loss":2.,"lambda_pair":.1,
        "median_prediction_to_target_sd_ratio":.8,"program":{"patient_equal_masked_huber":huber,
        "macro_program_pearson":.1,"macro_program_spearman":.1,"patient_equal_profile_pearson_secondary":.2,
        "patients_with_measurement":8,"measured_target_cells":400,"valid_pearson_programs":50,"coverage_fraction":1.},
        "retrieval_hard":{"image_to_gene":direction(gap),"gene_to_image":direction(gap)},
        "retrieval_random":{"image_to_gene":direction(gap/2 if gap is not None else None),"gene_to_image":direction(gap/2 if gap is not None else None)}}


def receipt(*,inner=True,selected=400,steps=400):
    hist=[]
    values={50:(.0503,.8),100:(.05,1.),200:(.0515,1.5),400:(.0504,2.)}
    for epoch in range(1,steps+1):
        row={"epoch":epoch,"optimizer_step":epoch,"train_program_loss":.2-epoch*.0001,
             "train_pair_loss":3.-epoch*.001,"train_auxiliary_loss":.5-epoch*.0002,
             "gradient_norms":{"w":.1,"bias":.01},"active_parameter_elements":64}
        if inner and epoch in values:row["validation"]=validation(*values[epoch])
        hist.append(row)
    return {"optimizer_steps":steps,"history":hist,"evaluate_valid":inner,"valid_patients":8 if inner else 0,
        "selected_epoch":selected,"fixed_epoch":None if inner else steps,"program_huber_tolerance":.001,
        "states":[{"epoch":e,"actual_replay":True,"replay_max_error":0.} for e in audit.CHECKPOINTS if e<=steps],
        "selected_checkpoint_sha256":"a"*64}


def source(role="inner",seed=31):
    return {"family":"MAIN","split":"outer1_inner0" if role=="inner" else "outer1_final",
            "role":role,"outer":1,"inner":0 if role=="inner" else None,"seed":seed}


def test_permission_gate_precedes_any_input_read(monkeypatch,tmp_path):
    monkeypatch.setattr(audit,"sha",lambda *a:(_ for _ in ()).throw(AssertionError("no inputs")))
    with pytest.raises(ValueError,match="AUTHORIZATION_REQUIRED_BEFORE_INPUT_READ"):
        audit.generate("unread","unread","unread",tmp_path/"not_created")


def test_selected400_is_not_same_as_raw_huber_min400_or_convergence():
    r=receipt();s=audit.summarize_cell(source(),r,receipt_sha="b"*64)
    assert s["selected_epoch_at400"] is True
    assert s["raw_Huber_minimum_epochs"]==[100] and not s["raw_Huber_has_minimum_at400"]
    assert s["changes"]["200_to_400"]["val_patient_equal_masked_huber"]["improvement"]==pytest.approx(.0011)
    assert s["changes"]["200_to_400"]["val_Huber_improvement_exceeds_original_selection_band"] is True
    assert s["budget400_is_not_a_convergence_certificate"] and not s["budget_extension_automatically_triggered"]


def test_huber_and_hardgap_directions_are_not_combined_into_one_score():
    r=receipt();r["history"][399]["validation"]=validation(.049,1.1)
    r["selected_epoch"]=audit.selection_epoch(r)
    s=audit.summarize_cell(source(),r,receipt_sha="b"*64)
    c=s["changes"]["200_to_400"]
    assert c["val_patient_equal_masked_huber"]["improvement"]>0
    assert c["val_hard_gap_available_direction_mean"]["improvement"]<0
    assert s["raw_Huber_strict_unique_minimum_at400"]


def test_original_huber_band_is_labelled_engineering_tolerance_not_plateau_proof():
    assert audit.change(.05,.05,lower_better=True)["status"]=="NUMERICALLY_FLAT"
    result=audit.change(.05,.05001,lower_better=True)
    assert result["status"]=="WORSE_DIRECTION"
    r=receipt();r["history"][199]["validation"]=validation(.0505,1.5)
    r["selected_epoch"]=audit.selection_epoch(r)
    s=audit.summarize_cell(source(),r,receipt_sha="b"*64)
    assert s["changes"]["200_to_400"]["val_Huber_change_within_original_selection_band"] is True


def test_outer_refit_has_only_real_executed_steps_and_no_validation():
    r=receipt(inner=False,selected=50,steps=50)
    s=audit.summarize_cell(source("outer_final"),r,receipt_sha="b"*64)
    assert s["optimizer_steps"]==50 and s["fixed_checkpoints"]=={}
    assert s["changes"]["200_to_400"]["val_patient_equal_masked_huber"]["status"]=="NA_UNSUPPORTED_OR_NOT_RUN"
    row=audit.flatten_epoch(source("outer_final"),r["history"][-1])
    assert row["validation_available"] is False and row["val_patient_equal_masked_huber"] is None
    r["history"][-1]["validation"]=validation(.05,1.)
    with pytest.raises(ValueError,match="CONTAINS_VALIDATION"):
        audit.summarize_cell(source("outer_final"),r,receipt_sha="b"*64)


def test_low_support_gap_stays_na_not_zero_and_nonfinite_is_rejected():
    v=validation(.1,None)
    assert audit.hard_gap(v) is None
    r=receipt();r["history"][99]["validation"]=v
    flat=audit.flatten_epoch(source(),r["history"][99])
    assert flat["val_hard_gap_supported_directions"]==0 and flat["val_hard_gap_available_direction_mean"] is None
    r["history"][0]["train_program_loss"]=float("nan")
    with pytest.raises(ValueError,match="NONFINITE"):
        audit.summarize_cell(source(),r,receipt_sha="b"*64)


def test_missing_epoch_duplicate_or_wrong_selection_cannot_be_certified():
    for bad in ("missing","duplicate","selection"):
        r=receipt()
        if bad=="missing":r["history"].pop()
        elif bad=="duplicate":r["history"][49]["epoch"]=49
        else:r["selected_epoch"]=50
        with pytest.raises(ValueError):audit.summarize_cell(source(),r,receipt_sha="b"*64)


def test_complete_csv_shards_obey_bytes_total_rows_and_float_roundtrip(tmp_path):
    writer=audit.CsvShards(tmp_path,byte_limit=180,row_limit=4)
    values=[1/3,1e-30,.12345678901234567,.5,1.]
    for i,value in enumerate(values):writer.append({"family":"MAIN","epoch":i+1,"loss":value,"validation":None})
    files=writer.finish();restored=[]
    assert writer.total==5 and sum(x["data_records"] for x in files)==5 and len(files)>=2
    for item in files:
        path=tmp_path/item["file"];assert path.stat().st_size<=180
        assert item["total_rows_including_header"]<=4 and audit.sha(path)==item["sha256"]
        rows=list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
        assert all(r["validation"]=="" for r in rows);restored.extend(float(r["loss"]) for r in rows)
    assert restored==values


def test_family_role_seed_grouping_counts_actual_updates_without_fake_outer400():
    inner=audit.summarize_cell(source(),receipt(),receipt_sha="a"*64)
    final=audit.summarize_cell(source("outer_final"),receipt(inner=False,selected=50,steps=50),receipt_sha="b"*64)
    result=audit.grouped_summary([inner,final],("family","role"))
    by_role={r["role"]:r for r in result}
    assert by_role["inner"]["optimizer_steps"]==400 and by_role["outer_final"]["optimizer_steps"]==50
    assert by_role["inner"]["selected_epoch_frequency"]["400"]==1
    assert by_role["outer_final"]["val_patient_equal_masked_huber_200to400"]["NA_fits"]==1


def test_json_shards_and_public_curve_index_remain_ordinary_readable_objects(tmp_path):
    rows=[{"family":"MAIN","seed":31,"ordinal":i} for i in range(1001)]
    files=audit.json_records(tmp_path,"synthetic",rows)
    assert len(files)==2 and sum(x["records"] for x in files)==1001
    assert all(isinstance(json.loads((tmp_path/x["file"]).read_text()),list) for x in files)

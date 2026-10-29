"""Synthetic corruption tests; no data, model, optimizer or formal-fit evidence."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics

import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


review=module("r7_independent",ROOT/"scripts/bgb_en_r7_independent_review.py")
fixtures=module("r7_report_fixture",ROOT/"tests/test_bgb_en_r7_report.py")


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding="utf-8")


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)


def make_public(root,mode="defined"):
    case=fixtures.make_tree(root/"private_fixture",mode=mode)
    public=root/"anonymous";fixtures.report.build(case["private"],case["config"],public)
    export=review.export_direct(case["private"]/"REPLAY_AUDIT.json",public,private_root=case["private"])
    return dict(case=case,public=public,export=export)


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    return make_public(tmp_path_factory.mktemp("r7_independent_template"))


@pytest.fixture
def public(template,tmp_path):
    target=tmp_path/"anonymous";shutil.copytree(template["public"],target)
    return target


def update_export_hash(public,name):
    path=public/review.EXPORT_NAME;audit=read(path)
    row=next(r for r in audit["artifacts"] if r["file"] == name);content=(public/name).read_bytes()
    row.update(bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),rows=len(read_csv(public/name)))
    write(path,audit)


def test_complete_export_and_independent_review_are_bound_anonymous_metadata(template,public,tmp_path):
    audit=template["export"]
    assert audit["direct_rows"] == 99 and audit["secondary_rows"] == 117
    assert len(audit["source_metadata"]) == 118
    assert audit["runtime"]["model_checkpoint_array_label_files_read"] is False
    assert audit["validated_upstream_certificate"]["independent_model_forwards_performed_here"] == 0
    output=tmp_path/"review";result=review.review(public,output)
    assert result["status"] == "PASS_ANONYMOUS_ARITHMETIC_AND_HASH_BINDINGS"
    assert result["verified_counts"] == dict(arms=13,primary_fold_rows=117,paired_comparisons=11,
        direct_partition_seed_rows=99,diagnostic_shards=33,diagnostic_model_rows=99)
    assert result["fixed_gate1_status"] == "DIRECTIONAL_PATIENT_CONTEXT_INCREMENT"
    assert result["secondary_review"] == "PASS_117_FROZEN_DSS_RISK_PFI_ROWS"
    assert result["declared_upstream_budget"]["formal_neural_fits"] == 99
    assert result["declared_upstream_budget"]["formal_optimizer_updates"] == 7920
    assert result["declared_upstream_budget"]["zero_fit_baseline_reference_rows"] == 18
    assert result["upstream_replay_metadata"]["actual_checkpoint_records"] == 297
    assert result["upstream_replay_metadata"]["actual_model_replay_performed_by_this_script"] is False
    assert result["runtime"]["optimizer_fits"] == result["runtime"]["model_forwards"] == 0
    assert set(p.name for p in output.iterdir()) == set(review.REVIEW_NAMES)
    text=(output/review.REVIEW_NAMES[0]).read_text(encoding="utf-8")
    assert str(tmp_path) not in text and "patient_ids" not in text and "state_dict" not in text
    with pytest.raises(ValueError,match="APPEND_ONLY_OUTPUT_EXISTS"):
        review.review(public,output)


def test_independent_weighted_arithmetic_differs_from_fold_mean_and_wrong_uno_weights(template):
    public=template["public"]
    grid=review.metric_grid(read_csv(public/"R7_FOLD_METRICS.csv"));summaries=review.aggregate(grid)
    pooled=summaries["GATE1"]["Harrell"]
    foldmean=statistics.mean(grid[("GATE1",p,s)]["Harrell"] for p in review.PARTITIONS for s in review.SEEDS)
    assert abs(pooled-foldmean)>.03
    pairs=review.paired(grid,summaries,read_csv(public/review.DIRECT_NAME))
    target=next(r for r in pairs if (r["candidate"],r["reference"]) == ("GATE1","B1"))
    for row in target["per_seed"]:
        assert row["repair"]-row["damage"] == pytest.approx(row["delta_Harrell"],abs=1e-12)
        assert row["Uno_repair"]-row["Uno_damage"] == pytest.approx(row["delta_Uno60"],abs=1e-12)
    rows=[r for r in read_csv(public/review.DIRECT_NAME) if r["candidate"] == "GATE1" and r["reference"] == "B1" and r["seed"] == "31"]
    wrong=sum(float(r["Uno_repair"])*float(r["Harrell_den"]) for r in rows)/sum(float(r["Harrell_den"]) for r in rows)
    assert target["per_seed"][0]["Uno_repair"] != pytest.approx(wrong)


@pytest.mark.parametrize("mode,expected",(("mixed","CURRENT_GATE1_NOT_SUPPORTED"),("undefined","INCONCLUSIVE_UNDEFINED_METRIC")))
def test_mixed_and_undefined_do_not_become_joint_success_or_zero(tmp_path,mode,expected):
    case=make_public(tmp_path/mode,mode=mode)
    result=review.review(case["public"],tmp_path/(mode+"_review"))
    assert result["fixed_gate1_status"] == expected
    pair=next(r for r in result["recomputed_pair_comparisons"] if r["candidate"] == "GATE1" and r["reference"] == "B1")
    assert pair["joint_directional_support"] is False
    if mode == "undefined":
        assert result["recomputed_primary_DSS"]["GATE1"]["Harrell"] is None
        assert pair["delta_Harrell"] is pair["repair"] is pair["damage"] is None
    else:
        assert pair["delta_Harrell"]>0 and pair["delta_Uno60"]<0 and pair["mixed_metric_direction"] is True


@pytest.mark.parametrize("corruption,code",(
    ("fold_duplicate","EXACT_117_ANONYMOUS_METRIC_ROWS_REQUIRED"),
    ("fold_aggregate","DSS_SEED_COMPONENT_AGGREGATION_MISMATCH"),
    ("direct_byte","EXPORTED_ANONYMOUS_CSV_BYTES_CHANGED"),
    ("direct_identity","DIRECT_PARTITION_REPAIR_DAMAGE_IDENTITY_FAILED"),
    ("replay_sha","DIRECT_EXPORT_SOURCE_REPLAY_BINDING_CHANGED"),
    ("gate","FIXED_GATE1_DECISION_OR_SCOPE_CHANGED"),
    ("diagnostic","DIAGNOSTIC_SHARD_BYTES_OR_SHA_CHANGED"),
    ("budget","R7_DECLARED_BUDGET_OR_AGGREGATION_CHANGED"),
))
def test_corruption_fails_without_complete_review_output(public,tmp_path,corruption,code):
    if corruption in ("fold_duplicate","fold_aggregate"):
        path=public/"R7_FOLD_METRICS.csv";rows=read_csv(path)
        if corruption == "fold_duplicate":rows.append(copy.deepcopy(rows[0]))
        else:
            rows[0]["Harrell_num"]=str(float(rows[0]["Harrell_num"])+.1)
            rows[0]["Harrell"]=str(float(rows[0]["Harrell_num"])/float(rows[0]["Harrell_den"]))
        write_csv(path,rows)
    elif corruption in ("direct_byte","direct_identity"):
        path=public/review.DIRECT_NAME;rows=read_csv(path)
        rows[0]["repair"]=str(float(rows[0]["repair"])+.01);write_csv(path,rows)
        if corruption == "direct_identity":update_export_hash(public,review.DIRECT_NAME)
    elif corruption == "replay_sha":
        path=public/review.EXPORT_NAME;value=read(path);value["source_replay_sha256"]="0"*64;write(path,value)
    elif corruption in ("gate","budget"):
        path=public/"R7_RESULTS.json";value=read(path)
        if corruption == "gate":value["decision"]["GATE2_selected_from_validation"]=True
        else:value["budget"]["formal_optimizer_updates"]=8000
        write(path,value)
    else:
        path=public/"R7_DIAGNOSTICS_ADD_p0.json";path.write_bytes(path.read_bytes()+b" ")
    with pytest.raises(ValueError,match=code):review.review(public,tmp_path/"must_not_complete")
    assert not (tmp_path/"must_not_complete"/review.REVIEW_NAMES[0]).exists()


def test_secondary_corruption_is_detected_independently(public,tmp_path):
    path=public/review.SECONDARY_NAME;rows=read_csv(path)
    rows[0]["Uno_num"]=str(float(rows[0]["Uno_num"])+.25)
    rows[0]["Uno60"]=str(float(rows[0]["Uno_num"])/float(rows[0]["Uno_den"]))
    write_csv(path,rows);update_export_hash(public,review.SECONDARY_NAME)
    with pytest.raises(ValueError,match="PFI_SEED_COMPONENT_AGGREGATION_MISMATCH"):
        review.review(public,tmp_path/"must_not_complete")


def test_export_refuses_incomplete_or_nonactual_checkpoint_metadata(template,tmp_path):
    path=tmp_path/"REPLAY_AUDIT.json";value=read(template["case"]["private"]/"REPLAY_AUDIT.json")
    value["units"][0]["checkpoints"][1]["actual_state_dict_forward_replayed"]=False;write(path,value)
    with pytest.raises(ValueError,match="UPSTREAM_ACTUAL_CHECKPOINT_REPLAY_NOT_VERIFIED"):
        review.export_direct(path,tmp_path/"must_not_exist")
    assert not (tmp_path/"must_not_exist").exists()


def test_export_completion_metadata_SHA_is_checked_and_never_reads_dummy_weight_bytes(template,tmp_path,monkeypatch):
    private=tmp_path/"private";shutil.copytree(template["case"]["private"],private)
    path=private/"cells/ADD_p0_seed31/COMPLETE.json";value=read(path);value["elapsed_seconds"]=123.;write(path,value)
    original=Path.read_bytes
    def no_weights(path):
        assert path.suffix not in (".pt",".npz",".npy",".h5ad")
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",no_weights)
    with pytest.raises(ValueError,match="BOUND_COMPLETION_OR_REFERENCE_METADATA_CHANGED"):
        review.export_direct(private/"REPLAY_AUDIT.json",tmp_path/"must_not_exist",private_root=private)
    assert not (tmp_path/"must_not_exist").exists()


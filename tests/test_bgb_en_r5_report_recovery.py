"""Exercise the real producer schema without training or model deserialization."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import bgb_en_r5_report as original
from scripts.bgb_en_r5_report_recovery import prepared_origin_guard, reference_inventory, run
from scripts.bgb_en_r5_run import make_references
from src.bgb_en.r5_protocol import PARTITIONS, canonical_sha256, make_protocol_lock, sha256, write_protocol_lock


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def environment(tmp_path):
    root, private = tmp_path / "repo", tmp_path / "private"
    actual = Path(__file__).resolve().parents[1]
    configs = ("configs/bgb_en/r5_main_v1.json", "configs/bgb_en/r5_mechanism_v1.json")
    source = "scripts/bgb_en_r5_report.py"
    for name in (*configs, source):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((actual / name).read_bytes())
    sentinel = root / "reports/bgb_en/KEEP.md"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("untouched historical result\n")
    lock = make_protocol_lock(root, *configs, [source], data_fingerprints={"cohort": "a" * 64},
        prepared_receipt_sha256={f"outer1_inner{i}": str(i) * 64 for i in PARTITIONS})
    write_protocol_lock(private / "LOCK.json", lock)
    save(private / "QUALIFICATION.json", dict(status="PASS_R5_REAL_DATA_AND_LEGACY_PARITY",
        contract_sha256=lock["contract_sha256"]))
    raw = dict(t=np.array([8., 14., 70., 9., 18., 72.]), e=np.array([1, 0, 1, 1, 1, 0]),
        pfi_valid=np.ones(6, dtype=bool), pfi_t=np.array([7., 12., 65., 6., 15., 64.]),
        pfi_e=np.array([1, 1, 0, 1, 1, 0]))
    prepared = dict(train=np.array([0, 1, 2]), valid=np.array([3, 4, 5]),
        CP=np.array([.5, -.2, -.7, .4, .1, -.8]), base=np.array([.6, -.3, -.6, .5, .0, -.9]))
    for part in PARTITIONS:
        origin = dict(prepared_receipt_sha256=str(part) * 64,
            prepared_artifact_sha256=canonical_sha256(["payload", part]),
            train_hash=canonical_sha256(["train", part]), valid_hash=canonical_sha256(["valid", part]),
            train_n=3, valid_n=3, teachers_reused=3, teacher_refits=0,
            upstream_supervised_fits_reused_not_new=14,
            risk_distribution={"train_mean": 0.0}, scalar_beta=.1)
        make_references(argparse.Namespace(private=private, partition=part),
                        dict(raw=raw, lock=lock), prepared, origin)
    return root, private, lock


def args(environment):
    root, private, _ = environment
    return argparse.Namespace(private=private, stage="main", output=root / "reports/bgb_en_r5")


def test_real_reference_writer_fails_old_guard_and_recovers_without_mutation(environment):
    root, private, lock = environment
    references = sorted((private / "references").glob("*.json"))
    assert len(references) == 18
    example = original.read(references[0])
    assert "prepared_origin" in example and "contract" not in example
    original_guard = original.prepared_origin_guard
    before = {p.name: sha256(p) for p in references}
    source_before = sha256(original.__file__)
    lock_before = sha256(private / "LOCK.json")
    with pytest.raises(ValueError, match="PREPARED_REFERENCE_PARTITION_ORIGIN_CHANGED"):
        original.run(args(environment), root=root)
    audit, decision = run(args(environment), root=root)
    assert original.prepared_origin_guard is original_guard
    assert audit["main_summary"]["arms"]["A0"]["status"] == "COMPLETE"
    assert audit["main_summary"]["arms"]["A1"]["status"] == "COMPLETE"
    assert audit["main_summary"]["arms"]["A2"]["status"] != "COMPLETE"
    assert decision["decision"] == "INCONCLUSIVE" and not decision["R6_formal_start_eligible"]
    certificate = original.read(args(environment).output / "R5_REPORT_SCHEMA_RECOVERY_main.json")
    assert certificate["baseline_references_qualified"] == 18
    assert certificate["contract_sha256"] == lock["contract_sha256"]
    assert certificate["new_optimizer_fits"] == certificate["model_deserializations"] == 0
    assert {p.name: sha256(p) for p in references} == before
    assert sha256(original.__file__) == source_before
    assert sha256(private / "LOCK.json") == lock_before
    assert (root / "reports/bgb_en/KEEP.md").read_text() == "untouched historical result\n"
    for path in args(environment).output.glob("*"):
        assert str(private).encode() not in path.read_bytes()


@pytest.mark.parametrize("change, message", [
    ("wrong_origin", "PREPARED_REFERENCE_PARTITION_ORIGIN_CHANGED"),
    ("wrong_identity", "FILENAME_OR_RECEIPT_IDENTITY"),
    ("missing_top", "REQUIRED_SCHEMA_MISSING"),
    ("conflict", "CONFLICTING_PREPARED_REFERENCE_ORIGINS"),
    ("missing_hash", "ORIGIN_HASH_MISSING_OR_INVALID"),
    ("wrong_contract", "TOPLEVEL_CONTRACT_CHANGED"),
])
def test_recovery_rejects_wrong_schema_or_source(environment, change, message):
    root, private, lock = environment
    path = private / "references/A0_p0_seed31.json"
    value = original.read(path)
    if change == "wrong_origin":
        value["prepared_origin"]["prepared_receipt_sha256"] = "1" * 64
    elif change == "wrong_identity":
        value["partition"] = 1
    elif change == "missing_top":
        value["contract"] = dict(prepared_origin=value.pop("prepared_origin"))
    elif change == "conflict":
        value["contract"] = dict(prepared_origin=copy.deepcopy(value["prepared_origin"]))
        value["contract"]["prepared_origin"]["prepared_receipt_sha256"] = "1" * 64
    elif change == "missing_hash":
        del value["prepared_origin"]["train_hash"]
    elif change == "wrong_contract":
        value["contract_sha256"] = "b" * 64
    save(path, value)
    guard = original.prepared_origin_guard
    with pytest.raises(ValueError, match=message):
        run(args(environment), root=root)
    assert original.prepared_origin_guard is guard


def test_missing_reference_or_duplicate_renaming_is_rejected(environment):
    _, private, lock = environment
    path = private / "references/A0_p0_seed31.json"
    path.rename(path.with_name("duplicate.json"))
    with pytest.raises(ValueError, match="EXACT_18"):
        reference_inventory(private, lock)


def test_fits_require_nested_origin_and_match_reference(environment):
    _, private, lock = environment
    ref = original.read(private / "references/A0_p0_seed31.json")
    value = dict(status="COMPLETE_R5_FIT", partition=0, prepared_origin=ref["prepared_origin"])
    with pytest.raises(ValueError, match="REQUIRED_SCHEMA_MISSING"):
        prepared_origin_guard(value, lock, 0)
    value["contract"] = dict(prepared_origin=value.pop("prepared_origin"))
    assert prepared_origin_guard(value, lock, 0) == ref["prepared_origin"]
    value["contract"]["prepared_origin"]["valid_hash"] = "a" * 64
    save(private / "cells/A3_MATCHED_p0_seed31/COMPLETE.json", value)
    with pytest.raises(ValueError, match="FIT_AND_REFERENCE_PREPARED_ORIGIN_DIFFER"):
        reference_inventory(private, lock)


def test_frozen_reporter_bytes_stay_pinned(environment):
    root, _, _ = environment
    (root / "scripts/bgb_en_r5_report.py").write_text("modified\n")
    with pytest.raises(ValueError, match="source bytes changed"):
        run(args(environment), root=root)

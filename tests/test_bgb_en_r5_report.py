"""Synthetic metadata fixtures test reporting; no actual model is loaded."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.bgb_en_r5_report import run
from src.bgb_en.r5_protocol import (
    ALL_MAIN_ARMS, MAIN_ARMS, MECHANISM_CONTROLS, PARTITIONS, SEEDS,
    make_protocol_lock, write_protocol_lock,
)

SCORES = dict(A0=(.77, .75), A1=(.78, .76), A2=(.79, .77), A3=(.785, .765),
              A4=(.82, .80), A5=(.80, .78), A6=(.798, .778), A7=(.797, .777), A8=(.792, .772))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def metric(h, u, ref=(.78, .76)):
    return dict(Harrell=h, Uno60=u, Harrell_num=h * 100, Harrell_den=100,
                Uno_num=u * 200, Uno_den=200, repair=max(h - ref[0], 0),
                damage=max(ref[0] - h, 0), Uno_repair=max(u - ref[1], 0),
                Uno_damage=max(ref[1] - u, 0), risk_shift_sd_ratio=.1)


@pytest.fixture
def environment(tmp_path):
    root, private = tmp_path / "repo", tmp_path / "private"
    actual = Path(__file__).resolve().parents[1]
    configs = ("configs/bgb_en/r5_main_v1.json", "configs/bgb_en/r5_mechanism_v1.json")
    for name in configs:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((actual / name).read_bytes())
    source = root / "src/frozen.py"
    source.parent.mkdir(parents=True)
    source.write_text("frozen = True\n")
    sentinel = root / "reports/bgb_en/KEEP.md"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("old formal report\n")
    lock = make_protocol_lock(root, *configs, ["src/frozen.py"],
        data_fingerprints={"cohort": "a" * 64},
        prepared_receipt_sha256={f"outer1_inner{i}": str(i) * 64 for i in PARTITIONS})
    write_protocol_lock(private / "LOCK.json", lock)
    save(private / "QUALIFICATION.json", dict(status="PASS_R5_REAL_DATA_AND_LEGACY_PARITY",
                                              contract_sha256=lock["contract_sha256"]))
    return root, private, lock


def add_unit(environment, arm, part, seed, control="MATCHED"):
    root, private, lock = environment
    h, u = SCORES[arm] if control == "MATCHED" else (.81, .79)
    metrics = metric(h, u)
    contract = dict(contract_sha256=lock["contract_sha256"],
        prepared_origin={"prepared_receipt_sha256": str(part) * 64}, arm=arm,
        partition=part, seed=seed, training_control=control)
    value = dict(arm=arm, report_arm=arm if control == "MATCHED" else control, training_control=control,
        partition=part, seed=seed, metrics=metrics,
        metrics_vs=dict(CP_SCALAR=metrics, CP_STRONG=metric(h, u, SCORES["A0"]),
                        PFI_frozen_DSS_risk=metric(h - .1, u - .1)),
        contract=contract, parameters=dict(registered=200, requires_grad=100))
    if arm in ("A0", "A1"):
        value.update(status="R5_BASELINE_REFERENCE", actual_new_optimizer_fits=0,
                     deterministic_seed_reference=seed != 31)
        path = private / "references" / f"{arm}_p{part}_seed{seed}.json"
    else:
        cell = private / "cells" / f"{arm}_{control}_p{part}_seed{seed}"
        cell.mkdir(parents=True, exist_ok=True)
        payload = (arm + control + str(part) + str(seed)).encode()
        (cell / "model.pt").write_bytes(payload)
        value.update(status="COMPLETE_R5_FIT", group="MAIN" if control == "MATCHED" else "MECHANISM",
            artifact_sha256=hashlib.sha256(payload).hexdigest(), epochs=80, optimizer_updates=80,
            actual_new_optimizer_fits=1, elapsed_seconds=2.0,
            initialization=dict(backbone_sha256="b" * 64, full_state_sha256=hashlib.sha256(arm.encode()).hexdigest()),
            hidden_private=dict(patient_id="DO_NOT_EXPORT_ME", absolute_path="/" + "root" + "/private_not_public"))
        path = cell / "COMPLETE.json"
    save(path, value)
    return path


def add_main(environment):
    for arm in ALL_MAIN_ARMS:
        for part in PARTITIONS:
            for seed in SEEDS:
                add_unit(environment, arm, part, seed)


def args_for(environment, stage="main"):
    root, private, _ = environment
    return argparse.Namespace(private=private, output=root / "reports/bgb_en_r5", stage=stage)


def add_selection_and_mechanism(environment):
    root, private, lock = environment
    audit, _ = run(args_for(environment), root=root)
    selection = dict(audit["proposed_selection"], contract_sha256=lock["contract_sha256"])
    assert selection["selected_arm"] == "A4"
    save(private / "SELECTION.json", selection)
    for control in MECHANISM_CONTROLS:
        for part in PARTITIONS:
            for seed in SEEDS:
                add_unit(environment, "A4", part, seed, control)
    direct = [dict(selected_arm="A4", control=control, partition=part, seed=seed,
                   matched_vs_control=metric(.82, .80, (.81, .79)), risk_SD_ratio=.1)
              for control in MECHANISM_CONTROLS for part in PARTITIONS for seed in SEEDS]
    main = [dict(selected_arm="A4", reference=reference, partition=part, seed=seed,
                 matched_vs_control=metric(.82, .80, SCORES[reference]), risk_SD_ratio=.1)
            for reference in ("A1", "A2", "A3", "A8") for part in PARTITIONS for seed in SEEDS]
    save(private / "REPLAY_AUDIT.json", dict(status="PASS_COMPLETE_R5_CHECKPOINT_REPLAY",
        contract_sha256=lock["contract_sha256"], passed_units=108, failed_units=0,
        main_passed=63, mechanism_passed=45, paired_retrained_controls=direct, paired_main_controls=main))


def test_partial_reports_keep_null_means_and_unknown_interruption_updates(environment):
    root, private, _ = environment
    add_unit(environment, "A3", 0, 31)
    save(private / "cells/A4_MATCHED_p0_seed31/START.json", dict(actual_optimizer_updates_completed=0))
    audit, decision = run(args_for(environment), root=root)
    assert audit["main_summary"]["arms"]["A3"]["completed_cells"] == 1
    assert audit["main_summary"]["arms"]["A3"]["Harrell"] is None
    assert decision["decision"] == "INCONCLUSIVE" and not decision["R6_formal_start_eligible"]
    ledger = json.loads((args_for(environment).output / "RUN_LEDGER.json").read_text())
    assert ledger["unknown_failed_or_interrupted_update_counts"] == 1
    started = next(row for row in ledger["units"] if row["status"].startswith("STARTED_"))
    assert started["actual_new_optimizer_fits"] is None and started["optimizer_updates"] is None
    assert (root / "reports/bgb_en/KEEP.md").read_text() == "old formal report\n"


def test_main_complete_still_cannot_authorize_R6(environment):
    root, _, _ = environment
    add_main(environment)
    audit, decision = run(args_for(environment), root=root)
    assert audit["main_summary"]["status"] == "COMPLETE"
    assert audit["proposed_selection"]["selected_arm"] == "A4"
    assert not decision["R6_formal_start_eligible"]


def test_complete_actual_receipt_grid_replay_and_direct_credits_are_separated(environment):
    root, _, _ = environment
    add_main(environment)
    add_selection_and_mechanism(environment)
    audit, decision = run(args_for(environment, "final"), root=root)
    assert audit["status"] == "COMPLETE_REQUIRED_R5_TRAINING_AND_REPLAY"
    assert decision["decision"] == "R5_MECHANISM_DIRECTIONAL_GO"
    assert audit["diagnostics"]["direct_retrained_pair_rows"] == 45
    assert audit["diagnostics"]["direct_main_pair_rows"] == 36
    out = args_for(environment).output
    assert len(list(csv.DictReader((out / "R5_SEED_FOLD.csv").open()))) == 252
    ledger = json.loads((out / "RUN_LEDGER.json").read_text())
    assert ledger["complete_new_supervised_optimizer_fits"] == 108
    assert ledger["complete_optimizer_updates"] == 8640
    assert ledger["reference_accounting"]["actual_new_optimizer_fits"] == 0
    for path in out.glob("*"):
        assert b"DO_NOT_EXPORT_ME" not in path.read_bytes()
        assert (b"/" + b"root" + b"/private_not_public") not in path.read_bytes()


def test_weights_are_only_hashed_and_bad_payload_is_rejected(environment):
    root, _, _ = environment
    path = add_unit(environment, "A4", 0, 31)
    (path.parent / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="PAYLOAD_SHA"):
        run(args_for(environment), root=root)


def test_wrong_prepared_partition_or_conflicting_terminal_is_rejected(environment):
    root, _, _ = environment
    path = add_unit(environment, "A4", 0, 31)
    value = json.loads(path.read_text())
    value["contract"]["prepared_origin"]["prepared_receipt_sha256"] = "1" * 64
    save(path, value)
    with pytest.raises(ValueError, match="PREPARED_REFERENCE"):
        run(args_for(environment), root=root)
    value["contract"]["prepared_origin"]["prepared_receipt_sha256"] = "0" * 64
    save(path, value)
    save(path.parent / "FAILED.json", dict(type="RuntimeError", message="DO_NOT_EXPORT_ME"))
    with pytest.raises(ValueError, match="CONFLICTING"):
        run(args_for(environment), root=root)


def test_mechanism_initialization_mismatch_does_not_change_locked_main_choice(environment):
    root, _, _ = environment
    add_main(environment)
    add_selection_and_mechanism(environment)
    path = environment[1] / "cells/A4_WRONG_QUERY_p0_seed31/COMPLETE.json"
    value = json.loads(path.read_text())
    value["initialization"]["full_state_sha256"] = "e" * 64
    save(path, value)
    audit, decision = run(args_for(environment, "final"), root=root)
    assert audit["proposed_selection"]["selected_arm"] == "A4"
    assert not audit["initializations"]["mechanism_fair"]
    assert decision["decision"] == "INCONCLUSIVE" and not decision["R6_formal_start_eligible"]


def test_report_cannot_overwrite_old_namespace_or_ignore_source_changes(environment):
    root, _, _ = environment
    args = args_for(environment)
    args.output = root / "reports/bgb_en"
    with pytest.raises(ValueError, match="CANNOT_WRITE_OLD"):
        run(args, root=root)
    (root / "src/frozen.py").write_text("changed = True\n")
    with pytest.raises(ValueError, match="source bytes"):
        run(args_for(environment), root=root)

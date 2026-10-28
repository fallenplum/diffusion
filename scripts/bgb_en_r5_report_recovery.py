"""Reporting-only R5 producer-schema recovery; frozen sources stay unchanged.

The actual reference writer stores prepared_origin at the receipt top level;
neural-fit receipts store it under contract. Only this schema guard is adapted
in memory. Metrics, selection, training, immutable locks and receipts are never
modified, and model payloads are only hashed by the original reporter.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_r5_report as frozen_report
from src.bgb_en.r5_protocol import PARTITIONS, SEEDS, canonical_sha256, require, sha256, verify_protocol_lock

REPORT_SOURCE = "scripts/bgb_en_r5_report.py"


def prepared_origin_guard(value, lock, partition):
    """Accept the actual writer's required location, never a fallback location."""
    require(value.get("partition") == partition, "PREPARED_REFERENCE_PARTITION_IDENTITY_CHANGED")
    contract = value.get("contract", {})
    require(isinstance(contract, dict), "PREPARED_REFERENCE_CONTRACT_NOT_MAPPING")
    top = value.get("prepared_origin")
    nested = contract.get("prepared_origin")
    if top is not None and nested is not None:
        require(top == nested, "CONFLICTING_PREPARED_REFERENCE_ORIGINS")
    if value.get("status") == "R5_BASELINE_REFERENCE":
        origin = top
    elif value.get("status") == "COMPLETE_R5_FIT":
        origin = nested
    else:
        raise ValueError("UNKNOWN_PREPARED_REFERENCE_RECEIPT_SCHEMA")
    require(isinstance(origin, dict), "PREPARED_REFERENCE_REQUIRED_SCHEMA_MISSING")
    expected = lock["prepared_receipt_sha256"][f"outer1_inner{partition}"]
    require(origin.get("prepared_receipt_sha256") == expected,
            "PREPARED_REFERENCE_PARTITION_ORIGIN_CHANGED")
    return origin


def reference_inventory(private, lock):
    """Qualify exactly 18 actual-writer references and return anonymous hashes."""
    paths = sorted((Path(private) / "references").glob("*.json"))
    expected = {f"{arm}_p{part}_seed{seed}.json": (arm, part, seed)
                for arm in ("A0", "A1") for part in PARTITIONS for seed in SEEDS}
    require({p.name for p in paths} == set(expected) and len(paths) == 18,
            "EXACT_18_BASELINE_REFERENCES_REQUIRED_FOR_RECOVERY")
    rows, origins = [], {}
    for path in paths:
        arm, part, seed = expected[path.name]
        value = frozen_report.read(path)
        require(value.get("status") == "R5_BASELINE_REFERENCE"
                and value.get("group") == "REFERENCE"
                and value.get("arm") == value.get("report_arm") == arm
                and value.get("partition") == part and value.get("seed") == seed,
                "BASELINE_REFERENCE_FILENAME_OR_RECEIPT_IDENTITY_CHANGED")
        require(value.get("contract_sha256") == lock["contract_sha256"],
                "BASELINE_REFERENCE_TOPLEVEL_CONTRACT_CHANGED")
        require(value.get("actual_new_optimizer_fits") == 0
                and value.get("deterministic_seed_reference") is (seed != 31),
                "BASELINE_REFERENCE_FIT_ACCOUNTING_CHANGED")
        origin = prepared_origin_guard(value, lock, part)
        for name in ("prepared_receipt_sha256", "prepared_artifact_sha256", "train_hash", "valid_hash"):
            require(isinstance(origin.get(name), str)
                    and re.fullmatch(r"[0-9a-f]{64}", origin[name]) is not None,
                    "BASELINE_REFERENCE_ORIGIN_HASH_MISSING_OR_INVALID")
        if part in origins:
            require(origin == origins[part], "BASELINE_REFERENCE_SAME_PARTITION_ORIGIN_CHANGED")
        origins[part] = origin
        rows.append(dict(arm=arm, partition=part, seed=seed, receipt_sha256=sha256(path),
                         prepared_origin_sha256=canonical_sha256(origin)))
    # The existing fit writer uses the same qualified prepared bundle. Require
    # consistency with all completed fits before the unchanged reporter loads it.
    for path in sorted((Path(private) / "cells").glob("*/COMPLETE.json")):
        value = frozen_report.read(path)
        part = value.get("partition")
        require(part in PARTITIONS, "FIT_PREPARED_PARTITION_OUTSIDE_MATRIX")
        require(prepared_origin_guard(value, lock, part) == origins[part],
                "FIT_AND_REFERENCE_PREPARED_ORIGIN_DIFFER")
    return rows


def run(args, *, root=ROOT):
    root, private = Path(root).resolve(), Path(args.private).resolve()
    lock_path = private / "LOCK.json"
    lock_before = sha256(lock_path)
    lock = frozen_report.read(lock_path)
    verify_protocol_lock(lock, root)
    pinned = lock["sources"].get(REPORT_SOURCE)
    require(pinned is not None and sha256(root / REPORT_SOURCE) == pinned
            and sha256(frozen_report.__file__) == pinned, "FROZEN_REPORTER_SOURCE_NOT_LOCKED_OR_CHANGED")
    old_reports = frozen_report.old_report_snapshot(root)
    references_before = reference_inventory(private, lock)
    completed_before = {p.parent.name: sha256(p) for p in sorted((private / "cells").glob("*/COMPLETE.json"))}
    original_guard = frozen_report.prepared_origin_guard
    frozen_report.prepared_origin_guard = prepared_origin_guard
    try:
        audit, decision = frozen_report.run(args, root=root)
    finally:
        frozen_report.prepared_origin_guard = original_guard
        require(sha256(lock_path) == lock_before, "RECOVERY_CHANGED_IMMUTABLE_LOCK")
        require(sha256(root / REPORT_SOURCE) == pinned and sha256(frozen_report.__file__) == pinned,
                "RECOVERY_CHANGED_FROZEN_REPORTER")
        require(frozen_report.old_report_snapshot(root) == old_reports, "RECOVERY_CHANGED_OLD_FORMAL_REPORTS")
        require(reference_inventory(private, lock) == references_before, "RECOVERY_CHANGED_REFERENCE_RECEIPTS")
        require({p.parent.name: sha256(p) for p in sorted((private / "cells").glob("*/COMPLETE.json"))}
                == completed_before, "RECOVERY_CHANGED_COMPLETE_RECEIPTS")
    certificate = dict(schema="bgb_en_r5_report_schema_recovery_v1", status="PASS_REPORTING_SCHEMA_ONLY_RECOVERY",
        stage=args.stage, contract_sha256=lock["contract_sha256"], private_lock_file_sha256=lock_before,
        frozen_reporter_sha256=pinned, recovery_wrapper_sha256=sha256(__file__),
        adaptation="reference_top_level_and_fit_nested_prepared_origin_only",
        baseline_references_qualified=18, reference_receipts=references_before,
        reference_manifest_sha256=canonical_sha256(references_before),
        completed_receipts_unchanged_sha256=canonical_sha256(completed_before),
        original_receipts_edited=False, immutable_lock_unchanged=True, frozen_sources_unchanged=True,
        old_formal_reports_unchanged=True, metrics_or_selection_rule_changed=False,
        model_deserializations=0, new_optimizer_fits=0, optimizer_updates=0)
    frozen_report.write(Path(args.output) / f"R5_REPORT_SCHEMA_RECOVERY_{args.stage}.json", certificate)
    return audit, decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--stage", choices=("main", "final"), required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bgb_en_r5")
    args = parser.parse_args()
    audit, decision = run(args)
    import json
    print(json.dumps(dict(status=audit["status"], stage=args.stage, decision=decision["decision"],
        R6_formal_start_eligible=decision["R6_formal_start_eligible"],
        model_deserializations=0, new_optimizer_fits=0, reporting_schema_recovered=True)), flush=True)
    if not audit["initializations"]["fair"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

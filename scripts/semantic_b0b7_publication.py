"""Explicit P0/P1 publication gate; unchanged privacy rules, no fitting or Git.

The inherited scanner inspects every readable field in every small JSON cell.
Only the server's existing identifier registry completes the final gate; local
preflight deliberately does not claim that protected-identifier scan occurred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_publication as inherited
from scripts import semantic_b0b7_report as reporter

BASE_COMMIT = "b579d8acf83bbf1e7b55a5f166c5aa98e4b94f0d"
SCANNER_SHA = "4c70929d57459a9e97774e6319bd455deedc29fbcedf077131c918c53db60e98"
DENYLIST_SHA = "d73b70939a38277f8f9f693e9958afcc9435893cd05f5c1d6a6d08a100ffe867"
FROZEN_SHA = "bf0de7a8c7b552634be4125cf4ce117f19a9d74e68a4ea65fa6813709b60d6a9"
GATE_SHA = "2282e18cbc697f352ac75923376e9db7460ec19c5d151d2cbdd08ae27de729cc"
INPUT_SHA = "93d2419a95c129e72e360fc24748e4fef6c909a01830208ecef5fc2d7d0b1b6c"
AGGREGATE_SHA = "2a368d408726344fbeebf5b264ac0319c879c6b320267971fbaa198f633b1cb2"
AUDIT = "reports/semantic_b0b7/PUBLICATION_SOURCE_AUDIT.json"
MANIFEST = "reports/semantic_b0b7/PUBLICATION_MANIFEST.json"
INDEX = "reports/semantic_b0b7/04_PRETEXT_LOSSLESS_INDEX.json"
REVIEW = "reports/semantic_b0b7/INDEPENDENT_P1_AUDIT.json"
CHANCE = "reports/semantic_b0b7/P1_EXACT_RETRIEVAL_CHANCE.json"
BUILD = "reports/semantic_b0b7/04_PRETEXT_REPORT_BUILD_AUDIT.json"
PROTOCOL = "B0_B7_语义条件互补三模态_严格实验方案_Codex.md"
POLICY = "configs/semantic_b0b7/evaluation_policy_20261031.json"
DIAGNOSTIC = "reports/semantic_b0b7/P1_SAVED_SCALE_ROUTE_AUDIT.json"
HISTORY_MANIFESTS = {
    "reports/spgel_s5/S5_PUBLICATION_MANIFEST.json":
        "104745978313887748d487f8e4240b6e87a01c9f503cf57663aa989b33d90682",
    "reports/spgel/HEST_OFFICIAL_PUBLICATION_MANIFEST_20261029.json":
        "7d1dd6afc1f01b667271e0db7d6d3557c73ddbbc8d2640567784bd07538b0a3b",
    "reports/spgel/HEST_NEURAL_FIXED12_PUBLICATION_MANIFEST_20261029.json":
        "c3114a85a9fa6ff8efc43ff14bed36e4498bf841ad42f61368498d11970e7ab3",
}

read = lambda path: json.loads(Path(path).read_text(encoding="utf-8-sig"))
sha = inherited.sha256
require = inherited.require


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def safe_name(name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name, "SEMANTIC_UNSAFE_PATH")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name, "SEMANTIC_UNSAFE_PATH")
    require(name not in {AUDIT, MANIFEST}, "SEMANTIC_GENERATED_OUTPUT_CANNOT_BE_INPUT")
    require(name == PROTOCOL or any(re.fullmatch(pattern, name) for pattern in (
        r"src/semantic_b0b7/[A-Za-z0-9_]+\.py",
        r"configs/semantic_b0b7/[A-Za-z0-9_]+\.json",
        r"scripts/(?:semantic_b0b7[A-Za-z0-9_]*|verify_semantic_b0b7[A-Za-z0-9_]*)\.py",
        r"tests/test_semantic_b0b7[A-Za-z0-9_]*\.py",
        r"reports/semantic_b0b7/[A-Za-z0-9_]+\.(?:json|md|csv)")), "SEMANTIC_UNKNOWN_PUBLIC_PATH")
    return name


def checked(root, name, expected=None, size=None):
    root, path = Path(root).resolve(), Path(root) / name
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            "SEMANTIC_MISSING_OR_ESCAPING_FILE:" + name)
    require(not any(p.is_symlink() for p in path.parents if p.is_relative_to(root)),
            "SEMANTIC_SYMLINK_PARENT:" + name)
    if expected is not None:
        require(sha(path) == expected, "SEMANTIC_FILE_SHA_CHANGED:" + name)
    if size is not None:
        require(path.stat().st_size == size, "SEMANTIC_FILE_SIZE_CHANGED:" + name)
    return path


def inventory(root, allowlist):
    value = read(allowlist)
    require(value.get("base_git_commit") == BASE_COMMIT, "SEMANTIC_BASE_COMMIT_CHANGED")
    rows = value.get("files")
    require(isinstance(rows, list) and bool(rows) and len(rows) == len({r["file"] for r in rows}),
            "SEMANTIC_EXPLICIT_UNIQUE_ALLOWLIST_REQUIRED")
    for row in rows:
        require(set(row) == {"file", "sha256", "bytes"}, "SEMANTIC_INVALID_ALLOWLIST_ROW")
        checked(root, safe_name(row["file"]), row["sha256"], row["bytes"])
    return sorted(rows, key=lambda r: r["file"])


def scan(root, rows, pattern):
    require(sha(inherited.__file__) == SCANNER_SHA, "SEMANTIC_ORIGINAL_SCANNER_CHANGED")
    original = inherited.ALLOWLIST
    inherited.ALLOWLIST = frozenset(r["file"] for r in rows)
    try:
        result = [inherited.scan_file(Path(root), r["file"], pattern) for r in rows]
    finally:
        inherited.ALLOWLIST = original
    require(result == rows, "SEMANTIC_SCANNED_BYTES_CHANGED")
    return result


def historical_files(root):
    result = {}
    for name, digest in HISTORY_MANIFESTS.items():
        manifest = read(checked(root, name, digest))
        for row in manifest["files"]:
            require(row["file"] not in result, "SEMANTIC_DUPLICATE_HISTORICAL_FILE")
            checked(root, row["file"], row["sha256"], row["bytes"])
            result[row["file"]] = row["sha256"]
        result[name] = digest
    require(len(result) == 121, "SEMANTIC_EXPECTED_54_PLUS_67_UNCHANGED_HISTORY")
    return result


def completion(root, historical_root, gate_path):
    gate = read(gate_path)
    require(sha(gate_path) == GATE_SHA and gate.get("status") == "PASS"
            and gate.get("scope") == "P1_AUXILIARY_QUALIFICATION_ONLY"
            and gate.get("input_sha256") == INPUT_SHA
            and gate.get("P2_DSS_authorized") is False
            and gate.get("scientific_mechanism_supported") is False, "SEMANTIC_GATE_BINDING_CHANGED")
    frozen = gate["frozen_sources"]
    require(len(frozen) == 13 and canonical(frozen) == FROZEN_SHA
            and gate["source_sha256"] == FROZEN_SHA, "SEMANTIC_FROZEN_SOURCE_SET_CHANGED")
    for name, digest in frozen.items():
        source_root = root if (Path(root) / name).is_file() else historical_root
        checked(source_root, name, digest)
    report, raw = reporter.restore_public_report(Path(root) / INDEX)
    require(canonical(report) == canonical(json.loads(raw.decode("utf-8"))),
            "SEMANTIC_RESTORED_VALUE_AND_ORIGINAL_BYTES_DIFFER")
    reporter.validate_aggregate(report)
    require(hashlib.sha256(raw).hexdigest() == AGGREGATE_SHA, "SEMANTIC_ORIGINAL_AGGREGATE_BYTES_CHANGED")
    require(report["input_sha256"] == INPUT_SHA and report["gate_sha256"] == GATE_SHA
            and report["frozen_sources"] == frozen, "SEMANTIC_REPORT_INPUT_OR_SOURCE_CHANGED")
    require(report["neural_fits"] == 12 and report["optimizer_steps"] == 4800
            and report["actual_checkpoint_replays"] == 48 and report["decoder_fits"] == 3
            and report["scalar_program_linear_system_solves"] == 150
            and report["dss_models_fitted"] == report["dss_scores_selected"] == 0
            and report["P2_started"] is False, "SEMANTIC_COMPLETION_SCOPE_CHANGED")
    review = read(Path(root) / REVIEW)
    require(review["status"] == "PASS_INDEPENDENT_REPORT_ARITHMETIC_AND_BYTE_HASH_AUDIT"
            and review["producer_report_sha256"] == AGGREGATE_SHA
            and review["input_sha256"] == INPUT_SHA and review["gate_sha256"] == GATE_SHA
            and review["frozen_sources"] == frozen and review["frozen_source_files_verified"] == 13,
            "SEMANTIC_INDEPENDENT_REVIEW_BINDING_CHANGED")
    require(review["neural_cell_receipts_verified"] == 12
            and review["producer_optimizer_updates_verified"] == 4800
            and review["checkpoint_files_sha256_verified"] == review["producer_actual_replays_recorded"] == 48
            and review["producer_replay_max_error"] == 0
            and review["ridge_decoder_families"] == 3 and review["scalar_program_linear_system_rhs"] == 150
            and review["independent_model_forwards"] == review["independent_optimizer_updates"] == 0
            and review["checkpoint_arrays_deserialized"] is False
            and review["raw_patient_arrays_loaded"] is False
            and review["DSS_models_fitted"] == review["DSS_scores_selected"] == 0
            and review["P2_started"] is False, "SEMANTIC_REVIEW_EXECUTION_OR_SCOPE_CHANGED")
    tuples = {(x["arm"], x["seed"], x["epoch"]) for x in review["checkpoint_sha256"]}
    require(len(review["checkpoint_sha256"]) == 48 and tuples == {(a, s, e)
        for a in ("B3", "B4", "B6a", "B4_MAP_SHUFFLE") for s in (31, 47, 83) for e in (50, 100, 200, 400)},
        "SEMANTIC_EXACT_REPLAY_UNIT_SET_CHANGED")
    expected_receipts = {(x["arm"], x["seed"]): x["private_receipt_sha256"] for x in report["cells"]}
    reviewed_receipts = {(x["arm"], x["seed"]): x["sha256"] for x in review["receipt_sha256"]}
    require(len(review["receipt_sha256"]) == 12 and reviewed_receipts == expected_receipts,
            "SEMANTIC_REVIEWED_PRIVATE_RECEIPT_SET_DIFFERS_FROM_AGGREGATE")
    checked(root, "scripts/verify_semantic_b0b7_p1.py", review["reviewer_source_sha256"])
    chance = read(Path(root) / CHANCE)
    require(chance["status"] == "COMPLETE" and chance["input_sha256"] == INPUT_SHA
            and chance["model_forwards"] == chance["fits"] == 0
            and chance["DSS_read"] is False and chance["checkpoint_selection_changed"] is False,
            "SEMANTIC_EXACT_CHANCE_INPUT_BINDING_CHANGED")
    checked(root, "scripts/semantic_b0b7_chance.py", chance["source_sha256"])
    build = read(Path(root) / BUILD)
    require(build["status"] == "PASS" and build["source_utf8_sha256"] == AGGREGATE_SHA
            and build["source_utf8_bytes"] == len(raw) and build["original_utf8_roundtrip_exact"] is True
            and build["canonical_roundtrip_exact"] is True and build["optimizer_steps_run_by_reporter"] == 0,
            "SEMANTIC_REPORT_BUILD_SOURCE_OR_ROUNDTRIP_CHANGED")
    required = {INDEX, REVIEW, CHANCE, BUILD, "scripts/verify_semantic_b0b7_p1.py",
        "scripts/semantic_b0b7_report.py", "scripts/semantic_b0b7_chance.py",
        "scripts/semantic_b0b7_publication.py", "tests/test_semantic_b0b7_publication.py"}
    lock = read(Path(root) / "reports/semantic_b0b7/LOCK.json")
    require(lock["frozen_scientific_sources"] == frozen and lock["pre_fit_gate_sha256"] == GATE_SHA
            and lock["pre_fit_input_sha256"] == INPUT_SHA and lock["P2_started"] is False,
            "SEMANTIC_PUBLIC_LOCK_PROJECTION_CHANGED")
    checked(root, PROTOCOL, lock["original_plan_sha256"])
    policy = read(Path(root) / POLICY)
    require(policy["compute_confidence_intervals"] is False
            and policy["compute_bootstrap_for_confidence_intervals"] is False
            and policy["confidence_interval_is_advancement_gate"] is False
            and policy["completed_P1_scientific_source_or_selection_modified"] is False,
            "SEMANTIC_LATEST_USER_EVALUATION_POLICY_NOT_PRESERVED")
    diagnostic = read(Path(root) / DIAGNOSTIC)
    require(diagnostic["status"] == "COMPLETE_DESCRIPTIVE_ONLY"
            and diagnostic["input_sha256"] == INPUT_SHA
            and diagnostic["source_aggregate_sha256"] == AGGREGATE_SHA
            and diagnostic["model_forwards"] == diagnostic["optimizer_updates"] == 0
            and diagnostic["checkpoint_selection_changed"] is False,
            "SEMANTIC_SAVED_STATE_DIAGNOSTIC_SCOPE_OR_SOURCE_CHANGED")
    checked(root, "scripts/semantic_b0b7_checkpoint_diagnostics.py", diagnostic["source_sha256"])
    state_hashes = {(x["arm"], x["seed"], x["epoch"]): x["sha256"] for x in review["checkpoint_sha256"]}
    require(len(diagnostic["rows"]) == 12 and {(x["arm"], x["seed"]) for x in diagnostic["rows"]}
            == set(expected_receipts) and all(state_hashes[(x["arm"], x["seed"], x["selected_epoch"])]
                == x["selected_state_sha256"] for x in diagnostic["rows"]),
            "SEMANTIC_DIAGNOSTIC_SELECTED_STATE_BINDING_CHANGED")
    completion_note = read(Path(root) / "reports/semantic_b0b7/09_COMPLETION_AND_REPLAY_AUDIT.json")
    require(completion_note["neural_fits"] == 12 and completion_note["optimizer_updates"] == 4800
            and completion_note["actual_saved_model_replays"] == 48 and completion_note["DSS_model_fits"] == 0
            and completion_note["P2_started"] is False
            and completion_note["independent_review_sha256"] == sha(Path(root) / REVIEW)
            and completion_note["official_alignment_public_projection_sha256"] == sha(Path(root) /
                "reports/semantic_b0b7/02_TEXT_ALIGNMENT_AND_CONCEPT_AUDIT.json"),
            "SEMANTIC_PUBLIC_COMPLETION_NOTE_BINDING_CHANGED")
    required.update({PROTOCOL, POLICY, DIAGNOSTIC, "scripts/semantic_b0b7_checkpoint_diagnostics.py"})
    required.update("reports/semantic_b0b7/" + name for name in (
        "LOCK.json", "00_PROTOCOL_LOCK.md", "01_DATA_ENDPOINT_FOLD_AUDIT.json",
        "02_TEXT_ALIGNMENT_AND_CONCEPT_AUDIT.json", "03_FAIRNESS_CAPACITY_AND_GRADIENT_AUDIT.json",
        "05_INNER_SELECTION_SUMMARY.md", "08_RISK_REPAIR_DAMAGE_AND_CALIBRATION.md",
        "09_COMPLETION_AND_REPLAY_AUDIT.json", "10_DECISION.md", "10_DECISION.json",
        "11_LIMITATIONS_AND_NEGATIVE_RESULTS.md", "12_REPRODUCE.md", "COMPUTE_COST.json",
        "EVALUATION_POLICY_ADDENDUM_20261031.md"))
    required.update(name for name in frozen if name != "configs/phase6a_titan_crc_morphology_concepts.json")
    index = read(Path(root) / INDEX)
    require(len(index["parts"]) == 12, "SEMANTIC_TWELVE_READABLE_CELL_PARTS_REQUIRED")
    require(len(build["files"]) == len({r["file"] for r in build["files"]}),
            "SEMANTIC_DUPLICATE_BUILT_PUBLIC_REPORT")
    for row in build["files"]:
        require(Path(row["file"]).name == row["file"], "SEMANTIC_REPORT_BUILD_PATH_ESCAPE")
        relative = safe_name("reports/semantic_b0b7/" + row["file"])
        checked(root, relative, row["sha256"], row["bytes"])
        required.add(relative)
    required.update(safe_name("reports/semantic_b0b7/" + x["file"]) for x in index["parts"])
    return {"frozen_sources": frozen, "input_sha256": INPUT_SHA, "gate_sha256": GATE_SHA,
        "source_report_original_sha256": AGGREGATE_SHA, "source_report_original_bytes": len(raw),
        "lossless_original_bytes_reconstruction": True, "ordinary_readable_json_cell_shards": 12,
        "independent_review_sha256": sha(Path(root) / REVIEW), "exact_chance_sha256": sha(Path(root) / CHANCE),
        "neural_fits": 12, "optimizer_updates": 4800, "actual_saved_checkpoint_replays": 48,
        "replay_maximum_error": 0, "Ridge_decoder_families": 3, "scalar_Ridge_right_hand_sides": 150,
        "DSS_models_fitted": 0, "P2_started": False, "required_public_file_closure": sorted(required)}


def require_closure(rows, required):
    require(set(required) <= {row["file"] for row in rows},
            "SEMANTIC_PUBLIC_ALLOWLIST_OMITS_REQUIRED_REPRODUCTION_OR_REVIEW_FILES")


def publish(root, historical_root, allowlist, gate, denylist):
    root = Path(root).resolve()
    rows = inventory(root, allowlist)
    history = historical_files(historical_root)
    require(not ({r["file"] for r in rows} & set(history)), "SEMANTIC_HISTORICAL_FILE_IN_NEW_REQUEST")
    evidence = completion(root, historical_root, gate)
    require_closure(rows, evidence["required_public_file_closure"])
    require(sha(denylist) == DENYLIST_SHA, "SEMANTIC_ESTABLISHED_DENYLIST_BYTES_CHANGED")
    pattern = inherited.identifier_pattern(inherited.validate_denylist(read(denylist)))
    scanned = scan(root, rows, pattern)
    audit = {"schema": "semantic_b0b7_publication_source_audit_v1", "status": "COMPLETE_P0_P1_ONLY_DSS_NOT_STARTED",
        "base_git_commit": BASE_COMMIT, "science_completion": evidence,
        "historical_files_unchanged": len(history), "historical_public_source_sha256": history,
        "original_privacy_scanner_sha256": SCANNER_SHA, "established_denylist_sha256": DENYLIST_SHA,
        "private_identifier_list_exported": False, "all_candidate_fields_exact_identifier_scanned": True,
        "new_model_fits": 0, "model_or_private_array_deserializations": 0, "Git_operations": 0,
        "publication_files": len(scanned), "publication_bytes": sum(x["bytes"] for x in scanned),
        "mechanism_or_DSS_gain_inferred_from_completion": False}
    encode = lambda x: (json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    encoded_audit = encode(audit)
    audit_row = {"file": AUDIT, "bytes": len(encoded_audit), "sha256": hashlib.sha256(encoded_audit).hexdigest()}
    manifest = {"schema": "semantic_b0b7_publication_manifest_v1", "status": audit["status"],
        "base_git_commit": BASE_COMMIT, "source_audit_sha256": audit_row["sha256"],
        "files": sorted(scanned + [audit_row], key=lambda r: r["file"]),
        "file_count_including_audit_excluding_manifest": len(scanned) + 1,
        "manifest_does_not_hash_itself": True, "Git_operations": 0}
    for name, value in ((AUDIT, audit), (MANIFEST, manifest)):
        raw = encode(value)
        inherited.validate_public_json(value)
        require(len(raw) <= 2 * 1024**2 and not pattern.search(raw.decode())
                and not any(p.search(raw.decode()) for p in inherited.PATTERNS.values()),
                "SEMANTIC_GENERATED_PUBLIC_OUTPUT_PRIVACY_FAILURE")
        require(not (root / name).exists(), "SEMANTIC_EXISTING_PUBLICATION_OUTPUT_PRESERVED")
    for name, raw in ((AUDIT, encoded_audit), (MANIFEST, encode(manifest))):
        with (root / name).open("xb") as handle:
            handle.write(raw)
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("local", "publish"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--denylist", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "local":
        rows = inventory(args.root, args.allowlist)
        scan(args.root, rows, re.compile(r"(?!x)x"))
        result = {"schema": "semantic_b0b7_local_publication_preflight_v1",
            "status": "PASS_EXISTING_PATTERN_SIZE_AND_NO_ARRAY_RULES_ONLY", "files": rows,
            "exact_protected_identifier_scan": "NOT_RUN_REQUIRES_ESTABLISHED_SERVER_REGISTRY",
            "new_model_fits": 0, "Git_operations": 0}
    else:
        require(args.gate is not None and args.denylist is not None, "SEMANTIC_FINAL_GATE_AND_DENYLIST_REQUIRED")
        result = publish(args.root, args.historical_root or args.root, args.allowlist, args.gate, args.denylist)
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"schema": result["schema"], "status": result["status"]}))


if __name__ == "__main__":
    main()

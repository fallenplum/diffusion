"""S5 metadata publication gate. No fitting, artifact decoding or Git operations.

Local preflight applies the unchanged inherited pattern and no-array rules only.
The complete gate additionally requires the established server identifier list,
all 84/252 source attribution units and all 40/120 shared-feature benchmark units.
Author-model SOTA reproduction is explicitly outside this completion statement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import hest_official_publication as prior

read, canonical, require, sha = prior.read, prior.canonical, prior.require, prior.sha
BASE_COMMIT = "3c6391c5a8d65ffd8cb416351dc1b8fa887e627a"
BASE_TREE = "b3a57dbc2de56904bbb21be2e876e5318d77e1c8"
A_SHA = "4ace8497f44f7c909835a8940fe4806b597b07e55c3580daffb75f79d03d5699"
B_SHA = "20966b4d2d92df28f70d18ee83f111e0b6db0e15154b662b70a9efc119057b24"
SCANNER_SHA = "4c70929d57459a9e97774e6319bd455deedc29fbcedf077131c918c53db60e98"
HISTORY = {
    "reports/spgel/HEST_OFFICIAL_PUBLICATION_MANIFEST_20261029.json":
        "7d1dd6afc1f01b667271e0db7d6d3557c73ddbbc8d2640567784bd07538b0a3b",
    "reports/spgel/HEST_NEURAL_FIXED12_PUBLICATION_MANIFEST_20261029.json":
        "c3114a85a9fa6ff8efc43ff14bed36e4498bf841ad42f61368498d11970e7ab3",
}
AUDIT = "reports/spgel_s5/S5_PUBLICATION_SOURCE_AUDIT.json"
MANIFEST = "reports/spgel_s5/S5_PUBLICATION_MANIFEST.json"
A_REPORT = "reports/spgel_s5/S5A_RESULTS.json"
B_REPORT = "reports/spgel_s5/S5B1_RESULTS.json"
B_REVIEW = "reports/spgel_s5/S5B1_INDEPENDENT_ARITHMETIC_AUDIT.json"
B_REVIEWER = "scripts/verify_spgel_s5b1_final.py"
COST = "reports/spgel_s5/S5B1_SUPERSEDED_PRETEST_COST.json"
QUAL = "reports/spgel_s5/S5B_OFFICIAL_QUALIFICATION.json"
ADMIN = {"scripts/spgel_s5_publication.py", "tests/test_spgel_s5_publication.py"}
QUAL_FILES = {QUAL, "reports/spgel_s5/S5B_OFFICIAL_QUALIFICATION.md",
    "reports/spgel_s5/S5B_BLEEP_RELEASED_DATASET_CONTRACT_PROBE_20261029.json",
    "reports/spgel_s5/S5B_BLEEP_SERVER_DATASET_CONTRACT_PROBE_20261029.json"}


def safe_name(name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name,
            "S5_UNSAFE_PUBLIC_PATH")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "S5_UNSAFE_PUBLIC_PATH")
    valid = any(re.fullmatch(pattern, name) for pattern in (
        r"src/spgel_s5(?:_hest)?/[A-Za-z0-9_]+\.py",
        r"configs/spgel_s5(?:_hest)?/[A-Za-z0-9_]+\.json",
        r"scripts/(?:spgel_s5[A-Za-z0-9_]*|verify_spgel_s5[A-Za-z0-9_]*)\.py",
        r"tests/test_spgel_s5[A-Za-z0-9_]*\.py",
        r"reports/spgel_s5/[A-Za-z0-9_]+\.(?:json|md|csv)"))
    require(valid and name not in {AUDIT, MANIFEST}, "S5_UNKNOWN_OR_GENERATED_INPUT_PATH")
    return name


def checked_file(root, name, expected):
    path = root / name
    require(path.is_file() and not path.is_symlink()
        and path.resolve().is_relative_to(root.resolve())
        and not any(p.is_symlink() for p in path.parents if p.is_relative_to(root)),
        "S5_MISSING_OR_ESCAPING_FILE:" + name)
    require(sha(path) == expected, "S5_FILE_SHA_CHANGED:" + name)
    return path


def inventory(root, manifest):
    rows = read(manifest)["files"]
    require(isinstance(rows, list) and rows and len(rows) == len({r["file"] for r in rows}),
            "S5_EXPLICIT_UNIQUE_INPUT_LIST_REQUIRED")
    for row in rows:
        path = checked_file(root, safe_name(row["file"]), row["sha256"])
        require(path.stat().st_size == row["bytes"], "S5_INPUT_FILE_SIZE_CHANGED")
    return rows


def local_scan(root, rows):
    require(sha(Path(prior.inherited.__file__)) == SCANNER_SHA, "S5_INHERITED_SCANNER_CHANGED")
    old = prior.inherited.ALLOWLIST
    prior.inherited.ALLOWLIST = frozenset(r["file"] for r in rows)
    try:
        scanned = [prior.inherited.scan_file(root, r["file"], re.compile(r"(?!x)x")) for r in rows]
    finally:
        prior.inherited.ALLOWLIST = old
    require(scanned == rows, "S5_LOCAL_SCANNED_BYTES_CHANGED")
    return dict(schema="spgel_s5_local_privacy_preflight_v1",
        status="PASS_EXISTING_PATTERN_AND_NO_ARRAY_RULES_ONLY", files=scanned,
        exact_private_identifier_scan="NOT_RUN_REQUIRES_ESTABLISHED_SERVER_REGISTRY",
        new_fits=0, model_or_array_deserializations=0, git_operations=0)


def historical_inventory(root):
    """Validate all 65 prior payloads plus their two immutable manifests."""
    result = {}
    for name, digest in HISTORY.items():
        path = checked_file(root, name, digest)
        for row in read(path)["files"]:
            file = row["file"]
            require(file not in result, "S5_DUPLICATE_HISTORICAL_FILE")
            previous = checked_file(root, file, row["sha256"])
            require(previous.stat().st_size == row["bytes"], "S5_HISTORICAL_SIZE_CHANGED")
            result[file] = row["sha256"]
        require(name not in result, "S5_HISTORICAL_SELF_HASH_UNEXPECTED")
        result[name] = digest
    require(len(result) == 67, "S5_EXACT_67_HISTORICAL_FILES_REQUIRED")
    return result


def sources(root, values, expected, count, history=None):
    require(isinstance(values, dict) and len(values) == count and canonical(values) == expected,
            "S5_FROZEN_SCIENTIFIC_SOURCE_SET_CHANGED")
    for name, digest in values.items():
        # Only the four original B dependencies may resolve from the old tree.
        inherited = name in (history or {})
        if inherited:
            require(history[name] == digest, "S5_INHERITED_B_SOURCE_CHANGED")
        else:
            checked_file(root, safe_name(name), digest)


def a_evidence(root, private, config, assets):
    """Rebuild anonymous aggregates from actual receipts; never decode weights."""
    from scripts import spgel_s5_report as producer
    cfg, lock, cells, ledger = producer.load_matrix(private, config, assets, root=root)
    sources(root, lock["sources"], A_SHA, 13)
    rebuilt = producer.build_report(cfg, lock, cells, ledger)
    require(canonical(rebuilt) == canonical(producer.restore_report(root / A_REPORT)),
            "S5_A_REPORT_NOT_DERIVED_FROM_CURRENT_COMPLETE_MATRIX")
    directory = root / "tmp"
    directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="s5_public_metadata_", dir=directory) as temp:
        producer.write_report(Path(temp), rebuilt)
        bindings = {}
        for name in producer.NAMES:
            relative = "reports/spgel_s5/" + name
            require(sha(Path(temp)/name) == sha(root/relative), "S5_A_PUBLIC_OUTPUT_DRIFT:" + name)
            bindings[relative] = sha(root/relative)
    return dict(schema="spgel_s5a_anonymous_completion_evidence_v1",
        status="PASS_84_FITS_72800_UPDATES_252_ACTUAL_REPLAYS_AND_EIGHT_PUBLIC_OUTPUTS",
        source_files=lock["sources"], source_canonical_sha256=A_SHA,
        contract_sha256=lock["contract_sha256"], actual_neural_fits=84,
        actual_optimizer_updates=72800, actual_saved_state_replays=252,
        public_outputs=bindings, producer_script="scripts/spgel_s5_report.py",
        producer_script_sha256=sha(root/"scripts/spgel_s5_report.py"),
        report_reconstructed_byte_exact=True, new_fits_by_publication=0,
        model_or_array_deserializations=0, checkpoint_and_prediction_file_hashing=True)


def superseded_cost(root):
    value = read(root / COST)
    require(value["schema"] == "spgel_s5b_superseded_pretest_cost_v1"
        and value["status"] == "INVALIDATED_BEFORE_FORMAL_TEST_DUE_TO_STATIC_OBJECTIVE_FIDELITY"
        and value["completed_development_fits"] == 8
        and value["interrupted_development_processes"] == 2
        and value["interrupted_initialized_fit_cells"] == value["interrupted_before_fit_initialization"] == 1
        and value["completed_formal_fits"] == value["official_test_evaluations"] == 0
        and value["known_complete_optimizer_updates"] == 2560
        and value["partial_optimizer_updates_exact"] is None
        and value["partial_optimizer_updates_upper_bound_total"] == 320
        and value["previous_PCA_fits"] == value["previous_scaler_fits"] == 4
        and value["performance_results_used_in_final_tables"] is False
        and value["evidence_used_to_change_model"] ==
            "static_objective_and_operator_audit_no_validation_or_test_score_used"
        and len(value["completed_units"]) == 8
        and sum(r["observed_optimizer_updates"] for r in value["completed_units"]) == 2560
        and value["archived_source_zip_sha256"] ==
            "aee67a092f6b4bf5a532694a4a8a5f87d862255eca8f8fa71a6f78a5a1dac377",
        "S5_SUPERSEDED_COST_OR_STATIC_REPAIR_DISCLOSURE_CHANGED")
    return sha(root / COST)


def verify_b_review(root, review, history):
    require(review["schema"] == "spgel_s5b1_independent_arithmetic_audit_v1"
        and review["status"] == "PASS_INDEPENDENT_METADATA_HASH_AND_ARITHMETIC_REVIEW",
        "S5_B_COMPLETE_INDEPENDENT_REVIEW_REQUIRED")
    require(review["counts"] == dict(development_fits=20, final_fits=20,
        saved_state_replays=120, fold_metric_rows=20, gene_metric_rows=1000),
        "S5_B_ACTUAL_COMPLETION_COUNTS_CHANGED")
    sources(root, review["source_files"], B_SHA, 12, history)
    require(review["source_canonical_sha256"] == B_SHA
        and review["producer_report_sha256"] == sha(root / B_REPORT)
        and review["operations"] == dict(model_fits=0, model_forwards=0,
            array_or_model_deserializations=0, current_artifact_file_hashing=True)
        and review["convergence_not_proven"] is True
        and review["SOTA_strict_reproduction_claimed"] is False
        and review["own_fold_selection_recomputed_independently"] is True
        and review["exact_120_unique_replay_tuple_set_verified"] is True
        and review["raw_prediction_Pearson_recomputed_by_reviewer"] is False
        and math.isfinite(review["production_actual_replay_maximum_prediction_error"])
        and 0 <= review["production_actual_replay_maximum_prediction_error"] <= 1e-6
        and math.isfinite(review["maximum_absolute_arithmetic_error"])
        and 0 <= review["maximum_absolute_arithmetic_error"] <= 2e-10,
        "S5_B_REVIEW_METHOD_OR_PRODUCER_BINDING_CHANGED")
    script = B_REVIEWER
    checked_file(root, script, review["reviewer_script_sha256"])
    report = read(root / B_REPORT)
    require(report["schema"] == "s5_hest_shared_feature_results_v2"
        and report["status"] == "COMPLETE_TWENTY_DEVELOPMENT_AND_TWENTY_FINAL_FITS_AND_120_REPLAYS"
        and report["actual_development_fits"] == report["actual_final_fits"] == 20
        and report["actual_saved_state_replays"] == 120 and len(report["cells"]) == 40
        and report["source_files"] == review["source_files"], "S5_B_PRODUCER_NOT_COMPLETE")
    boundary = report["boundaries"]
    require(report["new_PCA_fits"] == report["new_scaler_fits"] == 8
        and boundary["official_test_not_used_for_selection"] is True
        and boundary["own_fold_test_excluded_from_training_and_selection"] is True
        and boundary["optimizer_convergence_proved"] is False
        and boundary["BLEEP_or_TRIPLEX_reproduced"] is False
        and boundary["new_independent_cohort_claimed"] is False
        and boundary["tissue_truth_or_teacher_used"] is False
        and boundary["extra_gene_membership_used"] is False
        and boundary["DSS_PFI_or_R8_read"] is False,
        "S5_B_EVIDENCE_OR_INFORMATION_BUDGET_CHANGED")
    return script


def verify_completion(root, a, b, history):
    sources(root, a["source_files"], A_SHA, 13)
    require(a["schema"] == "spgel_s5a_anonymous_completion_evidence_v1"
        and a["status"] == "PASS_84_FITS_72800_UPDATES_252_ACTUAL_REPLAYS_AND_EIGHT_PUBLIC_OUTPUTS"
        and a["source_canonical_sha256"] == A_SHA and a["actual_neural_fits"] == 84
        and a["actual_optimizer_updates"] == 72800 and a["actual_saved_state_replays"] == 252
        and a["new_fits_by_publication"] == a["model_or_array_deserializations"] == 0
        and a["report_reconstructed_byte_exact"] is True and len(a["public_outputs"]) == 8,
        "S5_A_ACTUAL_COMPLETE_METADATA_REQUIRED")
    checked_file(root, safe_name(a["producer_script"]), a["producer_script_sha256"])
    reviewer = verify_b_review(root, b, history)
    required = set(a["source_files"]) | {n for n in b["source_files"] if n not in history}
    required |= ADMIN | QUAL_FILES | {COST, B_REPORT, B_REVIEW, a["producer_script"], reviewer,
        "tests/test_spgel_s5_report.py", "tests/test_spgel_s5b1_independent_review.py",
        "reports/spgel_s5/S5A_IMPLEMENTATION_FIDELITY.md",
        "reports/spgel_s5/S5_EXECUTION_PROTOCOL.md",
        "reports/spgel_s5/S5B1_PROTOCOL.md", "reports/spgel_s5/S5B1_CANDIDATE_ITC_REPAIR.md"}
    for evidence in (a, b):
        for name, digest in evidence["public_outputs"].items():
            checked_file(root, safe_name(name), digest)
            required.add(name)
    require(A_REPORT in a["public_outputs"], "S5_PUBLIC_PRODUCER_REPORT_BINDINGS_MISSING")
    require(read(root / QUAL)["status"] == "QUALIFIED_SOURCE_REVIEW_NOT_REPRODUCED",
            "S5_AUTHOR_SOTA_REPRODUCTION_NOT_COMPLETED")
    superseded_cost(root)
    return required


def publish(root, history_root, allowlist, denylist, a_path, b_path):
    rows = inventory(root, allowlist)
    require(sha(Path(prior.inherited.__file__)) == SCANNER_SHA, "S5_INHERITED_SCANNER_CHANGED")
    history = historical_inventory(history_root)
    a, b = read(a_path), read(b_path)
    checked_file(root, B_REVIEW, sha(b_path))
    required = verify_completion(root, a, b, history)
    names = {r["file"] for r in rows}
    require(not names.intersection(history) and required <= names,
            "S5_REQUIRED_NEW_SCIENTIFIC_DELIVERY_MISSING_OR_OLD_PAYLOAD_INCLUDED")
    scanned, pattern = prior.checked_scan(root, rows, denylist)
    audit = dict(schema="spgel_s5_publication_source_audit_v1",
        status="SCIENTIFIC_COMPLETE_S5A_ATTRIBUTION_AND_S5B1_SHARED_FEATURE_BENCHMARK",
        base_git_commit=BASE_COMMIT, base_git_tree=BASE_TREE,
        inherited_scanner_sha256=SCANNER_SHA, denylist_sha256=prior.DENY_SHA,
        historical_files_byte_unchanged=67, historical_manifest_sha256=HISTORY,
        source_attribution_evidence=a, shared_feature_benchmark_evidence=b,
        SOTA_author_implementation_reproductions="NOT_COMPLETED_QUALIFICATION_ONLY",
        independent_external_source_confirmation=False,
        superseded_pretest_cost_sha256=sha(root/COST),
        public_input_files=len(rows), public_input_bytes=sum(r["bytes"] for r in rows),
        new_fits=0, model_or_array_deserializations=0, git_operations=0)
    encoded = lambda value: (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    raw = encoded(audit)
    audit_row = dict(file=AUDIT, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest = dict(schema="spgel_s5_publication_manifest_v1",
        status=audit["status"], base_git_commit=BASE_COMMIT, base_git_tree=BASE_TREE,
        source_audit_sha256=audit_row["sha256"], files=sorted(scanned+[audit_row],key=lambda r:r["file"]),
        file_count_including_audit_excluding_manifest=len(rows)+1,
        manifest_does_not_hash_itself=True, git_operations=0)
    for value in (audit, manifest):
        prior.inherited.validate_public_json(value)
        text = encoded(value).decode()
        require(len(encoded(value)) <= 2*1024**2 and pattern.search(text) is None
            and not any(p.search(text) for p in prior.inherited.PATTERNS.values()),
            "S5_GENERATED_PUBLIC_OUTPUT_PRIVACY_OR_SIZE_FAILURE")
    require(not (root/AUDIT).exists() and not (root/MANIFEST).exists(),
            "S5_EXISTING_PUBLICATION_RESULTS_PRESERVED")
    with (root/AUDIT).open("xb") as stream:
        stream.write(raw)
    with (root/MANIFEST).open("xb") as stream:
        stream.write(encoded(manifest))
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices=("local", "evidence-a", "publish"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--historical-root", type=Path)
    for arg in ("allowlist", "denylist", "a-evidence", "b-review", "private", "config", "assets", "output"):
        parser.add_argument("--"+arg, type=Path)
    args = parser.parse_args()
    if args.mode == "local":
        result = local_scan(args.root, inventory(args.root, args.allowlist))
    elif args.mode == "evidence-a":
        result = a_evidence(args.root,args.private,
            args.config or args.root/"configs/spgel_s5/mechanism_v1.json",args.assets)
    else:
        result = publish(args.root,args.historical_root or args.root,args.allowlist,
            args.denylist,args.a_evidence,args.b_review)
    if args.output:
        with args.output.open("x",encoding="utf-8",newline="\n") as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({k:result[k] for k in ("schema","status")},ensure_ascii=False))


if __name__ == "__main__":
    main()

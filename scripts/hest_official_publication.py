"""Explicit metadata-only HEST publication guards; zero Git/model operations.

The local evidence step reads private JSON metadata only. Exact identifier
scanning stays beside the established private denylist; no identifier list,
patient array, image feature, prediction or model is transported by this tool.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_publication as inherited

require, sha = inherited.require, inherited.sha256
BASE_COMMIT = "8477be619105ba79827bb3c309c88249023f6a84"
BASE_TREE = "de8c924b1fdb5584d22ecbfaa4796e7862f039e9"
DATA_REVISION = "b039b5965ed7092f280d4f336da7972ef9a10e1a"
DENY_SHA = "d73b70939a38277f8f9f693e9958afcc9435893cd05f5c1d6a6d08a100ffe867"
RIDGE_SOURCE_SHA = "0c33b463e556ea22d461bc7e2be7747dfa1d528ac815e62dbdc5534bc8b2ec3e"
FEATURE_CODE_SHA = "7fb5571cfa80a8ec1a345cabbb35f23b60d8b650d461e4e3b912e3542a0e7938"
FEATURE_CONFIG_SHA = "a6dac53b274163f58d2f4b08613f55cb8131ddbfa2803660909f429ef7a5bc7e"
GEO_ORIGINAL_SHA = "a13af54caa72e3107b618faa32ac80d1f002a3f1b8046c332e472e81281f5232"
REPORT = "reports/spgel/HEST_OFFICIAL_RIDGE_RESULTS_20261029.json"
REVIEW = "reports/spgel/HEST_OFFICIAL_RIDGE_INDEPENDENT_REVIEW_20261029.json"
REVIEW_SCRIPT = "scripts/review_hest_official_ridge_delivery.py"
AUDIT = "reports/spgel/HEST_OFFICIAL_PUBLICATION_SOURCE_AUDIT_20261029.json"
MANIFEST = "reports/spgel/HEST_OFFICIAL_PUBLICATION_MANIFEST_20261029.json"
LEDGER = "reports/spgel/HEST_OFFICIAL_RIDGE_EXECUTION_LEDGER_20261029.json"
FOLD_CSV = "reports/spgel/HEST_OFFICIAL_RIDGE_FOLD_METRICS_20261029.csv"
GENE_CSV = "reports/spgel/HEST_OFFICIAL_RIDGE_GENE_METRICS_20261029.csv"
ADMIN = {"scripts/hest_official_publication.py", "tests/test_hest_official_publication.py"}
PROJECTION_MANIFEST = "reports/spgel/HEST_FEATURE_PUBLIC_PROJECTIONS_20261029.json"
PROJECTION_RESTORE = "reports/spgel/HEST_FEATURE_PUBLIC_PROJECTION_RESTORE_AUDIT_20261029.json"
MATERIALIZER = "scripts/materialize_hest8_public_projections.py"
MATERIALIZER_TEST = "tests/test_hest_public_projections.py"


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            require(key not in value, "DUPLICATE_METADATA_KEY")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_METADATA")))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def safe_name(name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name,
            "UNSAFE_PUBLICATION_PATH")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == name,
            "UNSAFE_PUBLICATION_PATH")
    code = (re.fullmatch(r"src/spgel_hest/[A-Za-z0-9_]+\.(?:py|md)", name)
        or re.fullmatch(r"tests/test_(?:spgel_hest_[A-Za-z0-9_]+|hest_official_publication)\.py", name)
        or re.fullmatch(r"configs/spgel_hest/[A-Za-z0-9_]+\.json", name))
    extra = name in ADMIN | {REVIEW_SCRIPT, MATERIALIZER, MATERIALIZER_TEST, "scripts/spgel_hest_ridge.py",
        "scripts/extract_hest8_official_uni2_bf16.py", "configs/spgel/official8_uni2_bf16_v1.json"}
    report = re.fullmatch(r"reports/spgel/(?:HEST_|INDEPENDENT_CRC_)[A-Za-z0-9_]+\.(?:md|json|csv)", name)
    require(bool(code or extra or report), "UNKNOWN_OR_HISTORICAL_PUBLICATION_FILE")
    require(name not in {AUDIT, MANIFEST}, "GENERATED_FILES_MUST_NOT_ENTER_INPUT_ALLOWLIST")
    return name


def inventory(root, path):
    record = read(path)
    rows = record.get("files", [])
    require(isinstance(rows, list) and rows and len(rows) == len({r["file"] for r in rows}),
            "EXPLICIT_UNIQUE_FILE_ALLOWLIST_REQUIRED")
    for row in rows:
        name = safe_name(row["file"]);p = root / name
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
            and p.stat().st_size == row["bytes"] and sha(p) == row["sha256"],
            "PUBLICATION_FILE_SHA_OR_SIZE_CHANGED:" + name)
    return rows


def restore_geo_projection(value):
    original = copy.deepcopy(value);lineage = original.pop("public_projection_lineage")
    require(lineage["schema"] == "lossless_aggregate_json_key_projection_v1"
        and lineage["original_raw_sha256"] == GEO_ORIGINAL_SHA
        and lineage["values_or_counts_removed"] is False
        and lineage["arrays_or_identifiers_added"] is False
        and lineage["rename"] == {"parent": ["new_studies", "GSE267401", "official_attachment_groups"],
                                   "from": "barcodes", "to": "barcode_attachment_summary"},
        "GEO_PROJECTION_LINEAGE_CHANGED")
    group = original["new_studies"]["GSE267401"]["official_attachment_groups"]
    require("barcode_attachment_summary" in group and "barcodes" not in group,
            "GEO_PROJECTION_KEY_MAPPING_CHANGED")
    original["new_studies"]["GSE267401"]["official_attachment_groups"] = {
        ("barcodes" if k == "barcode_attachment_summary" else k): v for k, v in group.items()}
    raw = (json.dumps(original, ensure_ascii=False, indent=2) + "\n").replace("\n", "\r\n").encode()
    require(len(raw) == lineage["original_raw_bytes"]
        and hashlib.sha256(raw).hexdigest() == GEO_ORIGINAL_SHA
        and canonical(original) == lineage["original_canonical_semantic_sha256"],
        "GEO_PROJECTION_LOSS_OR_ORIGINAL_BYTES_CHANGED")
    return dict(original_raw_sha256=GEO_ORIGINAL_SHA, byte_exact_reconstruction=True)


def verify_feature_public_projections(root, names, bundle):
    require({PROJECTION_MANIFEST,PROJECTION_RESTORE,MATERIALIZER,MATERIALIZER_TEST} <= names,
            "FEATURE_PUBLIC_PROJECTION_RESTORE_DELIVERY_REQUIRED")
    manifest, audit = read(root/PROJECTION_MANIFEST), read(root/PROJECTION_RESTORE)
    require(manifest["schema"] == "hest_lossless_public_artifact_projections_v1"
        and manifest["dataset_revision"] == DATA_REVISION
        and manifest["original_artifacts_modified"] is False
        and manifest["projection_execution_supported"] is False
        and len(manifest["aliases"]) == len({r["alias"] for r in manifest["aliases"]}) == 8,
        "FEATURE_PROJECTION_CONTRACT_CHANGED")
    require(audit["schema"] == "hest_public_projection_actual_restore_audit_v1"
        and audit["status"] == "PASS_THREE_ACTUAL_IMMUTABLE_PUBLIC_METADATA_RESTORES_AND_THREE_NEGATIVE_GUARD_TESTS"
        and audit["public_directory_requests_in_successful_restore"] == 2
        and audit["tests_passed"] == 3 and audit["raw_image_expression_weight_downloads"] == 0
        and audit["original_files_modified"] is False and audit["scientific_results_changed"] is False
        and audit["models_loaded"] == audit["new_fits"] == 0,
        "ACTUAL_FEATURE_PROJECTION_RESTORE_AUDIT_REQUIRED")
    expected={"scripts/extract_hest8_official_uni2_bf16.py":FEATURE_CODE_SHA,
        "configs/spgel/official8_uni2_bf16_v1.json":FEATURE_CONFIG_SHA,
        "reports/spgel/HEST_OFFICIAL8_BF16_FEATURE_QUALIFICATION_WITH_GEOMETRY_V2_20261029.json":bundle["feature_qualification_original_sha256"]}
    rows={r["original_file"]:r for r in manifest["artifacts"]}
    restored={r["original_file"]:r for r in audit["artifacts"]}
    require(len(rows)==len(restored)==3 and set(rows)==set(restored)==set(expected),
            "EXACT_THREE_FROZEN_ORIGINAL_PROJECTIONS_REQUIRED")
    for name,digest in expected.items():
        row,actual=rows[name],restored[name];path=root/row["projection_file"]
        require(row["projection_file"] in names
            and row["original_sha256"] == actual["original_sha256"] == actual["actual_restored_sha256"] == digest
            and row["projection_sha256"] == actual["projection_sha256"] == sha(path)
            and path.stat().st_size == row["projection_bytes"] and actual["byte_exact_equal"] is True,
            "FEATURE_PROJECTION_OR_ACTUAL_RESTORE_BYTES_CHANGED")
    return dict(original_artifacts=3,actual_byte_exact_restores=3,
        metadata_only_public_directory_requests=2,projection_is_not_executable_science=True,
        projection_manifest_sha256=sha(root/PROJECTION_MANIFEST),
        actual_restore_audit_sha256=sha(root/PROJECTION_RESTORE),materializer_sha256=sha(root/MATERIALIZER),
        materializer_test_sha256=sha(root/MATERIALIZER_TEST),original_artifact_sha256=expected)


def local_evidence(root, private, freeze_path, feature_path):
    """No H5/NPZ/joblib access: private JSON and prior independent review only."""
    freeze = read(freeze_path)
    require(freeze["sources_canonical_sha256"] == canonical(freeze["sources"]) == RIDGE_SOURCE_SHA
        and len(freeze["sources"]) == 11, "ELEVEN_FROZEN_RIDGE_SOURCES_REQUIRED")
    for name, expected in freeze["sources"].items():
        require(sha(root / name) == expected, "FROZEN_RIDGE_SOURCE_CHANGED:" + name)
    lock_path, replay_path = private / "HEST_RUN_LOCK.json", private / "REPLAY_AUDIT.json"
    lock, replay = read(lock_path), read(replay_path)
    report, review, feature, ledger = read(root / REPORT), read(root / REVIEW), read(feature_path), read(root / LEDGER)
    require(lock["schema"] == "spgel_hest_ridge_run_lock_v1"
        and lock["internal_contract_sha256"] == canonical({k:v for k,v in lock.items() if k != "internal_contract_sha256"})
        and lock["dataset_revision"] == DATA_REVISION
        and lock["status"] == "LOCKED_BEFORE_FOUR_OFFICIAL_RIDGE_FITS"
        and lock["source_files"] == freeze["sources"]
        and lock["official_Ridge_fits"] == 4 and lock["neural_fits"] == 0,
        "ACTUAL_FOUR_RIDGE_LOCK_REQUIRED")
    contract = lock["internal_contract_sha256"]
    require(report["status"] == "COMPLETE_FOUR_OFFICIAL_RIDGE_FITS_AND_ACTUAL_REPLAYS"
        and report["actual_Ridge_fits"] == report["actual_replayed_models"] == 4
        and report["source_run_lock_file_sha256"] == sha(lock_path)
        and report["internal_contract_sha256"] == replay["internal_contract_sha256"] == contract
        and replay["schema"] == "spgel_hest_official_replay_v1"
        and replay["status"] == "PASS_FOUR_ACTUAL_OFFICIAL_RIDGE_MODEL_REPLAYS"
        and replay["passed_models"] == len(replay["units"]) == 4 and replay["failed_models"] == 0,
        "FOUR_COMPLETE_FITS_AND_ACTUAL_REPLAYS_REQUIRED")
    require(review["schema"] == "hest_four_ridge_independent_arithmetic_v1"
        and review["status"] == "PASS_FOUR_OFFICIAL_CELLS_GENE_AND_ROUND4_AGGREGATION_AND_ARTIFACT_BINDINGS"
        and review["internal_contract_sha256"] == contract
        and review["script_sha256"] == sha(root / REVIEW_SCRIPT)
        and review["reviewer_model_loads"] == review["reviewer_new_fits"] == 0
        and review["actual_Ridge_fits_in_research"] == review["actual_model_replays_in_research"] == 4
        and review["fold_rows"] == 4 and review["gene_rows"] == 200
        and 0 <= review["maximum_absolute_arithmetic_error"] <= 2e-10,
        "INDEPENDENT_ROOT_REVIEW_REQUIRED")
    evidence = review["evidence_sha256"]
    require(evidence["public_report"] == sha(root / REPORT)
        and evidence["run_lock"] == sha(lock_path)
        and evidence["actual_model_replay"] == sha(replay_path), "ROOT_REVIEW_BINDING_CHANGED")
    pointer = read(private/"HEST_ASSET_POINTER.private.json")
    require(sha(Path(pointer["path"])) == pointer["sha256"] == lock["assets_sha256"],
            "CURRENT_ASSET_POINTER_OR_LOCK_BINDING_CHANGED")
    runtime_assets = read(pointer["path"])
    require(runtime_assets["feature_qualification"]["sha256"] == sha(feature_path),
            "LOCKED_ASSETS_DO_NOT_BIND_AUTHORITATIVE_EIGHT_FEATURE_QUALIFICATION")
    cells = {}
    expected = {(d, f, 1) for d in ("COAD", "READ") for f in (0, 1)}
    require({(r["dataset"],r["fold"],r["seed"]) for r in replay["units"]} == expected,
            "ACTUAL_FOUR_REPLAY_IDENTITIES_CHANGED")
    for dataset, fold, seed in expected:
        p = private / "cells" / f"{dataset}_fold{fold}_seed{seed}" / "COMPLETE.json"
        cell = read(p);row = next(r for r in replay["units"] if (r["dataset"],r["fold"],r["seed"]) == (dataset,fold,seed))
        require(cell["schema"] == "spgel_hest_official_ridge_cell_v1"
            and cell["status"] == "COMPLETE_ONE_OFFICIAL_RIDGE_FIT"
            and cell["internal_contract_sha256"] == contract
            and cell["new_Ridge_fits"] == cell["new_scaler_fits"] == cell["new_PCA_fits"] == 1
            and cell["neural_fits"] == 0 and cell["test_RNA_in_predict"] is False
            and cell["source_pretrained_weights_loaded"] is False
            and row["actual_saved_model_loaded"] is row["actual_forward_replayed"] is True
            and row["status"] == "PASS" and 0 <= row["maximum_prediction_error"] <= 1e-6
            and row["new_Ridge_fits"] == row["new_scaler_fits"] == row["new_PCA_fits"] == 0
            and row["COMPLETE_sha256"] == evidence[f"{dataset}_fold{fold}_COMPLETE"] == sha(p)
            and row["model_sha256"] == cell["model_sha256"], "ACTUAL_CELL_OR_REPLAY_METADATA_CHANGED")
        cells[(dataset,fold)] = cell
    require(feature["schema"] == "hest_official8_bf16_feature_qualification_v1"
        and feature["status"] == "PASS_ALL_EIGHT_OFFICIAL_BF16_FEATURES_RETURNED_E_RELOADED"
        and feature["dataset_revision"] == DATA_REVISION and feature["sample_count"] == 8
        and len(feature["samples"]) == 8 and feature["total_spots"] == 24058
        and feature["config_sha256"] == FEATURE_CONFIG_SHA and feature["extractor_sha256"] == FEATURE_CODE_SHA
        and feature["new_optimizer_fits"] == feature["new_optimizer_updates"] == 0,
        "EIGHT_ACTUAL_NEW_FEATURES_REQUIRED")
    require(sum(r["spots_count"] for r in feature["samples"]) == 24058
        and all(r["E_reloaded_barcode_coords_and_finite_raw_features"] is True
                and r["forward_precision"] == "bfloat16" and r["row_L2_normalized"] is False
                and r["feature_dimension"] == 1536
                and r["config_sha256"] == FEATURE_CONFIG_SHA and r["extractor_sha256"] == FEATURE_CODE_SHA
                and len(r["feature_sha256"]) == len(r["receipt_sha256"]) == 64 for r in feature["samples"]),
        "ACTUAL_FEATURE_RECEIPTS_OR_RELOAD_SCOPE_CHANGED")
    qualified = {(r["task"],r["official_sample_id"]):r for r in feature["samples"]}
    require(len(qualified) == 8 and set(qualified) == {(d,s) for d,t in runtime_assets["tasks"].items() for s in t["samples"]},
            "ACTUAL_EIGHT_FEATURE_IDENTITIES_CHANGED")
    for dataset, task in runtime_assets["tasks"].items():
        for specimen, spec in task["samples"].items():
            entry = qualified[(dataset,specimen)]
            require(entry["feature_sha256"] == spec["features"]["sha256"]
                and entry["source_patch_sha256"] == spec["patches"]["sha256"],
                "RUNTIME_FEATURES_DIFFER_FROM_ACTUAL_EIGHT_QUALIFICATION")
    require(ledger["schema"] == "hest_official_ridge_execution_ledger_v1"
        and ledger["status"] == "COMPLETE_FIXED_FOUR_RIDGE_FITS_AND_FOUR_ACTUAL_REPLAYS"
        and ledger["actual_Ridge_fits"] == ledger["actual_scaler_fits"] == ledger["actual_PCA_fits"]
            == ledger["actual_replayed_models"] == 4
        and ledger["neural_fits"] == ledger["new_fits_during_replay"] == ledger["automatic_retry_fits"] == 0,
        "ACTUAL_EXECUTION_LEDGER_COUNTS_OR_RETRY_SCOPE_CHANGED")
    require(ledger["lock_artifact"]["file_sha256"] == sha(lock_path)
        and ledger["lock_artifact"]["internal_contract_sha256"] == contract
        and ledger["runtime_assets_artifact"]["sha256"] == pointer["sha256"]
        and ledger["actual_replay_artifact"]["sha256"] == sha(replay_path)
        and ledger["report_artifact"]["sha256"] == sha(root/REPORT)
        and ledger["authoritative_feature_qualification"]["sha256"] == sha(feature_path)
        and ledger["data_qualification"]["sha256"] == runtime_assets["qualification"]["sha256"]
        and ledger["code_sources_canonical_sha256"] == RIDGE_SOURCE_SHA,
        "EXECUTION_LEDGER_ARTIFACT_SHA_BINDING_CHANGED")
    require(len(ledger["cells"]) == 4
        and {(r["dataset"],r["fold"],r["seed"]) for r in ledger["cells"]} == expected,
        "EXECUTION_LEDGER_CELL_SCOPE_CHANGED")
    for row in ledger["cells"]:
        cell=cells[(row["dataset"],row["fold"])];key=f"{row['dataset']}_fold{row['fold']}_COMPLETE"
        require(row["COMPLETE_sha256"] == evidence[key] and row["model_sha256"] == cell["model_sha256"]
            and row["predictions_sha256"] == cell["predictions_sha256"]
            and row["train_rows"] == cell["train_rows"] and row["test_rows"] == cell["test_rows"],
            "EXECUTION_LEDGER_CELL_ARTIFACTS_CHANGED")
    with (root/FOLD_CSV).open(encoding="utf-8-sig",newline="") as f: folds=list(csv.DictReader(f))
    with (root/GENE_CSV).open(encoding="utf-8-sig",newline="") as f: genes=list(csv.DictReader(f))
    require(len(folds) == 4 and len(genes) == 200, "ACTUAL_FOUR_FOLD_AND_TWO_HUNDRED_GENE_ROWS_REQUIRED")
    def number(text): return None if text == "" else float(text)
    for row in folds:
        key=(row["dataset"],int(row["fold"]));cell=cells[key]
        require(int(row["seed"]) == 1 and int(row["train_spots"]) == cell["train_rows"]
            and int(row["test_spots"]) == cell["test_rows"]
            and number(row["Pearson_mean_full_precision"]) == cell["full_precision_metrics"]["pearson_mean"]
            and number(row["Pearson_mean_official_round4"]) == cell["official_rounded4_split_metrics"]["pearson_mean"],
            "PUBLIC_FOLD_CSV_CHANGED_FROM_COMPLETION")
    require(len({(r["dataset"],int(r["fold"])) for r in folds}) == 4
        and len({(r["dataset"],int(r["fold"]),r["gene"]) for r in genes}) == 200,
        "PUBLIC_CSV_DUPLICATE_OR_MISSING_UNIT")
    for row in genes:
        cell=cells[(row["dataset"],int(row["fold"]))];i=cell["genes"].index(row["gene"])
        require(int(row["seed"]) == 1
            and number(row["Pearson_full_precision"]) == cell["full_precision_metrics"]["pearson_corrs"][i]["pearson_corr"]
            and number(row["Pearson_official_round4"]) == cell["official_rounded4_split_metrics"]["pearson_corrs"][i]["pearson_corr"],
            "PUBLIC_GENE_CSV_CHANGED_FROM_COMPLETION")
    return dict(schema="hest_official_local_metadata_evidence_v1",
        status="PASS_LOCAL_PRIVATE_JSON_AND_INDEPENDENT_ROOT_REVIEW_BINDINGS",
        dataset_revision=DATA_REVISION, source_canonical_sha256=RIDGE_SOURCE_SHA,
        sources=freeze["sources"], actual_features=8, actual_feature_patches=24058,
        actual_Ridge_fits=4, actual_model_replays=4, source_run_lock_sha256=sha(lock_path),
        internal_contract_sha256=contract, actual_replay_sha256=sha(replay_path),
        feature_qualification_original_sha256=sha(feature_path),
        independent_root_review_sha256=sha(root / REVIEW), public_report_sha256=sha(root / REPORT),
        independent_review_script_sha256=sha(root / REVIEW_SCRIPT), evidence_sha256=evidence,
        public_execution_ledger_sha256=sha(root/LEDGER), public_fold_CSV_sha256=sha(root/FOLD_CSV),
        public_gene_CSV_sha256=sha(root/GENE_CSV),
        new_fits_by_publication=0, model_or_prediction_arrays_read_by_publication=False,
        feature_or_label_arrays_transferred=False)


def checked_scan(root, rows, denylist_path):
    require(sha(denylist_path) == DENY_SHA, "ESTABLISHED_DENYLIST_BYTES_CHANGED")
    pattern = inherited.identifier_pattern(inherited.validate_denylist(read(denylist_path)))
    previous = inherited.ALLOWLIST;inherited.ALLOWLIST = frozenset(r["file"] for r in rows)
    try:
        scanned = [inherited.scan_file(root, r["file"], pattern) for r in rows]
    finally:
        inherited.ALLOWLIST = previous
    require(sorted(scanned,key=lambda x:x["file"]) == sorted(rows,key=lambda x:x["file"]),
            "SCANNED_BYTES_DIFFER_FROM_EXPLICIT_ALLOWLIST")
    return scanned, pattern


def publish_manifest(root, allowlist_path, denylist_path, bundle_path):
    rows = inventory(root, allowlist_path);scanned, pattern = checked_scan(root, rows, denylist_path)
    bundle = read(bundle_path);names = {r["file"] for r in rows}
    require(bundle["schema"] == "hest_official_local_metadata_evidence_v1"
        and bundle["status"] == "PASS_LOCAL_PRIVATE_JSON_AND_INDEPENDENT_ROOT_REVIEW_BINDINGS"
        and bundle["dataset_revision"] == DATA_REVISION and bundle["source_canonical_sha256"] == RIDGE_SOURCE_SHA
        and bundle["actual_features"] == 8 and bundle["actual_feature_patches"] == 24058
        and bundle["actual_Ridge_fits"] == bundle["actual_model_replays"] == 4
        and bundle["new_fits_by_publication"] == 0
        and bundle["model_or_prediction_arrays_read_by_publication"] is False,
        "COMPLETE_LOCAL_METADATA_EVIDENCE_REQUIRED")
    require({REPORT, REVIEW, REVIEW_SCRIPT, LEDGER, FOLD_CSV, GENE_CSV} | set(bundle["sources"]) | ADMIN <= names,
            "REQUIRED_SCIENTIFIC_AND_REVIEW_SOURCES_MISSING")
    require(sha(root / REPORT) == bundle["public_report_sha256"]
        and sha(root / REVIEW) == bundle["independent_root_review_sha256"]
        and sha(root / REVIEW_SCRIPT) == bundle["independent_review_script_sha256"]
        and sha(root / LEDGER) == bundle["public_execution_ledger_sha256"]
        and sha(root / FOLD_CSV) == bundle["public_fold_CSV_sha256"]
        and sha(root / GENE_CSV) == bundle["public_gene_CSV_sha256"]
        and all(sha(root / k) == v for k,v in bundle["sources"].items()),
        "COMPLETE_EVIDENCE_SOURCE_OR_REPORT_SHA_CHANGED")
    projections = [restore_geo_projection(read(root/r["file"])) for r in rows
        if r["file"].endswith("INDEPENDENT_CRC_SOURCE_QUALIFICATION_PUBLIC_PROJECTION_20261029.json")]
    require(len(projections) == 1, "GEO_LOSSLESS_PUBLIC_PROJECTION_REQUIRED")
    feature_projections = verify_feature_public_projections(root,names,bundle)
    audit = dict(schema="hest_official_publication_source_audit_v1",
        status="PASS_COMPLETE_METADATA_EVIDENCE_AND_ESTABLISHED_EXACT_IDENTIFIER_SCAN",
        base_git_commit=BASE_COMMIT, base_git_tree=BASE_TREE, denylist_sha256=DENY_SHA,
        inherited_scanner_sha256=sha(Path(inherited.__file__)),
        publication_script_sha256=sha(root/"scripts/hest_official_publication.py"),
        completion_evidence=bundle, geo_projection=projections[0], feature_projections=feature_projections,
        input_files=len(rows), input_bytes=sum(r["bytes"] for r in rows),
        old_scanner_or_scientific_sources_modified=False, existing_reports_overwritten=False,
        git_operations=0, private_identifiers_exported=False)
    audit_bytes=(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    audit_row=dict(file=AUDIT,bytes=len(audit_bytes),sha256=hashlib.sha256(audit_bytes).hexdigest())
    manifest=dict(schema="hest_official_publication_manifest_v1",status="COMPLETE_ANONYMOUS_OFFICIAL_STAGE",
        base_git_commit=BASE_COMMIT,base_git_tree=BASE_TREE,source_audit_sha256=audit_row["sha256"],
        files=sorted(scanned+[audit_row],key=lambda x:x["file"]),
        file_count_including_audit_excluding_this_manifest=len(rows)+1,
        total_bytes_including_audit_excluding_this_manifest=sum(r["bytes"] for r in rows)+len(audit_bytes),
        manifest_does_not_hash_itself=True,git_operations=0)
    for obj in (audit,manifest):
        inherited.validate_public_json(obj);text=json.dumps(obj,ensure_ascii=False)
        require(not any(p.search(text) for p in inherited.PATTERNS.values()) and pattern.search(text) is None,
                "GENERATED_PUBLICATION_OUTPUT_PRIVACY_FAILURE")
    require(not (root/AUDIT).exists() and not (root/MANIFEST).exists(),
            "EXISTING_OFFICIAL_PUBLICATION_OUTPUT_MUST_BE_ARCHIVED_SEPARATELY")
    (root/AUDIT).write_bytes(audit_bytes)
    with (root/MANIFEST).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    return manifest


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("mode",choices=("evidence","publish"))
    p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--private",type=Path)
    p.add_argument("--freeze",type=Path);p.add_argument("--feature-qualification",type=Path)
    p.add_argument("--files",type=Path);p.add_argument("--denylist",type=Path);p.add_argument("--bundle",type=Path,required=True)
    a=p.parse_args()
    if a.mode=="evidence":
        result=local_evidence(a.root,a.private,a.freeze,a.feature_qualification)
        with a.bundle.open("x",encoding="utf-8",newline="\n") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    else:result=publish_manifest(a.root,a.files,a.denylist,a.bundle)
    print(json.dumps({k:result[k] for k in ("schema","status")}))


if __name__=="__main__":main()

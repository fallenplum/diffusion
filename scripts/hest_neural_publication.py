"""Metadata-only publication for the separately frozen twelve HEST fits.

This administrative tool neither reads model/prediction arrays nor performs Git
or fitting. Local preflight omits the unavailable private identifier registry;
the complete stage must use the unchanged established scanner beside that list.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import hest_official_publication as prior

read, canonical, require, sha = prior.read, prior.canonical, prior.require, prior.sha
ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "c129fa6676d8d70198224e399a7694ade90895ee6bdf9729507619adf1e070e0"
BASE_COMMIT = "562ad1cdd42498bc38eedf5329e1668d34c4dc84"
BASE_TREE = "84ee19466b84a53cb9398b57113e142ca6893506"
PREFIX = "reports/spgel/HEST_NEURAL_FIXED12_"
REPORT = PREFIX + "RESULTS_20261029.json"
REVIEW = PREFIX + "INDEPENDENT_AUDIT_20261029.json"
LEDGER = PREFIX + "EXECUTION_LEDGER_20261029.json"
FOLD_CSV = PREFIX + "FOLD_METRICS_20261029.csv"
GENE_CSV = PREFIX + "GENE_METRICS_20261029.csv"
REVIEW_SCRIPT = "scripts/verify_hest_neural_fixed12.py"
AUDIT = PREFIX + "PUBLICATION_SOURCE_AUDIT_20261029.json"
MANIFEST = PREFIX + "PUBLICATION_MANIFEST_20261029.json"
ADMIN = {"scripts/hest_neural_publication.py", "tests/test_hest_neural_publication.py"}
SUPPLEMENT = {"scripts/fetch_hest8_pinned_primary.py",
    "reports/spgel/HEST_BF16_DEVICE_PUBLIC_NUMERIC_AUDIT_20261029.json",
    "reports/spgel/HEST_BF16_REPRODUCTION_DELIVERY_SUPPLEMENT_20261029.md"}
ARMS = ("MLP", "SPATIAL_GNN", "IDENTITY_GNN")
EXPECTED = {(d, f, a) for d in ("COAD", "READ") for f in (0, 1) for a in ARMS}


def safe_name(name):
    require(isinstance(name, str) and "\\" not in name and ":" not in name,
            "UNSAFE_NN_PUBLICATION_PATH")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and str(p) == name,
            "UNSAFE_NN_PUBLICATION_PATH")
    code = (re.fullmatch(r"src/spgel_hest_neural/[A-Za-z0-9_]+\.py", name)
        or re.fullmatch(r"tests/test_spgel_hest_neural_[A-Za-z0-9_]+\.py", name)
        or name == "configs/spgel_hest_neural/pilot_v1.json")
    report = re.fullmatch(PREFIX + r"[A-Z0-9_]+_20261029\.(?:json|csv|md)", name)
    require(bool(code or report or name in ADMIN | SUPPLEMENT |
        {REVIEW_SCRIPT, "scripts/spgel_hest_neural.py"}), "UNKNOWN_OR_HISTORICAL_NN_FILE")
    require(name not in {AUDIT, MANIFEST}, "GENERATED_AUDIT_MUST_NOT_ENTER_INPUT_LIST")
    return name


def inventory(root, path):
    rows = read(path)["files"]
    require(rows and len(rows) == len({r["file"] for r in rows}), "EXPLICIT_UNIQUE_NN_LIST_REQUIRED")
    for row in rows:
        name = safe_name(row["file"]); p = root / name
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
            and p.stat().st_size == row["bytes"] and sha(p) == row["sha256"],
            "NN_PUBLIC_FILE_BYTES_CHANGED:" + name)
    return rows


def local_scan(root, rows):
    """Use old size, credential, path and no-array rules; no private list claim."""
    old = prior.inherited.ALLOWLIST
    prior.inherited.ALLOWLIST = frozenset(r["file"] for r in rows)
    try:
        result = [prior.inherited.scan_file(root, r["file"], re.compile(r"(?!x)x")) for r in rows]
    finally:
        prior.inherited.ALLOWLIST = old
    require(result == rows, "NN_LOCAL_SCAN_BYTES_CHANGED")
    return dict(schema="hest_neural_local_privacy_preflight_v1",
        status="PASS_EXISTING_PATTERN_AND_NO_ARRAY_RULES_ONLY",
        files=result, file_count=len(result), total_bytes=sum(r["bytes"] for r in result),
        exact_private_identifier_scan="NOT_RUN_REQUIRES_SERVER_REGISTRY",
        raw_arrays_models_or_private_registry_transferred=False, new_fits=0, git_operations=0)


def metadata_evidence(root, private, freeze_path):
    """Only JSON receipts: actual checkpoint forwards belong to the replay owner."""
    freeze = read(freeze_path); sources = freeze["sources"]
    require(len(sources) == 11 and canonical(sources) == SOURCE_SHA
        and freeze["sources_canonical_sha256"] == SOURCE_SHA,
        "EXACT_ELEVEN_FROZEN_NN_FILES_REQUIRED")
    require(all(sha(root / n) == h for n, h in sources.items()), "FROZEN_NN_SOURCE_CHANGED")
    lock_path, replay_path = private / "HEST_NEURAL_RUN_LOCK.json", private / "NEURAL_REPLAY_AUDIT.json"
    lock, replay, report = read(lock_path), read(replay_path), read(root / REPORT)
    contract = lock["internal_contract_sha256"]
    require(lock["schema"] == "spgel_hest_neural_run_lock_v1"
        and lock["status"] == "LOCKED_BEFORE_TWELVE_NEURAL_FITS"
        and contract == canonical({k:v for k,v in lock.items() if k != "internal_contract_sha256"})
        and lock["source_files"] == sources and report["code_sha256"] == sources
        and lock["config_sha256"] == report["config_sha256"] == sha(root/"configs/spgel_hest_neural/pilot_v1.json")
        and lock["assets_sha256"] == report["assets_sha256"]
        and lock["actual_preparation_new_PCA_fits"] == 4
        and lock["actual_preparation_new_scaler_fits"] == lock["actual_preparation_new_Ridge_fits"]
            == lock["actual_preparation_new_neural_fits"] == 0,
        "ACTUAL_NN_LOCK_OR_PREPARATION_BINDING_CHANGED")
    require(len(lock["inherited_Ridge_source_files"]) == 11
        and canonical(lock["inherited_Ridge_source_files"]) == prior.RIDGE_SOURCE_SHA
        and all(sha(root/n) == h for n,h in lock["inherited_Ridge_source_files"].items()),
        "INHERITED_ELEVEN_RIDGE_FILES_CHANGED")
    require(report["schema"] == "spgel_hest_neural_fixed12_report_v1"
        and report["status"] == "COMPLETE_12_FIXED40_FITS_AND_36_ACTUAL_REPLAYS"
        and report["internal_contract_sha256"] == replay["internal_contract_sha256"] == contract
        and replay["schema"] == "spgel_hest_neural_replay_v1"
        and replay["status"] == "PASS_36_ACTUAL_SAVED_STATE_REPLAYS"
        and replay["passed_states"] == len(replay["units"]) == 36 and replay["passed_fits"] == 12
        and replay["new_neural_fits"] == replay["new_Ridge_fits"] == replay["new_PCA_fits"]
            == replay["optimizer_updates"] == 0 and replay["model_or_epoch_selection_performed"] is False
        and report["replay_audit_sha256"] == sha(replay_path), "COMPLETE_12_FITS_36_ACTUAL_REPLAYS_REQUIRED")
    require(report["ledger"] == dict(new_neural_fits=12, actual_optimizer_updates=480,
        actual_saved_state_replays=36, new_unsupervised_PCA_reconstructions=4,
        new_scaler_fits=0, new_Ridge_fits=0, old_source_weight_loads=0, automatic_retry_fits=0),
        "FIXED_NN_BUDGET_OR_RETRY_COUNTS_CHANGED")
    scope = report["scope"]
    require(scope["primary_epoch"] == 40 and scope["snapshot10_20_40_diagnostic_only"] is True
        and scope["fixed40_budget_proves_convergence"] is False
        and scope["test_labels_in_predict_or_epoch_selection"] is False
        and scope["new_independent_cohort_claimed"] is False
        and scope["READ_donors_previously_used_in_source_development"] is True
        and scope["COAD_preview_donor_identity"] == "UNCONFIRMED"
        and scope["graph_versus_MLP_difference_exclusively_attributed_to_topology"] is False,
        "UNSUPPORTED_SELECTION_CONVERGENCE_OR_INDEPENDENCE_CLAIM")
    parts = {(r["dataset"],r["fold"]):r for r in lock["partitions"]}
    require(len(parts) == len(lock["partitions"]) == 4 and set(parts) == {(d,f) for d,f,a in EXPECTED},
        "EXACT_FOUR_SHARED_INPUT_PARTITIONS_REQUIRED")
    public = {(r["dataset"],r["fold"],r["arm"]):r for r in report["per_cell"]}
    actual = {(r["dataset"],r["fold"],r["arm"],r["epoch"]):r for r in replay["units"]}
    require(len(public) == len(report["per_cell"]) == 12 and set(public) == EXPECTED
        and len(actual) == 36 and set(actual) == {(*k,e) for k in EXPECTED for e in (10,20,40)},
        "MISSING_OR_DUPLICATE_NN_CELL_OR_REPLAY")
    cells = {}; max_error = 0.
    for d,f,a in sorted(EXPECTED):
        cp = private/"cells"/f"{d}_fold{f}_seed1_{a}"/"COMPLETE.json"
        done = read(cp); row = public[(d,f,a)]; part = parts[(d,f)]
        require(done["schema"] == "spgel_hest_neural_cell_v1"
            and done["status"] == "COMPLETE_ONE_FIXED40_NEURAL_FIT"
            and (done["dataset"],done["fold"],done["arm"],done["seed"]) == (d,f,a,1)
            and done["internal_contract_sha256"] == contract
            and done["actual_epochs"] == done["actual_optimizer_updates"] == 40
            and done["new_neural_fits"] == 1
            and done["new_Ridge_fits"] == done["new_scaler_fits"] == done["new_PCA_fits"] == 0
            and done["input_cache_sha256"] == part["input_cache_sha256"]
            and done["train_loss_only"] is True and done["test_RNA_in_predict"] is False
            and done["test_labels_select_epoch"] is False and done["source_pretrained_weights_loaded"] is False
            and len(done["learning_curve"]) == 40
            and [s["epoch"] for s in done["snapshots"]] == [10,20,40]
            and row == {k:done[k] for k in row}, "NN_COMPLETE_OR_PUBLIC_REPORT_DRIFT")
        for snapshot in done["snapshots"]:
            unit = actual[(d,f,a,snapshot["epoch"])]; error = unit["maximum_prediction_error"]
            require(unit["status"] == "PASS" and unit["seed"] == 1
                and unit["actual_saved_state_loaded"] is True
                and unit["actual_train_and_test_forward_replayed"] is True
                and unit["new_neural_fits"] == unit["new_Ridge_fits"] == unit["new_PCA_fits"] == 0
                and unit["COMPLETE_sha256"] == sha(cp)
                and unit["state_sha256"] == snapshot["state_sha256"]
                and unit["predictions_sha256"] == snapshot["predictions_sha256"]
                and unit["input_cache_sha256"] == part["input_cache_sha256"]
                and isinstance(error,(float,int)) and math.isfinite(error) and 0 <= error <= 1e-6,
                "ACTUAL_NN_REPLAY_FLAGS_OR_ARTIFACT_SHA_CHANGED")
            max_error = max(max_error,error)
        cells[f"{d}_fold{f}_seed1_{a}"] = sha(cp)
    require(report["maximum_actual_replay_error"] == max_error, "PUBLIC_REPLAY_MAX_ERROR_CHANGED")
    return dict(schema="hest_neural_local_completion_evidence_v1",
        status="PASS_12_COMPLETES_36_ACTUAL_REPLAY_JSON_BINDINGS",
        source_canonical_sha256=SOURCE_SHA, sources=sources,
        lock_sha256=sha(lock_path), internal_contract_sha256=contract,
        actual_replay_sha256=sha(replay_path), public_report_sha256=sha(root/REPORT),
        source_freeze_manifest_sha256=sha(freeze_path),
        complete_file_sha256=cells, actual_neural_fits=12, actual_optimizer_updates=480,
        actual_state_replays=36, actual_unsupervised_PCA_reconstructions=4,
        inherited_Ridge_files_unchanged=True, new_fits_by_publication=0,
        arrays_or_models_read_by_publication=False)


def public_proofs(root, evidence):
    review, ledger = read(root/REVIEW), read(root/LEDGER)
    require(review["schema"] == "hest_neural_fixed12_metadata_arithmetic_audit_v1"
        and review["status"] == "PASS_12_FITS_480_UPDATES_36_ACTUAL_REPLAYS_AND_METRIC_ARITHMETIC"
        and review["independent_review_script_sha256"] == sha(root/REVIEW_SCRIPT)
        and review["producer_report_sha256"] == evidence["public_report_sha256"]
        and review["neural_lock_file_sha256"] == evidence["lock_sha256"]
        and review["actual_replay_file_sha256"] == evidence["actual_replay_sha256"]
        and review["source_freeze_manifest_sha256"] == evidence["source_freeze_manifest_sha256"]
        and review["source_canonical_sha256"] == SOURCE_SHA
        and review["internal_contract_sha256"] == evidence["internal_contract_sha256"]
        and review["actual_counts"] == dict(neural_fits=12,optimizer_updates=480,
            saved_state_replays=36,new_unsupervised_PCA_reconstructions=4,new_Ridge_fits=0)
        and review["reviewer_operations"] == dict(model_fits=0,model_forwards=0,PCA_fits=0,
            model_or_array_artifact_deserializations=0,current_artifact_file_hashing=True,
            actual_replays_run_by_producer_not_reviewer=True), "ACTUAL_INDEPENDENT_NN_REVIEW_REQUIRED")
    require(ledger["schema"] == "hest_neural_actual_execution_ledger_v1"
        and ledger["status"] == "PASS_12_ACTUAL_FIXED40_FITS_480_UPDATES_36_ACTUAL_REPLAYS_4_PCA"
        and ledger["actual_neural_fits"] == 12 and ledger["actual_optimizer_updates"] == 480
        and ledger["actual_saved_state_replays"] == 36 and ledger["new_unsupervised_PCA_reconstructions"] == 4
        and ledger["new_scaler_fits"] == ledger["new_Ridge_fits"] == ledger["automatic_retry_fits"] == 0
        and ledger["source_canonical_sha256"] == SOURCE_SHA
        and ledger["lock_file_sha256"] == evidence["lock_sha256"]
        and ledger["replay_file_sha256"] == evidence["actual_replay_sha256"]
        and ledger["public_report_sha256"] == evidence["public_report_sha256"]
        and ledger["internal_contract_sha256"] == evidence["internal_contract_sha256"]
        and ledger["expression_or_clinical_transferred"] is False
        and ledger["common_in_out_initialization_actual_equal"] is True
        and ledger["graph_identity_initialization_actual_equal"] is True
        and ledger["three_arms_share_one_cache_per_fold"] is True,
        "ACTUAL_NN_EXECUTION_LEDGER_BINDING_CHANGED")
    require(len(ledger["cells"]) == 12 and {(r["dataset"],r["fold"],r["arm"]) for r in ledger["cells"]} == EXPECTED,
        "TWELVE_EXECUTION_LEDGER_CELLS_REQUIRED")
    for row in ledger["cells"]:
        name=f"{row['dataset']}_fold{row['fold']}_seed1_{row['arm']}"
        require(row["seed"] == 1 and row["actual_neural_fits"] == 1
            and row["actual_epochs"] == row["optimizer_updates"] == 40
            and row["saved_epoch_states"] == [10,20,40]
            and row["COMPLETE_sha256"] == evidence["complete_file_sha256"][name],
            "LEDGER_COMPLETE_OR_UPDATE_COUNTS_CHANGED")
    for name,count in ((FOLD_CSV,36),(GENE_CSV,1800)):
        declared=review["outputs"][Path(name).name]
        with (root/name).open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
        require(len(rows) == declared["rows"] == count and sha(root/name) == declared["sha256"],
            "INDEPENDENT_NN_CSV_BYTES_OR_COUNTS_CHANGED")
    paths={REPORT,REVIEW,REVIEW_SCRIPT,LEDGER,FOLD_CSV,GENE_CSV}
    return {name:sha(root/name) for name in paths}


def completion_bundle(root, private, freeze_path):
    result=metadata_evidence(root,private,freeze_path)
    result["public_proof_sha256"]=public_proofs(root,result)
    result["status"]="PASS_ACTUAL_NN_COMPLETION_AND_INDEPENDENT_REVIEW_BINDINGS"
    return result


def verify_device_projection(root, receipt):
    name="reports/spgel/HEST_BF16_DEVICE_PUBLIC_NUMERIC_AUDIT_20261029.json"
    projection=read(root/name)
    require(projection["schema"] == "hest_bf16_device_qualification_anonymous_numeric_projection_v1"
        and projection["status"] == "PASS_EXACT_NUMERIC_PROJECTION_BOUND_TO_PRIVATE_ORIGINAL_RECEIPT"
        and projection["source_receipt_sha256"] == sha(receipt)
        == "dfefd1b00711593b5a89c50b2ff3d9875ee13ed2818444ea9ea815484d023141"
        and projection["projection"] == dict(removed_top_level_field_names=["profile_official_sample_id"],
            substituted_values=False,numeric_fields_changed=0,numeric_scalar_fields_preserved=26,
            all_other_original_fields_and_values_preserved_exactly=True),
        "DEVICE_PROJECTION_ORIGINAL_BINDING_CHANGED")
    original=read(receipt)
    require("profile_official_sample_id" in original,"DEVICE_ORIGINAL_PROFILE_FIELD_REQUIRED")
    original.pop("profile_official_sample_id")
    require(original == projection["original_device_qualification_without_provider_identifier"],
        "DEVICE_PUBLIC_PROJECTION_NUMERIC_DRIFT")
    return dict(original_receipt_sha256=sha(receipt),projection_sha256=sha(root/name),
        only_provider_identifier_removed=True,all_other_values_equal=True)


def publish_manifest(root, files, denylist, bundle_path, device_receipt):
    rows=inventory(root,files);bundle=read(bundle_path);names={r["file"] for r in rows}
    require(bundle["schema"] == "hest_neural_local_completion_evidence_v1"
        and bundle["status"] == "PASS_ACTUAL_NN_COMPLETION_AND_INDEPENDENT_REVIEW_BINDINGS"
        and bundle["source_canonical_sha256"] == SOURCE_SHA
        and canonical(bundle["sources"]) == SOURCE_SHA and len(bundle["sources"]) == 11
        and bundle["actual_neural_fits"] == 12 and bundle["actual_optimizer_updates"] == 480
        and bundle["actual_state_replays"] == 36 and bundle["actual_unsupervised_PCA_reconstructions"] == 4
        and bundle["new_fits_by_publication"] == 0 and bundle["arrays_or_models_read_by_publication"] is False,
        "COMPLETE_INDEPENDENTLY_BOUND_NN_EVIDENCE_REQUIRED")
    proofs=public_proofs(root,bundle)
    require(proofs == bundle["public_proof_sha256"]
        and all(sha(root/n) == h for n,h in bundle["sources"].items()), "NN_EVIDENCE_BYTES_CHANGED")
    require(set(bundle["sources"]) | set(proofs) | ADMIN | SUPPLEMENT
        | {PREFIX+"PROTOCOL_20261029.md",PREFIX+"RESULTS_20261029.md"} <= names,
        "REQUIRED_NN_CODE_OR_COMPLETE_REPORTS_MISSING")
    device=verify_device_projection(root,device_receipt)
    scanned,pattern=prior.checked_scan(root,rows,denylist)
    audit=dict(schema="hest_neural_publication_source_audit_v1",
        status="PASS_COMPLETE_NN_EVIDENCE_AND_ESTABLISHED_EXACT_IDENTIFIER_SCAN",
        base_git_commit=BASE_COMMIT,base_git_tree=BASE_TREE,
        denylist_sha256=prior.DENY_SHA,inherited_scanner_sha256=sha(Path(prior.inherited.__file__)),
        publication_script_sha256=sha(root/"scripts/hest_neural_publication.py"),
        completion_evidence=bundle,device_projection=device,input_files=len(rows),
        input_bytes=sum(r["bytes"] for r in rows),new_model_fits=0,model_deserializations=0,
        old_scientific_or_publication_files_modified=False,git_operations=0)
    raw=(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    ar=dict(file=AUDIT,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    manifest=dict(schema="hest_neural_publication_manifest_v1",status="COMPLETE_ANONYMOUS_FIXED12_STAGE",
        base_git_commit=BASE_COMMIT,base_git_tree=BASE_TREE,source_audit_sha256=ar["sha256"],
        files=sorted(scanned+[ar],key=lambda r:r["file"]),
        file_count_including_audit_excluding_this_manifest=len(rows)+1,
        total_bytes_including_audit_excluding_this_manifest=sum(r["bytes"] for r in rows)+len(raw),
        manifest_does_not_hash_itself=True,git_operations=0)
    for obj in (audit,manifest):
        prior.inherited.validate_public_json(obj)
        text=json.dumps(obj,ensure_ascii=False)
        require(not any(p.search(text) for p in prior.inherited.PATTERNS.values())
            and pattern.search(text) is None,"GENERATED_NN_PUBLICATION_PRIVACY_FAILURE")
    require(not (root/AUDIT).exists() and not (root/MANIFEST).exists(),"EXISTING_NN_PUBLICATION_OUTPUT_PRESERVED")
    (root/AUDIT).write_bytes(raw)
    with (root/MANIFEST).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(manifest,f,indent=2,allow_nan=False);f.write("\n")
    return manifest


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("mode",choices=("local-scan","evidence","publish"))
    p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--files",type=Path)
    p.add_argument("--private",type=Path);p.add_argument("--freeze",type=Path)
    p.add_argument("--denylist",type=Path);p.add_argument("--bundle",type=Path)
    p.add_argument("--device-receipt",type=Path)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.mode=="publish":
        result=publish_manifest(a.root,a.files,a.denylist,a.bundle,a.device_receipt)
        print(json.dumps(dict(schema=result["schema"],status=result["status"])));return
    result=(local_scan(a.root,inventory(a.root,a.files)) if a.mode=="local-scan"
        else completion_bundle(a.root,a.private,a.freeze))
    with a.output.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(schema=result["schema"],status=result["status"])))


if __name__ == "__main__": main()

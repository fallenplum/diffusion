"""Audit the explicit anonymous Phase22 publication scope; never commit/push.

Default/--audit is read-only, including the Git index. --write-manifest writes
only an anonymous hash inventory; --stage is a separate explicit Git mutation.
A local source mirror can pass file privacy but cannot pass the parent gate.
Partial/negative scientific results are publishable when described honestly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = "reports/phase22_spatial_bridge"
CONFIG = "configs/phase22_spatial_bridge_runtime.json"
PARENT = "dd9bddd943ab1484a224f28fe456bc34d0a96d1f"
BRANCH = "research/phase22-spatially-grounded-assimilation"
SNAPSHOT = Path("/root/autodl-tmp/phase22_private/PARENT_TRACKED_BYTES_PRIVATE.json")
SNAPSHOT_SHA = "7dfe32a4ed2af2dfe7a5c605e7a80663021e97b2366cd5f237a1e81812f9b634"
MAX_BYTES = 2 * 1024**2
MANIFEST = REPORT_ROOT + "/ARTIFACT_MANIFEST.json"

# Names are enumerated deliberately. A newly proposed report must be reviewed
# here instead of becoming public merely because it appears in a report folder.
MARKDOWN_REPORTS = {"RESULTS.md", "DATA_SOURCE_REVIEW.md", "QUALIFICATION_PLAN.md",
                    "DICTIONARY_NUMERICAL_DIAGNOSTIC.md"}
JSON_REPORT_FIELDS = {
    "DATA_QUALIFICATION.json": {
        "schema_version", "qualification_status", "source_repository", "source_revision",
        "locked_split_sha", "planned_sections", "sections_ready", "sections_pending",
        "sections_failed", "ready_sections_by_split", "ready_donors_by_split",
        "ready_sections_by_assay", "fixed_gene_count", "gene_contract",
        "missing_downloads_are_resumable", "donor_identity_split_checked",
        "exact_duplicate_source_files_crossing_splits", "actual_spatial_or_survival_model_fits",
        "survival_outcomes_read", "historical_embeddings_reused",
        "UNI2_feature_extraction_qualification", "original_WSI_image_sha",
        "known_foundation_pretraining_overlap", "license_status",
        "local_ST_data_qualification_is_not_TCGA_bulk_assay_qualification",
        "integer_native_crop_tolerance", "image_and_expression_capture_footprints_recorded_separately",
        "failure_categories", "data_locations"},
    "NUMERICAL_QUALIFICATION.json": {
        "status", "scope", "config", "independent_solver", "locked_parity_gates",
        "python", "numpy", "scipy", "source_sha256", "cases", "nonconvergence_case",
        "nonconvergence_example", "tests", "test_count", "test_result", "limitations",
        "forced_nonconvergence", "worst_scale_case", "qualification", "checks",
        "count_objective_tests", "projection_tests", "default_budget_stress_case"},
    "DICTIONARY_NUMERICAL_BUDGET_DIAGNOSTIC.json": {
        "status", "sklearn_version", "no_validation_or_test_expression_read", "no_DSS_or_PFI",
        "no_spatial_model_fits", "no_dictionary_or_coefficient_arrays_saved", "seed", "states",
        "gene_cap", "tolerance", "fixed_candidate_iteration_caps", "native_convergence_check_interval",
        "native_convergence_rule", "partition_role", "locked_split_sha256", "input_count_sha256",
        "gene_sequence_sha256", "sampled_profile_indices_sha256", "source_hashes", "budgets",
        "comparison_checks", "engineering_repair_recommendation"},
    "TRAINING_AUDIT.json": {
        "schema_version", "status", "scope", "source_hashes", "configuration_sha256",
        "data_qualification_sha256", "split_lock_sha256", "dictionary_sha256",
        "dictionary", "schedule", "fits", "cells", "expected_fits", "completed_fits",
        "formal_fits", "actual_formal_fits", "formal_model_fits", "actual_model_fits",
        "smoke_fits", "exploratory_fits", "survival_fits", "optimizer_steps",
        "failed_attempts", "interrupted_attempts", "training_curves_private",
        "test_read_during_training", "DSS_or_PFI_used", "no_DSS_or_PFI",
        "selected_arm", "selection", "validation", "limitations", "verification"},
    "SPATIAL_RESULTS.json": {
        "schema_version", "status", "scope", "selected_arm", "selection", "validation",
        "test", "aggregate", "arms", "seeds", "context_scope", "gene_metric_unit",
        "zero_count_spots_excluded_from_spatial_metrics", "initialization_seeds_not_independent_donors",
        "oracle_role", "no_ST_test_model_selection", "no_DSS_or_PFI",
        "limitations", "A_gate", "candidate_only", "independent_external_confirmation"},
    "WEAK_SAMPLING_RESULTS.json": {
        "schema_version", "schema", "status", "scope", "validation", "test", "selection",
        "aggregate", "arms", "paired_local_contrasts", "progress", "C_gate", "limitations",
        "output_contracts", "validation_selection_sha256", "no_DSS_or_PFI"},
    "decision.json": {
        "schema_version", "phase", "status", "decision", "scope", "evidence_level",
        "A", "B", "C", "A_gate", "B_gate", "C_gate", "stage_A", "stage_B", "stage_C",
        "completed", "pending", "blocked", "reasons", "limitations", "next_actions",
        "actual_model_fits", "survival_model_fits", "DSS_or_PFI_used",
        "no_survival_outcome_access", "scientific_conclusion", "data_qualification",
        "target_assay_qualification", "authorizes_TCGA_survival", "source_hashes"},
    "DELIVERY_VERIFICATION.json": {
        "schema_version", "status", "scope", "source_hashes", "tests", "test_count",
        "test_results", "parent", "publication", "privacy", "artifacts", "checks",
        "formal_fits", "data_downloads", "feature_extraction", "limitations",
        "historical_bytes_unchanged", "patient_level_outputs_public", "credentials_public"},
    "INDEPENDENT_ARTIFACT_AUDIT.json": {
        "schema_version", "status", "expected_formal_fits", "completed_receipts_found",
        "verified_formal_fits", "missing_formal_fits", "metadata_verified_fits", "replayed_fits",
        "require_complete", "replay_requested", "audit_reads", "DSS_or_PFI_read",
        "ST_test_expression_read", "public_report_contains_identifiers_or_arrays",
        "independent_replay_metrics", "training_update_journal_scope", "fits", "failures",
        "training_donors", "validation_donors", "training_sections", "validation_sections",
        "dictionary_states", "dictionary_genes", "maximum_absolute_replay_errors",
        "parameters_by_arm", "total_committed_optimizer_steps", "total_recorded_update_events"},
}
PROGRESS_FIELDS = {
    "schema", "status", "stage", "plan_sha256", "expected_cells", "completed_cells",
    "pending_cells", "actual_new_cells_this_invocation", "formal_model_fits",
    "projection_solver_calls", "projection_solver_iterations", "solver_max_iterations",
    "all_failures_retained", "not_converged_cells", "serial_section_real_overlap_evaluated",
    "synthetic_same_view_and_subdomain_only", "source_measurement_eligibility"}
JSON_REPORT_FIELDS.update({name: PROGRESS_FIELDS for name in
                          ("B_VALIDATION_PROGRESS.json", "B_TEST_PROGRESS.json")})
JSON_REPORT_FIELDS["B_SELECTION_LOCK.json"] = {
    "status", "schema", "invariant_sha256", "validation_plan_sha256",
    "validation_cells_sha256", "B2", "B3", "selection_rule", "B2_candidates",
    "B3_candidates", "test_predictions_or_test_expression_used_for_selection",
    "all_expected_validation_cells_completed_including_failures"}
JSON_REPORT_FIELDS["B_TEST_SUMMARY.json"] = {
    "status", "schema", "progress", "validation_selection_sha256", "arms",
    "paired_local_contrasts", "limitations", "C_gate", "output_contracts",
    "descriptive_assay_strata", "source_measurement_eligibility"}
MEASUREMENT_ELIGIBILITY_REPORTS = {
    "B_VALIDATION_MEASUREMENT_ELIGIBILITY.json", "B_TEST_MEASUREMENT_ELIGIBILITY.json"}
for _name in MEASUREMENT_ELIGIBILITY_REPORTS:
    JSON_REPORT_FIELDS[_name] = {
        "policy", "source_sections", "eligible_sections", "ineligible_sections",
        "eligible_assays", "fully_ineligible_assays", "by_assay",
        "fixed_train_dictionary_preserved", "no_zero_imputation_or_test_gene_reselection",
        "gate_generalization_to_ineligible_assays"}
JSON_REPORT_FIELDS["ARTIFACT_MANIFEST.json"] = {
    "schema_version", "status", "scope", "parent_commit", "branch",
    "parent_verification", "public_files", "total_files", "total_bytes",
    "manifest_self_excluded", "scientific_completion_claimed", "publication_policy"}

ALWAYS_SENSITIVE = re.compile(
    rb"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|"
    rb"\bhf_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|"
    rb"\bgh[pousr]_[A-Za-z0-9]{20,}|"
    rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----")
HEST_SOURCE_HANDLE = re.compile(
    rb"(?<![A-Za-z0-9])(?:TENX|ZEN|MISC)\d{2,}[A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*")
# Exactly four public source handles needed to reproduce a documented metadata
# adjudication. Hex encoding prevents this guard's own source from becoming an
# exception to its path rule. No runtime donor identifiers are exempted.
PUBLIC_SOURCE_HANDLES = frozenset(bytes.fromhex(x) for x in (
    "54454e58313437", "54454e58313438", "54454e58313439", "54454e58313532"))
PUBLIC_SOURCE_HANDLE_PATHS = frozenset({
    "src/phase22_spatial_bridge/data_contract.py",
    "scripts/qualify_phase22_spatial_data.py",
    "tests/test_phase22_data_contract.py"})
SOURCE_SECRET_ASSIGNMENT = re.compile(
    rb"\b(?:password|passwd|access_token|api_key)\s*[:=]\s*[\"'][^\"'\r\n]+[\"']", re.I)
DENIED_NAME = re.compile(
    r"(?:^|_)(?:private|tmp|checkpoint|predictions?|weights?|raw|coordinates|coords|"
    r"gene_list|gene_symbols|patient_ids?|donor_ids?|barcodes?)(?:_|\.|$)", re.I)
DENIED_FIELDS = {
    "patient_id", "patient_ids", "patient_index", "patient_indices", "sample_id", "sample_ids",
    "slide_id", "slide_ids", "donor_id", "donor_ids", "donor_indices", "section_id", "section_ids",
    "barcode", "barcodes", "risk", "risks", "raw_risk", "raw_risks", "risk_raw",
    "prediction", "predictions", "raw_predictions", "raw_prediction", "anchor_risk",
    "gene_ids", "gene_symbols", "gene_names", "selected_gene_ids", "selected_gene_symbols",
    "coordinates", "coords", "coords_um", "coords_px", "raw_maps", "raw_tokens", "token_values",
    "prototype_features", "pca_components", "pca_state", "model_state", "state_dict",
    "dss_time", "dss_event", "pfi_time", "pfi_event", "survival_time", "survival_event",
    "password", "access_token", "api_key", "private_key", "credentials", "signed_url"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sensitive_content_present(raw, relative):
    """Only the explicit source-path/source-handle conjunction is exempted."""
    if ALWAYS_SENSITIVE.search(raw) or SOURCE_SECRET_ASSIGNMENT.search(raw):
        return True
    return any(relative not in PUBLIC_SOURCE_HANDLE_PATHS or match.group() not in PUBLIC_SOURCE_HANDLES
               for match in HEST_SOURCE_HANDLE.finditer(raw))


def read_json(path):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=reject_constant, object_pairs_hook=unique_object)


def safe_path(relative):
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and rel.parts and "\\" not in relative and ":" not in relative
            and all(p not in {".", "..", ""} for p in rel.parts), "unsafe relative path")
    path = ROOT.joinpath(*rel.parts)
    require(path.resolve().is_relative_to(ROOT.resolve()), "escaping public path")
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink(), "symlink in public path")
        if ancestor == ROOT:
            break
    return path


def allowed_path(relative):
    p = PurePosixPath(relative)
    if p.parts[:2] == ("src", "phase22_spatial_bridge"):
        return len(p.parts) == 3 and re.fullmatch(r"[a-z_][a-z0-9_]*\.py", p.name) is not None
    if p.parts[:1] == ("scripts",):
        return len(p.parts) == 2 and re.fullmatch(r"(?=[a-z0-9_]*phase22)[a-z][a-z0-9_]*\.py", p.name) is not None
    if p.parts[:1] == ("tests",):
        return len(p.parts) == 2 and re.fullmatch(r"test_phase22_[a-z0-9_]+\.py", p.name) is not None
    if relative == CONFIG:
        return True
    return len(p.parts) == 3 and p.parent.as_posix() == REPORT_ROOT and (
        p.name in MARKDOWN_REPORTS or p.name in JSON_REPORT_FIELDS)


def numeric_axis_allowed(relative, location, values):
    """Only predeclared public axes/three-seed aggregate metrics are vectors."""
    axes = {
        "external_A.seeds": [31, 47, 83],
        "projection_B.rho_candidates": [.25, .5, .75, 1],
        "projection_B.lambda_candidates": [0, .1, 1, 10]}
    if relative == CONFIG:
        return location in axes and values == axes[location]
    if PurePosixPath(relative).name == "DICTIONARY_NUMERICAL_BUDGET_DIAGNOSTIC.json":
        return location == "fixed_candidate_iteration_caps" and values == [500, 1000, 2000]
    if PurePosixPath(relative).name == "NUMERICAL_QUALIFICATION.json":
        # An explicitly synthetic optimizer history is not a patient vector.
        return (location.startswith("default_budget_stress_case.diagnostics.")
                and location.rsplit(".", 1)[-1] in {"objective_history", "step_size_history", "kkt_history"}
                and len(values) <= 101)
    if PurePosixPath(relative).name == "SPATIAL_RESULTS.json":
        if location in {"seeds", "selection.selected_seeds"}:
            return values == [31, 47, 83]
        if re.fullmatch(r"selection\.(?:validation_scores_by_seed\.A[012]|delta_A1_minus_A0|delta_A2_minus_A1)", location):
            return len(values) == 3
    if PurePosixPath(relative).name == "INDEPENDENT_ARTIFACT_AUDIT.json":
        return (re.fullmatch(r"parameters_by_arm\.A[012]", location) is not None
                and len(values) <= 1 and all(isinstance(v, int) and 0 < v <= 20_000_000 for v in values))
    b_reports = {"B_TEST_SUMMARY.json", "WEAK_SAMPLING_RESULTS.json", *MEASUREMENT_ELIGIBILITY_REPORTS,
                 "B_VALIDATION_PROGRESS.json", "B_TEST_PROGRESS.json"}
    if PurePosixPath(relative).name in b_reports:
        field = location.rsplit(".", 1)[-1]
        if field in {"ordinary_95_interval", "six_primary_comparison_Bonferroni_interval"}:
            return len(values) == 2 and values[0] <= values[1]
        if field == "unmeasured_dictionary_gene_count_by_ineligible_section":
            return (len(values) <= 16 and all(isinstance(v, int) and 0 <= v <= 1024 for v in values)
                    and values == sorted(values))
    return False


def audit_json(value, relative, location=""):
    if isinstance(value, dict):
        for key, item in value.items():
            name = str(key).lower()
            require(name not in DENIED_FIELDS and not (name.endswith("_private") and not isinstance(item, bool)),
                    "protected JSON field in " + relative)
            require(not (name in {"selected_genes", "genes", "features", "counts", "tokens", "mu", "pi0"}
                         and isinstance(item, (list, dict))), "raw structured payload in " + relative)
            audit_json(item, relative, location + "." + key if location else key)
    elif isinstance(value, list):
        require(len(value) <= 1000, "oversized public record list in " + relative)
        require(not any(isinstance(v, list) for v in value), "raw nested array in " + relative)
        if any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            require(all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
                    and numeric_axis_allowed(relative, location, value),
                    "unapproved numeric vector in " + relative + ": " + location)
        for item in value:
            audit_json(item, relative, location + "[]")
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite report scalar")
    elif isinstance(value, str):
        require(len(value) <= 16384 and not re.search(r"[A-Za-z0-9+/]{512,}={0,2}", value),
                "encoded/raw payload in report string")
        require(not re.search(r"[?&](?:X-Amz-|Policy=|Signature=|token=)", value, re.I),
                "signed capability in report string")
    else:
        require(value is None or isinstance(value, (int, bool)), "unsupported report scalar")


def audit_public_file(relative):
    require(allowed_path(relative), "unexpected file outside explicit Phase22 publication scope; path SHA256="
            + sha(relative.encode("utf-8")))
    path = safe_path(relative)
    require(path.is_file() and path.stat().st_size <= MAX_BYTES, "missing/oversized public file: " + relative)
    require(not DENIED_NAME.search(path.stem), "private payload filename: " + relative)
    raw = path.read_bytes()
    require(not sensitive_content_present(raw, relative), "identifier/credential content detected: " + relative)
    raw.decode("utf-8-sig")
    if path.suffix == ".json":
        value = read_json(path)
        require(isinstance(value, dict), "public JSON root must be an object: " + relative)
        if relative != CONFIG:
            allowed = JSON_REPORT_FIELDS[path.name]
            require(set(value) <= allowed, "unreviewed public report schema: " + relative + ": "
                    + ",".join(sorted(set(value) - allowed)))
            require(any(k in value for k in ("status", "qualification_status"))
                    or path.name in MEASUREMENT_ELIGIBILITY_REPORTS and isinstance(value.get("policy"), str),
                    "report must state its actual status: " + relative)
            if path.name == "DICTIONARY_NUMERICAL_BUDGET_DIAGNOSTIC.json":
                require(value.get("seed") == 31 and value.get("states") == 32
                        and value.get("partition_role") == "ST-train"
                        and all(value.get(k) is True for k in (
                            "no_validation_or_test_expression_read", "no_DSS_or_PFI",
                            "no_spatial_model_fits", "no_dictionary_or_coefficient_arrays_saved")),
                        "dictionary numerical diagnostic scope mismatch")
                nested_schemas = {
                    "comparison_checks": {
                        "same_input_gene_and_sample_hashes_at_each_cap",
                        "500_prefix_matches_1000_native_trace_with_absolute_tolerance_1e_12",
                        "1000_and_2000_stop_at_same_iteration",
                        "1000_and_2000_dictionary_hash_identical_in_diagnostic_environment",
                        "500_original_server_sqrt_2KL", "500_local_server_objective_absolute_difference",
                        "server_environment",
                        "prior_server_dictionary_not_saved_no_elementwise_original_dictionary_replay_claim"},
                    "engineering_repair_recommendation": {
                        "status", "old_max_iterations", "new_max_iterations", "actual_convergence_iteration",
                        "change_tolerance", "change_seed", "change_data_or_sampling", "change_genes_or_states",
                        "validation_or_test_used_for_repair", "changes_A_model_schedule_or_nine_cell_budget",
                        "basis", "not_a_predictive_gain_claim", "not_global_optimum_or_KKT_certification"}}
                for field, allowed_nested in nested_schemas.items():
                    if field in value:
                        require(isinstance(value[field], dict) and set(value[field]) <= allowed_nested,
                                "unreviewed dictionary numerical diagnostic validation fields")
                budget_fields = {"iteration_cap", "actual_iterations", "converged", "reconstruction_sqrt_2KL",
                    "total_KL", "mean_profile_KL", "initial_sqrt_2KL", "final_native_scaled_error_decrease",
                    "previous_10step_relative_KL_decrease", "total_profiles", "genes", "states", "seconds",
                    "diagnostic_dictionary_sha256", "native_trace", "relative_KL_decrease_vs_previous_cap"}
                require(isinstance(value.get("budgets"), list) and len(value["budgets"]) <= 3,
                        "dictionary numerical budget record count invalid")
                for budget in value["budgets"]:
                    require(isinstance(budget, dict) and set(budget) <= budget_fields,
                            "unreviewed dictionary numerical budget fields")
                    require(isinstance(budget.get("native_trace"), list)
                            and len(budget["native_trace"]) <= 201,
                            "dictionary numerical trace too long")
                    for row in budget["native_trace"]:
                        require(isinstance(row, dict) and set(row) == {"iteration", "sqrt_2KL", "total_KL"}
                                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in row.values()),
                                "dictionary numerical trace must contain three aggregate scalars only")
            if path.name == "INDEPENDENT_ARTIFACT_AUDIT.json":
                require(value.get("schema_version") == "phase22_independent_artifact_audit_v1"
                        and value.get("expected_formal_fits") == 9
                        and value.get("DSS_or_PFI_read") is False
                        and value.get("ST_test_expression_read") is False
                        and value.get("public_report_contains_identifiers_or_arrays") is False,
                        "independent artifact audit scope mismatch")
                require(isinstance(value.get("fits"), list) and len(value["fits"]) <= 9
                        and isinstance(value.get("failures"), list) and len(value["failures"]) <= 10,
                        "independent artifact audit record count invalid")
                fit_fields = {"arm", "seed", "actual_epochs", "selected_epoch", "committed_optimizer_steps",
                    "parameters", "zero_count_regions_skipped", "replayed", "replay_absolute_errors",
                    "process_attempts", "completed_attempts", "recorded_optimizer_update_events",
                    "repeated_recorded_updates_after_interruptions"}
                failure_fields = {"arm", "seed", "code", "error_class"}
                for row in value["fits"]:
                    require(isinstance(row, dict) and set(row) <= fit_fields,
                            "unreviewed independent audit fit record fields")
                for row in value["failures"]:
                    require(isinstance(row, dict) and set(row) <= failure_fields,
                            "unreviewed independent audit failure record fields")
                if value.get("status") == "PASS_INDEPENDENT_REPLAY":
                    require(len(value["fits"]) == value.get("verified_formal_fits")
                            == value.get("metadata_verified_fits") == value.get("replayed_fits") == 9
                            and value.get("missing_formal_fits") == 0 and not value["failures"]
                            and all(row.get("replayed") is True for row in value["fits"]),
                            "independent replay PASS requires nine real replayed cells")
        audit_json(value, relative)
    return {"path": relative, "bytes": len(raw), "sha256": sha(raw)}


def publication_paths():
    candidates = []
    for base, pattern in (("src/phase22_spatial_bridge", "*.py"),
                          ("tests", "test_phase22_*.py"), ("scripts", "*phase22*.py")):
        candidates += list((ROOT / base).glob(pattern))
    candidates.append(ROOT / CONFIG)
    # Fail closed on every unexpected report, including hidden data files.
    candidates += [p for p in (ROOT / REPORT_ROOT).rglob("*") if p.is_file()]
    paths = sorted({p.relative_to(ROOT).as_posix() for p in candidates})
    require(paths and CONFIG in paths and "scripts/package_phase22.py" in paths,
            "source/configuration/publication guard missing")
    return paths


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def audit_parent():
    probe = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                           capture_output=True, text=True)
    if probe.returncode != 0:
        return {"status": "NOT_RUN_LOCAL_MIRROR", "scientific_or_git_publication_verified": False}
    require(Path(probe.stdout.strip()).resolve() == ROOT.resolve(), "not the intended repository root")
    require(git("branch", "--show-current").decode().strip() == BRANCH, "wrong Phase22 branch")
    subprocess.run(["git", "merge-base", "--is-ancestor", PARENT, "HEAD"], cwd=ROOT, check=True)
    require(SNAPSHOT.is_file() and sha(SNAPSHOT.read_bytes()) == SNAPSHOT_SHA,
            "missing or changed private parent snapshot")
    record = read_json(SNAPSHOT)
    snapshot = record["files_sha256"]
    require(record["parent_commit"] == PARENT and record["branch"] == BRANCH
            and record["tracked_file_count"] == len(snapshot) == 2508, "parent snapshot identity mismatch")
    tree = {}
    for item in git("ls-tree", "-r", "-z", PARENT).split(b"\0"):
        if not item:
            continue
        head, name = item.split(b"\t", 1)
        mode, kind, oid = head.decode().split()
        require(kind == "blob" and mode in {"100644", "100755"}, "unsupported inherited tracked object")
        tree[name.decode("utf-8")] = (mode, oid)
    require(set(tree) == set(snapshot), "parent snapshot path set differs from exact Git tree")
    for relative, expected in snapshot.items():
        path = safe_path(relative)
        require(path.is_file(), "inherited file missing")
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        require(sha(raw) == expected and blob == tree[relative][1], "inherited bytes changed")
    # Include mode/deletion changes, which content hashes alone would miss.
    changed = [p.decode("utf-8") for p in git("diff", "--name-only", "-z", PARENT, "--").split(b"\0") if p]
    require(not (set(changed) & set(snapshot)), "inherited tracked mode/index/worktree changed")
    tracked = [p.decode("utf-8") for p in git("ls-files", "-z").split(b"\0") if p]
    require(all(p in snapshot or allowed_path(p) for p in tracked), "new tracked file outside allowlist")
    staged = [p.decode("utf-8") for p in git("diff", "--cached", "--name-only", "-z").split(b"\0") if p]
    require(all(allowed_path(p) for p in staged), "existing staged change outside Phase22 scope")
    for relative in staged:
        require(git("show", ":" + relative) == safe_path(relative).read_bytes(),
                "existing staged bytes differ from audited working copy")
    return {"status": "PASS", "parent_commit": PARENT, "branch": BRANCH,
            "inherited_tracked_files_verified": len(snapshot),
            "all_inherited_tracked_bytes_unchanged": True,
            "private_parent_snapshot_sha256": SNAPSHOT_SHA,
            "new_tracked_and_staged_paths_within_allowlist": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="read-only audit (default)")
    parser.add_argument("--write-manifest", action="store_true", help="write anonymous inventory only")
    parser.add_argument("--stage", action="store_true", help="explicitly git-add audited allowlisted files")
    args = parser.parse_args()
    if args.audit and (args.write_manifest or args.stage):
        parser.error("--audit is read-only and cannot be combined with write/stage actions")
    try:
        paths = publication_paths()
        files = [audit_public_file(p) for p in paths if p != MANIFEST]
        parent = audit_parent()
        manifest = dict(schema_version="phase22_publication_inventory_v1",
            status="PASS" if parent["status"] == "PASS" else "PUBLIC_FILES_PASS_PARENT_NOT_RUN",
            scope="new_Phase22_source_configuration_tests_anonymous_aggregates_only",
            parent_commit=PARENT, branch=BRANCH, parent_verification=parent,
            public_files=files, total_files=len(files), total_bytes=sum(x["bytes"] for x in files),
            manifest_self_excluded=True, scientific_completion_claimed=False,
            publication_policy="no identifiers, patient/spot arrays, predictions, gene lists, weights or credentials")
        if MANIFEST in paths:
            audit_public_file(MANIFEST)
            if not args.write_manifest:
                require(read_json(ROOT / MANIFEST) == manifest, "existing public manifest is stale")
        if args.write_manifest:
            path = safe_path(MANIFEST)
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            audit_public_file(MANIFEST)
        if args.stage:
            require(parent["status"] == "PASS", "Git staging requires verified server parent gate")
            require((ROOT / MANIFEST).exists() and read_json(ROOT / MANIFEST) == manifest,
                    "write a current reviewed inventory before staging")
            stage_paths = sorted(set(paths + [MANIFEST]))
            subprocess.run(["git", "add", "--", *stage_paths], cwd=ROOT, check=True)
            audit_parent()
        print(json.dumps({"status": manifest["status"], "public_files": len(files),
            "public_bytes": manifest["total_bytes"], "parent": parent,
            "manifest_written": args.write_manifest, "git_index_modified": args.stage,
            "commit_or_push_performed": False}, indent=2, sort_keys=True))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        # No raw payload or exception traceback can leak into a public receipt.
        print(json.dumps({"status": "FAIL", "reason": str(error),
                          "commit_or_push_performed": False}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Assemble anonymous Phase22 decisions from completed frozen A/B receipts.

This reporter does not fit models, read survival outcomes, select candidates,
or export identifiers, checkpoints, coefficients or position-level arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spatial-private-out", type=Path, required=True)
    p.add_argument("--public-report-dir", type=Path,
                   default=Path(__file__).resolve().parents[1] / "reports/phase22_spatial_bridge")
    args = p.parse_args()
    out, reports = args.spatial_private_out, args.public_report_dir
    selected = read(out / "SELECTION_PRIVATE.json")
    val = read(out / "evaluation/validation/AGGREGATE_ANONYMOUS.json")
    test = read(out / "evaluation/test/AGGREGATE_ANONYMOUS.json")
    replay = read(reports / "INDEPENDENT_ARTIFACT_AUDIT.json")
    bval = read(reports / "B_VALIDATION_PROGRESS.json")
    bsel = read(reports / "B_SELECTION_LOCK.json")
    btest = read(reports / "B_TEST_SUMMARY.json")
    expected_val = {(arm, seed) for arm in ("A0", "A1", "A2") for seed in (31, 47, 83)}
    expected_test = {(selected["selected_arm"], seed) for seed in (31, 47, 83)}
    def keys(rows):
        return {(r["arm"], r["seed"]) for r in rows}
    if not (replay["status"] == "PASS_INDEPENDENT_REPLAY" and replay["replayed_fits"] == 9
            and selected["status"] == "SELECTED_ON_VALIDATION_ONLY"
            and selected["test_used_for_selection"] is False
            and selected["DSS_or_PFI_used"] is False
            and replay["DSS_or_PFI_read"] is False and replay["ST_test_expression_read"] is False
            and len(replay["fits"]) == 9 and keys(replay["fits"]) == expected_val
            and val["status"] == test["status"] == "COMPLETE"
            and len(val["aggregate"]) == 9 and len(test["aggregate"]) == 3
            and keys(val["aggregate"]) == expected_val and keys(test["aggregate"]) == expected_test
            and val["split"] == "validation" and test["split"] == "test"
            and val["no_DSS_or_PFI"] is test["no_DSS_or_PFI"] is True
            and bval["status"] == btest["status"] == "COMPLETE"
            and bval["pending_cells"] == btest["progress"]["pending_cells"] == 0
            and bsel["status"] == "LOCKED_ON_VALIDATION_ONLY"
            and btest["validation_selection_sha256"] == sha(reports / "B_SELECTION_LOCK.json")):
        raise ValueError("complete frozen A/B receipts required; no inferred completion")
    # Bind anonymous tables to their current private receipts and exact bytes.
    for r in replay["fits"]:
        model = selected["models"]["%s_seed%03d" % (r["arm"], r["seed"])]
        path = Path(model["model_path"])
        done_path = path.with_name("DONE_PRIVATE.json")
        done = read(done_path)
        if not (sha(path) == model["model_sha256"] == done["model_sha256"]
                and sha(done_path) == model["done_sha256"]
                and done["actual_epochs"] == r["actual_epochs"]
                and done["selected_epoch"] == r["selected_epoch"]
                and done["identity"]["source_hashes"] == selected["source_hashes"]
                and done["identity"]["split_lock_sha256"] == selected["split_lock_sha256"]
                and done["identity"]["dictionary_npz_sha256"] == selected["dictionary_npz_sha256"]
                and done["identity"]["data_qualification_sha256"] == selected["data_qualification_sha256"]):
            raise ValueError("independent replay/current checkpoint identity mismatch")
    for role, expected in (("validation", expected_val), ("test", expected_test)):
        receipt = read(out / "evaluation" / role / "EVALUATION_PRIVATE.json")
        if not (receipt["status"] == "COMPLETE"
                and receipt["split_lock_sha256"] == selected["split_lock_sha256"]
                and receipt["dictionary_npz_sha256"] == selected["dictionary_npz_sha256"]
                and receipt["source_hashes"] == selected["source_hashes"]
                and receipt["no_DSS_or_PFI"] is True
                and receipt["no_ST_test_model_selection"] is True
                and receipt["selection_receipt_sha256"] == (
                    sha(out / "SELECTION_PRIVATE.json") if role == "test" else None)):
            raise ValueError("A evaluation companion receipt identity mismatch")
        rows = receipt["rows_private"]
        unique = {(r["arm"], r["seed"], r["section_id_private"]) for r in rows}
        if len(rows) != 4 * len(expected) or len(unique) != len(rows) or keys(rows) != expected:
            raise ValueError("A evaluation exact cell set mismatch")
        for r in rows:
            identity = selected["models"]["%s_seed%03d" % (r["arm"], r["seed"])]["model_sha256"]
            if (r["split"] != role or r["source_model_sha256"] != identity
                    or sha(r["path_private"]) != r["sha256"]):
                raise ValueError("A frozen prediction/source-model mismatch")
    # Explicit fields prevent accidental inclusion of private checkpoint paths.
    fields = ("selected_arm", "selected_seeds", "A_gate", "candidate_only",
              "independent_external_confirmation", "gate_metric", "gate_rule",
              "gate_is_not_statistical_significance", "complexity_rule",
              "validation_scores_by_seed", "delta_A1_minus_A0", "delta_A2_minus_A1",
              "retained_A1", "retained_A2", "test_used_for_selection",
              "DSS_or_PFI_used", "split_lock_sha256", "data_qualification_sha256",
              "dictionary_npz_sha256", "source_hashes", "limitations")
    clean_selection = {k: selected[k] for k in fields}
    limitations = [
        "two_source_studies_and_two_validation_donors",
        "three_test_donors_not_three_independent_research_studies",
        "initialization_seeds_do_not_increase_independent_donor_count",
        "unknown_foundation_model_pretraining_overlap",
        "sampled_training_regions_versus_full_section_validation_and_test",
        "A_partial_measurement_uses_measured_gene_composition;B_requires_full_fixed_dictionary",
        "fixed_model_donor_bootstrap_is_not_full_pipeline_retraining_uncertainty",
        "no_TCGA_assay_transform_or_target_image_footprint_qualification_implied"]
    write(reports / "SPATIAL_RESULTS.json", {
        "schema_version": "phase22_spatial_results_v1", "status": "COMPLETE",
        "scope": "external_ST_A_only", "selected_arm": selected["selected_arm"],
        "selection": clean_selection, "validation": val, "test": test,
        "A_gate": selected["A_gate"], "candidate_only": True,
        "independent_external_confirmation": False, "no_DSS_or_PFI": True,
        "limitations": limitations})
    gate = btest["C_gate"]["status"]
    write(reports / "decision.json", {
        "schema_version": "phase22_decision_v1", "phase": "Phase22",
        "status": "COMPLETE_EXTERNAL_A_B_C_NOT_UNLOCKED",
        "scope": "locked_external_spatial_and_synthetic_weak_sampling_audit",
        "evidence_level": "small_donor_external_candidate_not_survival_confirmation",
        "stage_A": {"status": "COMPLETE", "formal_fits": 9,
                    "independent_replayed_fits": replay["replayed_fits"],
                    "selected_arm": selected["selected_arm"], "gate": selected["A_gate"],
                    "retained_A2": selected["retained_A2"],
                    "test_selected_models_only": True},
        "stage_B": {"status": "COMPLETE", "gate": gate,
                    "validation_cells": bval["completed_cells"],
                    "validation_not_converged_cells": bval["not_converged_cells"],
                    "test_cells": btest["progress"]["completed_cells"],
                    "test_not_converged_cells": btest["progress"]["not_converged_cells"],
                    "B2": bsel["B2"], "B3": bsel["B3"],
                    "selection_sha256": btest["validation_selection_sha256"],
                    "evaluated_assays": btest["C_gate"]["evaluated_assays"],
                    "not_evaluated_assays": btest["C_gate"]["not_evaluated_assays"]},
        "stage_C": {"status": "NOT_UNLOCKED", "new_survival_fits": 0,
                    "DSS_or_PFI_read": False},
        "target_assay_qualification": {
            "status": "NOT_QUALIFIED", "composition_transform_T": "NOT_ESTABLISHED",
            "target_UNI2_input_footprint": "NOT_QUALIFIED"},
        "actual_model_fits": 9, "survival_model_fits": 0,
        "DSS_or_PFI_used": False, "no_survival_outcome_access": True,
        "authorizes_TCGA_survival": False,
        "reasons": btest["C_gate"]["reasons"] + [
            "target_assay_transform_and_image_footprint_remain_unqualified"],
        "scientific_conclusion": (
            selected["selected_arm"] + "_is_a_validation_selected_spatial_candidate;"
            + ("patient_matched_weak_bulk_specificity_not_demonstrated_in_the_locked_B_comparison;"
               if gate == "NOT_SUPPORTED_UNDER_LOCKED_EXTERNAL_AUDIT" else
               "read_the_recorded_B_gate_and_its_scope;no_automatic_target_assay_transfer;")
            + "does_not_reject_all_spatial_supervision_or_stronger_constraint_solvers"),
        "limitations": limitations + [
            "large_lambda_validation_candidates_failed_the_fixed_100_step_KKT_gate",
            "no_real_serial_section_registration_or_measured_human_sampling_overlap"],
        "next_actions": [
            "preserve_frozen_external_results_and_private_failure_states",
            "do_not_expand_survival_or_model_matrix_from_this_receipt",
            "any_solver_repair_or_target_assay_transfer_requires_a_separate_locked_audit"],
        "source_hashes": selected["source_hashes"]})
    print(json.dumps({"status": "COMPLETE", "selected_arm": selected["selected_arm"],
                      "B_gate": gate, "survival_fits": 0}))


if __name__ == "__main__":
    main()

"""Read-only, endpoint-free mechanism audit of all 15 completed inner cells.

Only source-qualified kernel/readout checkpoints and DONE receipts are opened.
No data loader, endpoint vector, fitting routine or new patient prediction is
called. Public output contains aggregate scalars and cell identities only.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase23_cdei.runtime import (PRIVATE, atomic_json, cfg, sha, source, verify_done)

ARMS = ("REL", "DUAL", "ADD", "NO_C", "WRONG_O", "WRONG_C", "MEAN", "CENTERED")
SENSITIVITIES = ("DUAL_ABS_OFF", "DUAL_WRONG_O", "DUAL_WRONG_C", "DUAL_NO_C")


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def array(value, name, ndim=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    require(np.issubdtype(result.dtype, np.number) and not np.iscomplexobj(result), name + " is not real numeric")
    require(np.isfinite(result).all() and result.size > 0, name + " is empty or nonfinite")
    if ndim is not None:
        require(result.ndim == ndim, name + " has the wrong rank")
    return result.astype(np.float64, copy=False)


def summary(values):
    x = array(values, "anonymous summary").reshape(-1)
    result = {"count": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
              "population_SD": float(x.std(ddof=0)), "RMS": float(np.sqrt(np.mean(x * x))),
              "minimum": float(x.min()), "maximum": float(x.max()),
              "maximum_absolute": float(np.abs(x).max())}
    require(all(np.isfinite(v) for v in result.values()), "summary arithmetic overflow")
    return result


def correlation(x, y):
    a, b = array(x, "paired absolute feature", 1), array(y, "paired counterfactual feature", 1)
    require(a.shape == b.shape and len(a) >= 2, "Pearson needs matching nonempty patient rows")
    a, b = a - a.mean(), b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0. or nb == 0.:
        return {"defined": False, "reason": "CONSTANT_FEATURE", "patients": len(a)}
    value = float(np.dot(a / na, b / nb))
    require(np.isfinite(value) and abs(value) <= 1. + 1e-12, "invalid Pearson result")
    return {"defined": True, "Pearson": float(np.clip(value, -1., 1.)), "patients": len(a)}


def normalized_risk_summary(values, anchor_sd):
    result = summary(values)
    result.update(RMS_over_fit_anchor_SD=result["RMS"] / anchor_sd,
                  population_SD_over_fit_anchor_SD=result["population_SD"] / anchor_sd,
                  maximum_absolute_over_fit_anchor_SD=result["maximum_absolute"] / anchor_sd)
    return result


def _warmup(audit):
    rows = audit["history"]
    require(audit["requested_epochs"] == audit["actual_epochs"] == audit["optimizer_steps"] == len(rows) == 80,
            "warmup is not the locked 80 complete optimizer steps")
    require([r["epoch"] for r in rows] == list(range(1, 81)), "warmup epoch sequence changed")
    require(audit["budget_mode"] == "FORMAL_FIXED_80" and audit["backbone_frozen_eval"]
            and audit["frozen_anchor_unchanged"] and audit["heldout_endpoint_reads"] == 0,
            "warmup isolation/freeze contract changed")
    training = array([r["training_loss"] for r in rows], "warmup training losses", 1)
    fitted = array([r["fit_loss"] for r in rows], "updated warmup fit losses", 1)
    groups = set(rows[0]["gradients"])
    require(all(set(row["gradients"]) == groups for row in rows), "gradient group keys changed")
    gradients = {}
    for group in sorted(groups):
        values = array([r["gradients"][group] for r in rows], "warmup gradients", 1)
        gradients[group] = {"first": float(values[0]), "last": float(values[-1]),
                            "median": float(np.median(values)), "minimum": float(values.min()),
                            "maximum": float(values.max()), "zero_epochs": int((values == 0).sum())}
    improvement = float(fitted[-11] - fitted[-1])
    return {"actual_epochs": 80, "optimizer_steps": 80, "fixed_final_epoch": True,
            "training_loss_first": float(training[0]), "training_loss_last": float(training[-1]),
            "fit_loss_first_updated": float(fitted[0]), "fit_loss_last_updated": float(fitted[-1]),
            "last10_fit_loss_improvement": improvement,
            "last10_relative_fit_loss_improvement": improvement / max(abs(float(fitted[-11])), 1e-12),
            "last10_stochastic_training_loss_improvement": float(training[-11] - training[-1]),
            "last10_fit_loss_slope": float(np.polyfit(np.arange(10), fitted[-10:], 1)[0]),
            "gradient_groups": gradients, "gradient_global_norm": summary([r["grad_norm"] for r in rows]),
            "epoch_seconds": summary([r["epoch_seconds"] for r in rows]),
            "generalization_or_global_convergence_established": False,
            "scope": "training_objective_trends_only;fixed_budget_is_not_a_convergence_proof"}


def _pairing(data_audit):
    controls = data_audit["wrong_controls"]
    fields = ("patients", "primary_cancer_pattern_strata", "primary_singleton_patients",
              "missing_pattern_strata", "pattern_singleton_patients", "identity_count",
              "missing_pattern_mismatch_count", "relaxed_cancer_match_count")
    support_fields = ("patients", "token_slots", "own_valid_tokens_total", "donor_valid_tokens_total",
                      "shared_valid_tokens_total", "shared_valid_tokens_patient_min",
                      "shared_valid_tokens_patient_max", "shared_valid_tokens_patient_mean",
                      "support_changed_patients")
    result = {}
    for partition in ("fit", "valid"):
        result[partition] = {}
        for branch in ("clinical", "omics"):
            original = controls[partition][branch]
            require(original["status"] == "AVAILABLE", "a wrong-patient control is unavailable")
            record = {"status": original["status"]}
            record.update({key: original[key] for key in fields})
            if branch == "omics":
                support = original["token_support"]
                record["token_support"] = {key: support[key] for key in support_fields}
                record["RNA_vs_CNA_support_change_separately_persisted"] = False
                record["support_scope"] = "combined_24_token_support;separate_RNA_CNA_changes_not_reconstructed"
            result[partition][branch] = record
    return result


def _reference(kernel, readout):
    state = kernel["reference_state"]
    counts = array(state["_slot_counts"], "fit reference slot counts", 1)
    mask = np.asarray(state["_measured_mask"].detach().cpu(), dtype=bool)
    clinical = array(state["_clinical"], "fit-only reference clinical", 2)
    require(counts.shape == (24,) and mask.shape == (len(kernel["fit"]), 24), "reference slot shape changed")
    require(clinical.shape == (len(kernel["fit"]), int(kernel["clinical_dim"]))
            and clinical.shape[1] > 0, "fit-local clinical dimension changed")
    np.testing.assert_array_equal(mask.sum(0), counts)
    result = {"fit_patient_count": len(mask), "token_slots": 24,
              "fit_local_clinical_dimension": clinical.shape[1],
              "fit_slot_counts_before_query_exclusion": summary(counts),
              "actual_per_query_counts_persisted": False,
              "paired_query_counts_equal_receipt": bool(readout["feature_audit"]["paired_reference_counts_equal"])}
    for channel, lo in (("RNA", 0), ("CNA_gain", 8), ("CNA_loss", 16)):
        result[channel] = {"slot_donor_counts_before_query_exclusion": summary(counts[lo:lo + 8]),
                           "fit_patients_with_missing_slot": int((~mask[:, lo:lo + 8]).any(1).sum())}
    return result


def summarize_cell(kernel, readout, *, identity, seed, fold):
    for item in (kernel, readout):
        require(all(item.get(key) == value for key, value in
                    (("source", identity), ("role", "inner"), ("seed", seed), ("fold", fold))),
                "checkpoint identity mismatch")
    for partition in ("fit", "valid"):
        np.testing.assert_array_equal(kernel[partition], readout[partition])
    require(set(readout["arms"]) == set(ARMS) and set(readout["sensitivities"]) == set(SENSITIVITIES),
            "locked arms or same-head interventions changed")
    anchor_sd = float(kernel["warmup_audit"]["anchor_sd_fit"])
    require(np.isfinite(anchor_sd) and anchor_sd > 0, "invalid fit anchor scale")
    anchors = {p: array(readout["anchor_" + p], "frozen anchor", 1) for p in ("fit", "valid")}
    arms, features = {}, {}
    base = readout["arms"]["REL"]
    for arm in ARMS:
        value = readout["arms"][arm]
        weight = array(value["weights"], "readout weights", 1)
        dimensions = 64 if arm == "REL" else 66
        require(weight.shape == (dimensions,), "readout capacity changed")
        record = {"columns": dimensions, "weights_L2": float(np.linalg.norm(weight)),
                  "relative_weights_L2": float(np.linalg.norm(weight[:64])),
                  "extra_weights_L2": float(np.linalg.norm(weight[64:]))}
        for partition in ("fit", "valid"):
            x = array(value[partition + "_features"], "raw readout features", 2)
            require(x.shape == (len(anchors[partition]), dimensions), "readout row/dimension mismatch")
            np.testing.assert_array_equal(x[:, :64], base[partition + "_features"])
            mean = array(value["scaler"]["mean"], "fit scaler mean", 1)
            scale = array(value["scaler"]["scale"], "fit scaler scale", 1)
            require(mean.shape == scale.shape == (dimensions,) and (scale > 0).all(), "invalid fit standardizer")
            z = (x - mean) / scale
            stored_risk = array(value["fit_risk"] if partition == "fit" else value["risk"], "stored risk", 1)
            require(np.max(np.abs(stored_risk - (anchors[partition] + z @ weight))) <= 1e-12,
                    "saved readout risk replay changed")
            if partition == "fit":
                np.testing.assert_array_equal(mean, x.mean(0))
                sd = x.std(0, ddof=0)
                np.testing.assert_array_equal(scale, np.where(sd < 1e-8, 1., sd))
            features[(arm, partition)] = (x, z)
            variances = x.var(0, ddof=0)
            record[partition] = {"raw_column_variance": summary(variances),
                "raw_column_variance_below_1e_minus16_count": int((variances < 1e-16).sum()),
                "standardized_rank": int(np.linalg.matrix_rank(z)),
                "total_risk_correction": normalized_risk_summary(z @ weight, anchor_sd)}
            if dimensions == 66:
                record[partition]["extra_column_variance"] = {
                    "column_1": float(variances[-2]), "column_2": float(variances[-1])}
        arms[arm] = record
    dual = readout["arms"]["DUAL"]
    weight = array(dual["weights"], "DUAL weights", 1)
    absolute = {"standardized_coefficients": {"patch_evidence_mean": float(weight[-2]),
                                              "patch_evidence_population_SD": float(weight[-1])}}
    sensitivity, paired_correlation = {}, {}
    for partition in ("fit", "valid"):
        _, z = features[("DUAL", partition)]
        absolute[partition] = normalized_risk_summary(z[:, -2:] @ weight[-2:], anchor_sd)
    matched_risk = array(dual["risk"], "DUAL heldout risk", 1)
    for name in SENSITIVITIES:
        risk = array(readout["sensitivities"][name], "same-DUAL-head risk", 1)
        if name == "DUAL_ABS_OFF":
            z = features[("DUAL", "valid")][1].copy()
            z[:, -2:] = 0.
        else:
            arm = name.removeprefix("DUAL_")
            x = features[(arm, "valid")][0]
            z = (x - dual["scaler"]["mean"]) / dual["scaler"]["scale"]
        require(np.max(np.abs(risk - (anchors["valid"] + z @ weight))) <= 1e-12,
                "same-head intervention used another scaler or readout")
        sensitivity[name] = normalized_risk_summary(matched_risk - risk, anchor_sd)
    for arm in ("WRONG_O", "WRONG_C", "NO_C"):
        paired_correlation[arm] = {}
        for partition in ("fit", "valid"):
            actual = features[("DUAL", partition)][0][:, -2:]
            other = features[(arm, partition)][0][:, -2:]
            paired_correlation[arm][partition] = {
                "patch_evidence_mean": correlation(actual[:, 0], other[:, 0]),
                "patch_evidence_population_SD": correlation(actual[:, 1], other[:, 1])}
    return {"seed": seed, "fold": fold, "role": "inner", "fit_local_clinical_dimension": int(kernel["clinical_dim"]),
            "fit_anchor_population_SD": anchor_sd,
            "warmup": _warmup(kernel["warmup_audit"]), "arms": arms,
            "DUAL_absolute_only_risk": absolute, "same_DUAL_head_sensitivities": sensitivity,
            "absolute_feature_correlations": paired_correlation,
            "wrong_patient_controls": _pairing(readout["data_audit"]),
            "reference_counts": _reference(kernel, readout),
            "shared_relative_replayed": True, "scalers_replayed_fit_only": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root", type=Path, default=PRIVATE)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase23_cdei/MECHANISM_AUDIT.json")
    args = parser.parse_args()
    identity = source()
    config = cfg()
    matrix = [(int(s), int(f)) for f in config["folds"] for s in config["seeds"]]
    require(len(matrix) == len(set(matrix)) == 15, "mechanism audit requires all fifteen locked inner cells")
    # Validate all receipts/byte hashes before deserializing ANY model payload.
    directories = []
    for seed, fold in matrix:
        directory = args.private_root / "inner" / f"seed{seed}_fold{fold}"
        verify_done(directory, role="inner", seed=seed, fold=fold, identity=identity)
        directories.append((seed, fold, directory))
    cells = []
    for seed, fold, directory in directories:
        kernel = torch.load(directory / "kernel.pt", map_location="cpu", weights_only=False)
        readout = torch.load(directory / "readouts.pt", map_location="cpu", weights_only=False)
        cells.append(summarize_cell(kernel, readout, identity=identity, seed=seed, fold=fold))
    require(source() == identity, "source drift during read-only audit")
    record = {"schema": "phase23_inner_endpoint_free_mechanism_audit_v1", "status": "PASS",
              "source": identity, "diagnostic_script_sha256": sha(__file__), "cells_completed": len(cells),
              "new_predictor_fits": 0, "endpoint_loader_calls": 0, "endpoint_vector_reads": 0,
              "all_kernel_epochs_80": True, "all_readout_risks_replayed": True,
              "inner_heldout_is_selection_partition": True,
              "warmup_last10_fit_loss_improvement": summary([c["warmup"]["last10_fit_loss_improvement"] for c in cells]),
              "cells": cells, "warnings": [
                  "G0_FREEZES_FIT_TRAINED_ENCODERS_AND_COMPATIBILITY;NOT_END_TO_END_CDEI",
                  "FROZEN_J0_FIT_OFFSET_IS_IN_SAMPLE;NO_CROSSFITTED_TEACHER_OR_FREE_STACKING_COEFFICIENT",
                  "ALL_ABSOLUTE_CONTROLS_SHARE_OWN_WRONG_TOKEN_SUPPORT_INTERSECTION_AND_REFERENCE_EXCLUSION_UNION",
                  "TOKEN_INTERSECTION_DOES_NOT_EQUALIZE_WITHIN_TOKEN_GENE_SUPPORT_OR_PROVE_CAUSAL_MATCHING",
                  "SEPARATE_RNA_CNA_SUPPORT_CHANGED_PATIENT_COUNTS_NOT_PERSISTED;COMBINED_24_SLOT_COUNT_REPORTED",
                  "FIXED80_AND_TRAINING_LOSS_TRENDS_DO_NOT_ESTABLISH_GLOBAL_CONVERGENCE_OR_GENERALIZATION",
                  "COMPLETED_CELL_AUDIT_DOES_NOT_COUNT_SUPERSEDED_OR_FAILED_ATTEMPTS_AS_COMPLETED_FITS",
                  "ABSOLUTE_EVIDENCE_COMBINES_ROW_LEVEL_AND_DISPERSION;COEFFICIENTS_ARE_NOT_MEDICAL_TRUTH",
                  "SAME_HEAD_SENSITIVITY_AND_SEPARATELY_REFITTED_CONTROLS_ANSWER_DIFFERENT_QUESTIONS"]}
    atomic_json(args.output, record)
    print("PHASE23_MECHANISM_AUDIT", record["status"], len(cells), flush=True)


if __name__ == "__main__":
    main()

"""Read-only attribution of existing Phase15 scalar RNA risk corrections.

No raw RNA load, fitting, hyperparameter selection or bootstrap is performed.
Private risks/indices are used only in memory. New reports contain model-level
scalars and anonymous fold/seed summaries, never patient or gene coordinates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase10_trimodal.reference import metrics, uno_weights
from src.phase21_relative.evaluation import pair_transition_summary

EXPECTED_SOURCE = "0eb0e095b0f30287ec3793a382413674e0abe341cc35c27d89dc61120928bc5b"
SEEDS, FOLDS, GENES, ENDPOINTS = (31, 47, 83), (1, 2, 3, 4, 5), ("TOP", "HALL", "MATCH"), ("DSS", "PFI")


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def csv_file(path):
    with Path(path).open(newline="", encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def finite(value, name):
    array = np.asarray(value, float)
    require(np.isfinite(array).all(), name + " contains nonfinite values")
    return array


def agreement(actual, expected, name, *, tolerance=1e-12):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, name + " shape mismatch")
    error = float(np.max(np.abs(actual - expected), initial=0.))
    require(np.isfinite(error) and error <= tolerance, name + " replay changed")
    return error


def pairs(time, event):
    return np.where((event[:, None] == 1) & ((time[:, None] < time[None, :])
                    | ((time[:, None] == time[None, :]) & (event[None, :] == 0))))


def descriptive(values):
    values = finite(values, "summary")
    return {"n": len(values), "minimum": float(values.min()), "median": float(np.median(values)),
            "maximum": float(values.max()), "mean": float(values.mean())}


def verify_existing(private_root):
    """First qualify the historical source, receipt set and all ninety hashes."""
    report = ROOT / "reports/phase15_rna_complementarity"
    public = json_file(report / "PUBLIC_FILES.json")
    require(public.get("status") == "PASS", "historical public manifest incomplete")
    for row in public["files"]:
        path = ROOT / row["path"]
        require(path.is_file() and sha(path) == row["sha256"], "historical public bytes changed: " + row["path"])
    from src.phase15_complement.core import source
    require(source() == EXPECTED_SOURCE, "historical Phase15 training source changed")
    saved_audit = json_file(report / "INDEPENDENT_AUDIT.json")
    private_audit = json_file(private_root / "EVALUATION_DONE.json")
    require(saved_audit == private_audit and saved_audit.get("status") == "PASS"
            and saved_audit.get("source_hash") == EXPECTED_SOURCE, "historical evaluation receipt differs")
    hashes = {row["pipeline"]: row for row in csv_file(report / "PRIVATE_ARTIFACT_HASHES.csv")}
    require(len(hashes) == 90, "historical ninety-artifact inventory incomplete")
    records = []
    for endpoint in ENDPOINTS:
        for gene in GENES:
            for seed in SEEDS:
                for fold in FOLDS:
                    name = f"{endpoint}_{gene}_seed{seed}_fold{fold}"
                    receipt_path = private_root / "cells" / (name + ".json")
                    artifact = private_root / "artifacts" / (name + ".pt")
                    receipt = json_file(receipt_path)
                    for key, expected in (("status", "COMPLETE"), ("source_hash", EXPECTED_SOURCE),
                                          ("endpoint", endpoint), ("gene", gene), ("seed", seed), ("fold", fold)):
                        require(receipt.get(key) == expected, "old cell metadata changed: " + key)
                    require(receipt.get("outer_test_selection") is False, "old cell admits outer selection")
                    require(receipt["artifact_sha256"] == hashes[name]["artifact_sha256"] == sha(artifact),
                            "historical private artifact SHA mismatch")
                    require(artifact.stat().st_size == int(hashes[name]["artifact_bytes"]), "artifact byte count changed")
                    records.append((endpoint, gene, seed, fold, receipt, artifact))
    return records, {"public_manifest_files_checked": len(public["files"]), "private_artifacts_checked": 90,
                     "private_artifact_bytes_checked": sum(row[-1].stat().st_size for row in records),
                     "historical_source": EXPECTED_SOURCE, "historical_independent_audit": saved_audit,
                     "historical_teacher_perturbation_audit": json_file(report / "MECHANISM_AUDIT.json")}


def audit(private_root, report_dir):
    records, source_audit = verify_existing(private_root)
    artifacts = []
    labels = {ep: {"time": np.full(512, np.nan), "event": np.full(512, np.nan)} for ep in ENDPOINTS}
    for ep, gene, seed, fold, receipt, path in records:
        z = torch.load(path, map_location="cpu", weights_only=False)
        require(z["source_hash"] == EXPECTED_SOURCE, "serialized source changed")
        test = np.asarray(z["partitions"]["outer_test"], int)
        for key, saved in (("time", "test_time"), ("event", "test_event")):
            value = finite(z[saved], saved)
            require(value.shape == test.shape, "saved endpoint shape changed")
            previous = labels[ep][key][test]
            require(np.isnan(previous).all() or np.array_equal(previous, value), "historical endpoint differs across models")
            labels[ep][key][test] = value
        artifacts.append((ep, gene, seed, fold, receipt, z))
    for ep, expected_n, expected_events in (("DSS", 512, 65), ("PFI", 510, 129)):
        available = np.isfinite(labels[ep]["time"]) & np.isfinite(labels[ep]["event"])
        require(int(available.sum()) == expected_n and int(labels[ep]["event"][available].sum()) == expected_events,
                "historical endpoint population differs")
    row_lookup = {(row["endpoint"], row["gene"], int(row["seed"]), int(row["fold"]), row["model"]): row
                  for row in csv_file(ROOT / "reports/phase15_rna_complementarity/FOLD_METRICS.csv")}
    model_rows, coefficient_rows, transition_rows, teachers = [], [], [], {}
    full_rna_reference, equal_full_rna_comparisons = {}, 0
    max_risk_error = max_gradient = max_metric_error = 0.
    for ep, gene, seed, fold, receipt, z in artifacts:
        time, event = labels[ep]["time"], labels[ep]["event"]
        current = {"endpoint": ep, "gene": gene, "seed": seed, "fold": fold}
        for role in ("inner", "outer"):
            anchor_path = private_root / "anchors" / f"{ep}_seed{seed}_fold{fold}_{role}.pt"
            a = torch.load(anchor_path, map_location="cpu", weights_only=False)
            require(a["source_hash"] == EXPECTED_SOURCE, "anchor source changed")
            fit, test = np.asarray(a["fit"], int), np.asarray(a["test"], int)
            require(np.array_equal(fit, z["partitions"]["inner_train" if role == "inner" else "outer_train"])
                    and np.array_equal(test, z["partitions"]["inner_val" if role == "inner" else "outer_test"]),
                    "anchor/data partition mismatch")
            require(not np.intersect1d(fit, test).size, "fit/test overlap")
            rr = z["RNA"][role]
            require(np.array_equal(rr["preprocessing"]["fit_indices"], fit), "full RNA scaling scope mismatch")
            held_parts = []
            for k, (teacher, rteacher) in enumerate(zip(a["teachers"], rr["teachers"])):
                train, held = np.asarray(teacher["train"], int), np.asarray(teacher["held"], int)
                require(not np.intersect1d(train, held).size and np.array_equal(np.union1d(train, held), np.sort(fit)),
                        "teacher train/held is not exact fit partition")
                require(not np.intersect1d(train, test).size, "teacher trained on test")
                require(np.array_equal(rteacher["train"], train) and np.array_equal(rteacher["held"], held),
                        "RNA/anchor teacher boundaries differ")
                require(np.array_equal(teacher["preprocessing"]["fit_indices"], train)
                        and np.array_equal(rteacher["preprocessing"]["fit_indices"], train), "teacher scaling fit mismatch")
                require(np.all(np.asarray(a["group"])[held] == k), "teacher group labels changed")
                require(teacher["lambda"] in (10., 1., .1) and set(teacher["selection"]) == {10., 1., .1},
                        "teacher nested lambda inventory changed")
                require(np.asarray(rteacher["w"]).shape == (512,), "RNA teacher dimension changed")
                require(np.isfinite(rteacher["risk_sd"]) and rteacher["risk_sd"] > 0, "RNA teacher scaling invalid")
                held_parts.append(held)
            require(len(held_parts) == 3 and np.array_equal(np.sort(np.concatenate(held_parts)), np.sort(fit)),
                    "RNA/anchor teacher OOF not exact once-only cover")
            raw_i, raw_j = pairs(time[fit], event[fit])
            mask = np.asarray(a["group"])[fit[raw_i]] == np.asarray(a["group"])[fit[raw_j]]
            ii, jj = raw_i[mask], raw_j[mask]
            require(np.array_equal(ii, z["pairs"][role]["i"]) and np.array_equal(jj, z["pairs"][role]["j"]),
                    "same-teacher pair mask differs")
            for k, teacher in enumerate(a["teachers"]):
                use = np.asarray(a["group"])[fit[ii]] == k
                owners = np.union1d(fit[ii[use]], fit[jj[use]])
                require(not np.intersect1d(owners, teacher["train"]).size
                        and np.isin(owners, teacher["held"]).all(), "both pair labels not excluded")
            margin = finite(np.asarray(a["oof"])[fit[ii]] - np.asarray(a["oof"])[fit[jj]], "pair offset")
            agreement(margin, z["pairs"][role]["margin"], "saved margin")
            scalar_fit = finite(np.asarray(rr["oof"])[fit], "crossfit scalar")
            scalar_test = finite(rr["test"], "final scalar")
            if role == "outer":
                signature_key = ep, gene, fold
                signature = (np.asarray(z["indices"][role]), np.asarray(rr["w"]), scalar_test,
                             np.asarray(rr["preprocessing"]["median"]), np.asarray(rr["preprocessing"]["mean"]),
                             np.asarray(rr["preprocessing"]["sd"]), np.asarray([rr["risk_mean"], rr["risk_sd"]]))
                if signature_key in full_rna_reference:
                    require(all(np.array_equal(old, new) for old, new in zip(full_rna_reference[signature_key], signature)),
                            "same-fold final RNA risk differs across crossfit seeds")
                    equal_full_rna_comparisons += 1
                else:
                    full_rna_reference[signature_key] = signature
            difference = scalar_fit[ii] - scalar_fit[jj]
            for key, coef in z["coefficients"].items():
                if not key.startswith(role + "_SCALAR_") or key.endswith("_OFF"):
                    continue
                lam = float(key.rsplit("_", 1)[1])
                beta = finite(coef, "scalar beta")
                require(beta.shape == (1,), "SCALAR is not one-dimensional")
                beta = float(beta[0])
                gradient = float(np.mean(-expit(-(margin + difference * beta)) * difference) + lam * beta)
                max_gradient = max(max_gradient, abs(gradient))
                require(abs(gradient) <= 2.5e-5, "saved scalar coefficient is nonstationary")
                derivative_at_zero = float(np.mean(-expit(-margin) * difference))
                hessian = float(np.mean(expit(margin + difference * beta) * expit(-(margin + difference * beta))
                                                * difference ** 2) + lam)
                coefficient_rows.append({**current, "role": role, "lambda": lam, "beta": beta,
                    "gradient": gradient, "hessian": hessian, "derivative_at_zero": derivative_at_zero,
                    "direction_agrees_with_convex_gradient": bool(beta == 0 or beta * derivative_at_zero <= 0),
                    "anchor_selected_lambda": float(a["lambda"]), "fit_patients": len(fit),
                    "scalar_reference": "train-score-standardized RNA Cox risk; positive is earlier event"})
            teachers[ep, seed, fold, role] = {"source_sha256": sha(anchor_path), "teacher_blocks": 3,
                                             "both_pair_labels_excluded": True}
            if role != "outer":
                continue
            base = finite(a["test_risk"], "CS test risk")
            selected_lambda = z["selection"]["SCALAR"]
            candidates = z["selection"]["SCALAR_rows"]
            best = max(row["inner_Harrell"] for row in candidates)
            tied = [row for row in candidates if abs(row["inner_Harrell"] - best) <= 1e-12]
            winner = next((row for row in tied if row["lambda"] == "OFF"), tied[0])["lambda"]
            require(winner == selected_lambda, "SCALAR inner selection differs")
            w = uno_weights(time[fit], event[fit], time[test], event[test], tau=60.)
            for model, lam in (("SCALAR", selected_lambda), ("SCALAR_FIXED1", 1.)):
                beta = 0. if lam == "OFF" else float(z["coefficients"]["outer_SCALAR_" + str(lam)][0])
                correction = scalar_test * beta
                prediction = base + correction
                max_risk_error = max(max_risk_error, agreement(prediction, z["risks"][model], model))
                value = metrics(time[test], event[test], prediction, np.full(len(test), fold), w)
                old = row_lookup[ep, gene, seed, fold, model]
                for metric in ("Harrell", "Uno60"):
                    error = abs(value[metric] - float(old[metric])); max_metric_error = max(max_metric_error, error)
                    require(error <= 1e-12, "scalar metric differs from historical independent report")
                model_rows.append({**current, "model": model, "selected_lambda": lam, "beta": beta,
                    "anchor_selected_lambda": float(a["lambda"]),
                    "correction_mean": float(correction.mean()), "correction_SD": float(correction.std()),
                    "anchor_SD": float(base.std()), "correction_SD_over_anchor_SD": float(correction.std() / base.std()),
                    "absolute_correction_mean": float(np.abs(correction).mean()),
                    "scalar_test_SD": float(scalar_test.std()),
                    "anchor_scalar_correlation": float(np.corrcoef(base, scalar_test)[0, 1]), **value})
                transition = pair_transition_summary(time[test], event[test], base, prediction,
                                                     folds=np.full(len(test), fold), ipcw=w)
                for row in transition["rows"]:
                    transition_rows.append({**current, "model": model, "metric": row["metric"],
                        "denominator": row["denominator"], **{name: row[name] for name in ("C0", "C1", "repair", "damage", "net")}})
    summary, seed_metrics, transition_summary = [], [], []
    for ep in ENDPOINTS:
        for gene in GENES:
            for model in ("SCALAR", "SCALAR_FIXED1"):
                selected = [r for r in model_rows if (r["endpoint"], r["gene"], r["model"]) == (ep, gene, model)]
                for seed in SEEDS:
                    fold_rows = [r for r in selected if r["seed"] == seed]
                    require(len(fold_rows) == 5, "five folds required per seed")
                    seed_metrics.append({"endpoint": ep, "gene": gene, "model": model, "seed": seed,
                        "Harrell": sum(r["Harrell_num"] for r in fold_rows) / sum(r["Harrell_den"] for r in fold_rows),
                        "Uno60": sum(r["Uno_num"] for r in fold_rows) / sum(r["Uno_den"] for r in fold_rows)})
                summary.append({"endpoint": ep, "gene": gene, "model": model,
                    "Harrell": float(np.mean([r["Harrell"] for r in seed_metrics if (r["endpoint"], r["gene"], r["model"]) == (ep, gene, model)])),
                    "Uno60": float(np.mean([r["Uno60"] for r in seed_metrics if (r["endpoint"], r["gene"], r["model"]) == (ep, gene, model)])),
                    "OFF_cells": sum(r["selected_lambda"] == "OFF" for r in selected),
                    "positive_beta_cells": sum(r["beta"] > 0 for r in selected),
                    "negative_beta_cells": sum(r["beta"] < 0 for r in selected),
                    "beta": descriptive([r["beta"] for r in selected]),
                    "correction_SD_over_anchor_SD": descriptive([r["correction_SD_over_anchor_SD"] for r in selected])})
                for metric in ("Harrell", "Uno60"):
                    per_seed = []
                    for seed in SEEDS:
                        rows = [r for r in transition_rows if (r["endpoint"], r["gene"], r["model"], r["metric"], r["seed"])
                                == (ep, gene, model, metric, seed)]
                        mass = sum(r["denominator"] for r in rows)
                        per_seed.append({name: sum(r[name] * r["denominator"] for r in rows) / mass
                                         for name in ("C0", "C1", "repair", "damage", "net")})
                    transition_summary.append({"endpoint": ep, "gene": gene, "model": model, "metric": metric,
                        **{name: float(np.mean([r[name] for r in per_seed])) for name in ("C0", "C1", "repair", "damage", "net")}})
    focus = [r for r in model_rows if (r["endpoint"], r["gene"]) == ("DSS", "HALL")]
    historical_intervals = [{key: (float(row[key]) if key in ("delta", "CI_low", "CI_high") else
                                  int(row[key]) if key == "positive_seeds" else row[key])
                             for key in ("comparison", "metric", "delta", "CI_low", "CI_high", "positive_seeds", "primary")}
                            for row in csv_file(ROOT / "reports/phase15_rna_complementarity/PAIRED_COMPARISONS.csv")
                            if row["endpoint"] == "DSS" and row["gene"] == "HALL"
                            and row["comparison"] in ("SCALAR-CS", "SCALAR_FIXED1-CS")]
    report = {"status": "PASS_TARGETED_READ_ONLY_SCALAR_ATTRIBUTION", "scope": "post hoc audit of already evaluated Phase15; no new validation/confirmation cohort",
        "source_audit": source_audit, "new_model_fits": 0, "new_bootstrap_draws": 0, "raw_RNA_loaded": False,
        "patient_or_gene_arrays_public": False, "scalar_parameter_count": 1,
        "risk_definition": "frozen Clinical+S32 anchor (coefficient one) + beta * train-score-standardized RNA Cox risk",
        "semantic_or_trilinear_interaction_claimed": False,
        "strict_label_crossfit": "both patients share one held-out teacher; neither endpoint label enters that teacher model/penalty selection; partitions were endpoint-stratified",
        "strict_all_preprocessing_teacher_local": False,
        "feature_scope_limitation": "gene universe fixed by pipeline-train unsupervised expression selection; teacher-held covariates can affect that universe; scaling/imputation and fitted RNA risk are teacher-train-only",
        "historical_teacher_label_perturbation_scope": "three fixed DSS seed31 fold1 inner anchor teachers, not all ninety RNA teachers and not split regeneration",
        "reference_risk_scaling": "each RNA teacher standardizes by its own training RNA risk mean/SD; outer RNA refit uses whole outer-train mean/SD; coordinates are one risk axis, not latent embeddings",
        "audit": {"max_scalar_risk_replay_error": max_risk_error, "max_scalar_metric_error": max_metric_error,
                  "max_independent_scalar_gradient": max_gradient, "teacher_anchor_artifacts_checked": len(teachers),
                  "identical_full_RNA_axis_across_seed_comparisons": equal_full_rna_comparisons,
                  "both_pair_labels_excluded": True, "selection_replayed_without_refitting": True,
                  "independent_raw_feature_encoder_replay_claimed": False},
        "summaries": summary, "DSS_HALL_fold_decomposition": focus,
        "historical_DSS_HALL_paired_intervals_reused": historical_intervals,
        "DSS_HALL_coefficient_diagnostics": [r for r in coefficient_rows if (r["endpoint"], r["gene"]) == ("DSS", "HALL")],
        "seed_metrics": seed_metrics, "repair_damage_summary": transition_summary,
        "uncertainty": "reuse historical fixed-model CIs; no post hoc new bootstrap; historical intervals do not cover repeated-research or model-selection uncertainty"}
    write(report_dir / "SCALAR_ATTRIBUTION.json", report)
    focus_summary = [r for r in summary if (r["endpoint"], r["gene"]) == ("DSS", "HALL")]
    print(json.dumps({"status": report["status"], "audit": report["audit"], "DSS_HALL": focus_summary,
                      "DSS_HALL_repair_damage": [r for r in transition_summary if (r["endpoint"], r["gene"]) == ("DSS", "HALL")]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=Path("/root/phase15_complement_private"))
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/phase25_a")
    args = parser.parse_args()
    audit(args.private_root, args.report_dir)


if __name__ == "__main__":
    main()

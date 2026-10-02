from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from src.phase_m2.common import (
    atomic_csv, atomic_json, censoring_survival, harrell_c,
    lookup_censor_survival, uno_c_index,
)
from src.phase_m2.data import load_raw_inputs

SEEDS = (20261012, 20261013, 20261014, 20261015, 20261016)
RESIDUALS = ("RP", "RO", "RPO")
VARIANTS = ("A0", "C0-Full", "B0", *RESIDUALS)
COMPARISONS = (("RP", "A0"), ("RO", "A0"), ("RPO", "A0"), ("A0", "C0-Full"), ("RPO", "B0"))
BASE_SHA = "748d0273f78a778a9b615872060720e02d3def36"


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_offsets(rows, folds, seeds=SEEDS, tolerance=1e-7):
    """Fail closed on duplicate, missing, nonfinite or unmatched anchors."""
    cells = {}
    for row in rows:
        if row.get("diagnostic_level") == "summary" or row.get("seed") == "ALL":
            continue
        variant, seed, fold = row["variant"], int(row["seed"]), int(row["outer_fold"])
        if variant not in RESIDUALS or seed not in seeds or fold not in range(1, 6):
            raise ValueError("Unexpected residual diagnostic cell")
        patient = int(row["patient_index"])
        key = variant, seed, fold
        cell = cells.setdefault(key, {})
        if patient in cell:
            raise ValueError("Duplicate clinical offset")
        value = float(row["clinical_offset"])
        if not np.isfinite(value):
            raise ValueError("Nonfinite clinical offset")
        cell[patient] = value
    offsets, audits = {}, []
    for seed in seeds:
        for fold in range(1, 6):
            idx = np.flatnonzero(folds == fold)
            values = []
            for variant in RESIDUALS:
                cell = cells.get((variant, seed, fold), {})
                if set(cell) != set(idx):
                    raise ValueError("Clinical offset patient coverage mismatch")
                values.append(np.array([cell[int(i)] for i in idx], dtype=np.float64))
            max_diff = float(max(np.abs(values[0] - values[1]).max(),
                                 np.abs(values[0] - values[2]).max(),
                                 np.abs(values[1] - values[2]).max()))
            if max_diff > tolerance:
                raise ValueError(f"Anchor mismatch seed={seed} fold={fold} max_abs_diff={max_diff}")
            offsets[seed, fold] = values[0].copy()
            audits.append({"seed": seed, "outer_fold": fold, "n_test": len(idx),
                           "max_abs_diff": max_diff, "status": "PASS"})
    return offsets, audits


def bootstrap_counts(n, reps, seed):
    draws = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    counts = np.zeros((reps, n), dtype=np.float64)
    np.add.at(counts, (np.arange(reps)[:, None], draws), 1)
    return counts


def weighted_bootstrap_metrics(times, events, predictions, counts):
    """Exact count representation of paired resampling, including duplicates.

    Global Uno uses the unchanged M8-P0 diagnostic convention: censoring KM
    estimated from the original complete OOF cohort, held fixed per replicate.
    Fold Uno is calculated separately with the outer-training KM.
    """
    times, events = np.asarray(times), np.asarray(events)
    event_idx = np.flatnonzero(events == 1)
    pairs = (times[None, :] > times[event_idx, None]).astype(float)
    km_t, km_g = censoring_survival(times, events)
    g = lookup_censor_survival(times[event_idx], km_t, km_g)
    weights = np.zeros(len(g))
    valid = g > 1e-8
    weights[valid] = 1.0 / g[valid] ** 2
    patient_multiplicity = counts[:, event_idx]
    sampled_pairs = (counts @ pairs.T) * patient_multiplicity
    denom_h = sampled_pairs.sum(axis=1)
    denom_u = (sampled_pairs * weights).sum(axis=1)
    if np.any(denom_h <= 0) or np.any(denom_u <= 0):
        raise ValueError("Bootstrap replicate has no comparable pairs")
    result = {}
    for name, values in predictions.items():
        risk = np.asarray(values)
        concordance = (risk[event_idx, None] > risk[None, :]).astype(float)
        concordance += 0.5 * (risk[event_idx, None] == risk[None, :])
        numerators = (counts @ (pairs * concordance).T) * patient_multiplicity
        result[name] = (numerators.sum(axis=1) / denom_h,
                        (numerators * weights).sum(axis=1) / denom_u)
    return result


def classify(mean_h, mean_u, positive_h, positive_u, lower_h, lower_u):
    if not (mean_h > 0 and mean_u > 0 and positive_h >= 4 and positive_u >= 4):
        return "RESIDUAL_SIGNAL_NOT_CONFIRMED"
    if lower_h >= 0 or lower_u >= 0:
        return "RESIDUAL_SIGNAL_CONFIRMED"
    return "RESIDUAL_SIGNAL_DIRECTIONAL"


def run(args):
    repo, source, workers, out = map(Path, (args.repo_root, args.source_report, args.workers, args.output))
    out.mkdir(parents=True, exist_ok=True)
    raw = load_raw_inputs(Path(args.data_root), repo)
    if len(raw.patient_ids) != 512 or int(raw.dss_event.sum()) != 65:
        raise ValueError("Fixed cohort must be 512/65")
    source_paths = sorted(source.glob("*"))
    source_hashes = {p.name: digest(p) for p in source_paths if p.is_file()}
    diagnostic_rows = read_csv(source / "residual_diagnostics.csv")
    offsets, checks = extract_offsets(diagnostic_rows, raw.outer_folds)
    audit = {"status": "PASS", "tolerance": 1e-7, "cells_checked": len(checks),
             "max_abs_diff": max(r["max_abs_diff"] for r in checks), "rows": checks,
             "full_coverage": True, "duplicate_offsets": 0, "finite_offsets": True,
             "anchor_definition": "A0 equals saved clinical_offset; no refit"}
    atomic_json(out / "anchor_consistency_audit.json", audit)
    if args.audit_only:
        print(json.dumps(audit))
        return
    predictions, prediction_hashes = {}, {}
    oldfold = {(r["variant"], int(r["seed"]), int(r["outer_fold"])): r
               for r in read_csv(source / "fold_metrics.csv")}
    folds = []
    for seed in SEEDS:
        for fold in range(1, 6):
            idx = np.flatnonzero(raw.outer_folds == fold)
            train_idx = np.flatnonzero(raw.outer_folds != fold)
            for variant in VARIANTS:
                if variant == "A0":
                    risk = offsets[seed, fold]
                else:
                    p = workers / f"seed_{seed}" / "cells" / f"{variant}_{seed}_{fold}.npy"
                    prediction_hashes[str(p.relative_to(workers))] = digest(p)
                    risk = np.load(p)
                if risk.shape != (len(idx),) or not np.isfinite(risk).all():
                    raise ValueError("Prediction shape or finiteness mismatch")
                predictions[variant, seed, fold] = risk
                h = harrell_c(raw.dss_time[idx], raw.dss_event[idx], risk)
                u, _, _ = uno_c_index(raw.dss_time[idx], raw.dss_event[idx], risk,
                                     raw.dss_time[train_idx], raw.dss_event[train_idx])
                if variant != "A0":
                    row = oldfold[variant, seed, fold]
                    if abs(h - float(row["harrell_c"])) > 1e-10 or abs(u - float(row["uno_c"])) > 1e-10:
                        raise ValueError("Stored predictions do not reproduce source fold metrics")
                folds.append({"variant": variant, "seed": seed, "outer_fold": fold,
                              "n_test": len(idx), "events_test": int(raw.dss_event[idx].sum()),
                              "harrell_c": h, "uno_c": u})
    atomic_csv(out / "anchor_only_fold_metrics.csv", [r for r in folds if r["variant"] == "A0"])
    seedrows, variances = [], []
    for variant in VARIANTS:
        for seed in SEEDS:
            z = [r for r in folds if r["variant"] == variant and r["seed"] == seed]
            h, u = [np.array([r[k] for r in z]) for k in ("harrell_c", "uno_c")]
            seedrows.append({"variant": variant, "seed": seed, "n_patients": 512, "events": 65,
                             "mean_harrell": h.mean(), "mean_uno": u.mean(),
                             "worst_fold_harrell": h.min(), "worst_fold_uno": u.min(),
                             "within_seed_fold_variance_H": h.var(ddof=1),
                             "within_seed_fold_variance_U": u.var(ddof=1)})
        z = [r for r in seedrows if r["variant"] == variant]
        variances.append({"variant": variant,
                          "mean_harrell": np.mean([r["mean_harrell"] for r in z]),
                          "mean_uno": np.mean([r["mean_uno"] for r in z]),
                          "between_seed_variance_H": np.var([r["mean_harrell"] for r in z], ddof=1),
                          "between_seed_variance_U": np.var([r["mean_uno"] for r in z], ddof=1),
                          "within_seed_fold_variance_H": np.mean([r["within_seed_fold_variance_H"] for r in z]),
                          "within_seed_fold_variance_U": np.mean([r["within_seed_fold_variance_U"] for r in z]),
                          "worst_fold_H": min(r["worst_fold_harrell"] for r in z),
                          "worst_fold_U": min(r["worst_fold_uno"] for r in z)})
    atomic_csv(out / "anchor_only_seed_metrics.csv", [r for r in seedrows if r["variant"] == "A0"])
    atomic_csv(out / "variance_decomposition.csv", variances)
    delta_rows = []
    for a, b in COMPARISONS:
        for seed in SEEDS:
            for fold in range(1, 6):
                x = next(r for r in folds if (r["variant"], r["seed"], r["outer_fold"]) == (a, seed, fold))
                y = next(r for r in folds if (r["variant"], r["seed"], r["outer_fold"]) == (b, seed, fold))
                delta_rows.append({"level": "fold", "comparison": f"{a}_vs_{b}", "seed": seed,
                                   "outer_fold": fold, "delta_harrell": x["harrell_c"] - y["harrell_c"],
                                   "delta_uno": x["uno_c"] - y["uno_c"]})
            z = [r for r in delta_rows if r["level"] == "fold" and r["comparison"] == f"{a}_vs_{b}" and r["seed"] == seed]
            delta_rows.append({"level": "seed", "comparison": f"{a}_vs_{b}", "seed": seed,
                               "outer_fold": "MEAN", "delta_harrell": np.mean([r["delta_harrell"] for r in z]),
                               "delta_uno": np.mean([r["delta_uno"] for r in z])})
        z = [r for r in delta_rows if r["level"] == "seed" and r["comparison"] == f"{a}_vs_{b}"]
        delta_rows.append({"level": "summary", "comparison": f"{a}_vs_{b}", "seed": "ALL",
                           "outer_fold": "ALL", "delta_harrell": np.mean([r["delta_harrell"] for r in z]),
                           "delta_uno": np.mean([r["delta_uno"] for r in z]),
                           "positive_seeds_H": sum(r["delta_harrell"] > 0 for r in z),
                           "positive_seeds_U": sum(r["delta_uno"] > 0 for r in z)})
    atomic_csv(out / "paired_deltas.csv", delta_rows)
    # Fill global OOF in original patient order; fold predictions retain exact ordering.
    boot = []
    for seed in SEEDS:
        oof = {}
        for variant in VARIANTS:
            risk = np.full(512, np.nan)
            for fold in range(1, 6):
                risk[raw.outer_folds == fold] = predictions[variant, seed, fold]
            oof[variant] = risk
        counts = bootstrap_counts(512, 2000, 8502026 + seed)
        metrics = weighted_bootstrap_metrics(raw.dss_time, raw.dss_event, oof, counts)
        for a, b in COMPARISONS:
            h, u = metrics[a][0] - metrics[b][0], metrics[a][1] - metrics[b][1]
            boot.append({"comparison": f"{a}_vs_{b}", "seed": seed, "n_reps": 2000,
                         "mean_delta_harrell": h.mean(), "ci95_low_harrell": np.quantile(h, .025),
                         "ci95_high_harrell": np.quantile(h, .975), "mean_delta_uno": u.mean(),
                         "ci95_low_uno": np.quantile(u, .025), "ci95_high_uno": np.quantile(u, .975)})
        print(f"seed={seed} bootstrap completed", flush=True)
    atomic_csv(out / "paired_bootstrap.csv", boot)
    decisions = {}
    for variant in RESIDUALS:
        s = next(r for r in delta_rows if r["level"] == "summary" and r["comparison"] == f"{variant}_vs_A0")
        z = [r for r in boot if r["comparison"] == f"{variant}_vs_A0"]
        decisions[variant] = classify(s["delta_harrell"], s["delta_uno"], s["positive_seeds_H"],
                                     s["positive_seeds_U"], min(r["ci95_low_harrell"] for r in z),
                                     min(r["ci95_low_uno"] for r in z))
    final = {"RESIDUAL_SIGNAL_CONFIRMED": "A", "RESIDUAL_SIGNAL_DIRECTIONAL": "B",
             "RESIDUAL_SIGNAL_NOT_CONFIRMED": "C"}[decisions["RPO"]]
    # No mutation of the source report or frozen predictions is permitted.
    if any(digest(source / name) != value for name, value in source_hashes.items()):
        raise ValueError("Source report changed during audit")
    if any(digest(workers / name) != value for name, value in prediction_hashes.items()):
        raise ValueError("Frozen prediction changed during audit")
    manifest = {"phase": "M8-P0R", "status": "complete", "base_sha": BASE_SHA,
                "execution_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                "n_patients": 512, "dss_events": 65, "seeds": list(SEEDS), "outer_folds": 5,
                "model_training_runs": 0, "predictions_modified": False,
                "source_report_sha256": source_hashes, "prediction_sha256": prediction_hashes,
                "source_metrics_reproduced": True, "anchor_checks": "PASS",
                "bootstrap_reps_per_seed": 2000, "bootstrap_seed_rule": "8502026 + seed",
                "bootstrap_unit": "patient; identical resamples for all comparisons within a seed",
                "uno_bootstrap_KM": "fixed original 512-patient OOF KM, inherited M8-P0 diagnostic",
                "decision_CI_rule": "conservative: all five per-seed lower bounds >=0 for at least one metric",
                "final_decision_rule": "RPO combined additive primary comparison; no best(RP,RO,RPO) selection",
                "variant_decisions": decisions, "final_decision": final,
                "source_files_unchanged": True, "private_patient_arrays_not_committed": True}
    atomic_json(out / "run_manifest.json", manifest)
    report(out, variances, delta_rows, boot, decisions, final, audit)
    print(json.dumps({"final_decision": final, "variants": decisions, "anchor_max_abs_diff": audit["max_abs_diff"]}))


def report(out, variances, deltas, boot, decisions, final, audit):
    lines = ["# Phase M8-P0R — Exact Anchor Comparator Repair", "", "## Protocol and provenance", "",
             "512 patients / 65 DSS events; frozen 5 outer folds; seeds 20261012–20261016.",
             "No training, refit, repartitioning, parameter changes or prediction modification.",
             f"Exact anchor agreement across RP/RO/RPO: **PASS**, max absolute difference={audit['max_abs_diff']:.9g}.",
             "A0 equals the saved clinical_offset; source fold scores were independently reproduced from frozen predictions.",
             "", "## Anchor and variance", "",
             "| Model | Harrell | Uno | Between seed H/U variance | Within seed H/U variance | Worst fold H/U |",
             "|---|---:|---:|---|---|---|"]
    for r in variances:
        lines.append(f"| {r['variant']} | {r['mean_harrell']:.6f} | {r['mean_uno']:.6f} | "
                     f"{r['between_seed_variance_H']:.6g}/{r['between_seed_variance_U']:.6g} | "
                     f"{r['within_seed_fold_variance_H']:.6g}/{r['within_seed_fold_variance_U']:.6g} | "
                     f"{r['worst_fold_H']:.6f}/{r['worst_fold_U']:.6f} |")
    lines += ["", "## Matched comparisons", "",
              "Fold metrics are primary; positive seed counts refer to the five-fold mean differences.", "",
              "| Comparison | Mean delta Harrell | Mean delta Uno | Positive seeds H/U | Decision |",
              "|---|---:|---:|---|---|"]
    for r in deltas:
        if r["level"] == "summary":
            v = r["comparison"].split("_vs_")[0] if r["comparison"].endswith("_vs_A0") else ""
            lines.append(f"| {r['comparison']} | {r['delta_harrell']:+.6f} | {r['delta_uno']:+.6f} | "
                         f"{r['positive_seeds_H']}/5, {r['positive_seeds_U']}/5 | {decisions.get(v, 'context')} |")
    lines += ["", "## Paired patient bootstrap", "",
              "2,000 paired patient resamples per seed. Global OOF is diagnostic, not model selection.",
              "Global Uno uses the unchanged complete-OOF censoring KM; fold Uno uses each outer-train KM.",
              "The decision uses the most conservative lower bound over the five per-seed intervals.", "",
              "| Comparison | Seed | Bootstrap delta H | H 95% CI | Bootstrap delta U | U 95% CI |",
              "|---|---:|---:|---|---:|---|"]
    for r in boot:
        lines.append(f"| {r['comparison']} | {r['seed']} | {r['mean_delta_harrell']:+.6f} | "
                     f"[{r['ci95_low_harrell']:+.6f}, {r['ci95_high_harrell']:+.6f}] | "
                     f"{r['mean_delta_uno']:+.6f} | [{r['ci95_low_uno']:+.6f}, {r['ci95_high_uno']:+.6f}] |")
    acomp = next(r for r in deltas if r["level"] == "summary" and r["comparison"] == "A0_vs_C0-Full")
    lines += ["", "## Interpretation", "",
              f"A0 vs C0-Full mean differences: Harrell {acomp['delta_harrell']:+.6f}, Uno {acomp['delta_uno']:+.6f}. "
              "The previous residual-vs-C0 comparisons mixed changes in Clinical anchor quality with modality corrections. "
              "Only residual-vs-A0 tests estimate the change relative to the exact frozen Clinical risk.",
              "An apparent change from the original Decision C must be justified by the matched seed directions and intervals, "
              "not by selecting the best residual variant.", "", "## Advantages and limitations", "",
              "Advantage: identical Clinical offsets and frozen predictions isolate the residual correction without new training.",
              "Limitations: only 65 events; five-seed intervals assess the fixed cohort, not external generalization. "
              "This repair does not certify unrelated M8-P0 model or preprocessing choices. "
              "The inherited M8-P0 missingness census remains 35 additional DSS-labelled patients / 4 observed events.",
              "Confidence: high for frozen-anchor identity and numerical reproduction; medium for generalizable incremental information.",
              "", "## Final decision", "", f"**{final}. {decisions['RPO']}**", "",
              "Stop Clinical-residual architecture development; retain CommonOnly/Clinical baselines and position missing-modality work "
              "as coverage/robustness." if final == "C" else
              "A subsequent shrinkage residual model is justified for a separate phase; do not add complex cross-attention. "
              "No next-phase training was performed."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True)
    p.add_argument("--source-report", required=True)
    p.add_argument("--workers", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--audit-only", action="store_true")
    run(p.parse_args())

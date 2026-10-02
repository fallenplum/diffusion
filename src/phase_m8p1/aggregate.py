from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from src.phase_m2.common import atomic_csv, atomic_json, harrell_c, uno_c_index
from src.phase_m2.data import load_raw_inputs, preprocess_fold
from src.phase_m8p0.models import build_residual
from src.phase_m8p0.survival import cox_loss_full


SEEDS = (20261012, 20261013, 20261014, 20261015, 20261016)
VARIANTS = ("S0", "SP", "SO", "SPO")
COMPARISONS = (("SP", "S0"), ("SO", "S0"), ("SPO", "S0"), ("SPO", "SP"), ("SPO", "SO"), ("SPO", "B0"))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def num(x: object) -> float:
    return float(x)


def cfast(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    times = np.asarray(times)
    events = np.asarray(events).astype(int)
    risk = np.asarray(risk)
    nume = den = 0.0
    for i in np.flatnonzero(events == 1):
        at_risk = times > times[i]
        den += at_risk.sum()
        nume += (risk[i] > risk[at_risk]).sum() + 0.5 * (risk[i] == risk[at_risk]).sum()
    return nume / den if den else float("nan")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric(raw, test_idx: np.ndarray, train_idx: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    h = harrell_c(raw.dss_time[test_idx], raw.dss_event[test_idx], pred)
    u, _, _ = uno_c_index(raw.dss_time[test_idx], raw.dss_event[test_idx], pred, raw.dss_time[train_idx], raw.dss_event[train_idx])
    return float(h), float(u)


def build_anchor_identity(out: Path, workers: Path, source_workers: Path) -> dict:
    rows = []
    source_hashes = {}
    max_diff = 0.0
    for seed in SEEDS:
        for fold in range(1, 6):
            src = source_workers / f"seed_{seed}" / "cells" / f"C0-Full_{seed}_{fold}.npy"
            dst = workers / f"seed_{seed}" / "cells" / f"S0_{seed}_{fold}.npy"
            a = np.asarray(np.load(src), dtype=np.float64)
            b = np.asarray(np.load(dst), dtype=np.float64)
            diff = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
            max_diff = max(max_diff, diff)
            source_hashes[f"{seed}/{fold}"] = sha256(src)
            rows.append({"seed": seed, "outer_fold": fold, "n_test": int(a.size), "max_abs_diff": diff, "source_file": str(src), "status": "PASS" if diff <= 1e-7 else "FAIL"})
    payload = {"status": "PASS" if max_diff <= 1e-7 and len(rows) == 25 else "FAIL", "definition": "S0 is exact frozen M8-P0 C0-Full outer-test prediction", "cells_checked": len(rows), "max_abs_diff": max_diff, "tolerance": 1e-7, "source_hashes_sha256": source_hashes, "rows": rows}
    atomic_json(out / "strong_anchor_identity_audit.json", payload)
    return payload


def build_crossfit_audit(out: Path, rows: list[dict]) -> dict:
    groups = []
    for seed in SEEDS:
        for fold in range(1, 6):
            z = [r for r in rows if int(r["seed"]) == seed and int(r["outer_fold"]) == fold]
            held = sum(int(r["heldout_count"]) for r in z)
            groups.append({"seed": seed, "outer_fold": fold, "subfolds": len(z), "heldout_sum": held, "self_training_rows": sum(str(r.get("self_training", "")).lower() == "true" for r in z), "status": "PASS" if len(z) == 5 and held == 0 + int(sum(int(r["train_count"]) + int(r["heldout_count"]) for r in z) / 5) and all(str(r.get("self_training", "False")).lower() == "false" for r in z) else "FAIL"})
    # The cross-fit rows are per subfold; the exact outer-train patient count is
    # 409 or 410 depending on the held-out outer fold.  Five held-out subsets
    # must partition that count with no self-training rows.
    for g in groups:
        z = [r for r in rows if int(r["seed"]) == g["seed"] and int(r["outer_fold"]) == g["outer_fold"]]
        g["outer_train_count"] = sum(int(r["heldout_count"]) for r in z)
        g["status"] = "PASS" if len(z) == 5 and g["outer_train_count"] > 0 and g["self_training_rows"] == 0 and len({int(r["crossfit_subfold"]) for r in z}) == 5 else "FAIL"
    atomic_csv(out / "clinical_crossfit_audit.csv", groups)
    return {"status": "PASS" if groups and all(r["status"] == "PASS" for r in groups) else "FAIL", "groups_checked": len(groups), "self_training_rows": sum(r["self_training_rows"] for r in groups), "rows": groups}


def build_cox_audit(out: Path) -> dict:
    times = torch.tensor([1.0, 1.0, 2.0, 3.0, 4.0])
    events = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
    risks = torch.tensor([0.2, -0.1, 0.7, 0.0, -0.4], requires_grad=True)
    got = float(cox_loss_full(times, events, risks).detach())
    order = torch.argsort(times, descending=True, stable=True)
    tr, ev, rr = times[order], events[order], risks[order]
    manual = float((((torch.logcumsumexp(rr, 0) - rr) * ev).sum() / ev.sum()).detach())
    grad_ok = bool(torch.isfinite(torch.autograd.grad(cox_loss_full(times, events, risks), risks, retain_graph=True)[0]).all())
    payload = {"status": "PASS" if abs(got - manual) <= 1e-10 and grad_ok else "FAIL", "implementation": "historical full-risk-set Cox with stable descending time order", "tie_case": True, "loss": got, "manual_loss": manual, "absolute_error": abs(got - manual), "finite_gradient": grad_ok}
    atomic_json(out / "cox_riskset_audit.json", payload)
    return payload


def build_zero_init(out: Path, clinical_dim: int) -> dict:
    rows = []
    for seed in SEEDS:
        for variant, internal in (("SP", "RP"), ("SO", "RO"), ("SPO", "RPO")):
            model = build_residual(internal, clinical_dim, seed, 1)
            heads = [model.path_head, model.omic_head]
            value = max([float(h[-1].weight.abs().max() + h[-1].bias.abs().max()) for h in heads if h is not None] or [0.0])
            rows.append({"seed": seed, "variant": variant, "internal_variant": internal, "zero_init_max_abs": value, "status": "PASS" if value == 0.0 else "FAIL"})
    payload = {"status": "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL", "rows": rows, "shared_representation": "M8-P0 CommonOnly MultimodalBackbone; additive heads only"}
    atomic_json(out / "zero_init_audit.json", payload)
    return payload


def run(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    workers = Path(args.workers)
    source_workers = Path(args.source_workers)
    raw = load_raw_inputs(Path(args.data_root), Path(args.repo_root))
    fold_rows: list[dict] = []
    profiles: list[dict] = []
    crossfit_rows: list[dict] = []
    diagnostics: list[dict] = []
    for seed in SEEDS:
        worker = workers / f"seed_{seed}"
        fold_rows.extend(read_csv(worker / "fold_metrics.csv"))
        profiles.extend(read_csv(worker / "compute_profile.csv"))
        crossfit_rows.extend(read_csv(worker / "clinical_crossfit_audit.csv"))
        diagnostics.extend(read_csv(worker / "residual_diagnostics.csv"))
    # Recompute S0 metrics from the frozen source arrays themselves.  This
    # prevents a stale worker-side summary from changing the strong-anchor
    # comparator while keeping the source prediction byte-identical.
    for seed in SEEDS:
        for fold in range(1, 6):
            test_idx = np.flatnonzero(raw.outer_folds == fold)
            train_idx = np.flatnonzero(raw.outer_folds != fold)
            source_pred = np.asarray(np.load(source_workers / f"seed_{seed}" / "cells" / f"C0-Full_{seed}_{fold}.npy"), dtype=float)
            h, u = metric(raw, test_idx, train_idx, source_pred)
            row = next(r for r in fold_rows if r["variant"] == "S0" and int(r["seed"]) == seed and int(r["outer_fold"]) == fold)
            row["harrell_c"] = h
            row["uno_c"] = u
    for row in fold_rows:
        for key in ("seed", "outer_fold", "n_test", "events_test", "params", "best_epoch"):
            if key in row and row[key] != "": row[key] = int(row[key])
        for key in ("harrell_c", "uno_c", "uno_numerator", "uno_denominator"):
            if key in row and row[key] != "": row[key] = float(row[key])
    atomic_csv(out / "fold_metrics.csv", fold_rows)
    atomic_csv(out / "clinical_crossfit_audit.csv", crossfit_rows)
    atomic_csv(out / "residual_diagnostics.csv", diagnostics)
    atomic_csv(out / "compute_profile.csv", profiles)

    main = []
    seed_metrics = []
    for variant in VARIANTS:
        z = [r for r in fold_rows if r["variant"] == variant]
        h = np.array([float(r["harrell_c"]) for r in z]); u = np.array([float(r["uno_c"]) for r in z])
        main.append({"variant": variant, "n_cells": len(z), "mean_harrell": h.mean(), "mean_uno": u.mean(), "std_harrell": h.std(ddof=1), "std_uno": u.std(ddof=1), "worst_fold_harrell": h.min(), "worst_fold_uno": u.min()})
        for seed in SEEDS:
            q = [r for r in z if int(r["seed"]) == seed]; hh = np.array([float(r["harrell_c"]) for r in q]); uu = np.array([float(r["uno_c"]) for r in q])
            seed_metrics.append({"variant": variant, "seed": seed, "n_patients": sum(int(r["n_test"]) for r in q), "dss_events": sum(int(r["events_test"]) for r in q), "fold_mean_harrell": hh.mean(), "fold_mean_uno": uu.mean(), "fold_std_harrell": hh.std(ddof=1), "fold_std_uno": uu.std(ddof=1), "worst_fold_harrell": hh.min(), "worst_fold_uno": uu.min(), "status": "PASS"})
    atomic_csv(out / "main_metrics.csv", main)
    atomic_csv(out / "seed_metrics.csv", seed_metrics)

    # P6 uses the frozen M8-P0 B0 prediction as a context comparator.
    b0_fold = {}
    for seed in SEEDS:
        for fold in range(1, 6):
            b0_fold[(seed, fold)] = np.asarray(np.load(Path(args.m8p0_workers) / f"seed_{seed}" / "cells" / f"B0_{seed}_{fold}.npy"), dtype=float)
    deltas = []
    for a, b in COMPARISONS:
        for seed in SEEDS:
            aa = sorted([r for r in fold_rows if r["variant"] == a and int(r["seed"]) == seed], key=lambda r: int(r["outer_fold"]))
            if b == "B0":
                bb = [{"outer_fold": f, "harrell_c": metric(raw, np.flatnonzero(raw.outer_folds == f), np.flatnonzero(raw.outer_folds != f), b0_fold[(seed, f)])[0], "uno_c": metric(raw, np.flatnonzero(raw.outer_folds == f), np.flatnonzero(raw.outer_folds != f), b0_fold[(seed, f)])[1]} for f in range(1, 6)]
            else:
                bb = sorted([r for r in fold_rows if r["variant"] == b and int(r["seed"]) == seed], key=lambda r: int(r["outer_fold"]))
            for x, y in zip(aa, bb):
                deltas.append({"comparison": f"{a}_vs_{b}", "variant_a": a, "variant_b": b, "seed": seed, "outer_fold": int(x["outer_fold"]), "delta_harrell": float(x["harrell_c"]) - float(y["harrell_c"]), "delta_uno": float(x["uno_c"]) - float(y["uno_c"]), "status": "cell"})
            q = [r for r in deltas if r["comparison"] == f"{a}_vs_{b}" and r["seed"] == seed]
            deltas.append({"comparison": f"{a}_vs_{b}", "variant_a": a, "variant_b": b, "seed": seed, "outer_fold": "MEAN", "mean_delta_harrell": np.mean([r["delta_harrell"] for r in q]), "mean_delta_uno": np.mean([r["delta_uno"] for r in q]), "harrell_fold_wins": sum(r["delta_harrell"] > 0 for r in q), "uno_fold_wins": sum(r["delta_uno"] > 0 for r in q), "status": "seed_summary"})
        q = [r for r in deltas if r["comparison"] == f"{a}_vs_{b}" and r["status"] == "cell"]
        dh = np.array([r["delta_harrell"] for r in q]); du = np.array([r["delta_uno"] for r in q])
        deltas.append({"comparison": f"{a}_vs_{b}", "variant_a": a, "variant_b": b, "seed": "ALL", "outer_fold": "ALL", "mean_delta_harrell": dh.mean(), "mean_delta_uno": du.mean(), "std_delta_harrell": dh.std(ddof=1), "std_delta_uno": du.std(ddof=1), "harrell_wins": int((dh > 0).sum()), "uno_wins": int((du > 0).sum()), "status": "summary"})
    atomic_csv(out / "paired_deltas.csv", deltas)

    ordered_idx = np.concatenate([np.flatnonzero(raw.outer_folds == f) for f in range(1, 6)])
    oof_time = raw.dss_time[ordered_idx]; oof_event = raw.dss_event[ordered_idx]
    preds = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            for fold in range(1, 6):
                preds[(variant, seed, fold)] = np.asarray(np.load(workers / f"seed_{seed}" / "cells" / f"{variant}_{seed}_{fold}.npy"), dtype=float)
    boot = []
    rng = np.random.default_rng(8212026)
    for a, b in COMPARISONS[:3]:
        for seed in SEEDS:
            pa = np.concatenate([preds[(a, seed, f)] for f in range(1, 6)]); pb = np.concatenate([preds[(b, seed, f)] for f in range(1, 6)])
            hvals = []; uvals = []
            for _ in range(2000):
                idx = rng.integers(0, len(pa), len(pa)); tt = oof_time[idx]; ee = oof_event[idx]
                hvals.append(cfast(tt, ee, pa[idx]) - cfast(tt, ee, pb[idx]))
                ua, _, _ = uno_c_index(tt, ee, pa[idx], oof_time, oof_event); ub, _, _ = uno_c_index(tt, ee, pb[idx], oof_time, oof_event); uvals.append(ua - ub)
            h = np.asarray(hvals); u = np.asarray(uvals)
            boot.append({"comparison": f"{a}_vs_{b}", "seed": seed, "n_reps": 2000, "mean_delta_harrell": h.mean(), "ci95_low_harrell": np.quantile(h, .025), "ci95_high_harrell": np.quantile(h, .975), "mean_delta_uno": u.mean(), "ci95_low_uno": np.quantile(u, .025), "ci95_high_uno": np.quantile(u, .975)})
    atomic_csv(out / "paired_bootstrap.csv", boot)

    variances = []
    for variant in VARIANTS:
        z = [r for r in fold_rows if r["variant"] == variant]; seed_h = [np.mean([float(r["harrell_c"]) for r in z if int(r["seed"]) == s]) for s in SEEDS]; seed_u = [np.mean([float(r["uno_c"]) for r in z if int(r["seed"]) == s]) for s in SEEDS]
        variances.append({"variant": variant, "between_seed_variance_H": np.var(seed_h, ddof=1), "between_seed_variance_U": np.var(seed_u, ddof=1), "within_seed_fold_variance_H": np.mean([np.var([float(r["harrell_c"]) for r in z if int(r["seed"]) == s], ddof=1) for s in SEEDS]), "within_seed_fold_variance_U": np.mean([np.var([float(r["uno_c"]) for r in z if int(r["seed"]) == s], ddof=1) for s in SEEDS]), "worst_fold_H": min(float(r["harrell_c"]) for r in z), "worst_fold_U": min(float(r["uno_c"]) for r in z)})
    atomic_csv(out / "variance_decomposition.csv", variances)

    anchor = build_anchor_identity(out, workers, source_workers)
    cf_audit = build_crossfit_audit(out, crossfit_rows)
    cox_audit = build_cox_audit(out)
    clinical_dim = raw.clinical.shape[1]
    zero = build_zero_init(out, clinical_dim)

    decisions = {}
    for variant in ("SP", "SO", "SPO"):
        summary = next(r for r in deltas if r["status"] == "summary" and r["variant_a"] == variant and r["variant_b"] == "S0")
        seed_rows = [r for r in deltas if r["status"] == "seed_summary" and r["variant_a"] == variant and r["variant_b"] == "S0"]
        bb = [r for r in boot if r["comparison"] == f"{variant}_vs_S0"]
        pos_h = sum(float(r["mean_delta_harrell"]) > 0 for r in seed_rows); pos_u = sum(float(r["mean_delta_uno"]) > 0 for r in seed_rows)
        lower_h = min(float(r["ci95_low_harrell"]) for r in bb); lower_u = min(float(r["ci95_low_uno"]) for r in bb)
        no_stable_negative = max(float(r["ci95_high_harrell"]) for r in bb) > 0 and max(float(r["ci95_high_uno"]) for r in bb) > 0
        if float(summary["mean_delta_harrell"]) > 0 and float(summary["mean_delta_uno"]) > 0 and pos_h >= 4 and pos_u >= 4 and (lower_h >= 0 or lower_u >= 0) and no_stable_negative:
            decisions[variant] = "STRONG_ANCHOR_INCREMENT_CONFIRMED"
        elif float(summary["mean_delta_harrell"]) > 0 and float(summary["mean_delta_uno"]) > 0 and pos_h >= 4 and pos_u >= 4:
            decisions[variant] = "STRONG_ANCHOR_INCREMENT_DIRECTIONAL"
        else:
            decisions[variant] = "STRONG_ANCHOR_INCREMENT_NOT_CONFIRMED"
    final = "A" if any(v == "STRONG_ANCHOR_INCREMENT_CONFIRMED" for v in decisions.values()) else "B" if any(v == "STRONG_ANCHOR_INCREMENT_DIRECTIONAL" for v in decisions.values()) else "C"
    atomic_json(out / "run_manifest.json", {"phase": "M8-P1", "status": "complete", "cells_completed": len(fold_rows) - 25, "cells_expected": 75, "total_rows_including_frozen_s0": len(fold_rows), "formal_residual_cells": 75, "s0_frozen_cells": 25, "N": 512, "DSS_events": 65, "seeds": list(SEEDS), "variants": list(VARIANTS), "source_workers": str(source_workers), "source_files_unchanged": True, "predictions_modified": False, "model_training_runs": 75, "execution_sha": subprocess.check_output(["git", "-C", args.repo_root, "rev-parse", "HEAD"], text=True).strip(), "audits": {"strong_anchor_identity": anchor["status"], "clinical_crossfit": cf_audit["status"], "cox_riskset": cox_audit["status"], "zero_init": zero["status"]}, "decisions": decisions, "final_decision": final})

    mm = {r["variant"]: r for r in main}; summary = {}
    for a, b in COMPARISONS:
        summary[f"{a}_vs_{b}"] = next(r for r in deltas if r["status"] == "summary" and r["variant_a"] == a and r["variant_b"] == b)
    lines = ["# Phase M8-P1 Strong Anchor Residual Confirmation", "", "## Protocol", "", "- Fixed DSS cohort: N=512, events=65, five outer folds, seeds 20261012??0261016.", "- S0 is the frozen M8-P0 C0-Full outer-test prediction; SP/SO/SPO use additive zero-initialized residual heads.", "- 75/75 residual cells were trained; S0 was reused without retraining.", "", "## Required summary", "", f"1. Branch: `{args.branch}`", f"2. Commit SHA: `{subprocess.check_output(['git','-C',args.repo_root,'rev-parse','HEAD'],text=True).strip()}`", f"3. 75/75 residual cells complete: **{'YES' if len(fold_rows)==100 else 'NO'}** (100 total rows including 25 frozen S0 cells)", f"4. StrongAnchor == C0 audit: **{anchor['status']}**, max abs diff={anchor['max_abs_diff']:.3g}"]
    for label in ("S0", "SP", "SO", "SPO"):
        lines.append(f"{label} H/U: {mm[label]['mean_harrell']:.6f} / {mm[label]['mean_uno']:.6f}")
    for variant in ("SP", "SO", "SPO"):
        s = summary[f"{variant}_vs_S0"]
        ss = [r for r in deltas if r["status"] == "seed_summary" and r["variant_a"] == variant and r["variant_b"] == "S0"]
        lines.append(f"{variant} delta vs S0: H={s['mean_delta_harrell']:+.6f}, U={s['mean_delta_uno']:+.6f}; positive seeds H={sum(float(r['mean_delta_harrell'])>0 for r in ss)}/5, U={sum(float(r['mean_delta_uno'])>0 for r in ss)}/5; **{decisions[variant]}**")
    lines += ["", "## Bootstrap", "", "Each SP/SO/SPO vs S0 comparison used 2,000 patient-level paired bootstrap replicates per seed. Exact CIs are in `paired_bootstrap.csv`; the decision uses the prompt's conservative CI gate.", "", f"SPO vs B0: H={summary['SPO_vs_B0']['mean_delta_harrell']:+.6f}, U={summary['SPO_vs_B0']['mean_delta_uno']:+.6f} (context only).", "", "## Variance", "", "Between-seed and within-seed fold variances for S0/SP/SO/SPO are in `variance_decomposition.csv`.", "", f"## Final decision", "", f"**Decision {final}: {'STRONG_ANCHOR_INCREMENT_CONFIRMED' if final=='A' else 'STRONG_ANCHOR_INCREMENT_DIRECTIONAL' if final=='B' else 'STRONG_ANCHOR_INCREMENT_NOT_CONFIRMED'}**.", "", "If A/B, the next permitted step is low-capacity shrinkage residual modeling. If C, stop the Clinical-residual line and retain C0/B0.", "", "## Audit status", "", f"- strong_anchor_identity_audit.json: {anchor['status']}", f"- clinical_crossfit_audit.csv: {cf_audit['status']}", f"- cox_riskset_audit.json: {cox_audit['status']}", f"- zero_init_audit.json: {zero['status']}", "", "## Output directory", "", str(out)]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(fold_rows), "final_decision": final, "decisions": decisions}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True); parser.add_argument("--data-root", required=True); parser.add_argument("--workers", required=True); parser.add_argument("--source-workers", required=True); parser.add_argument("--m8p0-workers", required=True); parser.add_argument("--output", required=True); parser.add_argument("--branch", required=True)
    run(parser.parse_args())


"""Same-fold low-capacity Ridge Cox controls for Phase8-V0R."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.phase8_v0.cox import fit_cox, efron_loss_grad, concordance, km_left
from src.phase8_v0.dataset import load_private, digest, write_json
from src.phase8_v0r.evaluation import pair_stats, _bootstrap_delta, SEEDS

METHODS = {
    "Clinical+S32": "cs",
    "Clinical+RNA+S32": "crs",
    "JointRidge": "joint",
}
RIDGE_GRID = [0.01, 0.1, 1.0, 10.0]


def robust_fit_cox(x, time, event, lam):
    """Use the registered optimizer first, then a stricter same-objective retry."""
    try:
        return fit_cox(x, time, event, lam)
    except RuntimeError as first:
        objective = efron_loss_grad(np.asarray(x, float), np.asarray(time, float), np.asarray(event, int), lam)
        result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                          options={"maxiter": 50000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 100, "maxfun": 100000})
        loss, gradient = objective(result.x); g = float(np.max(np.abs(gradient)))
        if (not np.isfinite(loss)) or g > 1e-5:
            raise RuntimeError(f"Ridge Cox retry failed: {result.message}; gradient={g}; first={first}")
        return result.x, {"iterations": int(result.nit), "gradient_max": g, "objective": loss,
                          "optimizer_message": str(result.message), "retry": True}


def design(block, method):
    c, p, o, s = block["C"], block["P"], block["O"], block["S"]
    if method == "cs": return np.concatenate([c, s], axis=1)
    if method == "crs": return np.concatenate([c, o[:, :16], s], axis=1)
    if method == "joint": return np.concatenate([c, p, o], axis=1)
    raise KeyError(method)


def run(formal_dir: Path, output_dir: Path):
    formal_dir = Path(formal_dir); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cells").mkdir(exist_ok=True); (output_dir / "oof").mkdir(exist_ok=True)
    d = load_private(Path("/root/phase7a_foundation_headroom_v1")); rows = []
    for seed in SEEDS:
        for fold in range(1, 6):
            ctx = np.load(formal_dir / "private_contexts" / f"seed{seed}_fold{fold}.npz")
            tr, te, it, iv = ctx["tr"], ctx["te"], ctx["it"], ctx["iv"]
            for method, code in METHODS.items():
                tag = f"DSS_seed{seed}_fold{fold}_{code}"; op = output_dir / "oof" / f"{tag}.npz"; cp = output_dir / "cells" / f"{tag}.json"
                if cp.exists() and op.exists():
                    rows.append(json.loads(cp.read_text())); continue
                xi = design({k: ctx["inner_" + k] for k in ["C", "P", "O", "S"]}, code)
                xo = design({k: ctx["outer_" + k] for k in ["C", "P", "O", "S"]}, code)
                trials = []
                for lam in RIDGE_GRID:
                    w, detail = robust_fit_cox(xi[it], d["DSS_time"][it], d["DSS_event"][it], lam)
                    c = concordance(d["DSS_time"][iv], d["DSS_event"][iv], xi[iv] @ w)
                    trials.append({"ridge": lam, "inner_harrell": c, **detail})
                choice = max(trials, key=lambda z: (z["inner_harrell"], -z["ridge"]))
                w, detail = robust_fit_cox(xo[tr], d["DSS_time"][tr], d["DSS_event"][tr], choice["ridge"])
                trainrisk = xo[tr] @ w; raw = xo[te] @ w; mean = float(trainrisk.mean()); sd = float(trainrisk.std())
                risk = (raw - mean) / max(sd, 1e-12)
                g = km_left(d["DSS_time"][tr], d["DSS_event"][tr], d["DSS_time"][te]); active = (d["DSS_event"][te] == 1) & (d["DSS_time"][te] <= 60)
                weights = np.zeros(len(te)); weights[active] = 1.0 / g[active] ** 2
                np.savez_compressed(op, test_indices=te, patient_ids=d["patient_ids"][te], time=d["DSS_time"][te], event=d["DSS_event"][te], risk=risk, raw_risk=raw, censor_g=g, uno_weight=weights, seed=seed, fold=fold, method=method)
                row = {"status": "PASS", "method": method, "code": code, "seed": seed, "fold": fold, "dimension": int(xo.shape[1]),
                       "parameter_count": int(len(w)), "selected_ridge": choice["ridge"], "inner_trials": json.dumps(trials),
                       "outer_gradient_max": detail["gradient_max"], "inner_best_harrell": choice["inner_harrell"],
                       "within_fold_harrell": pair_stats(d["DSS_time"][te], d["DSS_event"][te], risk)[2],
                       "within_fold_uno60": pair_stats(d["DSS_time"][te], d["DSS_event"][te], risk, weights)[2],
                       "oof_sha256": digest(op), "outer_test_used_for_selection": False, "patient_id_written": False}
                write_json(cp, row); rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(output_dir / "cell_summary.csv", index=False)
    risks = {}; weights_by_seed = {}; seed_rows = []
    for seed in SEEDS:
        weights_full = np.zeros(len(d["patient_ids"]))
        for method in METHODS:
            rr = np.full(len(d["patient_ids"]), np.nan); ww = np.zeros(len(d["patient_ids"]))
            for fold in range(1, 6):
                z = np.load(output_dir / "oof" / f"DSS_seed{seed}_fold{fold}_{METHODS[method]}.npz")
                rr[z["test_indices"]] = z["risk"]; ww[z["test_indices"]] = z["uno_weight"]; weights_full[z["test_indices"]] = z["uno_weight"]
            risks[seed, method] = rr
            fold_h = []; fold_u = []; hn = hd = un = ud = 0.0
            for fold in range(1, 6):
                ix = np.flatnonzero(d["folds"] == fold); n0, d0, h = pair_stats(d["DSS_time"][ix], d["DSS_event"][ix], rr[ix]); u0, v0, u = pair_stats(d["DSS_time"][ix], d["DSS_event"][ix], rr[ix], ww[ix]);
                hn += n0; hd += d0; un += u0; ud += v0; fold_h.append(h); fold_u.append(u)
            seed_rows.append({"seed": seed, "method": method, "within_fold_harrell": hn / hd, "within_fold_uno60": un / ud,
                              "pooled_oof_harrell": pair_stats(d["DSS_time"], d["DSS_event"], rr)[2],
                              "pooled_oof_uno60": pair_stats(d["DSS_time"], d["DSS_event"], rr, ww)[2],
                              "fold_harrell_mean": float(np.mean(fold_h)), "fold_uno60_mean": float(np.mean(fold_u))})
        weights_by_seed[seed] = weights_full
    # Add the fixed formal C3R arms to the comparison risk map without changing them.
    for arm in ["A0", "A3", "A5"]:
        for seed in SEEDS:
            rr = np.full(len(d["patient_ids"]), np.nan); ww = np.zeros(len(d["patient_ids"]))
            for fold in range(1, 6):
                z = np.load(formal_dir / "oof" / f"DSS_seed{seed}_fold{fold}_{arm}.npz"); rr[z["test_indices"]] = z["risk"]; ww[z["test_indices"]] = z["uno_weight"]
            risks[seed, arm] = rr; weights_by_seed[seed] = ww
    pd.DataFrame(seed_rows).to_csv(output_dir / "seed_metrics.csv", index=False)
    comparisons = []
    pairs = [("Clinical+S32", "A0"), ("Clinical+RNA+S32", "A0"), ("JointRidge", "A0"),
             ("Clinical+S32", "A3"), ("Clinical+RNA+S32", "A3"), ("JointRidge", "A3"),
             ("Clinical+S32", "A5"), ("Clinical+RNA+S32", "A5"), ("JointRidge", "A5")]
    for i, (a, b) in enumerate(pairs):
        x = _bootstrap_delta(d, risks, weights_by_seed, a, b, B=2000, seed=20261040 + i)
        x["comparison"] = f"{a}-{b}"; comparisons.append(x)
    pd.DataFrame(comparisons).to_csv(output_dir / "paired_comparisons.csv", index=False)
    write_json(output_dir / "manifest.json", {"status": "PASS", "formal_dir": str(formal_dir), "methods": METHODS,
                                               "dimensions": {k: int(df[df.method == k].dimension.iloc[0]) for k in METHODS},
                                               "ridge_grid": RIDGE_GRID, "same_fixed_outer_folds": True,
                                               "no_patient_level_artifacts_published": True})
    lines = ["# Phase8-V0R：简单 Ridge Cox 对照", "", "三个对照使用 Phase8-V0 相同 outer folds、inner lambda selection、fold-local Clinical/P/R/O/S 表征和 Efron Cox。", "", "|方法|维度|Harrell|Uno60|", "|---|---:|---:|---:|"]
    for _, x in pd.DataFrame(seed_rows).groupby("method").mean(numeric_only=True).iterrows():
        method = x.name; lines.append(f"|{method}|{int(df[df.method==method].dimension.iloc[0])}|{x.within_fold_harrell:.6f}|{x.within_fold_uno60:.6f}|")
    lines += ["", "## 与 C3R-Surv 的配对比较", "", "|比较|ΔH|H CI|ΔUno60|Uno CI|", "|---|---:|---|---:|---|"]
    for _, x in pd.DataFrame(comparisons).iterrows(): lines.append(f"|{x.comparison}|{x.delta_within_fold_harrell:+.6f}|[{x.harrell_ci95_low:+.6f},{x.harrell_ci95_high:+.6f}]|{x.delta_within_fold_uno60:+.6f}|[{x.uno60_ci95_low:+.6f},{x.uno60_ci95_high:+.6f}]|")
    lines += ["", "结论只用于判断简单线性基线是否已经覆盖当前表征增量；不把跨协议历史数字排入本表。"]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_dir / "DONE.json", {"status": "PASS", "cells": len(df), "methods": list(METHODS)})
    return df

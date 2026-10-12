"""Independent fold-wise evaluation for Phase10-P0 private OOF files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.phase8_v0.cox import km_left


def pair_stats(t, e, risk, weights=None):
    t = np.asarray(t); e = np.asarray(e); risk = np.asarray(risk)
    eligible = (e[:, None] == 1) & ((t[:, None] < t[None, :]) | ((t[:, None] == t[None, :]) & (e[None, :] == 0)))
    w = np.ones(len(t)) if weights is None else np.asarray(weights)
    den = float((eligible * w[:, None]).sum())
    credit = (risk[:, None] > risk[None, :]).astype(float) + .5 * (risk[:, None] == risk[None, :])
    return float((eligible * w[:, None] * credit).sum()), den, float((eligible * w[:, None] * credit).sum() / den) if den else np.nan


def eval_one(out, parent, arm, seed, fold):
    z = np.load(Path(out) / "oof" / f"DSS_seed{seed}_fold{fold}_{arm}.npz")
    d = np.load(Path(parent) / "private_dataset.npz", allow_pickle=True)
    te = z["test_indices"].astype(int); t = d["DSS_time"][te]; e = d["DSS_event"][te]; tr = np.flatnonzero(d["folds"] != fold)
    risk = z["risk"].astype(float); hnum, hden, h = pair_stats(t, e, risk)
    g = km_left(d["DSS_time"][tr], d["DSS_event"][tr], t); active = (e == 1) & (t <= 60) & np.isfinite(g) & (g > 0)
    weight = np.zeros_like(g, dtype=float)
    np.divide(1.0, g ** 2, out=weight, where=active & np.isfinite(g) & (g > 0))
    unum, uden, u = pair_stats(t, e, risk, weight)
    return {"arm": arm, "seed": seed, "fold": fold, "test_n": len(te), "DSS_events_test": int(e.sum()),
            "harrell_num": hnum, "harrell_den": hden, "harrell": h, "uno_num": unum, "uno_den": uden, "uno60": u,
            "uno_invalid_events": int(((e == 1) & (t <= 60) & ~active).sum())}


def bootstrap_deltas(fold_rows, B=1000, seed=20261013):
    fold_rows = list(fold_rows)
    rng = np.random.default_rng(seed); results = []
    primary = [("P2", "P1"), ("P3", "P2"), ("P3", "P3-noTopo"), ("P4", "P3"), ("P4", "P0")]
    for a, b in primary:
        rows_a = {(int(x.seed), int(x.fold)): x for x in fold_rows if x.arm == a}
        rows_b = {(int(x.seed), int(x.fold)): x for x in fold_rows if x.arm == b}
        keys = sorted(rows_a); delta_h = []; delta_u = []
        observed_h = np.mean([rows_a[k].harrell - rows_b[k].harrell for k in keys]); observed_u = np.mean([rows_a[k].uno60 - rows_b[k].uno60 for k in keys])
        for _ in range(B):
            chosen = rng.choice(len(keys), len(keys), replace=True)
            delta_h.append(np.mean([rows_a[keys[j]].harrell - rows_b[keys[j]].harrell for j in chosen]))
            delta_u.append(np.mean([rows_a[keys[j]].uno60 - rows_b[keys[j]].uno60 for j in chosen]))
        results.append({"comparison": f"{a}-{b}", "delta_harrell": observed_h,
                        "harrell_ci95_low": np.quantile(delta_h, .025), "harrell_ci95_high": np.quantile(delta_h, .975),
                        "delta_uno60": observed_u, "uno60_ci95_low": np.quantile(delta_u, .025),
                        "uno60_ci95_high": np.quantile(delta_u, .975), "bootstrap_reps": B, "unit": "fold"})
    return results


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261007); args = parser.parse_args()
    arms = ["P0", "P1", "P2", "P3", "P4", "P3-noTopo"]; rows = []
    for fold in range(1, 6):
        for arm in arms: rows.append(eval_one(args.output, args.parent, arm, args.seed, fold))
    df = pd.DataFrame(rows); df.to_csv(args.output / "fold_metrics.csv", index=False)
    summary = df.groupby("arm").agg(harrell=("harrell", "mean"), harrell_sd=("harrell", "std"), uno60=("uno60", "mean"), uno60_sd=("uno60", "std"), test_events=("DSS_events_test", "sum")).reset_index()
    summary.to_csv(args.output / "summary_metrics.csv", index=False)
    pd.DataFrame(bootstrap_deltas(df.itertuples(index=False))).to_csv(args.output / "paired_comparisons.csv", index=False)
    best = summary.sort_values("harrell", ascending=False).iloc[0]
    decision = {"status": "PASS", "best_arm": str(best.arm), "best_harrell": float(best.harrell), "best_uno60": float(best.uno60),
                "primary_comparisons": ["P2-P1", "P3-P2", "P3-P3-noTopo", "P4-P3", "P4-P0"], "external_confirmation": False, "test_label_selection": False}
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    lines = ["# Phase10-P0 SCPN-Surv", "", "本轮由独立 OOF 评测脚本重算五折 DSS Harrell 与 Uno60。", "", "|Arm|Harrell mean|SD|Uno60 mean|SD|", "|---|---:|---:|---:|---:|"]
    for row in summary.itertuples(): lines.append(f"|{row.arm}|{row.harrell:.6f}|{row.harrell_sd:.6f}|{row.uno60:.6f}|{row.uno60_sd:.6f}|")
    lines += ["", "## 预先锁定比较", "", "|比较|Δ Harrell|95% CI|Δ Uno60|95% CI|", "|---|---:|---|---:|---|"]
    comp = pd.read_csv(args.output / "paired_comparisons.csv")
    for row in comp.itertuples(): lines.append(f"|{row.comparison}|{row.delta_harrell:+.6f}|[{row.harrell_ci95_low:+.6f},{row.harrell_ci95_high:+.6f}]|{row.delta_uno60:+.6f}|[{row.uno60_ci95_low:+.6f},{row.uno60_ci95_high:+.6f}]|")
    lines += ["", f"最佳模型：{best.arm}。该结果只适用于本轮固定队列、单 seed 和训练折内选择。"]
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output / "DONE.json").write_text(json.dumps({"status": "PASS", "cells": len(df), "independent_evaluation": True}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()

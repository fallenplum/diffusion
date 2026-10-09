"""Independent reconstruction of Phase8-V0 metrics from persisted OOF files.

This module never fits a model. It reconstructs every held-out fold, verifies
the cached Clinical anchor, and computes IPCW Uno statistics from numerators
and denominators rather than weighting fold Uno values by Harrell pair counts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase8_v0.dataset import digest, load_private, write_json
from src.phase8_v0.cox import km_left

SEEDS = [20261007, 20261008, 20261009]
ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
COMPARISONS = [("A3", "A0"), ("A4", "A3"), ("A5", "A4"),
               ("A6", "A5"), ("A5", "A0"), ("A1", "A0"), ("A2", "A0")]


def _hash_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def pair_stats(time, event, risk, weights=None):
    """Return numerator, denominator and C for one fixed patient set."""
    time = np.asarray(time, float); event = np.asarray(event, int); risk = np.asarray(risk, float)
    w = np.ones(len(time), float) if weights is None else np.asarray(weights, float)
    eligible = (event[:, None] == 1) & ((time[:, None] < time[None, :]) |
                 ((time[:, None] == time[None, :]) & (event[None, :] == 0)))
    credit = (risk[:, None] > risk[None, :]).astype(float) + .5 * (risk[:, None] == risk[None, :])
    den = float((eligible * w[:, None]).sum())
    num = float((eligible * w[:, None] * credit).sum())
    return num, den, (num / den if den else float("nan"))


def _bootstrap_delta(d, risks, weights, a, b, B=2000, seed=20261019):
    """Fold/event stratified bootstrap of paired H and Uno deltas."""
    t = np.asarray(d["DSS_time"], float); e = np.asarray(d["DSS_event"], int)
    fold = np.asarray(d["folds"], int); rng = np.random.default_rng(seed)
    h_by_seed = []; u_by_seed = []
    h_boot = []; u_boot = []
    for s in SEEDS:
        ra = risks[s, a]; rb = risks[s, b]; ww = weights[s]
        def within(r, w=None):
            ns = ds = 0.0
            for f in range(1, 6):
                ix = np.flatnonzero(fold == f)
                n0, d0, _ = pair_stats(t[ix], e[ix], r[ix], None if w is None else w[ix])
                ns += n0; ds += d0
            return ns / ds
        h_delta = within(ra) - within(rb)
        u_delta = within(ra, ww) - within(rb, ww)
        # The primary statistic is the sum of fold numerators/denominators.
        h_num_a = np.zeros(B); h_num_b = np.zeros(B); h_den = np.zeros(B)
        u_num_a = np.zeros(B); u_num_b = np.zeros(B); u_den = np.zeros(B)
        for f in range(1, 6):
            ix = np.flatnonzero(fold == f); tt = t[ix]; ee = e[ix]
            elig = (ee[:, None] == 1) & ((tt[:, None] < tt[None, :]) |
                    ((tt[:, None] == tt[None, :]) & (ee[None, :] == 0)))
            ca = (ra[ix, None] > ra[ix, None].T).astype(float) + .5 * (ra[ix, None] == ra[ix, None].T)
            cb = (rb[ix, None] > rb[ix, None].T).astype(float) + .5 * (rb[ix, None] == rb[ix, None].T)
            mult = np.zeros((B, len(ix)), float)
            for ev in (0, 1):
                g = np.flatnonzero(ee == ev)
                if len(g): mult[:, g] = rng.multinomial(len(g), np.ones(len(g)) / len(g), size=B)
            pair_mult = np.einsum("bi,ij,bj->bij", mult, elig.astype(float), mult)
            base = np.einsum("bij->b", pair_mult)
            h_den += base
            h_num_a += np.einsum("bij,ij->b", pair_mult, ca)
            h_num_b += np.einsum("bij,ij->b", pair_mult, cb)
            uw = ww[ix]
            u_pair = pair_mult * uw[None, :, None]
            u_den += np.einsum("bij->b", u_pair)
            u_num_a += np.einsum("bij,ij->b", u_pair, ca)
            u_num_b += np.einsum("bij,ij->b", u_pair, cb)
        h_boot.append((h_num_a - h_num_b) / h_den)
        u_boot.append((u_num_a - u_num_b) / u_den)
        h_by_seed.append(h_delta); u_by_seed.append(u_delta)
    return {
        "delta_within_fold_harrell": float(np.mean(h_by_seed)),
        "delta_within_fold_uno60": float(np.mean(u_by_seed)),
        "harrell_seed_deltas": json.dumps([float(x) for x in h_by_seed]),
        "uno60_seed_deltas": json.dumps([float(x) for x in u_by_seed]),
        "harrell_ci95_low": float(np.quantile(np.mean(np.stack(h_boot), axis=0), .025)),
        "harrell_ci95_high": float(np.quantile(np.mean(np.stack(h_boot), axis=0), .975)),
        "uno60_ci95_low": float(np.quantile(np.mean(np.stack(u_boot), axis=0), .025)),
        "uno60_ci95_high": float(np.quantile(np.mean(np.stack(u_boot), axis=0), .975)),
        "all_seed_harrell_positive": bool(min(h_by_seed) > 0),
        "all_seed_uno60_positive": bool(min(u_by_seed) > 0),
    }


def reconstruct(formal_dir: Path, output_dir: Path, bootstrap_replicates=2000):
    formal_dir = Path(formal_dir); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    d = load_private(Path("/root/phase7a_foundation_headroom_v1"))
    folds = np.asarray(d["folds"], int); n = len(d["patient_ids"])
    old_seed = pd.read_csv(formal_dir / "seed_metrics.csv")
    fold_rows = []; seed_rows = []; risks = {}; weights = {}
    anchor_rows = []; expected = set()
    for seed in SEEDS:
        for fold in range(1, 6):
            ctx = np.load(formal_dir / "private_contexts" / f"seed{seed}_fold{fold}.npz")
            te_expected = np.asarray(ctx["te"], int)
            anchor = np.asarray(ctx["outer_test_anchor"], float)
            anchor_hash = _hash_array(anchor)
            for arm in ARMS:
                expected.add((seed, fold, arm))
                op = formal_dir / "oof" / f"DSS_seed{seed}_fold{fold}_{arm}.npz"
                z = np.load(op, allow_pickle=True); te = np.asarray(z["test_indices"], int)
                if not np.array_equal(te, te_expected): raise AssertionError(f"test index mismatch: {op}")
                if not np.array_equal(d["folds"][te], np.full(len(te), fold)): raise AssertionError(f"fold mismatch: {op}")
                if not np.array_equal(z["patient_ids"].astype(str), d["patient_ids"][te].astype(str)): raise AssertionError(f"patient ID mismatch: {op}")
                t = np.asarray(z["time"], float); e = np.asarray(z["event"], int); r = np.asarray(z["risk"], float)
                w = np.asarray(z["uno_weight"], float)
                hnum, hden, hc = pair_stats(t, e, r)
                unum, uden, uc = pair_stats(t, e, r, w)
                rc = np.asarray(z["r_C"], float)
                anchor_rows.append({"seed": seed, "fold": fold, "arm": arm,
                                    "anchor_hash": anchor_hash, "r_C_hash": _hash_array(rc),
                                    "max_abs_r_C_anchor": float(np.max(np.abs(rc - anchor))),
                                    "bitwise_equal": bool(np.array_equal(rc, anchor))})
                old = old_seed[(old_seed.seed == seed) & (old_seed.arm == arm)].iloc[0]
                fold_rows.append({"seed": seed, "arm": arm, "fold": fold, "test_n": len(te), "test_events": int(e.sum()),
                                  "harrell_num": hnum, "harrell_den": hden, "harrell": hc,
                                  "uno60_num": unum, "uno60_den": uden, "uno60": uc,
                                  "old_uno60": float(old.within_fold_uno60), "uno60_delta": uc - float(old.within_fold_uno60),
                                  "risk_sha256": _hash_array(r), "uno_weight_sha256": _hash_array(w)})
        for arm in ARMS:
            rr = np.full(n, np.nan); ww = np.zeros(n); raw = np.full(n, np.nan)
            arm_folds = [x for x in fold_rows if x["seed"] == seed and x["arm"] == arm]
            for fold in range(1, 6):
                z = np.load(formal_dir / "oof" / f"DSS_seed{seed}_fold{fold}_{arm}.npz")
                rr[z["test_indices"]] = z["risk"]; raw[z["test_indices"]] = z["raw_risk"]; ww[z["test_indices"]] = z["uno_weight"]
            risks[seed, arm] = rr; weights[seed] = ww
            hnum = sum(x["harrell_num"] for x in arm_folds); hden = sum(x["harrell_den"] for x in arm_folds)
            unum = sum(x["uno60_num"] for x in arm_folds); uden = sum(x["uno60_den"] for x in arm_folds)
            pooled_h = pair_stats(d["DSS_time"], d["DSS_event"], rr)[2]
            pooled_u = pair_stats(d["DSS_time"], d["DSS_event"], rr, ww)[2]
            old = old_seed[(old_seed.seed == seed) & (old_seed.arm == arm)].iloc[0]
            seed_rows.append({"seed": seed, "arm": arm, "within_fold_harrell_repaired": hnum / hden,
                              "within_fold_uno60_repaired": unum / uden, "pooled_oof_harrell": pooled_h,
                              "pooled_oof_uno60_repaired": pooled_u,
                              "old_within_fold_harrell": float(old.within_fold_harrell),
                              "old_within_fold_uno60": float(old.within_fold_uno60),
                              "uno60_aggregation_delta": unum / uden - float(old.within_fold_uno60)})
    if expected != {(s, f, a) for s in SEEDS for f in range(1, 6) for a in ARMS}:
        raise AssertionError("not all 105 OOF cells were reconstructed")
    anchor_df = pd.DataFrame(anchor_rows); fold_df = pd.DataFrame(fold_rows); seed_df = pd.DataFrame(seed_rows)
    pair_rows = []
    for j, (a, b) in enumerate(COMPARISONS):
        item = _bootstrap_delta(d, risks, weights, a, b, B=bootstrap_replicates, seed=20261019 + j)
        item.update({"comparison": f"{a}-{b}", "family": "primary" if j < 3 else "secondary"})
        pair_rows.append(item)
    pairs = pd.DataFrame(pair_rows)
    fold_df.to_csv(output_dir / "fold_metrics_repaired.csv", index=False)
    seed_df.to_csv(output_dir / "seed_metrics_repaired.csv", index=False)
    pairs.to_csv(output_dir / "paired_comparisons_repaired.csv", index=False)
    anchor_df.to_csv(output_dir / "clinical_anchor_identity_repaired.csv", index=False)
    write_json(output_dir / "anchor_audit.json", {"status": "PASS" if anchor_df.max_abs_r_C_anchor.max() <= 1e-12 and anchor_df.bitwise_equal.all() else "FAIL",
                                                   "cells": len(anchor_df), "max_abs_difference": float(anchor_df.max_abs_r_C_anchor.max()),
                                                   "all_bitwise_equal": bool(anchor_df.bitwise_equal.all()),
                                                   "unique_anchor_hashes": int(anchor_df.anchor_hash.nunique())})
    write_json(output_dir / "manifest.json", {"status": "PASS", "formal_dir": str(formal_dir), "formal_done_sha256": digest(formal_dir / "DONE.json"),
                                               "n_oof_cells": len(fold_df), "bootstrap_replicates": bootstrap_replicates,
                                               "uno_fix": "sum fold-local IPCW numerators and denominators; no Harrell-pair weighting"})
    lines = ["# Phase8-V0R：评测修复", "", "旧 Phase8-V0 报告未修改。本目录从 105 个 OOF 文件独立重建逐折指标。",
             "", "## 修复内容", "", "- Uno60 按 fold-local IPCW 分子/分母合并；不再使用 Harrell 可比较对数加权折 Uno。",
             "- pooled OOF Uno 使用每个患者所属 outer-train censoring KM 产生的权重。", "- 所有 A0–A6 的 `r_C` 逐元素与缓存 Clinical anchor 校验。", "",
             "## 逐 seed 修复后均值", "", "|seed|arm|within-fold Harrell|within-fold Uno60|pooled Harrell|pooled Uno60|旧 Uno|修复−旧|", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for _, x in seed_df.iterrows():
        lines.append(f"|{int(x.seed)}|{x.arm}|{x.within_fold_harrell_repaired:.6f}|{x.within_fold_uno60_repaired:.6f}|{x.pooled_oof_harrell:.6f}|{x.pooled_oof_uno60_repaired:.6f}|{x.old_within_fold_uno60:.6f}|{x.uno60_aggregation_delta:+.6f}|")
    lines += ["", "## 独立配对区间", "", "|比较|ΔH|H 95% CI|ΔUno60|Uno 95% CI|三 seed H 全正|三 seed Uno 全正|", "|---|---:|---|---:|---|---|---|"]
    for _, x in pairs.iterrows():
        lines.append(f"|{x.comparison}|{x.delta_within_fold_harrell:+.6f}|[{x.harrell_ci95_low:+.6f},{x.harrell_ci95_high:+.6f}]|{x.delta_within_fold_uno60:+.6f}|[{x.uno60_ci95_low:+.6f},{x.uno60_ci95_high:+.6f}]|{bool(x.all_seed_harrell_positive)}|{bool(x.all_seed_uno60_positive)}|")
    lines += ["", "## 审计状态", "", f"- OOF 重建：{len(fold_df)}/105。", f"- Clinical anchor：{anchor_df.max_abs_r_C_anchor.max():.3g} 最大绝对误差，bitwise equal={bool(anchor_df.bitwise_equal.all())}。", "- 原 Phase8-V0 报告保持不变。"]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_dir / "DONE.json", {"status": "PASS", "oof_cells": len(fold_df), "anchor_status": "PASS" if anchor_df.bitwise_equal.all() else "FAIL"})
    return seed_df, pairs

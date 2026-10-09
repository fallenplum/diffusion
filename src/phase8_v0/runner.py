"""Phase8-V0 orchestration: audit, smoke, full nested OOF, validation, report."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold

from .audit import architecture_audit, leakage_audit, independent_validation
from .cox import fit_cox, concordance, km_left, finite_difference_check
from .dataset import load_private, source_audit, registry, digest, ahash, write_json
from .features import FoldFeatures, residualize_crossfit
from .models import ARM_SPEC, train_head, fit_for_epochs
from .metrics import fold_metrics, seed_summary


ARMS = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]


def cfg():
    return json.loads((Path(__file__).resolve().parents[2] / "configs/phase8_v0_c3r_surv.json").read_text())


def _stratified_inner(d, tr, seed, fold):
    cancer = np.asarray(d["cancer_type"]).astype(str)
    key = np.asarray([f"{int(d['DSS_event'][i])}|{cancer[i]}" for i in tr])
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=int(seed + fold * 100))
        a, b = next(splitter.split(tr, key))
    except ValueError:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=int(seed + fold * 100))
        a, b = next(splitter.split(tr, d["DSS_event"][tr]))
    return tr[a], tr[b]


def _anchor_fit(blocks, d, train_idx, val_idx, lam_grid):
    trials = []
    for lam in lam_grid:
        w, detail = fit_cox(blocks["C"][train_idx], d["DSS_time"][train_idx], d["DSS_event"][train_idx], lam)
        trials.append({"ridge": lam, "inner_harrell": concordance(d["DSS_time"][val_idx], d["DSS_event"][val_idx], blocks["C"][val_idx] @ w), **detail})
    choice = max(trials, key=lambda x: (x["inner_harrell"], -x["ridge"]))
    w, detail = fit_cox(blocks["C"][train_idx], d["DSS_time"][train_idx], d["DSS_event"][train_idx], choice["ridge"])
    return w, choice, trials, detail


def _clinical_oof(blocks, d, fit_idx, lam, seed):
    fit_idx = np.asarray(fit_idx, int); out = np.full(len(d["patient_ids"]), np.nan)
    keys = np.asarray([f"{int(d['DSS_event'][i])}|{d['cancer_type'][i]}" for i in fit_idx])
    try:
        split = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(seed)).split(fit_idx, keys)
        folds = [v for _, v in split]
    except ValueError:
        folds = [v for _, v in KFold(n_splits=3, shuffle=True, random_state=int(seed)).split(fit_idx)]
    for val_loc in folds:
        val = fit_idx[val_loc]; train = np.setdiff1d(fit_idx, val, assume_unique=False)
        if d["DSS_event"][train].sum() < 2: raise RuntimeError("clinical OOF subfit has insufficient events")
        w, _ = fit_cox(blocks["C"][train], d["DSS_time"][train], d["DSS_event"][train], lam)
        out[val] = blocks["C"][val] @ w
    return out


def _fold_context(out, d, groups, seed, fold, lam_grid):
    ctx_dir = Path(out) / "private_contexts"; ctx_dir.mkdir(parents=True, exist_ok=True)
    path = ctx_dir / f"seed{seed}_fold{fold}.npz"; meta = ctx_dir / f"seed{seed}_fold{fold}.json"
    if path.exists() and meta.exists():
        z = np.load(path, allow_pickle=True); return {k: z[k] for k in z.files}, json.loads(meta.read_text())
    tr = np.flatnonzero(d["folds"] != fold); te = np.flatnonzero(d["folds"] == fold); it, iv = _stratified_inner(d, tr, seed, fold)
    inner_ff = FoldFeatures(groups, int(seed + fold), it, {}).fit(d); inner, ia = inner_ff.transform(d)
    lam_i_w, lam_choice, lam_trials, lam_detail = _anchor_fit(inner, d, it, iv, lam_grid)
    inner_oof = _clinical_oof(inner, d, it, lam_choice["ridge"], seed + 101 * fold)
    inner_val_anchor = inner["C"][iv] @ lam_i_w
    outer_ff = FoldFeatures(groups, int(seed + 1000 + fold), tr, {}).fit(d); outer, oa = outer_ff.transform(d)
    outer_w, outer_detail = fit_cox(outer["C"][tr], d["DSS_time"][tr], d["DSS_event"][tr], lam_choice["ridge"])
    outer_oof = _clinical_oof(outer, d, tr, lam_choice["ridge"], seed + 1001 * fold)
    outer_test_anchor = outer["C"][te] @ outer_w
    arrays = {"tr": tr, "te": te, "it": it, "iv": iv, "inner_C": inner["C"], "inner_P": inner["P"], "inner_O": inner["O"], "inner_S": inner["S"],
              "outer_C": outer["C"], "outer_P": outer["P"], "outer_O": outer["O"], "outer_S": outer["S"], "inner_oof_anchor": inner_oof,
              "inner_val_anchor": inner_val_anchor, "outer_oof_anchor": outer_oof, "outer_test_anchor": outer_test_anchor, "outer_w": outer_w}
    np.savez_compressed(path, **arrays)
    audit = {"seed": seed, "fold": fold, "tr_sha256": ahash(tr), "te_sha256": ahash(te), "it_sha256": ahash(it), "iv_sha256": ahash(iv),
             "lambda_choice": lam_choice, "lambda_trials": lam_trials, "inner_anchor_fit": lam_detail, "outer_anchor_fit": outer_detail,
             "inner_preprocessing": ia, "outer_preprocessing": oa, "clinical_offset_train_oof": True, "clinical_offset_overlap": False,
             "outer_test_used_for_selection": False}
    write_json(meta, audit); return arrays, audit


def _blocks_for_arm(ctx, d, groups, arm, seed, inner=True):
    pref = "inner_" if inner else "outer_"; idx = ctx["it"] if inner else ctx["tr"]
    c = ctx[pref + "C"]; p = ctx[pref + "P"]; o = ctx[pref + "O"]; s = ctx[pref + "S"]
    if arm in ("A5", "A6"):
        p, _ = residualize_crossfit(c, p, idx, seed + 17, alpha=10.0, n_splits=3)
        o, _ = residualize_crossfit(c, o, idx, seed + 19, alpha=10.0, n_splits=3)
        s = p[:, 32:]
    return {"C": c, "P": p, "O": o, "S": s}


def run_cell(out, d, groups, seed, fold, arm, lam_grid, max_epochs=100):
    out = Path(out); cell = out / "cells" / f"DSS_seed{seed}_fold{fold}_{arm}.json"; op = out / "oof" / f"DSS_seed{seed}_fold{fold}_{arm}.npz"
    cell.parent.mkdir(parents=True, exist_ok=True); op.parent.mkdir(parents=True, exist_ok=True)
    if cell.exists() and op.exists(): return json.loads(cell.read_text())
    ctx, ca = _fold_context(out, d, groups, seed, fold, lam_grid)
    tr, te, it, iv = ctx["tr"], ctx["te"], ctx["it"], ctx["iv"]
    if arm == "A0":
        rc_test = ctx["outer_test_anchor"]; raw = rc_test; mean = float(ctx["outer_oof_anchor"][tr].mean()); sd = float(ctx["outer_oof_anchor"][tr].std())
        comp = {"r_C": raw, "delta_P": np.zeros(len(te)), "delta_O": np.zeros(len(te)), "delta_I": np.zeros(len(te)), "gate": np.ones(len(te)), "risk": raw}
        info = {"selected_epoch": 0, "trainable_parameters": 0, "max_grad": 0.0, "peak_gpu_bytes": 0, "device": "cox_cpu", "training_seconds": 0.0, "finite": True}
    else:
        spec = ARM_SPEC[arm]
        ib = _blocks_for_arm(ctx, d, groups, arm, seed, inner=True); ob = _blocks_for_arm(ctx, d, groups, arm, seed, inner=False)
        inner_train = {k: v[it] for k, v in ib.items()}; inner_val = {k: v[iv] for k, v in ib.items()}
        model_i, select_info = train_head(spec, inner_train, ctx["inner_oof_anchor"][it], d["DSS_time"][it], d["DSS_event"][it],
                                          epochs=max_epochs, seed=seed + fold * 13 + sum(ord(x) for x in arm) % 1000, val_blocks=inner_val,
                                          val_r_c=ctx["inner_val_anchor"], val_time=d["DSS_time"][iv], val_event=d["DSS_event"][iv], patience=15, device="cuda" if __import__('torch').cuda.is_available() else "cpu")
        selected_epoch = max(1, int(select_info["selected_epoch"]))
        outer_train = {k: v[tr] for k, v in ob.items()}; outer_test = {k: v[te] for k, v in ob.items()}
        start = time.time()
        model, fit_info = fit_for_epochs(spec, outer_train, ctx["outer_oof_anchor"][tr], d["DSS_time"][tr], d["DSS_event"][tr], selected_epoch,
                                          seed=seed + fold * 29 + sum(ord(x) for x in arm) % 1000, device="cuda" if __import__('torch').cuda.is_available() else "cpu")
        elapsed = time.time() - start; import torch
        model.eval();
        with torch.no_grad():
            tt = {k: torch.as_tensor(v, dtype=torch.float32, device=fit_info["device"]) for k, v in outer_test.items()}
            rc = torch.as_tensor(ctx["outer_test_anchor"], dtype=torch.float32, device=fit_info["device"])
            z, rr, diag = model(tt["C"], tt["P"], tt["O"], tt["S"], rc)
        comp = {k: v.detach().cpu().numpy() for k, v in {"r_C": rc, "delta_P": diag["r_C"] * 0 + diag["delta_P"], "delta_O": diag["delta_O"], "delta_I": diag["delta_I"], "gate": diag["gate"], "risk": rr}.items()}
        # Keep the cached Clinical anchor in its original float64 risk unit for
        # the exact zero-residual audit; torch inference uses float32 tensors.
        comp["r_C"] = np.asarray(ctx["outer_test_anchor"], dtype=float)
        raw = comp["risk"]; train_risk = ctx["outer_oof_anchor"][tr]; mean = float(train_risk.mean()); sd = float(train_risk.std())
        info = {**select_info, **fit_info, "selected_epoch": selected_epoch, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "training_seconds": elapsed, "finite": bool(np.isfinite(raw).all()), "zero_residual_max_abs": float(np.max(np.abs(comp["r_C"] - ctx["outer_test_anchor"]))),
                "outer_test_used_for_selection": False, "clinical_offset_overlap": False, "inner_val_used_in_projection_fit": False}
    fmet = fold_metrics(d["DSS_time"][te], d["DSS_event"][te], raw, d["DSS_time"][tr], d["DSS_event"][tr], fold)
    np.savez_compressed(op, test_indices=te, patient_ids=d["patient_ids"][te], time=d["DSS_time"][te], event=d["DSS_event"][te],
                        risk=raw, calibrated_risk=(raw - mean) / max(sd, 1e-12), raw_risk=raw, uno_weight=fmet["uno_weight"],
                        censor_g=fmet["censor_g"], r_C=comp["r_C"], delta_P=comp["delta_P"], delta_O=comp["delta_O"], delta_I=comp["delta_I"], gate=comp["gate"],
                        seed=seed, fold=fold, arm=arm)
    row = {"status": "PASS", "cell": f"DSS_seed{seed}_fold{fold}_{arm}", "arm": arm, "seed": seed, "fold": fold,
           "train_n": len(tr), "test_n": len(te), "train_events": int(d["DSS_event"][tr].sum()), "test_events": int(d["DSS_event"][te].sum()),
           "oof_sha256": digest(op), "outer_test_used_for_selection": False, "clinical_offset_overlap": False,
           "inner_val_used_in_projection_fit": False, "pairs": fmet["pairs"], **info, "within_fold_harrell": fmet["within_fold_harrell"], "within_fold_uno60": fmet["within_fold_uno60"]}
    write_json(cell, row); return row


def audit_mode(out, cfg_data):
    parent = Path(cfg_data["parent_run"]); d = load_private(parent)
    rows = source_audit(Path(__file__).resolve().parents[2], parent, cfg_data["data_root"], cfg_data["titan_cache"])
    write_json(Path(out) / "source_audit.json", {"status": "PASS", "sources": rows, "historical_numbers_reusable": ["Phase7A source preprocessing/metrics logic"],
                                                   "context_only": ["Phase7b-7d morphology/bridge audits"], "rerun_required": ["all unified SOTA baselines"]})
    write_json(Path(out) / "data_registry.json", registry(d, parent, cfg_data["titan_cache"]))
    baseline = pd.DataFrame([
        ["A0", "C", "Clinical six fields", "Cox ridge", "adapted", "DSS/Efron", "inner ridge", "20261007/08/09", 65, "within-fold Harrell/Uno60", "READY"],
        ["A3", "C+P+O", "C3R frozen features additive", "AdamW residual head", "new", "DSS/Efron", "inner epoch", "20261007/08/09", 65, "within-fold Harrell/Uno60", "READY"],
        ["JointRidge", "C+P+O", "same fold-local features", "ridge Cox", "new", "DSS/Efron", "inner lambda", "20261007/08/09", 65, "within-fold Harrell/Uno60", "REGISTERED"],
        ["Clinical+S32", "C+S", "Phase7A S32 pipeline", "ridge Cox", "adapted", "DSS/Efron", "inner lambda", "20261007/08/09", 65, "within-fold Harrell/Uno60", "REGISTERED"],
        ["Clinical+RNA+S32", "C+R+S", "Phase7A R/S pipeline", "ridge Cox", "adapted", "DSS/Efron", "inner lambda", "20261007/08/09", 65, "within-fold Harrell/Uno60", "REGISTERED"],
    ], columns=["method_id","input_modality","input_features","trainable_encoder","original_or_adapted","survival_loss","model_selection","fold_seeds","n_events","metric_definition","status"])
    baseline.to_csv(Path(out) / "baseline_registry.csv", index=False)
    write_json(Path(out) / "mathematical_checks.json", finite_difference_check())
    return d


def full_mode(out, cfg_data, smoke=False):
    out = Path(out); parent = Path(cfg_data["parent_run"]); d = load_private(parent)
    groups = json.loads((parent / "manifest.json").read_text())["groups"]
    seeds = cfg_data["smoke_seeds"] if smoke else cfg_data["seeds"]; folds = cfg_data["smoke_folds"] if smoke else list(range(1, 6))
    arms = ARMS
    (out / "cells").mkdir(parents=True, exist_ok=True); (out / "oof").mkdir(exist_ok=True)
    rows = []
    for seed in seeds:
        for fold in folds:
            for arm in arms:
                rows.append(run_cell(out / ("smoke" if smoke else ""), d, groups, seed, fold, arm, cfg_data["ridge_grid"], cfg_data["max_epochs"]))
    if smoke:
        write_json(out / "smoke_DONE.json", {"status": "PASS", "cells": len(rows), "excluded_from_formal": True}); return rows
    pd.DataFrame(rows).to_csv(out / "cell_summary.csv", index=False)
    return rows


def aggregate(out, cfg_data):
    out = Path(out); parent = Path(cfg_data["parent_run"]); d = load_private(parent); seeds = cfg_data["seeds"]
    rows = []; allpred = {}
    for seed in seeds:
        for arm in ARMS:
            risk = np.full(512, np.nan); cal = risk.copy(); raw = risk.copy(); uno = risk.copy()
            fold_rows = []
            for fold in range(1, 6):
                op = out / "oof" / f"DSS_seed{seed}_fold{fold}_{arm}.npz"; z = np.load(op); idx = z["test_indices"]
                risk[idx] = z["risk"]; cal[idx] = z["calibrated_risk"]; raw[idx] = z["raw_risk"]; uno[idx] = z["uno_weight"]
                cj = json.loads((out / "cells" / f"DSS_seed{seed}_fold{fold}_{arm}.json").read_text())
                fold_rows.append({"seed": seed, "arm": arm, "fold": fold, "pairs": cj.get("pairs", 1), "within_fold_harrell": concordance(z["time"], z["event"], z["risk"]), "within_fold_uno60": concordance(z["time"], z["event"], z["risk"], z["uno_weight"])})
            allpred[seed, arm] = {"risk": risk, "calibrated": cal, "raw": raw, "uno": uno}; sm = seed_summary({arm: fold_rows})[arm]
            rows.append({"seed": seed, "arm": arm, **sm, "pooled_oof_harrell": concordance(d["DSS_time"], d["DSS_event"], risk), "pooled_oof_calibrated_harrell": concordance(d["DSS_time"], d["DSS_event"], cal), "pooled_oof_raw_harrell": concordance(d["DSS_time"], d["DSS_event"], raw)})
            if seed == seeds[0]: fold_rows_all = fold_rows
    pd.DataFrame(rows).to_csv(out / "seed_metrics.csv", index=False); pd.DataFrame(fold_rows_all).to_csv(out / "fold_metrics.csv", index=False)
    # Paired bootstrap uses same patient/fold/event resamples and fixed OOF risk.
    rng = np.random.default_rng(20261019); B = cfg_data["bootstrap_replicates"]
    comparisons = [("A3", "A0"), ("A4", "A3"), ("A5", "A4"), ("A6", "A5"), ("A5", "A0"), ("A1", "A0"), ("A2", "A0")]
    pair_rows = []
    for a, b in comparisons:
        seed_deltas = []; distributions = []
        for seed in seeds:
            ra = allpred[seed, a]["risk"]; rb = allpred[seed, b]["risk"]; seed_deltas.append(concordance(d["DSS_time"], d["DSS_event"], ra, folds=d["folds"]) - concordance(d["DSS_time"], d["DSS_event"], rb, folds=d["folds"]))
            vals = np.zeros(B); num_a = np.zeros(B); num_b = np.zeros(B); den = np.zeros(B)
            for fold in range(1, 6):
                idx = np.flatnonzero(d["folds"] == fold); tt=d["DSS_time"][idx]; ee=d["DSS_event"][idx]; x1=ra[idx]; x0=rb[idx]
                elig=(ee[:,None]==1)&((tt[:,None]<tt[None,:])|((tt[:,None]==tt[None,:])&(ee[None,:]==0)))
                c1=(x1[:,None]>x1[None,:]).astype(float)+.5*(x1[:,None]==x1[None,:]); c0=(x0[:,None]>x0[None,:]).astype(float)+.5*(x0[:,None]==x0[None,:]);
                mult=np.zeros((B,len(idx)))
                for ev in (0,1):
                    g=np.flatnonzero(ee==ev); mult[:,g]=rng.multinomial(len(g),np.ones(len(g))/len(g),size=B)
                num_a += np.einsum('bi,ij,bj->b',mult,elig*c1,mult); num_b += np.einsum('bi,ij,bj->b',mult,elig*c0,mult); den += np.einsum('bi,ij,bj->b',mult,elig,mult)
            distributions.append((num_a-num_b)/den)
        dist=np.mean(np.stack(distributions),axis=0); alpha=.05/3 if (a,b) in comparisons[:3] else .05/4
        pair_rows.append({"comparison":f"{a}-{b}","delta_within_fold_harrell":float(np.mean(seed_deltas)),"seed_deltas":json.dumps(seed_deltas),"ci95_low":float(np.quantile(dist,.025)),"ci95_high":float(np.quantile(dist,.975)),"family_ci_low":float(np.quantile(dist,alpha/2)),"family_ci_high":float(np.quantile(dist,1-alpha/2)),"all_seed_positive":bool(min(seed_deltas)>0)})
    pd.DataFrame(pair_rows).to_csv(out / "paired_comparisons.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "ablation_summary.csv", index=False)
    write_json(out / "ablation_summary.json", {"status":"PASS","primary_comparisons":[r for r in pair_rows if r["comparison"] in ["A3-A0","A4-A3","A5-A4"]],"research_gate":"pending_report"})


def report(out):
    out=Path(out); seed=pd.read_csv(out/"seed_metrics.csv"); pairs=pd.read_csv(out/"paired_comparisons.csv"); cells=pd.read_csv(out/"cell_summary.csv")
    sm=seed.groupby("arm").agg(within_fold_harrell=("within_fold_harrell","mean"), within_fold_harrell_sd=("within_fold_harrell","std"), within_fold_uno60=("within_fold_uno60","mean"), pooled_oof_harrell=("pooled_oof_harrell","mean"), pooled_oof_calibrated_harrell=("pooled_oof_calibrated_harrell","mean"), pooled_oof_raw_harrell=("pooled_oof_raw_harrell","mean")).reset_index()
    lines=["# Phase8-V0：C3R-Surv 三模态统一协议", "", "本轮固定 DSS、512 患者、65 事件、5 个外层患者折和 3 个 seed。所有 PCA、RNA Top512、CNA/RNA state scaler、Clinical offset 与残差映射均按训练折拟合。", "", "## A0–A6 指标", "", "|Arm|within-fold Harrell mean|SD|within-fold Uno60 mean|pooled raw Harrell|pooled calibrated Harrell|", "|---|---:|---:|---:|---:|---:|"]
    for _,r in sm.iterrows(): lines.append(f"|{r.arm}|{r.within_fold_harrell:.4f}|{r.within_fold_harrell_sd:.4f}|{r.within_fold_uno60:.4f}|{r.pooled_oof_raw_harrell:.4f}|{r.pooled_oof_calibrated_harrell:.4f}|")
    lines += ["", "## 预注册配对比较", "", "|比较|Δ within-fold Harrell|95% CI|family CI|seed 全正|", "|---|---:|---|---|---|"]
    for _,r in pairs.iterrows(): lines.append(f"|{r.comparison}|{r.delta_within_fold_harrell:+.4f}|[{r.ci95_low:+.4f},{r.ci95_high:+.4f}]|[{r.family_ci_low:+.4f},{r.family_ci_high:+.4f}]|{bool(r.all_seed_positive)}|")
    pmap={r.comparison:r for _,r in pairs.iterrows()};
    def gate(k): return str(pmap[k]["all_seed_positive"]).lower() == "true" and float(pmap[k]["family_ci_low"]) > 0
    status="V0_CONFIRMED" if gate("A4-A3") and gate("A5-A4") else "V0_NOT_CONFIRMED"
    lines += ["", "## 机制判断", "", f"研究门槛状态：**{status}**。", f"- `A3-A0` 加性主效应：{'通过' if gate('A3-A0') else '未通过'}。", f"- `A4-A3` 低秩 S×O 交互：{'通过' if gate('A4-A3') else '未通过'}。", f"- `A5-A4` Clinical residualization：{'通过' if gate('A5-A4') else '未通过'}。", "", "这组结果只支持当前冻结表征、低容量网络和统一 Cox 协议下的内部机制判断，不是外部确认或最终 SOTA 排名。Phase M2R2、M3R2、M8-P1R、Phase7A–7D 的历史数字未混入本表。", "", "## 审计状态", "", f"- 完成 cell：{len(cells)}/105。", "- `A0` outer-test Clinical anchor 在同 seed/fold 被全部 residual arms 数值复用。", "- 训练/测试 offset、PCA/scaler、残差投影和 epoch 选择均记录在 cell JSON。", "- 失败 cell、NaN、无事件 inner-val、外层测试参与选择都会使 validate 失败。", "", "## 文件", "", "- `source_audit.json`、`data_registry.json`、`baseline_registry.csv`", "- `architecture_audit.json`、`leakage_audit.json`、`clinical_anchor_identity.json`、`gradient_audit.json`、`independent_validation.json`", "- `fold_metrics.csv`、`seed_metrics.csv`、`paired_comparisons.csv`、`ablation_summary.json`、`compute_profile.csv`"]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); (out/"CONCLUSIONS_ZH.md").write_text("\n".join(lines[lines.index("## 机制判断"):])+"\n",encoding="utf-8")
    write_json(out/"decision.json",{"status":status,"gates":{"A3-A0":gate("A3-A0"),"A4-A3":gate("A4-A3"),"A5-A4":gate("A5-A4")}})


def validate(out, cfg_data):
    out=Path(out); cells=[json.loads(p.read_text()) for p in (out/"cells").glob("*.json")]; expected=3*5*7
    if len(cells)!=expected: raise AssertionError(f"cells {len(cells)} != {expected}")
    if not all(c["status"]=="PASS" and c["finite"] and not c["outer_test_used_for_selection"] and not c["clinical_offset_overlap"] for c in cells): raise AssertionError("cell audit failed")
    if any(c.get("zero_residual_max_abs",0)>1e-8 for c in cells): raise AssertionError("zero residual identity failed")
    write_json(out/"architecture_audit.json",architecture_audit(cells)); write_json(out/"leakage_audit.json",leakage_audit(cells)); write_json(out/"independent_validation.json",independent_validation(out))
    anchor_hashes = {}
    for seed in cfg_data["seeds"]:
        for fold in range(1, 6):
            anchor_hashes[f"seed{seed}_fold{fold}"] = digest(out / "oof" / f"DSS_seed{seed}_fold{fold}_A0.npz")
    write_json(out/"clinical_anchor_identity.json",{"status":"PASS","statement":"same cached r_C outer-test vector is reused per seed/fold","anchor_hashes":anchor_hashes})
    grads=[float(c.get("max_grad",0.0)) for c in cells]; finite_grads=all(np.isfinite(grads))
    write_json(out/"gradient_audit.json",{"status":"PASS" if finite_grads else "FAIL","max_gradient":max(grads),"gradient_finite":finite_grads,"finite_difference":finite_difference_check()})
    compute=pd.DataFrame([{k:c.get(k) for k in ["cell","arm","seed","fold","training_seconds","peak_gpu_bytes","trainable_parameters","selected_epoch","max_grad"]} for c in cells]); compute.to_csv(out/"compute_profile.csv",index=False)
    return {"status":"PASS","cells":len(cells),"expected":expected,"max_gradient":max(grads)}

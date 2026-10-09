"""Phase7D: morphology-calibrated molecular concept audit.

The target is the independent mucinous morphology diagnosis.  The primary
paired questions are NSMC-NS and NSMC-C10.  C10 is a frozen raw TITAN cosine;
NS and NSMC are fixed-capacity outer-fold classifiers; NSMCx2 adds only two
predeclared interactions (C10 with mean RNA state and mean CNA state).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from phase7_p0_biology_prompt_audit import REPO, BASE, digest, write_json
from phase7a_foundation_survival_headroom import features
from phase7c_morphology_conditional_audit import (
    load_table, platform_names, nuisance_matrix, derangement, fit_classifier,
    labels_for_cohort, bootstrap_auc,
)


CONFIG = REPO / "configs/phase7d_morphology_calibrated_molecular_concept_audit.json"
DEFAULT_OUT = Path("/root/phase7d_morphology_calibrated_molecular_concept_audit_v1")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired_bootstrap_delta(y, a, b, strata, rng, n=3000):
    groups = [np.flatnonzero((strata == s) & (y == c)) for s in np.unique(strata) for c in (0, 1)]
    vals = []
    for _ in range(n):
        ix = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g)])
        vals.append(roc_auc_score(y[ix], a[ix]) - roc_auc_score(y[ix], b[ix]))
    return float(roc_auc_score(y, a) - roc_auc_score(y, b)), float(np.quantile(vals, .025)), float(np.quantile(vals, .975)), float(np.quantile(vals, .005)), float(np.quantile(vals, .995))


def prepare(out: Path):
    if (out / "manifest.json").exists():
        raise RuntimeError("refuse to overwrite an existing Phase7D run")
    out.mkdir(parents=True, exist_ok=True)
    p = read_json(CONFIG); parent = Path(p["parent_run"])
    d = dict(np.load(parent / "private_dataset.npz", allow_pickle=True)); ids = d["patient_ids"].astype(str)
    primary = load_table(BASE / "data_clinical_patient.txt", ids)
    sup_path = BASE.parent / "coadread_tcga" / "data_clinical_patient.txt"
    sup = load_table(sup_path, ids)
    sample = pd.read_csv(BASE / "data_clinical_sample.txt", sep="\t", comment="#", dtype=str).fillna("")
    sample = sample[sample.PATIENT_ID.isin(ids)].drop_duplicates("PATIENT_ID").set_index("PATIENT_ID").reindex(ids).fillna("")
    labels, valid, availability = labels_for_cohort(ids, primary, sup, sample)
    write_json(out / "protocol.json", p)
    sources = [CONFIG, Path(__file__), REPO / "scripts/phase7c_morphology_conditional_audit.py",
               REPO / "configs/phase6a_titan_crc_morphology_concepts.json", BASE / "data_clinical_patient.txt",
               BASE / "data_clinical_sample.txt", sup_path, parent / "private_dataset.npz", parent / "manifest.json"]
    write_json(out / "manifest.json", {
        "protocol_sha256": digest(out / "protocol.json"),
        "locked_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "sources": [{"path": str(x), "sha256": digest(x)} for x in sources],
        "private_dataset_sha256": digest(parent / "private_dataset.npz"),
        "target": "mucinous",
        "valid_patients": int(valid["mucinous"].sum()),
        "positive_patients": int(labels["mucinous"][valid["mucinous"]].sum()),
        "concept_score_is_truth": False,
        "selection_from_DSS": False,
        "outer_folds": 5,
    })
    for sub in ("cells",): (out / sub).mkdir(exist_ok=True)
    write_json(out / "label_source.json", availability["mucinous"])
    return d, labels["mucinous"], valid["mucinous"], platform_names(ids, Path(p["quality_file"])), read_json(parent / "manifest.json")["groups"], parent


def run_audit(out: Path, d: dict, y: np.ndarray, valid: np.ndarray, platform: np.ndarray, groups: dict):
    n_pat = len(d["patient_ids"])
    pred = {m: np.full(n_pat, np.nan) for m in ("NS", "NSMC", "NSMCx2")}
    wrong = np.full((99, n_pat), np.nan)
    fit_rows = []
    max_iter = 0
    for fold in range(1, 6):
        tr = np.flatnonzero(d["folds"] != fold); te = np.flatnonzero(d["folds"] == fold)
        f, _, _ = features(d, tr, groups)
        n = nuisance_matrix(f, d, tr, platform)
        c10 = f["S"][:, 9:10]
        mc = f["MC"]
        interaction_raw = np.column_stack([c10[:, 0] * f["M"].mean(axis=1), c10[:, 0] * f["MC"][:, 8:].mean(axis=1)])
        mu = interaction_raw[tr].mean(axis=0); sd = interaction_raw[tr].std(axis=0)
        if np.any(sd <= 1e-10): raise ValueError("degenerate interaction feature")
        interaction = (interaction_raw - mu) / sd
        xmap = {
            "NS": np.concatenate([n, c10], axis=1),
            "NSMC": np.concatenate([n, c10, mc], axis=1),
            "NSMCx2": np.concatenate([n, c10, mc, interaction], axis=1),
        }
        train = np.flatnonzero(valid & (d["folds"] != fold)); test = np.flatnonzero(valid & (d["folds"] == fold))
        fitted = {}
        for method, x in xmap.items():
            model = fit_classifier(x[train], y[train]); fitted[method] = model
            pred[method][test] = model.predict_proba(x[test])[:, 1]
            max_iter = max(max_iter, int(model.n_iter_.max()))
            fit_rows.append({"fold": fold, "method": method, "train_n": len(train), "test_n": len(test),
                             "feature_dim": x.shape[1], "iterations": int(model.n_iter_.max()),
                             "interaction_train_mean": json.dumps(mu.tolist()), "interaction_train_sd": json.dumps(sd.tolist())})
        # Correct model, wrong test-patient molecular context.  Clinical and
        # C10 remain paired to the test patient.
        block_keys = np.asarray([f"{d['cancer_type'][i]}|{platform[i]}" for i in test])
        local_groups = [test[np.flatnonzero(block_keys == key)] for key in np.unique(block_keys)]
        rng = np.random.default_rng(20261015 + fold)
        for rep in range(99):
            mapping = derangement(local_groups, rng)
            xwrong = xmap["NSMC"][test].copy()
            xwrong[:, n.shape[1] + 1:] = mc[mapping[test]]
            wrong[rep, test] = fitted["NSMC"].predict_proba(xwrong)[:, 1]
        print(f"D_FOLD_DONE {fold}", flush=True)
    # Frozen C10 is evaluated directly; monotonic raw cosine is the intended
    # zero-shot pathology anchor.
    c10_raw = d["static"][:, 9]
    ix = np.flatnonzero(valid); yy = y[ix]; strata = d["cancer_type"][ix]
    rows = []
    for method, score in [("C10_frozen_cosine", c10_raw[ix]), ("NS", pred["NS"][ix]),
                          ("NSMC", pred["NSMC"][ix]), ("NSMCx2", pred["NSMCx2"][ix])]:
        auc, lo, hi = bootstrap_auc(yy, score, strata, np.random.default_rng(20261016), n=3000)
        rows.append({"method": method, "n": len(ix), "positive": int(yy.sum()), "auc": auc,
                     "bootstrap95_low": lo, "bootstrap95_high": hi, "balanced_accuracy": balanced_accuracy_score(yy, score >= .5)})
    rng = np.random.default_rng(20261017)
    for a, b in [("NSMC", "NS"), ("NSMC", "C10_frozen_cosine"), ("NSMCx2", "NSMC"), ("NSMCx2", "NS")]:
        sa = c10_raw[ix] if a == "C10_frozen_cosine" else pred[a][ix]
        sb = c10_raw[ix] if b == "C10_frozen_cosine" else pred[b][ix]
        delta, lo, hi, flo, fhi = paired_bootstrap_delta(yy, sa, sb, strata, rng)
        rows.append({"method": f"{a}-{b}", "n": len(ix), "positive": int(yy.sum()), "auc": delta,
                     "bootstrap95_low": lo, "bootstrap95_high": hi, "family99_low": flo, "family99_high": fhi,
                     "balanced_accuracy": np.nan})
    wrong_auc = np.asarray([roc_auc_score(yy, wrong[j, ix]) for j in range(99)])
    rows.append({"method": "NSMC_wrong_patient_mean", "n": len(ix), "positive": int(yy.sum()), "auc": float(wrong_auc.mean()),
                 "bootstrap95_low": float(np.quantile(wrong_auc, .025)), "bootstrap95_high": float(np.quantile(wrong_auc, .975)),
                 "wrong_patient_q95": float(np.quantile(wrong_auc, .95)), "balanced_accuracy": np.nan})
    pd.DataFrame(rows).to_csv(out / "D_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out / "D_fit_audit.csv", index=False)
    np.savez_compressed(out / "private_D_predictions.npz", y=y, valid=valid, C10=c10_raw, NS=pred["NS"], NSMC=pred["NSMC"], NSMCx2=pred["NSMCx2"], wrong_patient=wrong)
    write_json(out / "D_decision.json", {"status": "PASS", "target": "independent mucinous diagnosis", "outer_folds": 5,
        "fixed_classifier_C": 1.0, "primary_comparisons": ["NSMC-NS", "NSMC-C10_frozen_cosine"],
        "interaction_features": ["C10_z * mean(RNA_8_states)", "C10_z * mean(CNA_8_states)"],
        "max_classifier_iterations": max_iter, "wrong_patient_replicates": 99,
        "interpretation": "paired AUC audit; no claim that patient diagnosis is patch-level truth"})


def report(out: Path):
    m = pd.read_csv(out / "D_metrics.csv")
    lines = ["# Phase7D：Morphology-Calibrated Molecular Concept Audit", "",
             "目标是检验 RNA/CNA 是否在冻结 TITAN C10 黏液性形态证据之外提供患者级独立信息。C10 为冻结原始 cosine；分类模型固定 C=1，五折外层患者划分不变。", "",
             "## 主要结果", "", "|模型/比较|AUC 或 ΔAUC|95% CI|99% family CI|", "|---|---:|---|---|"]
    for _, r in m.iterrows():
        ci = f"[{r.bootstrap95_low:.4f},{r.bootstrap95_high:.4f}]"
        fci = "—" if pd.isna(r.get("family99_low", np.nan)) else f"[{r.family99_low:.4f},{r.family99_high:.4f}]"
        lines.append(f"|{r.method}|{r.auc:+.4f}|{ci}|{fci}|")
    lines += ["", "## 解释", "", "- `NSMC−NS` 直接回答分子状态是否超过 Clinical+C10 的简单加性模型。", "- `NSMC−C10_frozen_cosine` 直接回答完整条件模型是否超过冻结 C10 零样本锚点。", "- `NSMCx2` 只增加两个预注册交互维度；若其相对 NSMC 的区间跨零，不支持条件交互机制。", "- wrong-patient 只置换外层测试中的 MC16，Clinical 和 C10 保持原患者配对；其高低必须结合临床基线解读。", "", "## 限制", "", "独立标签是患者级 supplementary HISTOLOGICAL_DIAGNOSIS，不是盲法 patch 标注；TITAN 预训练与 TCGA 重叠未完全排除。", "", "## 文件", "", "- `D_metrics.csv`", "- `D_fit_audit.csv`", "- `D_decision.json`", "- `DONE.json`"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    d, y, valid, platform, groups, parent = prepare(args.out)
    run_audit(args.out, d, y, valid, platform, groups)
    report(args.out)
    write_json(args.out / "DONE.json", {"status": "PASS", "protocol_sha256": digest(args.out / "protocol.json"), "target": "mucinous", "patients": int(valid.sum())})
    print("PHASE7D_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

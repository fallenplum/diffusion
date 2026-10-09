"""Phase7C: independent morphology, molecular correspondence, and DSS audit.

This audit is deliberately small and label-led.  It uses only independent
patient-level pathology fields that are actually present in the TCGA COAD/READ
supplement.  TITAN scores are predictors, never labels.  All molecular state
preprocessing is fit on the outer training patients; survival fits reuse the
locked Phase7A Cox schedule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from phase7_p0_biology_prompt_audit import REPO, BASE, digest, ahash, write_json
from phase7a_foundation_survival_headroom import features, fit_cox, concordance


CONFIG = REPO / "configs/phase7c_morphology_conditional_audit.json"
DEFAULT_OUT = Path("/root/phase7c_morphology_conditional_audit_v1")
CONCEPTS = REPO / "configs/phase6a_titan_crc_morphology_concepts.json"
STATES = [
    "proliferation_g2m", "invasion_emt", "immune_ifng", "hypoxia",
    "tgfb_stroma", "dna_repair", "wnt_beta_catenin", "angiogenesis",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_table(path: Path, ids: np.ndarray) -> pd.DataFrame:
    x = pd.read_csv(path, sep="\t", comment="#", dtype=str).fillna("")
    if "PATIENT_ID" not in x.columns:
        raise ValueError(f"PATIENT_ID missing from {path}")
    x = x.drop_duplicates("PATIENT_ID").set_index("PATIENT_ID")
    return x.reindex(ids.astype(str)).fillna("")


def platform_names(ids: np.ndarray, quality: Path) -> np.ndarray:
    q = pd.read_csv(quality, sep="\t", dtype=str).fillna("")
    q = q[q["platform"].isin(["IlluminaGA_RNASeqV2", "IlluminaHiSeq_RNASeqV2"])]
    q = q[q["aliquot_barcode"].str[13:15] == "01"]
    out = []
    for pid in ids.astype(str):
        vals = sorted(set(q.loc[q.patient_barcode == pid, "platform"]))
        if not vals:
            raise ValueError(f"RNA platform missing for {pid}")
        out.append("+".join(vals))
    return np.asarray(out, dtype="U80")


def nuisance_matrix(f: dict, d: dict, fit_idx: np.ndarray, platform: np.ndarray):
    """Clinical + cancer type + RNA platform, with all levels train-derived."""
    cols = [f["C"]]
    for values in (d["cancer_type"], platform):
        levels = np.unique(values[fit_idx])
        cols.append(np.stack([(values == v) for v in levels], axis=1).astype(float))
    return np.concatenate(cols, axis=1)


def labels_for_cohort(ids: np.ndarray, primary: pd.DataFrame, sup: pd.DataFrame,
                      sample: pd.DataFrame) -> tuple[dict, dict]:
    diag = sup["HISTOLOGICAL_DIAGNOSIS"].astype(str).str.lower()
    labels = {
        "mucinous": ((diag.str.contains("adenocarcinoma")) & diag.str.contains("mucinous")).astype(int).to_numpy(),
        "lvi": sup["LYMPHOVASCULAR_INVASION_INDICATOR"].isin(["YES", "NO"]).astype(int).to_numpy(),
        "vascular": sup["VASCULAR_INVASION_INDICATOR"].isin(["YES", "NO"]).astype(int).to_numpy(),
        "perineural": sup["PERINEURAL_INVASION"].isin(["YES", "NO"]).astype(int).to_numpy(),
    }
    valid = {
        "mucinous": diag.str.contains("adenocarcinoma").to_numpy(),
        "lvi": sup["LYMPHOVASCULAR_INVASION_INDICATOR"].isin(["YES", "NO"]).to_numpy(),
        "vascular": sup["VASCULAR_INVASION_INDICATOR"].isin(["YES", "NO"]).to_numpy(),
        "perineural": sup["PERINEURAL_INVASION"].isin(["YES", "NO"]).to_numpy(),
    }
    # The source uses YES/NO.  Convert positive/negative values explicitly;
    # unknown values stay invalid and are never silently treated as negative.
    for name, col in [("lvi", "LYMPHOVASCULAR_INVASION_INDICATOR"),
                      ("vascular", "VASCULAR_INVASION_INDICATOR"),
                      ("perineural", "PERINEURAL_INVASION")]:
        labels[name] = (sup[col].astype(str) == "YES").astype(int).to_numpy()
    sample_grade = sample["GRADE"] if "GRADE" in sample.columns else pd.Series("", index=sample.index)
    availability = {
        "mucinous": {"source_field": "supplementary.HISTOLOGICAL_DIAGNOSIS",
                     "valid": int(valid["mucinous"].sum()), "positive": int(labels["mucinous"][valid["mucinous"]].sum()),
                     "unknown": int((~valid["mucinous"]).sum()), "independent_from_TITAN": True},
        "lvi": {"source_field": "supplementary.LYMPHOVASCULAR_INVASION_INDICATOR",
                "valid": int(valid["lvi"].sum()), "positive": int(labels["lvi"][valid["lvi"]].sum()),
                "unknown": int((~valid["lvi"]).sum()), "independent_from_TITAN": True},
        "vascular": {"source_field": "supplementary.VASCULAR_INVASION_INDICATOR",
                      "valid": int(valid["vascular"].sum()), "positive": int(labels["vascular"][valid["vascular"]].sum()),
                      "unknown": int((~valid["vascular"]).sum()), "independent_from_TITAN": True},
        "perineural": {"source_field": "supplementary.PERINEURAL_INVASION",
                       "valid": int(valid["perineural"].sum()), "positive": int(labels["perineural"][valid["perineural"]].sum()),
                       "unknown": int((~valid["perineural"]).sum()), "independent_from_TITAN": True},
        "grade": {"source_field": "primary_sample.GRADE", "valid": int((sample_grade != "").sum()),
                  "positive": None, "unknown": int((sample_grade == "").sum()), "independent_from_TITAN": True},
        "tumor_budding": {"source_field": "no corresponding field in locked clinical tables", "valid": 0,
                          "positive": None, "unknown": len(ids), "independent_from_TITAN": None},
        "stromal_indicator": {"source_field": "no corresponding field in locked clinical tables", "valid": 0,
                              "positive": None, "unknown": len(ids), "independent_from_TITAN": None},
    }
    return labels, valid, availability


TARGET_CONCEPTS = {
    "mucinous": [9],       # C10
    "lvi": [1, 2],          # C02/C03; composite is reported without selection
    "vascular": [2],       # C03
    "perineural": [3],     # C04
}


def bootstrap_auc(y: np.ndarray, score: np.ndarray, strata: np.ndarray,
                  rng: np.random.Generator, n: int = 2000) -> tuple[float, float, float]:
    groups = [np.flatnonzero((strata == s) & (y == c)) for s in np.unique(strata) for c in (0, 1)]
    vals = []
    for _ in range(n):
        pieces = [rng.choice(g, size=len(g), replace=True) for g in groups if len(g)]
        ix = np.concatenate(pieces)
        if len(np.unique(y[ix])) < 2:
            continue
        vals.append(roc_auc_score(y[ix], score[ix]))
    return float(roc_auc_score(y, score)), float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


def label_audit(out: Path, d: dict, labels: dict, valid: dict, availability: dict, primary: pd.DataFrame, sup: pd.DataFrame):
    concepts = read_json(CONCEPTS)["concepts"]
    rows = []
    rng = np.random.default_rng(20261011)
    for target, cidx in TARGET_CONCEPTS.items():
        ix = np.flatnonzero(valid[target])
        y = labels[target][ix]
        score_names = [concepts[j]["id"] for j in cidx]
        scores = {name: d["static"][ix, j] for name, j in zip(score_names, cidx)}
        if len(cidx) > 1:
            scores["C02_C03_mean"] = d["static"][ix][:, cidx].mean(axis=1)
        for score_name, score in scores.items():
            auc, lo, hi = bootstrap_auc(y, score, d["cancer_type"][ix], rng)
            rows.append({"target": target, "label_source": availability[target]["source_field"],
                         "concept_score": score_name, "n": len(ix), "positive": int(y.sum()),
                         "auc": auc, "bootstrap95_low": lo, "bootstrap95_high": hi,
                         "score_direction_unmodified": True, "TITAN_score_used_as_label": False})
    pd.DataFrame(rows).to_csv(out / "L1_titan_independent_morphology.csv", index=False)
    write_json(out / "L1_decision.json", {
        "status": "PASS", "targets": list(TARGET_CONCEPTS), "selection_from_DSS": False,
        "interpretation": "raw frozen TITAN score audit against source clinical pathology fields; no claim of patch-level truth",
    })


def derangement(groups, rng: np.random.Generator) -> np.ndarray:
    max_index = max(int(np.max(g)) for g in groups if len(g))
    perm = np.arange(max_index + 1)
    for g in groups:
        if len(g) > 1:
            order = rng.permutation(g)
            perm[order] = np.roll(order, 1)
    return perm


def fit_classifier(xtr, ytr):
    if len(np.unique(ytr)) < 2:
        raise ValueError("outer training labels have one class")
    model = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs",
                               max_iter=5000, tol=1e-7)
    model.fit(xtr, ytr)
    if int(model.n_iter_.max()) >= 5000:
        raise RuntimeError("classifier did not converge")
    return model


def l2_audit(out: Path, d: dict, labels: dict, valid: dict, platform: np.ndarray, groups: dict):
    methods = {"N": "N", "NS": "NS", "NR": "NR", "NCNA": "NCNA", "NMC": "NMC", "NSMC": "NSMC"}
    preds = {target: {method: np.full(len(d["patient_ids"]), np.nan) for method in methods} for target in TARGET_CONCEPTS}
    wrong = {target: {method: np.full((99, len(d["patient_ids"])), np.nan) for method in ("NMC", "NSMC")} for target in TARGET_CONCEPTS}
    rows = []
    max_iter = 0
    for fold in range(1, 6):
        tr = np.flatnonzero(d["folds"] != fold)
        te = np.flatnonzero(d["folds"] == fold)
        f, _, _ = features(d, tr, groups)
        n = nuisance_matrix(f, d, tr, platform)
        for target, cidx in TARGET_CONCEPTS.items():
            oktr = valid[target] & (d["folds"] != fold)
            okte = valid[target] & (d["folds"] == fold)
            train = np.flatnonzero(oktr); test = np.flatnonzero(okte)
            if len(train) < 20 or len(np.unique(labels[target][train])) < 2:
                raise ValueError(f"insufficient label support for {target} fold {fold}")
            static = f["S"][:, cidx]
            if len(cidx) > 1:
                static = np.concatenate([static, static[:, 0:1].mean(axis=1, keepdims=True)], axis=1)
                # Replace the last column with the C02/C03 mean, without
                # allowing the model to select the better individual score.
                static[:, -1] = f["S"][:, cidx].mean(axis=1)
            xmap = {
                "N": n,
                "NS": np.concatenate([n, static], axis=1),
                "NR": np.concatenate([n, f["M"]], axis=1),
                "NCNA": np.concatenate([n, f["MC"][:, 8:]], axis=1),
                "NMC": np.concatenate([n, f["MC"]], axis=1),
                "NSMC": np.concatenate([n, static, f["MC"]], axis=1),
            }
            fitted = {}
            for method, x in xmap.items():
                model = fit_classifier(x[train], labels[target][train])
                max_iter = max(max_iter, int(model.n_iter_.max()))
                preds[target][method][test] = model.predict_proba(x[test])[:, 1]
                rows.append({"fold": fold, "target": target, "method": method, "train_n": len(train),
                             "test_n": len(test), "positive_train": int(labels[target][train].sum()),
                             "positive_test": int(labels[target][test].sum()), "iterations": int(model.n_iter_.max())})
                if method in ("NMC", "NSMC"):
                    fitted[method] = model
            block_keys = np.asarray([f"{d['cancer_type'][i]}|{platform[i]}" for i in test])
            block_groups = [np.flatnonzero(block_keys == k) for k in np.unique(block_keys)]
            local_groups = [test[g] for g in block_groups]
            rng = np.random.default_rng(20261012 + fold)
            for rep in range(99):
                pidx = derangement(local_groups, rng)
                for method in ("NMC", "NSMC"):
                    if method not in fitted:
                        continue
                    xwrong = xmap[method][test].copy()
                    # N and static columns remain paired to the patient; only
                    # the molecular block is moved to another patient.
                    mstart = n.shape[1] + (static.shape[1] if method == "NSMC" else 0)
                    xwrong[:, mstart:] = f["MC"][pidx[test]]
                    wrong[target][method][rep, test] = fitted[method].predict_proba(xwrong)[:, 1]
        print(f"L2_FOLD_DONE {fold}", flush=True)
    pd.DataFrame(rows).to_csv(out / "L2_fit_audit.csv", index=False)
    metric_rows = []
    rng = np.random.default_rng(20261013)
    for target in TARGET_CONCEPTS:
        ix = np.flatnonzero(valid[target]); y = labels[target][ix]
        strata = d["cancer_type"][ix]
        for method in methods:
            score = preds[target][method][ix]
            auc, lo, hi = bootstrap_auc(y, score, strata, rng, n=2000)
            bacc = balanced_accuracy_score(y, score >= .5)
            metric_rows.append({"target": target, "method": method, "n": len(ix), "positive": int(y.sum()),
                                "auc": auc, "bootstrap95_low": lo, "bootstrap95_high": hi,
                                "balanced_accuracy": bacc})
        for method in ("NR", "NCNA", "NMC", "NSMC"):
            a = preds[target][method][ix]; b = preds[target]["N"][ix]
            vals = []
            groups_y = [np.flatnonzero((strata == s) & (y == c)) for s in np.unique(strata) for c in (0, 1)]
            for _ in range(2000):
                draw = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups_y if len(g)])
                vals.append(roc_auc_score(y[draw], a[draw]) - roc_auc_score(y[draw], b[draw]))
            metric_rows.append({"target": target, "method": method + "-N_delta", "n": len(ix), "positive": int(y.sum()),
                                "auc": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
                                "bootstrap95_low": float(np.quantile(vals, .025)),
                                "bootstrap95_high": float(np.quantile(vals, .975)), "balanced_accuracy": np.nan})
        for method in ("NMC", "NSMC"):
            w = wrong[target][method][:, ix]
            aucs = [roc_auc_score(y, row) for row in w]
            pd.DataFrame({"target": target, "method": method, "replicate": np.arange(99), "wrong_patient_auc": aucs}).to_csv(
                out / f"L2_wrong_patient_{target}_{method}.csv", index=False)
            metric_rows.append({"target": target, "method": method + "_wrong_patient_mean", "n": len(ix), "positive": int(y.sum()),
                                "auc": float(np.mean(aucs)), "bootstrap95_low": float(np.quantile(aucs, .025)),
                                "bootstrap95_high": float(np.quantile(aucs, .975)), "balanced_accuracy": np.nan})
    pd.DataFrame(metric_rows).to_csv(out / "L2_oof_metrics.csv", index=False)
    np.savez_compressed(out / "private_L2_predictions.npz", **{
        f"{target}__{method}": preds[target][method] for target in TARGET_CONCEPTS for method in methods
    })
    write_json(out / "L2_decision.json", {"status": "PASS", "outer_folds": 5, "fixed_classifier_C": 1.0,
        "molecular_preprocessing_outer_train_only": True, "wrong_patient_replicates": 99,
        "maximum_classifier_iterations": max_iter,
        "interpretation": "OOF association with independent patient pathology fields; does not establish causality or patch-level correspondence"})


def load_parent_oof(root: Path, method: str, seed: int, d: dict):
    risk = np.full(len(d["patient_ids"]), np.nan); raw = risk.copy(); weight = risk.copy()
    for fold in range(1, 6):
        a = np.load(root / "oof" / f"DSS_seed{seed}_fold{fold}_{method}.npz")
        idx = a["test_indices"]
        risk[idx] = a["risk"]; raw[idx] = a["raw_risk"]; weight[idx] = a["uno_weight"]
    assert np.isfinite(risk).all() and np.isfinite(weight).all()
    return {"risk": risk, "raw_risk": raw, "uno_weight": weight}


def l3_survival(out: Path, d: dict, parent: Path, groups: dict):
    pp = read_json(parent / "protocol.json"); pm = read_json(parent / "manifest.json")
    methods = {"CUM": ["C", "U", "MC"], "CTM": ["C", "T", "MC"]}
    new = {}
    for seed in pp["seeds"]:
        for fold in range(1, 6):
            tag = f"DSS_seed{seed}_fold{fold}"
            pr = np.load(parent / "preprocessing" / f"{tag}_outer.npz")
            tr, te, it, iv = pr["tr"], pr["te"], pr["it"], pr["iv"]
            fi, _, _ = features(d, it, groups); fo, _, _ = features(d, tr, groups)
            for method, blocks in methods.items():
                name = f"{tag}_{method}"; op = out / "l3_oof" / f"{name}.npz"; cp = out / "l3_cells" / f"{name}.json"
                if op.exists() and cp.exists():
                    continue
                xi = np.concatenate([fi[k] for k in blocks], axis=1); xo = np.concatenate([fo[k] for k in blocks], axis=1)
                trials = []
                for lam in pp["ridge_grid"]:
                    w, detail = fit_cox(xi[it], d["DSS_time"][it], d["DSS_event"][it], lam)
                    trials.append({"ridge": lam, "inner_harrell": concordance(d["DSS_time"][iv], d["DSS_event"][iv], xi[iv] @ w), **detail})
                choice = max(trials, key=lambda z: (z["inner_harrell"], z["ridge"]))
                w, detail = fit_cox(xo[tr], d["DSS_time"][tr], d["DSS_event"][tr], choice["ridge"])
                train_r = xo[tr] @ w; test_r = xo[te] @ w
                mean = train_r.mean(); sd = train_r.std(); assert sd > 1e-10
                # The parent OOF already contains the locked IPCW weights; copy
                # them by fold so the only changed object is the representation.
                base = np.load(parent / "oof" / f"{tag}_CU.npz")
                np.savez_compressed(op, test_indices=te, patient_ids=d["patient_ids"][te], time=d["DSS_time"][te],
                    event=d["DSS_event"][te], risk=(test_r - mean) / sd, raw_risk=test_r,
                    uno_weight=base["uno_weight"], seed=seed, fold=fold, method=method,
                    protocol_sha256=pm["protocol_sha256"])
                write_json(cp, {"status": "PASS", "seed": int(seed), "fold": fold, "method": method,
                    "blocks": blocks, "feature_dim": int(xo.shape[1]), "trials": trials,
                    "selected_ridge": choice["ridge"], "outer_fit": detail,
                    "outer_test_used_for_selection": False, "same_parent_split_and_schedule": True})
        print(f"L3_SEED_DONE {seed}", flush=True)
    # Aggregate by seed first; no 15-cell pseudo-seed averaging.
    rows = []
    for seed in pp["seeds"]:
        for method in ("CU", "CT", "CUM", "CTM"):
            root = parent if method in ("CU", "CT") else out
            risk = np.full(len(d["patient_ids"]), np.nan); raw = risk.copy(); weight = risk.copy()
            for fold in range(1, 6):
                if root == parent:
                    a = np.load(root / "oof" / f"DSS_seed{seed}_fold{fold}_{method}.npz")
                else:
                    a = np.load(root / "l3_oof" / f"DSS_seed{seed}_fold{fold}_{method}.npz")
                idx = a["test_indices"]; risk[idx] = a["risk"]; raw[idx] = a["raw_risk"]; weight[idx] = a["uno_weight"]
            rows.append({"seed": seed, "method": method, "oof_harrell": concordance(d["DSS_time"], d["DSS_event"], risk),
                         "oof_uno60": concordance(d["DSS_time"], d["DSS_event"], risk, weight),
                         "raw_oof_harrell": concordance(d["DSS_time"], d["DSS_event"], raw)})
    table = pd.DataFrame(rows); table.to_csv(out / "L3_seed_oof_metrics.csv", index=False)
    # Paired patient bootstrap on seed-wise metric differences.
    rng = np.random.default_rng(20261014); boot = np.zeros((5000, len(d["patient_ids"])))
    for ev in [0, 1]:
        g = np.flatnonzero(d["DSS_event"] == ev); boot[:, g] = rng.multinomial(len(g), np.ones(len(g)) / len(g), size=5000)
    pair_rows = []
    for a, b in [("CUM", "CU"), ("CTM", "CT")]:
        ds_h = []; ds_u = []; dist_h = []; dist_u = []
        for seed in pp["seeds"]:
            base = load_parent_oof(parent, b, seed, d) if b in ("CU", "CT") else None
            if a in ("CUM", "CTM"):
                rr = np.full(len(d["patient_ids"]), np.nan); ww = rr.copy()
                for fold in range(1, 6):
                    z = np.load(out / "l3_oof" / f"DSS_seed{seed}_fold{fold}_{a}.npz"); idx = z["test_indices"]; rr[idx] = z["risk"]; ww[idx] = z["uno_weight"]
            ds_h.append(concordance(d["DSS_time"], d["DSS_event"], rr) - concordance(d["DSS_time"], d["DSS_event"], base["risk"]))
            ds_u.append(concordance(d["DSS_time"], d["DSS_event"], rr, ww) - concordance(d["DSS_time"], d["DSS_event"], base["risk"], base["uno_weight"]))
            def dist(metric):
                # Vectorized patient-bootstrap concordance.  Rebuilding a
                # 512x512 pair matrix inside 5,000 loops would be needlessly
                # expensive and can make this audit look GPU-bound.
                time = d["DSS_time"]; event = d["DSS_event"]
                eligible = (event[:, None] == 1) & ((time[:, None] < time[None, :]) |
                    ((time[:, None] == time[None, :]) & (event[None, :] == 0)))
                if metric == "h":
                    p1 = eligible * ((rr[:, None] > rr[None, :]).astype(float) + .5 * (rr[:, None] == rr[None, :]))
                    p0 = eligible * ((base["risk"][:, None] > base["risk"][None, :]).astype(float) + .5 * (base["risk"][:, None] == base["risk"][None, :]))
                else:
                    p1 = eligible * ww[:, None] * ((rr[:, None] > rr[None, :]).astype(float) + .5 * (rr[:, None] == rr[None, :]))
                    p0 = eligible * base["uno_weight"][:, None] * ((base["risk"][:, None] > base["risk"][None, :]).astype(float) + .5 * (base["risk"][:, None] == base["risk"][None, :]))
                den = eligible if metric == "h" else eligible * ww[:, None]
                den0 = eligible if metric == "h" else eligible * base["uno_weight"][:, None]
                n1 = np.einsum("bi,ij,bj->b", boot, p1, boot); d1 = np.einsum("bi,ij,bj->b", boot, den, boot)
                n0 = np.einsum("bi,ij,bj->b", boot, p0, boot); d0 = np.einsum("bi,ij,bj->b", boot, den0, boot)
                return n1 / d1 - n0 / d0
            dist_h.append(dist("h")); dist_u.append(dist("u"))
        dh = np.mean(np.stack(dist_h), axis=0); du = np.mean(np.stack(dist_u), axis=0)
        pair_rows.append({"comparison": f"{a}-{b}", "delta_harrell": float(np.mean(ds_h)),
                          "harrell_seed_deltas": json.dumps(ds_h), "harrell_ci95_low": float(np.quantile(dh, .025)),
                          "harrell_ci95_high": float(np.quantile(dh, .975)), "harrell_family99_low": float(np.quantile(dh, .005)),
                          "harrell_family99_high": float(np.quantile(dh, .995)), "delta_uno60": float(np.mean(ds_u)),
                          "uno_seed_deltas": json.dumps(ds_u), "uno_ci95_low": float(np.quantile(du, .025)),
                          "uno_ci95_high": float(np.quantile(du, .975)),
                          "stable_positive": bool(min(ds_h) > 0 and float(np.quantile(dh, .005)) > 0)})
    pd.DataFrame(pair_rows).to_csv(out / "L3_paired_increments.csv", index=False)
    write_json(out / "L3_decision.json", {"status": "PASS", "new_methods": ["CUM", "CTM"],
        "reused_anchors": ["CU", "CT"], "new_outer_fits": 30, "new_inner_fits": 120,
        "interpretation": "conditional DSS headroom with original pathology anchors; this is not a causal information estimate"})


def report(out: Path, availability: dict):
    l1 = pd.read_csv(out / "L1_titan_independent_morphology.csv")
    l2 = pd.read_csv(out / "L2_oof_metrics.csv")
    l3 = pd.read_csv(out / "L3_seed_oof_metrics.csv")
    pairs = pd.read_csv(out / "L3_paired_increments.csv")
    lines = ["# Phase7C：独立形态标签、分子对应关系与 DSS 条件增量审计", "",
             "本轮只使用实际存在的患者级病理字段；TITAN concept score 始终是预测变量，不是标签。RNA/CNA 状态和所有 scaler 均在外层训练患者上拟合。", "",
             "## 1. 标签可用性", "", "|项目|来源|有效人数|阳性人数|未知人数|", "|---|---|---:|---:|---:|"]
    for k, v in availability.items():
        lines.append(f"|{k}|{v['source_field']}|{v['valid']}|{v['positive'] if v['positive'] is not None else '—'}|{v['unknown']}|")
    lines += ["", "GRADE、tumor budding 和 stromal indicator 没有可用字段，因此未进入模型，也没有用 TITAN 分数补标签。", "",
              "## 2. 病理语义可靠性（冻结 TITAN）", "", "|目标|concept|n|阳性|AUC|95% bootstrap CI|", "|---|---|---:|---:|---:|---|"]
    for _, r in l1.iterrows():
        lines.append(f"|{r.target}|{r.concept_score}|{int(r.n)}|{int(r.positive)}|{r.auc:.4f}|[{r.bootstrap95_low:.4f},{r.bootstrap95_high:.4f}]|")
    lines += ["", "## 3. 分子状态对真实形态标签的患者级 OOF 检验", "", "|目标|模型|n|阳性|AUC|95% CI|BA|", "|---|---|---:|---:|---:|---|---:|"]
    for _, r in l2[l2.method.str.contains("_delta|wrong_patient", regex=True) == False].iterrows():
        lines.append(f"|{r.target}|{r.method}|{int(r.n)}|{int(r.positive)}|{r.auc:.4f}|[{r.bootstrap95_low:.4f},{r.bootstrap95_high:.4f}]|{r.balanced_accuracy:.4f}|")
    lines += ["", "### 相对临床基线 N 的分子增量与错误患者对照", "", "|目标|比较|AUC/均值|95% CI 或 wrong-patient 分位区间|", "|---|---|---:|---|"]
    for _, r in l2[l2.method.str.contains("_delta|wrong_patient", regex=True)].iterrows():
        label = "wrong-patient" if "wrong_patient" in r.method else "delta vs N"
        lines.append(f"|{r.target}|{r.method} ({label})|{r.auc:+.4f}|[{r.bootstrap95_low:+.4f},{r.bootstrap95_high:+.4f}]|")
    lines += ["", "正确配对只有在分子特征仍有增量、而 wrong-patient 对照回落时，才支持患者级对应关系；若临床基线已很强，wrong-patient 可能仍保持较高 AUC，这不能被误读为分子匹配有效。", "",
              "## 4. Clinical + 原始病理证据下的 DSS", "", "|比较|三 seed Harrell 增量|99% CI|三 seed Uno 增量|稳定正向|", "|---|---:|---|---:|---:|"]
    for _, r in pairs.iterrows():
        lines.append(f"|{r.comparison}|{r.delta_harrell:+.4f}|[{r.harrell_family99_low:+.4f},{r.harrell_family99_high:+.4f}]|{r.delta_uno60:+.4f}|{bool(r.stable_positive)}|")
    lines += ["", "## 决策边界", "", "- 独立形态字段只支持对应概念的语义审计；不能推断其他概念或 patch 级真值。", "- 分子对形态标签的 OOF 关联不等于因果，也不能把 TNM 代理结果替代真实形态标注。", "- DSS 结果只判断当前冻结表示和低容量 Cox 头的条件增量；若区间跨零，不把点估计写成已验证机制。", "", "## 可复核文件", "", "- `L1_titan_independent_morphology.csv`", "- `L2_oof_metrics.csv`、`L2_wrong_patient_*.csv`", "- `L3_seed_oof_metrics.csv`、`L3_paired_increments.csv`", "- `manifest.json`、`protocol.json`、`DONE.json`"]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(out: Path):
    if (out / "manifest.json").exists():
        raise RuntimeError("refuse to overwrite an existing Phase7C run")
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
    sources = [CONFIG, Path(__file__), CONCEPTS, BASE / "data_clinical_patient.txt", BASE / "data_clinical_sample.txt", sup_path,
               parent / "private_dataset.npz", parent / "protocol.json", parent / "manifest.json"]
    manifest = {"protocol_sha256": digest(out / "protocol.json"), "locked_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "sources": [{"path": str(x), "sha256": digest(x)} for x in sources], "private_dataset_sha256": digest(parent / "private_dataset.npz"),
                "patients": int(len(ids)), "availability": availability, "DSS_labels_used_for_label_selection": False,
                "concept_score_used_as_truth": False, "outer_folds": 5, "molecular_preprocessing": "outer-train only"}
    write_json(out / "manifest.json", manifest); write_json(out / "label_availability.json", availability)
    for sub in ("l3_oof", "l3_cells"): (out / sub).mkdir(exist_ok=True)
    print("PHASE7C_LOCKED", flush=True)
    return d, labels, valid, availability, primary, sup, sample


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["all", "prepare"], default="all"); ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(); p = read_json(CONFIG); parent = Path(p["parent_run"])
    if args.mode == "prepare":
        prepare(args.out); return
    d, labels, valid, availability, primary, sup, sample = prepare(args.out)
    platform = platform_names(d["patient_ids"], Path(p["quality_file"]))
    groups = read_json(parent / "manifest.json")["groups"]
    label_audit(args.out, d, labels, valid, availability, primary, sup)
    l2_audit(args.out, d, labels, valid, platform, groups)
    l3_survival(args.out, d, parent, groups)
    report(args.out, availability)
    write_json(args.out / "DONE.json", {"status": "PASS", "protocol_sha256": digest(args.out / "protocol.json"), "patients": 512,
                                        "independent_label_targets": list(TARGET_CONCEPTS), "DSS_primary": True})
    print("PHASE7C_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

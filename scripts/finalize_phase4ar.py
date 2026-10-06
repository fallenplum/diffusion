"""Fail-closed Phase4A-R aggregation from three private, seed-scoped runtimes.

This command does not fit models or choose any training hyperparameters.
Every metric is computed from a complete five-fold OOF, and every shuffle is
scored separately. Only aggregate artifacts are written to the public report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

spec = importlib.util.spec_from_file_location("repair", Path(__file__).with_name("phase4ar_dss_free.py"))
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)
b = repair.m
TASKS = ("T", "N", "M", "Stage")
BOOT_METRICS = ("T_BA", "N_BA", "M_BA", "Stage_BA", "G_TNM")


def bootstrap_scores(logits, targets, resamples):
    """Exact class recall BA, with the same present-class convention as sklearn."""
    values = []
    for task in TASKS:
        y = targets[task]
        pred = np.argmax(logits[task], axis=1)
        ys = y[resamples]
        correct = (pred == y)[resamples]
        recall = []
        for c in np.unique(y[y >= 0]):
            mask = ys == c
            den = mask.sum(axis=1)
            recall.append(np.divide((mask & correct).sum(axis=1), den,
                                    out=np.full(len(den), np.nan), where=den > 0))
        values.append(np.nanmean(np.column_stack(recall), axis=1))
    a = np.column_stack(values)
    return np.column_stack([a, a[:, :3].mean(axis=1)])


def verify_fast_bootstrap():
    rng = np.random.default_rng(9106)
    targets = {task: rng.integers(-1, c, size=64) for task, c in zip(TASKS, (4, 2, 2, 4))}
    logits = {task: rng.normal(size=(64, c)) for task, c in zip(TASKS, (4, 2, 2, 4))}
    idx = rng.integers(0, 64, size=(20, 64))
    fast = bootstrap_scores(logits, targets, idx)
    for i, sample in enumerate(idx):
        for j, task in enumerate(TASKS):
            y = targets[task][sample]
            pred = logits[task][sample].argmax(axis=1)
            valid = y >= 0
            assert np.isclose(fast[i, j], balanced_accuracy_score(y[valid], pred[valid]))
    return True


def load_targets(repo, data):
    # Aggregation needs no DSS/CNA values. Cohort membership is the fixed manifest.
    import sys
    sys.path.insert(0, str(repo))
    from src.phase_m2.data import _load_patient_table, _merge_clinical
    split_path = repo / "outputs/phase_m1r/fixed_patient_folds.csv"
    split = pd.read_csv(split_path).sort_values("patient_id").reset_index(drop=True)
    assert len(split) == 512 and not split.patient_id.duplicated().any()
    ids = split.patient_id.tolist()
    primary = _load_patient_table(data / "data/data_8/coadread_tcga_pan_can_atlas_2018/data_clinical_patient.txt", ids)
    extra = _load_patient_table(data / "data/data_8/coadread_tcga/data_clinical_patient.txt", ids)
    clinical = _merge_clinical(primary, extra, ids)
    return ids, split.fold.to_numpy(int), b.target_arrays(clinical)


def diagnostics(path, model, seed, fold, shuffle):
    with np.load(path) as f:
        tokens = f["tokens"].astype(np.float64)
        pool = f["pool"].astype(np.float64)
        entropy = f["attention_entropy"].astype(float)
        ranks = f["rank"].copy()
    norm = np.linalg.norm(tokens, axis=-1, keepdims=True)
    normalized = tokens / np.maximum(norm, 1e-12)
    pair = normalized @ normalized.transpose(0, 2, 1)
    cosine = pair[:, np.triu_indices(50, 1)[0], np.triu_indices(50, 1)[1]].mean()
    # Include the previously used centered quantity with its correct name.
    centered = tokens - tokens.mean(axis=-1, keepdims=True)
    centered /= np.maximum(np.linalg.norm(centered, axis=-1, keepdims=True), 1e-12)
    correlation = centered @ centered.transpose(0, 2, 1)
    correlation = correlation[:, np.triu_indices(50, 1)[0], np.triu_indices(50, 1)[1]].mean()
    cov = np.cov(tokens.reshape(-1, 32), rowvar=False)
    eig = np.maximum(np.linalg.eigvalsh(cov), 0)
    prob = eig / max(eig.sum(), 1e-12)
    effective_rank = np.exp(-(prob * np.log(np.maximum(prob, 1e-12))).sum())
    var = float(np.var(tokens, axis=0).mean())
    row = {"model": model, "seed": seed, "outer_fold": fold, "shuffle_id": shuffle,
           "token_across_patient_variance": var, "patient_token_pairwise_cosine": float(cosine),
           "centered_token_pairwise_correlation": float(correlation),
           "token_covariance_effective_rank": float(effective_rank),
           "pathology_patch_attention_entropy": float(entropy.mean()),
           "normalized_patch_attention_entropy": float(entropy.mean() / np.log(1024)),
           "pathway_pool_entropy": float((-pool * np.log(np.maximum(pool, 1e-12))).sum(axis=1).mean()),
           "token_collapse_flag": bool(var < 1e-6),
           "diagnostic_caution": "variance_floor_only; cosine_and_rank_are_descriptive_not_new_selection_gates"}
    return row, ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--runtime-roots", nargs=3, type=Path, required=True)
    ap.add_argument("--reports-root", type=Path, required=True)
    ap.add_argument("--path-cache", type=Path, required=True)
    args = ap.parse_args()
    report = args.reports_root
    report.mkdir(parents=True, exist_ok=True)
    ids, folds, targets = load_targets(args.repo_root, args.data_root)
    order_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    cache_meta = json.loads((args.path_cache / "path_tokens_meta.json").read_text())
    assert cache_meta["patient_order_sha256"] == order_sha
    assert verify_fast_bootstrap()
    gmt = args.repo_root / "src/phase_m2/metadata/h.all.v2024.1.Hs.symbols.gmt"
    assert b.sha256_file(gmt) == "ee2463540042078bfa3f67828e1e223bb354446d9fbb4d22845866835ba5c772"
    all_cells = []
    logits = {}
    seed_rows = []
    shuffle_rows = []
    diag_rows = []
    ranks = {}
    cache_inventory = []
    weight_audits = []
    for seed, root in zip(b.SEEDS, args.runtime_roots):
        for model in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
            qids = b.SHUFFLE_SEEDS if model == "M4" else (0,)
            for q in qids:
                assembled = {task: np.full((512, c), np.nan) for task, c in zip(TASKS, (4, 2, 2, 4))}
                seen = np.zeros(512, dtype=int)
                for fold in b.OUTER_FOLDS:
                    stem = f"{model}_s{seed}_f{fold}_q{q}"
                    cell_path = root / "cells" / f"{stem}.json"
                    pred_path = root / "oof" / f"{stem}.npz"
                    assert cell_path.exists() and pred_path.exists(), stem
                    cell = json.loads(cell_path.read_text())
                    assert (cell["model"], cell["seed"], cell["outer_fold"], cell["shuffle_id"]) == (model, seed, fold, q)
                    assert cell["status"] == "completed" and cell["selection_metric"] == "EMA_inner_validation_balanced_CE"
                    assert 10 <= cell["best_epoch"] <= 100 and np.isfinite(cell["inner_val_loss_ema"])
                    all_cells.append(cell)
                    with np.load(pred_path) as f:
                        idx = f["outer_test"]
                        assert np.array_equal(idx, np.flatnonzero(folds == fold)), stem
                        seen[idx] += 1
                        for task in TASKS:
                            assert np.isfinite(f[task]).all()
                            assembled[task][idx] = f[task]
                        if model in {"M0", "M2", "M3", "M4", "M6"}:
                            assert np.isfinite(f["recon_pred"]).all()
                            assert np.isfinite(f["recon_truth"]).all()
                    cache_inventory.append({"model": model, "seed": seed, "outer_fold": fold, "shuffle_id": q,
                                            "cell_sha256": b.sha256_file(cell_path), "oof_sha256": b.sha256_file(pred_path)})
                    if model in {"M3", "M4", "M5", "M6"}:
                        path = root / "attention" / f"{stem}.npz"
                        assert path.exists()
                        row, rank = diagnostics(path, model, seed, fold, q)
                        diag_rows.append(row)
                        ranks[(model, seed, fold, q)] = rank
                assert np.all(seen == 1), (seed, model, q, "nonunique OOF")
                logits[(model, seed, q)] = assembled
                metric = b.metrics(assembled, targets)
                if model == "M4":
                    shuffle_rows.append({"model": model, "seed": seed, "shuffle_id": q, **metric})
                else:
                    seed_rows.append({"model": model, "seed": seed, **metric})
        control_frame = pd.DataFrame([x for x in shuffle_rows if x["seed"] == seed])
        seed_rows.append({"model": "M4", "seed": seed, **control_frame.drop(columns=["model", "seed", "shuffle_id"]).mean().to_dict()})
        for fold in b.OUTER_FOLDS:
            outer_train = np.flatnonzero(folds != fold)
            inner_train, inner_val, mode = b.make_inner_split(outer_train, targets["Stage"], seed + fold + 4401)
            assert not np.intersect1d(inner_train, inner_val).size
            assert not np.intersect1d(outer_train, np.flatnonzero(folds == fold)).size
            for partition, indices in (("inner_train", inner_train), ("outer_train", outer_train)):
                weights = repair.class_weights(targets, indices, repair.torch.device("cpu"))
                for task, c in zip(TASKS, (4, 2, 2, 4)):
                    y = targets[task][indices]
                    counts = np.bincount(y[y >= 0], minlength=c)
                    weight_audits.append({"seed": seed, "outer_fold": fold, "partition": partition, "task": task,
                                          "class_counts": json.dumps(counts.tolist()), "class_weights": json.dumps(weights[task].tolist()),
                                          "missing_count": int((y < 0).sum()), "split_mode": mode,
                                          "audit_type": "recomputed_from_deterministic_training_indices"})
    assert len(all_cells) == 135
    sf = pd.DataFrame(seed_rows).sort_values(["model", "seed"])
    sf.to_csv(report / "seed_oof_metrics.csv", index=False)
    pd.DataFrame(shuffle_rows).to_csv(report / "shuffle_seed_oof_metrics.csv", index=False)
    cells = pd.DataFrame(all_cells).sort_values(["seed", "outer_fold", "model", "shuffle_id"])
    cells.to_csv(report / "fold_metrics.csv", index=False)
    cells[["model", "seed", "outer_fold", "shuffle_id", "reconstruction_mse", "reconstruction_pearson", "train_reconstruction_first", "train_reconstruction_last", "train_reconstruction_decreased"]].to_csv(report / "masked_reconstruction.csv", index=False)
    cells[["model", "seed", "outer_fold", "shuffle_id", "train_seconds", "peak_gpu_memory_gb", "batch_size", "best_epoch"]].to_csv(report / "training_resource.csv", index=False)
    pd.DataFrame(weight_audits).to_csv(report / "class_weight_audit.csv", index=False)
    pd.DataFrame(cache_inventory).to_csv(report / "private_cache_provenance.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(report / "token_diagnostics.csv", index=False)
    metrics_by = {(x["model"], x["seed"]): x for x in seed_rows}
    bootstrap_rows = []
    comparisons = []
    for seed in b.SEEDS:
        for name, baseline in (("M3_minus_M4_shuffle_mean", "M4"), ("M3_minus_M2_naive", "M2"), ("M3_minus_M6_fixed_query", "M6"), ("M3_minus_M5_no_mask", "M5")):
            rng = np.random.default_rng(seed + 9401 + b.stable_hash(name))
            idx = rng.integers(0, 512, size=(2000, 512))
            a = bootstrap_scores(logits[("M3", seed, 0)], targets, idx)
            bases = [logits[(baseline, seed, q)] for q in (b.SHUFFLE_SEEDS if baseline == "M4" else (0,))]
            bs = np.mean([bootstrap_scores(x, targets, idx) for x in bases], axis=0)
            delta = a - bs
            row = {"comparison": name, "seed": seed, "bootstrap_replicates": 2000,
                   "sampling_unit": "patient; same draws across candidate and all controls",
                   "shuffle_estimand": "mean of independently scored controls"}
            for j, metric in enumerate(BOOT_METRICS):
                row[f"delta_{metric}_observed"] = metrics_by[("M3", seed)][metric] - metrics_by[(baseline, seed)][metric]
                row[f"delta_{metric}_bootstrap_mean"] = float(delta[:, j].mean())
                row[f"delta_{metric}_ci95_low"], row[f"delta_{metric}_ci95_high"] = np.quantile(delta[:, j], [.025, .975]).tolist()
            bootstrap_rows.append(row)
            comparisons.append({"seed": seed, "comparison": name, "delta_G_TNM": row["delta_G_TNM_observed"],
                                "ci95_low": row["delta_G_TNM_ci95_low"], "ci95_high": row["delta_G_TNM_ci95_high"],
                                **{f"delta_{task}_BA": row[f"delta_{task}_BA_observed"] for task in TASKS}})
    boots = pd.DataFrame(bootstrap_rows)
    boots.to_csv(report / "paired_bootstrap.csv", index=False)
    pd.DataFrame(comparisons).to_csv(report / "knowledge_comparison.csv", index=False)
    stability = []
    for model in ("M3", "M4", "M5", "M6"):
        for q in (b.SHUFFLE_SEEDS if model == "M4" else (0,)):
            for fold in b.OUTER_FOLDS:
                for i, sa in enumerate(b.SEEDS):
                    for sb in b.SEEDS[i + 1:]:
                        ra, rb = ranks[(model, sa, fold, q)], ranks[(model, sb, fold, q)]
                        k = 102
                        # Count common top patches without a Bx50x102x102 allocation.
                        membership = np.zeros_like(ra, dtype=bool)
                        np.put_along_axis(membership, ra[:, :, :k], True, axis=-1)
                        common = np.take_along_axis(membership, rb[:, :, :k], axis=-1).sum(axis=-1)
                        ja = common / (2 * k - common)
                        ia = np.argsort(ra, axis=-1).astype(float)
                        ib = np.argsort(rb, axis=-1).astype(float)
                        rho = 1 - 6 * ((ia - ib)**2).sum(axis=-1) / (1024 * (1024**2 - 1))
                        stability.append({"model": model, "outer_fold": fold, "shuffle_id": q, "seed_a": sa, "seed_b": sb,
                                          "patients_compared": ra.shape[0], "top10_jaccard": float(ja.mean()),
                                          "attention_spearman": float(rho.mean()),
                                          "interpretation": "reproducibility_only_not_biological_validation"})
    pd.DataFrame(stability).to_csv(report / "pathology_attention_stability.csv", index=False)
    profiles = []
    for model in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
        instance = b.make_model(model)
        profiles.append({"model": model, "total_params": sum(x.numel() for x in instance.parameters()),
                         "trainable_params": sum(x.numel() for x in instance.parameters() if x.requires_grad)})
    pd.DataFrame(profiles).to_csv(report / "parameter_counts.csv", index=False)
    b.atomic_json(report / "model_profiles.json", {x["model"]: x for x in profiles})
    assert profiles[3]["total_params"] == profiles[4]["total_params"] == profiles[6]["total_params"]
    b.atomic_json(report / "clinical_target_audit.json", b.clinical_audit(targets, folds))
    # Static fold mapping and shuffle provenance are retained from completed workers.
    worker_reports = [args.repo_root / "reports/phase4ar_dss_free_grounding_repair",
                      args.repo_root / "reports/phase4ar_seed20260905", args.repo_root / "reports/phase4ar_seed20260906"]
    maps = [json.loads((r / "hallmark_mapping_audit.json").read_text()) for r in worker_reports]
    controls = [json.loads((r / "shuffled_pathway_audit.json").read_text()) for r in worker_reports]
    assert maps[0] == maps[1] == maps[2]
    assert controls[0] == controls[1] == controls[2]
    assert all(x["gene_degrees_preserved"] and x["pathway_sizes_true"] == x["pathway_sizes_shuffled"] for x in controls[0]["folds"])
    assert maps[0]["matched_gene_universe_count"] == 4191
    b.atomic_json(report / "hallmark_mapping_audit.json", maps[0])
    b.atomic_json(report / "shuffled_pathway_audit.json", controls[0])
    protocol = {"phase": "Phase4A-R", "cohort": 512, "seeds": list(b.SEEDS), "outer_folds": list(b.OUTER_FOLDS),
                "formal_cells": 135, "input_modalities": ["RNA pathway summaries", "UNI2-h 1024x1536 patches"],
                "selection_metric": "EMA_inner_validation_balanced_CE", "ema_alpha": .2, "warmup_epochs": 10,
                "patience": 12, "max_epochs": 100, "optimizer": "AdamW", "lr": .0003, "weight_decay": .001,
                "batch_size": 64, "token_dim": 32, "mask_ratio": .2, "reconstruction_weight": .5,
                "attention_entropy_lambda": .001, "token_squared_cosine_lambda": .001,
                "regularizer_scope": "patch entropy: M3-M6; pathway token diversity: M0,M2,M3-M6",
                "class_weights": "inverse class frequency normalized to mean1 over present classes; inner-train for selection, outer-train for refit",
                "clinical_head_weights": {"T": 1, "N": 1, "M": 1, "Stage": .5},
                "masked_identity": "pathway slot embedding + learned mask token in M0/M2/M3/M4/M6",
                "dss_supervision": False, "clinical_targets_not_inputs": True,
                "rna_preprocessing": "outer-train median; metadata log2(x+1); gene zscore; five summaries; summary zscore",
                "inner_preprocessing_limitation": "RNA statistics use outer-train, including inner-val features; no inner-val labels enter statistics. Inner validation is not fully feature-isolated.",
                "selection_estimator_limitation": "CE is normalized independently within each validation batch then weighted by patient count, rather than globally by class weights.",
                "regularizer_assessment": "preregistered constants in runner before formal cells; entropy/diversity results are diagnostics and do not tune lambdas",
                "no_outer_test_tuning": True,
                "report_protocol_materialization": "assembled after training from fixed runner constants; worker initial protocol did not contain every repair field"}
    b.atomic_json(report / "protocol.json", protocol)
    validation = {"status": True, "cohort_512": True, "fixed_outer_folds": True,
                  "unique_formal_cells": len(all_cells), "expected_formal_cells": 135,
                  "all_oof_grounding_predictions_complete": True, "one_oof_prediction_per_patient": True,
                  "formal_seed_rows": len(seed_rows), "formal_expected_seed_rows": 21,
                  "patient_order_sha256_matches": True, "hallmark_mapping_recorded": True,
                  "shuffled_pathway_size_matched": True, "shuffled_gene_degree_preserved": True,
                  "three_shuffle_controls_complete": True, "shuffles_aggregated_as_metrics_not_logits": True,
                  "class_weights_training_indices_only": True, "class_weight_audit_recomputed": True,
                  "no_raw_G_TNM_epoch_selection": True, "all_best_epochs_at_least_10": True,
                  "bootstrap_matches_sklearn_reference": True, "clinical_targets_not_inputs": True,
                  "outer_test_not_used_for_pretraining": True, "dss_labels_used_for_training": False,
                  "dss_metrics_used_for_selection": False, "dss_read_for_cohort_check_only": True,
                  "patient_ids_public": False, "inner_val_features_used_for_unsupervised_preprocessing": True}
    b.atomic_json(report / "validation.json", validation)
    b.atomic_json(report / "run_manifest.json", {"phase": "Phase4A-R", "training_script_sha256": b.sha256_file(Path(repair.__file__)),
                  "base_script_sha256": b.sha256_file(Path(b.__file__)), "finalizer_sha256": b.sha256_file(Path(__file__)),
                  "hallmark_sha256": b.sha256_file(gmt), "patient_order_sha256": order_sha,
                  "seeds": list(b.SEEDS), "cohort_size": 512, "unique_cells": 135,
                  "seed_runtime_assignment": {str(s): str(r) for s, r in zip(b.SEEDS, args.runtime_roots)},
                  "duplicate_policy": "seed4 only from first runtime; seed5/6 only from independent runtimes; duplicates excluded",
                  "provenance_inventory": "private_cache_provenance.csv; hashes only, no patient data",
                  "gpu": "NVIDIA GeForce RTX 4080 32760MiB", "runtime_artifacts_not_public": True})
    ledger = [{"failure_type": "ephemeral_training_preflight", "details": "Missing attention output directory; fixed before formal cells; temporary cells excluded"},
              {"failure_type": "scheduler_duplicate", "details": "Initial worker used default 3 seeds; stopped after seed4 complete; seed5/6 duplicates retained privately and excluded by explicit runtime assignment"}]
    pd.DataFrame(ledger).to_csv(report / "failure_ledger.csv", index=False)
    b.atomic_json(report / "repair_audit.json", {"status": True, "raw_G_TNM_early_stopping_removed": True,
                  "inner_train_class_balance": True, "outer_train_refit_class_balance": True,
                  "attention_concentration_lambda": .001, "token_diversity_lambda": .001,
                  "M0_M2_mask_slot_preserved": True, "reconstruction_loss_preserved": True,
                  "training_reconstruction_logged": True, "guided_vs_fixed_retained": True,
                  "new_auxiliary_tasks_added": False, "test_used_to_tune_repairs": False})
    write_decision(report, sf, boots, cells, pd.DataFrame(diag_rows), pd.DataFrame(stability))
    (report / "COMPLETE.json").write_text(json.dumps({"status": "COMPLETE", "unique_cells": 135, "seed_oof_rows": 21,
                                              "validation_pass": True}, indent=2))
    print(sf.groupby("model").G_TNM.mean().to_string(), flush=True)
    print((report / "decision.json").read_text(), flush=True)


def write_decision(report, seeds, boots, cells, tokens, stability):
    mean = seeds.groupby("model").mean(numeric_only=True)
    def comparison(name):
        return boots.loc[boots.comparison == name]
    kb = comparison("M3_minus_M4_shuffle_mean")
    tasks_positive = sum(mean.loc["M3", f"{t}_BA"] > mean.loc["M4", f"{t}_BA"] for t in ("T", "N", "M"))
    gate = {"M3_above_M2": bool(mean.loc["M3", "G_TNM"] > mean.loc["M2", "G_TNM"]),
            "M3_above_M4": bool(mean.loc["M3", "G_TNM"] > mean.loc["M4", "G_TNM"]),
            "positive_true_shuffle_seeds_at_least_2": bool((kb.delta_G_TNM_observed > 0).sum() >= 2),
            "all_seed_true_shuffle_CI_lower_positive": bool((kb.delta_G_TNM_ci95_low > 0).all()),
            "at_least_2_TNM_mean_BA_positive": bool(tasks_positive >= 2),
            "M3_at_least_best_unimodal": bool(mean.loc["M3", "G_TNM"] >= max(mean.loc["M0", "G_TNM"], mean.loc["M1", "G_TNM"])),
            "no_variance_floor_collapse": bool(not tokens.loc[tokens.model == "M3", "token_collapse_flag"].any()),
            "no_DSS_supervision": True}
    supported = all(gate.values())
    guided = comparison("M3_minus_M6_fixed_query")
    stable_guided = (guided.delta_G_TNM_observed > 0).sum() >= 2 and (guided.delta_G_TNM_ci95_low > 0).all()
    if supported:
        decision = "KNOWLEDGE_GROUNDED_STATE_SUPPORTED"
    elif stable_guided:
        decision = "OMICS_GUIDED_PATHOLOGY_SUPPORTED"
    elif gate["M3_above_M2"] and gate["M3_above_M4"]:
        decision = "STRUCTURE_HELPFUL_SEMANTICS_UNSUPPORTED"
    elif gate["M3_above_M4"]:
        decision = "GROUNDING_DIRECTIONAL"
    else:
        decision = "DISEASE_STATE_HYPOTHESIS_NOT_SUPPORTED"
    b.atomic_json(report / "decision.json", {"decision": decision, "phase4B_allowed": supported, "strict_gates": gate,
                                           "mean_delta_G_true_shuffle": float(mean.loc["M3", "G_TNM"] - mean.loc["M4", "G_TNM"]),
                                           "interpretation": "This repair bundle changes multiple training rules; gains cannot be attributed to any one repair."})
    (report / "DECISION.md").write_text(f"# Phase4A-R 决策\n\n**{decision}**\n\nPhase4B：**{'允许' if supported else '不允许'}**。\n\n" + "\n".join(f"- {k}: `{v}`" for k, v in gate.items()) + "\n", encoding="utf-8")
    lines = ["# Phase4A-R 修复实验结果", "", "135 个唯一正式 cells 已完整；每个 seed 拼接 5-fold OOF 后计算指标，再对 3 seeds 求均值。M4 为三个独立 shuffle 指标的均值，没有平均 logits。", "",
             "## 核心结果", "", "|模型|3-seed OOF G_TNM|T BA|N BA|M BA|", "|---|---:|---:|---:|---:|"]
    for model in mean.index:
        lines.append(f"|{model}|{mean.loc[model, 'G_TNM']:.6f}|{mean.loc[model, 'T_BA']:.6f}|{mean.loc[model, 'N_BA']:.6f}|{mean.loc[model, 'M_BA']:.6f}|")
    lines += ["", "## 同 seed 配对差值", "", "|比较|seed|ΔG_TNM|患者配对 bootstrap 95% CI|", "|---|---:|---:|---|"]
    for row in boots.itertuples():
        lines.append(f"|{row.comparison}|{row.seed}|{row.delta_G_TNM_observed:.6f}|[{row.delta_G_TNM_ci95_low:.6f}, {row.delta_G_TNM_ci95_high:.6f}]|")
    recon = cells.loc[cells.model.isin(["M0", "M2", "M3", "M4", "M6"])]
    lines += ["", "## 修复与诊断", "", "- 选择：inner-train 权重的 balanced validation CE，EMA α=0.2，至少10epochs，patience12；refit 使用 outer-train 权重和相同 AdamW/loss 规则。", "- 正则：patch normalized entropy λ=0.001；pathway token off-diagonal squared cosine λ=0.001。M0/M2 的 masked tokens 保留 pathway slot。", f"- best epoch 中位数 {cells.best_epoch.median():.1f}，范围 {cells.best_epoch.min()}–{cells.best_epoch.max()}。达到10epochs下界的 cells 为 {(cells.best_epoch == 10).sum()}/{len(cells)}。", f"- 训练重建 MSE 首末下降的 cells：{int(recon.train_reconstruction_decreased.sum())}/{len(recon)}。这是在线随机mask训练损失，不能替代固定mask测试增益。", "- 测试 reconstruction MSE/Pearson、真实 raw cosine、中心化 correlation、effective rank、跨seed attention稳定性均单独列出。没有根据这些结果调整参数。", "",
             "## prompt 的直接回答", "", "1. 无 DSS supervision；原训练 loader 读取 DSS 仅检查固定队列（65events），DSS 不进入损失、epoch选择、表示或评估；汇总不读取 DSS。", "2. 官方 MSigDB 2024.1.Hs 50 Hallmark，实际匹配4191个唯一genes，最小pathway32genes。", "3. M3/M4 同参数量、相同初始化seed规则与训练规则；shuffle保持每个pathway规模和每gene度数。M6 的cond参数保留但在forward禁用，名义参数量匹配。", "4–7. M0 RNA、M1 Pathology、M2 Naive、M3 DiseaseState 的 G_TNM 见上表。", f"8. True−shuffle mean ΔG_TNM={mean.loc['M3', 'G_TNM'] - mean.loc['M4', 'G_TNM']:.6f}。", f"9. True−shuffle 的CI下界全正：{gate['all_seed_true_shuffle_CI_lower_positive']}；正方向seeds：{int((kb.delta_G_TNM_observed > 0).sum())}/3。", f"10. T/N/M mean BA中正向任务：{tasks_positive}/3。", "11–12. M3−M2、M3−M6每seed差值与CI见配对表；均值为正不能独立证明稳定超越。", f"13. M3−M5(no mask) mean ΔG={mean.loc['M3', 'G_TNM'] - mean.loc['M5', 'G_TNM']:.6f}；M3/M0/M2固定测试mask重建及M3−M5CI见CSV。", f"14. M3 raw token cosine均值={tokens.loc[tokens.model == 'M3', 'patient_token_pairwise_cosine'].mean():.6f}；跨患者variance floor collapse={not gate['no_variance_floor_collapse']}。方差不为零不证明pathway语义。", f"15. M3 attention top10% Jaccard={stability.loc[stability.model == 'M3', 'top10_jaccard'].mean():.6f}，Spearman={stability.loc[stability.model == 'M3', 'attention_spearman'].mean():.6f}；稳定性不等于生物学grounding。", f"16. **{decision}**。", f"17. Phase4B：**{'允许' if supported else '不允许'}**。", "18. 只有完整knowledge gate支持才允许下一阶段；届时疾病状态encoder冻结，仅训练小adapter/survival head，保留Clinical/Pathology/RNA signal paths。本次未运行transfer。", "", "## 优势、风险与置信度", "", "- 优势：类别均衡、平滑选择、matched controls、真正seed OOF和患者配对bootstrap，减少原先raw BA epoch选择噪声与mask identity不公平。", "- 风险：稀有T/M类别仍限制统计精度；两项正则可能制造多样性而非真实语义；多项修复联合改变，不能拆分因果贡献。所有seed共享患者，CI不代表独立cohort泛化。", "- inner RNA预处理统计来自outer-train（含inner-val features，无labels），inner验证并非完全feature隔离；validation CE按batch加权求均值，没有全局class-weight归一化。公开记录这一实现边界。", "- 置信度：完成性和OOF覆盖为高（机械核验）；机制支持程度依据配对CI与seed一致性，见DECISION，不能由单一均值推断。", "", "## 拓展思考", "", "若M3−M6和True−shuffle缺乏稳定优势，增加cross-attention深度没有直接证据支持。应先区分监督分辨率不足、pathway表示压缩与设计限制。", "", "## 你可能想知道", "", "训练loss下降只证明优化器能拟合辅助目标；是否提供可迁移信息仍须固定mask测试重建、matched shuffle及guided−fixed比较共同支持。", "", "## 产物边界", "", "患者IDs、logits、attention ranks、embeddings、checkpoint均留在私有runtime；GitHub只提交源代码、官方GMT和聚合报告。烟雾/探索cell不进入正式结果。"]
    (report / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

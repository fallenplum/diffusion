from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.phase_k0.concepts import load_concept_bank
from src.phase_k2.run import (
    BATCH,
    D,
    EPOCHS,
    PATH_EDGES,
    FrozenTextPathStudent,
    fold_omics,
    load_label_free_inputs,
    sha256_file,
)


SEEDS = (20260904, 20260905, 20260906)
GRAPH_CONCEPTS = tuple(PATH_EDGES)  # K1, K2, K4, K7
PERMUTATIONS = tuple(itertools.permutations(GRAPH_CONCEPTS))
IDENTITY = GRAPH_CONCEPTS
TAU = 0.07


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=keys).to_csv(tmp, index=False)
    tmp.replace(path)


def pearson(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def fit_permutation(model, path, queries, teacher, train_idx, mapping, seed, fold, device):
    """Fit one independent query-to-teacher permutation.

    `mapping[query_concept]` is the teacher concept assigned to that query.
    The teacher is external to the pathology forward and is never supplied as
    a model input. Every permutation gets a fresh model and optimizer.
    """
    torch.manual_seed(seed + fold * 1000)
    np.random.seed(seed + fold * 1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed + fold * 101 + 7919)
    query_edges = np.asarray(GRAPH_CONCEPTS, dtype=int)
    teacher_edges = np.asarray(mapping, dtype=int)
    model.train()
    for _epoch in range(EPOCHS):
        order = rng.permutation(np.asarray(train_idx, dtype=int))
        for start in range(0, len(order), BATCH):
            ix = order[start : start + BATCH]
            pred, _attention = model.student_forward(
                torch.as_tensor(path[ix], device=device),
                torch.as_tensor(queries, device=device),
            )
            target = torch.as_tensor(teacher[ix][:, teacher_edges], dtype=torch.float32, device=device)
            predicted = pred[:, query_edges]
            predicted = predicted - predicted.mean(dim=0, keepdim=True)
            target = target - target.mean(dim=0, keepdim=True)
            loss = 1.0 - F.cosine_similarity(predicted.reshape(-1), target.reshape(-1), dim=0)
            pairs = torch.triu_indices(len(query_edges), len(query_edges), offset=1, device=device)
            sign = torch.sign(target[:, pairs[0]] - target[:, pairs[1]])
            loss = loss + 0.25 * F.relu(
                0.10 - sign * (predicted[:, pairs[0]] - predicted[:, pairs[1]])
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()


@torch.no_grad()
def evaluate(model, path, queries, teacher, test_idx, mapping, device):
    scores = []
    query_edges = np.asarray(GRAPH_CONCEPTS, dtype=int)
    teacher_edges = np.asarray(mapping, dtype=int)
    for start in range(0, len(test_idx), BATCH):
        ix = np.asarray(test_idx[start : start + BATCH], dtype=int)
        pred, _attention = model.student_forward(
            torch.as_tensor(path[ix], device=device),
            torch.as_tensor(queries, device=device),
        )
        pred = pred.cpu().numpy()[:, query_edges]
        target = teacher[ix][:, teacher_edges]
        scores.append(np.asarray([pearson(pred[:, k], target[:, k]) for k in range(len(query_edges))]))
    return float(np.mean(np.concatenate(scores)))


def revised_edges(repo: Path) -> list[dict]:
    """Collapse deterministic K2 Omics/Clinical rows to five unique folds."""
    report = repo / "reports" / "phase_k2_sparse_medical_concept_graph"
    rows: list[dict] = []
    omics = pd.read_csv(report / "omics_edge_metrics.csv").drop_duplicates(["outer_fold", "concept_id"])
    for concept, group in omics.groupby("concept_id", sort=True):
        delta = group["real_minus_shuffled"].to_numpy(float)
        rows.append({
            "modality": "Omics", "edge": f"Omics->{concept}", "concept_id": concept,
            "n_unique_outer_folds": int(len(group)), "positive_folds": int(np.sum(delta > 0)),
            "positive_fraction": float(np.mean(delta > 0)), "mean_control_delta": float(np.mean(delta)),
            "interpretation": "RNA/CNA pathway activity versus frozen shuffled gene membership",
        })
    clinical = pd.read_csv(report / "clinical_edge_metrics.csv").drop_duplicates(["outer_fold", "concept_id"])
    for concept, group in clinical.groupby("concept_id", sort=True):
        value = group["expected_direction_spearman"].to_numpy(float)
        rows.append({
            "modality": "Clinical", "edge": f"Clinical->{concept}", "concept_id": concept,
            "n_unique_outer_folds": int(len(group)), "positive_folds": int(np.sum(value > 0)),
            "positive_fraction": float(np.mean(value > 0)), "mean_control_delta": float(np.mean(value)),
            "interpretation": "shared disease-extent context from the same TNM composite; K4/K7 are not independent clinical evidence",
        })
    return rows


def main(args):
    repo = Path(args.repo_root); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    raw = load_label_free_inputs(Path(args.data_root), repo)
    bank = load_concept_bank(repo / "src" / "phase_k0" / "concept_bank_v1.json")
    runtime = Path(args.runtime)
    path = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    artifact = torch.load(args.text_embeddings, map_location="cpu", weights_only=False)
    text = F.normalize(artifact["embeddings"].float(), dim=-1).numpy()
    # Same fixed SVD geometry as K2, with no trainable text parameters.
    _, singular, vt = np.linalg.svd(text, full_matrices=False)
    rank = int(np.sum(singular > singular[0] * 1e-3))
    basis = vt[:rank].T
    queries = text @ basis
    queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    queries32 = np.zeros((8, D), dtype=np.float32); queries32[:, :rank] = queries
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    metrics: list[dict] = []
    started = time.time()
    for seed in SEEDS:
        for fold in range(1, 6):
            train = np.flatnonzero(raw.outer_folds != fold)
            test = np.flatnonzero(raw.outer_folds == fold)
            real, _half = fold_omics(raw, bank, train, "mapped_gene_symbols")
            teacher = 0.5 * (real["rna"] + real["cna"])
            for perm_index, mapping in enumerate(PERMUTATIONS):
                # Initialize every cell identically across the 24 mappings;
                # the only changed factor is query-to-teacher assignment.
                init_seed = seed + fold * 1000
                torch.manual_seed(init_seed)
                np.random.seed(init_seed)
                model = FrozenTextPathStudent().to(device)
                fit_permutation(model, path, queries32, teacher, train, mapping, seed, fold, device)
                score = evaluate(model, path, queries32, teacher, test, mapping, device)
                metrics.append({
                    "seed": seed, "outer_fold": fold, "permutation_index": perm_index,
                    "mapping": "+".join(f"K{x+1}" for x in mapping),
                    "is_real": bool(mapping == IDENTITY), "score_alignment": score,
                    "query_set": "+".join(f"K{x+1}" for x in GRAPH_CONCEPTS),
                    "teacher_set": "+".join(f"K{x+1}" for x in mapping),
                })
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
        print(json.dumps({"seed": seed, "device": str(device), "elapsed": round(time.time() - started, 1)}), flush=True)
    metrics_df = pd.DataFrame(metrics)
    real_cells = metrics_df[metrics_df["is_real"]].set_index(["seed", "outer_fold"])["score_alignment"]
    perm_cells = metrics_df[~metrics_df["is_real"]].groupby(["seed", "outer_fold"])["score_alignment"].mean()
    deltas = (real_cells - perm_cells).reset_index(name="real_minus_permutation_mean")
    deltas["real_better"] = deltas["real_minus_permutation_mean"] > 0
    delta_rows = deltas.to_dict("records")
    perm_summary = metrics_df[~metrics_df["is_real"]].groupby(["permutation_index", "mapping"], as_index=False)["score_alignment"].agg(["mean", "std", "min", "max"]).reset_index()
    perm_summary = perm_summary.rename(columns={"mean": "permutation_mean", "std": "permutation_std", "min": "permutation_min", "max": "permutation_max"})
    real_mean = float(real_cells.mean())
    null_means = perm_summary["permutation_mean"].to_numpy(float)
    permutation_mean = float(np.mean(null_means)); permutation_std = float(np.std(null_means, ddof=1))
    count_ge = int(np.sum(null_means >= real_mean))
    exact_p = float((1 + count_ge) / 24)
    real_percentile = float(100.0 * (1 + np.sum(null_means < real_mean)) / 24.0)
    positive_fraction = float(deltas["real_better"].mean())
    topology_pass = bool(real_mean > permutation_mean and real_percentile >= 95.0 and exact_p <= 0.05 and positive_fraction >= 0.80)
    decision = "A. TOPOLOGY_SPECIFIC_INDUCTIVE_BIAS_CONFIRMED" if topology_pass else ("B. TOPOLOGY_DIRECTIONAL" if real_mean > permutation_mean and positive_fraction >= 0.50 else "C. TOPOLOGY_NOT_SUPPORTED")
    atomic_csv(output / "topology_permutation_metrics.csv", metrics)
    atomic_csv(output / "topology_null_distribution.csv", perm_summary.to_dict("records"))
    atomic_csv(output / "fold_seed_deltas.csv", delta_rows)
    atomic_csv(output / "revised_edge_summary.csv", revised_edges(repo))
    atomic_json(output / "run_manifest.json", {
        "phase": "K2R_TOPOLOGY_PERMUTATION_AUDIT", "status": "complete", "decision": decision,
        "n_patients": 512, "outer_folds": 5, "seeds": list(SEEDS), "permutations": 24,
        "identity_mapping": "+".join(f"K{x+1}" for x in IDENTITY), "query_set": "+".join(f"K{x+1}" for x in GRAPH_CONCEPTS),
        "device": str(device), "d": D, "tau": TAU, "epochs": EPOCHS,
        "trainable_text_parameters": 0, "trainable_path_parameters": sum(p.numel() for p in FrozenTextPathStudent().parameters() if p.requires_grad),
        "real_mean": real_mean, "permutation_mean": permutation_mean, "permutation_std": permutation_std,
        "real_percentile": real_percentile, "exact_p_value": exact_p, "positive_seed_fold_fraction": positive_fraction,
        "dss_used": False, "survival_run": False, "raw_wsi_used": False,
        "concept_bank_sha256": bank["content_sha256"], "text_embedding_sha256": sha256_file(Path(args.text_embeddings)),
        "fixed_split": "outputs/phase_m1r/fixed_patient_folds.csv",
    })
    report = "\n".join([
        "# Phase K2R — Topology-Specific Permutation Audit", "",
        f"- Decision: **{decision}**",
        f"- Real topology mean: **{real_mean:.6f}**",
        f"- 23-permutation mean ± std: **{permutation_mean:.6f} ± {permutation_std:.6f}**",
        f"- Real percentile: **{real_percentile:.2f}%**; exact p-value: **{exact_p:.6f}**",
        f"- Positive seed/fold fraction: **{positive_fraction:.6f}**",
        "- Omics and Clinical summaries use five unique outer folds; Clinical K4/K7 share one TNM disease-extent context.",
        "- DSS, survival, and raw WSI/SVS were not used.", "",
        "Each of the 24 query-to-teacher permutations was trained independently with the same model capacity and protocol.",
    ])
    (output / "RESULTS.md").write_text(report + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "real_mean": real_mean, "permutation_mean": permutation_mean, "permutation_std": permutation_std, "real_percentile": real_percentile, "exact_p": exact_p, "positive_fraction": positive_fraction, "out": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--repo-root", required=True); p.add_argument("--data-root", required=True); p.add_argument("--runtime", required=True); p.add_argument("--text-embeddings", required=True); p.add_argument("--output", required=True); p.add_argument("--cpu", action="store_true"); main(p.parse_args())

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "outputs/pathology_context_phase1"
SEED = 20260723


def load_embeddings(model: str) -> pd.DataFrame | None:
    base = PHASE / "embeddings" / model
    if model == "B0_mean":
        path = base / f"seed_{SEED}.csv"
        return pd.read_csv(path) if path.exists() else None
    paths = sorted(base.glob(f"fold_*_seed_{SEED}.csv"))
    if len(paths) != 5:
        return None
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def parse_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray([json.loads(value) for value in frame.embedding], dtype=np.float32)


def diagnostics(model: str, frame: pd.DataFrame):
    x = parse_matrix(frame)
    norms = np.linalg.norm(x, axis=1)
    stds = np.std(x, axis=0)
    normalized = x / np.maximum(norms[:, None], 1e-12)
    cosine = normalized @ normalized.T
    pair = cosine[np.triu_indices(len(x), 1)]
    return {
        "model": model, "n_patients": len(frame), "embedding_dim": x.shape[1],
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "feature_std_mean": float(stds.mean()), "feature_std_min": float(stds.min()),
        "near_zero_std_features": int(np.sum(stds < 1e-6)),
        "pairwise_cosine_mean": float(pair.mean()), "pairwise_cosine_std": float(pair.std()),
        "pairwise_cosine_p05": float(np.quantile(pair, .05)),
        "pairwise_cosine_p95": float(np.quantile(pair, .95)),
        "collapse_flag": bool(stds.mean() < 1e-4 or pair.mean() > .995),
    }


def plot_tsne(model: str, frame: pd.DataFrame, manifest: pd.DataFrame):
    x = parse_matrix(frame)
    x = PCA(n_components=min(32, x.shape[1], len(x) - 1), random_state=SEED).fit_transform(StandardScaler().fit_transform(x))
    xy = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=SEED).fit_transform(x)
    data = frame.merge(manifest[["patient_id", "tissue_source_site"]], on="patient_id", how="left")
    data["risk_quartile"] = pd.qcut(data.oof_risk, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    event = data.dss_status.to_numpy(int)
    axes[0].scatter(xy[:, 0], xy[:, 1], c=event, cmap="coolwarm", s=16, alpha=.8)
    axes[0].set_title("DSS event")
    risk = data.oof_risk.to_numpy(float)
    p = axes[1].scatter(xy[:, 0], xy[:, 1], c=risk, cmap="viridis", s=16, alpha=.8)
    fig.colorbar(p, ax=axes[1], label="OOF risk")
    axes[1].set_title("OOF risk")
    top_sites = data.tissue_source_site.value_counts().head(8).index
    site = data.tissue_source_site.where(data.tissue_source_site.isin(top_sites), "other")
    codes, labels = pd.factorize(site, sort=True)
    axes[2].scatter(xy[:, 0], xy[:, 1], c=codes, cmap="tab10", s=16, alpha=.8)
    axes[2].set_title("TSS (top 8 + other)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{model} patient embedding t-SNE (exploratory only)")
    out = PHASE / "figures/embeddings"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{model}_tsne.png", dpi=180)
    plt.close(fig)


def main():
    manifest = pd.read_csv(PHASE / "audit/patient_manifest_used.csv")
    models = ["B0_mean", "B1", "B2", "P0", "P1", "P2", "P0_random040", "P0_nocoords", "P0_spatial025"]
    rows = []
    for model in models:
        frame = load_embeddings(model)
        if frame is None:
            continue
        if len(frame) != 487 or frame.patient_id.nunique() != 487:
            raise RuntimeError(f"{model}: incomplete embedding OOF")
        rows.append(diagnostics(model, frame))
        if model in {"B0_mean", "B1", "B2", "P0", "P1", "P2"}:
            plot_tsne(model, frame, manifest)
    out = PHASE / "statistical_tests/embedding_diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

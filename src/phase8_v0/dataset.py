"""Phase8-V0 private dataset registry and source audit helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def ahash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); tmp.replace(path)


def load_private(parent):
    d = dict(np.load(Path(parent) / "private_dataset.npz", allow_pickle=True))
    ids = d["patient_ids"].astype(str)
    if len(ids) != 512 or len(set(ids)) != len(ids): raise ValueError("patient ID count/uniqueness failure")
    if d["DSS_event"].sum() != 65: raise ValueError("DSS event count is not 65")
    for key in ("static", "titan", "uni2"):
        if not np.isfinite(d[key]).all(): raise ValueError(f"nonfinite {key}")
    # raw RNA/CNA matrices intentionally retain missing measurements; the
    # fold-local Top512/state routines perform train-only imputation.
    for key in ("raw_rna", "raw_cna", "rna_genes", "cna_genes"):
        if not np.isfinite(d[key][np.isfinite(d[key])]).all(): raise ValueError(f"invalid {key}")
    return d


def source_audit(repo, parent, data_root, cache_root):
    repo = Path(repo); parent = Path(parent); data_root = Path(data_root); cache_root = Path(cache_root)
    required = [
        repo / "reports/phase_m2r2_omics_fidelity/RESULTS.md", repo / "reports/phase_m2r2_omics_fidelity/protocol.json",
        repo / "reports/phase_m3r2_simple_fusion_audit/RESULTS.md", repo / "reports/phase_m8p1r_strong_anchor_residual_repair/RESULTS.md",
        repo / "reports/phase7a_foundation_headroom/RESULTS.md", repo / "reports/phase7a_foundation_headroom/protocol.json",
        repo / "reports/phase7a_foundation_headroom/independent_validation.json", repo / "reports/phase7b_bridge_evidence/RESULTS.md",
        repo / "reports/phase7c_morphology_conditional_audit/RESULTS.md", repo / "reports/phase7d_morphology_calibrated_molecular_concept_audit/RESULTS.md",
        repo / "outputs/phase_m1r/fixed_patient_folds.csv", parent / "private_dataset.npz", parent / "uni2_patient_means.npz",
        cache_root / "patient_representations.npz", parent / "manifest.json", parent / "protocol.json",
    ]
    rows = []
    for p in required:
        rows.append({"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() else None,
                     "sha256": digest(p) if p.exists() else None})
    if not all(x["exists"] for x in rows): raise FileNotFoundError([x["path"] for x in rows if not x["exists"]])
    return rows


def registry(d, parent, cache_root):
    ids = d["patient_ids"].astype(str); cache_root = Path(cache_root)
    return {
        "patients": int(len(ids)), "unique_patient_ids": int(len(set(ids))), "DSS_events": int(d["DSS_event"].sum()),
        "DSS_valid": int(d["DSS_valid"].sum()), "RNA_shape": list(d["raw_rna"].shape), "CNA_shape": list(d["raw_cna"].shape),
        "RNA_gene_shape": list(d["rna_genes"].shape), "CNA_gene_shape": list(d["cna_genes"].shape),
        "UNI2_shape": list(d["uni2"].shape), "TITAN_shape": list(d["titan"].shape), "static_concept_shape": list(d["static"].shape),
        "patient_id_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "fold_counts": {str(k): int((d["folds"] == k).sum()) for k in np.unique(d["folds"])},
        "frozen_cache": str(cache_root / "patient_representations.npz"), "parent_run": str(parent),
    }

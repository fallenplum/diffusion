from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.phase_m2.data import RawInputs, _transform_matrix


@dataclass(frozen=True)
class ConceptFoldArrays:
    """Fold-local, label-free concept evidence."""

    omics_evidence: np.ndarray  # [patient, concept, (RNA, CNA)]
    audit: list[dict[str, Any]]
    concept_bank_hash: str


def _membership_hash(symbols: list[str]) -> str:
    return hashlib.sha256("\n".join(symbols).encode()).hexdigest()


def _fit_modality(raw_values: np.ndarray, symbols: np.ndarray, train_idx: np.ndarray, modality: str, bank: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    transformed, transform_audit = _transform_matrix(raw_values, train_idx, modality)
    symbol_to_indices: dict[str, list[int]] = {}
    for index, symbol in enumerate(symbols.tolist()):
        clean = str(symbol).strip().upper()
        if clean:
            symbol_to_indices.setdefault(clean, []).append(index)

    outputs: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for concept in bank["concepts"]:
        genes = [str(g).strip().upper() for g in concept["mapped_gene_symbols"]]
        matched = [gene for gene in genes if gene in symbol_to_indices]
        if not matched:
            raise ValueError(f"{modality} concept {concept['concept_id']} has no feature overlap")
        gene_values = np.column_stack([
            np.mean(transformed[:, symbol_to_indices[gene]], axis=1)
            for gene in matched
        ]).astype(np.float32)
        summary = np.mean(gene_values, axis=1).astype(np.float32)
        train_values = summary[np.asarray(train_idx, dtype=int)]
        mean = float(np.mean(train_values))
        std = float(np.std(train_values))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        outputs.append(((summary - mean) / std).astype(np.float32))
        audits.append({
            "concept_id": concept["concept_id"],
            "modality": modality,
            "official_gene_count": len(genes),
            "matched_gene_count": len(matched),
            "matched_gene_symbols_sha256": _membership_hash(matched),
            "summary": "mean of mapped gene-level means",
            "train_standardizer_mean": mean,
            "train_standardizer_std": std,
            "transform": transform_audit.get("transform", "none"),
            "imputation": "outer_train_median",
            "selection_uses_outcomes": False,
        })
    return np.stack(outputs, axis=1), audits


def build_concept_evidence(raw: RawInputs, train_idx: np.ndarray, bank: dict[str, Any]) -> ConceptFoldArrays:
    """Build fixed concept summaries with train-only imputation and scaling."""
    rna, rna_audit = _fit_modality(raw.rna, raw.rna_symbols, train_idx, "RNA", bank)
    cna, cna_audit = _fit_modality(raw.cna, raw.cna_symbols, train_idx, "CNA", bank)
    evidence = np.stack([rna, cna], axis=-1).astype(np.float32)
    audit = rna_audit + cna_audit
    for row in audit:
        row["concept_bank_sha256"] = bank["content_sha256"]
        row["outer_fold_train_count"] = int(len(train_idx))
    return ConceptFoldArrays(evidence, audit, bank["content_sha256"])


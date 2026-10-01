from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.phase_m2.common import stable_int
from src.phase_m2.data import (
    METADATA_ROOT,
    FoldArrays,
    RawInputs,
    _fit_clinical,
    _fit_group_matrix,
    _fit_numeric,
    _read_signature_groups,
    load_raw_inputs,
    prepare_path_cache,
)


@dataclass
class M6AFoldArrays:
    """Fold-local arrays for the M6A representation audit.

    The original FoldArrays fields are retained for the six-signature control;
    the three 50-group families are packed as [patient, group, width, channel],
    where channel 0 is RNA and channel 1 is CNA.
    """

    path_tokens: np.ndarray
    rna: np.ndarray
    cna: np.ndarray
    clinical: np.ndarray
    mcat_groups: np.ndarray
    mcat_widths: tuple[tuple[int, int], ...]
    hallmark_rna_groups: np.ndarray
    hallmark_rna_widths: tuple[int, ...]
    hallmark_rc_groups: np.ndarray
    hallmark_rc_widths: tuple[tuple[int, int], ...]
    shuffled_rc_groups: np.ndarray
    shuffled_rc_widths: tuple[tuple[int, int], ...]
    group_names: tuple[str, ...]
    shuffled_group_names: tuple[str, ...]
    audit: dict[str, Any]


def _pack_groups(
    rna_values: np.ndarray,
    rna_widths: tuple[int, ...],
    cna_values: np.ndarray | None,
    cna_widths: tuple[int, ...] | None,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    cna_values = np.zeros((rna_values.shape[0], rna_values.shape[1], 1), dtype=np.float32) if cna_values is None else cna_values
    cna_widths = tuple(0 for _ in rna_widths) if cna_widths is None else cna_widths
    if len(rna_widths) != len(cna_widths):
        raise ValueError("RNA/CNA group count mismatch")
    width = max(rna_values.shape[-1], cna_values.shape[-1])
    packed = np.zeros((rna_values.shape[0], rna_values.shape[1], width, 2), dtype=np.float32)
    packed[:, :, : rna_values.shape[-1], 0] = rna_values
    packed[:, :, : cna_values.shape[-1], 1] = cna_values
    return packed, tuple((int(r), int(c)) for r, c in zip(rna_widths, cna_widths))


def _unique_symbols(symbols: np.ndarray) -> list[str]:
    return sorted({str(x).strip().upper() for x in symbols.tolist() if str(x).strip()})


def _shuffled_groups(
    names: tuple[str, ...],
    rna_widths: tuple[int, ...],
    cna_widths: tuple[int, ...],
    rna_universe: list[str],
    cna_universe: list[str],
    fold: int,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]], dict[str, Any]]:
    """Make one outcome-blind, size-matched assignment per outer fold."""
    seed = 20260925 + stable_int(f"m6a-shuffle:{fold}", modulo=1000003)
    rng = np.random.default_rng(seed)
    if sum(rna_widths) > len(rna_universe) or sum(cna_widths) > len(cna_universe):
        raise RuntimeError("size-matched shuffled assignment exceeds available gene universe")
    rna_perm = rng.permutation(np.asarray(rna_universe, dtype=object))
    cna_perm = rng.permutation(np.asarray(cna_universe, dtype=object))
    rna_groups: list[tuple[str, list[str]]] = []
    cna_groups: list[tuple[str, list[str]]] = []
    rpos = cpos = 0
    for i, name in enumerate(names):
        rw = int(rna_widths[i])
        cw = int(cna_widths[i])
        rna_groups.append((f"SHUFFLED_{i:02d}_{name}", rna_perm[rpos : rpos + rw].tolist()))
        cna_groups.append((f"SHUFFLED_{i:02d}_{name}", cna_perm[cpos : cpos + cw].tolist()))
        rpos += rw
        cpos += cw
    audit = {
        "shuffle_seed": int(seed),
        "outer_fold": int(fold),
        "outcome_blind": True,
        "size_matched": True,
        "assignment": "deterministic permutation of fold-available RNA/CNA symbol universes",
        "original_rna_widths": list(map(int, rna_widths)),
        "original_cna_widths": list(map(int, cna_widths)),
        "shuffled_rna_widths": list(map(len, [x[1] for x in rna_groups])),
        "shuffled_cna_widths": list(map(len, [x[1] for x in cna_groups])),
        "rna_universe_size": len(rna_universe),
        "cna_universe_size": len(cna_universe),
        "assignment_sha256": hashlib.sha256(
            "\n".join(g + ":" + ",".join(v) for g, v in rna_groups + cna_groups).encode()
        ).hexdigest(),
    }
    return rna_groups, cna_groups, audit


def _group_audit_rows(
    fold: int,
    family: str,
    names: tuple[str, ...],
    rna_audit: list[dict[str, Any]],
    cna_audit: list[dict[str, Any]] | None,
    original_rna_widths: tuple[int, ...],
    original_cna_widths: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cna_audit = cna_audit or [{"local_gene_count": 0, "official_gene_count": 0, "selected_symbols_sha256": ""} for _ in names]
    for i, name in enumerate(names):
        ra = rna_audit[i]
        ca = cna_audit[i]
        rows.append(
            {
                "outer_fold": int(fold),
                "family": family,
                "group_index": int(i),
                "group_name": name,
                "rna_width": int(original_rna_widths[i]),
                "cna_width": int(original_cna_widths[i]),
                "rna_overlap": int(ra.get("local_gene_count", 0)),
                "cna_overlap": int(ca.get("local_gene_count", 0)),
                "rna_missing_symbols": int(max(0, ra.get("official_gene_count", 0) - ra.get("local_gene_count", 0))),
                "cna_missing_symbols": int(max(0, ca.get("official_gene_count", 0) - ca.get("local_gene_count", 0))),
                "rna_symbols_sha256": ra.get("selected_symbols_sha256", ""),
                "cna_symbols_sha256": ca.get("selected_symbols_sha256", ""),
                "zero_overlap_fail_closed": True,
                "zero_overlap_detected": bool(int(ra.get("local_gene_count", 0)) == 0 or int(ca.get("local_gene_count", 0)) == 0),
            }
        )
    return rows


def preprocess_fold_m6a(raw: RawInputs, path_tokens: np.ndarray, fold: int) -> M6AFoldArrays:
    train_idx = np.flatnonzero(raw.outer_folds != fold)
    rna, rna_audit = _fit_numeric(raw.rna, train_idx, output_dim=512, modality="RNA")
    cna, cna_audit = _fit_numeric(raw.cna, train_idx, output_dim=256, modality="CNA")
    clinical, clinical_audit = _fit_clinical(raw.clinical, train_idx)

    mcat_groups = _read_signature_groups(METADATA_ROOT / "mcat_signatures.csv")
    mcat_rna, mcat_rna_widths, mcat_rna_audit = _fit_group_matrix(raw.rna, raw.rna_symbols, train_idx, mcat_groups, "RNA")
    mcat_cna, mcat_cna_widths, mcat_cna_audit = _fit_group_matrix(raw.cna, raw.cna_symbols, train_idx, mcat_groups, "CNA")
    mcat_packed, mcat_widths = _pack_groups(mcat_rna, mcat_rna_widths, mcat_cna, mcat_cna_widths)

    hallmark_groups = _read_signature_groups(METADATA_ROOT / "mmp_hallmarks_signatures.csv")
    names = tuple(name for name, _ in hallmark_groups)
    h_rna, h_rna_widths, h_rna_audit = _fit_group_matrix(raw.rna, raw.rna_symbols, train_idx, hallmark_groups, "RNA")
    h_cna, h_cna_widths, h_cna_audit = _fit_group_matrix(raw.cna, raw.cna_symbols, train_idx, hallmark_groups, "CNA")
    h_rna_packed, _ = _pack_groups(h_rna, h_rna_widths, None, None)
    h_rc_packed, h_rc_widths = _pack_groups(h_rna, h_rna_widths, h_cna, h_cna_widths)

    shuffled_rna_groups, shuffled_cna_groups, shuffled_meta = _shuffled_groups(
        names,
        h_rna_widths,
        h_cna_widths,
        _unique_symbols(raw.rna_symbols),
        _unique_symbols(raw.cna_symbols),
        fold,
    )
    s_rna, s_rna_widths, s_rna_audit = _fit_group_matrix(raw.rna, raw.rna_symbols, train_idx, shuffled_rna_groups, "RNA")
    s_cna, s_cna_widths, s_cna_audit = _fit_group_matrix(raw.cna, raw.cna_symbols, train_idx, shuffled_cna_groups, "CNA")
    s_rc_packed, s_rc_widths = _pack_groups(s_rna, s_rna_widths, s_cna, s_cna_widths)

    group_audit = _group_audit_rows(fold, "hallmark", names, h_rna_audit, h_cna_audit, h_rna_widths, h_cna_widths)
    group_audit += _group_audit_rows(fold, "shuffled", tuple(x[0] for x in shuffled_rna_groups), s_rna_audit, s_cna_audit, s_rna_widths, s_cna_widths)
    audit = {
        "outer_fold": int(fold),
        "rna": rna_audit,
        "cna": cna_audit,
        "clinical": clinical_audit,
        "mcat_signature_groups": mcat_rna_audit,
        "mcat_signature_cna_groups": mcat_cna_audit,
        "hallmark_group_count": len(names),
        "hallmark_rna_audit": h_rna_audit,
        "hallmark_cna_audit": h_cna_audit,
        "shuffled_rna_audit": s_rna_audit,
        "shuffled_cna_audit": s_cna_audit,
        "hallmark_group_audit_rows": group_audit,
        "shuffled_pathway_audit": shuffled_meta,
    }
    return M6AFoldArrays(
        path_tokens=path_tokens,
        rna=rna,
        cna=cna,
        clinical=clinical,
        mcat_groups=mcat_packed,
        mcat_widths=mcat_widths,
        hallmark_rna_groups=h_rna_packed,
        hallmark_rna_widths=tuple(map(int, h_rna_widths)),
        hallmark_rc_groups=h_rc_packed,
        hallmark_rc_widths=h_rc_widths,
        shuffled_rc_groups=s_rc_packed,
        shuffled_rc_widths=s_rc_widths,
        group_names=names,
        shuffled_group_names=tuple(x[0] for x in shuffled_rna_groups),
        audit=audit,
    )


__all__ = ["M6AFoldArrays", "RawInputs", "load_raw_inputs", "prepare_path_cache", "preprocess_fold_m6a"]

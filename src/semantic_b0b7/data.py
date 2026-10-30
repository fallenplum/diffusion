"""Outcome-free, partition-local molecular and clinical background contracts.

No loader in this module opens an endpoint file. All returned per-person arrays
are private execution artifacts. The primary molecular inputs are measured,
already-log2 RNA and signed CNA dosage; this code applies no logarithm.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def array_sha(value):
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()


def fit_rows(indices, n):
    idx = np.asarray(indices)
    if idx.ndim != 1 or idx.dtype.kind not in "iu" or len(idx) < 2:
        raise ValueError("at least two unique integer training rows required")
    if len(np.unique(idx)) != len(idx) or (idx < 0).any() or (idx >= n).any():
        raise ValueError("invalid or duplicate training row")
    return idx.astype(np.int64)


def read_membership(gmt, symbols):
    """External GMT order is retained; no cohort target selects a pathway."""
    lookup = {str(s): i for i, s in enumerate(symbols)}
    names, rows = [], []
    for line in gmt.splitlines():
        fields = line.rstrip().split("\t")
        if len(fields) < 3:
            continue
        names.append(fields[0])
        row = np.zeros(len(lookup), bool)
        for gene in fields[2:]:
            if gene in lookup:
                row[lookup[gene]] = True
        rows.append(row)
    if len(names) != 50 or len(set(names)) != 50 or len(lookup) != len(symbols):
        raise ValueError("unique canonical symbols and exactly 50 fixed pathways required")
    return tuple(names), np.stack(rows)


@dataclass
class ObservedScale:
    mean: np.ndarray
    sd: np.ndarray
    measured: np.ndarray
    train_hash: str

    @classmethod
    def fit(cls, values, rows):
        x = np.asarray(values, np.float64)
        if x.ndim != 2 or np.isinf(x).any():
            raise ValueError("finite-or-NaN measurement matrix required")
        idx = fit_rows(rows, len(x))
        v, m = x[idx], np.isfinite(x[idx])
        count = m.sum(0)
        mean = np.where(m, v, 0).sum(0) / np.maximum(count, 1)
        var = np.where(m, (v - mean) ** 2, 0).sum(0) / np.maximum(count, 1)
        sd = np.sqrt(var)
        return cls(mean, np.where(sd > 1e-8, sd, 1.0), count, array_sha(idx))

    def transform(self, values):
        x = np.asarray(values, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or np.isinf(x).any():
            raise ValueError("measurement matrix changed")
        mask = np.isfinite(x) & (self.measured[None] > 0)
        return np.where(mask, (x - self.mean) / self.sd, 0).astype(np.float32), mask


@dataclass
class ProgramTransform:
    rna_scale: ObservedScale
    gain_scale: ObservedScale
    loss_scale: ObservedScale
    rna_membership: np.ndarray
    cna_membership: np.ndarray
    min_coverage: float = 0.5
    min_genes: int = 5

    @classmethod
    def fit(cls, rna, cna, rows, rna_membership, cna_membership, *, min_coverage=0.5, min_genes=5):
        rna, cna = np.asarray(rna), np.asarray(cna)
        rm, cm = np.asarray(rna_membership, bool), np.asarray(cna_membership, bool)
        if len(rna) != len(cna) or rm.shape != (50, rna.shape[1]) or cm.shape != (50, cna.shape[1]):
            raise ValueError("unaligned 50-pathway measurements")
        if not 0 < min_coverage <= 1 or min_genes < 1:
            raise ValueError("invalid locked coverage rule")
        return cls(ObservedScale.fit(rna, rows), ObservedScale.fit(np.maximum(cna, 0), rows),
                   ObservedScale.fit(np.maximum(-cna, 0), rows), rm, cm, min_coverage, min_genes)

    def transform(self, rna, cna):
        channels, masks, coverages = [], [], []
        for x, st, membership in ((rna, self.rna_scale, self.rna_membership),
                (np.maximum(cna, 0), self.gain_scale, self.cna_membership),
                (np.maximum(-cna, 0), self.loss_scale, self.cna_membership)):
            z, measured = st.transform(x)
            member = membership.astype(np.float32)
            count = measured.astype(np.float32) @ member.T
            total = member.sum(1)
            coverage = count / np.maximum(total, 1)[None]
            valid = (coverage >= self.min_coverage) & (count >= self.min_genes)
            value = z @ member.T / np.maximum(count, 1)
            channels.append(np.where(valid, value, 0))
            masks.append(valid)
            coverages.append(coverage)
        return {"values": np.stack(channels, -1).astype(np.float32),
                "measured": np.stack(masks, -1), "coverage": np.stack(coverages, -1),
                "fit_hash": self.rna_scale.train_hash}


def pretext_split(folds, *, outer_fold=1, inner_partition=0, split_seed=103001):
    """P1 uses no event stratification: no outcome vector is even accepted."""
    folds = np.asarray(folds)
    outer_train = np.flatnonzero(folds != outer_fold)
    outer_test = np.flatnonzero(folds == outer_fold)
    parts = list(KFold(3, shuffle=True, random_state=split_seed + outer_fold).split(outer_train))
    a, b = parts[inner_partition]
    return outer_train[a], outer_train[b], outer_test


CLINICAL_FIELDS = ("AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE")


def clinical_design(raw, rows):
    """Fit categories, age imputation and scaling solely on specified rows."""
    if set(raw) != set(CLINICAL_FIELDS):
        raise ValueError("exact six baseline fields required")
    n = len(raw["AGE"])
    fit = fit_rows(rows, n)
    age = np.array([float(s) if str(s).replace(".", "", 1).isdigit() else np.nan for s in raw["AGE"]])
    if not np.isfinite(age[fit]).any():
        raise ValueError("no observed training age")
    median = float(np.nanmedian(age[fit]))
    columns = [np.where(np.isfinite(age), age, median)[:, None]]
    levels = {}
    for field in CLINICAL_FIELDS[1:]:
        value = np.asarray(raw[field]).astype(str)
        if len(value) != n:
            raise ValueError("unaligned clinical fields")
        levels[field] = np.unique(value[fit])
        columns.append(value[:, None] == levels[field][None])
    x = np.concatenate(columns, 1).astype(np.float64)
    mean, sd = x[fit].mean(0), x[fit].std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return (x - mean) / sd, {"age_median": median, "levels": levels,
                            "mean": mean, "sd": sd, "fit_hash": array_sha(fit)}


def background_crossfit(clinical, representation, train, evaluation, *, alpha=1.0, seed=103017, clinical_raw=None):
    """OOF regression in ONE frozen encoder's coordinates, never across encoders.

    This function receives no labels. Its caller must freeze the representation
    first. OOF background models exclude each held patient's representation from
    regression fitting; evaluation backgrounds use the full training fit.
    """
    c, z = np.asarray(clinical, np.float64), np.asarray(representation, np.float64)
    tr, ev = fit_rows(train, len(c)), np.asarray(evaluation, np.int64)
    if len(c) != len(z) or c.ndim != 2 or not np.isfinite(c).all() or not np.isfinite(z).all():
        raise ValueError("finite aligned frozen features required")
    if np.intersect1d(tr, ev).size or (ev < 0).any() or (ev >= len(c)).any():
        raise ValueError("background evaluation overlaps fit")
    flat = z.reshape(len(z), -1)
    mu = np.full_like(flat, np.nan)
    groups = []
    def design(fit):
        if clinical_raw is not None:
            return clinical_design(clinical_raw, fit)
        # Numeric mode is for already-observed synthetic/controlled covariates.
        # Refit its scaling per background fold, never use global fold moments.
        mean, sd = c[fit].mean(0), c[fit].std(0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        return (c - mean) / sd, {"fit_hash": array_sha(fit)}
    for a, b in KFold(3, shuffle=True, random_state=seed).split(tr):
        fit, held = tr[a], tr[b]
        x, state = design(fit)
        model = Ridge(alpha=alpha, solver="svd").fit(x[fit], flat[fit])
        mu[held] = model.predict(x[held])
        groups.append({"fit_hash": array_sha(fit), "held_hash": array_sha(held),
                       "clinical_preprocessing_fit_hash": state["fit_hash"],
                       "fit_n": len(fit), "held_n": len(held)})
    x, full_state = design(tr)
    full = Ridge(alpha=alpha, solver="svd").fit(x[tr], flat[tr])
    if len(ev):
        mu[ev] = full.predict(x[ev])
    return {"mean": mu.reshape(z.shape), "full_model": full, "full_clinical_state": full_state,
            "audit": {"background_models": 4, "train_hash": array_sha(tr),
                      "evaluation_hash": array_sha(ev), "folds": groups, "DSS_read": False,
                      "encoder_coordinates": "one_frozen_current_partition_model",
                      "clinical_preprocessing": "raw_six_field_refit_in_each_background_fit" if clinical_raw is not None else "numeric_observed_covariates_rescaled_per_fit",
                      "training_predictions": "out_of_fold_background_regression",
                      "evaluation_predictions": "full_training_background_regression"}}

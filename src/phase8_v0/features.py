"""Fold-local clinical, pathology and omics representations."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from phase7_p0_biology_prompt_audit import raw_top512, gene_z, group_scores, zscale


def ahash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


CAT = ["SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"]


@dataclass
class FoldFeatures:
    groups: dict
    seed: int
    fit_idx: np.ndarray
    audit: dict

    def __post_init__(self):
        self._fit_idx = np.asarray(self.fit_idx, dtype=int)
        self.audit = {"fit_n": int(len(self._fit_idx)), "fit_indices_sha256": ahash(self._fit_idx),
                      "clinical_levels": {}, "pca": {}, "molecular": {}}
        self._fitted = False

    def fit(self, d):
        idx = self._fit_idx
        age = np.asarray(d["clinical_AGE"], dtype="U80")
        aval = np.array([float(x) if str(x).strip() not in ("", "UNKNOWN", "nan") else np.nan for x in age])
        med = float(np.nanmedian(aval[idx])); aval = np.where(np.isfinite(aval), aval, med)
        clinical = [aval[:, None]]
        for cat in CAT:
            vals = np.asarray(d["clinical_" + cat], dtype="U80")
            levels = np.unique(vals[idx]); self.audit["clinical_levels"][cat] = levels.tolist()
            clinical.append(np.stack([vals == v for v in levels], axis=1).astype(float))
        c = np.concatenate(clinical, axis=1); c, self.c_mean, self.c_sd = zscale(c, idx)
        self.age_median = med; self.c_dim = c.shape[1]

        self.pca_u, self.u_mean, self.u_sd = self._fit_pca(d["uni2"], idx, "U")
        self.pca_t, self.t_mean, self.t_sd = self._fit_pca(d["titan"], idx, "T")
        sraw = np.asarray(d["static"], float)
        s, self.s_mean, self.s_sd = zscale(sraw, idx)

        rraw, keep, rstats = raw_top512(d["raw_rna"], idx)
        rz, rstat = gene_z(d["rna_genes"], idx)
        cz, cstat = gene_z(d["cna_genes"], idx)
        rstate, rsets = group_scores(rz, d["rna_gene_symbols"], self.groups, rstat["usable"])
        cstate, csets = group_scores(cz, d["cna_gene_symbols"], self.groups, cstat["usable"])
        self.r_keep = keep; self.r_mean = rraw[idx].mean(0); self.r_sd = rraw[idx].std(0); self.r_sd[self.r_sd < 1e-8] = 1.
        self.r_pca = PCA(n_components=16, svd_solver="full", random_state=self.seed).fit((rraw[idx] - self.r_mean) / self.r_sd)
        self.m_mean = rstate[idx].mean(0); self.m_sd = rstate[idx].std(0); self.m_sd[self.m_sd < 1e-8] = 1.
        self.a_mean = cstate[idx].mean(0); self.a_sd = cstate[idx].std(0); self.a_sd[self.a_sd < 1e-8] = 1.
        self.audit.update({"clinical_dim": int(self.c_dim), "pca_rank": {"U": 16, "T": 16, "R": 16},
                           "pca_explained_variance": {"U": self.pca_u.explained_variance_ratio_.tolist(),
                                                       "T": self.pca_t.explained_variance_ratio_.tolist(),
                                                       "R": self.r_pca.explained_variance_ratio_.tolist()},
                           "RNA_top512_sha256": ahash(keep),
                           "molecular_gene_counts": {"RNA": [len(x) for x in rsets], "CNA": [len(x) for x in csets]}})
        self._fitted = True
        return self

    def _fit_pca(self, x, idx, name):
        x = np.asarray(x, float); mu = x[idx].mean(0); sd = x[idx].std(0); sd[sd < 1e-8] = 1.
        pca = PCA(n_components=16, svd_solver="full", random_state=self.seed).fit((x[idx] - mu) / sd)
        return pca, mu, sd

    def transform(self, d):
        if not self._fitted: raise RuntimeError("fit before transform")
        idx = self._fit_idx
        age = np.asarray(d["clinical_AGE"], dtype="U80")
        aval = np.array([float(x) if str(x).strip() not in ("", "UNKNOWN", "nan") else np.nan for x in age])
        aval = np.where(np.isfinite(aval), aval, self.age_median)
        clinical = [aval[:, None]]
        for cat in CAT:
            vals = np.asarray(d["clinical_" + cat], dtype="U80")
            levels = self.audit["clinical_levels"][cat]
            clinical.append(np.stack([vals == v for v in levels], axis=1).astype(float))
        c = np.concatenate(clinical, axis=1); c = (c - self.c_mean) / self.c_sd
        u = self.pca_u.transform((d["uni2"] - self.u_mean) / self.u_sd)
        t = self.pca_t.transform((d["titan"] - self.t_mean) / self.t_sd)
        s = (d["static"] - self.s_mean) / self.s_sd
        rraw, _, _ = raw_top512(d["raw_rna"], idx)
        r = self.r_pca.transform((rraw - self.r_mean) / self.r_sd)
        rz, rstat = gene_z(d["rna_genes"], idx); cz, cstat = gene_z(d["cna_genes"], idx)
        rs, _ = group_scores(rz, d["rna_gene_symbols"], self.groups, rstat["usable"])
        cs, _ = group_scores(cz, d["cna_gene_symbols"], self.groups, cstat["usable"])
        m = (rs - self.m_mean) / self.m_sd; a = (cs - self.a_mean) / self.a_sd
        return {"C": c.astype(np.float64), "U": u.astype(np.float64), "T": t.astype(np.float64),
                "S": s.astype(np.float64), "P": np.concatenate([u, t, s], axis=1).astype(np.float64),
                "R": r.astype(np.float64), "M": m.astype(np.float64), "A": a.astype(np.float64),
                "O": np.concatenate([r, m, a], axis=1).astype(np.float64)}, self.audit


def residualize_crossfit(c, x, fit_idx, seed, alpha=10.0, n_splits=3):
    """Clinical residuals with train OOF predictions and full-fit test map."""
    fit_idx = np.asarray(fit_idx, dtype=int); x = np.asarray(x, float); c = np.asarray(c, float)
    out = np.empty_like(x); kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    local = np.arange(len(fit_idx))
    for trloc, val_loc in kf.split(local):
        model = Ridge(alpha=alpha).fit(c[fit_idx[trloc]], x[fit_idx[trloc]])
        out[fit_idx[val_loc]] = model.predict(c[fit_idx[val_loc]])
    full = Ridge(alpha=alpha).fit(c[fit_idx], x[fit_idx])
    rest = np.setdiff1d(np.arange(len(x)), fit_idx, assume_unique=False)
    if len(rest): out[rest] = full.predict(c[rest])
    residual = x - out
    return residual, {"alpha": alpha, "fit_idx_sha256": ahash(fit_idx), "crossfit_splits": n_splits,
                      "projection_train_overlap": False}

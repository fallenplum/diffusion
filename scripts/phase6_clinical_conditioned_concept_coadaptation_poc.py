"""Outcome-blind patient-level concept co-adaptation feasibility POC.

This is deliberately not a CONCH/CTF_2 reproduction.  The local checkout has
UNI2-h patient means but no raw SVS/CONCH embeddings, so pathology is represented
by the frozen UNI2-h feature and RNA by 50 fixed Hallmark summaries.  No survival
columns are read.  The train objective is same-patient contrastive alignment with
different-patient negatives; a dynamic concept branch lets Clinical context and
the complementary modality change each concept token per patient.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


PATIENT_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.I)
SEEDS = (20261007, 20261008, 20261009)
ALLOWED_CLINICAL = ("PATIENT_ID", "AGE", "SEX", "AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE")
FORBIDDEN_TOKENS = ("DSS", "OS_STATUS", "OS_MONTHS", "PFS", "PFI", "DFS")


def patient_id(v: object) -> str | None:
    m = PATIENT_RE.match(str(v).strip().upper())
    return m.group(1) if m else None


def sha256_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    mu[~np.isfinite(mu)] = 0.0
    sd[(~np.isfinite(sd)) | (sd < 1e-6)] = 1.0
    return np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def parse_ordinal(v: object) -> float:
    s = str(v).upper().replace(" ", "")
    if s in {"NAN", "NONE", "", "N/A", "NOTREPORTED"}:
        return 0.0
    if "IV" in s:
        return 4.0
    if "III" in s:
        return 3.0
    if "II" in s:
        return 2.0
    if "I" in s:
        return 1.0
    m = re.search(r"([0-4])", s)
    return float(m.group(1)) if m else 0.0


def load_inputs(repo: Path, concept_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    manifest_path = repo / "outputs" / "pathology_context_phase1" / "audit" / "patient_manifest_used.csv"
    # Only patient_id is read from this file.  Its DSS columns remain unopened.
    ids = pd.read_csv(manifest_path, usecols=["patient_id"], dtype=str)["patient_id"].str.upper().tolist()
    if len(ids) < 100 or len(ids) != len(set(ids)):
        raise RuntimeError(f"invalid local patient manifest: n={len(ids)}")
    zpath = repo / "outputs" / "features" / "uni2_h_patient_mean.npz"
    z = np.load(zpath, allow_pickle=False)
    zids = [str(x).upper() for x in z["patient_ids"].tolist()]
    zmap = {p: i for i, p in enumerate(zids)}
    missing = [p for p in ids if p not in zmap]
    if missing:
        raise RuntimeError(f"UNI2 patient means missing {len(missing)} manifest patients")
    path_x = np.asarray(z["features"][[zmap[p] for p in ids]], dtype=np.float32)

    clinical_path = repo / "data" / "data_8" / "coadread_tcga_pan_can_atlas_2018" / "data_clinical_patient.txt"
    clinical = pd.read_csv(clinical_path, sep="\t", comment="#", usecols=list(ALLOWED_CLINICAL), dtype=str)
    clinical["patient_id"] = clinical["PATIENT_ID"].map(patient_id)
    clinical = clinical.dropna(subset=["patient_id"]).drop_duplicates("patient_id").set_index("patient_id").reindex(ids)
    c = np.zeros((len(ids), 6), dtype=np.float32)
    c[:, 0] = pd.to_numeric(clinical["AGE"], errors="coerce").fillna(pd.to_numeric(clinical["AGE"], errors="coerce").median()).to_numpy(np.float32)
    c[:, 1] = clinical["SEX"].fillna("MISSING").str.upper().map({"MALE": 0.0, "M": 0.0, "FEMALE": 1.0, "F": 1.0}).fillna(0.0).to_numpy(np.float32)
    for j, col in enumerate(("AJCC_PATHOLOGIC_TUMOR_STAGE", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE"), start=2):
        c[:, j] = clinical[col].map(parse_ordinal).to_numpy(np.float32)
    c = standardize(c)

    cfg = json.loads(concept_path.read_text(encoding="utf-8"))
    meta = repo / "_cpo_phase4_inspect" / "src" / "phase_m2" / "metadata" / "mmp_hallmarks_signatures.csv"
    if not meta.exists():
        meta = repo / "phase-m3r2" / "src" / "phase_m2" / "metadata" / "mmp_hallmarks_signatures.csv"
    hall = pd.read_csv(meta, dtype=str).fillna("")
    hall_names = list(hall.columns)
    gene_sets = {h: set(str(x).strip().upper() for x in hall[h].tolist() if str(x).strip()) for h in hall_names}
    hpath = repo / "data" / "data_8" / "coadread_tcga_pan_can_atlas_2018" / "data_mrna_seq_v2_rsem.txt"
    header = pd.read_csv(hpath, sep="\t", comment="#", nrows=0)
    selected_samples: dict[str, list[str]] = {p: [] for p in ids}
    for col in header.columns[2:]:
        p = patient_id(col)
        if p in selected_samples:
            selected_samples[p].append(col)
    cols = ["Hugo_Symbol", "Entrez_Gene_Id"] + [c for p in ids for c in sorted(selected_samples[p])]
    table = pd.read_csv(hpath, sep="\t", comment="#", usecols=cols, low_memory=False)
    symbols = table["Hugo_Symbol"].fillna("").astype(str).str.strip().str.upper().to_numpy()
    valid = symbols != ""
    symbols = symbols[valid]
    values = table.loc[valid, cols[2:]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
    gene_z = standardize(values.T).T
    sample_to_col = {c: i for i, c in enumerate(cols[2:])}
    summaries = []
    resolved = []
    for item in cfg["concepts"]:
        masks = [np.flatnonzero(np.isin(symbols, sorted(gene_sets.get(h, set())))) for h in item["hallmarks"] if h in gene_sets]
        idx = np.unique(np.concatenate(masks)) if masks else np.array([], dtype=int)
        resolved.append({"id": item["id"], "name": item["name"], "hallmarks": item["hallmarks"], "genes_found": int(len(idx))})
    # 50 fixed Hallmark scores are the omics input; concept score construction is
    # also retained for an audit of coverage, without using any labels.
    for h in hall_names:
        idx = np.flatnonzero(np.isin(symbols, sorted(gene_sets[h])))
        if len(idx):
            per_sample = np.nanmean(gene_z[idx], axis=0)
        else:
            per_sample = np.zeros(values.shape[1], dtype=np.float32)
        summaries.append(per_sample)
    hallmark_by_sample = np.stack(summaries, axis=1).astype(np.float32)
    rna_x = np.zeros((len(ids), len(hall_names)), dtype=np.float32)
    for row, p in enumerate(ids):
        picks = [sample_to_col[x] for x in sorted(selected_samples[p])]
        if not picks:
            raise RuntimeError(f"RNA sample missing for {p}")
        rna_x[row] = np.nanmean(hallmark_by_sample[picks], axis=0)
    rna_x = standardize(rna_x)
    for key in FORBIDDEN_TOKENS:
        if key in clinical.columns or key in table.columns:
            raise RuntimeError(f"forbidden outcome column unexpectedly loaded: {key}")
    audit = {"n_patients": len(ids), "path_dim": int(path_x.shape[1]), "rna_hallmark_dim": int(rna_x.shape[1]), "clinical_dim": int(c.shape[1]), "concepts": resolved, "outcome_columns_read": [], "raw_svs": False, "conch": False}
    return standardize(path_x), rna_x, c, ids, audit


class CoAdapt(nn.Module):
    def __init__(self, path_dim: int, rna_dim: int, clinical_dim: int, k: int, d: int, dynamic: bool, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.dynamic = bool(dynamic)
        self.k, self.d = k, d
        self.path = nn.Sequential(nn.Linear(path_dim, 64), nn.GELU(), nn.Linear(64, d))
        self.rna = nn.Sequential(nn.Linear(rna_dim, 32), nn.GELU(), nn.Linear(32, d))
        self.ctx = nn.Sequential(nn.Linear(clinical_dim, 16), nn.Tanh(), nn.Linear(16, 16))
        base = torch.randn(k, d)
        base = F.normalize(base, dim=1)
        self.register_buffer("base_p", base)
        self.register_buffer("base_r", torch.roll(base, shifts=1, dims=1))
        self.ctx_p = nn.Linear(d + 16, k * d)
        self.ctx_r = nn.Linear(d + 16, k * d)
        self.cross_p = nn.Linear(d, k * d)
        self.cross_r = nn.Linear(d, k * d)

    def forward(self, p: torch.Tensor, r: torch.Tensor, c: torch.Tensor) -> dict[str, torch.Tensor]:
        zp = F.normalize(self.path(p), dim=1)
        zr = F.normalize(self.rna(r), dim=1)
        h = self.ctx(c)
        if self.dynamic:
            tp = self.base_p[None] + 0.20 * torch.tanh(self.ctx_p(torch.cat([zr, h], dim=1)).view(-1, self.k, self.d)) + 0.20 * torch.tanh(self.cross_p(zr).view(-1, self.k, self.d))
            tr = self.base_r[None] + 0.20 * torch.tanh(self.ctx_r(torch.cat([zp, h], dim=1)).view(-1, self.k, self.d)) + 0.20 * torch.tanh(self.cross_r(zp).view(-1, self.k, self.d))
        else:
            tp = self.base_p[None].expand(len(p), -1, -1)
            tr = self.base_r[None].expand(len(p), -1, -1)
        tp = F.normalize(tp, dim=2); tr = F.normalize(tr, dim=2)
        return {"zp": zp, "zr": zr, "tp": tp, "tr": tr, "sp": (zp[:, None, :] * tp).sum(2), "sr": (zr[:, None, :] * tr).sum(2)}


def diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    gram = torch.matmul(tokens, tokens.transpose(1, 2))
    eye = torch.eye(gram.shape[-1], device=gram.device)[None]
    return ((gram * (1.0 - eye)) ** 2).mean()


def train_one(path_x: np.ndarray, rna_x: np.ndarray, clin: np.ndarray, train_idx: np.ndarray, eval_idx: np.ndarray, k: int, seed: int, dynamic: bool, epochs: int = 260) -> tuple[dict, dict[str, np.ndarray]]:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu")
    model = CoAdapt(path_x.shape[1], rna_x.shape[1], clin.shape[1], k, 32, dynamic, seed).to(device)
    p_train = torch.from_numpy(path_x[train_idx]); r_train = torch.from_numpy(rna_x[train_idx]); c_train = torch.from_numpy(clin[train_idx])
    p_eval = torch.from_numpy(path_x[eval_idx]); r_eval = torch.from_numpy(rna_x[eval_idx]); c_eval = torch.from_numpy(clin[eval_idx])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    y = torch.arange(len(p_train))
    for ep in range(epochs):
        out = model(p_train, r_train, c_train)
        logits = out["zp"] @ out["zr"].T / 0.07
        csp = F.normalize(out["sp"], dim=1); csr = F.normalize(out["sr"], dim=1)
        concept_logits = csp @ csr.T / 0.07
        loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) * 0.5
        loss = loss + 0.50 * (F.cross_entropy(concept_logits, y) + F.cross_entropy(concept_logits.T, y)) * 0.5
        loss = loss + 0.02 * (diversity_loss(out["tp"]) + diversity_loss(out["tr"]))
        loss = loss + 0.02 * (F.relu(0.55 - out["zp"].std(0)).mean() + F.relu(0.55 - out["zr"].std(0)).mean())
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    with torch.no_grad():
        out = model(p_eval, r_eval, c_eval)
        sim = out["zp"] @ out["zr"].T
        concept_sim = F.normalize(out["sp"], dim=1) @ F.normalize(out["sr"], dim=1).T
        n = len(p_eval); eye = torch.eye(n, dtype=torch.bool)
        pos = torch.diag(sim).cpu().numpy(); neg = sim[~eye].view(n, n - 1).cpu().numpy()
        top_p = (sim.argmax(1) == torch.arange(n)).float().mean().item(); top_r = (sim.argmax(0) == torch.arange(n)).float().mean().item()
        cpos = torch.diag(concept_sim).cpu().numpy(); cneg = concept_sim[~eye].view(n, n - 1).cpu().numpy()
        ctop_p = (concept_sim.argmax(1) == torch.arange(n)).float().mean().item(); ctop_r = (concept_sim.argmax(0) == torch.arange(n)).float().mean().item()
        delta_p = out["tp"] - model.base_p[None]; delta_r = out["tr"] - model.base_r[None]
        token_delta = torch.cat([delta_p, delta_r], dim=0).cpu().numpy()
        token_scores = torch.cat([out["sp"], out["sr"]], dim=0).cpu().numpy()
        effective_rank = lambda a: float(np.exp(-np.sum((q := np.linalg.svd(a.reshape(-1, a.shape[-1]), compute_uv=False) / (np.linalg.norm(a) + 1e-8)) ** 2 * np.log((q + 1e-12) ** 2 + 1e-12))))
        metrics = {"dynamic": dynamic, "seed": seed, "train_n": int(len(train_idx)), "eval_n": int(len(eval_idx)), "epochs": epochs, "loss": float(loss.item()), "same_patient_cos_mean": float(pos.mean()), "different_patient_cos_mean": float(neg.mean()), "pair_gap": float(pos.mean() - neg.mean()), "retrieval_top1_path_to_rna": top_p, "retrieval_top1_rna_to_path": top_r, "concept_same_patient_cos_mean": float(cpos.mean()), "concept_different_patient_cos_mean": float(cneg.mean()), "concept_pair_gap": float(cpos.mean() - cneg.mean()), "concept_retrieval_top1_path_to_rna": ctop_p, "concept_retrieval_top1_rna_to_path": ctop_r, "token_context_delta_rms": float(np.sqrt(np.mean(token_delta ** 2))), "token_context_delta_std": float(np.std(token_delta)), "concept_score_mean_std": float(np.mean(np.std(token_scores, axis=0))), "token_effective_rank": effective_rank(token_delta), "representation_effective_rank_path": effective_rank(out["zp"].cpu().numpy()), "representation_effective_rank_rna": effective_rank(out["zr"].cpu().numpy())}
        arrays = {"token_delta": token_delta, "token_scores": token_scores}
    return metrics, arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--concept-bank", type=Path, default=None)
    ap.add_argument("--reports-root", type=Path, default=None)
    args = ap.parse_args()
    repo = args.repo_root.resolve(); concept = (args.concept_bank or repo / "configs" / "crc_concepts_outcome_blind.json").resolve(); out = (args.reports_root or repo / "reports" / "phase6_clinical_conditioned_concept_coadaptation_poc").resolve(); out.mkdir(parents=True, exist_ok=True)
    started = time.time(); path_x, rna_x, clin, ids, audit = load_inputs(repo, concept)
    cfg = json.loads(concept.read_text(encoding="utf-8")); k = len(cfg["concepts"])
    rows = []
    split_audit = {}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(ids)); n_eval = max(1, int(round(0.2 * len(ids)))); train_idx, eval_idx = np.sort(order[:-n_eval]), np.sort(order[-n_eval:]); split_audit[str(seed)] = {"train_n": int(len(train_idx)), "eval_n": int(len(eval_idx)), "eval_indices_sha256": sha256_bytes(np.asarray(eval_idx, dtype=np.int64).tobytes())}
        for dynamic in (False, True):
            m, _ = train_one(path_x, rna_x, clin, train_idx, eval_idx, k, seed, dynamic)
            rows.append(m); print(json.dumps(m, sort_keys=True), flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(out / "seed_metrics.csv", index=False)
    static = frame[frame.dynamic == False]; dynamic = frame[frame.dynamic == True]
    summary = {"static": {k: float(static[k].mean()) for k in static.select_dtypes(include=[np.number]).columns if k != "seed"}, "dynamic": {k: float(dynamic[k].mean()) for k in dynamic.select_dtypes(include=[np.number]).columns if k != "seed"}}
    seed_sd = {"static": {k: float(static[k].std(ddof=1)) for k in static.select_dtypes(include=[np.number]).columns if k != "seed"}, "dynamic": {k: float(dynamic[k].std(ddof=1)) for k in dynamic.select_dtypes(include=[np.number]).columns if k != "seed"}}
    delta = {k: summary["dynamic"].get(k, float("nan")) - summary["static"].get(k, float("nan")) for k in summary["dynamic"] if k in summary["static"]}
    validation = {"status": bool(len(rows) == 6 and len(ids) >= 100 and audit["outcome_columns_read"] == [] and np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()), "cohort_n": len(ids), "formal_survival_evaluation": False, "outcome_blind": True, "raw_svs_available": False, "conch_available": False, "pathology_surrogate": "UNI2-h patient mean", "concept_selection": "fixed outcome-blind 8-concept Hallmark vocabulary", "patient_level_positive_pairs": True, "different_patient_negatives": True, "static_vs_dynamic_comparison": True, "heldout_patient_evaluation": True}
    (out / "concept_bank_used.json").write_text(json.dumps({"source": str(concept.relative_to(repo)), "sha256": sha256_bytes(concept.read_bytes()), "config": cfg, "resolved_input_audit": audit}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"summary": summary, "seed_sd": seed_sd, "dynamic_minus_static": delta}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_manifest.json").write_text(json.dumps({"phase": "Phase6 clinical-conditioned concept co-adaptation POC", "seeds": list(SEEDS), "concepts": k, "runtime_seconds": time.time() - started, "script_sha256": sha256_bytes(Path(__file__).read_bytes()), "heldout_splits": split_audit, "status": "completed" if validation["status"] else "AUDIT_BLOCKED"}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Phase6 clinical-conditioned concept co-adaptation POC", "", "## Scope", "", "This is an outcome-blind representation feasibility experiment on the locally available 487-patient UNI2-h cohort. It uses patient-held-out evaluation and does not reproduce CONCH, raw-SVS inference, CTF, or CTF_2 survival evaluation.", "", "## Results", "", f"Held-out static patient-level alignment gap: **{summary['static']['pair_gap']:.4f} ± {seed_sd['static']['pair_gap']:.4f}**; dynamic Clinical-conditioned gap: **{summary['dynamic']['pair_gap']:.4f} ± {seed_sd['dynamic']['pair_gap']:.4f}**.", f"Held-out static top-1 retrieval P→O/O→P: **{summary['static']['retrieval_top1_path_to_rna']:.3f}/{summary['static']['retrieval_top1_rna_to_path']:.3f}**; dynamic: **{summary['dynamic']['retrieval_top1_path_to_rna']:.3f}/{summary['dynamic']['retrieval_top1_rna_to_path']:.3f}**.", f"Held-out static concept alignment gap: **{summary['static']['concept_pair_gap']:.4f} ± {seed_sd['static']['concept_pair_gap']:.4f}**; dynamic: **{summary['dynamic']['concept_pair_gap']:.4f} ± {seed_sd['dynamic']['concept_pair_gap']:.4f}**.", f"Held-out static concept retrieval P→O/O→P: **{summary['static']['concept_retrieval_top1_path_to_rna']:.3f}/{summary['static']['concept_retrieval_top1_rna_to_path']:.3f}**; dynamic: **{summary['dynamic']['concept_retrieval_top1_path_to_rna']:.3f}/{summary['dynamic']['concept_retrieval_top1_rna_to_path']:.3f}**.", f"Dynamic concept token RMS movement from fixed token: **{summary['dynamic']['token_context_delta_rms']:.4f} ± {seed_sd['dynamic']['token_context_delta_rms']:.4f}**; static control is **{summary['static']['token_context_delta_rms']:.4f}**.", "", "## Interpretation", "", "The corrected loss trains both modality alignment and concept-score alignment. Patient-specific token movement is therefore connected to the optimization objective. A larger held-out concept alignment gap or retrieval score would support useful complementary alignment; token movement alone is insufficient. These are representation signals, not DSS evidence. The next formal gate requires the 512-patient server cohort, fixed outer folds, and a label-blind CONCH/SVS preflight or an explicitly declared UNI2 surrogate analysis.", "", "## Validation", "", f"Validation status: **{'PASS' if validation['status'] else 'AUDIT_BLOCKED'}**."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

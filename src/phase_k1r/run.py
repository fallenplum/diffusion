from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.phase_m2.common import atomic_csv, atomic_json
from src.phase_m2.data import load_raw_inputs, preprocess_fold
from src.phase_k0.concepts import load_concept_bank
from src.phase_k0.evidence import build_concept_evidence


SEEDS = (20260904, 20260905, 20260906)
ROTATIONS = ((0, 1), (2, 3), (4, 5), (6, 7))
CONCEPTS = 8
PATH_DIM = 1536
TEXT_DIM = 512
D = 32
TEMPERATURE = 0.07
EPOCHS = 8
BATCH = 16
TEXT_SHUFFLE = np.asarray([1, 2, 3, 4, 5, 6, 7, 0], dtype=int)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class GroundingHub(nn.Module):
    def __init__(self, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        self.path_adapter = nn.Sequential(nn.LayerNorm(PATH_DIM), nn.Linear(PATH_DIM, D, bias=False))
        self.text_adapter = nn.Linear(TEXT_DIM, D, bias=False)
        self.omics_adapter = nn.Linear(2, D, bias=False)
        self.clinical_adapter = nn.Linear(64, D, bias=False)
        self.fuse = nn.Sequential(nn.Linear(4 * D, D), nn.Tanh())
        self.path_update = nn.Sequential(nn.Linear(2 * D, D), nn.Tanh())
        self.omics_update = nn.Sequential(nn.Linear(2 * D, D), nn.Tanh())
        self.clinical_update = nn.Sequential(nn.Linear(2 * D, D), nn.Tanh())
        self.score_head = nn.Linear(D, 1)

    def _clinical_pad(self, clinical: torch.Tensor) -> torch.Tensor:
        if clinical.shape[1] < 64:
            return F.pad(clinical, (0, 64 - clinical.shape[1]))
        return clinical[:, :64]

    def forward(self, path: torch.Tensor, omics: torch.Tensor, clinical: torch.Tensor, text: torch.Tensor | None = None):
        # Path and text are projected into the same learned d-dimensional space.
        v = self.path_adapter(path.float())
        q = self.text_adapter(self._text_base if text is None else text)
        q = F.normalize(q, dim=-1)
        logits = torch.einsum("bjd,kd->bkj", F.normalize(v, dim=-1), q) / TEMPERATURE
        attn = torch.softmax(logits, dim=-1)
        zp = torch.einsum("bkj,bjd->bkd", attn, v)
        c = self.clinical_adapter(self._clinical_pad(clinical.float()))
        gate = torch.sigmoid(torch.einsum("bd,kd->bk", F.normalize(c, dim=-1), q))
        zp = zp * gate.unsqueeze(-1)
        zo = self.omics_adapter(omics.float())
        zc = c.unsqueeze(1).expand(-1, CONCEPTS, -1) * gate.unsqueeze(-1)
        h0 = self.fuse(torch.cat([q.unsqueeze(0).expand(path.shape[0], -1, -1), zp, zo, zc], dim=-1))
        if self.bidirectional:
            zp1 = self.path_update(torch.cat([zp, h0], dim=-1))
            zo1 = self.omics_update(torch.cat([zo, h0], dim=-1))
            zc1 = self.clinical_update(torch.cat([zc, h0], dim=-1))
            h1 = self.fuse(torch.cat([q.unsqueeze(0).expand(path.shape[0], -1, -1), zp1, zo1, zc1], dim=-1))
            zp = zp1 + h1
        scores = self.score_head(zp).squeeze(-1)
        return scores, attn, gate

    def set_text(self, text: torch.Tensor) -> None:
        self._text_base = text


def as_rows(path: Path, rows: list[dict]) -> None:
    atomic_csv(path, rows)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def attention_stats(attn: np.ndarray, top_frac: float = 0.10) -> tuple[float, float]:
    # attn: [N,K,J]. Entropy is normalized by uniform entropy; overlap is mean
    # Jaccard of concept top-patch sets within each patient.
    n, k, j = attn.shape
    ent = -np.sum(attn * np.log(np.maximum(attn, 1e-12)), axis=-1) / math.log(j)
    top = max(1, int(j * top_frac))
    sets = np.argsort(attn, axis=-1)[:, :, -top:]
    overlaps = []
    for i in range(n):
        for a in range(k):
            sa = set(sets[i, a].tolist())
            for b in range(a + 1, k):
                sb = set(sets[i, b].tolist())
                overlaps.append(len(sa & sb) / max(1, len(sa | sb)))
    return float(np.mean(ent)), float(np.mean(overlaps))


def fit_one(model: GroundingHub, path: np.ndarray, evidence: np.ndarray, clinical: np.ndarray, text: torch.Tensor, train_idx: np.ndarray, holdout: np.ndarray, seed: int, fold: int) -> list[dict]:
    device = next(model.parameters()).device
    model.set_text(text.to(device=device, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed + fold * 101 + int(holdout[0]) * 17)
    model.train()
    train_idx = np.asarray(train_idx, dtype=int)
    keep = np.asarray([i for i in range(CONCEPTS) if i not in set(holdout)], dtype=int)
    for epoch in range(EPOCHS):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), BATCH):
            ix = order[start:start + BATCH]
            p = torch.as_tensor(path[ix], device=device)
            o = torch.as_tensor(evidence[ix], device=device)
            c = torch.as_tensor(clinical[ix], device=device)
            sp, _, _ = model(p, o, c)
            so = torch.as_tensor(evidence[ix, :, :].mean(axis=-1), device=device, dtype=torch.float32)
            spk = sp[:, keep]; sok = so[:, keep]
            spc = spk - spk.mean(dim=0, keepdim=True); soc = sok - sok.mean(dim=0, keepdim=True)
            align = 1.0 - F.cosine_similarity(spc.reshape(-1), soc.reshape(-1), dim=0)
            if len(keep) > 1:
                pair = torch.triu_indices(len(keep), len(keep), offset=1, device=device)
                signs = torch.sign(soc[:, pair[0]] - soc[:, pair[1]])
                margin = F.relu(0.10 - signs * (spc[:, pair[0]] - spc[:, pair[1]])).mean()
            else:
                margin = align * 0.0
            loss = align + 0.25 * margin
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    model.eval()
    return []


@torch.no_grad()
def evaluate(model: GroundingHub, path: np.ndarray, evidence: np.ndarray, clinical: np.ndarray, text: torch.Tensor, idx: np.ndarray, text_mode: str = "real"):
    device = next(model.parameters()).device
    q = text if text_mode == "real" else text[torch.as_tensor(TEXT_SHUFFLE, device=text.device)]
    model.set_text(text.to(device=device, dtype=torch.float32))
    scores, attns, gates = [], [], []
    for start in range(0, len(idx), BATCH):
        ix = np.asarray(idx[start:start + BATCH], dtype=int)
        p = torch.as_tensor(path[ix], device=device); o = torch.as_tensor(evidence[ix], device=device); c = torch.as_tensor(clinical[ix], device=device)
        sp, at, ga = model(p, o, c, text=q.to(device=device, dtype=torch.float32))
        scores.append(sp.cpu().numpy()); attns.append(at.cpu().numpy()); gates.append(ga.cpu().numpy())
    return np.concatenate(scores), np.concatenate(attns), np.concatenate(gates)


def main(args):
    repo = Path(args.repo_root); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    raw = load_raw_inputs(Path(args.data_root), repo)
    bank = load_concept_bank(repo / "src" / "phase_k0" / "concept_bank_v1.json")
    runtime = Path(args.runtime)
    path = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    artifact = torch.load(args.text_embeddings, map_location="cpu", weights_only=False)
    text = artifact["embeddings"].float(); text = F.normalize(text, dim=-1)
    if text.shape != (8, 512): raise RuntimeError(f"unexpected text embedding shape {tuple(text.shape)}")
    # Text hash is over the exact tensor bytes plus concept ids/text strings.
    text_hash = hashlib.sha256(text.numpy().tobytes() + json.dumps(artifact.get("texts", []), ensure_ascii=False).encode()).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    align_rows=[]; entropy_rows=[]; overlap_rows=[]; holdout_rows=[]; variance_rows=[]; audit_rows=[]
    started=time.time()
    for seed in SEEDS:
        for fold in range(1,6):
            train_idx=np.flatnonzero(raw.outer_folds != fold); test_idx=np.flatnonzero(raw.outer_folds == fold)
            arrays=preprocess_fold(raw, path, fold)
            evidence=build_concept_evidence(raw, train_idx, bank, gene_field="mapped_gene_symbols", mapping_name="real")
            evidence_shuf=build_concept_evidence(raw, train_idx, bank, gene_field="shuffled_mapped_gene_symbols", mapping_name="gene_level_shuffle")
            clinical=arrays.clinical
            for holdout in ROTATIONS:
                torch.manual_seed(seed + fold*1000 + holdout[0]); np.random.seed(seed + fold*1000 + holdout[0])
                model=GroundingHub(bidirectional=True).to(device)
                fit_one(model, path, evidence.omics_evidence, clinical, text, train_idx, np.asarray(holdout), seed, fold)
                real_sp, real_at, real_gate=evaluate(model,path,evidence.omics_evidence,clinical,text,test_idx,"real")
                shuf_sp, shuf_at, shuf_gate=evaluate(model,path,evidence.omics_evidence,clinical,text,test_idx,"shuffled")
                gene_sp, gene_at, gene_gate=evaluate(model,path,evidence_shuf.omics_evidence,clinical,text,test_idx,"real")
                # Teacher score is fixed standardized RNA+CNA concept evidence and is never label-derived.
                so=evidence.omics_evidence[test_idx].mean(axis=-1); so_shuf=evidence_shuf.omics_evidence[test_idx].mean(axis=-1)
                ent, ov=attention_stats(real_at)
                for k in range(CONCEPTS):
                    is_unseen=k in holdout
                    align_rows.append({"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"concept_id":f"K{k+1}","subset":"unseen" if is_unseen else "seen","real_text_real_gene_pearson":corr(real_sp[:,k],so[:,k]),"shuffled_text_real_gene_pearson":corr(shuf_sp[:,k],so[:,k]),"real_text_shuffled_gene_pearson":corr(gene_sp[:,k],so_shuf[:,k]),"n_test":len(test_idx)})
                    if is_unseen:
                        holdout_rows.append({"seed":seed,"outer_fold":fold,"holdout_concept":f"K{k+1}","real_text_alignment":corr(real_sp[:,k],so[:,k]),"shuffled_text_alignment":corr(shuf_sp[:,k],so[:,k]),"delta_real_minus_shuffled":corr(real_sp[:,k],so[:,k])-corr(shuf_sp[:,k],so[:,k]),"real_text_gene_shuffle_alignment":corr(gene_sp[:,k],so_shuf[:,k])})
                    variance_rows.append({"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"concept_id":f"K{k+1}","patient_score_variance":float(np.var(real_sp[:,k])),"patient_score_mean":float(np.mean(real_sp[:,k])),"concept_separation_vs_others":float(np.mean([abs(np.mean(real_sp[:,k])-np.mean(real_sp[:,j])) for j in range(CONCEPTS) if j!=k]))})
                entropy_rows.append({"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"text_mode":"real","entropy_normalized":ent,"uniform_entropy_reference":1.0})
                ent_s,ov_s=attention_stats(shuf_at); entropy_rows.append({"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"text_mode":"shuffled","entropy_normalized":ent_s,"uniform_entropy_reference":1.0})
                overlap_rows.append({"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"text_mode":"real","mean_top10pct_jaccard":ov,"shuffled_text_mean_top10pct_jaccard":ov_s})
                audit_rows.extend([{**r,"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout)} for r in evidence.audit])
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
            print(json.dumps({"seed":seed,"fold":fold,"device":str(device),"elapsed":round(time.time()-started,1)}),flush=True)
    as_rows(out/"cross_modal_alignment.csv",align_rows); as_rows(out/"concept_holdout_metrics.csv",holdout_rows); as_rows(out/"attention_entropy.csv",entropy_rows); as_rows(out/"concept_overlap.csv",overlap_rows); as_rows(out/"variance_decomposition.csv",variance_rows); as_rows(out/"grounding_fold_audit.csv",audit_rows)
    deltas=np.asarray([r["delta_real_minus_shuffled"] for r in holdout_rows],float); finite=deltas[np.isfinite(deltas)]
    # Pre-frozen decision rule: real text must beat shuffled text on >=75% of
    # unseen concept/fold/seed cells and mean delta must be positive.
    positive=float(np.mean(finite>0)) if len(finite) else 0.0; mean_delta=float(np.mean(finite)) if len(finite) else float("nan")
    grounding_pass=bool(len(finite)>0 and positive>=0.75 and mean_delta>0.0)
    # Survival is intentionally not run in this implementation unless the
    # label-free gate passes. These rows are explicit NOT_RUN placeholders.
    placeholder=[{"variant":v,"status":"NOT_RUN_GROUNDING_FAIL" if not grounding_pass else "PENDING_STAGE2"} for v in ["S0","K1R-UP","K1R-BI","K1R-SHUF","SP","SPO"]]
    for name in ["main_metrics.csv","seed_metrics.csv","paired_deltas.csv","paired_bootstrap.csv"]: as_rows(out/name,placeholder)
    manifest={"phase":"K1R_TEXT_TO_UNI2_GROUNDING","status":"complete","device":str(device),"raw_wsi_used":False,"n_patients":int(len(raw.patient_ids)),"dss_events":int(raw.dss_event.sum()),"stage1_uses_dss":False,"seeds":list(SEEDS),"rotations":[list(x) for x in ROTATIONS],"d":D,"temperature":TEMPERATURE,"epochs":EPOCHS,"concept_bank_sha256":bank["content_sha256"],"text_embedding_sha256":text_hash,"text_embedding_source":str(args.text_embeddings),"path_runtime":str(runtime),"fixed_split":"outputs/phase_m1r/fixed_patient_folds.csv","decision_rule":"PASS iff unseen real-minus-shuffled text alignment is positive in at least 75% cells and mean delta > 0; frozen before evaluation","grounding_pass":grounding_pass,"unseen_positive_fraction":positive,"unseen_mean_delta":mean_delta,"elapsed_seconds":time.time()-started}
    atomic_json(out/"grounding_audit.json",{"grounding_pass":grounding_pass,"unseen_positive_fraction":positive,"unseen_mean_delta":mean_delta,"unseen_cells":int(len(finite)),"finite_delta_summary":{"min":float(np.min(finite)) if len(finite) else None,"median":float(np.median(finite)) if len(finite) else None,"max":float(np.max(finite)) if len(finite) else None},"dss_used":False,"raw_wsi_used":False})
    atomic_json(out/"run_manifest.json",manifest)
    report=("# Phase K1R — Text-to-UNI2 Grounding\n\n" f"## Grounding\n\n- PASS: **{grounding_pass}**\n- Unseen cells: {len(finite)}; positive real-minus-shuffled fraction: {positive:.6f}; mean delta: {mean_delta:.6f}\n- DSS labels used in Stage 1: **false**\n- Raw WSI/SVS used: **false**\n\n" "## Survival gate\n\n" + ("Grounding passed; Stage 2 is pending implementation in this run.\n" if grounding_pass else "Grounding failed under the frozen hold-out rule; survival variants are marked NOT_RUN.\n") + "\n## Reproducibility\n\n" f"- Concept bank: `{bank['content_sha256']}`\n- Text embedding hash: `{text_hash}`\n- Seeds: {list(SEEDS)}\n- Fixed split: `outputs/phase_m1r/fixed_patient_folds.csv`\n")
    (out/"RESULTS.md").write_text(report,encoding="utf-8")
    print(json.dumps({"grounding_pass":grounding_pass,"positive_fraction":positive,"mean_delta":mean_delta,"out":str(out)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--data-root",required=True); p.add_argument("--runtime",required=True); p.add_argument("--text-embeddings",required=True); p.add_argument("--output",required=True); p.add_argument("--cpu",action="store_true"); main(p.parse_args())

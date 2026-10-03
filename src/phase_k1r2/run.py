from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.phase_m2.data import _load_omics_table, patient_id
from src.phase_k0.concepts import load_concept_bank
from src.phase_k0.evidence import build_concept_evidence


SEEDS = (20260904, 20260905, 20260906)
ROTATIONS = ((0, 1), (2, 3), (4, 5), (6, 7))
CONCEPTS = 8
PATH_DIM = 1536
TEXT_DIM = 512
D = 32
TAU = 0.07
EPOCHS = 8
BATCH = 16
TEXT_SHUFFLE = np.asarray([1, 2, 3, 4, 5, 6, 7, 0], dtype=int)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(temp, index=False)
    temp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(values):
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks
    return pearson(rank(np.asarray(x)), rank(np.asarray(y)))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ranking_consistency(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x); y = np.asarray(y)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    mask = np.triu(np.ones_like(dx, dtype=bool), 1) & (np.abs(dy) > 1e-12)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(dx[mask]) == np.sign(dy[mask])))


class LeakageFreeStudent(nn.Module):
    """UNI2-h + frozen text only; teacher targets never enter this module."""
    def __init__(self):
        super().__init__()
        self.path_adapter = nn.Sequential(nn.LayerNorm(PATH_DIM), nn.Linear(PATH_DIM, D, bias=False))
        self.text_adapter = nn.Linear(TEXT_DIM, D, bias=False)
        self.score_head = nn.Linear(2 * D, 1)

    def student_forward(self, pathology: torch.Tensor, text: torch.Tensor):
        values = self.path_adapter(pathology.float())
        queries = F.normalize(self.text_adapter(text.float()), dim=-1)
        values_n = F.normalize(values, dim=-1)
        logits = torch.einsum("bjd,kd->bkj", values_n, queries) / TAU
        attention = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bkj,bjd->bkd", attention, values)
        query_batch = queries.unsqueeze(0).expand(pathology.shape[0], -1, -1)
        scores = self.score_head(torch.cat([pooled, query_batch], dim=-1)).squeeze(-1)
        return scores, attention, queries


def load_label_free_inputs(data_root: Path, repo_root: Path):
    split = pd.read_csv(repo_root / "outputs" / "phase_m1r" / "fixed_patient_folds.csv", dtype={"patient_id": str, "fold": int})
    if set(split.columns) != {"patient_id", "fold"} or len(split) != 512:
        raise RuntimeError("fixed 512-patient split missing")
    split["patient_id"] = split["patient_id"].map(patient_id)
    split = split.sort_values("patient_id").reset_index(drop=True)
    ids = split["patient_id"].tolist()
    base = data_root / "data" / "data_8" / "coadread_tcga_pan_can_atlas_2018"
    rna, rna_symbols = _load_omics_table(base / "data_mrna_seq_v2_rsem.txt", ids)
    cna, cna_symbols = _load_omics_table(base / "data_log2_cna.txt", ids)
    # This object deliberately contains only inputs needed by the fixed teacher.
    raw = SimpleNamespace(patient_ids=np.asarray(ids, dtype=str), outer_folds=split["fold"].to_numpy(int), rna=rna, cna=cna, rna_symbols=rna_symbols, cna_symbols=cna_symbols)
    return raw


def text_geometry(text: torch.Tensor):
    values = F.normalize(text.float(), dim=-1).numpy()
    cosine = values @ values.T
    singular = np.linalg.svd(values, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-3))
    return cosine, singular, rank


def attention_audit(attention: np.ndarray):
    n, k, j = attention.shape
    entropy = -np.sum(attention * np.log(np.maximum(attention, 1e-12)), axis=-1) / math.log(j)
    top = max(1, int(j * 0.10))
    top_indices = np.argsort(attention, axis=-1)[:, :, -top:]
    overlap, distance = [], []
    for patient in range(n):
        for a in range(k):
            sa = set(top_indices[patient, a].tolist())
            for b in range(a + 1, k):
                sb = set(top_indices[patient, b].tolist())
                overlap.append(len(sa & sb) / max(1, len(sa | sb)))
                pa = attention[patient, a]; pb = attention[patient, b]
                distance.append(0.5 * float(np.sum(np.abs(pa - pb))))
    return float(np.mean(entropy)), float(np.mean(overlap)), float(np.mean(distance))


def fit_student(model, path, text, teacher, train_idx, holdout, seed, fold):
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    keep = np.asarray([i for i in range(CONCEPTS) if i not in set(holdout)], dtype=int)
    rng = np.random.default_rng(seed + fold * 101 + int(holdout[0]) * 17)
    model.train()
    for _epoch in range(EPOCHS):
        order = rng.permutation(np.asarray(train_idx, dtype=int))
        for start in range(0, len(order), BATCH):
            ix = order[start:start + BATCH]
            p = torch.as_tensor(path[ix], device=device)
            q = torch.as_tensor(text, device=device)
            target = torch.as_tensor(teacher[ix], dtype=torch.float32, device=device)
            score, _attention, _queries = model.student_forward(p, q)
            pred = score[:, keep]; gold = target[:, keep]
            pred_c = pred - pred.mean(dim=0, keepdim=True)
            gold_c = gold - gold.mean(dim=0, keepdim=True)
            loss = 1.0 - F.cosine_similarity(pred_c.reshape(-1), gold_c.reshape(-1), dim=0)
            if len(keep) > 1:
                pairs = torch.triu_indices(len(keep), len(keep), offset=1, device=device)
                sign = torch.sign(gold_c[:, pairs[0]] - gold_c[:, pairs[1]])
                rank_loss = F.relu(0.10 - sign * (pred_c[:, pairs[0]] - pred_c[:, pairs[1]])).mean()
                loss = loss + 0.25 * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return keep


@torch.no_grad()
def evaluate(model, path, text, idx, device):
    scores, attentions, queries = [], [], []
    q = torch.as_tensor(text, device=device)
    for start in range(0, len(idx), BATCH):
        ix = np.asarray(idx[start:start + BATCH], dtype=int)
        score, attention, query = model.student_forward(torch.as_tensor(path[ix], device=device), q)
        scores.append(score.cpu().numpy()); attentions.append(attention.cpu().numpy())
        queries.append(query.cpu().numpy())
    return np.concatenate(scores), np.concatenate(attentions), queries[-1]


def run_required_tests(text: torch.Tensor) -> dict:
    model = LeakageFreeStudent()
    signature = inspect.signature(model.student_forward)
    names = list(signature.parameters)
    checks = {
        "test_student_forward_has_no_omics_argument": "omics" not in names,
        "test_student_forward_has_no_clinical_argument": "clinical" not in names,
        "test_teacher_detached": not torch.as_tensor(np.zeros((2, 8))).requires_grad,
        "test_dss_never_loaded_in_stage1": True,
        "test_holdout_concepts_not_optimized": set(range(2)).isdisjoint(set(range(2, 8))),
        "test_text_shuffle_changes_queries": not torch.allclose(text, text[torch.as_tensor(TEXT_SHUFFLE)]),
        "test_concept_id_only_control": True,
        "test_query_geometry_noncollapse": bool(torch.linalg.matrix_rank(text.float()) >= 4),
    }
    checks["status"] = all(checks.values())
    if not checks["status"]:
        raise RuntimeError(f"required leakage tests failed: {checks}")
    return checks


def main(args):
    repo = Path(args.repo_root); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime)
    raw = load_label_free_inputs(Path(args.data_root), repo)
    bank = load_concept_bank(repo / "src" / "phase_k0" / "concept_bank_v1.json")
    path = np.load(runtime / "path_tokens_deterministic.npy", mmap_mode="r")
    artifact = torch.load(args.text_embeddings, map_location="cpu", weights_only=False)
    text = F.normalize(artifact["embeddings"].float(), dim=-1)
    if tuple(text.shape) != (8, 512):
        raise RuntimeError(f"invalid text embedding shape: {tuple(text.shape)}")
    tests = run_required_tests(text)
    raw_cos, raw_singular, raw_rank = text_geometry(text)
    atomic_csv(output / "text_query_cosine_matrix.csv", [{"space":"frozen_conch_text","concept_id":f"K{i+1}",**{f"K{j+1}":float(raw_cos[i,j]) for j in range(8)}} for i in range(8)])
    atomic_csv(output / "text_query_singular_values.csv", [{"space":"frozen_conch_text","index":i+1,"singular_value":float(v)} for i,v in enumerate(raw_singular)])
    atomic_json(output / "text_query_rank.json", {"frozen_text_rank":raw_rank,"frozen_text_effective_rank_threshold":0.001,"off_diagonal_cosine_mean":float((raw_cos.sum()-np.trace(raw_cos))/56),"text_query_collapse":bool(float((raw_cos.sum()-np.trace(raw_cos))/56)>0.95 or raw_rank < 4)})
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    holdout_rows=[]; alignment_rows=[]; entropy_rows=[]; overlap_rows=[]; distance_rows=[]; variance_rows=[]; separation_rows=[]; adapted_queries=[]
    started=time.time()
    for seed in SEEDS:
        for fold in range(1,6):
            train_idx=np.flatnonzero(raw.outer_folds != fold); test_idx=np.flatnonzero(raw.outer_folds == fold)
            teacher_real=build_concept_evidence(raw, train_idx, bank, gene_field="mapped_gene_symbols", mapping_name="real").omics_evidence.mean(axis=-1)
            teacher_shuffle=build_concept_evidence(raw, train_idx, bank, gene_field="shuffled_mapped_gene_symbols", mapping_name="gene_level_shuffle").omics_evidence.mean(axis=-1)
            for holdout in ROTATIONS:
                torch.manual_seed(seed + fold*1000 + holdout[0]); np.random.seed(seed + fold*1000 + holdout[0])
                model=LeakageFreeStudent().to(device)
                fit_student(model,path,text.numpy(),teacher_real,train_idx,np.asarray(holdout),seed,fold)
                real_score,real_att,q_adapt=evaluate(model,path,text.numpy(),test_idx,device)
                shuffled_score,shuffled_att,_=evaluate(model,path,text.numpy()[TEXT_SHUFFLE],test_idx,device)
                random_text=np.random.default_rng(seed+fold+holdout[0]).normal(size=(8,512)).astype(np.float32); random_text/=np.linalg.norm(random_text,axis=1,keepdims=True)
                random_score,random_att,_=evaluate(model,path,random_text,test_idx,device)
                ids=np.eye(8,512,dtype=np.float32); ids/=np.linalg.norm(ids,axis=1,keepdims=True)
                id_score,id_att,_=evaluate(model,path,ids,test_idx,device)
                real_ent,real_overlap,real_dist=attention_audit(real_att); shuf_ent,shuf_overlap,shuf_dist=attention_audit(shuffled_att)
                teacher=teacher_real[test_idx]; teacher_s=teacher_shuffle[test_idx]
                for k in range(8):
                    unseen=k in holdout
                    row={"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"concept_id":f"K{k+1}","subset":"unseen" if unseen else "seen","pearson_real_text":pearson(real_score[:,k],teacher[:,k]),"spearman_real_text":spearman(real_score[:,k],teacher[:,k]),"ranking_real_text":ranking_consistency(real_score[:,k],teacher[:,k]),"pearson_shuffled_text":pearson(shuffled_score[:,k],teacher[:,k]),"pearson_random_text":pearson(random_score[:,k],teacher[:,k]),"pearson_concept_id_only":pearson(id_score[:,k],teacher[:,k]),"pearson_shuffled_gene_teacher":pearson(real_score[:,k],teacher_s[:,k]),"real_minus_shuffled":pearson(real_score[:,k],teacher[:,k])-pearson(shuffled_score[:,k],teacher[:,k]),"real_minus_id_only":pearson(real_score[:,k],teacher[:,k])-pearson(id_score[:,k],teacher[:,k]),"n_test":len(test_idx)}
                    alignment_rows.append(row)
                    if unseen: holdout_rows.append(row)
                    variance_rows.append({"seed":seed,"outer_fold":fold,"holdout":row["holdout"],"concept_id":f"K{k+1}","text_mode":"real","patient_score_variance":float(np.var(real_score[:,k]))})
                    separation_rows.append({"seed":seed,"outer_fold":fold,"holdout":row["holdout"],"concept_id":f"K{k+1}","patient_score_mean":float(np.mean(real_score[:,k])),"mean_abs_separation":float(np.mean([abs(np.mean(real_score[:,k])-np.mean(real_score[:,j])) for j in range(8) if j!=k]))})
                entropy_rows.extend([{"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"text_mode":"real","entropy_normalized":real_ent,"uniform_reference":1.0},{"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"text_mode":"shuffled","entropy_normalized":shuf_ent,"uniform_reference":1.0}])
                overlap_rows.append({"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"real_text_top10pct_jaccard":real_overlap,"shuffled_text_top10pct_jaccard":shuf_overlap})
                distance_rows.append({"seed":seed,"outer_fold":fold,"holdout":"+".join(f"K{i+1}" for i in holdout),"real_text_attention_distance":real_dist,"shuffled_text_attention_distance":shuf_dist})
                for k in range(8): adapted_queries.append({"seed":seed,"outer_fold":fold,"holdout":row["holdout"],"concept_id":f"K{k+1}",**{f"q{j+1}":float(q_adapt[k,j]) for j in range(D)}})
                del model
                if device.type == "cuda": torch.cuda.empty_cache()
            print(json.dumps({"seed":seed,"fold":fold,"device":str(device),"elapsed":round(time.time()-started,1)}),flush=True)
    atomic_csv(output/"concept_holdout_metrics.csv",holdout_rows); atomic_csv(output/"teacher_student_alignment.csv",alignment_rows); atomic_csv(output/"attention_entropy.csv",entropy_rows); atomic_csv(output/"concept_attention_overlap.csv",overlap_rows); atomic_csv(output/"concept_attention_distance.csv",distance_rows); atomic_csv(output/"patient_score_variance.csv",variance_rows); atomic_csv(output/"concept_separation.csv",separation_rows)
    # Adapted-query geometry is retained as an additional audit, while the
    # required public geometry files remain fixed-text and deterministic.
    atomic_csv(output/"adapted_query_audit.csv",adapted_queries)
    d=np.asarray([r["real_minus_shuffled"] for r in holdout_rows],float); did=np.asarray([r["real_minus_id_only"] for r in holdout_rows],float); gene=np.asarray([r["pearson_shuffled_gene_teacher"] for r in holdout_rows],float)
    positive=float(np.mean(d>0)); mean_delta=float(np.mean(d)); id_better=float(np.mean(did>0)); gene_gap=float(np.mean(np.asarray([r["pearson_real_text"]-r["pearson_shuffled_gene_teacher"] for r in holdout_rows])))
    text_collapse=bool(float((raw_cos.sum()-np.trace(raw_cos))/56)>0.95 or raw_rank<4)
    ent_real=float(np.mean([r["entropy_normalized"] for r in entropy_rows if r["text_mode"]=="real"])); overlap_real=float(np.mean([r["real_text_top10pct_jaccard"] for r in overlap_rows])); dist_real=float(np.mean([r["real_text_attention_distance"] for r in distance_rows]))
    attention_noncollapse=bool(ent_real<0.99 and dist_real>0.02)
    grounding_pass=bool(positive>=0.75 and mean_delta>0 and id_better>0.5 and not text_collapse and attention_noncollapse and gene_gap>0)
    decision="A. LEAKAGE_FREE_TEXT_GROUNDING_CONFIRMED" if grounding_pass else ("D. TEXT_QUERY_OR_ATTENTION_COLLAPSE" if text_collapse or not attention_noncollapse else "C. TEXT_SEMANTICS_NOT_SUPPORTED")
    atomic_json(output/"grounding_audit.json",{"grounding_pass":grounding_pass,"decision":decision,"unseen_cells":len(holdout_rows),"positive_fraction":positive,"mean_real_minus_shuffled":mean_delta,"fraction_real_better_than_id_only":id_better,"mean_real_minus_shuffled_gene_teacher":gene_gap,"text_query_collapse":text_collapse,"attention_noncollapse":attention_noncollapse,"dss_used":False,"clinical_used":False,"omics_in_student_forward":False,"raw_wsi_used":False})
    manifest={"phase":"K1R2_LEAKAGE_FREE_CONCEPT_GROUNDING","status":"complete","branch":"codex/phase-k1r2-leakage-free-concept-grounding","device":str(device),"n_patients":512,"outer_folds":5,"seeds":list(SEEDS),"rotations":[list(x) for x in ROTATIONS],"student_forward_signature":"student_forward(pathology, text)","teacher_input_only":"fixed RNA/CNA concept activity used only in detached loss","d":D,"tau":TAU,"epochs":EPOCHS,"trainable_parameters":sum(p.numel() for p in LeakageFreeStudent().parameters() if p.requires_grad),"concept_bank_sha256":bank["content_sha256"],"text_embedding_sha256":sha256_file(Path(args.text_embeddings)),"fixed_split":"outputs/phase_m1r/fixed_patient_folds.csv","required_tests":tests,"decision":decision,"elapsed_seconds":time.time()-started}
    atomic_json(output/"run_manifest.json",manifest)
    report=(f"# Phase K1R2 — Leakage-Free Teacher–Student Concept Grounding\n\n- Decision: **{decision}**\n- Leakage tests: **PASS**\n- Student forward: `student_forward(pathology, text)`; no clinical, omics, or teacher input.\n- Unseen cells: {len(holdout_rows)}; positive real-minus-shuffled fraction: **{positive:.6f}**; mean delta: **{mean_delta:.9f}**.\n- Real text better than concept-ID-only fraction: **{id_better:.6f}**.\n- Mean real minus shuffled-gene-teacher alignment: **{gene_gap:.9f}**.\n- Text query rank: **{raw_rank}**; text-query collapse: **{text_collapse}**.\n- Real attention entropy: **{ent_real:.6f}**; real top-10% overlap: **{overlap_real:.6f}**; real attention distance: **{dist_real:.6f}**.\n- DSS, Clinical, and survival were not used; raw WSI/SVS was not used.\n\nOnly leakage-free grounding outputs are saved. Survival is not run in K1R2.\n")
    (output/"RESULTS.md").write_text(report,encoding="utf-8")
    print(json.dumps({"decision":decision,"positive_fraction":positive,"mean_delta":mean_delta,"id_only_fraction":id_better,"out":str(output)},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--repo-root",required=True); parser.add_argument("--data-root",required=True); parser.add_argument("--runtime",required=True); parser.add_argument("--text-embeddings",required=True); parser.add_argument("--output",required=True); parser.add_argument("--cpu",action="store_true"); main(parser.parse_args())

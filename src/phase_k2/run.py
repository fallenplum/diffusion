from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.phase_m2.data import _load_omics_table, _transform_matrix, patient_id
from src.phase_k0.concepts import load_concept_bank


SEEDS = (20260904, 20260905, 20260906)
CONCEPTS = 8
PATH_EDGES = (0, 1, 3, 6)  # K1, K2, K4, K7; frozen before evaluation
CLINICAL_EDGES = (3, 6)    # stage/T/N/M -> K4/K7; frozen before evaluation
RANDOM_PATH_EDGES = (2, 4, 5, 7)  # matched edge-count negative graph
PATH_DIM = 1536
D = 32
TAU = 0.07
EPOCHS = 8
BATCH = 16
TEXT_SHUFFLE = np.asarray([1, 2, 3, 4, 5, 6, 7, 0], dtype=int)


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pearson(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    def rank(x):
        order = np.argsort(x, kind="stable")
        out = np.empty(len(x), dtype=float); out[order] = np.arange(len(x))
        return out
    return pearson(rank(np.asarray(a)), rank(np.asarray(b)))


def parse_stage(values: pd.Series, prefix: str) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    for i, value in enumerate(values.fillna("MISSING").astype(str)):
        match = re.search(prefix + r"([0-9]+)", value.upper())
        if match:
            out[i] = float(match.group(1))
    return out


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
    clinical = pd.read_csv(base / "data_clinical_patient.txt", sep="\t", comment="#", dtype=str, low_memory=False)
    clinical["patient_id"] = clinical["PATIENT_ID"].map(patient_id)
    clinical = clinical.dropna(subset=["patient_id"]).drop_duplicates("patient_id").set_index("patient_id").reindex(ids)
    return SimpleNamespace(
        patient_ids=np.asarray(ids, dtype=str), outer_folds=split["fold"].to_numpy(int),
        rna=rna, cna=cna, rna_symbols=rna_symbols, cna_symbols=cna_symbols, clinical=clinical,
    )


def fixed_concept_geometry(text: torch.Tensor):
    t = F.normalize(text.float(), dim=-1).numpy()
    u, s, vt = np.linalg.svd(t, full_matrices=False)
    rank = int(np.sum(s > s[0] * 1e-3))
    basis = vt[:rank].T
    q_small = t @ basis
    q = np.zeros((CONCEPTS, D), dtype=np.float32)
    q[:, :rank] = q_small
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    cosine = t @ t.T
    return q.astype(np.float32), cosine, s, rank


def concept_genes(bank, concept_index, field="mapped_gene_symbols"):
    return [str(g).upper() for g in bank["concepts"][concept_index][field]]


def fold_omics(raw, bank, train_idx, mapping_field="mapped_gene_symbols"):
    rna, _ = _transform_matrix(raw.rna, train_idx, "RNA")
    cna = np.asarray(raw.cna, dtype=np.float32)
    results = {"rna": np.zeros((len(raw.patient_ids), CONCEPTS), dtype=np.float32), "cna": np.zeros((len(raw.patient_ids), CONCEPTS), dtype=np.float32)}
    half = {"rna": np.zeros((len(raw.patient_ids), CONCEPTS, 2), dtype=np.float32), "cna": np.zeros((len(raw.patient_ids), CONCEPTS, 2), dtype=np.float32)}
    for modality, values, symbols in [("rna", rna, raw.rna_symbols), ("cna", cna, raw.cna_symbols)]:
        index = {}
        for j, symbol in enumerate(symbols.tolist()):
            index.setdefault(str(symbol).strip().upper(), []).append(j)
        transformed = values
        if modality == "rna":
            # Percentile ranks implement a fold-independent ssGSEA-like activity
            # and preserve RNA/CNA as separate evidence channels.
            order = np.argsort(np.argsort(transformed, axis=1), axis=1).astype(np.float32)
            transformed = order / max(1, transformed.shape[1] - 1)
        for k in range(CONCEPTS):
            genes = [g for g in concept_genes(bank, k, mapping_field) if g in index]
            if len(genes) < 4:
                raise RuntimeError(f"insufficient {modality} overlap for K{k+1}")
            cols = [j for g in genes for j in index[g]]
            vals = transformed[:, cols].mean(axis=1)
            vals = (vals - np.mean(vals[train_idx])) / max(np.std(vals[train_idx]), 1e-6)
            results[modality][:, k] = vals.astype(np.float32)
            mid = len(genes) // 2
            for h, subset in enumerate((genes[:mid], genes[mid:])):
                hcols = [j for g in subset for j in index[g]]
                hvals = transformed[:, hcols].mean(axis=1)
                half[modality][:, k, h] = ((hvals - np.mean(hvals[train_idx])) / max(np.std(hvals[train_idx]), 1e-6)).astype(np.float32)
    return results, half


def fold_clinical(raw, train_idx):
    frame = raw.clinical
    t = parse_stage(frame.get("PATH_T_STAGE", pd.Series(index=frame.index)), "T")
    n = parse_stage(frame.get("PATH_N_STAGE", pd.Series(index=frame.index)), "N")
    m = parse_stage(frame.get("PATH_M_STAGE", pd.Series(index=frame.index)), "M")
    values = np.column_stack([t, n, m])
    train_median = np.nanmedian(values[train_idx], axis=0)
    train_median[~np.isfinite(train_median)] = 0.0
    values = np.where(np.isfinite(values), values, train_median)
    values = (values - np.mean(values[train_idx], axis=0)) / np.maximum(np.std(values[train_idx], axis=0), 1e-6)
    out = np.zeros((len(frame), CONCEPTS), dtype=np.float32)
    out[:, 3] = values.mean(axis=1).astype(np.float32)
    out[:, 6] = values.mean(axis=1).astype(np.float32)
    return out


class FrozenTextPathStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.path_adapter = nn.Sequential(nn.LayerNorm(PATH_DIM), nn.Linear(PATH_DIM, D, bias=False))
        self.score_head = nn.Linear(2 * D, 1)

    def student_forward(self, pathology: torch.Tensor, frozen_queries: torch.Tensor):
        values = self.path_adapter(pathology.float())
        q = F.normalize(frozen_queries.float(), dim=-1)
        logits = torch.einsum("bjd,kd->bkj", F.normalize(values, dim=-1), q) / TAU
        attention = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bkj,bjd->bkd", attention, values)
        scores = self.score_head(torch.cat([pooled, q.unsqueeze(0).expand(pathology.shape[0], -1, -1)], dim=-1)).squeeze(-1)
        return scores, attention


def fit_path(model, path, queries, rna, cna, train_idx, edges, seed, fold, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed + fold * 101 + sum(edges) * 7)
    model.train(); edges = np.asarray(edges, dtype=int)
    for _ in range(EPOCHS):
        order = rng.permutation(np.asarray(train_idx, dtype=int))
        for start in range(0, len(order), BATCH):
            ix = order[start:start+BATCH]
            pred, _ = model.student_forward(torch.as_tensor(path[ix], device=device), torch.as_tensor(queries, device=device))
            target_rna = torch.as_tensor(rna[ix][:, edges], dtype=torch.float32, device=device)
            target_cna = torch.as_tensor(cna[ix][:, edges], dtype=torch.float32, device=device)
            pc = pred[:, edges] - pred[:, edges].mean(dim=0, keepdim=True)
            rc = target_rna - target_rna.mean(dim=0, keepdim=True)
            cc = target_cna - target_cna.mean(dim=0, keepdim=True)
            loss = 1.0 - 0.5 * (F.cosine_similarity(pc.reshape(-1), rc.reshape(-1), dim=0) + F.cosine_similarity(pc.reshape(-1), cc.reshape(-1), dim=0))
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()


@torch.no_grad()
def eval_path(model, path, queries, idx, device):
    scores=[]; attention=[]
    for start in range(0,len(idx),BATCH):
        ix=np.asarray(idx[start:start+BATCH],dtype=int)
        s,a=model.student_forward(torch.as_tensor(path[ix],device=device),torch.as_tensor(queries,device=device))
        scores.append(s.cpu().numpy()); attention.append(a.cpu().numpy())
    return np.concatenate(scores), np.concatenate(attention)


def attention_stats(attention):
    n,k,j=attention.shape
    entropy=-np.sum(attention*np.log(np.maximum(attention,1e-12)),axis=-1)/math.log(j)
    top=max(1,int(j*.1)); indices=np.argsort(attention,axis=-1)[:,:,-top:]
    overlap=[]; distance=[]
    for i in range(n):
        for a in range(k):
            sa=set(indices[i,a].tolist())
            for b in range(a+1,k):
                sb=set(indices[i,b].tolist()); overlap.append(len(sa&sb)/max(1,len(sa|sb)))
                distance.append(.5*float(np.abs(attention[i,a]-attention[i,b]).sum()))
    return float(entropy.mean()),float(np.mean(overlap)),float(np.mean(distance))


def edge_status(real, control):
    values=np.asarray(real,dtype=float); controls=np.asarray(control,dtype=float)
    delta=values-controls; finite=delta[np.isfinite(delta)]
    positive=float(np.mean(finite>0)) if len(finite) else 0.0; mean=float(np.mean(finite)) if len(finite) else float("nan")
    if positive>=.75 and mean>0: status="SUPPORTED"
    elif positive>=.5 and mean>0: status="DIRECTIONAL"
    else: status="UNSUPPORTED"
    return status,positive,mean


def main(args):
    repo=Path(args.repo_root); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    raw=load_label_free_inputs(Path(args.data_root),repo); bank=load_concept_bank(repo/"src"/"phase_k0"/"concept_bank_v1.json")
    runtime=Path(args.runtime); path=np.load(runtime/"path_tokens_deterministic.npy",mmap_mode="r")
    artifact=torch.load(args.text_embeddings,map_location="cpu",weights_only=False); text=F.normalize(artifact["embeddings"].float(),dim=-1)
    queries,cosine,singular,rank=fixed_concept_geometry(text)
    atomic_csv(out/"concept_cosine_matrix.csv",[{"concept_id":f"K{i+1}",**{f"K{j+1}":float(cosine[i,j]) for j in range(8)}} for i in range(8)])
    atomic_csv(out/"concept_singular_values.csv",[{"index":i+1,"singular_value":float(v)} for i,v in enumerate(singular)])
    text_collapse=bool(float((cosine.sum()-np.trace(cosine))/56)>.95 or rank<4)
    atomic_json(out/"frozen_concept_geometry.json",{"rank":rank,"effective_rank_threshold":.001,"off_diagonal_cosine_mean":float((cosine.sum()-np.trace(cosine))/56),"text_collapse":text_collapse,"basis":"right singular vectors of frozen CONCH text matrix; zero-padded to d=32"})
    incidence={"Pathology":{"K1":1,"K2":1,"K3":0,"K4":1,"K5":0,"K6":0,"K7":1,"K8":0},"Omics":{"K1":1,"K2":1,"K3":1,"K4":1,"K5":1,"K6":1,"K7":1,"K8":1},"Clinical":{"K1":0,"K2":0,"K3":0,"K4":1,"K5":0,"K6":0,"K7":1,"K8":0},"clinical_fields":{"K4":["PATH_T_STAGE","PATH_N_STAGE","PATH_M_STAGE"],"K7":["PATH_T_STAGE","PATH_N_STAGE","PATH_M_STAGE"]}}
    atomic_json(out/"modality_concept_incidence.json",{"incidence":incidence,"basis":"fixed before evaluation from CRC medical concept definitions","random_path_edges":[f"K{i+1}" for i in RANDOM_PATH_EDGES]})
    device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    path_rows=[]; text_rows=[]; gene_rows=[]; clinical_rows=[]; entropy_rows=[]; overlap_rows=[]; specificity_rows=[]; random_rows=[]; start_time=time.time()
    for seed in SEEDS:
        for fold in range(1,6):
            train=np.flatnonzero(raw.outer_folds!=fold); test=np.flatnonzero(raw.outer_folds==fold)
            real,half=fold_omics(raw,bank,train,"mapped_gene_symbols"); shuf,half_shuf=fold_omics(raw,bank,train,"shuffled_mapped_gene_symbols"); clinical=fold_clinical(raw,train)
            torch.manual_seed(seed+fold*1000); np.random.seed(seed+fold*1000)
            model=FrozenTextPathStudent().to(device); fit_path(model,path,queries,real["rna"],real["cna"],train,PATH_EDGES,seed,fold,device)
            score,att=eval_path(model,path,queries,test,device); score_shuf_text,_=eval_path(model,path,queries[TEXT_SHUFFLE],test,device)
            ent,ov,dist=attention_stats(att)
            entropy_rows.append({"seed":seed,"outer_fold":fold,"text_mode":"real","entropy_normalized":ent,"uniform_reference":1.0}); overlap_rows.append({"seed":seed,"outer_fold":fold,"real_top10pct_jaccard":ov}); specificity_rows.append({"seed":seed,"outer_fold":fold,"real_attention_distance":dist})
            for k in PATH_EDGES:
                rr=0.5*(pearson(score[:,k],real["rna"][test,k])+pearson(score[:,k],real["cna"][test,k])); ss=0.5*(pearson(score_shuf_text[:,k],real["rna"][test,k])+pearson(score_shuf_text[:,k],real["cna"][test,k]))
                path_rows.append({"seed":seed,"outer_fold":fold,"edge":f"Pathology->K{k+1}","concept_id":f"K{k+1}","text_mode":"REAL","mean_pearson":rr,"shuffled_text_pearson":ss,"real_minus_shuffled":rr-ss,"rna_pearson":pearson(score[:,k],real["rna"][test,k]),"cna_pearson":pearson(score[:,k],real["cna"][test,k])})
            for k in range(CONCEPTS):
                real_st=pearson(half["rna"][test,k,0],half["rna"][test,k,1]); sh_st=pearson(half_shuf["rna"][test,k,0],half_shuf["rna"][test,k,1]); real_c=pearson(half["cna"][test,k,0],half["cna"][test,k,1]); sh_c=pearson(half_shuf["cna"][test,k,0],half_shuf["cna"][test,k,1])
                gene_rows.append({"seed":seed,"outer_fold":fold,"edge":f"Omics->K{k+1}","concept_id":f"K{k+1}","real_rna_split_half":real_st,"shuffled_rna_split_half":sh_st,"real_cna_split_half":real_c,"shuffled_cna_split_half":sh_c,"real_minus_shuffled":.5*((real_st-sh_st)+(real_c-sh_c))})
            for k in CLINICAL_EDGES:
                value=clinical[test,k]; expected=.5*(spearman(value,real["rna"][test,k])+spearman(value,real["cna"][test,k])); clinical_rows.append({"seed":seed,"outer_fold":fold,"edge":f"Clinical->K{k+1}","concept_id":f"K{k+1}","expected_direction_spearman":expected,"pathology_stage_score_variance":float(np.var(value))})
            # Matched-edge random graph, fixed before metrics and trained independently.
            torch.manual_seed(seed+fold*1000+991); random_model=FrozenTextPathStudent().to(device); fit_path(random_model,path,queries,real["rna"],real["cna"],train,RANDOM_PATH_EDGES,seed,fold,device); random_score,_=eval_path(random_model,path,queries,test,device)
            random_real=[]
            for k in RANDOM_PATH_EDGES: random_real.append(.5*(pearson(random_score[:,k],real["rna"][test,k])+pearson(random_score[:,k],real["cna"][test,k])))
            random_rows.append({"seed":seed,"outer_fold":fold,"graph":"REAL","mean_path_alignment":float(np.mean([r["mean_pearson"] for r in path_rows[-len(PATH_EDGES):]])),"edge_count":len(PATH_EDGES)}); random_rows.append({"seed":seed,"outer_fold":fold,"graph":"RANDOM_MATCHED","mean_path_alignment":float(np.mean(random_real)),"edge_count":len(RANDOM_PATH_EDGES)})
            del model,random_model
            if device.type=="cuda": torch.cuda.empty_cache()
        print(json.dumps({"seed":seed,"device":str(device),"elapsed":round(time.time()-start_time,1)}),flush=True)
    # Edge support statuses are computed only after all fixed fold/seed rows exist.
    edge_rows=[]
    for k in PATH_EDGES:
        rows=[r for r in path_rows if r["concept_id"]==f"K{k+1}"]; status,pos,delta=edge_status([r["mean_pearson"] for r in rows],[r["shuffled_text_pearson"] for r in rows]); edge_rows.append({"modality":"Pathology","concept_id":f"K{k+1}","status":status,"positive_fraction":pos,"mean_control_delta":delta})
    for k in range(CONCEPTS):
        rows=[r for r in gene_rows if r["concept_id"]==f"K{k+1}"]; status,pos,delta=edge_status([r["real_minus_shuffled"] for r in rows],[0.0 for _ in rows]); edge_rows.append({"modality":"Omics","concept_id":f"K{k+1}","status":status,"positive_fraction":pos,"mean_control_delta":delta})
    for k in CLINICAL_EDGES:
        rows=[r for r in clinical_rows if r["concept_id"]==f"K{k+1}"]; values=[r["expected_direction_spearman"] for r in rows]; status="SUPPORTED" if np.mean(np.asarray(values)>0)>=.75 and np.mean(values)>0 else ("DIRECTIONAL" if np.mean(np.asarray(values)>0)>=.5 and np.mean(values)>0 else "UNSUPPORTED"); edge_rows.append({"modality":"Clinical","concept_id":f"K{k+1}","status":status,"positive_fraction":float(np.mean(np.asarray(values)>0)),"mean_control_delta":float(np.mean(values))})
    atomic_csv(out/"pathology_edge_metrics.csv",path_rows); atomic_csv(out/"omics_edge_metrics.csv",gene_rows); atomic_csv(out/"clinical_edge_metrics.csv",clinical_rows); atomic_csv(out/"edge_support_matrix.csv",edge_rows); atomic_csv(out/"attention_entropy.csv",entropy_rows); atomic_csv(out/"attention_overlap.csv",overlap_rows); atomic_csv(out/"concept_specificity.csv",specificity_rows); atomic_csv(out/"real_vs_random_graph.csv",random_rows); atomic_csv(out/"real_vs_shuffled_text.csv",path_rows); atomic_csv(out/"real_vs_shuffled_gene.csv",gene_rows)
    real_graph=np.mean([r["mean_path_alignment"] for r in random_rows if r["graph"]=="REAL"]); random_graph=np.mean([r["mean_path_alignment"] for r in random_rows if r["graph"]=="RANDOM_MATCHED"]); supported=[r for r in edge_rows if r["status"] in ("SUPPORTED","DIRECTIONAL")]; concepts={c for r in supported for c in [r["concept_id"]]}; multimodal={c for c in concepts if sum(r["concept_id"]==c for r in supported)>=2}
    if text_collapse: decision="D. SEMANTIC_OR_TEACHER_AUDIT_FAIL"
    elif real_graph>random_graph and len(supported)>=3 and len(multimodal)>=1: decision="A. SPARSE_MEDICAL_CONCEPT_GRAPH_SUPPORTED"
    elif real_graph>random_graph and len(supported)>=1: decision="B. PARTIAL_CONCEPT_GRAPH_SUPPORTED"
    else: decision="C. MEDICAL_GRAPH_NOT_SUPPORTED"
    atomic_json(out/"run_manifest.json",{"phase":"K2_SPARSE_MEDICAL_CONCEPT_GRAPH","status":"complete","seeds":list(SEEDS),"n_patients":512,"outer_folds":5,"device":str(device),"d":D,"tau":TAU,"epochs":EPOCHS,"trainable_path_parameters":sum(p.numel() for p in FrozenTextPathStudent().parameters() if p.requires_grad),"trainable_text_parameters":0,"student_forward":"pathology + frozen concept queries only","dss_used":False,"survival_run":False,"raw_wsi_used":False,"concept_bank_sha256":bank["content_sha256"],"text_embedding_sha256":sha256_file(Path(args.text_embeddings)),"decision":decision,"real_graph_mean":float(real_graph),"random_graph_mean":float(random_graph),"supported_edges":len(supported),"multimodal_concepts":sorted(multimodal),"fixed_incidence":incidence,"random_path_edges":[f"K{i+1}" for i in RANDOM_PATH_EDGES]})
    atomic_json(out/"concept_geometry_audit.json",{"text_collapse":text_collapse,"rank":rank,"off_diagonal_cosine_mean":float((cosine.sum()-np.trace(cosine))/56),"attention_entropy_mean":float(np.mean([r["entropy_normalized"] for r in entropy_rows])),"attention_overlap_mean":float(np.mean([r["real_top10pct_jaccard"] for r in overlap_rows])),"real_graph_mean":float(real_graph),"random_graph_mean":float(random_graph)})
    lines=["# Phase K2 — Sparse Medical Concept Graph Audit","",f"- Decision: **{decision}**",f"- Frozen text geometry: rank {rank}; collapse={text_collapse}",f"- Real graph mean path alignment: {real_graph:.6f}; matched random graph: {random_graph:.6f}",f"- Supported/directional edges: {len(supported)}; multimodal concepts: {sorted(multimodal)}",f"- DSS, survival, and raw WSI/SVS were not used.","", "Edge statuses are retained in `edge_support_matrix.csv`; no failed edge was removed."]
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"decision":decision,"real_graph":real_graph,"random_graph":random_graph,"supported_edges":len(supported),"multimodal":sorted(multimodal)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--data-root",required=True); p.add_argument("--runtime",required=True); p.add_argument("--text-embeddings",required=True); p.add_argument("--output",required=True); p.add_argument("--cpu",action="store_true"); main(p.parse_args())

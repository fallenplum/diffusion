from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from src.phase_m2 import data as m2
from src.phase_m2.data import RawInputs, FoldArrays, load_raw_inputs, prepare_path_cache, preprocess_fold
SHUFFLE_SIGNATURE_SEED=7102026
@dataclass
class M7RFoldArrays:
    path_tokens: np.ndarray
    rna: np.ndarray
    cna: np.ndarray
    clinical: np.ndarray
    mcat_groups: np.ndarray
    mcat_widths: tuple[tuple[int,int],...]
    shuffled_groups: np.ndarray
    shuffled_widths: tuple[tuple[int,int],...]
    audit: dict

def _visible_symbols(values,symbols,train_idx):
    vals=np.asarray(values); syms=np.asarray(symbols).astype(str)
    good=np.any(np.isfinite(vals[train_idx]),axis=0)
    out=[]; seen=set()
    for s,g in zip(syms.tolist(),good.tolist()):
        s=s.strip().upper()
        if s and g and s not in seen: out.append(s); seen.add(s)
    return out

def _shuffle_group_defs(raw, arrays, fold):
    train_idx=np.flatnonzero(raw.outer_folds!=fold)
    official=m2._read_signature_groups(m2.METADATA_ROOT/"mcat_signatures.csv")
    rna_universe=_visible_symbols(raw.rna,raw.rna_symbols,train_idx); cna_universe=_visible_symbols(raw.cna,raw.cna_symbols,train_idx)
    rng=np.random.default_rng(SHUFFLE_SIGNATURE_SEED)
    rna_perm=np.asarray(rna_universe,dtype=object); cna_perm=np.asarray(cna_universe,dtype=object); rng.shuffle(rna_perm); rng.shuffle(cna_perm)
    rw=[int(x[0]) for x in arrays.mcat_widths]; cw=[int(x[1]) for x in arrays.mcat_widths]
    if sum(rw)>len(rna_perm) or sum(cw)>len(cna_perm): raise RuntimeError("outer-train visible universe smaller than official widths")
    r_groups=[]; c_groups=[]; ro=0; co=0
    for i,(r,c) in enumerate(zip(rw,cw)):
        r_groups.append((f"SHUFFLED_{i:02d}",r_perm[ro:ro+r].tolist())); c_groups.append((f"SHUFFLED_{i:02d}",c_perm[co:co+c].tolist())); ro+=r; co+=c
    rpack,rwidth,r_audit=m2._fit_group_matrix(raw.rna,raw.rna_symbols,train_idx,r_groups,"RNA")
    cpack,cwidth,c_audit=m2._fit_group_matrix(raw.cna,raw.cna_symbols,train_idx,c_groups,"CNA")
    packed=np.zeros((len(raw.patient_ids),6,max(rpack.shape[-1],cpack.shape[-1]),2),dtype=np.float32); packed[:,:,:rpack.shape[-1],0]=rpack; packed[:,:,:cpack.shape[-1],1]=cpack
    def digest(groups): return hashlib.sha256("\n".join("|".join(x) for x in groups).encode()).hexdigest()
    audit={"outer_fold":int(fold),"shuffle_seed":SHUFFLE_SIGNATURE_SEED,"rna_universe_count":len(rna_universe),"cna_universe_count":len(cna_universe),"original_rna_widths":rw,"original_cna_widths":cw,"shuffled_rna_widths":[int(x) for x in rwidth],"shuffled_cna_widths":[int(x) for x in cwidth],"rna_assignment_sha256":digest(r_groups),"cna_assignment_sha256":digest(c_groups),"mapping_reproducibility":"fixed outer-train-visible universe, seed-independent"}
    return packed,tuple(zip(rwidth,cwidth)),audit

def preprocess_m7r(raw:RawInputs,path_tokens:np.ndarray,fold:int):
    base=preprocess_fold(raw,path_tokens,fold); shuffled,widths,audit=_shuffle_group_defs(raw,base,fold)
    return M7RFoldArrays(base.path_tokens,base.rna,base.cna,base.clinical,base.mcat_groups,tuple(base.mcat_widths),shuffled,widths,{**base.audit,"shuffled_signature":audit}),audit
__all__=["RawInputs","FoldArrays","M7RFoldArrays","load_raw_inputs","prepare_path_cache","preprocess_fold","preprocess_m7r","SHUFFLE_SIGNATURE_SEED"]

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_uni2_cox_baseline import bootstrap_ci, concordance_index, fit_cox
from scipy.optimize import minimize
from scipy.special import logsumexp

def fast_cox(t, e, x, alpha=0.1):
    idx=np.flatnonzero(e==1); masks=[t>=t[i] for i in idx]
    def fun(b):
        eta=x@b; ll=0.; g=np.zeros_like(b)
        for i,m in zip(idx,masks):
            w=np.exp(eta[m]-logsumexp(eta[m])); ll += eta[i]-logsumexp(eta[m]); g += x[i]-w@x[m]
        n=max(1,len(idx)); return -ll/n+alpha*(b@b)/2, -g/n+alpha*b
    return minimize(fun,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':30}).x

PAT = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.I)
SAFE_CAT = ["SEX", "PATH_T_STAGE", "PATH_N_STAGE", "PATH_M_STAGE", "AJCC_PATHOLOGIC_TUMOR_STAGE", "AJCC_STAGING_EDITION", "HISTOLOGICAL_DIAGNOSIS", "PRIMARY_SITE_PATIENT"]

def pid(x):
    m = PAT.match(str(x).upper().strip())
    return m.group(1) if m else None

def load_table(path):
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)

def load(root):
    base = root / "data" / "data_8" / "coadread_tcga_pan_can_atlas_2018"
    clinical = load_table(base / "data_clinical_patient.txt")
    supp = load_table(root / "data" / "data_8" / "coadread_tcga" / "data_clinical_patient.txt")
    clinical["patient_id"] = clinical.PATIENT_ID.map(pid)
    supp["patient_id"] = supp.PATIENT_ID.map(pid)
    for c in ["HISTOLOGICAL_DIAGNOSIS", "PRIMARY_SITE_PATIENT"]:
        if c not in clinical:
            clinical[c] = np.nan
        if c in supp:
            clinical[c] = clinical[c].where(clinical[c].notna(), clinical["patient_id"].map(supp.set_index("patient_id")[c]))
    clinical = clinical.drop_duplicates("patient_id").set_index("patient_id")
    rna = load_table(base / "data_mrna_seq_v2_rsem.txt")
    cna = load_table(base / "data_log2_cna.txt")
    def omics(d):
        cols = [c for c in d.columns if pid(c)]
        x = d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).T
        ids = [pid(c) for c in cols]
        out = pd.DataFrame(x, index=ids).groupby(level=0).mean()
        return out
    rna, cna = omics(rna), omics(cna)
    bundle = np.load(root / "outputs" / "features" / "uni2_h_patient_mean.npz")
    path = pd.DataFrame(bundle["features"], index=bundle["patient_ids"].astype(str))
    common = sorted(set(clinical.index) & set(rna.index) & set(cna.index) & set(path.index))
    frame = clinical.loc[common, ["DSS_STATUS", "DSS_MONTHS", "PFS_STATUS", "PFS_MONTHS", "CANCER_TYPE_ACRONYM"]].copy()
    frame.columns = ["dss_event", "dss_time", "pfs_event", "pfs_time", "cancer_type"]
    for c in ["dss_event", "pfs_event"]:
        frame[c] = frame[c].astype(str).str.extract(r"^(0|1)", expand=False)
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    for c in ["dss_time", "pfs_time"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame["patient_id"] = common
    return frame.reset_index(drop=True), rna.loc[common], cna.loc[common], path.loc[common], clinical.loc[common]

def clinical_x(train, ev):
    cont = ["AGE"]
    ts = StandardScaler(); a = ts.fit_transform(train[cont].apply(pd.to_numeric, errors="coerce").fillna(train[cont].apply(pd.to_numeric, errors="coerce").median()))
    b = ts.transform(ev[cont].apply(pd.to_numeric, errors="coerce").fillna(train[cont].apply(pd.to_numeric, errors="coerce").median()))
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    c = enc.fit_transform(train[SAFE_CAT].fillna("MISSING").astype(str)); d = enc.transform(ev[SAFE_CAT].fillna("MISSING").astype(str))
    return np.c_[a, c], np.c_[b, d]

def high_x(train, ev, kind, seed):
    a, b = train.to_numpy(float), ev.to_numpy(float)
    med = np.nanmedian(a, axis=0); med[~np.isfinite(med)] = 0
    a = np.where(np.isfinite(a), a, med); b = np.where(np.isfinite(b), b, med)
    if kind == "rna":
        a = np.log2(np.clip(a, 0, None) + 1); b = np.log2(np.clip(b, 0, None) + 1)
        keep = np.mean(np.isfinite(a), axis=0) >= 0.5
        var = np.nanvar(a[:, keep], axis=0); idx = np.where(keep)[0][np.argsort(var)[-min(3000, keep.sum()):]]
        a, b = a[:, idx], b[:, idx]
        if a.shape[1] > 64:
            idx = np.argsort(np.var(a, axis=0))[-64:]; a, b = a[:, idx], b[:, idx]
        sc = StandardScaler(); return sc.fit_transform(a), sc.transform(b)
    keep = np.mean(np.abs(a) > 1e-8, axis=0) >= 0.05; a, b = a[:, keep], b[:, keep]
    if a.shape[1] > 8:
        idx = np.argsort(np.var(a, axis=0))[-8:]; a, b = a[:, idx], b[:, idx]
    sc = StandardScaler(); return sc.fit_transform(a), sc.transform(b)

def run_endpoint(name, frame, rna, cna, path, clin, endpoint, out, seed):
    t, e = frame[f"{endpoint}_time"].to_numpy(float), pd.to_numeric(frame[f"{endpoint}_event"], errors="coerce").fillna(0).to_numpy(int)
    strata = frame.cancer_type.astype(str) + "_" + e.astype(str)
    outer = StratifiedKFold(5, shuffle=True, random_state=seed)
    if endpoint == "dss":
        split_rows = []
        for fold, (_, te) in enumerate(outer.split(frame, strata), 1):
            split_rows.extend({"patient_id": frame.iloc[i].patient_id, "fold": fold} for i in te)
        pd.DataFrame(split_rows).sort_values("patient_id").to_csv(out / "fixed_patient_folds.csv", index=False)
    specs = {"C0":["C"],"R0":["R"],"N0":["N"],"O0":["R","N"],"P0":["P"],"PO0":["P","R","N"],"PC0":["P","C"],"OC0":["R","N","C"],"POC0":["P","R","N","C"]}
    pred = frame[["patient_id", f"{endpoint}_time", f"{endpoint}_event"]].copy(); rows=[]
    for model, mods in specs.items():
        oof=np.full(len(frame), np.nan)
        for fold,(tr,te) in enumerate(outer.split(frame,strata),1):
            xs, ys = [], []
            for m in mods:
                if m=="C": a,b=clinical_x(clin.iloc[tr],clin.iloc[te])
                elif m=="P": a,b=high_x(path.iloc[tr],path.iloc[te],"cna",seed+fold)
                elif m=="R": a,b=high_x(rna.iloc[tr],rna.iloc[te],"rna",seed+fold)
                else: a,b=high_x(cna.iloc[tr],cna.iloc[te],"cna",seed+fold)
                xs.append(a); ys.append(b)
            xtr, xte = np.c_[*xs], np.c_[*ys]
            beta=fast_cox(t[tr],e[tr],xtr,0.1); oof[te]=xte@beta
            rows.append({"endpoint":endpoint,"model":model,"fold":fold,"n_test":len(te),"events_test":int(e[te].sum()),"fold_c":concordance_index(t[te],e[te],oof[te])})
        pred[f"{model}_risk"] = oof
        rows.append({"endpoint":endpoint,"model":model,"fold":"OOF","n_test":len(frame),"events_test":int(e.sum()),"fold_c":concordance_index(t,e,oof),"ci_low":bootstrap_ci(t,e,oof,seed=seed)[0],"ci_high":bootstrap_ci(t,e,oof,seed=seed)[1]})
    out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/f"{endpoint}_metrics.csv",index=False); pred.to_csv(out/f"{endpoint}_predictions.csv",index=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument("--seed",type=int,default=20260904); args=ap.parse_args(); root=args.root.resolve()
    frame,rna,cna,path,clin=load(root); out=root/"outputs"/"phase_m1_complete"; out.mkdir(parents=True,exist_ok=True)
    frame.to_csv(out/"aligned_patients.csv",index=False); frame.groupby(["cancer_type","dss_event"]).size().reset_index(name="n").to_csv(out/"alignment_audit.csv",index=False)
    run_endpoint("DSS",frame,rna,cna,path,clin,"dss",out,args.seed); run_endpoint("PFS",frame,rna,cna,path,clin,"pfs",out,args.seed)
    (out/"run_manifest.json").write_text(json.dumps({"seed":args.seed,"n_patients":len(frame),"dss_events":int(frame.dss_event.sum()),"pfs_events":int(frame.pfs_event.sum()),"models":9,"feature_note":"RNA/CNA fold-local preprocessing followed by variance cap at 64 dimensions for the existing Cox optimizer"},indent=2),encoding="utf-8")
    print(json.dumps({"n_patients":len(frame),"dss_events":int(frame.dss_event.sum()),"pfs_events":int(frame.pfs_event.sum()),"output":str(out)},ensure_ascii=False))
if __name__=="__main__": main()

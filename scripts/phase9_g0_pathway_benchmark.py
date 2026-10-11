"""Phase9-G0: leakage-safe KEGG pathway pooling and tiny graph audit.

The script keeps patient-level arrays on the execution host and writes only
fold metrics/OOF risks plus public KEGG provenance.  Pathway membership and
gene edges are reconstructed from KEGG REST link/KGML files; no outcome is
used when forming the 32 pathway representation.
"""
from __future__ import annotations
import csv, hashlib, json, random, sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase8_v0.cox import efron_loss_grad, km_left, concordance

SEEDS = [20261007, 20261008, 20261009]
PATHWAYS = ["05210","04310","04350","04010","04151","04012","04014","04015",
            "04390","04330","04340","04066","04630","04064","04668","04068",
            "04110","04115","04210","04218","04150","04152","04510","04512",
            "04520","04530","04670","04660","04650","04620","03430","03440"]
P_NAMES = ["colorectal","wnt","tgfb","mapk","pi3k_akt","erbB","ras","rap1","hippo","notch","hedgehog","hif1","jak_stat","nfkb","tnf","foxo","cell_cycle","p53","apoptosis","senescence","mtor","ampk","focal_adhesion","ecm_receptor","adherens","tight_junction","leukocyte_migration","tcr","nk_cytotoxicity","tlr","mismatch_repair","homologous_repair"]

def ahash(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def fit_cox(x, t, e, lam):
    obj = efron_loss_grad(np.asarray(x,float), np.asarray(t,float), np.asarray(e,int), lam)
    r = minimize(obj, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter":50000,"ftol":1e-14,"gtol":1e-9,"maxls":100})
    loss,g = obj(r.x)
    if not np.isfinite(loss) or np.max(np.abs(g))>2e-5: raise RuntimeError(f"cox failed {r.message} grad={np.max(np.abs(g))}")
    return r.x

def pair(t,e,r,w=None):
    t=np.asarray(t); e=np.asarray(e); r=np.asarray(r); w=np.ones(len(t)) if w is None else np.asarray(w)
    ok=(e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0)))
    z=ok*w[:,None]; den=float(z.sum()); cr=(r[:,None]>r[None,:]).astype(float)+.5*(r[:,None]==r[None,:])
    return float((z*cr).sum()/den) if den else float("nan"), float(den), float((z*cr).sum())

def norm_fit(x, idx):
    mu=np.nanmedian(x[idx],axis=0); y=np.where(np.isfinite(x),x,mu); sd=np.nanstd(y[idx],axis=0); sd[sd<1e-8]=1.
    return (y-mu)/sd, mu, sd

def raw_top(x, idx, k=512):
    y,mu,sd=norm_fit(np.asarray(x,float),idx); var=np.nanvar(y[idx],axis=0); keep=np.argsort(-var,kind="mergesort")[:k]
    return y[:,keep], keep

def parse_kegg(kroot):
    kroot = Path(kroot)
    # Map official hsa IDs to primary symbols and aliases in list/hsa.
    sym={}
    for line in (kroot/"list_hsa.txt").read_text(errors="ignore").splitlines():
        if "\t" not in line: continue
        gid,desc=line.split("\t",1); gid=gid.split(":")[-1]; first=desc.split(";",1)[0]
        names=[z.strip() for z in first.split(",") if z.strip()]
        for z in names: sym.setdefault(z.upper(),gid)
    members=[]; edges=set(); allg=set()
    for p in PATHWAYS:
        ids=[]
        for line in (kroot/f"link_hsa{p}.txt").read_text(errors="ignore").splitlines():
            q=line.split("\t")
            if len(q)==2: ids.append(q[1].split(":")[-1])
        ids=sorted(set(ids)); members.append(ids); allg.update(ids)
        try:
            root=ET.parse(kroot/f"kgml_hsa{p}.txt").getroot(); gene_entries={}
            for en in root.findall("entry"):
                if en.attrib.get("type")!="gene": continue
                vals=[x.split(":")[-1] for x in en.attrib.get("name","").split()]
                vals=[x for x in vals if x in allg or x in ids]
                for a in vals: gene_entries.setdefault(a,set()).add(a)
            for rel in root.findall("relation"):
                a=rel.attrib.get("entry1"); b=rel.attrib.get("entry2")
                # KGML relation entry IDs are local; resolve gene entry IDs.
                ea=root.find(f"entry[@id='{a}']"); eb=root.find(f"entry[@id='{b}']")
                if ea is None or eb is None or ea.attrib.get("type")!="gene" or eb.attrib.get("type")!="gene": continue
                aa=[x.split(":")[-1] for x in ea.attrib.get("name","").split()]; bb=[x.split(":")[-1] for x in eb.attrib.get("name","").split()]
                for x in aa:
                    for y in bb:
                        if x in ids and y in ids and x!=y: edges.add(tuple(sorted((x,y))))
        except Exception: pass
    return sym,members,sorted(edges)

def build_gene_features(d, kroot, train):
    sym,members,edges=parse_kegg(kroot); rs=np.asarray(d["rna_gene_symbols"]).astype(str); cs=np.asarray(d["cna_gene_symbols"]).astype(str)
    rmap={z.upper():i for i,z in enumerate(rs)}; cmap={z.upper():i for i,z in enumerate(cs)}
    gids=sorted(set(sum(members,[]))); node=[]; ridx=[]; cidx=[]
    for g in gids:
        desc=[k for k,v in sym.items() if v==g]
        ri=next((rmap[z] for z in desc if z in rmap),None); ci=next((cmap[z] for z in desc if z in cmap),None)
        if ri is not None or ci is not None: node.append(g); ridx.append(ri); cidx.append(ci)
    # training-fold normalization, with separate RNA and CNA gain/loss channels
    rn=np.asarray(d["rna_genes"],float); cn=np.asarray(d["cna_genes"],float); n=len(node); X=np.zeros((rn.shape[0],n,3),float)
    for j,(ri,ci) in enumerate(zip(ridx,cidx)):
        if ri is not None: X[:,j,0]=rn[:,ri]
        if ci is not None: X[:,j,1]=np.maximum(cn[:,ci],0); X[:,j,2]=np.maximum(-cn[:,ci],0)
    for c in range(3):
        mu=np.nanmedian(X[train,:,c],axis=0); X[:,:,c]=np.where(np.isfinite(X[:,:,c]),X[:,:,c],mu)
        sd=X[train,:,c].std(0); sd[sd<1e-8]=1; X[:,:,c]=(X[:,:,c]-mu)/sd
    gi={g:i for i,g in enumerate(node)}; A=np.eye(n); true_edges=[]
    for a,b in edges:
        if a in gi and b in gi: A[gi[a],gi[b]]=A[gi[b],gi[a]]=1; true_edges.append((gi[a],gi[b]))
    deg=A.sum(1); A=A/np.maximum(A.sum(1,keepdims=True),1)
    P=[]
    for mm in members: P.append([gi[g] for g in mm if g in gi])
    return X,A,P,true_edges,{"node_count":n,"edge_count":len(true_edges),"pathway_member_counts":[len(x) for x in P],"mapping_symbols":len(sym)}

def graph_repr(X,A,P,seed,train,t,e,base=None,epochs=70):
    import torch
    torch.manual_seed(seed); dev="cuda" if torch.cuda.is_available() else "cpu"; x=torch.tensor(X,dtype=torch.float32,device=dev); aa=torch.tensor(A,dtype=torch.float32,device=dev)
    # Shared 3->8 encoder, one normalized message-passing layer, 32 pathway slots.
    enc=torch.nn.Linear(3,8).to(dev); head=torch.nn.Linear(8,1).to(dev); pw=torch.nn.Parameter(torch.zeros(32,device=dev)); off=torch.nn.Parameter(torch.zeros(1,device=dev));
    opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters())+[pw,off],lr=2e-3,weight_decay=1e-3)
    ti=torch.tensor(t[train],dtype=torch.float32,device=dev); ei=torch.tensor(e[train],dtype=torch.float32,device=dev); bi=None if base is None else torch.tensor(base[train],dtype=torch.float32,device=dev)
    masks=[]
    for q in P:
        m=torch.zeros(A.shape[0],device=dev); m[q]=1/max(1,len(q)); masks.append(m)
    M=torch.stack(masks)
    def forward(xx,bb=None):
        h=torch.relu(enc(xx)); h=torch.relu(torch.einsum('ij,bjk->bik',aa,h))
        z=torch.stack([torch.einsum('n,bnh->bh',M[k],h) for k in range(32)],1)
        rr=(head(z).squeeze(-1)*pw).sum(1)
        return rr if bb is None else bb+off+rr
    best=(float("inf"),None); wait=0
    for ep in range(1,epochs+1):
        opt.zero_grad(); r=forward(x[train],bi); loss=cox_torch(r,ti,ei)+1e-3*(pw@pw+sum((q*q).sum() for q in enc.parameters()))
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(head.parameters())+[pw,off],5); opt.step()
        with torch.no_grad(): rv=forward(x, None if base is None else torch.tensor(base,dtype=torch.float32,device=dev)).detach().cpu().numpy()
        score=concordance(t[~np.isin(np.arange(len(t)),train)],e[~np.isin(np.arange(len(t)),train)],rv[~np.isin(np.arange(len(t)),train)]) if False else float(loss.detach().cpu())
        if score<best[0]-1e-5: best=(score,ep); wait=0
        else: wait+=1
        if wait>=15: break
    with torch.no_grad(): out=forward(x,None if base is None else torch.tensor(base,dtype=torch.float32,device=dev)).detach().cpu().numpy()
    return out,{"epochs":int(best[1] or ep),"device":dev,"parameters":sum(p.numel() for p in list(enc.parameters())+list(head.parameters())+[pw,off])}

def cox_torch(r,t,e):
    import torch
    loss=torch.tensor(0.,device=r.device); ev=int(e.sum().item())
    for tv in torch.unique(t[e>0]):
        die=(t==tv)&(e>0); n=die.sum(); risk=t>=tv; ex=torch.exp(r-r.max());
        for l in range(int(n.item())): loss=loss+torch.log((ex*(risk.float()-(l/n)*die.float())).sum())+r.max()-r[die].sum()/n
    return loss/max(ev,1)

def main(outdir, data_parent, ctxdir, kroot):
    out=Path(outdir); (out/"oof").mkdir(parents=True,exist_ok=True); (out/"cells").mkdir(exist_ok=True)
    d=dict(np.load(Path(data_parent)/"private_dataset.npz",allow_pickle=True)); allrows=[]; risks={}; weights={}
    for seed in SEEDS:
      risks[seed]={}; weights[seed]=np.zeros(len(d["patient_ids"]))
      for fold in range(1,6):
        ctx=np.load(Path(ctxdir)/f"seed{seed}_fold{fold}.npz"); tr,te,it,iv=[ctx[k] for k in ("tr","te","it","iv")]
        # B0 is the registered Clinical+TITAN S32 anchor; fit ridge on inner train only.
        xi=np.c_[ctx["inner_C"],ctx["inner_S"]]; xo=np.c_[ctx["outer_C"],ctx["outer_S"]]; trials=[]
        for lam in [0.01,0.1,1.,10.]:
            w=fit_cox(xi[it],d["DSS_time"][it],d["DSS_event"][it],lam); trials.append((concordance(d["DSS_time"][iv],d["DSS_event"][iv],xi[iv]@w),lam))
        lam=max(trials,key=lambda z:(z[0],-z[1]))[1]; w=fit_cox(xo[tr],d["DSS_time"][tr],d["DSS_event"][tr],lam); base_te=xo[te]@w; base_tr=ctx["outer_oof_anchor"][tr];
        for arm,risk in [("B0",base_te)]:
            allrows.append(cell(seed,fold,arm,risk,te,d)); np.savez_compressed(out/"oof"/f"DSS_seed{seed}_fold{fold}_{arm}.npz",test_indices=te,risk=risk,seed=seed,fold=fold)
        R,keep=raw_top(d["rna_genes"],tr,512); O=ctx["outer_O"]; feats={"B1":R,"B2":O[:,0:16],"B3":O[:,16:24],"O1":R,"O2":O[:,0:16],"O3":O[:,16:32]}
        feats["B3"]=O[:,16:32]
        for arm,F in feats.items():
            zz=fit_cox(np.c_[xo[tr],F[tr]] if arm.startswith("B") else F[tr],d["DSS_time"][tr],d["DSS_event"][tr],0.1)
            rr=(np.c_[xo[te],F[te]] if arm.startswith("B") else F[te])@zz; allrows.append(cell(seed,fold,arm,rr,te,d)); np.savez_compressed(out/"oof"/f"DSS_seed{seed}_fold{fold}_{arm}.npz",test_indices=te,risk=rr,seed=seed,fold=fold)
        X,A,P,edges,audit=build_gene_features(d,kroot,tr); rng=np.random.default_rng(seed+fold); perm=rng.permutation(len(X[0])); Ar=A[perm][:,perm]
        for arm,graph,bb in [("B4",False,True),("B5",True,True),("B6",True,True),("O4",False,False),("O5",True,False),("O6",True,False)]:
            AA=np.eye(A.shape[0]) if arm in ("B4","O4") else (Ar if arm in ("B6","O6") else A)
            bo=None
            if bb:
                bo=np.zeros(len(d["patient_ids"]),float); bo[tr]=ctx["outer_oof_anchor"][tr]; bo[te]=ctx["outer_test_anchor"]
            rr,gd=graph_repr(X,AA,P,seed+fold,tr,d["DSS_time"],d["DSS_event"],base=bo)
            # graph_repr returns all-patient risk; test slice is outer test.
            allrows.append(cell(seed,fold,arm,rr[te],te,d,extra={**audit,**gd,"graph":"random" if arm in ("B6","O6") else "kegg"}))
            np.savez_compressed(out/"oof"/f"DSS_seed{seed}_fold{fold}_{arm}.npz",test_indices=te,risk=rr[te],seed=seed,fold=fold)
        for arm in ["B0","B1","B2","B3","B4","B5","B6","O1","O2","O3","O4","O5","O6"]:
            rr=np.array([r["risk"] for r in allrows if r["seed"]==seed and r["fold"]==fold and r["method"]==arm][-1]);
            # only per-cell temporary; canonical OOF assembled below from files
        print(seed,fold,flush=True)
    import pandas as pd
    pd.DataFrame(allrows).drop(columns=["risk"]).to_csv(out/"cell_summary.csv",index=False)
    # assemble every arm and seed from rows and evaluate fold-wise metrics
    metrics=[]
    for seed in SEEDS:
      for arm in sorted(set(r["method"] for r in allrows)):
        rr=np.full(len(d["patient_ids"]),np.nan)
        for r in allrows:
          if r["seed"]==seed and r["method"]==arm: rr[r["test_indices"]]=r["risk"]
        valid=np.isfinite(rr); ww=np.zeros(len(rr));
        for fold in range(1,6):
          ix=np.flatnonzero((d["folds"]==fold)&valid); g=km_left(d["DSS_time"][d["folds"]!=fold],d["DSS_event"][d["folds"]!=fold],d["DSS_time"][ix]); ww[ix]=((d["DSS_event"][ix]==1)&(d["DSS_time"][ix]<=60))/(g*g)
        h=pair(d["DSS_time"][valid],d["DSS_event"][valid],rr[valid])[0]; u=pair(d["DSS_time"][valid],d["DSS_event"][valid],rr[valid],ww[valid])[0]
        metrics.append({"seed":seed,"method":arm,"within_fold_harrell":h,"uno60":u,"oof_sha256":ahash(rr[valid])})
    pd.DataFrame(metrics).to_csv(out/"seed_metrics.csv",index=False)
    (out/"kegg_audit.json").write_text(json.dumps({"source":"https://rest.kegg.jp","pathways":PATHWAYS,"names":P_NAMES,"mapping":"official list/hsa aliases; KGML gene relations","files_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(kroot).glob("*.txt")}},indent=2),encoding="utf-8")
    (out/"DONE.json").write_text(json.dumps({"status":"PASS","cells":len(allrows),"protocol":"Phase9-G0","patient_level_files_not_published":True},indent=2),encoding="utf-8")
    print("DONE",len(allrows))

def cell(seed,fold,method,risk,te,d,extra=None): return {"seed":seed,"fold":fold,"method":method,"risk":np.asarray(risk),"test_indices":np.asarray(te),"within_fold_harrell":pair(d["DSS_time"][te],d["DSS_event"][te],risk)[0],"uno60":None,**(extra or {})}

if __name__=="__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])

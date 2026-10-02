from __future__ import annotations
import argparse,csv,hashlib,json,re
from pathlib import Path
import numpy as np,pandas as pd
from src.phase_m2.common import atomic_csv,atomic_json
PAT=re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})",re.I)
def pid(x):
 m=PAT.match(str(x).strip().upper());return m.group(1) if m else None
def sha(x):return hashlib.sha256(str(x).encode()).hexdigest()
def sample_ids(path):
 t=pd.read_csv(path,sep="\t",comment="#",dtype=str,low_memory=False); out=set()
 for c in t.columns[2:]:
  p=pid(c)
  if p:out.add(p)
 return out
def run(args):
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);private=Path(args.private);private.mkdir(parents=True,exist_ok=True)
 canonical=pd.read_csv(args.manifest,dtype=str).fillna(""); canonical["patient_id"]=canonical.patient_id.map(pid); canonical=canonical.dropna(subset=["patient_id"]).drop_duplicates("patient_id")
 data=Path(args.data_root); h5ids=set()
 for p in (data/"TCGA").glob("TCGA-*/*.h5"):
  x=pid(p.name)
  if x:h5ids.add(x)
 rnaids=sample_ids(data/"data/data_8/coadread_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt"); cnaids=sample_ids(data/"data/data_8/coadread_tcga_pan_can_atlas_2018/data_log2_cna.txt")
 rows=[]
 for _,r in canonical.iterrows():
  dss=bool(str(r.get("DSS_valid","")).lower()=="true");pfs=bool(str(r.get("PFS_valid","")).lower()=="true"); pathcanon=bool(str(r.get("UNI2h_available","")).lower()=="true"); rnac=bool(str(r.get("RNA_available","")).lower()=="true");cnac=bool(str(r.get("CNA_available","")).lower()=="true");clin=bool(str(r.get("Clinical_available","")).lower()=="true"); pathrun=r.patient_id in h5ids
  row={"patient_hash":sha(r.patient_id),"project":"TCGA-"+str(r.get("cancer_type","UNKNOWN")),"DSS_valid":dss,"DSS_time":float(r.DSS_MONTHS) if dss else "","DSS_event":int(float(r.DSS_event)) if dss else "","PFS_valid":pfs,"PFS_time":float(r.PFS_MONTHS) if pfs else "","PFS_event":int(float(r.PFS_event)) if pfs else "","clinical_available":clin,"pathology_available":pathcanon,"rna_available":rnac,"cna_available":cnac,"genomics_available":rnac and cnac,"pathology_runtime_staging_missing":pathcanon and not pathrun,"pathology_canonical_source_missing":not pathcanon,"rna_source_missing":not rnac,"cna_source_missing":not cnac}
  row["pattern_CPRA"]="["+",".join(str(int(x)) for x in [clin,pathcanon,rnac,cnac])+"]";row["pattern_CPG"]="["+",".join(str(int(x)) for x in [clin,pathcanon,rnac and cnac])+"]";rows.append(row)
 private_csv=private/"modality_availability_patient_level.csv";atomic_csv(private_csv,rows)
 def aggregate(endpoint):
  valid=[r for r in rows if r[endpoint+"_valid"]];groups={}
  for r in valid:
   pats=[]
   if r["clinical_available"]:pats.append("C")
   if r["pathology_available"]:pats.append("P")
   if r["genomics_available"]:pats.append("G")
   key="+".join(pats) if pats else "other";groups.setdefault(key,[]).append(r)
  outrows=[]
  for k,z in sorted(groups.items()):
   ev=sum(int(r[endpoint+"_event"]) for r in z);outrows.append({"endpoint":endpoint,"pattern":k,"N":len(z),"events":ev,"censored":len(z)-ev,"event_fraction":ev/len(z)})
  return outrows
 census=aggregate("DSS")+aggregate("PFS");atomic_csv(out/"modality_pattern_census.csv",census)
 dss=[r for r in rows if r["DSS_valid"]];pfs=[r for r in rows if r["PFS_valid"]]
 def summary(z,e):
  n=len(z);ev=sum(int(r[e+"_event"]) for r in z);return {"N":n,"events":ev,"censored":n-ev}
 s={"DSS_complete_case":{"N":sum(r["clinical_available"] and r["pathology_available"] and r["genomics_available"] for r in dss),"events":sum(int(r["DSS_event"]) for r in dss if r["clinical_available"] and r["pathology_available"] and r["genomics_available"])},"DSS_all_outcome_labeled":summary(dss,"DSS"),"PFS_complete_case":{"N":sum(r["clinical_available"] and r["pathology_available"] and r["genomics_available"] for r in pfs),"events":sum(int(r["PFS_event"]) for r in pfs if r["clinical_available"] and r["pathology_available"] and r["genomics_available"])},"PFS_all_outcome_labeled":summary(pfs,"PFS"),"canonical_patient_count":len(rows),"runtime_h5_patient_count":len(h5ids),"runtime_staging_missing_pathology":sum(r["pathology_runtime_staging_missing"] for r in rows)}
 s["DSS_incomplete"]={"N":s["DSS_all_outcome_labeled"]["N"]-s["DSS_complete_case"]["N"],"events":s["DSS_all_outcome_labeled"]["events"]-s["DSS_complete_case"]["events"]};s["PFS_incomplete"]={"N":s["PFS_all_outcome_labeled"]["N"]-s["PFS_complete_case"]["N"],"events":s["PFS_all_outcome_labeled"]["events"]-s["PFS_complete_case"]["events"]};s["PotentialPatientGain_DSS"]=s["DSS_incomplete"]["N"];s["PotentialEventGain_DSS"]=s["DSS_incomplete"]["events"];s["PotentialEventGainRatio_DSS"]=s["PotentialEventGain_DSS"]/max(1,s["DSS_complete_case"]["events"]);atomic_json(out/"cohort_expansion_summary.json",s);atomic_json(out/"source_missingness_audit.json",{"canonical_manifest":str(args.manifest),"pathology":{"canonical_available_from_manifest":sum(bool(r["pathology_available"]) for r in rows),"runtime_h5_available":len(h5ids),"runtime_staging_missing":sum(r["pathology_runtime_staging_missing"] for r in rows),"true_source_missing":sum(r["pathology_canonical_source_missing"] for r in rows)},"rna":{"canonical_available":sum(r["rna_available"] for r in rows),"source_missing":sum(r["rna_source_missing"] for r in rows)},"cna":{"canonical_available":sum(r["cna_available"] for r in rows),"source_missing":sum(r["cna_source_missing"] for r in rows)},"interpretation":"runtime staging missing is kept separate from canonical source missing"})
 print(json.dumps(s,ensure_ascii=False))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--data-root",required=True);p.add_argument("--private",required=True);p.add_argument("--output",required=True);run(p.parse_args())

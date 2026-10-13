"""Audit data, source versions and dependencies; no formal outer metrics."""
import argparse, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np, torch
from src.phase11_semantic.data import audit_cache
from src.phase10_trimodal.data import digest,write_json


def main():
    a=argparse.ArgumentParser();a.add_argument('--cache',default='/root/phase11_semantic_anchor_private/cache');args=a.parse_args()
    report=ROOT/'reports/phase11_semantic_anchored_trimodal';report.mkdir(parents=True,exist_ok=True)
    p10=json.loads((ROOT/'reports/phase10_trimodal_mechanism_audit/SOURCE_REGISTRY.json').read_text())
    paths=[ROOT/'reports/phase10_trimodal_mechanism_audit'/n for n in ['RESULTS.md','DECISION.md','KNOWN_LIMITATIONS.md','DATA_AUDIT.md','INDEPENDENT_AUDIT.md','MODEL_MATRIX.csv','PROTOCOL_LOCK.json','PAPER_CODE_AUDIT.md']]
    paths+=list((ROOT/'src/phase10_trimodal').glob('*.py'))+list((ROOT/'reports/phase10_trimodal_mechanism_audit/training_source_v2').rglob('*.py'))
    paths+=[ROOT/p for p in ['reports/phase6a_titan_semantic_preflight/RESULTS.md','reports/phase6_conch_v15_download_audit/RESULTS.md','reports/phase8_v0r/simple_ridge_controls/RESULTS.md']]
    paths+=[ROOT/s['path'] for s in p10['historical_sources']]
    registry=[]
    for p in dict.fromkeys(paths):
        if p.exists():
            content=p.read_text();registry.append({'ref':'9040f5cb32d311832b3107bbba1569a8f6f98761','path':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':len(content.encode()),'status':'READ_AVAILABLE'})
        else:registry.append({'path':str(p.relative_to(ROOT)),'status':'SOURCE_MISSING_NO_SUBSTITUTION'})
    result=audit_cache('/root/phase7a_foundation_headroom_v1/private_dataset.npz','/root/phase10_trimodal_private/cache/private_cohort_manifest.csv','/root/autodl-tmp/phase10_conch',args.cache)
    deps={}
    from importlib.metadata import version
    for name in ['torch','numpy','pandas','scipy','scikit-learn','h5py','scikit-survival']:
        deps[name]=version(name)
    source={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'history':registry,
            'dependencies':deps,'python':sys.version,'platform':platform.platform(),'CPU':platform.processor(),'CPU_count':os.cpu_count(),
            'cuda':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'data':result,
            'paper_sources':p10['official_sources'] if 'official_sources' in p10 else p10.get('official_repositories'),
            'feature_models':p10['feature_models'],'feature_model_provenance':p10['feature_model_provenance']}
    write_json(report/'SOURCE_REGISTRY.json',source);write_json(report/'PATIENT_SLIDE_ALIGNMENT_AUDIT.json',result)
    print(json.dumps({'data':result,'sources':len(registry),'dependencies':deps}))
if __name__=='__main__':main()

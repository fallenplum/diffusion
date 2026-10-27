"""Verify saved preprocessing replay on actual formal partitions, no refits."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.bgb_en_run import load_inputs
from src.bgb_en.data import sha256
from src.bgb_en.replay import apply_saved_preprocessing,replay_full_risks


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/bgb_en/prototype_v1.json');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(2)
    _,_,raw,_,lineage=load_inputs(args);rows=[]
    for path in sorted((args.private/'prepared').glob('*.pt')):
        receipt=json.loads(path.with_suffix('.json').read_text())
        if receipt['lineage']!=lineage or receipt['sha256']!=sha256(path):raise ValueError('Prepared source changed')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        rebuilt=apply_saved_preprocessing(raw,saved['features']['state']);errors={}
        for key,value in rebuilt.items():
            if key=='state':continue
            expected=saved['features'][key]
            error=float(np.max(np.abs(value.astype(float)-expected.astype(float))))
            tolerance=2e-12 if value.dtype==np.float64 else 0.
            if error>tolerance:raise ValueError('Saved feature transformation replay mismatch: '+key)
            errors[key]=error
        risks=replay_full_risks(rebuilt,saved['full_models'])
        risk_errors={key:float(np.max(np.abs(value-saved['full_risks'][key]))) for key,value in risks.items()}
        if max(risk_errors.values())>2e-12:raise ValueError('Rebuilt anchor risk differs')
        rows.append(dict(partition=path.stem,status='PASS',feature_max_errors=errors,risk_max_errors=risk_errors,prepared_sha256=sha256(path)))
    result=dict(schema='bgb_en_saved_preprocessing_replay_v1',status='PASS' if len(rows)==3 else 'PARTIAL',
        optimizer_fits=0,PCA_fits=0,cohort_arrays_exported=False,partitions=rows,
        source_sha256={name:sha256(ROOT/name) for name in ('src/bgb_en/replay.py','scripts/bgb_en_replay_feature_audit.py')})
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],partitions=len(rows),maximum_risk_error=max((max(row['risk_max_errors'].values()) for row in rows),default=None))))


if __name__=='__main__':main()

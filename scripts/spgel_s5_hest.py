"""S5-B1 finite shared-feature benchmark; no author-SOTA reproduction claim."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="2"
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("operation",choices=("prepare-dev","fit-dev","select","prepare-final","fit-final","evaluate","replay","report"))
    p.add_argument("--assets",required=True,type=Path)
    p.add_argument("--config",type=Path,default=ROOT/"configs/spgel_s5_hest/shared_feature_v1.json")
    p.add_argument("--private",required=True,type=Path)
    p.add_argument("--dataset",choices=("COAD","READ"));p.add_argument("--fold",type=int,choices=(0,1))
    p.add_argument("--arm",choices=("MLP","GRAPHSAGE","IDENTITY","QUERY4","SPGEL50_ADAPT"))
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    from src.spgel_s5_hest import execution as x
    common=(args.assets,args.config,args.private)
    if args.operation in ("fit-dev","fit-final"):
        if args.dataset is None or args.fold is None or args.arm is None:p.error("fit needs dataset/fold/arm")
        fn=x.fit_development if args.operation=="fit-dev" else x.fit_final
        r=fn(*common,args.dataset,args.fold,args.arm)
    elif args.operation=="report":
        if args.output is None:p.error("report needs output")
        r=x.report(*common,args.output)
    else:
        fn={"prepare-dev":x.prepare_development,"select":x.select_schedule,"prepare-final":x.prepare_final,"evaluate":x.evaluate_final,"replay":x.replay_all}[args.operation]
        r=fn(*common)
    print(json.dumps({k:r[k] for k in ("status","internal_contract_sha256") if k in r},allow_nan=False))

if __name__=="__main__":main()

"""S5-A separately locked mechanism fits; historical source contracts stay read-only."""
import os
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[name]="2"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("qualify-lock","fit","replay"))
    parser.add_argument("--assets",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/spgel_s5/mechanism_v1.json")
    parser.add_argument("--private",type=Path,required=True)
    parser.add_argument("--arm",choices=tuple(f"M{i}" for i in range(7)))
    parser.add_argument("--donor",type=int,choices=range(6))
    parser.add_argument("--seed",type=int,choices=(31,47))
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    from src.spgel_s5 import execution
    config=execution.recipe(args.config);source=execution.load_source(args.assets,config)
    if args.operation=="qualify-lock":
        result=execution.make_lock(source,args.config,args.private,device=args.device)
    else:
        if args.arm is None or args.donor is None or args.seed is None:parser.error("fit/replay require --arm --donor --seed")
        fn=execution.fit_cell if args.operation=="fit" else execution.replay_cell
        result=fn(source,args.config,args.private,arm=args.arm,donor=args.donor,seed=args.seed,device=args.device)
    print(json.dumps({k:result[k] for k in ("status","arm","donor","seed","contract_sha256") if k in result}),flush=True)


if __name__=="__main__":main()

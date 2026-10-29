"""New official 50-HVG lock/fit/replay/report; old source contracts are invalid."""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.spgel_hest.execution import make_lock, fit_cell, replay_all, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("qualify-lock", "fit", "replay", "report"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/spgel_hest/ridge_v1.json")
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--dataset", choices=("COAD", "READ"))
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.operation == "qualify-lock":
        result = make_lock(args.assets, args.config, args.private)
    elif args.operation == "fit":
        parser.error("fit requires --dataset and --fold") if args.dataset is None or args.fold is None else None
        result = fit_cell(args.assets, args.config, args.private, args.dataset, args.fold)
    elif args.operation == "replay":
        result = replay_all(args.assets, args.config, args.private)
    else:
        if args.output is None:
            parser.error("report requires --output")
        result = report(args.assets, args.config, args.private, args.output)
    print(json.dumps({key: result[key] for key in ("status", "internal_contract_sha256") if key in result}))


if __name__ == "__main__":
    main()

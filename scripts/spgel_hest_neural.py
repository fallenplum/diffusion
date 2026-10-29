"""Separately locked 12-fit HEST neural controls and 36 saved-state replays.

All expression, preprocessing caches, checkpoints and predictions remain in the
private local namespace. Official Ridge artifacts and older source runs are
read-only. There is no epoch selection or automatic fit retry.
"""
import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "2"

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("qualify-lock", "fit", "replay", "report"))
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--config", type=Path,
        default=ROOT / "configs/spgel_hest_neural/pilot_v1.json")
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--ridge-private", type=Path)
    parser.add_argument("--dataset", choices=("COAD", "READ"))
    parser.add_argument("--fold", type=int, choices=(0, 1))
    parser.add_argument("--arm", choices=("MLP", "SPATIAL_GNN", "IDENTITY_GNN"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from src.spgel_hest_neural import neural_execution as execution
    if args.operation == "qualify-lock":
        if args.ridge_private is None:
            parser.error("qualify-lock requires --ridge-private (read-only unsupervised pipeline)")
        result = execution.make_lock(args.assets, args.config, args.private, args.ridge_private)
    elif args.operation == "fit":
        if args.dataset is None or args.fold is None or args.arm is None:
            parser.error("fit requires --dataset, --fold and --arm")
        result = execution.fit_cell(args.assets, args.config, args.private,
            args.dataset, args.fold, args.arm)
    elif args.operation == "replay":
        result = execution.replay_all(args.assets, args.config, args.private)
    else:
        if args.output is None:
            parser.error("report requires --output")
        result = execution.report(args.assets, args.config, args.private, args.output)
    print(json.dumps({key: result[key] for key in ("status", "internal_contract_sha256")
        if key in result}, allow_nan=False))


if __name__ == "__main__":
    main()

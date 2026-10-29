"""Run exactly one prelocked SPGEL source-only optimizer fit."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.spgel.training import read_json, validate_recipe, load_assets, fit_cell


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/spgel/pilot_v1.json")
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--stage", choices=("pilot", "s4"), default="pilot")
    p.add_argument("--arm", required=True)
    p.add_argument("--heldout-donor", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    config = read_json(args.config); validate_recipe(config)
    source = load_assets(args.assets, config)
    result = fit_cell(source, config, args.config, args.private, arm=args.arm,
        donor=args.heldout_donor, seed=args.seed, device=args.device, stage=args.stage)
    print(json.dumps({k: result[k] for k in ("status", "arm", "heldout_donor", "seed", "stage", "optimizer_updates")}))


if __name__ == "__main__":
    main()

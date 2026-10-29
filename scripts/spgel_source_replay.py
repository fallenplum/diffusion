"""Actually reload all planned source states; never refit a model or optimizer."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.spgel.training import read_json, validate_recipe, load_assets, replay_all


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/spgel/pilot_v1.json")
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--stage", choices=("pilot", "s4"), default="pilot")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    config = read_json(args.config); validate_recipe(config)
    source = load_assets(args.assets, config)
    result = replay_all(source, config, args.config, args.private, stage=args.stage, device=args.device)
    print(json.dumps({k: result[k] for k in ("status", "passed_units", "failed_units", "passed_states")}))


if __name__ == "__main__":
    main()

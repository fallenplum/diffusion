"""Source data / leakage / gradient qualification, followed by immutable LOCK."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.spgel.training import read_json, validate_recipe, load_assets, qualify_source, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/spgel/pilot_v1.json")
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    config = read_json(args.config); validate_recipe(config)
    source = load_assets(args.assets, config)
    result = qualify_source(source, config, args.config, args.private, device=args.device)
    saved_assets = args.private / "SOURCE_ASSETS.private.json"
    if saved_assets.exists():
        if read_json(saved_assets) != read_json(args.assets):
            raise ValueError("EXISTING_PRIVATE_SOURCE_ASSET_POINTERS_CHANGED")
    else:
        write_json(saved_assets, read_json(args.assets))
    print(json.dumps({k: result[k] for k in ("status", "contract_sha256", "new_optimizer_fits", "optimizer_updates")}))


if __name__ == "__main__":
    main()

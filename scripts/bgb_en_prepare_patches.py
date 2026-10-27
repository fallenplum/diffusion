"""Create outcome-blind patch tensors from original immutable H5 assets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.bgb_en.data import load_raw, prepare_native_patch_cache, prepare_uni2_patch_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-json', required=True)
    parser.add_argument('--encoder', choices=('CONCH', 'UNI2'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--selector', choices=('GRID', 'DIVERSE', 'RAND', 'REGIONS'), default='GRID')
    parser.add_argument('--seed', type=int, default=270003)
    args = parser.parse_args()
    assets = json.loads(Path(args.assets_json).read_text(encoding='utf-8-sig'))
    if args.encoder == 'CONCH':
        receipt = prepare_native_patch_cache(assets['conch_inventory'], args.output, budget=args.budget,
                    selector=args.selector, seed=args.seed, root_override=assets.get('conch_root'))
    else:
        raw = load_raw(assets, include_endpoints=False)
        receipt = prepare_uni2_patch_cache(assets['uni2_root'], raw['patient_ids'], args.output,
                    budget=args.budget, selector=args.selector, seed=args.seed)
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

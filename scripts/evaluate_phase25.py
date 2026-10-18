"""Evaluate only six complete Phase25 prototype cells; no new fitting."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase25_local.evaluation import evaluate_stage, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prototype',), required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--private-root', type=Path, default=Path('/root/autodl-tmp/phase25_private/b'))
    parser.add_argument('--report-dir', type=Path, default=ROOT/'reports/phase25_b')
    args = parser.parse_args()
    from src.phase25_local import runtime
    if runtime.source() != args.source:
        raise ValueError('current training source differs from requested identity')
    config = runtime.cfg()
    if (config['prototype']['seeds'] != [31, 47] or config['prototype']['folds'] != [1, 2, 3]
            or config['prototype']['model_slots'] != 24 or config['training']['epochs'] != 80):
        raise ValueError('prototype matrix or budget changed')
    holder = {}

    def endpoints():
        from src.phase13_rna import data as parent
        holder['data'] = parent.load()
        return parent.endpoints(holder['data'])

    def partitions(fold):
        from src.phase13_rna import data as parent
        part = parent.splits(holder['data'], 'DSS', fold)
        return {'fit': part['inner_train'], 'valid': part['inner_val']}

    def baselines(fold):
        import json
        import torch
        from src.phase25_local.data import validate_prepared
        path = args.private_root/'prepared'/f'{args.stage}_fold{fold}.pt'
        receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))
        if (receipt.get('status') != 'COMPLETE' or receipt.get('source') != args.source
                or receipt.get('stage') != args.stage or receipt.get('fold') != fold
                or receipt.get('sha256') != sha256_file(path)
                or receipt.get('convex_fits') != 5 or receipt.get('shared_across_neural_seeds') is not True):
            raise ValueError('independent prepared baseline receipt/hash/source differs')
        saved = torch.load(path, map_location='cpu', weights_only=False)
        validate_prepared(saved, args.source)
        if saved.get('stage') != args.stage or saved.get('fold') != fold:
            raise ValueError('independent prepared baseline partition differs')
        return {key: saved[key] for key in ('fit', 'valid', 'R0_valid', 'CP_valid')}

    result, gate = evaluate_stage(args.private_root, args.report_dir, args.stage, args.source,
                                  endpoints, partition_loader=partitions, baseline_loader=baselines)
    print('PHASE25_PROTOTYPE_EVALUATION', result['status'], gate['classification'], flush=True)


if __name__ == '__main__':
    main()

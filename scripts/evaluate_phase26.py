"""Evaluate all six qualified Phase26 prototype cells once; never fit or launch outer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase26_multiquery.evaluation import evaluate_stage, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prototype',), required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    from src.phase26_multiquery import data, runtime
    if runtime.source() != args.source:
        raise ValueError('current training source differs from requested identity')
    config = runtime.cfg()
    if (config['prototype'] != {'folds': [1, 2, 3], 'seeds': [31, 47], 'model_slots': 24}
            or config['training']['epochs'] != 80 or config['arms'] != ['M0', 'QUERY', 'ATTN', 'COMP']):
        raise ValueError('locked prototype matrix or budget changed')

    def baselines(fold):
        loaded = data.read_prepared(fold)
        saved, receipt = loaded if isinstance(loaded, tuple) else (loaded, {})
        if saved.get('source') != args.source or saved.get('fold') != fold or saved.get('stage') != 'prototype':
            raise ValueError('independent prepared main source/partition differs')
        result = {key: saved[key] for key in ('fit', 'valid', 'CP_valid', 'RNA_valid')}
        key = 'M0_valid' if 'M0_valid' in saved else 'R0_valid'
        result['M0_valid'] = saved[key]
        result['_fit_offset'] = np.asarray(saved['base'])[np.asarray(saved['fit'], int)]
        if receipt:
            if receipt.get('source') != args.source or receipt.get('status') != 'COMPLETE':
                raise ValueError('independent prepared receipt source/status differs')
            result['_prepared_sha256'] = receipt['sha256']
        return result

    def partitions(fold):
        from src.phase13_rna import data as parent
        path = parent.PARENT/'formal/manifests'/f'R0_128_seed31_fold{fold}.json'
        saved = json.loads(path.read_text())
        return {'fit': saved['inner_train'], 'valid': saved['inner_val']}

    def endpoints():
        from src.phase13_rna import data as parent
        registry = json.loads((parent.REPORT/'SOURCE_REGISTRY.json').read_text())
        endpoint_path = parent.PRIVATE/'endpoints.npz'
        if sha256_file(endpoint_path) != registry['endpoints_npz_sha256']:
            raise ValueError('original registered endpoint bytes changed')
        with np.load(parent.DATA, allow_pickle=True) as saved:
            ids = np.array(saved['patient_ids'])
        if hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest() != \
                '76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f':
            raise ValueError('canonical patient identity changed')
        with np.load(endpoint_path) as saved:
            if str(saved['patient_hash']) != parent.ahash(ids):
                raise ValueError('DSS patient order differs')
            return {'DSS': {key: np.array(saved['DSS_'+key]) for key in ('time', 'event', 'valid')}}

    result, gate = evaluate_stage(runtime.PRIVATE, runtime.REPORT, args.stage, args.source, endpoints,
                                  baseline_loader=baselines, partition_loader=partitions)
    if runtime.source() != args.source:
        raise ValueError('training source changed during evaluation')
    print('PHASE26_PROTOTYPE_EVALUATION', result['status'], gate['decision'], flush=True)


if __name__ == '__main__':
    main()

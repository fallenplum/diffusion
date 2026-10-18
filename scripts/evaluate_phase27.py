"""Once-only Phase27 saved-head evaluation; no fit, forward, or automatic outer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase27_readout.evaluation import LEGACY_SOURCE, evaluate_stage, require, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prototype',), required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    from src.phase27_readout import runtime
    from src.phase26_multiquery import data as legacy_data, runtime as legacy_runtime
    require(runtime.source() == args.source, 'requested Phase27 source changed')
    require(legacy_runtime.source() == LEGACY_SOURCE, 'frozen Phase26 source changed')
    lock = runtime.cfg()
    require(lock['folds'] == [1, 2, 3] and lock['seeds'] == [31, 47]
            and lock['arms'] == ['QUERY', 'ATTN', 'COMP'] and lock['readout']['rank'] == 16
            and lock['readout']['lambda'] == 10, 'Phase27 readout lock changed')

    def provenance(seed, fold):
        directory = legacy_runtime.PRIVATE/'prototype'/f'seed{seed}_fold{fold}'
        receipt_path, checkpoint_path = directory/'DONE_PRIVATE.json', directory/'checkpoint.pt'
        receipt = json.loads(receipt_path.read_text())
        require(all(receipt.get(k) == v for k, v in (
            ('status', 'COMPLETE'), ('source', LEGACY_SOURCE), ('stage', 'prototype'), ('seed', seed), ('fold', fold),
            ('model_slots', 4), ('neural_models', 3), ('epochs', 80), ('optimizer_steps', 240))),
            'frozen encoder DONE contract differs')
        old_sha = sha256_file(checkpoint_path)
        require(old_sha == receipt.get('checkpoint_sha256'), 'frozen encoder checkpoint SHA differs')
        path = runtime.PRIVATE/'features'/f'seed{seed}_fold{fold}.pt'
        feature_receipt_path = path.with_suffix('.json')
        feature = json.loads(feature_receipt_path.read_text())
        require(all(feature.get(k) == v for k, v in (('status', 'COMPLETE'), ('source', args.source),
            ('seed', seed), ('fold', fold), ('legacy_checkpoint_sha256', old_sha)))
            and feature.get('sha256') == sha256_file(path), 'frozen feature receipt/SHA differs')
        # This is a read of existing private saved risk vectors, never a forward.
        old = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        require(old.get('source') == LEGACY_SOURCE and old.get('seed') == seed and old.get('fold') == fold,
                'frozen encoder record identity differs')
        return {'legacy_source': LEGACY_SOURCE, 'legacy_checkpoint_sha256': old_sha,
                'prepared_sha256': feature['prepared_sha256'],
                'feature_receipt_sha256': sha256_file(feature_receipt_path),
                'feature_sha256': feature['sha256'],
                'old_risks': {arm: old['arms'][arm]['risk'] for arm in ('QUERY', 'ATTN', 'COMP')}}

    def baselines(fold):
        saved, receipt = legacy_data.read_prepared(fold)
        require(saved.get('source') == LEGACY_SOURCE and saved.get('fold') == fold,
                'frozen prepared source/fold differs')
        return {key: saved[key] for key in ('fit', 'valid', 'M0_valid', 'CP_valid', 'RNA_valid')} | {
            'prepared_sha256': receipt['sha256']}

    def partitions(fold):
        from src.phase13_rna import data as parent
        saved = json.loads((parent.PARENT/'formal/manifests'/f'R0_128_seed31_fold{fold}.json').read_text())
        return {'fit': saved['inner_train'], 'valid': saved['inner_val']}

    def endpoints():
        from src.phase13_rna import data as parent
        registry = json.loads((parent.REPORT/'SOURCE_REGISTRY.json').read_text())
        endpoint_path = parent.PRIVATE/'endpoints.npz'
        require(sha256_file(endpoint_path) == registry['endpoints_npz_sha256'], 'original endpoint bytes changed')
        with np.load(parent.DATA, allow_pickle=True) as saved:
            ids = np.array(saved['patient_ids'])
        require(hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest() ==
            '76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f', 'canonical patient identity changed')
        with np.load(endpoint_path) as saved:
            require(str(saved['patient_hash']) == parent.ahash(ids), 'DSS patient order differs')
            return {'DSS': {key: np.array(saved['DSS_'+key]) for key in ('time', 'event', 'valid')}}

    result, gate = evaluate_stage(runtime.PRIVATE, runtime.REPORT, args.stage, args.source, endpoints,
                                  provenance_loader=provenance, baseline_loader=baselines, partition_loader=partitions)
    require(runtime.source() == args.source, 'Phase27 source changed during evaluation')
    print('PHASE27_PROTOTYPE_EVALUATION', result['status'], gate['decision'], flush=True)


if __name__ == '__main__': main()

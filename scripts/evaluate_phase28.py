"""Once-only Phase28 DSS evaluation after thirty fits and independent replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prototype',), required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--execute-evaluation', action='store_true')
    args = parser.parse_args()
    from src.phase28_reread import runtime, data
    from src.phase28_reread.evaluation import ARMS, CONTROLS, evaluate_stage, require, sha256_file
    require(args.execute_evaluation, 'explicit --execute-evaluation is required after replay authorization')

    def source_guard():
        require(runtime.source() == args.source, 'requested Phase28 training source changed')
        runtime.verify_qualifications(args.source)

    source_guard()
    lock = runtime.cfg()
    require(lock['stage'] == 'prototype' and lock['folds'] == [1, 2, 3] and lock['seeds'] == [31, 47]
            and tuple(lock['arms']) == ARMS and tuple(lock['controls']) == CONTROLS,
            'fixed prototype matrix/control lock changed')
    require(lock['gate']['primary'] == ['C11-C00', 'C11-DIRECT']
            and lock['gate']['matched_controls'] == ['GENE_MEAN', 'GENE_WRONG', 'PATH_MEAN',
                'PATH_WRONG', 'BOTH_WRONG', 'CONSTANT_VALUES'], 'Phase28 admission definitions changed')

    def baselines(fold):
        saved = data.read_frozen_main(fold)
        return {key: saved[key] for key in ('fit', 'valid', 'M0_valid', 'CP_valid', 'RNA_valid', 'prepared_sha256')}

    def partitions(fold):
        from src.phase13_rna import data as parent
        saved = json.loads((parent.PARENT/'formal/manifests'/f'R0_128_seed31_fold{fold}.json').read_text())
        return {'fit': saved['inner_train'], 'valid': saved['inner_val']}

    def endpoints():
        from src.phase13_rna import data as parent
        registry = json.loads((parent.REPORT/'SOURCE_REGISTRY.json').read_text())
        endpoint_path = parent.PRIVATE/'endpoints.npz'
        require(sha256_file(endpoint_path) == registry['endpoints_npz_sha256'], 'original endpoint bytes changed')
        # The already audited loader accesses canonical features only, including
        # patient IDs. It never loads central DSS fields.
        ids = data.read_feature_npz()['patient_ids'].astype(str)
        require(hashlib.sha256('\n'.join(ids).encode()).hexdigest() == data.PATIENT_SHA256,
                'canonical patient identity changed')
        with np.load(endpoint_path) as saved:
            require(str(saved['patient_hash']) == parent.ahash(ids), 'DSS patient order differs')
            return {'DSS': {key: np.array(saved['DSS_'+key]) for key in ('time', 'event', 'valid')}}

    result, gate = evaluate_stage(runtime.PRIVATE, runtime.REPORT, args.stage, args.source, endpoints,
        source_files=runtime.source_files(), baseline_loader=baselines, partition_loader=partitions,
        source_guard=source_guard, project_root=ROOT)
    print('PHASE28_PROTOTYPE_EVALUATION', result['status'], gate['decision'], flush=True)


if __name__ == '__main__':
    main()

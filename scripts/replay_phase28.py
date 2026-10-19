"""Explicit all-30-model inference replay; no training or heldout endpoints."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('prototype',), required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--execute-replay', action='store_true')
    args = parser.parse_args()
    if not args.execute_replay:
        raise ValueError('explicit --execute-replay is required after all thirty fits')
    from src.phase28_reread.replay import run_replay
    report = run_replay(args.source)
    print('PHASE28_REPLAY', report['status'], report['neural_models'], report['maximum_absolute_risk_error'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        from src.phase28_reread import runtime
        failure = runtime.PRIVATE/'REPLAY_FAILED_PRIVATE.json'
        if not failure.exists():
            runtime.atomic_json(failure, {'status': 'FAILED_NOT_RETRIED', 'error_type': type(error).__name__,
                'error_message_private': str(error), 'script_sha256': runtime.sha(Path(__file__))})
        print('PHASE28_REPLAY_FAILED', type(error).__name__, flush=True)
        raise SystemExit(1)

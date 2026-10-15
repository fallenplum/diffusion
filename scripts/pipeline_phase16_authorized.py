"""Resume the authorized Phase16 pipeline against its private prerequisites."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase16_semantic.core import MODEL, base, sha


def run(script, env, extra=()):
    print('START', script, *extra, flush=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *extra],
                   cwd=ROOT, env=env, check=True)
    print('FINISHED', script, flush=True)


def main(publish=False):
    if sys.platform != 'linux':
        raise RuntimeError('Run on the authorized Linux GPU server with private data mounted.')
    env = os.environ.copy()
    env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OPENBLAS_NUM_THREADS='2',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    os.environ.update(env)
    prerequisites = [base.P / 'ASSET_INVENTORY.json', base.P / 'INPUT_PINS.json',
                     base.parent.DATA, base.parent.PRIVATE / 'endpoints.npz',
                     MODEL / 'DOWNLOAD_COMPLETE.json']
    for path in prerequisites:
        if not path.is_file():
            raise FileNotFoundError('Private prerequisite unavailable: ' + str(path))
    for row in json.loads((MODEL / 'DOWNLOAD_COMPLETE.json').read_text())['files']:
        path = MODEL / row['file']
        if not path.is_file() or path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Official model file checksum mismatch: ' + row['file'])
    old = json.loads((base.R / 'ARTIFACT_MANIFEST.json').read_text())
    for row in old['public_files']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Previous batch changed: ' + row['path'])
    cache = Path('/dev/shm/phase16_routing_full.npy')
    if not cache.exists():
        # Rebuild identical private inputs without rewriting historical reports.
        spec = importlib.util.spec_from_file_location('phase16_private_cache_rebuild',
                                                       ROOT / 'scripts/prepare_phase16.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.R = base.P / 'cache_rebuild_report'
        module.R.mkdir(exist_ok=True)
        module.main()
    run('pin_phase16_inputs.py', env)
    steps = [
        ('execute_phase16_semantic.py', ()),
        ('validate_phase16_semantic_independent.py', ()),
        ('audit_phase16_semantic_retrieval.py', ()),
        ('replay_phase16_semantic_queries.py', ()),
        ('ablate_phase16_semantic_inference.py', ()),
        ('launch_phase16_omics.py', ('--pilot',)),
        ('launch_phase16_omics.py', ()),
        ('validate_phase16_omics_independent.py', ()),
        ('launch_phase16_mil_two_stage.py', ()),
        ('validate_phase16_mil_two_stage.py', ()),
        ('compare_phase16_direct.py', ()),
        ('finalize_phase16_tables.py', ()),
        ('audit_phase16_mechanisms.py', ()),
        ('report_phase16_authorized.py', ()),
    ]
    for script, args in steps:
        run(script, env, args)
    if publish:
        run('publish_phase16_authorized.py', env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    main(parser.parse_args().publish)

"""Bind a pretraining patch cache to hops inferred from the original full XY.

This qualification changes metadata and adds a hop array; feature, selection,
mask and patient-ownership arrays remain byte-identical. It must run before
model fitting. Public metadata uses an explicit geometric attribute allowlist.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from src.bgb_en.data import PUBLIC_GEOMETRY_KEYS, coordinate_step, native_conch_catalog, require, sha256


def sanitize(value):
    if isinstance(value, dict):
        return {key: ({name: entries for name, entries in item.items() if name in PUBLIC_GEOMETRY_KEYS}
                      if key == 'coordinate_attributes' else sanitize(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--audit', help='Optional own initial R0 report to sanitize before first publication')
    args = parser.parse_args()
    directory = Path(args.cache)
    path = directory/'READY.json'
    original_sha = sha256(path)
    ready = sanitize(json.loads(path.read_text()))
    previous = dict(ready['array_sha256'])
    rows = native_conch_catalog(args.inventory)
    steps = []
    for row in rows:
        with h5py.File(row['path'], 'r') as handle:
            xy = np.asarray(handle['coords']).reshape(-1, 2)
            steps.append(coordinate_step(xy))
    steps = np.asarray(steps, np.int64)
    np.save(directory/'coordinate_step.npy', steps)
    ready['array_sha256']['coordinate_step'] = sha256(directory/'coordinate_step.npy')
    ready['coordinate_step_counts'] = {str(int(step)): int((steps == step).sum()) for step in np.unique(steps)}
    ready['coordinate_step_source'] = 'GCD of all positive native distinct XY differences, before selection'
    for name, digest in previous.items():
        require(sha256(directory/(name+'.npy')) == digest, 'original prepared feature selection bytes changed')
    text = json.dumps(ready, indent=2, allow_nan=False)+'\n'
    require('TCGA-' not in text and '/root/' not in text and '/lustre/' not in text, 'private H5 attribute escaped metadata whitelist')
    path.write_text(text, encoding='utf-8')
    qualification = {'status': 'PASS_BEFORE_MODEL_FITS', 'original_READY_sha256': original_sha,
                     'READY_sha256': sha256(path), 'all_previous_array_hashes_unchanged': True,
                     'added_array': 'coordinate_step.npy', 'coordinate_step_counts': ready['coordinate_step_counts'],
                     'geometry_attributes_public_allowlist': sorted(PUBLIC_GEOMETRY_KEYS)}
    (directory/'GEOMETRY_QUALIFICATION.json').write_text(json.dumps(qualification,indent=2)+'\n',encoding='utf-8')
    if args.audit:
        audit_path = Path(args.audit)
        audit = sanitize(json.loads(audit_path.read_text()))
        audit['new_feature_only_cache'] = ready
        audit['CONCH']['coordinate_step_counts'] = ready['coordinate_step_counts']
        audit['CONCH']['coordinate_step_source'] = ready['coordinate_step_source']
        text = json.dumps(audit, indent=2, allow_nan=False)+'\n'
        require('TCGA-' not in text and '/root/' not in text and '/lustre/' not in text, 'private fields escaped public R0 audit')
        audit_path.write_text(text, encoding='utf-8')
    print(json.dumps(qualification), flush=True)


if __name__ == '__main__':
    main()

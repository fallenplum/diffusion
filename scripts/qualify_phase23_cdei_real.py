"""Real-data, heldout-poisoning qualification; no model/array persistence.

Run only after final source/configuration freezing. This performs exactly two
two-step numerical warmups and sixteen conditional Cox solves. Neither warmup
is a formal experiment. Only aggregate qualification counters, scalar losses,
timings, memory and hashes are written into the protected private directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def tree_hash(value):
    """Hash private nested numeric state without exporting its contents."""
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(b'TENSOR')
            h.update(str(tensor.dtype).encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError('object arrays are not qualification numeric state')
            array = np.ascontiguousarray(item)
            h.update(b'ARRAY'); h.update(str(array.dtype).encode())
            h.update(str(tuple(array.shape)).encode()); h.update(array.tobytes())
        elif isinstance(item, dict):
            h.update(b'DICT{')
            for key in sorted(item):
                if not isinstance(key, str):
                    raise ValueError('qualification state keys must be strings')
                h.update(json.dumps(key).encode()); visit(item[key])
            h.update(b'}')
        elif isinstance(item, (list, tuple)):
            h.update(b'LIST[')
            for child in item: visit(child)
            h.update(b']')
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None or isinstance(item, (str, int, float, bool)):
            h.update(b'SCALAR'); h.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
        else:
            raise TypeError('unsupported qualification state type: ' + type(item).__name__)
    visit(value)
    return h.hexdigest()


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


@contextmanager
def poison_nonfit_dss_endpoints(fit, cohort_size):
    """Change the actual legacy endpoint-loader output, never disk artifacts."""
    from src.phase13_rna import data as endpoint_data
    original = endpoint_data.endpoints
    nonfit = np.setdiff1d(np.arange(cohort_size), np.asarray(fit), assume_unique=True)
    calls = {'loader_calls': 0, 'poisoned_patients': int(len(nonfit))}
    def changed(raw):
        result = {name: {key: np.array(value, copy=True) for key, value in record.items()}
                  for name, record in original(raw).items()}
        endpoint = result['DSS']
        endpoint['time'][nonfit] = np.arange(len(nonfit), dtype=float) + 170013.
        endpoint['event'][nonfit] = np.arange(len(nonfit)) % 2
        # The observed endpoint-validity mask and all patient partitions remain
        # unchanged. PFI is not used to train, select or fit anything here.
        calls['loader_calls'] += 1
        return result
    with patch.object(endpoint_data, 'endpoints', changed):
        yield calls


def install_label_barriers(ds):
    """Only exact fit-order supervision can reach either training stage."""
    original = ds.fit_labels
    expected_fit = np.array(ds.fit, copy=True)
    expected = tree_hash(original())
    counts = {'fit_label_calls': 0, 'selection_label_calls': 0}
    def fit_only():
        if not np.array_equal(ds.fit, expected_fit):
            raise ValueError('qualification fit indices changed')
        labels = original()
        if set(labels) != {'time', 'event'} or any(np.asarray(v).shape != (len(expected_fit),)
                                                  for v in labels.values()):
            raise ValueError('fit accessor returned non-fit supervision')
        if tree_hash(labels) != expected:
            raise ValueError('fit supervision changed')
        counts['fit_label_calls'] += 1
        return labels
    def forbidden_selection():
        counts['selection_label_calls'] += 1
        raise AssertionError('qualification training accessed heldout selection labels')
    ds.fit_labels = fit_only
    ds.selection_labels = forbidden_selection
    return counts, expected


def component_hashes(backbone, warmup_head, payload, reference_state, warmup_audit):
    from src.phase23_cdei.model import module_fingerprint
    arms = payload['arms']
    required = ('weights', 'risk', 'fit_risk', 'fit_features', 'valid_features', 'scaler')
    if len(arms) != 8 or any(any(key not in arm for key in required) for arm in arms.values()):
        raise ValueError('all eight complete readouts are required for qualification')
    history = [{key: value for key, value in row.items() if key != 'epoch_seconds'}
               for row in warmup_audit['history']]
    return {
        'kernel_sha256': module_fingerprint(backbone),
        'warmup_head_sha256': tree_hash(warmup_head),
        'reference_state_sha256': tree_hash(reference_state),
        'features_sha256': tree_hash({arm: {key: row[key] for key in ('fit_features', 'valid_features')}
                                     for arm, row in arms.items()}),
        'scalers_sha256': tree_hash({arm: row['scaler'] for arm, row in arms.items()}),
        'readout_weights_sha256': tree_hash({arm: row['weights'] for arm, row in arms.items()}),
        'fit_risks_sha256': tree_hash({arm: row['fit_risk'] for arm, row in arms.items()}),
        'heldout_risks_sha256': tree_hash({arm: row['risk'] for arm, row in arms.items()}),
        'sensitivity_risks_sha256': tree_hash(payload.get('sensitivities', {})),
        'frozen_offsets_sha256': tree_hash({key: payload[key] for key in ('anchor_fit', 'anchor_valid')}),
        'training_losses_and_gradients_sha256': tree_hash(history),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--expected-source')
    args = parser.parse_args()
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise ValueError('real-data qualification requires an available server CUDA device')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('qualification requires the fixed deterministic CUBLAS workspace')
    from src.phase23_cdei import runtime, train
    from src.phase23_cdei.data import Data
    from src.phase23_cdei.warmup import train_kernel
    identity = runtime.source()
    config = runtime.cfg()
    if args.expected_source is not None and args.expected_source != identity:
        raise ValueError('final training source differs from the requested qualification source')
    private = getattr(runtime, 'PRIVATE', None)
    if private is None:
        private = runtime.P
    output = Path(private) / 'REAL_DATA_QUALIFICATION.json'
    if output.exists():
        old = json.loads(output.read_text(encoding='utf-8'))
        if old.get('status') == 'PASS' and old.get('source') == identity:
            print('VERIFIED_REAL_DATA_QUALIFICATION', identity, flush=True)
            return
        raise ValueError('existing qualification receipt must be reviewed and preserved before any retry')
    started = time.monotonic()
    report = {'status': 'RUNNING', 'source': identity,
              'configuration_sha256': tree_hash(config), 'fold': 1, 'seed': 31, 'role': 'inner',
              'qualification_short_fit_attempts': 0, 'qualification_short_fits_completed': 0,
              'kernel_optimizer_steps': 0, 'conditional_head_solve_attempts': 0,
              'conditional_head_solves_completed': 0, 'formal_model_fits': 0,
              'persistent_weight_or_array_writes': 0, 'runs': []}
    write_report(output, report)
    first_hashes = None
    baseline_fit = None
    baseline_selected_hash = None
    baseline_fit_label_hash = None
    cohort_size = None
    stage = 'initializing'
    solver = train.fit_conditional_cox
    def counted_solver(*values, **keywords):
        report['conditional_head_solve_attempts'] += 1
        write_report(output, report)
        answer = solver(*values, **keywords)
        report['conditional_head_solves_completed'] += 1
        write_report(output, report)
        return answer
    def stepped(kind, epoch):
        if kind != 'kernel_warmup':
            raise ValueError('unexpected qualification optimizer')
        report['kernel_optimizer_steps'] += 1
        write_report(output, report)
    try:
        for run in (0, 1):
            stage = f'load_dataset_{run+1}'
            poison = None
            if run == 0:
                ds = Data(1, 'inner', 31, device=args.device)
                baseline_fit = np.array(ds.fit, copy=True)
                cohort_size = len(ds.ft['C'])
            else:
                with poison_nonfit_dss_endpoints(baseline_fit, cohort_size) as poison:
                    ds = Data(1, 'inner', 31, device=args.device)
                if poison['loader_calls'] < 1:
                    raise ValueError('heldout poison never reached the real endpoint loader')
            selection_hash = tree_hash(ds.selection_labels())
            barrier, fit_label_hash = install_label_barriers(ds)
            if run == 0:
                baseline_selected_hash, baseline_fit_label_hash = selection_hash, fit_label_hash
                report.update(fit_patients=len(ds.fit), heldout_patients=len(ds.valid),
                              fit_WSIs=int(np.isin(ds.owner, ds.fit).sum()))
            elif selection_hash == baseline_selected_hash or fit_label_hash != baseline_fit_label_hash:
                raise ValueError('poison must change heldout labels and preserve exact fit supervision')
            if runtime.source() != identity or tree_hash(runtime.cfg()) != report['configuration_sha256']:
                raise ValueError('source/configuration changed during qualification')
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            run_start = time.monotonic()
            report['qualification_short_fit_attempts'] += 1
            write_report(output, report)
            stage = f'kernel_warmup_{run+1}'
            backbone, head, warmup_audit = train_kernel(ds, 31, 1, epochs=2, step_callback=stepped)
            if warmup_audit['optimizer_steps'] != 2 or warmup_audit['actual_epochs'] != 2:
                raise ValueError('qualification must use exactly two kernel updates per run')
            stage = f'evidence_and_heads_{run+1}'
            with patch.object(train, 'fit_conditional_cox', counted_solver):
                payload, reference_state = train.extract_and_fit(backbone, ds, identity, fit_readouts=True)
            if barrier['selection_label_calls'] != 0 or barrier['fit_label_calls'] != 2:
                raise ValueError('unexpected endpoint access in qualification stages')
            if payload.get('feature_audit', {}).get('readout_solver_calls') != 8:
                raise ValueError('qualification did not solve all eight fixed-capacity heads')
            hashes = component_hashes(backbone, head, payload, reference_state, warmup_audit)
            if first_hashes is None:
                first_hashes = hashes
            elif hashes != first_hashes:
                raise ValueError('heldout endpoint poisoning changed trained or extracted numeric state')
            torch.cuda.synchronize()
            row = {'run': run+1, 'poisoned_nonfit_DSS': run == 1,
                   'kernel_updates': warmup_audit['optimizer_steps'],
                   'conditional_heads': len(payload['arms']),
                   'fit_events': warmup_audit['fit_events'],
                   'kernel_parameters': warmup_audit['kernel_parameters'],
                   'warmup_readout_parameters': warmup_audit['readout_parameters'],
                   'fit_label_calls': barrier['fit_label_calls'],
                   'selection_label_calls': barrier['selection_label_calls'],
                   'anchor_only_fit_loss': warmup_audit['anchor_only_fit_loss'],
                   'training_curve': [{key: item[key] for key in ('epoch','training_loss','fit_loss')}
                                      for item in warmup_audit['history']],
                   'peak_GPU_allocated_bytes': int(torch.cuda.max_memory_allocated()),
                   'peak_GPU_reserved_bytes': int(torch.cuda.max_memory_reserved()),
                   'wall_seconds': float(time.monotonic()-run_start),
                   'component_hashes': hashes}
            if poison is not None:
                row['poisoned_nonfit_patient_count'] = poison['poisoned_patients']
                row['real_endpoint_loader_calls'] = poison['loader_calls']
                row['heldout_label_hash_changed'] = selection_hash != baseline_selected_hash
            report['runs'].append(row)
            report['qualification_short_fits_completed'] += 1
            write_report(output, report)
            del ds, backbone, head, payload, reference_state
            gc.collect(); torch.cuda.empty_cache()
        if runtime.source() != identity or tree_hash(runtime.cfg()) != report['configuration_sha256']:
            raise ValueError('final source/configuration verification failed')
        if (report['kernel_optimizer_steps'] != 4
                or report['conditional_head_solves_completed'] != 16
                or report['conditional_head_solve_attempts'] != 16):
            raise ValueError('actual qualification budget differs from the fixed two-run design')
        report.update(status='PASS', heldout_poisoning='REAL_LOADER_DSS_NONFIT_CHANGED_FIT_IDENTICAL',
                      numeric_state_bitwise_identical=True, source_unchanged=True,
                      wall_seconds=float(time.monotonic()-started),
                      interpretation='technical_qualification_only_no_predictive_mechanism_claim')
        write_report(output, report)
        print('REAL_DATA_QUALIFICATION_PASS', identity, flush=True)
    except BaseException as error:
        report.update(status='FAIL', failure_stage=stage, error_type=type(error).__name__,
                      wall_seconds=float(time.monotonic()-started))
        write_report(output, report)
        raise


if __name__ == '__main__':
    main()

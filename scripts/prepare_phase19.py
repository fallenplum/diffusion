"""Fail-closed prerequisites for the new attention experiment, before any OOF fit."""
import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from src.phase19_attention.data import ROOT, P, R, Data, cfg, source, source_files, write
from src.phase19_attention.train import train_cell, simple_models
from src.phase10_trimodal.data import digest


def initialize_branch():
    c = cfg()
    def git(*args):
        return subprocess.check_output(['git', '-c',
            'core.sshCommand=ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new',
            *args], cwd=ROOT, text=True).strip()
    if git('branch', '--show-current') == c['branch']:
        subprocess.check_call(['git', 'merge-base', '--is-ancestor', c['parent_commit'], 'HEAD'], cwd=ROOT)
        return
    assert git('rev-parse', 'HEAD') == c['parent_commit']
    assert not git('diff', '--name-only', c['parent_commit'])
    assert not git('branch', '--list', c['branch'])
    assert not git('ls-remote', 'origin', 'refs/heads/'+c['branch']), 'remote research branch exists; inspect rather than overwrite'
    git('switch', '-c', c['branch'], c['parent_commit'])


def run_tests():
    tests = ['attention', 'data', 'model', 'training', 'baselines']
    targets = ['tests/test_phase19_'+name+'.py' for name in tests]
    command = [sys.executable, '-m', 'pytest', '-q', *targets,
               '--basetemp', str(P/'pytest_workspace')]
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (P/'logs').mkdir(parents=True, exist_ok=True)
    (P/'logs/qualification_tests.log').write_text(done.stdout+done.stderr)
    if done.returncode:
        raise RuntimeError('Phase19 qualification tests failed: '+done.stdout[-8000:]+done.stderr[-2000:])
    write(R/'OPERATOR_AND_PIPELINE_TESTS.json', {'status': 'PASS', 'command': command,
          'stdout_summary': done.stdout[-1500:], 'training_source': source(),
          'test_files': {path: digest(ROOT/path) for path in targets}})


def prepare_data():
    rows = []
    for role in ['inner', 'outer']:
        for fold in cfg()['folds']:
            ds = Data(fold, role, 31)
            simple = simple_models(ds)
            assert len(ds.fit)+len(ds.valid) == len(ds.active)
            assert ds.audit['active_patient_patch_max'] <= 512
            if role == 'inner':
                assert not np.intersect1d(ds.active, ds.part['outer_test']).size
            for key in ds.f:
                assert np.isfinite(ds.f[key]).all(), key
            row = {'fold': fold, 'role': role, 'fit': len(ds.fit), 'validation_or_test': len(ds.valid),
                   'fit_events': int(ds.ep['event'][ds.fit].sum()),
                   'validation_or_test_events': int(ds.ep['event'][ds.valid].sum()),
                   'audit': ds.audit, 'simple_ridge_stationarity': {k: v['diagnostic'] for k, v in simple['models'].items()}}
            rows.append(row)
            print('PREPARED_DATA', role, fold, flush=True)
            del ds; gc.collect(); torch.cuda.empty_cache()
    write(R/'DATA_AUDIT.json', {'status': 'PASS', 'source': source(), 'patients': 512, 'DSS_events': 65,
          'cells': rows, 'neural_history_reused': False, 'historical_patient_neural_weights_reused': 0,
          'patient_PCA_and_genes_private': True, 'raw_image_QA': False})


def smoke_and_leakage():
    # Only a completed pair proves this check. No short-fit state/risk is saved.
    ledger_path = P/'SMOKE_LEDGER.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'attempts': []}
    def short_fit(arm, dataset=None):
        record = {'id': len(ledger['attempts'])+1, 'source': source(), 'arm': arm,
                  'status': 'RUNNING', 'generator_steps': 0, 'discriminator_steps': 0}
        ledger['attempts'].append(record); write(ledger_path, ledger)
        def stepped(kind, epoch):
            record[kind+'_steps'] += 1
            record['last_epoch'] = epoch
            write(ledger_path, ledger)
        try:
            result = train_cell(arm, 31, 1, 'inner', epochs=2, dataset=dataset, progress_callback=stepped)
        except Exception as error:
            record['status'] = 'FAILED'; record['error_type'] = type(error).__name__
            write(ledger_path, ledger)
            raise
        record['status'] = 'RETURNED_SUCCESSFULLY'
        write(ledger_path, ledger)
        return result
    original = Data(1, 'inner', 31)
    test = original.part['outer_test']
    first = short_fit('A3', original)
    # Poison the common endpoint provider before constructing the alternate
    # role, and keep it poisoned throughout fitting. Existing upstream caches
    # remain hash-locked and cannot be changed by outcome mutation.
    from src.phase19_attention.data import parent
    original_endpoints = parent.endpoints
    def poisoned(d):
        changed = {ep: {k: np.array(v, copy=True) for k, v in value.items()}
                   for ep, value in original_endpoints(d).items()}
        changed['DSS']['time'][test] = np.arange(len(test))+9137.
        changed['DSS']['event'][test] = np.arange(len(test)) % 2
        return changed
    try:
        parent.endpoints = poisoned
        alternate = Data(1, 'inner', 31)
        second = short_fit('A3', alternate)
    finally:
        parent.endpoints = original_endpoints
    assert first['history'] == second['history'] and first['selected_epoch'] == second['selected_epoch']
    errors = []
    for name, a in first['state'].items():
        b = second['state'][name]
        if a.is_floating_point():
            errors.append(float((a-b).abs().max()))
        else:
            assert torch.equal(a, b), name
    assert max(errors, default=0) == 0
    assert np.array_equal(first['risk'], second['risk'])
    write(R/'DATA_LEAKAGE_AUDIT.json', {'status': 'PASS', 'source': source(),
          'test': 'outer-test endpoint poisoning with two real 2-epoch inner A3 fits',
          'maximum_parameter_error': max(errors, default=0), 'risk_error': 0,
          'history_identical': True, 'selected_epoch_identical': True,
          'outer_labels_absent_from_inner_data_by_default': True,
          'smoke_checkpoints_or_predictions_persisted': False})
    del first, second, original, alternate; gc.collect(); torch.cuda.empty_cache()
    # Qualify the actual native cores/auxiliary optimization before their formal
    # 15-cell selection. These short fits never choose a reported architecture.
    native_summaries = []
    for arm in ['MGCT', 'CAMR_REPAIRED', 'MMP']:
        short = short_fit(arm)
        native_summaries.append({'arm': arm, 'summary': short['summary'],
                                 'finite_risk': bool(np.isfinite(short['risk']).all())})
        del short; gc.collect(); torch.cuda.empty_cache()
    write(R/'NATIVE_QUALIFICATION.json', {'status': 'PASS', 'source': source(),
          'short_fit_diagnostics': native_summaries, 'native_protocol_reproduction': False,
          'smoke_checkpoints_or_predictions_persisted': False})
    write(P/'SMOKE_COUNTS.json', {'status': 'PASS',
          'real_short_fits': sum(r['generator_steps'] > 0 for r in ledger['attempts']),
          'completed_short_fits': sum(r['status'] == 'RETURNED_SUCCESSFULLY' for r in ledger['attempts']),
          'generator_optimizer_steps': sum(r['generator_steps'] for r in ledger['attempts']),
          'discriminator_optimizer_steps': sum(r['discriminator_steps'] for r in ledger['attempts']),
          'no_saved_smoke_state_or_patient_risk': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-branch', action='store_true')
    parser.add_argument('--tests-only', action='store_true')
    args = parser.parse_args()
    P.mkdir(parents=True, exist_ok=True); R.mkdir(parents=True, exist_ok=True)
    initialize_branch()
    if args.init_branch:
        print('PHASE19_BRANCH_READY', flush=True)
        return
    from src.phase19_attention.baselines import prepare_baseline_sources
    native = prepare_baseline_sources()
    write(R/'NATIVE_SOURCE_AUDIT.json', native)
    run_tests()
    if args.tests_only:
        return
    write(P/'PREFLIGHT_DESIGN_SNAPSHOT.json', {'status': 'BEFORE_REAL_DATA_SMOKE',
          'unix_time': time.time(), 'source': source(), 'configuration': cfg()})
    prepare_data(); smoke_and_leakage()
    if (R/'PROTOCOL_LOCK.json').exists():
        assert json.loads((R/'PROTOCOL_LOCK.json').read_text())['source'] == source()
    else:
        write(R/'PROTOCOL_LOCK.json', {'status': 'LOCKED_BEFORE_FIRST_FORMAL_TRAINING',
              'unix_time': time.time(), 'source': source(), 'training_files': source_files(),
              'configuration': cfg(), 'parent_commit': cfg()['parent_commit'],
              'repeated_cohort_development': True})
    write(P/'PREPARATION_DONE.json', {'status': 'PASS', 'source': source(),
          'planned_formal_inner_fits': 150, 'planned_formal_outer_fits': 150})
    print('PHASE19_PREPARATION_PASS', flush=True)


if __name__ == '__main__':
    main()

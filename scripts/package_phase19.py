"""Fail-closed completion/privacy packaging for the finite Phase19 experiment.

This entrypoint never trains or pushes. Unresolved scientific/optimization
results may be published when the complete evidence accurately reports them.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase19_attention.data import P, R, cfg, source, source_files, write
from src.phase10_trimodal.data import digest
sys.path.insert(0, str(ROOT/'src'))
from reproducibility import protected_path_violations

BRANCH = 'research/phase19-clinical-trilinear-attention'
PARENT = '099f3d8cd0815a1f994f22b5ce5801b1f4621105'
PARENT_REPORT = 'reports/phase18a_local_state_readout'
METRICS = ['Harrell', 'Uno60']
PRIMARY = ['A3-'+a for a in ['A0', 'A1', 'A2', 'A4', 'A5', 'A6']]
MAX_BYTES = 8*1024**2
SENSITIVE_CONTENT = re.compile(
    rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|'
    rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----|'
    rb'\bENS[GT]\d{11}(?:\.\d+)?\b|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}')
DENIED_FIELDS = {
    'patient_id', 'patient_ids', 'patient_index', 'patient_indices',
    'sample_id', 'sample_ids', 'slide_id', 'slide_ids', 'barcode', 'barcodes',
    'risk', 'risks', 'risk_raw', 'raw_risk', 'patient_risk', 'anchor_risk',
    'prediction', 'predictions', 'raw_predictions', 'prediction_raw',
    'target_value', 'position_probe_target_value', 'gene_ids', 'gene_symbols',
    'selected_genes', 'selected_gene_ids', 'selected_gene_symbols',
    'centroids', 'prototype_features', 'pca_components', 'pca_state',
    'dss_time', 'dss_event', 'pfi_time', 'pfi_event', 'survival_time',
}
JSON_EXTENSIONS = {'.py', '.json', '.md', '.csv'}
SCRIPT_NAMES = {
    name+'_phase19.py' for name in
    ['prepare', 'run', 'schedule', 'start', 'validate', 'report', 'diagnose', 'package', 'publish']}
GATES = [
    'OPERATOR_AND_PIPELINE_TESTS', 'NATIVE_SOURCE_AUDIT', 'NATIVE_QUALIFICATION',
    'DATA_AUDIT', 'DATA_LEAKAGE_AUDIT', 'EVALUATION_MATH_AUDIT',
    'INDEPENDENT_VALIDATION', 'CHECKPOINT_REPLAY', 'BOOTSTRAP_AUDIT',
    'CONTRAST_POINT_AUDIT', 'FIT_AND_REFERENCE_COUNTS', 'BUDGET_VERIFICATION',
    'EXECUTION_COUNTS', 'REPORT_INPUT_AUDIT', 'MMP_TOKEN_DIAGNOSTICS', 'PROBE_COVERAGE_AUDIT',
]


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def safe_path(relative):
    posix = PurePosixPath(relative)
    require(not posix.is_absolute() and '..' not in posix.parts and '\\' not in relative,
            'noncanonical public relative path')
    path = ROOT.joinpath(*posix.parts)
    require(path.resolve().is_relative_to(ROOT.resolve()) and not path.is_symlink(),
            'public path escapes workspace or is a symlink')
    return path


def allowed_path(relative):
    p = PurePosixPath(relative)
    if p.parts[:2] == ('src', 'phase19_attention'):
        return len(p.parts) == 3 and p.suffix == '.py'
    if p.parts[:1] == ('configs',):
        return relative in {'configs/phase19_clinical_trilinear_attention.json',
                            'configs/phase19_baseline_sources.json'}
    if p.parts[:1] == ('scripts',):
        return len(p.parts) == 2 and p.name in SCRIPT_NAMES
    if p.parts[:1] == ('tests',):
        return len(p.parts) == 2 and p.name.startswith('test_phase19_') and p.suffix == '.py'
    report = PurePosixPath(R.relative_to(ROOT).as_posix())
    return p.is_relative_to(report) and p.suffix in JSON_EXTENSIONS


def _json_privacy(value, location='root'):
    if isinstance(value, dict):
        for key, item in value.items():
            require(str(key).lower() not in DENIED_FIELDS,
                    'protected JSON field: '+location+'.'+str(key))
            _json_privacy(item, location+'.'+str(key))
    elif isinstance(value, list):
        # Anonymous numeric patient vectors are still patient-level output.
        require(not (len(value) >= 128 and all(isinstance(x, (int, float, type(None))) for x in value)),
                'long raw numeric vector in public JSON: '+location)
        for item in value:
            _json_privacy(item, location+'[]')


def audit_public_file(path):
    relative = path.relative_to(ROOT).as_posix()
    require(allowed_path(relative), 'file outside explicit Phase19 allowlist: '+relative)
    safe_path(relative)
    require(path.is_file() and path.suffix in JSON_EXTENSIONS and path.stat().st_size < MAX_BYTES,
            'unsupported or oversized public artifact: '+relative)
    require(not protected_path_violations([relative]), 'protected public artifact path: '+relative)
    require(not re.search(r'(?:OOF_PRIVATE|raw_predictions|patient_predictions|selected_genes|model_state|private_cache)',
                          path.name, re.IGNORECASE), 'private artifact filename: '+relative)
    raw = path.read_bytes()
    require(not SENSITIVE_CONTENT.search(raw), 'sensitive content detected in: '+relative)
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError as error:
        raise ValueError('public artifact is not UTF-8 text: '+relative) from error
    if path.suffix == '.json':
        _json_privacy(json.loads(text))
    elif path.suffix == '.csv':
        columns = next(csv.reader(text.splitlines()), [])
        require(not {c.strip().lower() for c in columns} & DENIED_FIELDS,
                'patient-level CSV field in: '+relative)
    return {'path': relative, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _frame(name):
    path = R/(name+'.csv')
    require(path.is_file(), 'missing required aggregate table: '+name)
    return pd.read_csv(path)


def _keys(frame, fields):
    require(not frame.duplicated(fields).any(), 'duplicated aggregate cell identity')
    return set(frame[fields].itertuples(index=False, name=None))


def verify_completion():
    """Read existing evidence only; no training, new selection or report rewrite."""
    config, identity = cfg(), source()
    require(config['parent_commit'] == PARENT and config['branch'] == BRANCH, 'wrong parent/branch protocol')
    require(config['seeds'] == [31, 47, 83] and config['folds'] == [1, 2, 3, 4, 5], 'locked CV changed')
    require(config['arms'] == ['A0','A1','A2','A3','A4','A5','A6','MGCT','CAMR_REPAIRED','MMP'],
            'finite architecture/baseline matrix changed')
    require(config['bootstrap']['draws'] == 20000 and config['bootstrap']['family_tests'] == 12,
            'locked bootstrap changed')
    done, preparation, lock = read(P/'TRAINING_DONE.json'), read(P/'PREPARATION_DONE.json'), read(R/'PROTOCOL_LOCK.json')
    require(done['status'] == preparation['status'] == 'PASS', 'formal training/preparation incomplete')
    require(done['source'] == preparation['source'] == lock['source'] == identity, 'stale source/completion lock')
    require(lock['status'] == 'LOCKED_BEFORE_FIRST_FORMAL_TRAINING' and lock['configuration'] == config,
            'protocol was not locked before formal training')
    require(lock['training_files'] == source_files(), 'training-file receipts changed')
    for field, value in [('new_inner_fits',150),('new_outer_fits',150),('new_formal_fits',300),
                         ('unique_convex_reference_fits',20),('old_neural_checkpoints_used',0)]:
        require(done[field] == value, 'formal completion count mismatch: '+field)
    gates = {}
    for name in GATES:
        item = read(R/(name+'.json'))
        require(item['status'] == 'PASS', 'incomplete technical gate: '+name)
        for field in ['source', 'training_source']:
            if field in item:
                require(item[field] == identity, 'stale technical gate: '+name)
        gates[name] = item
    tests = gates['OPERATOR_AND_PIPELINE_TESTS']
    require(tests['training_source'] == identity, 'operator tests from another training source')
    for relative, expected in tests['test_files'].items():
        require(digest(safe_path(relative)) == expected, 'tested qualification source changed')
    native = gates['NATIVE_SOURCE_AUDIT']
    require(native['files_verified'] == 10 and native['manifest_sha256'] == digest(ROOT/'configs/phase19_baseline_sources.json'),
            'native source qualification incomplete')
    math_audit = gates['EVALUATION_MATH_AUDIT']
    evaluator_hash = digest(ROOT/'scripts/validate_phase19.py')
    require(math_audit['server_actual_tests'] == 5 and math_audit['failures_errors_skips'] == 0
            and math_audit['evaluator_sha256'] == evaluator_hash
            and math_audit['test_source_sha256'] == digest(ROOT/'tests/test_phase19_evaluation.py'),
            'independent evaluation tests are stale or incomplete')
    evaluated = read(P/'EVALUATION_DONE.json')
    require(evaluated['status'] == 'PASS' and evaluated['source'] == identity
            and evaluated['evaluator_sha256'] == evaluator_hash
            and evaluated['actual_new_checkpoint_replays'] == 300 and evaluated['bootstrap_draws'] == 20000,
            'complete independent evaluator pipeline receipt missing/stale')
    require(gates['DATA_AUDIT']['patients'] == 512 and gates['DATA_AUDIT']['DSS_events'] == 65,
            'DSS cohort changed')
    validation = gates['INDEPENDENT_VALIDATION']
    require(validation['model_seed_vectors'] == 51 and validation['outer_fold_metric_rows'] == 255,
            'current OOF matrix incomplete')
    require(validation['independent_sksurv_Harrell_Uno60_each_fold'] is True
            and validation['risk_ensembling'] is False and validation['primary_tests'] == 12
            and validation['old_125_inner_pilot_fits_counted_as_OOF'] is False,
            'independent metric or old-pilot exclusion gate failed')
    replay = gates['CHECKPOINT_REPLAY']
    require(replay['new_neural_checkpoints'] == 300 and replay['modes_per_checkpoint'] == 5
            and replay['maximum_error'] < 2e-6 and replay['own_main_off_maximum_error'] < 5e-6,
            'complete checkpoint/own-main replay failed')
    bootstrap = gates['BOOTSTRAP_AUDIT']
    require(bootstrap['draws'] == bootstrap['valid'] == 20000 and bootstrap['zero_denominator_draws'] == 0
            and bootstrap['primary_family_tests'] == 12 and bootstrap['risk_ensemble'] is False,
            '20k shared-patient Bonferroni12 bootstrap incomplete')
    matrices = bootstrap['shared_patient_count_matrices']
    require(len(matrices) == 5 and {row['fold'] for row in matrices} == set(config['folds'])
            and all(re.fullmatch('[0-9a-f]{64}', row['counts_sha256']) for row in matrices),
            'five shared patient-count matrix hashes required')
    require(gates['CONTRAST_POINT_AUDIT']['primary_family_tests'] == 12, 'paired statistic identity not qualified')
    coverage = gates['PROBE_COVERAGE_AUDIT']
    coverage_rows = coverage['rows']
    require(len(coverage_rows) == 20
            and {(row['arm'],row['role']) for row in coverage_rows} == {(a,r) for a in config['arms'] for r in ['inner','outer']}
            and all(row['checkpoint_count'] == 15 for row in coverage_rows)
            and coverage['additional_forward_or_gradient_calls'] == 0
            and coverage['raw_patient_indices_public'] is False,
            'distinct query probe coverage metadata incomplete')
    for row in coverage_rows:
        available = row['arm'].startswith('A')
        require(row['isolated_input_gradient_available'] == available
                and row['risk_attention_probe_per_checkpoint_patients_min'] == 32
                and row['risk_attention_probe_per_checkpoint_patients_max'] == 32
                and row['isolated_input_gradient_probe_per_checkpoint_patients_min'] == int(available)
                and row['isolated_input_gradient_probe_per_checkpoint_patients_max'] == int(available),
                'native/unavailable gradients or query counts misrepresented')
    mmp = gates['MMP_TOKEN_DIAGNOSTICS']
    require(mmp['current_source_successful_private_caches_audited'] == 10
            and mmp['all_ten_role_fold_caches_present'] is True
            and mmp['outcomes_read'] is False and mmp['parameters_or_training_source_changed'] is False
            and mmp['models_retrained'] == 0 and mmp['NaN_count'] == mmp['Inf_count'] == 0
            and mmp['CSV_sha256'] == digest(R/'MMP_TOKEN_DIAGNOSTICS.csv')
            and mmp['diagnostic_script_sha256'] == digest(ROOT/'scripts/diagnose_phase19.py'),
            'complete frozen label-blind MMP token audit missing/stale')
    counts = gates['FIT_AND_REFERENCE_COUNTS']
    for field, value in [('new_inner_neural_fits',150),('new_outer_neural_fits',150),('new_unique_convex_models',20),
            ('convex_seed_role_metric_references',60),('historical_LOCAL_outer_references',15),
            ('historical_pilot_125_new_fit_or_OOF_count',0)]:
        require(counts[field] == value, 'independent fit/reference counts differ: '+field)
    manifest, compute = _frame('CELL_MANIFEST'), _frame('COMPUTE_PROFILE')
    neural = manifest[manifest.fit_classification == 'new_neural_fit'].copy()
    convex = manifest[manifest.fit_classification == 'new_unique_convex_fit'].copy()
    require(len(manifest) == 320 and len(neural) == 300 and len(convex) == 20, 'formal manifest incomplete')
    neural['seed'] = neural.seed.astype(int)
    expected = {(a,s,f,r) for a in config['arms'] for s in config['seeds']
                for f in config['folds'] for r in ['inner','outer']}
    require(_keys(neural, ['arm','seed','fold','role']) == expected
            and _keys(compute, ['arm','seed','fold','role']) == expected, 'formal neural identities incomplete')
    require(_keys(convex, ['arm','fold','role']) == {(a,f,r) for a in ['J0','J_CS']
            for f in config['folds'] for r in ['inner','outer']}, 'unique convex identities incomplete')
    require(set(manifest.source) == {identity}, 'formal manifest source changed')
    selection = read(R/'EPOCH_SELECTION_LOCK.json')
    require(selection['status'] == 'ALL150_INNER_SELECTIONS_LOCKED' and selection['source'] == identity
            and selection['no_outer_scores_used'] is True, 'inner-only epoch selection not locked')
    selected = pd.DataFrame(selection['selections'])
    require(_keys(selected, ['arm','seed','fold']) == {(a,s,f) for a in config['arms']
            for s in config['seeds'] for f in config['folds']}, 'inner epoch lock incomplete')
    for row in selected.itertuples(index=False):
        matching = neural[(neural.arm == row.arm) & (neural.seed == row.seed) & (neural.fold == row.fold)]
        require(len(matching) == 2 and set(matching.selected_epoch) == {row.epoch}, 'fresh outer budget differs from inner')
        inner = matching[matching.role == 'inner'].iloc[0]
        require(inner.checkpoint_sha256 == row.checkpoint_sha256, 'inner selection checkpoint identity changed')
    _verify_metrics_and_decision(config, identity, gates['BUDGET_VERIFICATION'])
    _verify_execution_counts(config, identity, done, gates['EXECUTION_COUNTS'], compute)
    report = gates['REPORT_INPUT_AUDIT']
    require(report['report_script_sha256'] == digest(ROOT/'scripts/report_phase19.py')
            and report['no_patient_level_output'] is True and report['no_new_training_or_selection'] is True,
            'report script/review scope changed')
    for name, expected_hash in report['input_hashes'].items():
        require('/' not in name and '\\' not in name and digest(R/name) == expected_hash, 'report input changed: '+name)
    for name, expected_hash in report['private_ledgers_sha256'].items():
        require('/' not in name and '\\' not in name and digest(P/name) == expected_hash, 'private ledger changed: '+name)
    for name in ['RESULTS.md','DECISION.md','PROTOCOL.md','FAILURE_LOG.md','REPRODUCE.md',
                 'LITERATURE_AND_BASELINE_AUDIT.md','BASELINE_SOURCE_REGISTRY.csv']:
        require((R/name).is_file() and (R/name).stat().st_size > 0, 'missing required report: '+name)
    return {'status':'PASS', 'source':identity, 'formal_new_neural_fits':300,
            'unique_convex_reference_fits':20, 'primary_family_tests':12, 'bootstrap_draws':20000,
            'scientific_decision':read(R/'decision.json')['status'],
            'private_training_done_sha256':digest(P/'TRAINING_DONE.json'),
            'private_evaluation_done_sha256':digest(P/'EVALUATION_DONE.json'),
            'private_smoke_ledger_sha256':digest(P/'SMOKE_LEDGER.json')}


def _verify_metrics_and_decision(config, identity, budget):
    folds, seeds = _frame('FOLD_METRICS'), _frame('SEED_METRICS')
    require(len(folds) == 255 and len(seeds) == 51, 'complete OOF fold/seed tables required')
    require(len(_keys(folds,['arm','seed','fold'])) == 255 and len(_keys(seeds,['arm','seed'])) == 51,
            'duplicate metric rows')
    for (arm, seed), group in folds.groupby(['arm','seed']):
        require(set(group.fold) == set(config['folds']) and group.patients.sum() == 512
                and group.events.sum() == 65, 'DSS OOF patient/event/fold coverage differs')
        row = seeds[(seeds.arm == arm) & (seeds.seed == seed)].iloc[0]
        require(abs(group.H_num.sum()/group.H_den.sum()-row.Harrell) < 1e-12
                and abs(group.U_num.sum()/group.U_den.sum()-row.Uno60) < 1e-12,
                'per-seed within-fold numerator/denominator aggregation differs')
    for prefix in ['PFI_']:
        require(len(_frame(prefix+'FOLD_METRICS')) == 255 and len(_frame(prefix+'SEED_METRICS')) == 51,
                'frozen secondary-endpoint transfer matrix incomplete')
    paired = _frame('PAIRED_COMPARISONS')
    primary = paired[paired.family == 'primary']
    require(_keys(primary,['comparison','metric']) == {(p,m) for p in PRIMARY for m in METRICS}
            and set(primary.family_tests) == {12} and set(primary.draws_requested) == {20000}
            and set(primary.draws_valid) == {20000}, 'primary paired family incomplete')
    per_seed = _frame('SEED_PAIRED_CONTRASTS')
    pseed = per_seed[per_seed.family == 'primary']
    require(_keys(pseed,['comparison','metric','seed']) == {(p,m,s) for p in PRIMARY for m in METRICS
            for s in config['seeds']}, 'primary paired seed table incomplete')
    means = seeds.groupby('arm')[METRICS].mean()
    direction = True
    corrected = []
    for row in primary.itertuples(index=False):
        a,b = row.comparison.split('-',1)
        values = pseed[(pseed.comparison == row.comparison) & (pseed.metric == row.metric)]
        for seed_row in values.itertuples(index=False):
            ar = seeds[(seeds.arm == a) & (seeds.seed == seed_row.seed)].iloc[0]
            br = seeds[(seeds.arm == b) & (seeds.seed == seed_row.seed)].iloc[0]
            require(abs(ar[row.metric]-br[row.metric]-seed_row.delta) < 1e-11,
                    'individual paired-seed contrast differs from its own predictions')
        require(abs(values.delta.mean()-row.delta) < 1e-11
                and abs(means.loc[a,row.metric]-means.loc[b,row.metric]-row.delta) < 1e-11
                and int((values.delta > 0).sum()) == row.positive_seeds,
                'paired point/seed statistic changed')
        require(np.isfinite([row.delta,row.CI95_low,row.CI95_high,row.Bonferroni_low,row.Bonferroni_high]).all()
                and row.Bonferroni_low <= row.CI95_low <= row.CI95_high <= row.Bonferroni_high,
                'invalid nested paired confidence intervals')
        direction &= row.delta > 0 and row.positive_seeds >= 2
        if row.Bonferroni_low > 0:
            corrected.append(row.comparison+':'+row.metric)
    no_harm = all(means.loc['A3',m] >= means.loc[a,m]
                  for a in config['engineering_gate']['no_mean_harm_against'] for m in METRICS)
    original = read(R/'BUDGET_AUDIT.json')
    require(original['source'] == identity and original['no_outer_results_used'] is True
            and original['status'] in ['PASS','BUDGET_UNRESOLVED'], 'preouter budget audit invalid')
    require(budget['original_preouter_status'] == original['status'], 'preouter budget status overwritten')
    rows = budget['rows']
    require({q['arm'] for q in rows} == set(config['arms']) and len(rows) == 10, 'budget arms incomplete')
    compute = _frame('COMPUTE_PROFILE')
    for row in rows:
        actual = compute[(compute.arm == row['arm']) & (compute.role == 'inner')]
        cap = float((actual.actual_epochs == config['training']['max_epochs']).mean())
        require(len(actual) == 15 and abs(row['cap_fraction']-cap) < 1e-12
                and abs(original['inner_cap_fractions'][row['arm']]-cap) < 1e-12,
                'reported inner budget differs from actual optimizer schedule')
    sufficient = {q['arm']:q['cap_fraction'] <= config['engineering_gate']['maximum_cap_fraction_per_arm'] for q in rows}
    require(all(q['inner_cells'] == 15 and q['budget_gate_pass'] == sufficient[q['arm']] for q in rows),
            'budget counts/fractions inconsistent')
    mechanism_budget = all(sufficient[a] for a in config['arms'] if a.startswith('A'))
    all_budget = all(sufficient.values())
    require(budget['mechanism_budget_sufficient'] == mechanism_budget
            and budget['all_method_budgets_sufficient'] == all_budget, 'budget summary booleans differ')
    mechanism_status = ('BUDGET_UNRESOLVED' if not mechanism_budget else
            'DIRECTIONAL_CANDIDATE' if direction and no_harm else 'MECHANISM_NOT_SUPPORTED_IN_THIS_ARCHITECTURE')
    status = mechanism_status if all_budget else 'BUDGET_UNRESOLVED'
    decision = read(R/'decision.json')
    require(decision['source'] == identity and decision['status'] == status
            and decision['mechanism_only_status'] == mechanism_status
            and decision['all_primary_direction_and_seed_gate'] == bool(direction)
            and decision['strong_reference_no_mean_harm'] == bool(no_harm)
            and decision['mechanism_budget_sufficient'] == mechanism_budget
            and decision['all_method_budgets_sufficient'] == all_budget
            and set(decision['corrected_positive_primary_statistics']) == set(corrected)
            and decision['primary_family_tests'] == 12
            and decision['independent_external_confirmation'] is False, 'decision differs from locked actual gates')


def _verify_execution_counts(config, identity, done, execution, compute):
    ledger, smoke = read(P/'SMOKE_LEDGER.json'), read(P/'SMOKE_COUNTS.json')
    attempts = ledger['attempts']
    require(attempts and len({q['id'] for q in attempts}) == len(attempts)
            and all(q['status'] != 'RUNNING' for q in attempts), 'smoke ledger incomplete/duplicated')
    actual = {
        'real_short_fits':sum(q.get('generator_steps',0) > 0 for q in attempts),
        'completed_short_fits':sum(q['status'] == 'RETURNED_SUCCESSFULLY' for q in attempts),
        'generator_optimizer_steps':sum(q.get('generator_steps',0) for q in attempts),
        'discriminator_optimizer_steps':sum(q.get('discriminator_steps',0) for q in attempts)}
    require(smoke['status'] == 'PASS' and all(smoke[k] == v for k,v in actual.items())
            and smoke['no_saved_smoke_state_or_patient_risk'] is True and done['smoke'] == smoke,
            'actual smoke steps differ from completion ledger')
    require(execution['source'] == identity and execution['new_formal_neural_fits'] == 300
            and execution['unique_current_source_convex_fits'] == 20
            and execution['smoke_attempts_recorded'] == len(attempts)
            and execution['smoke_real_short_fits_with_generator_steps'] == actual['real_short_fits']
            and execution['smoke_generator_optimizer_steps'] == actual['generator_optimizer_steps']
            and execution['smoke_discriminator_optimizer_steps'] == actual['discriminator_optimizer_steps']
            and execution['formal_generator_optimizer_steps'] == int(compute.actual_epochs.sum())
            and execution['formal_discriminator_optimizer_steps'] == int(compute.discriminator_optimizer_steps.sum())
            and execution['all_recorded_real_neural_fits'] == 300+actual['real_short_fits'],
            'reported formal/diagnostic/auxiliary counts differ')
    require(execution['MMP_fit_attempt_ledger_available'] is False
            and execution['MMP_failed_or_aborted_attempt_count'] == 'NOT_RECORDED',
            'unrecorded prototype attempts must not be invented as zero')
    current = []
    for receipt_path in sorted((P/'baseline_features').glob('MMP_*.json')):
        receipt = read(receipt_path)
        path = receipt_path.with_suffix('.pt')
        require(path.is_file() and digest(path) == receipt['sha256'], 'prototype cache hash changed')
        provenance = receipt['provenance']
        if provenance['preprocess_audit'].get('source') == identity:
            current.append((provenance['role'],provenance['fold']))
            require(provenance['parameters']['kmeans_n_init'] == 1
                    and provenance['parameters']['prototypes'] == 16, 'locked prototype budget changed')
    require(len(current) == 10 and set(current) == {(r,f) for r in ['inner','outer'] for f in config['folds']}
            and execution['verified_completed_MMP_prototype_cache_producers_current_source'] == 10,
            'ten unique successful fit-local prototype cache products required')
    # Successful cache products establish no count of unrecorded failed producer attempts.


def audit_parent_immutability():
    require(git('branch','--show-current').decode().strip() == BRANCH, 'wrong publication branch')
    subprocess.check_call(['git','merge-base','--is-ancestor',PARENT,'HEAD'], cwd=ROOT)
    tree = git('ls-tree','-r','-z',PARENT)
    parent_files = {}
    for record in tree.split(b'\0'):
        if not record:
            continue
        header, encoded_path = record.split(b'\t',1)
        mode, kind, oid = header.decode().split()
        require(kind == 'blob' and mode in ['100644','100755'], 'unsupported inherited tracked object')
        relative = encoded_path.decode('utf-8')
        path = safe_path(relative)
        require(path.is_file(), 'inherited tracked file removed: '+relative)
        raw = path.read_bytes()
        blob = b'blob '+str(len(raw)).encode()+b'\0'+raw
        require(hashlib.sha1(blob).hexdigest() == oid, 'inherited tracked bytes changed: '+relative)
        parent_files[relative] = hashlib.sha256(raw).hexdigest()
    parent = read(ROOT/PARENT_REPORT/'ARTIFACT_MANIFEST.json')
    require(parent['status'] == 'PASS' and len(parent['public_files']) == 78, 'parent artifact manifest identity differs')
    for row in parent['public_files']:
        require(parent_files.get(row['path']) == row['sha256'], 'parent public artifact changed: '+row['path'])
    old_scope = json.loads(git('show',PARENT+':'+PARENT_REPORT+'/PUBLICATION_SCOPE_AUDIT.json'))
    legacy = old_scope['unchanged_inherited_violations']
    require(len(legacy) == 5 and all(Path(q['path']).suffix.lower() == '.zip' for q in legacy),
            'legacy protected-path exception is not exactly five inherited ZIPs')
    tracked = [x.decode() for x in git('ls-files','-z').split(b'\0') if x]
    require(set(protected_path_violations(tracked)) == {q['path'] for q in legacy},
            'repository guard found new violations')
    for row in legacy:
        require(parent_files.get(row['path']) == row['sha256'], 'legacy protected artifact changed')
    return {'status':'PASS', 'parent_commit':PARENT, 'inherited_tracked_files_verified':len(parent_files),
            'parent_public_manifest_files_verified':78, 'all_inherited_tracked_bytes_unchanged':True,
            'verification':'Git blob identity plus SHA256 for each inherited tracked file'}, legacy


def publication_paths():
    files = list((ROOT/'src/phase19_attention').glob('*.py'))
    files += [ROOT/'configs/phase19_clinical_trilinear_attention.json', ROOT/'configs/phase19_baseline_sources.json']
    files += [ROOT/'scripts'/name for name in SCRIPT_NAMES]
    files += list((ROOT/'tests').glob('test_phase19_*.py'))
    files += [p for p in R.rglob('*') if p.is_file() and p.name != 'ARTIFACT_MANIFEST.json']
    require(all(p.is_file() for p in files), 'expected source/report file missing')
    return sorted(set(files), key=lambda p:p.relative_to(ROOT).as_posix())


def main():
    completion = verify_completion()
    parent, inherited = audit_parent_immutability()
    write(R/'PARENT_IMMUTABILITY_AUDIT.json', parent)
    write(R/'PUBLICATION_SCOPE_AUDIT.json', {
        'status':'PASS', 'scope':'only new Phase19 code/config and anonymous aggregate reports',
        'base_commit':PARENT, 'repository_wide_guard':'LEGACY_EXTENSION_VIOLATIONS',
        'unchanged_inherited_violations':inherited, 'new_artifacts_strict_privacy':True,
        'no_repository_wide_PASS_claim':True})
    rows = [audit_public_file(p) for p in publication_paths()]
    write(R/'ARTIFACT_MANIFEST.json', {**completion, 'public_files':rows,
        'parent_commit':PARENT, 'branch':BRANCH,
        'MMP_unique_successful_current_source_cache_products':10,
        'MMP_failed_or_retried_producer_attempt_count':'not recorded; not inferred from successful cache products',
        'private_excluded':['patient OOF and risks','trained weights','raw matrices','patient/gene identifiers',
                            'donor mappings','prototype/PCA states','foundation tokens','upstream source vendor','credentials'],
        'no_automatic_followup_training':True})
    audit_public_file(R/'ARTIFACT_MANIFEST.json')
    print('PHASE19_PACKAGE_PASS',len(rows),completion['scientific_decision'],flush=True)


if __name__ == '__main__':
    main()

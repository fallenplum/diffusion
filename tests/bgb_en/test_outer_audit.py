"""Synthetic zero-optimizer qualification for the independent outer audit."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import bgb_en_outer_audit as audit
from src.bgb_en import protocol


def test_changed_source_fails_before_checkpoint_loading_or_gpu(tmp_path, monkeypatch):
    from scripts import bgb_en_run
    config = tmp_path / 'config.json'; config.write_text('{}')
    monkeypatch.setattr(bgb_en_run, 'load_inputs', lambda args: ({}, {}, {}, None,
        dict(protocol_sha256='deliberately_wrong', source={'synthetic': 'not_a_real_source'})))
    def refuse(*args, **kwargs):
        raise AssertionError('Wrong-source audit reached weights or GPU')
    monkeypatch.setattr(torch, 'load', refuse)
    monkeypatch.setattr(torch.cuda, 'set_device', refuse)
    with pytest.raises(audit.AuditError, match='CONFIG_SHA256_MISMATCH'):
        audit.run(SimpleNamespace(config=config))


def test_beta_derivative_is_independent_and_uses_only_within_teacher_pairs():
    cp = np.zeros(6); rna = np.array([0., 0., 0., 10., 10., 10.])
    time = np.arange(1., 7.); event = np.ones(6, int); groups = np.array([0, 0, 0, 1, 1, 1])
    saved = dict(pairs=6, lambda_beta=1., loss=np.log(2.))
    result = audit.independent_beta_check(cp, rna, time, event, groups, 0., saved)
    assert result['same_teacher_pairs'] == 6  # All-cohort pairing would be 15.
    assert result['independent_gradient_abs'] == 0
    assert result['optimizer_steps'] == 0
    with pytest.raises(audit.AuditError, match='BETA_INDEPENDENT_GRADIENT'):
        audit.independent_beta_check(cp, rna, time, event, groups, .1,
            dict(saved, loss=np.log(2.) + .005))


def test_beta_analytic_derivative_matches_finite_difference_with_event_time_ties():
    cp = np.array([.1, -.4, .7, .2, .6, -.3]); rna = np.array([-.2, .5, .3, -.1, .8, -.6])
    time = np.array([1., 2., 2., 3., 4., 4.]); event = np.array([1, 0, 1, 1, 0, 1])
    groups = np.array([0, 0, 0, 1, 1, 1]); beta = .27
    left = np.array([0, 0, 2, 3, 3, 5]); right = np.array([1, 2, 1, 4, 5, 4])
    def objective(b):
        margins = cp[left] - cp[right] + b * (rna[left] - rna[right])
        return np.logaddexp(0., -margins).mean() + b * b / 2.
    result = audit.independent_beta_check(cp, rna, time, event, groups, beta,
        dict(pairs=6, lambda_beta=1., loss=objective(beta)), tolerance=10.)
    derivative = (objective(beta + 1e-5) - objective(beta - 1e-5)) / 2e-5
    assert abs(result['independent_gradient_abs'] - abs(derivative)) < 1e-10


@pytest.mark.parametrize('mutation,code', [('pair', 'BETA_PAIR_COUNT'), ('group', 'BETA_UNCROSSFITTED'),
                                         ('lambda', 'BETA_REGULARIZATION')])
def test_beta_audit_rejects_inconsistent_saved_contract(mutation, code):
    saved = dict(pairs=6, lambda_beta=1., loss=np.log(2.))
    groups = np.repeat([0, 1], 3)
    if mutation == 'pair': saved['pairs'] = 15
    if mutation == 'group': groups[0] = -1
    if mutation == 'lambda': saved['lambda_beta'] = .1
    with pytest.raises(audit.AuditError, match=code):
        audit.independent_beta_check(np.zeros(6), np.zeros(6), np.arange(6.), np.ones(6), groups, 0., saved)


def test_oof_coverage_is_patient_exact_once_not_cross_fold_ranking():
    folds = np.repeat(np.arange(1, 6), 4); pfi = np.ones(20, bool); pfi[-2:] = False
    units = [dict(kind='BASELINE', fold=f, status='PASS') for f in range(1, 6)]
    result, complete = audit.coverage_report(units, folds, pfi, ['CP_STRONG', 'CP_SCALAR'], n=20, pfi_expected=18)
    assert complete and len(result) == 6
    assert all(r['DSS_patients_covered'] == 20 and r['PFI_patients_covered'] == 18 for r in result)
    result, complete = audit.coverage_report(units, folds, pfi, ['CP_STRONG', 'CP_SCALAR', 'NESTED_INTERACTION'], n=20, pfi_expected=18)
    assert not complete
    assert all(r['coverage_status'] == 'PARTIAL' and r['DSS_patients_covered'] == 0
               for r in result if r['model'] == 'NESTED_INTERACTION')
    with pytest.raises(audit.AuditError, match='OOF_DUPLICATE_FOLD'):
        audit.coverage_report(units + [units[0]], folds, pfi, ['CP_STRONG'], n=20, pfi_expected=18)


def test_independent_metric_summary_uses_counts_before_seed_mean():
    units = []
    for fold in range(1, 6):
        metrics = []
        for seed in (31, 47, 83):
            h = .3 + .05 * fold + seed * .0001
            u = .4 + .01 * fold + seed * .0002
            metrics.append(dict(model='CP_SCALAR', seed=seed, endpoint='DSS',
                                Harrell_num=h * fold, Harrell_den=float(fold),
                                Uno_num=u * fold * fold, Uno_den=float(fold * fold)))
        units.append(dict(kind='BASELINE', fold=fold, status='PASS', independent_fold_metrics=metrics))
    summary = audit.summarize_fold_metrics(units, ['CP_SCALAR'])
    dss = next(row for row in summary if row['endpoint'] == 'DSS')
    assert dss['complete_all_five_folds_three_seeds']
    assert abs(dss['Harrell'] - np.mean([sum((.3 + .05 * f + s * .0001) * f for f in range(1, 6)) / 15 for s in (31, 47, 83)])) < 1e-15
    pfi = next(row for row in summary if row['endpoint'] == 'PFI')
    assert not pfi['complete_all_five_folds_three_seeds']
    assert 'Harrell' not in pfi


def _raw():
    rng = np.random.default_rng(27028); n, g = 96, 520
    symbols = np.array([f'SYN_GENE_{j}' for j in range(g)])
    clinical = {name: (np.arange(n) % 3).astype(str) for name in protocol.CLINICAL}
    clinical['AGE'] = (30 + np.arange(n) % 50).astype(str)
    rna = rng.normal(size=(n, g)); rna[:, -1] = np.nan
    return dict(t=np.arange(1., n + 1), e=(np.arange(n) % 3 == 0).astype(int),
                rna=rna, cna=rng.integers(-2, 3, size=(n, g)).astype(float),
                rna_symbols=symbols, cna_symbols=symbols.copy(), clinical_raw=clinical,
                S32=rng.normal(size=(n, 32)), conch_local=rng.normal(size=(n, 40)),
                conch_mean=rng.normal(size=(n, 40)))


def _teachers(raw):
    train, test = np.arange(72), np.arange(72, 96); groups = np.full(96, -1, int)
    cf = {name: np.zeros(96) for name in ('LOCAL', 'CS', 'R')}; teachers = []
    for group, held in enumerate(np.array_split(train, 3)):
        fit = np.setdiff1d(train, held)
        features = protocol.prepare_partition(raw, fit, np.ones((2, 520), bool))
        models = {}
        for name in cf:
            coefficient = np.sin(np.arange(features[name].shape[1]) + 1.) / features[name].shape[1]
            risk = features[name] @ coefficient
            scale = dict(mean=float(risk[fit].mean()), sd=float(risk[fit].std()))
            cf[name][held] = (risk[held] - scale['mean']) / scale['sd']
            models[name] = dict(coef=coefficient, scale=scale)
        teachers.append(dict(train=fit, held=held, preprocessing=features['state'], models=models))
        groups[held] = group
    full = dict(LOCAL=np.linspace(-1., 1., 96), CS=np.linspace(.2, -.3, 96), R=np.linspace(-.7, .8, 96))
    cp, rna, cs = (full[name].copy() for name in ('LOCAL', 'R', 'CS'))
    cp[train] = cf['LOCAL'][train]; rna[train] = cf['R'][train]; cs[train] = cf['CS'][train]
    calibrated = dict(teachers=teachers, teacher_groups=groups, CP_cf=cf['LOCAL'], RNA_cf=cf['R'],
                      CS_cf=cf['CS'], CP=cp, RNA=rna, CS=cs, base=cp.copy(), beta=0., scalar_audit={})
    return dict(train=train, test=test, full_risks=full, seed31_CS_risk=cs, teachers={31:calibrated})


def test_actual_teacher_coefficients_and_saved_preprocessing_replay_without_fitting(monkeypatch):
    raw = _raw(); common = _teachers(raw)
    def beta_scope(cp, rna, time, event, groups, *args, **kwargs):
        assert len(time) == 72
        assert set(time).isdisjoint(set(raw['t'][common['test']]))
        assert (groups >= 0).all()
        return dict(independent_gradient_checked_separately=True)
    monkeypatch.setattr(audit, 'independent_beta_check', beta_scope)
    def refuse(*args, **kwargs):
        raise AssertionError('Audit refitted preprocessing or optimized a parameter')
    monkeypatch.setattr(protocol, 'prepare_partition', refuse)
    monkeypatch.setattr(protocol, 'scale_fit', refuse)
    monkeypatch.setattr(torch.optim, 'Adam', refuse)
    result = audit.replay_teachers(raw, common, 31)
    assert result['teacher_Cox_models_replayed'] == 9
    assert result['maximum_crossfit_risk_error'] < 1e-12
    seed47 = copy.deepcopy(common['teachers'][31])
    for teacher in seed47['teachers']:
        del teacher['models']['CS']
    common['teachers'][47] = seed47
    result47 = audit.replay_teachers(raw, common, 47)
    assert result47['teacher_Cox_models_replayed'] == 6
    assert result47['CS_teacher_scope'] == 'NOT_FITTED_FOR_THIS_SEED_SHARED_SEED31_CS'
    bad = copy.deepcopy(common); bad['teachers'][31]['CP_cf'][0] += .01
    with pytest.raises(audit.AuditError, match='CHECKPOINT_FORWARD_RISK_MISMATCH'):
        audit.replay_teachers(raw, bad, 31)
    bad = copy.deepcopy(common); bad['teachers'][31]['teachers'][0]['train'][0] = 72
    with pytest.raises(audit.AuditError, match='TEACHER_OUTER_TEST_LEAKAGE'):
        audit.replay_teachers(raw, bad, 31)
    bad = copy.deepcopy(common); bad['teachers'][31]['teachers'][0]['preprocessing']['S32']['fit_hash'] = 'wrong'
    with pytest.raises(audit.AuditError, match='SCALER_FIT_HASH'):
        audit.replay_teachers(raw, bad, 31)


class _FakeFinalModel(torch.nn.Module):
    def __init__(self, stage, arm, prepared, membership, bank, device, gamma):
        super().__init__(); self.coef = torch.nn.Parameter(torch.tensor(-.1, dtype=torch.float64))
        self.device = device; self.query_map = np.arange(len(prepared['base']))
        self.register_buffer('x', torch.as_tensor(prepared['features']['R'][:, 0], dtype=torch.float64))
        self.register_buffer('offset', torch.as_tensor(prepared['base'], dtype=torch.float64))
    def forward(self, idx, control=None):
        delta = self.coef * self.x[idx]
        return dict(risk=self.offset[idx] + delta, delta=delta)


def test_neural_audit_loads_actual_weights_and_recounts_saved_risks(monkeypatch):
    from src.bgb_en import training
    monkeypatch.setattr(training, 'MechanismModel', _FakeFinalModel)
    def refuse(*args, **kwargs):
        raise AssertionError('Outer audit created an optimizer')
    monkeypatch.setattr(torch.optim, 'Adam', refuse)
    n = 12; train, test = np.arange(8), np.arange(8, 12)
    raw = dict(t=np.arange(1., n + 1), e=np.ones(n, int), pfi_t=np.arange(1., n + 1),
               pfi_e=np.ones(n, int), pfi_valid=np.ones(n, bool), cancer=np.repeat('SYN', n),
               platform=np.repeat('SYN', n), clinical_raw=dict(AJCC_PATHOLOGIC_TUMOR_STAGE=np.repeat('SYN', n)))
    features = dict(R=np.linspace(-1., 1., n)[:, None]); zeros = np.zeros(n)
    calibrated = dict(CP=zeros, RNA=zeros, base=zeros, beta=0.)
    common = dict(train=train, test=test, full_risks=dict(LOCAL=zeros), seed31_CS_risk=zeros,
                  teachers={31:calibrated})
    prototype = _FakeFinalModel('R3', 'F5', dict(features=features, base=zeros), None, None, 'cpu', .1)
    with torch.no_grad(): prototype.coef.fill_(.4)
    state = {k: v.clone() for k, v in prototype.state_dict().items()}
    risk = .4 * features['R'][test, 0]
    metric = audit.metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][test], raw['e'][test], risk, zeros[test])
    pfi = audit.metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][test], raw['e'][test], risk)
    controls = {}
    for name in ('wrong_query', 'mean_query', 'whole_omics', 'attention_permutation', 'interaction_off', 'direct_off'):
        cr = risk + np.linspace(-.1, .1, len(test))
        controls[name] = dict(risk=cr, metrics=audit.metrics_from_risk(raw['t'][train], raw['e'][train], raw['t'][test], raw['e'][test], risk, cr))
    receipt = dict(seed=31, fold=1, selected_arm='F5', epochs=80, contract=dict(synthetic='no_data'),
                   metrics=dict(CP_STRONG=metric, CP_SCALAR=metric), PFI_frozen_DSS_risk=pfi,
                   frozen_controls={k: v['metrics'] for k, v in controls.items()})
    artifact = dict(train=train, test=test, contract=receipt['contract'], epochs=80, optimizer_updates=80,
                    outer_evaluations_during_training=0, curves=[dict(outer_test_read=False)],
                    final_state_dict=state, risk=risk, delta=risk, metrics=receipt['metrics'],
                    PFI_frozen_DSS_risk=pfi, frozen_controls=controls)
    result = audit.audit_neural(raw, artifact, receipt, common, features, None, None, 'cpu')
    assert result['actual_state_dict_forward']
    assert result['maximum_risk_error'] < 1e-15
    assert result['independent_metric_max_error'] == 0
    assert result['frozen_controls_checked'] == 6
    bad = copy.deepcopy(artifact); bad['final_state_dict']['coef'] += .1
    with pytest.raises(audit.AuditError, match='CHECKPOINT_FORWARD_RISK_MISMATCH'):
        audit.audit_neural(raw, bad, receipt, common, features, None, None, 'cpu')
    bad = copy.deepcopy(artifact); bad['outer_evaluations_during_training'] = 1
    with pytest.raises(audit.AuditError, match='NEURAL_OUTER_READ'):
        audit.audit_neural(raw, bad, receipt, common, features, None, None, 'cpu')

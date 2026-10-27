"""Independent synthetic checks of the common scientific comparison contract."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from src.bgb_en import baselines, protocol, survival
from src.phase10_trimodal.math import log_weights, torch_cox, fit_ridge
from src.phase10_trimodal.reference import reference_efron


class FakeRiskModel(nn.Module):
    """Small deterministic nonlinear residual; no pathology/RNA data required."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.17, -.23, .09], dtype=torch.float64))
        self.register_buffer('x', torch.tensor([[.2, 1., -.2], [1., -.2, .3], [.1, .7, 1.],
                                              [-.4, -.2, 1.], [.4, .1, -.5], [1.2, .4, -.1],
                                              [.3, -.6, .2], [.1, .2, -.1]], dtype=torch.float64))
        self.register_buffer('offset', torch.linspace(-.3, .4, 8, dtype=torch.float64))

    def forward(self, idx):
        delta = .2 * torch.tanh(self.x[idx] @ self.weight)
        return {'risk': self.offset[idx] + delta, 'delta': delta}

    def regularization(self):
        return .017 * self.weight.square().sum()


def synthetic_raw():
    rng = np.random.default_rng(91)
    n, genes = 96, 640
    rna = rng.normal(size=(n, genes))
    rna[:4, 2] = np.nan
    rna[:, -1] = np.nan
    cna = rng.choice([-2., -1., 0., 1., 2.], size=(n, genes))
    cna[1:5, 4] = np.nan
    symbols = np.array([f'SYN_G{j:04d}' for j in range(genes)])
    clinical = {name: np.array([str(j % 3) for j in range(n)]) for name in protocol.CLINICAL}
    clinical['AGE'] = np.array([str(30 + j % 41) for j in range(n)])
    return {'rna': rna, 'cna': cna, 'rna_symbols': symbols, 'cna_symbols': symbols.copy(),
            'clinical_raw': clinical, 'S32': rng.normal(size=(n, 32)),
            'conch_local': rng.normal(size=(n, 48)), 'conch_mean': rng.normal(size=(n, 48)),
            't': rng.integers(1, 120, size=n).astype(float), 'e': (np.arange(n) % 3 == 0).astype(int)}


def test_full_risk_gradient_cache_matches_direct_full_batch_with_ties_and_penalty():
    cached = FakeRiskModel()
    direct = copy.deepcopy(cached)
    time = np.array([1., 2., 2., 2., 4., 6., 6., 9.])
    event = np.array([1, 1, 1, 0, 0, 1, 0, 0])
    batches = [torch.tensor([0, 1, 2]), torch.tensor([3]), torch.tensor([4, 5]), torch.tensor([6, 7])]
    opt_cache = torch.optim.SGD(cached.parameters(), lr=.03)
    opt_direct = torch.optim.SGD(direct.parameters(), lr=.03)
    record = survival.cached_full_risk_step(cached, batches, time, event, opt_cache, 'cpu', extra_weight=.13)
    opt_direct.zero_grad()
    output = direct(torch.arange(8))
    cox = torch_cox(output['risk'], log_weights(time, event, 'cpu'), torch.tensor(event))
    loss = cox + .13 * output['delta'].square().mean() + direct.regularization()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(direct.parameters(), 5.)
    opt_direct.step()
    assert abs(record['cox_loss'] - reference_efron(time, event, output['risk'].detach().numpy())) < 2e-14
    assert record['replay_max'] == 0
    assert torch.allclose(cached.weight.grad, direct.weight.grad, atol=2e-14, rtol=2e-14)
    assert torch.allclose(cached.weight, direct.weight, atol=2e-14, rtol=2e-14)


def test_non_deterministic_gradient_cache_is_rejected():
    class RandomRisk(FakeRiskModel):
        def forward(self, idx):
            out = super().forward(idx)
            out['risk'] = out['risk'] + torch.rand_like(out['risk'])
            return out
    torch.manual_seed(4)
    model = RandomRisk()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    with pytest.raises(RuntimeError, match='not deterministic'):
        survival.cached_full_risk_step(model, [torch.arange(8)], np.arange(1, 9),
                                       np.array([1, 1, 0, 1, 0, 1, 0, 0]), optimizer, 'cpu')


def test_efron_ties_extreme_logits_and_offset_ridge_match_independent_oracles():
    time = np.array([1., 2., 2., 2., 4., 5.])
    event = np.array([1, 1, 1, 0, 0, 1])
    risk = np.array([1000., 999., -999., 700., -1000., .3])
    actual = torch_cox(torch.tensor(risk), log_weights(time, event, 'cpu'), torch.tensor(event))
    assert abs(float(actual) - reference_efron(time, event, risk)) < 1e-11
    x = np.array([[.2, 1.], [1., -.1], [-.5, .3], [.1, -.7], [.3, .4], [-.1, .9]])
    ours, audit = survival.fit_offset_ridge(x, time, event, lam=.4)
    old, _ = fit_ridge(x, time, event, .4)
    assert np.allclose(ours, old, atol=2e-6, rtol=0)
    offset = np.array([.1, -.1, .2, -.2, -.1, .1])
    coef, audit = survival.fit_offset_ridge(x, time, event, lam=.4, offset=offset)
    objective = lambda w: reference_efron(time, event, offset + x @ w) + .2 * np.dot(w, w)
    finite_diff = np.array([(objective(coef + np.eye(2)[j] * 1e-5)
                            - objective(coef - np.eye(2)[j] * 1e-5)) / 2e-5 for j in range(2)])
    assert np.max(np.abs(finite_diff)) < 2e-6
    assert audit['gradient_max'] < 2e-5


def test_heldout_changes_leave_train_preprocessing_and_rna_selection_unchanged():
    raw = synthetic_raw(); changed = copy.deepcopy(raw)
    train = np.arange(72); held = np.arange(72, 96)
    member = np.ones((3, 640), bool)
    changed['rna'][held, :-1] = 9000.
    changed['cna'][held] = -800.
    changed['S32'][held] *= 10000.
    changed['conch_local'][held] *= 10000.
    changed['conch_mean'][held] *= -10000.
    for key in protocol.CLINICAL:
        changed['clinical_raw'][key][held] = '999' if key == 'AGE' else 'never_seen'
    before = protocol.prepare_partition(raw, train, member)
    after = protocol.prepare_partition(changed, train, member)
    for key in ('C', 'CS', 'LOCAL', 'GLOBAL', 'R', 'rna', 'rna_mask', 'cna_gain', 'cna_loss', 'cna_mask'):
        assert np.array_equal(before[key][train], after[key][train]), key
    for key in ('median', 'measured', 'selected'):
        assert np.array_equal(before['state']['molecular'][key], after['state']['molecular'][key]), key
    assert np.array_equal(before['state']['fit'], train)
    for name in ('LOCAL', 'GLOBAL'):
        for key in ('components', 'mean'):
            assert np.array_equal(before['state']['visual'][name][key], after['state']['visual'][name][key])


def test_cna_channels_split_signed_dosage_before_centering_and_mask_unknown_genes():
    raw = synthetic_raw(); train = np.arange(72)
    # A deterministic dosage has both gains and losses in the training rows.
    raw['cna'][:, 0] = np.resize(np.array([-2., -1., 0., 1., 2.]), 96)
    raw['cna_symbols'][-1] = 'SYN_CNA_ONLY'
    result = protocol.molecular_fit(raw, train, np.ones((2, 640), bool))
    dosage = raw['cna'][:, 0]
    for key, v in [('cna_gain', np.maximum(dosage, 0.)), ('cna_loss', np.maximum(-dosage, 0.))]:
        expected = (v - v[train].mean()) / v[train].std()
        assert np.allclose(result[key][:, 0], expected, atol=2e-7)
    assert not result['cna_mask'][:, -1].any()
    assert np.array_equal(result['cna_gain'][:, -1], np.zeros(96))
    assert np.array_equal(result['cna_loss'][:, -1], np.zeros(96))
    assert not result['rna_mask'][:, -1].any()
    assert np.isfinite(result['rna']).all()


def test_ipcw_uses_current_training_censor_distribution_and_pair_repairs_reconcile():
    # At t=2: five at risk, remove one death before one censor => G=.75;
    # at t=6: two at risk and one censor => G=.375.
    train_t = np.array([2., 2., 4., 6., 8.]); train_e = np.array([1, 0, 1, 0, 1])
    time = np.array([2., 3., 5., 7.]); event = np.ones(4, int)
    risk = np.array([3., 1., 2., 0.]); reference = np.array([1., 3., 0., 2.])
    result = survival.evaluate(train_t, train_e, time, event, risk, reference)
    weights = 1. / np.array([.75, .75, .75, .375])**2
    # Six ordered comparable pairs, weighted by the earlier event patient.
    expected_den = 3 * weights[0] + 2 * weights[1] + weights[2]
    expected_num = 3 * weights[0] + weights[1] + weights[2]
    assert abs(result['Uno_den'] - expected_den) < 1e-12
    assert abs(result['Uno_num'] - expected_num) < 1e-12
    ref = survival.evaluate(train_t, train_e, time, event, reference)
    assert abs(result['repair'] - result['damage'] - (result['Harrell'] - ref['Harrell'])) < 1e-12
    assert abs(result['Uno_repair'] - result['Uno_damage'] - (result['Uno60'] - ref['Uno60'])) < 1e-12
    changed = survival.evaluate(train_t, np.ones(5, int), time, event, risk)
    assert changed['Harrell'] == result['Harrell']
    assert changed['Uno_den'] != result['Uno_den']


def test_scalar_pair_calibration_excludes_cross_teacher_pairs():
    # Every within-teacher CP/RNA difference is zero; only cross-teacher pairs
    # carry RNA separation, so an honest same-teacher objective yields beta=0.
    cp = np.zeros(6); rna = np.array([0., 0., 0., 10., 10., 10.])
    time = np.arange(1., 7.); event = np.ones(6, int); group = np.array([0, 0, 0, 1, 1, 1])
    beta, audit = survival.fit_scalar_same_teacher(cp, rna, time, event, group)
    assert beta == 0
    assert audit['pairs'] == 6  # Not all 15 cohort pairs.


def test_teacher_excludes_both_pair_members_and_external_validation(monkeypatch):
    raw = synthetic_raw(); train = np.arange(72); valid = np.arange(72, 96)
    fit_calls = []

    def small_fit(x, time, event, lam, **kwargs):
        fit_calls.append((np.asarray(time).copy(), np.asarray(event).copy()))
        # Deterministic synthetic stand-in: exercise actual fold-local
        # preprocessing and teacher assignment without 512D optimization.
        w = np.sin(np.arange(x.shape[1]) + 1.) / np.sqrt(x.shape[1])
        return w, {'synthetic_test_only': True}

    monkeypatch.setattr(baselines, 'fit_offset_ridge', small_fit)
    prepared = baselines.prepare_anchors(raw, train, valid, np.ones((2, 640), bool))
    assert len(fit_calls) == 13
    assert prepared['upstream_optimizer_fits'] == 13
    assert np.array_equal(np.sort(np.concatenate([t['held'] for t in prepared['teachers']])), train)
    for group, teacher in enumerate(prepared['teachers']):
        teacher_train, held = teacher['train'], teacher['held']
        assert not np.intersect1d(teacher_train, held).size
        assert not np.intersect1d(teacher_train, valid).size
        assert not np.intersect1d(held, valid).size
        assert set(teacher_train) | set(held) == set(train)
        assert np.array_equal(teacher['preprocessing']['fit'], teacher_train)
        assert np.array_equal(prepared['teacher_groups'][held], np.full(len(held), group))
        for left in held:
            for right in held:
                assert left not in teacher_train and right not in teacher_train
    # Changing outer-held-out labels must not affect teacher-generation risks.
    changed = copy.deepcopy(raw); changed['t'][valid] = 500.; changed['e'][valid] = 1 - changed['e'][valid]
    again = baselines.prepare_anchors(changed, train, valid, np.ones((2, 640), bool))
    assert prepared['beta'] == again['beta']
    assert np.array_equal(prepared['base'][train], again['base'][train])


@pytest.mark.parametrize('strata', [np.array(['a', 'a', 'b', 'b', 'c']), np.arange(5).astype(str)])
def test_patient_mismatch_has_no_fixed_points_or_cross_partition_donors(strata):
    mapping, audit = protocol.patient_mismatch(np.arange(5), strata, seed=72)
    assert np.all(mapping != np.arange(5))
    assert set(mapping) == set(range(5))
    assert audit['fixed_points'] == 0


def test_aggregation_uses_fold_numerators_not_cross_fold_risks_or_average_ratios():
    rows = [dict(Harrell_num=2., Harrell_den=3., Uno_num=2., Uno_den=4.),
            dict(Harrell_num=9., Harrell_den=10., Uno_num=10., Uno_den=12.)]
    result = survival.combine_metrics(rows)
    assert result['Harrell'] == 11. / 13.
    assert result['Uno60'] == 12. / 16.

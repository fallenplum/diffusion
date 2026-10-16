"""Patient aggregation, full-risk gradients and fresh-fit checks, no real data."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.phase19_attention import data, train
from src.phase19_attention.model import AttentionModel
from src.phase17_ssd.losses import full_risk_efron


@pytest.fixture(autouse=True)
def restore_determinism_after_seed_all():
    # train_cell deliberately enables deterministic kernels.  Do not leak that
    # process-wide switch into later independent CUDA operator tests.
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def configuration():
    return {'hidden': 8, 'rank': 4, 'training': {
        'dropout': .25, 'microbatch_patients': 2, 'max_epochs': 4,
        'minimum_epoch': 1, 'early_stopping_patience': 3, 'validation_ema_alpha': .2,
        'learning_rate': .001, 'weight_decay': .01, 'main_effect_l2': 10.,
        'gradient_clip': 5.}}


def synthetic_data(role='inner', seed=31, dtype=torch.float64, device='cpu'):
    rng = np.random.default_rng(1919)
    n, width = 512, 16
    f = {k: rng.normal(size=(n, d)).astype(np.float64) for k, d in
         [('C', 3), ('S', 2), ('X', 7), ('R', width), ('G', width), ('L', width)]}
    f['CS'] = np.c_[f['C'], f['S']]
    for key in ['RM', 'CM']:
        f[key] = np.ones((n, width), float)
    member = np.zeros((8, width), bool)
    for k in range(8):
        member[k, k*2:k*2+2] = True
    f['membership'] = member
    owner = np.array([0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 7])
    visual = torch.as_tensor(rng.normal(size=(len(owner), 4, 1536)), dtype=dtype, device=device)
    visual[:, 0, 0] = torch.arange(1, len(owner)+1, dtype=dtype, device=device)
    mask = torch.ones(len(owner), 4, dtype=torch.bool, device=device)
    mask[0, 1:] = False
    fit, valid = np.arange(6), np.array([6, 7])
    wrong = np.arange(n); wrong[fit] = np.roll(fit, 1); wrong[valid] = valid[::-1]
    ds = SimpleNamespace(fold=1, role=role, seed=seed, f=f,
        ft={k: torch.as_tensor(v, dtype=torch.bool if k == 'membership' else dtype, device=device)
            for k, v in f.items()}, owner=owner, visual=visual, mask=mask,
        global_wsi_indices=np.arange(len(owner)) + 30, fit=fit, valid=valid, active=np.arange(8),
        wrong_c=wrong.copy(), wrong_o=wrong.copy(),
        audit={'fit_hash': 'same', 'molecular': {'membership_sizes': np.array([2] * 8)},
               'wrong_controls': {'seed': seed}})
    ds.ep = {'time': np.full(n, np.nan), 'event': np.full(n, np.nan), 'valid': np.zeros(n, bool)}
    ds.ep['time'][:8] = [1, 2, 2, 3, 4, 5, 6, 7]
    ds.ep['event'][:8] = [1, 1, 0, 1, 0, 1, 1, 0]
    ds.ep['valid'][:8] = True
    ds.check_indices = lambda ix: data.Data.check_indices(ds, ix)
    return ds


class SlideScalar(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))

    def forward(self, visual, *args, **kwargs):
        return visual[:, 0, 0] * self.weight


def test_survival_aggregation_is_equal_wsi_scalar_not_patch_weighted(monkeypatch):
    monkeypatch.setattr(train, 'cfg', configuration)
    ds = synthetic_data()
    risk = train.forward(SlideScalar(), ds, np.array([3, 0, 5, 1]))
    expected = [np.arange(1, len(ds.owner)+1)[ds.owner == p].mean() for p in [3, 0, 5, 1]]
    torch.testing.assert_close(risk, torch.tensor(expected, dtype=torch.float64))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_multiple_microbatch_checkpoint_dropout_full_risk_gradient_matches_complete_graph(monkeypatch, device):
    monkeypatch.setattr(train, 'cfg', configuration)
    ds = synthetic_data(device=device)
    ds.visual.requires_grad_()
    for key in ['C', 'R', 'G', 'L', 'X']:
        ds.ft[key].requires_grad_()
    reference = copy.deepcopy(ds)
    reference.check_indices = lambda ix: data.Data.check_indices(reference, ix)
    torch.manual_seed(314)
    model = AttentionModel(3, ds.f['membership'], np.linspace(-.1, .1, 7), np.zeros(3), 'A3', configuration()).to(device=device, dtype=torch.float64)
    with torch.no_grad():
        model.attention.c.weight[:, 1:].normal_(std=.2)
    oracle = copy.deepcopy(model)
    model.train(); oracle.train()
    torch.manual_seed(889)
    actual = train.forward(model, ds, ds.fit, training=True)
    loss = full_risk_efron(actual, ds.ep['time'][ds.fit], ds.ep['event'][ds.fit])
    loss.backward()
    torch.manual_seed(889)
    expected = train.forward(oracle, reference, reference.fit, training=False)
    oracle_loss = full_risk_efron(expected, reference.ep['time'][reference.fit], reference.ep['event'][reference.fit])
    oracle_loss.backward()
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(loss, oracle_loss, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ds.visual.grad, reference.visual.grad, atol=1e-11, rtol=1e-11)
    for key in ['C', 'R', 'G', 'L', 'X']:
        torch.testing.assert_close(ds.ft[key].grad, reference.ft[key].grad, atol=1e-11, rtol=1e-11, msg=key)
    for (name, par), (_, ref) in zip(model.named_parameters(), oracle.named_parameters()):
        if par.grad is None or ref.grad is None:
            assert par.grad is None and ref.grad is None, name
        else:
            torch.testing.assert_close(par.grad, ref.grad, atol=2e-11, rtol=2e-11, msg=name)
    # Six fit patients span three distinct microbatches and unequal slide counts;
    # a closure bound to the final native rows/WSI IDs would fail native test below.
    assert len(ds.fit) // configuration()['training']['microbatch_patients'] == 3


def test_simple_models_reuses_seed_independent_fit_with_nested_numpy_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(train, 'cfg', configuration)
    monkeypatch.setattr(train, 'P', tmp_path)
    monkeypatch.setattr(train, 'source', lambda: 'synthetic-source')
    monkeypatch.setattr(train, 'file_lock', data._cache_lock)
    calls = []
    def fit(x, t, e, penalty):
        calls.append((x.copy(), t.copy(), e.copy(), penalty))
        return np.linspace(-.1, .1, x.shape[1]), {'gradient_max': 0.}
    monkeypatch.setattr(train, 'fit_ridge', fit)
    a = train.simple_models(synthetic_data(seed=31))
    b = train.simple_models(synthetic_data(seed=83))
    assert len(calls) == 2  # J0 and J_CS fitted once each; no per-seed fit.
    assert 'wrong_controls' not in a['preprocess_audit']
    assert a['preprocess_audit'] == b['preprocess_audit']
    assert a['preprocess_audit']['molecular']['membership_sizes'] == [2] * 8
    for name in ['J0', 'J_CS']:
        np.testing.assert_array_equal(a['models'][name]['w'], b['models'][name]['w'])
        np.testing.assert_array_equal(a['models'][name]['risk'], b['models'][name]['risk'])
        assert a['models'][name]['w'].dtype == np.float64
        assert a['models'][name]['risk'].dtype == np.float32


class NativeRowsProbe(train.NativeModel):
    """No official dependency: exercise the actual native checkpoint route."""
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor([.2, -.1, .15], dtype=torch.float64))
        self.aux_weight = torch.nn.Parameter(torch.tensor([.01, -.02, .03], dtype=torch.float64))

    def forward(self, visual, mask, clinical, r, gain, loss, rm, cm, main_x,
                native_features=None, wsi_indices=None, return_extra=False, **kwargs):
        # Different microbatches have 3,4,2 WSIs, so late-bound native_rows or
        # WSI IDs cannot accidentally pass from the final closure into earlier
        # checkpoint recomputations. Both influence trainable gradients.
        features = (torch.nn.functional.dropout(visual[:, 0, :3], .25, self.training)
                    + native_features['C'] * wsi_indices[:, None] / 10.)
        risk = features @ self.weight
        extra = (native_features['C'] @ self.aux_weight).square().mean()
        return (risk, extra) if return_extra else risk


def test_native_checkpoint_binds_per_microbatch_rows_wsi_ids_and_auxiliary_graph(monkeypatch):
    monkeypatch.setattr(train, 'cfg', configuration)
    ds = synthetic_data()
    ds.visual.requires_grad_()
    ds.ft['C'].requires_grad_()
    reference = copy.deepcopy(ds)
    reference.check_indices = lambda ix: data.Data.check_indices(reference, ix)
    model = NativeRowsProbe().train()
    oracle = copy.deepcopy(model)
    torch.manual_seed(617)
    risk, extra = train.forward(model, ds, ds.fit, training=True, with_aux=True)
    objective = full_risk_efron(risk, ds.ep['time'][ds.fit], ds.ep['event'][ds.fit]) + extra
    objective.backward()
    torch.manual_seed(617)
    risk_o, extra_o = train.forward(oracle, reference, reference.fit, training=False, with_aux=True)
    reference_objective = full_risk_efron(risk_o, reference.ep['time'][reference.fit], reference.ep['event'][reference.fit]) + extra_o
    reference_objective.backward()
    torch.testing.assert_close(risk, risk_o, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(extra, extra_o, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ds.visual.grad, reference.visual.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(ds.ft['C'].grad, reference.ft['C'].grad, atol=1e-12, rtol=1e-12)
    for (name, actual), (_, expected) in zip(model.named_parameters(), oracle.named_parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-12, rtol=1e-12, msg=name)


def test_outer_requires_selected_epochs_and_is_a_fresh_exact_length_refit(monkeypatch):
    monkeypatch.setattr(train, 'cfg', configuration)
    monkeypatch.setattr(train, 'source', lambda: 'synthetic-source')
    monkeypatch.setattr(train, 'Data', lambda fold, role, seed: synthetic_data(role, seed, dtype=torch.float32))
    monkeypatch.setattr(train, 'simple_models', lambda ds: {'models': {'J0': {'w': np.linspace(-.1, .1, 7)}}})
    monkeypatch.setattr(train, 'probe', lambda *args: {})
    original_factory = train.make_model
    initials = []
    def factory(ds, arm, w):
        model = original_factory(ds, arm, w)
        initials.append({name: value.detach().clone() for name, value in model.state_dict().items()})
        return model
    monkeypatch.setattr(train, 'make_model', factory)
    with pytest.raises(ValueError, match='inner-selected epoch'):
        train.train_cell('A3', 31, 1, 'outer')
    initials.clear()
    inner = train.train_cell('A3', 31, 1, 'inner', epochs=2)
    outer = train.train_cell('A3', 31, 1, 'outer', epochs=3)
    assert len(initials) == 2
    for name in initials[0]:
        torch.testing.assert_close(initials[0][name], initials[1][name], atol=0, rtol=0, msg=name)
    assert any(not torch.equal(inner['state'][name], initials[1][name]) for name in initials[1])
    assert outer['selected_epoch'] == 3 and outer['summary']['actual_epochs'] == 3
    assert [row['epoch'] for row in outer['history']] == [1, 2, 3]
    assert all('validation_loss' not in row for row in outer['history'])

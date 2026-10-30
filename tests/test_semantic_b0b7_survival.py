"""Synthetic mathematical qualification only; no cohort or model fitting."""
import ast
import hashlib
import importlib.util
import numbers
from pathlib import Path

import numpy as np
import pytest
import torch

from src.semantic_b0b7.survival import (
    aggregate_folds, efron_cox_loss, efron_numpy, evaluate_fold,
    fit_censoring_km, harrell_components, pair_components, tau_support,
    uno60_components, uno_weights,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tmp/semantic_b0b7_scorer_reference_20261030"
REFERENCE_SHA = {
    "metrics.py": "0bcbd62444350b2b1b686e0a467400edc1ceb6d0d9b0b18d01a2f50d256071c2",
    "nonparametric.py": "5cd013036bd8a6ca57ab37e54fc724477cee3aa470048e74ccc70bc2fb392716",
}


def _official_source_namespace():
    """Execute unmodified v0.25.0 core functions from optional local cache.

    The GPL upstream files remain test-only in tmp and are not vendored here.
    Since the compiled sksurv package is absent locally, its validation wrapper
    is replaced only by the narrow synthetic-input adapter below. The original
    concordance/count/KM estimator bodies themselves are executed unchanged.
    """
    from sklearn.base import BaseEstimator
    from sklearn.utils._param_validation import Interval, StrOptions
    from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted

    namespace = {"np": np, "numbers": numbers, "Interval": Interval,
                 "StrOptions": StrOptions, "BaseEstimator": BaseEstimator, "check_array": check_array,
                 "check_consistent_length": check_consistent_length,
                 "check_is_fitted": check_is_fitted,
                 "NoComparablePairException": ValueError}

    def check_y_survival(y, *other, allow_all_censored=False):
        if not other:
            names = y.dtype.names
            e, t = y[names[0]], y[names[1]]
            if not allow_all_censored and not e.any():
                raise ValueError("All samples are censored")
            return np.asarray(e, bool), np.asarray(t, float)
        return np.asarray(y, bool), other[0], np.asarray(other[1], float)

    namespace["check_y_survival"] = check_y_survival
    wanted = {"metrics.py": {"_iter_comparable", "_estimate_concordance_index"},
              "nonparametric.py": {"_compute_counts", "kaplan_meier_estimator",
                                   "SurvivalFunctionEstimator", "CensoringDistributionEstimator"}}
    for name in ("nonparametric.py", "metrics.py"):
        path = REFERENCE / name
        if not path.exists():
            pytest.skip("optional pinned official-source cache is absent")
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA[name]
        parsed = ast.parse(raw.decode("utf-8"))
        nodes = [n for n in parsed.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in wanted[name]]
        assert {n.name for n in nodes} == wanted[name]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _structured(e, t):
    y = np.empty(len(t), dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = e, t
    return y


def test_three_subject_ties_counts_and_direction():
    # Same-time event/censor is comparable; both-time events are not.
    x = harrell_components([1, 1, 2], [1, 0, 1], [3, 2, 1])
    assert x.score == 1 and x.denominator == 2
    assert x.event_censor_tied_time_pairs == 1
    assert harrell_components([1, 1], [1, 1], [2, 1]).score is None
    assert harrell_components([1, 1, 2], [1, 0, 1], [-3, -2, -1]).score == 0


def test_risk_tolerance_applies_before_concordance_credit():
    x = harrell_components([1, 2, 3], [1, 1, 0], [1, 1 + 5e-9, 0])
    assert x.numerator == 2.5 and x.denominator == 3
    assert x.tied_risk_pairs == 1 and x.concordant_pairs == 2


def test_ipcw_anchor_weights_are_squared_once():
    x = pair_components([1, 2, 3], [1, 1, 0], [3, 1, 2], [4, 9, 0])
    assert x.numerator == 8 and x.denominator == 17
    assert x.score == 8 / 17


def test_reverse_km_event_leaves_before_censor_at_tie_and_post_value():
    km = fit_censoring_km([1, 1, 2, 3], [1, 0, 1, 0])
    assert np.allclose(km.probability, [2 / 3, 2 / 3, 0])
    assert np.allclose(km.predict([0, .99, 1, 1.5, 2, 3, 4]), [1, 1, 2/3, 2/3, 2/3, 0, 0])
    assert not km.times.flags.writeable and not km.probability.flags.writeable


def test_uno_tau_is_strict_and_keeps_later_comparators():
    t = [10, 59, 60, 70]
    e = [1, 1, 1, 0]
    w = uno_weights([10, 20, 80, 90], [1, 0, 1, 1], t, e)
    assert np.allclose(w, [1, 2.25, 0, 0])
    u = uno60_components([10, 20, 80, 90], [1, 0, 1, 1], t, e, [4, 3, 2, 1])
    assert u.score == 1
    assert u.denominator == pytest.approx(3 + 2.25 * 2)


def test_zero_censor_support_is_not_clipped():
    with pytest.raises(ValueError, match="IPCW_ZERO_CENSOR_SURVIVAL"):
        uno_weights([1, 2], [1, 0], [3, 4], [1, 0])
    assert tau_support([1, 2], [1, 0])["supported"] is False


def test_unknown_censor_tail_is_not_extrapolated():
    with pytest.raises(ValueError, match="EXCEEDS_TRAIN_SUPPORT"):
        uno_weights([1, 2], [1, 1], [3, 4], [1, 0])
    assert tau_support([1, 2], [1, 1])["supported"] is False


def test_tau_diagnostic_does_not_silently_change_statistic():
    # Eligible events have support, but tau itself does not: report both facts.
    row = evaluate_fold([1, 2, 3], [1, 1, 0], [.5, 2.5, 4], [1, 0, 0], [3, 2, 1])
    assert row["Uno60"] == 1
    assert row["tau_support"]["supported"] is False


def test_no_pairs_returns_none_not_artificial_chance():
    assert harrell_components([1, 2], [0, 0], [0, 0]).score is None
    assert pair_components([1, 2], [1, 0], [2, 1], [0, 0]).score is None


def test_fold_aggregation_uses_each_statistic_own_denominator():
    rows = [{"Harrell_num": 1, "Harrell_den": 1, "Uno_num": 1, "Uno_den": 4},
            {"Harrell_num": 0, "Harrell_den": 9, "Uno_num": 6, "Uno_den": 6}]
    out = aggregate_folds(rows)
    assert out["Harrell"] == .1 and out["Uno60"] == .7
    assert out["Uno60"] != (1 * .25 + 9 * 1) / 10


def test_independent_fold_shifts_do_not_change_primary_metric():
    training = ([1, 2, 4, 8], [1, 0, 1, 1])
    a = evaluate_fold(*training, [1, 2, 3], [1, 1, 0], [3, 2, 1], tau=4)
    b = evaluate_fold(*training, [1, 2, 3], [1, 1, 0], [103, 102, 101], tau=4)
    combined = aggregate_folds([a, b])
    assert combined["Harrell"] == combined["Uno60"] == 1


@pytest.mark.parametrize("time,event,risk", [
    ([1, np.nan], [1, 0], [1, 0]),
    ([-1, 2], [1, 0], [1, 0]),
    ([1, 2], [2, 0], [1, 0]),
    ([1, 2], [1, 0], [np.inf, 0]),
    ([1, 2], [1, 0], [1]),
])
def test_metric_inputs_fail_closed(time, event, risk):
    with pytest.raises(ValueError):
        harrell_components(time, event, risk)


@pytest.mark.parametrize("time,event", [
    ([1, 2, 3], [1, 1, 0]),
    ([1, 1, 2], [1, 1, 0]),
    ([1, 1, 1], [1, 1, 1]),
    ([1, 1, 1, 2], [1, 1, 0, 1]),
])
def test_efron_torch_matches_independent_scalar_oracle(time, event):
    r = torch.linspace(-.4, .6, len(time), dtype=torch.float64, requires_grad=True)
    got = efron_cox_loss(r, time, event)
    assert got.item() == pytest.approx(efron_numpy(r.detach().numpy(), time, event), abs=2e-14)
    got.backward()
    assert torch.isfinite(r.grad).all()


def test_efron_finite_difference_and_shift_invariance():
    time = [1, 1, 1, 2, 3, 3, 4]
    event = [1, 1, 0, 1, 1, 0, 0]
    raw = np.array([.1, -.2, .3, -.4, .5, .6, -.1])
    risk = torch.tensor(raw, dtype=torch.float64, requires_grad=True)
    loss = efron_cox_loss(risk, time, event)
    grad = torch.autograd.grad(loss, risk)[0].numpy()
    step = 1e-6
    numerical = []
    for j in range(len(raw)):
        shift = np.zeros(len(raw)); shift[j] = step
        numerical.append((efron_numpy(raw+shift, time, event)-efron_numpy(raw-shift, time, event))/(2*step))
    assert np.max(np.abs(grad-numerical)) < 5e-10
    assert abs(grad.sum()) < 1e-14
    assert efron_cox_loss(risk+1000, time, event).item() == pytest.approx(loss.item(), abs=2e-13)


def test_efron_differs_from_breslow_only_with_ties():
    r = [.1, .4, -.2, .3]
    assert efron_numpy(r, [1, 1, 2, 3], [1, 1, 1, 0]) < efron_numpy(r, [1, 1, 2, 3], [1, 1, 1, 0], ties="breslow")
    assert efron_numpy(r, [1, 2, 3, 4], [1, 1, 1, 0]) == efron_numpy(r, [1, 2, 3, 4], [1, 1, 1, 0], ties="breslow")


def test_efron_extreme_risk_and_endpoint_precision():
    risk = torch.tensor([10000., -10000., 0.], dtype=torch.float64, requires_grad=True)
    loss = efron_cox_loss(risk, [1, 1, 2], [1, 1, 0])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(risk.grad).all()
    time = [1, 1+1e-9, 2]
    r = torch.tensor([.2, -.1, .3], dtype=torch.float32)
    assert efron_cox_loss(r, time, [1, 1, 0]).item() == pytest.approx(efron_numpy(r.numpy(), time, [1, 1, 0]), abs=2e-7)


def test_efron_cannot_replace_full_risk_by_microbatch_average():
    risk = torch.tensor([.1, -.5, .3, -.4], dtype=torch.float64)
    full = efron_cox_loss(risk, [1, 2, 3, 4], [1, 0, 1, 0])
    micro = (efron_cox_loss(risk[:2], [1, 2], [1, 0]) + efron_cox_loss(risk[2:], [3, 4], [1, 0])) / 2
    assert abs(full.item()-micro.item()) > .1


def test_efron_rejects_no_events_or_invalid_labels():
    with pytest.raises(ValueError, match="NO_EVENTS"):
        efron_cox_loss(torch.tensor([0., 1.]), [1, 2], [0, 0])
    with pytest.raises(ValueError):
        efron_cox_loss(torch.tensor([0., 1.]), [1, 2], [1, 2])


def test_actual_pinned_official_core_functions_100_synthetic_cases():
    upstream = _official_source_namespace()
    rng = np.random.default_rng(313147)
    for _ in range(100):
        train_t = rng.integers(1, 20, size=35).astype(float)
        train_t[-1] = 30
        train_e = rng.binomial(1, .6, size=35).astype(bool)
        train_e[-1] = True
        t = rng.integers(1, 20, size=20).astype(float)
        e = rng.binomial(1, .6, size=20).astype(bool)
        e[0] = True; t[0] = 1; t[-1] = 21
        r = rng.integers(-3, 4, size=20).astype(float)
        expected_km = upstream["CensoringDistributionEstimator"]().fit(_structured(train_e, train_t))
        mine = fit_censoring_km(train_t, train_e)
        q = np.arange(0, 31, dtype=float)
        assert np.allclose(mine.predict(q), expected_km.predict_proba(q), atol=0, rtol=1e-14)
        w = uno_weights(train_t, train_e, t, e, tau=15)
        eligible = e & (t < 15)
        official_weights = np.zeros(len(t))
        official_weights[eligible] = expected_km.predict_ipcw(_structured(e[eligible], t[eligible]))**2
        assert np.allclose(w, official_weights, atol=0, rtol=1e-14)
        for weight in (np.ones(len(t)), official_weights):
            want = upstream["_estimate_concordance_index"](e, t, r, weight)
            got = pair_components(t, e, r, weight)
            assert got.score == pytest.approx(want[0], abs=2e-15)
            assert (got.concordant_pairs, got.discordant_pairs, got.tied_risk_pairs, got.event_censor_tied_time_pairs) == want[1:]


def test_installed_sksurv_api_when_available():
    if importlib.util.find_spec("sksurv") is None:
        pytest.skip("compiled scikit-survival package is not installed; pinned official core source is tested separately")
    from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
    train = _structured([1, 0, 1, 1], [1, 1, 4, 8])
    test = _structured([1, 0, 1], [1, 1, 3])
    r = [3., 2., 1.]
    assert harrell_components(test["time"], test["event"], r).score == concordance_index_censored(test["event"], test["time"], r)[0]
    assert uno60_components(train["time"], train["event"], test["time"], test["event"], r, tau=5).score == concordance_index_ipcw(train, test, r, tau=5)[0]


def test_historical_reverse_km_matches_but_phase8_left_limit_differs():
    from src.phase10_trimodal.reference import censor_probability
    from src.phase8_v0.cox import km_left
    t = np.array([1, 1, 2, 3], float); e = np.array([1, 0, 1, 0])
    query = np.array([1, 2], float)
    new = fit_censoring_km(t, e).predict(query)
    assert np.allclose(new, censor_probability(t, e, query))
    assert not np.allclose(new, km_left(t, e, query))

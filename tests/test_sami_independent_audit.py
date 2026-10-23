"""Independent replay must catch split and preprocessing corruption."""
import importlib.util
import copy
import json
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('sami_independent_audit', Path(__file__).resolve().parents[1]/'scripts/audit_sami_stage_a.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture_cell(tmp_path):
    x = np.arange(54, dtype=np.float32).reshape(18, 3)/13
    y = np.tile([0, 1, 2], 6)
    donor = np.repeat(['a', 'b', 'c'], 6)
    section = np.repeat(['a1', 'a2', 'b1', 'b2', 'c1', 'c2'], 3)
    tr, ev = np.arange(12), np.arange(12, 18)
    mean, scale = x[tr].astype(float).mean(0), x[tr].astype(float).std(0)
    W = np.array([[.1, .2, -.1], [.3, -.2, .1], [.1, .0, -.2]])
    b = np.array([.2, -.1, -.1])
    logits = ((x[ev]-mean)/scale) @ W + b
    p = np.exp(logits-logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    np.savez(tmp_path/'model.npz', W=W, b=b, mean=mean, scale=scale)
    np.savez(tmp_path/'predictions.npz', row_index=ev, probabilities=p)
    receipt = dict(status='COMPLETE', lineage={'test': 'fixed'}, train_index=tr.tolist(), evaluation_index=ev.tolist(),
                   model_sha256=audit.sha(tmp_path/'model.npz'), predictions_sha256=audit.sha(tmp_path/'predictions.npz'),
                   metrics=audit.metrics(y[ev], p, donor[ev], section[ev]),
                   fit_audit={'success': True, 'gradient_inf': 0., 'accepted_gradient_inf': 1e-5})
    (tmp_path/'COMPLETE.json').write_text(json.dumps(receipt))
    data = dict(X=x, y=y, donor=donor, section=section)
    return data, tr, ev, receipt


def test_hierarchical_weights_equalize_donors_without_labels():
    w = audit.weights(np.array(['a', 'a', 'a', 'b']), np.array(['x', 'x', 'y', 'z']))
    np.testing.assert_allclose(w, [.125, .125, .25, .5])


def test_perfect_multiclass_scores_have_zero_proper_losses():
    y = np.array([0, 1, 2])
    m = audit.metrics(y, np.eye(3), np.array(['a']*3), np.array(['s']*3))
    assert m['balanced_accuracy'] == 1
    assert m['negative_log_likelihood'] == 0
    assert m['brier_multiclass_sum'] == 0
    assert m['ece_10_equal_width_bins'] == 0


def test_independent_cell_replay(tmp_path):
    data, tr, ev, receipt = fixture_cell(tmp_path)
    result = audit.replay_cell(tmp_path, data, tr, ev, receipt['lineage'])
    assert result[-2] < 1e-12 and result[-1] < 1e-12


def test_test_influenced_scaler_is_rejected_even_with_updated_hash(tmp_path):
    data, tr, ev, receipt = fixture_cell(tmp_path)
    with np.load(tmp_path/'model.npz') as f:
        arrays = dict(f)
    arrays['mean'] = data['X'].astype(float).mean(0)
    np.savez(tmp_path/'model.npz', **arrays)
    receipt['model_sha256'] = audit.sha(tmp_path/'model.npz')
    (tmp_path/'COMPLETE.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='training-only'):
        audit.replay_cell(tmp_path, data, tr, ev, receipt['lineage'])


def test_crossed_donor_is_rejected(tmp_path):
    data, tr, ev, receipt = fixture_cell(tmp_path)
    data['donor'][ev] = 'a'
    with pytest.raises(ValueError, match='donor crosses'):
        audit.replay_cell(tmp_path, data, tr, ev, receipt['lineage'])


def test_metric_tamper_is_rejected(tmp_path):
    data, tr, ev, receipt = fixture_cell(tmp_path)
    receipt['metrics']['balanced_accuracy'] += .01
    (tmp_path/'COMPLETE.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='metric mismatch'):
        audit.replay_cell(tmp_path, data, tr, ev, receipt['lineage'])


def test_ece_locks_linspace_nextafter_boundary():
    a = .6
    b = np.nextafter(a, 1.)
    p = np.array([[a, .2, .2], [b, 1.-b-.1, .1], [0., 0., 1.]])
    y = np.array([0, 1, 2])
    result = audit.metrics(y, p, np.array(['d']*3), np.array(['s']*3))
    # Linspace places a below its 0.6 endpoint and b exactly at the endpoint.
    # Distinct bins keep the one-correct/one-wrong contributions separate.
    expected = ((1.-a)+b)/3
    assert np.linspace(0., 1., 11)[6] == b
    assert result['ece_10_equal_width_bins'] == pytest.approx(expected, abs=1e-15)
    # The old b/10 endpoints incorrectly combine the opposing contributions.
    wrong_rational_endpoint_result = abs((1.-a)-b)/3
    assert abs(result['ece_10_equal_width_bins']-wrong_rational_endpoint_result) > .25


@pytest.mark.parametrize('limit', [1e-4, 1., np.inf, np.nan, -1.])
def test_receipt_cannot_relax_optimizer_threshold(tmp_path, limit):
    data, tr, ev, receipt = fixture_cell(tmp_path)
    receipt['fit_audit']['accepted_gradient_inf'] = limit
    (tmp_path/'COMPLETE.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='threshold differs'):
        audit.replay_cell(tmp_path, data, tr, ev, receipt['lineage'], 1e-5)


@pytest.mark.parametrize('gradient', [-1., np.inf, np.nan, np.nextafter(1e-5, np.inf)])
def test_optimizer_gradient_must_be_finite_nonnegative_and_within_lock(tmp_path, gradient):
    data, tr, ev, receipt = fixture_cell(tmp_path)
    receipt['fit_audit']['gradient_inf'] = gradient
    (tmp_path/'COMPLETE.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='optimizer not qualified'):
        audit.replay_cell(tmp_path, data, tr, ev, receipt['lineage'], 1e-5)


def test_config_optimizer_threshold_cannot_relax(tmp_path):
    data, tr, ev, receipt = fixture_cell(tmp_path)
    with pytest.raises(ValueError, match='unexpected locked'):
        audit.replay_cell(tmp_path, data, tr, ev, receipt['lineage'], 1e-4)


def public_fixture():
    lineage = {'source_sha256': 'synthetic-fixed'}
    classes = ['tumor', 'stroma', 'tumor_stroma_mixed']
    public = dict(lineage=lineage, schema_version='sami_stage_a_results_v1',
                  status='COMPLETE_NOT_YET_INDEPENDENTLY_AUDITED',
                  n_donors=6, n_sections=12, n_spots=13011, completed_fits=96,
                  class_order=classes, outer_folds=[dict(outer_fold=i, selected_C=.1) for i in range(6)])
    return public, lineage, classes


def test_public_metadata_and_selected_config_are_verified():
    public, lineage, classes = public_fixture()
    audit.validate_public_header(public, lineage, classes)
    for i, fold in enumerate(public['outer_folds']):
        audit.validate_public_fold(fold, i, .1)


def test_public_lineage_tamper_rejected():
    public, lineage, classes = public_fixture()
    public['lineage'] = {'source_sha256': 'wrong'}
    with pytest.raises(ValueError, match='public lineage'):
        audit.validate_public_header(public, lineage, classes)


@pytest.mark.parametrize('kind', ['reorder', 'duplicate', 'omit'])
def test_public_fold_inventory_tamper_rejected(kind):
    public, lineage, classes = public_fixture()
    if kind == 'reorder':
        public['outer_folds'][0], public['outer_folds'][1] = public['outer_folds'][1], public['outer_folds'][0]
    elif kind == 'duplicate':
        public['outer_folds'][1] = copy.deepcopy(public['outer_folds'][0])
    else:
        public['outer_folds'].pop()
    with pytest.raises(ValueError, match='outer fold inventory'):
        audit.validate_public_header(public, lineage, classes)


def test_public_fold_index_and_selected_C_tamper_rejected():
    with pytest.raises(ValueError, match='outer fold index'):
        audit.validate_public_fold({'outer_fold': 1, 'selected_C': .1}, 0, .1)
    with pytest.raises(ValueError, match='selected C'):
        audit.validate_public_fold({'outer_fold': 0, 'selected_C': 1.}, 0, .1)

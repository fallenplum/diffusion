"""Matrix/provenance tests; no real patients or new PCA/teacher fits."""
from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest

from scripts import bgb_en_pca_geometry as geometry
from scripts.bgb_en_checkpoint_audit import AuditError
from src.phase10_trimodal.reference import reference_efron
from scripts.bgb_en_pca_geometry_audit import independent_objective_gradient
from scripts.bgb_en_pca_geometry_solver_repair import strict_fit
from src.bgb_en.survival import fit_offset_ridge


def example():
    rng = np.random.default_rng(27029)
    x = rng.normal(size=(20, 8)); train = np.arange(14)
    q, _ = np.linalg.qr(rng.normal(size=(8, 3)))
    center = x[train].mean(0); scores = (x - center) @ q
    state = dict(components=q.T, mean=center, post_mean=scores[train].mean(0), post_scale=scores[train].std(0))
    return x, train, state


def test_unwhitened_PC_l2_is_gene_l2_in_the_same_orthonormal_subspace():
    x, train, state = example()
    score, _, diagnostic = geometry.pca_coordinates(x, state, train, 3)
    beta = np.array([.2, -.3, .4]); gene_beta = state['components'].T @ beta
    assert abs(beta @ beta - gene_beta @ gene_beta) < 1e-15
    assert np.allclose(score @ beta, (x - state['mean']) @ gene_beta, atol=1e-15, rtol=0)
    event = (np.arange(20) % 3 == 0).astype(int); time = np.arange(1., 21.)
    a = reference_efron(time, event, score @ beta) + .5 * (beta @ beta)
    b = reference_efron(time, event, x @ gene_beta) + .5 * (gene_beta @ gene_beta)
    assert abs(a - b) < 1e-14  # Different centering is only a Cox-invariant intercept.
    assert diagnostic['orthonormality_max_error'] < 1e-14


def test_PC_unit_scaling_changes_the_gene_space_penalty():
    pc_sd = np.array([.1, 1., 10.]); beta_pc = np.array([1., 2., 3.])
    coefficient_on_scaled_PC = pc_sd * beta_pc
    assert coefficient_on_scaled_PC @ coefficient_on_scaled_PC == pytest.approx(np.sum(pc_sd**2 * beta_pc**2))
    assert not np.isclose(coefficient_on_scaled_PC @ coefficient_on_scaled_PC, beta_pc @ beta_pc)
    # Equal unwhitened effect sizes receive 10,000-fold different old penalties.
    assert pc_sd[-1]**2 / pc_sd[0]**2 == pytest.approx(10000.)


def test_saved_post_scaling_replays_original_coordinates_and_rejects_drift():
    x, train, state = example()
    score, old, _ = geometry.pca_coordinates(x, state, train, 3)
    assert np.allclose(old[train].mean(0), 0., atol=1e-15)
    assert np.allclose(old[train].std(0), 1., atol=1e-15)
    changed = copy.deepcopy(state); changed['post_scale'][0] *= 2.
    with pytest.raises(AuditError, match='SAVED_PC_SCALING_NOT_TRAIN_LOCAL'):
        geometry.pca_coordinates(x, changed, train, 3)
    changed = copy.deepcopy(state); changed['components'][0] *= 2.
    with pytest.raises(AuditError, match='NOT_ORTHONORMAL'):
        geometry.pca_coordinates(x, changed, train, 3)


def test_risk_amplitude_scaling_uses_each_own_train_not_heldout_moments():
    risk = np.array([-2., 0., 2., 1000., -1000.]); train = np.arange(3)
    standardized, state = geometry.own_training_standardize(risk, train)
    altered = risk.copy(); altered[3:] *= 100.
    second, second_state = geometry.own_training_standardize(altered, train)
    assert np.array_equal(standardized[train], second[train])
    assert state == second_state
    assert state['mean'] == 0 and state['sd'] == pytest.approx(np.sqrt(8 / 3))


def test_missing_prefit_lock_and_source_change_fail_closed(tmp_path):
    args = SimpleNamespace(private=tmp_path / 'isolated_private')
    contract = dict(specification=dict(dimensions=[32,128,256]), source='synthetic')
    with pytest.raises(AuditError, match='PRE_FIT_LOCK_MISSING'):
        geometry.enforce_lock(args, contract)
    first = geometry.enforce_lock(args, contract, create=True)
    assert first['optimizer_fits_before_lock'] == 0
    assert geometry.enforce_lock(args, contract) == first
    with pytest.raises(AuditError, match='SOURCE_OR_CONFIG_LOCK_CHANGED'):
        geometry.enforce_lock(args, dict(contract, source='changed'))


def test_partial_report_retains_all_locked_dimensions_without_best_selection(tmp_path):
    args = SimpleNamespace(private=tmp_path/'private', output=tmp_path/'public')
    contract = dict(specification=dict(dimensions=[32,128,256], inner_partitions=[0,1,2]))
    geometry.enforce_lock(args, contract, create=True)
    result = geometry.report(args, contract)
    assert result['status'] == 'RUNNING'
    assert result['new_Cox_fits'] == 0 and result['PCA_refits'] == result['teacher_refits'] == 0
    assert [r['dimension'] for r in result['summaries']] == [32,128,256]
    assert all('new_Harrell' not in row for row in result['summaries'])
    assert result['changes_R4_finalists'] is False
    assert 'not_independent_confirmation' in result['analysis_role']
    public = json.loads((args.output/'R2_PCA_GEOMETRY_CONTROL.json').read_text())
    assert public['all_dimensions_reported'] is True


def test_independent_efron_PC_gradient_matches_finite_difference_with_ties():
    x = np.array([[.2, 1.], [.3, -.4], [-.2, .6], [.1, -.8], [.5, .4], [-.1, .2]])
    time = np.array([1., 2., 2., 2., 4., 5.]); event = np.array([1,1,1,0,0,1])
    coefficient = np.array([.13, -.21])
    loss, gradient = independent_objective_gradient(x, time, event, coefficient)
    def objective(w): return reference_efron(time, event, x @ w) + .5 * (w @ w)
    finite = np.array([(objective(coefficient+np.eye(2)[j]*1e-5)-objective(coefficient-np.eye(2)[j]*1e-5))/2e-5 for j in range(2)])
    assert abs(loss-objective(coefficient)) < 1e-15
    assert np.allclose(gradient, finite, atol=1e-10, rtol=0)


def test_precision_retry_keeps_same_convex_target_and_external_gradient_guard():
    x = np.array([[.2,1.],[.3,-.4],[-.2,.6],[.1,-.8],[.5,.4],[-.1,.2]])
    time = np.array([1.,2.,2.,2.,4.,5.]); event=np.array([1,1,1,0,0,1])
    old, info = fit_offset_ridge(x,time,event,1.)
    repaired, detail = strict_fit(x,time,event,1.)
    loss, gradient = independent_objective_gradient(x,time,event,repaired)
    assert np.allclose(old,repaired,atol=2e-6,rtol=0)
    assert np.allclose(x@old,x@repaired,atol=2e-6,rtol=0)
    assert abs(loss-detail['loss'])<1e-12
    assert np.max(np.abs(gradient))<=detail['external_gradient_guard']==2e-5
    assert detail['precision_options']['ftol']==1e-15
    assert detail['precision_options']['gtol']==1e-9
    assert abs(info['loss']-detail['loss'])<1e-12
    with pytest.raises(AuditError,match='OBJECTIVE_CHANGED'):
        strict_fit(x,time,event,.1)

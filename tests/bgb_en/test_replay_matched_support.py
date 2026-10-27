"""Fixed replay gate, tie-credit diagnostics and actual constructor guards."""
import numpy as np
import pytest

from scripts.bgb_en_checkpoint_audit import AuditError
from scripts.bgb_en_replay_matched_support import (
    RISK_TOLERANCE, risk_diagnostics, synthetic_qualification, verify_risk_with_fixed_guard,
)


def test_synthetic_replay_qualification_never_uses_O9_or_wrong_gamma():
    row = synthetic_qualification()
    assert row['status'] == 'PASS_SYNTHETIC_CPU_ONLY'
    assert [(r['base_arm'], r['gamma']) for r in row['checks']] == [('O4', .3), ('F5', .1)]
    assert all(r['same_membership_object'] and not r['second_shuffle_applied'] for r in row['checks'])
    assert row['actual_model_forwards'] == row['optimizer_calls'] == row['GPU_calls'] == 0
    assert row['cohort_inputs_read'] is False


def test_forward_gate_is_fixed_and_cannot_be_relaxed_by_caller():
    assert RISK_TOLERANCE == 2e-6
    assert verify_risk_with_fixed_guard(np.zeros(3), np.full(3, 1e-6), 3) == pytest.approx(1e-6)
    with pytest.raises(AuditError, match='FORWARD_RISK'):
        verify_risk_with_fixed_guard(np.zeros(3), np.full(3, 2.1e-6), 3)


def test_tiny_accepted_error_can_change_tie_credits_and_must_be_disclosed():
    train_t = np.array([1., 2., 3., 4.]); train_e = np.array([1, 1, 1, 0])
    t = np.array([1., 2., 3.]); e = np.array([1, 1, 0])
    saved = np.zeros(3); replayed = np.array([1e-7, 0., 0.])
    verify_risk_with_fixed_guard(saved, replayed, 3)
    row = risk_diagnostics(train_t, train_e, t, e, saved, replayed, saved)
    assert row['rank_sensitivity_present']
    assert row['changed_pair_credit_count'] == 2
    assert row['delta_Harrell_rebuilt_minus_saved'] == pytest.approx(1 / 3)
    assert row['delta_Harrell_numerator'] == pytest.approx(1.)
    assert row['pair_credit_gain_count'] == pytest.approx(1.)
    assert row['pair_credit_loss_count'] == 0.


def test_identical_risks_have_zero_metric_and_pair_credit_change():
    train_t = np.array([1., 2., 3., 4.]); train_e = np.array([1, 1, 1, 0])
    t = np.array([1., 2., 3.]); e = np.array([1, 1, 0]); r = np.array([.2, -.1, .3])
    row = risk_diagnostics(train_t, train_e, t, e, r, r, np.zeros(3))
    assert not row['rank_sensitivity_present']
    assert row['delta_Harrell_rebuilt_minus_saved'] == row['delta_Uno60_rebuilt_minus_saved'] == 0

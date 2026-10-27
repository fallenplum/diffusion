import numpy as np
from scripts.bgb_en_budget_diagnostics import classify, frozen_scaling


def curve(epoch, loss, valid_h, valid_u=.7, gradient=.2):
    changes = dict(repair=.02, damage=.01, Uno_repair=.03, Uno_damage=.01)
    return dict(epoch=epoch, cox_loss=loss, train=dict(Harrell=.9, Uno60=.85, **changes),
                valid=dict(Harrell=valid_h, Uno60=valid_u, **changes), gradient_norm=gradient,
                gradient_tensor_nonzero_ratio=.8, delta_train_SD=.02, delta_valid_SD=.01)


def test_train_loss_alone_does_not_trigger_budget_direction():
    rows = [curve(20, 3., .72), curve(40, 2., .71), curve(80, 1., .69)]
    result = classify(rows)
    assert result['train_Cox_40to80_decreased']
    assert not result['strict_recorded_follow_up_direction']


def test_strict_joint_direction_and_inactive_gradient_are_separate():
    rows = [curve(20, 3., .70), curve(40, 2., .71), curve(80, 1., .72)]
    result = classify(rows)
    assert result['strict_recorded_follow_up_direction']
    rows[-1]['gradient_norm'] = 0.
    assert not classify(rows)['strict_recorded_follow_up_direction']


def test_frozen_gamma_diagnostic_replays_float32_saved_risk_and_never_retrains():
    base = np.array([.100000001, -.200000007, .8, 1.2], dtype=np.float64)
    delta = np.array([.02, -.03, .001, -.04], dtype=np.float32)
    saved = (base.astype(np.float32)+delta).astype(np.float64)
    time = np.array([1., 2., 3., 4.]); event = np.array([1, 0, 1, 1])
    rows = frozen_scaling(base, delta, saved, time, event, time, event)
    assert [row['gamma'] for row in rows] == [0., .03, .1, .3]
    assert rows[2]['gamma01_saved_risk_elementwise_equal'] is True
    assert all(row['retrained'] is False for row in rows)
    assert rows[0]['repair'] == 0 and rows[0]['damage'] == 0

"""Hard gate: one optimizer step must reduce negative Cox loss."""

from phase1_5r2.functional_tests import run_cox_descent_test


def test_one_step_cox_descent():
    result = run_cox_descent_test()
    assert result["status"] is True
    assert result["loss_after"] < result["loss_before"]

"""Hard gate: comparable-pair ranking derivative."""

from phase1_5r2.functional_tests import run_combined_gradient_test, run_rank_gradient_test


def test_rank_analytic_matches_autograd_and_finite_difference():
    result = run_rank_gradient_test()
    assert result["status"] is True
    assert result["analytic_vs_autograd"]["max_abs_error"] <= 1e-7
    assert result["analytic_vs_finite_difference"]["max_abs_error"] <= 1e-5


def test_combined_cox_rank_gradient():
    result = run_combined_gradient_test()
    assert result["status"] is True
    assert len(result["lambdas"]) == 2

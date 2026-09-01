"""Hard gate: corrected Cox analytic gradient."""

from phase1_5r2.functional_tests import run_cox_gradient_test


def test_cox_analytic_matches_autograd_and_finite_difference():
    result = run_cox_gradient_test()
    assert result["status"] is True
    assert result["analytic_vs_autograd"]["max_abs_error"] <= 1e-7
    assert result["analytic_vs_finite_difference"]["max_abs_error"] <= 1e-5
    assert result["corrected_cosine_similarity"] > 0.999999
    assert result["legacy_wrong_cosine_similarity"] < -0.999999

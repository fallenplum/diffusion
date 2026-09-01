"""Hard gate: two-pass exact risk-set update equals direct autograd."""

from phase1_5r2.functional_tests import run_two_pass_equivalence_test


def test_two_pass_matches_direct_autograd():
    result = run_two_pass_equivalence_test()
    assert result["status"] is True
    assert result["max_abs_parameter_error"] <= 1e-5

from src.phase8_v0.cox import finite_difference_check


def test_efron_finite_difference():
    result = finite_difference_check()
    assert result["status"] == "PASS"
    assert result["finite_difference_max_error"] < 1e-6

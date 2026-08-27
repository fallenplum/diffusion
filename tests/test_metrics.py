import numpy as np
import pytest

from phase1_utils import censoring_km, harrell_c, uno_c


def test_harrell_perfect_and_reverse_ordering():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.ones(4, dtype=int)
    assert harrell_c(durations, events, np.array([4.0, 3.0, 2.0, 1.0])) == 1.0
    assert harrell_c(durations, events, np.array([1.0, 2.0, 3.0, 4.0])) == 0.0


def test_harrell_ties_and_censoring():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.array([1, 1, 0, 0])
    assert harrell_c(durations, events, np.ones(4)) == 0.5
    censor = censoring_km(durations, events)
    score = uno_c(durations, events, np.array([4.0, 3.0, 2.0, 1.0]), censor(durations))
    assert 0.0 <= score <= 1.0
    assert np.isfinite(censor(2.5))


def test_uno_matches_harrell_without_censoring():
    durations = np.array([1.0, 2.0, 3.0, 4.0])
    events = np.ones(4, dtype=int)
    risks = np.array([4.0, 3.0, 2.0, 1.0])
    censor = censoring_km(durations, events)
    assert np.isclose(uno_c(durations, events, risks, censor(durations)), harrell_c(durations, events, risks))


@pytest.mark.parametrize(
    "durations,events,risks",
    [
        ([1, np.nan], [1, 0], [1, 0]),
        ([1, 2], [1, 0], [1, np.inf]),
        ([1, 2], [1, 2], [1, 0]),
        ([1, 2], [1], [1, 0]),
    ],
)
def test_survival_metrics_reject_invalid_inputs(durations, events, risks):
    with pytest.raises(ValueError):
        harrell_c(durations, events, risks)


def test_uno_rejects_bad_censoring_weights():
    with pytest.raises(ValueError):
        uno_c([1, 2], [1, 0], [2, 1], [1, np.nan])
    with pytest.raises(ValueError):
        uno_c([1, 2], [1, 0], [2, 1], [1, -0.1])

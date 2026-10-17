"""Qualification tests for count/exposure axes and donor-balanced objectives."""
import math
import pytest
import torch

from src.phase22_spatial_bridge.strank_objective import spatial_objective


DTYPE = torch.float64


def objective(mu, counts, tissue=None, donor=None, **kwargs):
    mu = torch.as_tensor(mu, dtype=DTYPE)
    counts = torch.as_tensor(counts, dtype=DTYPE)
    n = len(mu)
    return spatial_objective(mu.log(), counts, tissue or ["T"] * n,
                             donor or ["D"] * n, **kwargs)


def test_independent_two_spot_two_gene_calculation_uses_spot_axis():
    answer = objective([[.8, .2], [.8, .2]], [[8, 2], [2, 6]])
    # Equal composition, different exposure: each gene's two spot probabilities
    # are 10/(10+8) and 8/(10+8), unlike either row's gene probabilities.
    expected = -(10 * math.log(5 / 9) + 8 * math.log(4 / 9)) / 18
    point = -(10 * math.log(.8) + 8 * math.log(.2)) / 18
    assert answer.relative.item() == pytest.approx(expected, abs=1e-14)
    assert answer.composition.item() == pytest.approx(point, abs=1e-14)
    wrong_axis = point
    assert abs(answer.relative.item() - wrong_axis) > .01


def test_official_list_formula_parity_up_to_declared_mass_normalization():
    mu = torch.tensor([[.7, .3], [.2, .8], [.55, .45]], dtype=DTYPE)
    counts = torch.tensor([[7., 3.], [2., 6.], [5., 4.]], dtype=DTYPE)
    # Fixed G: log mean-count and log total-count differ by log(G), which
    # cancels exactly along the official dim=-2 (spot) normalization.
    official_logits = mu.log() + counts.mean(-1, keepdim=True).log()
    official_elements = -torch.log_softmax(official_logits, dim=-2) * counts
    normalized_official = official_elements.sum() / counts.sum()
    answer = objective(mu, counts)
    torch.testing.assert_close(answer.relative, normalized_official, atol=1e-14, rtol=0)
    assert answer.relative.item() != pytest.approx(official_elements.mean().item())


def test_separate_tissues_never_share_relative_normalizer():
    mu = [[.8, .2], [.3, .7], [.01, .99], [.99, .01]]
    counts = [[8, 2], [3, 7], [1, 9], [9, 1]]
    one = objective(mu[:2], counts[:2])
    two = objective(mu[2:], counts[2:])
    grouped = objective(mu, counts, ["A", "A", "B", "B"], ["D1", "D1", "D2", "D2"])
    torch.testing.assert_close(grouped.relative, (one.relative + two.relative) / 2)
    crossed = objective(mu, counts)
    assert abs(crossed.relative.item() - grouped.relative.item()) > .2


def test_donor_equal_weight_not_spot_or_count_weighted():
    mu = [[.8, .2], [.3, .7], [.01, .99], [.99, .01]]
    counts = [[8, 2], [3, 7], [100, 900], [900, 100]]
    one = objective(mu[:2], counts[:2])
    two = objective(mu[2:], counts[2:])
    answer = objective(mu, counts, ["A", "A", "B", "B"], ["D1", "D1", "D2", "D2"])
    torch.testing.assert_close(answer.loss, (one.loss + two.loss) / 2)
    repeated = objective(mu + mu[:2], counts + counts[:2],
                         ["A", "A", "B", "B", "C", "C"],
                         ["D1", "D1", "D2", "D2", "D1", "D1"])
    torch.testing.assert_close(repeated.loss, answer.loss)
    assert repeated.tissue_count == 3 and repeated.donor_count == 2


def test_global_count_scale_cancels_after_tissue_mass_normalization():
    mu = [[.7, .3], [.3, .7]]
    counts = torch.tensor([[3, 2], [4, 9]], dtype=DTYPE)
    a, b = objective(mu, counts), objective(mu, counts * 19)
    torch.testing.assert_close(a.composition, b.composition, atol=1e-14, rtol=0)
    torch.testing.assert_close(a.relative, b.relative, atol=1e-14, rtol=0)


def test_exposure_changes_spatial_probabilities_but_not_composition():
    a = objective([[.5, .5], [.5, .5]], [[8, 0], [2, 0]])
    b = objective([[.5, .5], [.5, .5]], [[8, 0], [2, 0]],
                  exposure=torch.tensor([1., 1.], dtype=DTYPE))
    assert a.relative.item() < b.relative.item()
    torch.testing.assert_close(a.composition, b.composition)


def test_measured_zero_is_candidate_but_missing_is_excluded():
    kwargs = {"exposure": torch.tensor([1., 1.], dtype=DTYPE)}
    counts = [[4, 1], [0, 1]]
    zeros = objective([[.5, .5], [.5, .5]], counts, **kwargs)
    mask = torch.tensor([[True, True], [False, True]])
    missing = objective([[.5, .5], [.5, .5]], counts, measured_mask=mask, **kwargs)
    assert zeros.relative.item() > missing.relative.item()
    assert zeros.composition.item() > missing.composition.item()


def test_missing_and_padding_nan_payload_does_not_change_output_or_gradient():
    log_mu = torch.tensor([[.8, .2], [.3, .7]], dtype=DTYPE).log().requires_grad_()
    counts = torch.tensor([[8, 2], [3, 7]], dtype=DTYPE)
    a = spatial_objective(log_mu, counts, [1, 1], [2, 2])
    a.loss.backward()
    grad = log_mu.grad.clone()
    padded = torch.cat([log_mu.detach(), torch.full((1, 2), float("nan"), dtype=DTYPE)]).requires_grad_()
    yy = torch.cat([counts, torch.full((1, 2), float("nan"), dtype=DTYPE)])
    b = spatial_objective(padded, yy, [1, 1, 1], [2, 2, 999],
                          spot_mask=torch.tensor([True, True, False]))
    b.loss.backward()
    torch.testing.assert_close(a.loss, b.loss)
    torch.testing.assert_close(grad, padded.grad[:2])
    assert torch.isfinite(padded.grad).all() and padded.grad[2].count_nonzero() == 0


def test_unmeasured_gene_is_never_imputed_as_zero():
    a = objective([[.8, .2], [.3, .7]], [[8, 2], [3, 7]])
    mu = torch.tensor([[.8, .2, float("nan")], [.3, .7, float("nan")]], dtype=DTYPE)
    counts = torch.tensor([[8., 2., float("nan")], [3., 7., -999.]], dtype=DTYPE)
    mask = torch.tensor([[True, True, False], [True, True, False]])
    b = spatial_objective(mu.log(), counts, [1, 1], [2, 2], measured_mask=mask)
    torch.testing.assert_close(a.loss, b.loss)


def test_correct_correspondence_beats_wrong_pairing_in_spatial_loss():
    mu = [[.95, .05], [.05, .95]]
    counts = [[19, 1], [1, 19]]
    matched = objective(mu, counts)
    wrong = objective(list(reversed(mu)), counts)
    assert matched.relative.item() < wrong.relative.item()
    assert matched.composition.item() < wrong.composition.item()


def test_joint_spot_permutation_equivariance_and_finite_nonzero_gradients():
    logits = torch.tensor([[1.3, -.4], [.2, .5], [-.9, 1.8]], dtype=DTYPE, requires_grad=True)
    yy = torch.tensor([[8., 2.], [4., 3.], [1., 9.]], dtype=DTYPE)
    a = spatial_objective(logits.log_softmax(1), yy, [1, 1, 1], [2, 2, 2])
    order = torch.tensor([2, 0, 1])
    b = spatial_objective(logits[order].log_softmax(1), yy[order], [1, 1, 1], [2, 2, 2])
    torch.testing.assert_close(a.loss, b.loss)
    a.loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > .01


def test_singleton_tissue_has_zero_relative_loss_and_disabled_rank_is_exact():
    a = objective([[.7, .3]], [[7, 3]], lambda_relative=0.)
    assert a.relative.item() == 0
    torch.testing.assert_close(a.loss, a.composition, rtol=0, atol=0)


@pytest.mark.parametrize("counts", [[[1, -1], [1, 1]], [[.1, .9], [1, 1]],
                                    [[float("nan"), 1], [1, 1]]])
def test_non_raw_negative_nonfinite_or_fractional_counts_rejected(counts):
    with pytest.raises(ValueError):
        objective([[.5, .5], [.5, .5]], counts)


def test_zero_exposure_positive_counts_and_ambiguous_tissue_donor_rejected():
    with pytest.raises(ValueError, match="positive exposure"):
        objective([[.5, .5], [.5, .5]], [[1, 2], [1, 2]], exposure=torch.tensor([0., 1.]))
    with pytest.raises(ValueError, match="multiple donors"):
        objective([[.5, .5], [.5, .5]], [[1, 2], [1, 2]], ["T", "T"], ["D1", "D2"])
    with pytest.raises(ValueError, match="no measured positive count"):
        objective([[.5, .5], [.5, .5]], [[0, 0], [0, 0]])

"""Synthetic qualification; no real data fits or artifacts are created."""
import numpy as np
import pytest
import torch
from torch import nn

from src.bgb_en.pathology import (PathologyEncoder, build_geometry,
    permute_coordinate_assignment, spatial_knn, spatial_segments)


class SyntheticCausalScan(nn.Module):
    """Explicit differentiable CPU test operator; never named/used as Mamba."""
    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, width)
        self.calls = 0

    def forward(self, packed):
        self.calls += 1
        state = torch.zeros_like(packed[:, 0])
        outputs = []
        for time in range(packed.shape[1]):
            state = .6*state+self.project(packed[:, time])
            outputs.append(state)
        return torch.stack(outputs, 1)


def fixture(input_dim=12):
    torch.manual_seed(919)
    x = torch.randn(2, 2, 8, input_dim)
    coords = torch.tensor([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [7, 7], [8, 7]], dtype=torch.float32)
    coords = coords[None, None].expand(2, 2, -1, -1).clone()
    mask = torch.ones(2, 2, 8, dtype=torch.bool)
    mask[1, 1, :] = False
    mask[0, 1, 6:] = False
    return x, coords, mask


def model(arm, input_dim=12):
    return PathologyEncoder(arm, input_dim=input_dim, width=8, embedding_dim=5,
            state_dim=8, coordinate_step=1,
            test_only_scan=SyntheticCausalScan(8) if arm in ("P3", "P4", "P5", "P6") else None)


@pytest.mark.parametrize("arm", ["P1", "P2", "P3", "P4", "P5", "P6"])
def test_forward_gradient_and_padding(arm):
    x, coords, mask = fixture()
    x.requires_grad_(True)
    net = model(arm)
    output = net(x, coords, mask, return_details=True)
    assert output["embedding"].shape == (2, 5)
    assert output["patch_states"].shape == (2, 2, 8, 8)
    assert torch.equal(output["patch_states"][~mask], torch.zeros_like(output["patch_states"][~mask]))
    output["embedding"].square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[mask].abs().sum() > 0
    assert x.grad[~mask].abs().sum() == 0
    assert net.project[0].weight.grad.abs().sum() > 0
    assert output["attention"][~mask].abs().sum() == 0
    if net.spatial is not None:
        assert net.spatial.scan.calls == 1
        assert net.spatial.scan.project.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", ["P1", "P2", "P3", "P4", "P6"])
def test_joint_row_permutation_is_equivariant(arm):
    x, coords, mask = fixture()
    net = model(arm).eval()
    permutation = torch.tensor([7, 2, 4, 0, 6, 1, 5, 3])
    initial = net(x, coords, mask, return_details=True)
    permuted = net(x[:, :, permutation], coords[:, :, permutation], mask[:, :, permutation], return_details=True)
    torch.testing.assert_close(initial["embedding"], permuted["embedding"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(initial["patch_states"][:, :, permutation], permuted["patch_states"], atol=2e-6, rtol=2e-6)


def test_nonspatial_sequence_uses_order_and_not_coordinates():
    x, coords, mask = fixture()
    net = model("P5").eval()
    original = net(x, coords, mask)
    torch.testing.assert_close(original, net(x, coords*0+float("nan"), mask))
    permutation = torch.tensor([7, 2, 4, 0, 6, 1, 5, 3])
    assert not torch.allclose(original, net(x[:, :, permutation], None, mask[:, :, permutation]), atol=1e-6)


@pytest.mark.parametrize("arm", ["P1", "P2", "P3", "P4", "P5", "P6"])
def test_slide_reset_equal_aggregation_and_padding_nan(arm):
    x, coords, mask = fixture()
    net = model(arm).eval()
    whole = net(x[:1], coords[:1], mask[:1], return_details=True)
    first = net(x[:1, :1], coords[:1, :1], mask[:1, :1])
    second = net(x[:1, 1:], coords[:1, 1:], mask[:1, 1:])
    torch.testing.assert_close(whole["embedding"], (first+second)/2, atol=2e-6, rtol=2e-6)
    corrupted = x.clone(); corrupted[~mask] = float("nan")
    badcoords = coords.clone(); badcoords[~mask] = float("nan")
    torch.testing.assert_close(net(x, coords, mask), net(corrupted, badcoords, mask), atol=2e-6, rtol=2e-6)


def test_runs_reset_at_real_holes_and_do_not_invent_points():
    coords = np.array([[0, 0], [1, 0], [4, 0], [4, 1], [4, 3]])
    runs = spatial_segments(coords, 1)
    assert np.array_equal(np.bincount(np.concatenate(runs)), np.full(5, 4))
    for run in runs:
        if len(run) > 1:
            assert np.all(np.abs(np.diff(coords[run], axis=0)).sum(-1) == 1)
    assert max(map(len, runs)) == 2


def test_graph_edges_are_real_local_and_coordinate_ties_stable():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [20, 20]])
    index, mask = spatial_knn(coords, 1, k=3)
    assert not mask[4].any()
    for row, (neighbors, valid) in enumerate(zip(index, mask)):
        assert row not in neighbors[valid]
        assert np.all(np.linalg.norm(coords[neighbors[valid]]-coords[row], axis=1) <= 1.5)
    permutation = np.array([4, 1, 3, 0, 2])
    pi, pm = spatial_knn(coords[permutation], 1, k=2)
    base, bm = spatial_knn(coords, 1, k=2)
    for newrow, oldrow in enumerate(permutation):
        assert np.array_equal(permutation[pi[newrow, pm[newrow]]], base[oldrow, bm[oldrow]])


def test_p6_reassigns_coordinate_feature_pairing_but_is_row_invariant():
    coords = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]])
    shuffled = permute_coordinate_assignment(coords, 11)
    assert not (coords == shuffled).all(1).any()
    assert set(map(tuple, coords)) == set(map(tuple, shuffled))
    permutation = np.array([4, 1, 3, 0, 2])
    assert np.array_equal(permute_coordinate_assignment(coords[permutation], 11), shuffled[permutation])


@pytest.mark.parametrize("arm", ["P2", "P3", "P4", "P6"])
def test_invalid_spatial_geometry_fails_closed(arm):
    with pytest.raises(ValueError, match="coordinate_step"):
        PathologyEncoder(arm, input_dim=12, width=8)
    x, coords, mask = fixture()
    net = model(arm)
    with pytest.raises(ValueError, match="requires original"):
        net(x, None, mask)
    duplicate = coords.clone(); duplicate[0, 0, 1] = duplicate[0, 0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        net(x, duplicate, mask)
    invalid = coords.clone(); invalid[0, 0, 1] = float("nan")
    with pytest.raises(ValueError, match="invalid real"):
        net(x, invalid, mask)


def test_p3_rejects_nonlattice_and_constant_coordinates():
    x, coords, mask = fixture()
    coords[0, 0, 1, 0] += .25
    with pytest.raises(ValueError, match="lattice"):
        model("P3")(x, coords, mask)
    with pytest.raises(ValueError, match="duplicate"):
        model("P3")(x, torch.zeros_like(coords), mask)


def test_mask_empty_patient_and_stale_geometry_fail_closed():
    x, coords, mask = fixture()
    empty = mask.clone(); empty[0] = False
    with pytest.raises(ValueError, match="at least one"):
        model("P1")(x, coords, empty)
    net = model("P2")
    geometry = net.geometry(coords, mask)
    changed = mask.clone(); changed[0, 0, 0] = False
    with pytest.raises(ValueError, match="identity mismatch"):
        net(x, coords, changed, geometry=geometry)
    mismatched = coords.clone(); mismatched[0, 0, 0, 0] += .5
    with pytest.raises(ValueError, match="coordinates or arm"):
        net(x, mismatched, mask, geometry=geometry)


def test_uni2_dimension_and_all_unmasked_parameter_gradients():
    x, coords, mask = fixture(1536)
    net = model("P4", input_dim=1536)
    net(x, coords, mask).square().sum().backward()
    assert net.graph[0].neighbor_project.weight.grad.abs().sum() > 0
    assert net.mil.score.weight.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in net.parameters())


def test_official_backend_never_silently_falls_back():
    if __import__("importlib").util.find_spec("mamba_ssm") is None:
        with pytest.raises(ModuleNotFoundError):
            PathologyEncoder("P3", input_dim=12, width=8, coordinate_step=1)


@pytest.mark.parametrize("arm", ["P2", "P3", "P4", "P6"])
def test_adjacent_windows_reset_geometry_without_becoming_slides(arm):
    torch.manual_seed(78)
    x = torch.randn(1, 1, 4, 12)
    coords = torch.tensor([[[[0., 0.], [1., 0.], [2., 0.], [3., 0.]]]])
    mask = torch.ones(1, 1, 4, dtype=torch.bool)
    regions = torch.tensor([[[0, 0, 1, 1]]])
    net = model(arm)
    geometry = net.geometry(coords, mask, regions)
    assert geometry.nonempty_slides == 1 and geometry.real_regions == 2
    assert geometry.longest_segment <= 2
    if geometry.neighbor is not None:
        for row, neighbors in enumerate(geometry.neighbor):
            assert set(regions.reshape(-1)[neighbors[neighbors >= 0]].tolist()) <= {int(regions.reshape(-1)[row])}
    original = net(x, coords, mask, region_ids=regions, return_details=True)
    modified = x.clone(); modified[:, :, 2:] += 20
    changed = net(modified, coords, mask, region_ids=regions, return_details=True)
    torch.testing.assert_close(original["patch_states"][:, :, :2], changed["patch_states"][:, :, :2])
    permutation = torch.tensor([3, 1, 0, 2])
    permuted = net(x[:, :, permutation], coords[:, :, permutation], mask[:, :, permutation],
                   region_ids=regions[:, :, permutation], return_details=True)
    torch.testing.assert_close(original["embedding"], permuted["embedding"], atol=2e-6, rtol=2e-6)
    assert original["slide_embedding"].shape[1] == 1
    with pytest.raises(ValueError, match="coordinates or arm"):
        net(x, coords, mask, region_ids=torch.zeros_like(regions), geometry=geometry)


def test_region_ids_do_not_split_nonspatial_sequence_or_change_slide_pooling():
    x, coords, mask = fixture()
    regions = torch.arange(x.shape[-2]).expand_as(mask).clone()
    for arm in ("P1", "P5"):
        net = model(arm)
        base = net(x, coords, mask)
        regional = net(x, coords, mask, region_ids=regions, return_details=True)
        torch.testing.assert_close(base, regional["embedding"])
        assert regional["diagnostics"]["region_scan_reset"] is False
        assert regional["diagnostics"]["window_pooling"] is False

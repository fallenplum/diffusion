"""Physical-neighborhood, padding and train-only dictionary qualification."""
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
import torch

from src.phase22_spatial_bridge.spatial_encoder import (
    LinearCompositionEncoder, LocalSpatialEncoder, physical_neighbors,
)
from src.phase22_spatial_bridge.molecular_dictionary import (
    DictionaryTrainingContract, counts_sha256, sequence_sha256, fit_molecular_dictionary,
)


def dictionary(states=32, genes=17):
    values = torch.arange(1, states * genes + 1, dtype=torch.float32).reshape(states, genes)
    return values / values.sum(1, keepdim=True)


def test_declared_architecture_budget_fixed_dictionary_and_simplexes():
    model = LocalSpatialEncoder(dictionary(), radius_um=160.)
    assert len(model.blocks) == 2
    assert model.input_projection.in_features == 1536
    assert model.input_projection.out_features == 128
    assert model.blocks[0].heads == 4
    assert model.blocks[0].ffn[0].out_features == 256
    assert model.neighbor_cap == 16
    assert sum(p.numel() for p in model.parameters()) < 1_000_000
    assert "dictionary" not in dict(model.named_parameters())
    prediction = model.eval()(torch.randn(5, 1536), torch.tensor([[i * 40., 0.] for i in range(5)]), ["A"] * 5)
    torch.testing.assert_close(prediction.states.sum(1), torch.ones(5))
    torch.testing.assert_close(prediction.mu.sum(1), torch.ones(5))
    torch.testing.assert_close(prediction.log_mu.exp(), prediction.mu)


def test_no_cross_wsi_edges_and_radius_in_micrometres():
    coords = torch.tensor([[0., 0.], [10., 0.], [0., 1.], [100., 100.]])
    ids = ["A", "A", "B", "A"]
    neighbors, allowed = physical_neighbors(coords, ids, torch.ones(4, dtype=torch.bool),
                                             radius_um=15., neighbor_cap=3)
    for i in range(len(coords)):
        for j in neighbors[i][allowed[i]].tolist():
            assert ids[i] == ids[j]
            assert torch.linalg.vector_norm(coords[i] - coords[j]) <= 15.
    assert allowed[3].sum() == 1 and neighbors[3][allowed[3]].item() == 3
    torch.manual_seed(4)
    model = LocalSpatialEncoder(dictionary(), radius_um=15.).eval()
    features = torch.randn(4, 1536)
    baseline = model(features, coords, ids).states
    changed = features.clone()
    changed[2] = changed[2] * 99 + 11
    intervention = model(changed, coords, ids).states
    torch.testing.assert_close(baseline[[0, 1, 3]], intervention[[0, 1, 3]], rtol=0, atol=0)


def test_spot_permutation_equivariance_even_at_distance_ties_and_cap():
    torch.manual_seed(7)
    model = LocalSpatialEncoder(dictionary(), radius_um=120., neighbor_cap=4).eval()
    # A regular lattice has many exact distance ties at the neighbor cutoff.
    coords = torch.tensor([[x * 40., y * 40.] for x in range(4) for y in range(4)])
    features = torch.randn(16, 1536)
    ids = ["A"] * 16
    order = torch.tensor([8, 6, 14, 1, 0, 12, 11, 15, 4, 3, 13, 7, 2, 10, 9, 5])
    first = model(features, coords, ids)
    second = model(features[order], coords[order], ids)
    torch.testing.assert_close(first.states[order], second.states, atol=2e-7, rtol=1e-6)
    torch.testing.assert_close(first.mu[order], second.mu, atol=2e-7, rtol=1e-6)


def test_all_padding_nan_payload_is_finite_zero_and_no_parameter_gradient():
    model = LocalSpatialEncoder(dictionary(), radius_um=112.)
    features = torch.full((4, 1536), float("nan"))
    coords = torch.full((4, 2), float("nan"))
    valid = torch.zeros(4, dtype=torch.bool)
    answer = model(features, coords, [None] * 4, valid)
    for value in (answer.states, answer.mu, answer.log_mu):
        assert torch.isfinite(value).all() and value.count_nonzero() == 0
    answer.states.sum().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all() and parameter.grad.count_nonzero() == 0


def test_padding_cannot_change_real_spots_and_a0_uses_same_fixed_dictionary():
    torch.manual_seed(19)
    models = [LocalSpatialEncoder(dictionary(), radius_um=180.).eval(),
              LinearCompositionEncoder(dictionary()).eval()]
    features = torch.randn(3, 1536)
    coords = torch.tensor([[0., 0.], [50., 0.], [100., 0.]])
    for model in models:
        first = model(features, coords, ["A"] * 3)
        second = model(torch.cat([features, torch.full((1, 1536), float("nan"))]),
                       torch.cat([coords, torch.full((1, 2), float("nan"))]),
                       ["A", "A", "A", None], torch.tensor([True, True, True, False]))
        torch.testing.assert_close(first.states, second.states[:3], atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(first.mu, first.states @ model.dictionary)


def test_unknown_radius_bad_dictionary_and_duplicate_physical_samples_fail():
    with pytest.raises(TypeError):
        LocalSpatialEncoder(dictionary())
    with pytest.raises(ValueError, match="rows must sum"):
        LinearCompositionEncoder(dictionary() * 2)
    with pytest.raises(ValueError, match="duplicate physical"):
        physical_neighbors(torch.zeros(2, 2), ["A", "A"], torch.ones(2, dtype=torch.bool), radius_um=112.)


def contract(counts, genes, donors, st, target, roles=None):
    return DictionaryTrainingContract(
        partition_role="ST-train", split_lock_sha256="d" * 64,
        counts_sha256=counts_sha256(counts), gene_names_sha256=sequence_sha256(genes),
        donor_sequence_sha256=sequence_sha256(donors),
        st_measured_genes_sha256=sequence_sha256(sorted(st)),
        target_measured_genes_sha256=sequence_sha256(sorted(target)),
        donor_roles=tuple(roles or [(d, "train") for d in sorted(set(donors))]),
    )


def count_example():
    counts = np.array([[9, 1, 2, 0, 99], [8, 2, 3, 0, 99], [3, 7, 1, 0, 99],
                       [1, 9, 4, 0, 99], [5, 5, 2, 0, 99], [0, 0, 0, 0, 99],
                       [7, 3, 2, 0, 99], [2, 8, 4, 0, 99]], dtype=np.int64)
    genes = ["g1", "g2", "g3", "zero_measured", "not_in_target"]
    donors = ["D1"] * 6 + ["D2"] * 2
    st = set(genes)
    target = set(genes[:-1])
    return counts, genes, donors, st, target


def fit_example(counts, genes, donors, st, target, lock=None, **kwargs):
    lock = lock or contract(counts, genes, donors, st, target)
    return fit_molecular_dictionary(counts, genes, donors, st_measured_genes=st,
        target_measured_genes=target, contract=lock, n_states=2,
        seed=31, max_iter=1500, tolerance=1e-5, **kwargs)


def test_dictionary_simplex_measured_intersection_zero_handling_and_donor_balance():
    args = count_example()
    fit = fit_example(*args)
    np.testing.assert_allclose(fit.dictionary.sum(1), 1., atol=1e-14)
    assert np.isfinite(fit.dictionary).all() and (fit.dictionary >= 0).all()
    assert set(fit.gene_names_private) == {"g1", "g2", "g3"}
    assert fit.metadata["training_donors"] == 2
    assert fit.metadata["profiles_per_donor"] == 2
    assert fit.metadata["total_profiles"] == 4
    assert fit.metadata["zero_common_gene_profiles_excluded"] == 1
    ids = np.array(args[2])[fit.sampled_spot_indices_private]
    assert (ids == "D1").sum() == (ids == "D2").sum() == 2
    assert not fit.metadata["survival_labels_read"]


def test_dictionary_deterministic_seed_sparse_inputs_and_count_scale():
    counts, genes, donors, st, target = count_example()
    a = fit_example(counts, genes, donors, st, target)
    b = fit_example(counts, genes, donors, st, target)
    c = fit_example(sparse.csr_matrix(counts), genes, donors, st, target)
    d = fit_example(counts * 11, genes, donors, st, target)
    np.testing.assert_array_equal(a.dictionary, b.dictionary)
    np.testing.assert_allclose(a.dictionary, c.dictionary, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(a.dictionary, d.dictionary, rtol=1e-12, atol=1e-12)


def test_non_train_donor_input_and_altered_provenance_fail_closed():
    counts, genes, donors, st, target = count_example()
    lock = contract(counts, genes, donors, st, target,
                    roles=[("D1", "train"), ("D2", "test")])
    with pytest.raises(ValueError, match="non-train"):
        fit_example(counts, genes, donors, st, target, lock)
    clean_lock = contract(counts, genes, donors, st, target)
    changed = counts.copy()
    changed[0, 0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        fit_example(changed, genes, donors, st, target, clean_lock)
    with pytest.raises(ValueError, match="ST-train"):
        fit_example(counts, genes, donors, st, target, replace(clean_lock, partition_role="ST-test"))
    with pytest.raises(ValueError, match="split-lock"):
        fit_example(counts, genes, donors, st, target, replace(clean_lock, split_lock_sha256="unknown"))


def test_dictionary_rejects_fractional_data_empty_intersection_and_empty_donor():
    counts, genes, donors, st, target = count_example()
    fractional = counts.astype(float)
    fractional[0, 0] += .25
    with pytest.raises(ValueError, match="integer raw counts"):
        fit_example(fractional, genes, donors, st, target)
    with pytest.raises(ValueError, match="no common measured"):
        fit_example(counts, genes, donors, st, {"absent"})
    empty_donor = counts.copy()
    empty_donor[-2:, :4] = 0
    with pytest.raises(ValueError, match="no positive"):
        fit_example(empty_donor, genes, donors, st, target)


def test_train_only_gene_cap_and_target_measurement_hash_guard():
    counts, genes, donors, st, target = count_example()
    fit = fit_example(counts, genes, donors, st, target, gene_cap=2)
    assert fit.dictionary.shape == (2, 2)
    assert set(fit.gene_names_private).issubset(target)
    lock = contract(counts, genes, donors, st, target)
    with pytest.raises(ValueError, match="target_measured_genes_sha256"):
        fit_example(counts, genes, donors, st, {"g1", "g2"}, lock)
    with pytest.raises(ValueError, match="gene_cap"):
        fit_example(counts, genes, donors, st, target, gene_cap=1025)

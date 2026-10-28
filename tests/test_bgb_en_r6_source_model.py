"""R6 mathematical, donor-isolation and complete-transfer tests; zero fits."""
from copy import deepcopy
import inspect
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from src.bgb_en.r6_source_model import (
    PROGRAMS, SourceProgramModel, SourceFitStatistics, donor_derangement_feasibility,
    lodo_derangement_feasibility, wrong_donor_permutation, joint_permutation_audit,
    local_neighbor_indices, spot_alignment_loss, neighborhood_difference_loss, source_loss,
)
from src.bgb_en.r6_omics import (
    mapping_from_programs, GeneValueBank, GeneEvidenceTransform, SparseProgramEncoder,
)


def unit_features(n=9, d=1536, seed=61031):
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=generator), dim=-1)


def model(arm="S4", **kwargs):
    return SourceProgramModel(arm, tissue_prior=[.3, .4, .3], init_seed=61031, **kwargs)


def gene_fixture(n=9):
    mapping = mapping_from_programs([
        dict(id=name, up_genes=[chr(65 + 2 * k)], down_genes=[chr(66 + 2 * k)])
        for k, name in enumerate(PROGRAMS)
    ])
    bank = GeneValueBank(mapping.gene_symbols, mapping.gene_symbols)
    stem = GeneEvidenceTransform()
    reader = SparseProgramEncoder(mapping, gene_encoder=stem)
    x = torch.arange(n * 8, dtype=torch.float32).reshape(n, 8) / 20
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, 1] = False
    x[~mask] = float("nan")
    return bank, reader, x, mask


def source_targets(n=9):
    y = torch.arange(n * 4, dtype=torch.float32).reshape(n, 4) / 7
    tissue = torch.arange(n) % 3
    return y, tissue


def test_visual_inference_has_no_tissue_truth_or_gene_measurement_argument():
    source = model()
    parameters = inspect.signature(source.forward).parameters
    assert not {"tissue_y", "tissue_truth", "gene_values", "program_targets"} & set(parameters)
    out = source(unit_features())
    assert out["embedding"].shape == (9, 64)
    assert out["queries"].shape == (9, 4, 64)
    assert out["program_predictions"].shape == (9, 4)
    assert torch.equal(out["tissue_probs"], out["tissue_probabilities"])
    with pytest.raises(TypeError):
        source(unit_features(), tissue_y=torch.arange(9) % 3)
    assert not source.descriptor()["tissue_truth_is_forward_input"]
    assert source.descriptor()["pretrained_status_requires_separate_fit_and_replay_receipt"]


def test_program_conditioning_gradient_reaches_predicted_tissue_route():
    source = model("S4")
    target, _ = source_targets()
    out = source(unit_features())
    F.huber_loss(out["program_predictions"], target).backward()
    assert source.tissue_head.weight.grad.abs().sum() > 0
    assert source.condition_gate.weight.grad.abs().sum() > 0
    assert source.condition_basis.grad.abs().sum() > 0
    additive = model("S3")
    F.huber_loss(additive(unit_features())["program_predictions"], target).backward()
    assert additive.tissue_head.weight.grad is None
    assert additive.condition_gate.weight.grad is None


def test_true_training_prior_recovers_static_queries_and_dynamic_gate_changes_them():
    source = model("S4")
    prior = source.train_tissue_prior.expand(3, -1)
    assert torch.equal(source.queries(prior), source.program_queries.unsqueeze(0).expand(3, -1, -1))
    condition = torch.eye(3)
    assert not torch.equal(source.queries(condition), source.queries(prior))
    with pytest.raises(ValueError, match="PROBABILITIES"):
        source.queries(torch.full((3, 3), .5))
    with pytest.raises(ValueError, match="TRAIN_ONLY_PRIOR"):
        SourceProgramModel("S4")


def test_frozen_arms_and_matched_initial_visual_backbone_are_explicit():
    random = model("S0")
    assert random.parameter_budget()["requires_grad"] == 0
    tissue, program, additive = model("S1"), model("S2"), model("S3")
    assert not tissue.program_queries.requires_grad and tissue.tissue_head.weight.requires_grad
    assert program.program_queries.requires_grad and not program.tissue_head.weight.requires_grad
    for name, value in additive.project.state_dict().items():
        assert torch.equal(value, tissue.project.state_dict()[name])
        assert torch.equal(value, program.project.state_dict()[name])
        assert torch.equal(value, random.project.state_dict()[name])
    assert not additive.parameter_budget()["same_effective_capacity_or_FLOPs_claimed"]


@pytest.mark.parametrize("x", [torch.zeros(2, 1536), torch.ones(2, 1536),
                               torch.full((2, 1536), float("nan")),
                               unit_features(2).double()])
def test_raw_zero_nonfinite_or_wrong_dtype_visual_evidence_is_rejected(x):
    with pytest.raises(ValueError, match="SOURCE_"):
        model("S3")(x)


def test_spatial_index_and_explicit_neighbor_features_produce_same_forward_and_gradients():
    source = model("S6")
    x = unit_features(4)
    graph = torch.tensor([[0, 1, -1], [0, 1, 2], [1, 2, 3], [2, 3, -1]])
    mask = graph >= 0
    explicit = x[graph.clamp_min(0)].clone()
    explicit[~mask] = float("nan")
    indexed = source(x, graph)
    indexed["program_predictions"].sum().backward()
    indexed_gradient = source.project[0].weight.grad.clone()
    source.zero_grad(set_to_none=True)
    supplied = source(x, neighbor_features=explicit, neighbor_mask=mask)
    supplied["program_predictions"].sum().backward()
    for key in ("embedding", "tissue_logits", "program_predictions", "queries"):
        torch.testing.assert_close(indexed[key], supplied[key], atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(indexed_gradient, source.project[0].weight.grad, atol=1e-6, rtol=2e-5)
    assert source.spatial_down.weight.grad.abs().sum() > 0


def test_missing_spatial_neighborhood_is_no_change_and_bad_scope_is_rejected():
    source = model("S6")
    x = unit_features(2)
    explicit = torch.full((2, 3, 1536), float("nan"))
    mask = torch.zeros(2, 3, dtype=torch.bool)
    torch.testing.assert_close(source.encode(x, neighbor_features=explicit, neighbor_mask=mask), source.project(x))
    with pytest.raises(ValueError, match="USE_INDEX_OR_EXPLICIT"):
        source(x)
    with pytest.raises(ValueError, match="INDEX_OUT_OF_RANGE"):
        source(x, torch.tensor([[0, 2], [0, 1]]))
    observed = mask.clone(); observed[0, 0] = True
    with pytest.raises(ValueError, match="NONFINITE_OBSERVED"):
        source(x, neighbor_features=explicit, neighbor_mask=observed)


def test_spatial_graph_remapping_cannot_use_heldout_or_other_section_rows():
    graph = np.array([[0, 1, -1], [0, 1, -1], [2, 3, -1], [2, 3, -1]])
    donors = np.array([0, 0, 1, 1])
    sections = np.array([0, 0, 2, 2])
    mapped = local_neighbor_indices(graph, np.array([3, 2]), section_codes=sections, donor_codes=donors)
    assert np.array_equal(mapped, np.array([[1, 0, -1], [1, 0, -1]]))
    bad = graph.copy(); bad[0, 1] = 2
    with pytest.raises(ValueError, match="HELDOUT_PARTITION"):
        local_neighbor_indices(bad, np.array([0, 1]), section_codes=sections, donor_codes=donors)
    bad = graph.copy(); bad[0, 1] = 2
    with pytest.raises(ValueError, match="SECTION_OR_DONOR"):
        local_neighbor_indices(bad, np.arange(4), section_codes=sections, donor_codes=donors)


def test_training_statistics_are_invariant_to_all_heldout_target_and_gene_changes():
    rng = np.random.default_rng(61031)
    targets = rng.normal(size=(12, 4))
    tissue = np.arange(12) % 3
    donor = np.repeat(np.arange(3), 4)
    genes = rng.normal(size=(12, 8)); gene_mask = np.ones_like(genes, bool)
    gene_mask[:8, 7] = False; genes[~gene_mask] = np.nan
    train = np.arange(8)
    before = SourceFitStatistics.fit(targets, tissue, donor, train, heldout_donors=[2],
                                     gene_values=genes, gene_mask=gene_mask)
    targets[8:] = 1e20; genes[8:] = -1e30; tissue[8:] = -99
    after = SourceFitStatistics.fit(targets, tissue, donor, train, heldout_donors=[2],
                                    gene_values=genes, gene_mask=gene_mask)
    for name in ("target_mean", "target_scale", "tissue_prior", "class_weights", "gene_mean", "gene_scale"):
        assert np.array_equal(getattr(before, name), getattr(after, name))
    assert before.gene_mean[7] == 0 and before.gene_scale[7] == 1
    assert before.audit()["source_gene_scaler_must_not_be_applied_to_TCGA_bulk"]
    with pytest.raises(ValueError, match="HELDOUT_DONOR"):
        SourceFitStatistics.fit(targets, tissue, donor, np.arange(9), heldout_donors=[2])


def test_source_transforms_preserve_true_missingness_and_uniform_training_scale():
    targets, tissue = source_targets(12)
    y = targets.numpy(); labels = tissue.numpy(); donor = np.repeat(np.arange(3), 4)
    train = np.arange(8)
    mask = np.ones_like(y, bool); mask[0, 1] = False; y[~mask] = np.nan
    stats = SourceFitStatistics.fit(y, labels, donor, train, valid_mask=mask)
    z = stats.transform_targets(y, mask)
    assert np.isfinite(z).all() and z[0, 1] == 0
    for k in range(4):
        assert abs(z[train, k][mask[train, k]].mean()) < 1e-14
        assert abs(z[train, k][mask[train, k]].std() - 1) < 1e-14
    assert stats.audit()["training_spot_weights"] == "uniform"


def test_donor_derangement_is_strict_reproducible_joint_and_leaves_nontraining_rows_unchanged():
    counts = [11, 8, 7, 6, 5]
    donor = np.repeat(np.arange(5), counts)
    tissue = np.arange(len(donor)) % 3
    library = np.exp(np.linspace(5, 11, len(donor)))
    train = np.flatnonzero(donor != 4)
    mapping, audit = wrong_donor_permutation(donor, tissue, library, train, seed=61)
    repeated, _ = wrong_donor_permutation(donor, tissue, library, train, seed=61)
    assert np.array_equal(mapping, repeated)
    assert np.array_equal(np.sort(mapping[train]), train)
    assert np.all(donor[mapping[train]] != donor[train])
    assert np.array_equal(mapping[donor == 4], np.flatnonzero(donor == 4))
    assert audit["final_tissue_matches"] >= audit["initial_tissue_matches"]
    assert not audit["outcomes_or_program_targets_used"] and not audit["dense_N_squared_allocation"]
    y = np.arange(len(donor) * 4).reshape(-1, 4).astype(float)
    mask = np.ones_like(y, bool); mask[1, 1] = False; y[~mask] = np.nan
    uniform = joint_permutation_audit(y, mask, mapping, train)
    assert uniform["unweighted_joint_vector_and_mask_multiset_preserved"]
    assert uniform["weighted_mean_max_abs_drift"] < 1e-12
    assert uniform["weighted_covariance_max_abs_drift"] < 1e-9
    weighted = joint_permutation_audit(y, mask, mapping, train, weights=np.arange(1, len(train) + 1))
    assert not weighted["weighted_marginal_preservation_claimed"]
    assert weighted["weighted_mean_max_abs_drift"] > 0


def test_impossible_strict_wrong_donor_control_blocks_instead_of_sampling_with_replacement():
    donor = np.array([0] * 6 + [1] * 2 + [2] * 2)
    assert not donor_derangement_feasibility(donor)["no_same_donor_bijection_possible"]
    with pytest.raises(ValueError, match="BLOCKED_STRICT"):
        wrong_donor_permutation(donor, np.arange(10) % 3, np.ones(10), np.arange(10), seed=1)
    boundary = np.repeat(np.arange(2), 5)
    mapping, _ = wrong_donor_permutation(boundary, np.arange(10) % 3, np.ones(10), np.arange(10), seed=1)
    assert np.all(boundary[mapping] != boundary)


def test_actual_anonymous_lodo_marginals_admit_six_strict_bijections():
    donor = np.repeat(np.arange(6), [1984, 4119, 1091, 1827, 932, 3058])
    audit = lodo_derangement_feasibility(donor)
    assert len(audit) == 6 and all(row["no_same_donor_bijection_possible"] for row in audit)
    assert [r["train_spots"] for r in audit] == [11027, 8892, 11920, 11184, 12079, 9953]


def test_wrong_donor_loss_moves_joint_program_targets_and_masks_but_never_tissue_labels():
    source = model("S5")
    out = source(unit_features(6))
    y, tissue = source_targets(6)
    mask = torch.ones_like(y, dtype=torch.bool); mask[1, 2] = False; y[~mask] = float("nan")
    donor = torch.tensor([0, 0, 1, 1, 2, 2])
    mapping = torch.tensor([2, 3, 4, 5, 0, 1])
    loss, audit = source_loss(source, out, y, tissue, valid_mask=mask,
                              program_target_permutation=mapping, training_donor_codes=donor)
    safe = torch.where(mask, y, torch.zeros_like(y))[mapping]
    permuted_mask = mask[mapping]
    expected = torch.where(permuted_mask, F.huber_loss(out["program_predictions"], safe, reduction="none"), 0.).sum(0)
    expected = (expected / permuted_mask.sum(0)).mean()
    expected_ce = F.cross_entropy(out["tissue_logits"], tissue, reduction="none").mean()
    torch.testing.assert_close(audit["program_huber"], expected)
    torch.testing.assert_close(audit["tissue_CE"], expected_ce)
    torch.testing.assert_close(loss, expected + .1 * expected_ce)
    with pytest.raises(ValueError, match="EVERY_ROW"):
        source_loss(source, out, y, tissue, valid_mask=mask,
                    program_target_permutation=torch.arange(6), training_donor_codes=donor)
    with pytest.raises(ValueError, match="EXPLICIT_WRONG"):
        source_loss(source, out, y, tissue, valid_mask=mask)


def test_additive_source_objective_equals_uniform_huber_plus_train_class_weighted_ce():
    source = model("S3")
    out = source(unit_features())
    y, tissue = source_targets()
    weights = torch.tensor([.8, 1., 1.3])
    loss, audit = source_loss(source, out, y, tissue, class_weights=weights)
    expected = F.huber_loss(out["program_predictions"], y)
    expected += .1 * F.cross_entropy(out["tissue_logits"], tissue, weight=weights, reduction="none").mean()
    torch.testing.assert_close(loss, expected)
    assert audit["training_spot_weights"] == "uniform"
    assert not audit["tissue_truth_was_forward_input"]


def test_auxiliary_gene_branch_uses_actual_values_without_changing_visual_prediction():
    bank, reader, raw, mask = gene_fixture()
    source = model("S7", gene_reader=reader)
    x = unit_features()
    before = source(x)["program_predictions"].detach().clone()
    gene_before = source.encode_gene_programs(bank(raw, mask))
    changed = raw.clone(); changed[:, 0] += 2
    gene_after = source.encode_gene_programs(bank(changed, mask))
    assert not torch.equal(gene_before["tokens"], gene_after["tokens"])
    assert torch.equal(before, source(x)["program_predictions"])
    assert sum(p.numel() for p in source.gene_reader.parameters()) == 9088
    y, tissue = source_targets()
    negatives = torch.stack(((torch.arange(9) + 1) % 9, (torch.arange(9) + 3) % 9), -1)
    loss, audit = source_loss(source, source(x), y, tissue, gene_programs=gene_before, negative_index=negatives)
    loss.backward()
    assert source.gene_reader.gene_encoder.value_encoder.weight.grad.abs().sum() > 0
    assert audit["alignment_audit"]["negatives_compare_same_program_only"]


def test_same_program_alignment_cannot_be_solved_by_static_program_identity():
    basis = torch.eye(4).unsqueeze(0).expand(5, -1, -1).clone().requires_grad_()
    mask = torch.ones(5, 4, dtype=torch.bool)
    negatives = torch.stack(((torch.arange(5) + 1) % 5, (torch.arange(5) + 2) % 5), -1)
    loss, audit = spot_alignment_loss(basis, basis, mask, negatives)
    torch.testing.assert_close(loss, loss.new_tensor(math.log(3)))
    assert audit["active_spot_programs"] == 20
    with pytest.raises(ValueError, match="SAME_SPOT"):
        spot_alignment_loss(basis, basis, mask, torch.arange(5)[:, None])


def test_alignment_uses_matched_spot_and_ignores_missing_program_nan_values():
    eye = torch.eye(4)[:, None, :].expand(-1, 4, -1).clone()
    mask = torch.ones(4, 4, dtype=torch.bool)
    negatives = ((torch.arange(4) + 1) % 4)[:, None]
    matched, _ = spot_alignment_loss(eye, eye, mask, negatives)
    wrong, _ = spot_alignment_loss(eye, eye.roll(1, 0), mask, negatives)
    assert matched < wrong
    missing = torch.full((4, 4, 4), float("nan"), requires_grad=True)
    empty, audit = spot_alignment_loss(missing, missing, torch.zeros_like(mask), negatives)
    assert empty.item() == 0 and torch.isfinite(empty)
    empty.backward()
    assert torch.isfinite(missing.grad).all() and missing.grad.abs().sum() == 0
    assert audit["no_valid_negative_anchors"]


def test_neighbor_loss_preserves_true_program_contrasts_instead_of_forcing_smoothness():
    truth = torch.arange(4, dtype=torch.float32)[:, None].expand(-1, 4).clone()
    graph = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]])
    mask = torch.ones_like(truth, dtype=torch.bool)
    exact, audit = neighborhood_difference_loss(truth.clone(), truth, mask, graph)
    shifted, _ = neighborhood_difference_loss(truth + 10, truth, mask, graph)
    oversmooth, _ = neighborhood_difference_loss(torch.ones_like(truth), truth, mask, graph)
    assert exact == 0 and shifted == 0 and oversmooth > 0
    assert audit["targets_are_neighbor_differences_not_forced_smoothness"]
    source = model("S6")
    y, tissue = source_targets(4)
    loss, parts = source_loss(source, source(unit_features(4), graph), y, tissue, true_neighbor_index=graph)
    assert torch.isfinite(loss) and parts["neighborhood_audit"]["active_spots"] == 4


def test_full_source_transfer_requires_tissue_and_queries_and_freezes_every_parameter():
    original = model("S4")
    artifact = original.transfer_state()
    assert "program_queries" in artifact["state_dict"] and "tissue_head.weight" in artifact["state_dict"]
    received = model("S4")
    audit = received.load_transfer_state(artifact)
    assert audit["source_frozen"] and not any(p.requires_grad for p in received.parameters())
    x = unit_features()
    assert torch.equal(original(x)["program_predictions"], received(x)["program_predictions"])
    assert received.training is False
    missing = deepcopy(artifact); del missing["state_dict"]["program_queries"]
    with pytest.raises(ValueError, match="FULL_STATE_MISSING"):
        model("S4").load_transfer_state(missing)
    with pytest.raises(ValueError, match="DIFFERENT_R6_TRANSFER"):
        model("S3").load_transfer_state(artifact)
    altered = deepcopy(artifact); altered["projection_only_transfer"] = True
    with pytest.raises(ValueError, match="TRANSFER_CONTRACT"):
        model("S4").load_transfer_state(altered)


def test_complete_gene_transfer_binds_signed_mapping_not_just_matching_tensor_shapes():
    _, reader, _, _ = gene_fixture()
    original = model("S7", gene_reader=reader)
    artifact = original.transfer_state()
    changed_mapping = reader.mapping.reindex_genes(torch.arange(7, -1, -1))
    changed_reader = SparseProgramEncoder(changed_mapping, gene_encoder=GeneEvidenceTransform())
    with pytest.raises(ValueError, match="TRANSFER_CONTRACT"):
        model("S7", gene_reader=changed_reader).load_transfer_state(artifact)
    _, reader2, _, _ = gene_fixture()
    received = model("S7", gene_reader=reader2)
    received.load_transfer_state(artifact)
    assert not any(p.requires_grad for p in received.gene_reader.parameters())
    for key, value in original.state_dict().items():
        assert torch.equal(value, received.state_dict()[key])


def test_no_survival_or_r5_dependency_is_imported_by_source_module():
    import src.bgb_en.r6_source_model as module
    text = inspect.getsource(module)
    assert "from .r5" not in text and "import r5" not in text
    assert "DSS" not in text and "PFI" not in text
    assert not any(isinstance(layer, (nn.Dropout, nn.modules.batchnorm._BatchNorm))
                   for layer in model("S4").modules())

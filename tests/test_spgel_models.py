"""Synthetic invariance / gradient tests; never evidence of source performance."""
import numpy as np
import pytest
import torch

from src.spgel.contracts import GeneBatch, ImageBatch
from src.spgel.losses import bidirectional_contrastive_loss, fixed_gene_mask
from src.spgel.models import OriginalGeneAttention, SpatialProgramEncoder, match_image_mlp_capacity


torch.set_num_threads(2)


def inputs(n=6, u=24):
    rng = np.random.default_rng(91)
    x = torch.tensor(rng.normal(size=(n, 1536)), dtype=torch.float32)
    x = x / x.norm(dim=-1, keepdim=True)
    neighbors = torch.stack((x, x.roll(1, 0), x.roll(2, 0)), 1)
    xy = torch.tensor([[[0., 0.], [100., 0.], [200., 0.]]] * n)
    image = ImageBatch(x, neighbors, xy, torch.ones((n, 3), dtype=torch.bool))
    values = torch.tensor(rng.normal(size=(n, u)), dtype=torch.float32)
    genes = GeneBatch(values, torch.ones_like(values, dtype=torch.bool), torch.arange(u))
    up, down = np.zeros((4, u), bool), np.zeros((4, u), bool)
    for k in range(4):
        up[k, (np.arange(6) + k * 3) % u] = True
        down[k, (np.arange(6) + k * 3 + 8) % u] = True
    return image, genes, up, down, [f"SYN_GENE_{g}" for g in range(u)]


def make(arm="S_A5", n=6):
    image, genes, up, down, symbols = inputs(n)
    model = SpatialProgramEncoder(arm, up, down, symbols, seed=31)
    return model, image, genes, up, down, symbols


def test_image_only_signature_and_prediction_have_no_gene_dependency():
    model, image, genes, *_ = make()
    first = model.image_only_predict(image)["program_predictions"]
    with pytest.raises(TypeError):
        model.image_only_predict(image, genes=genes)
    with pytest.raises(TypeError):
        model.image_only_predict(image, tissue_truth=torch.arange(6))
    with pytest.raises(TypeError):
        model.image_only_predict(image, library_size=torch.ones(6))
    prediction = first.square().mean()
    model.zero_grad(set_to_none=True)
    prediction.backward()
    assert model.backbone.project[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.gene_encoder.parameters())
    assert all(p.grad is None for p in model.gene_to_image.parameters())
    assert all(p.grad is None for p in model.image_to_gene.parameters())
    changed = GeneBatch(genes.values * 500., genes.observed, genes.gene_ids)
    model.encode_gene_programs(changed, paired_mode=True)
    assert torch.equal(first, model.image_only_predict(image)["program_predictions"])


def test_original_gene_joint_permutation_is_invariant():
    _, genes, up, down, symbols = inputs()
    torch.manual_seed(8)
    encoder = OriginalGeneAttention(up, down, symbols)
    a = encoder(genes)["tokens"]
    order = torch.randperm(len(symbols))
    permuted = GeneBatch(genes.values[:, order], genes.observed[:, order], genes.gene_ids[order])
    b = encoder(permuted)["tokens"]
    assert torch.allclose(a, b, atol=2e-6, rtol=2e-6)
    misaligned = GeneBatch(genes.values[:, order], genes.observed[:, order], genes.gene_ids)
    assert not torch.allclose(a, encoder(misaligned)["tokens"], atol=1e-6, rtol=1e-6)


def test_global_mask_removes_gene_from_every_program_copy():
    _, genes, up, down, symbols = inputs()
    encoder = OriginalGeneAttention(up, down, symbols)
    hidden = torch.zeros_like(genes.observed)
    hidden[:, 8] = True
    assert (up[:, 8] | down[:, 8]).sum() >= 2
    masked = genes.masked(hidden)
    changed = genes.values.clone()
    changed[:, 8] += 100000
    other = GeneBatch(changed, genes.observed, genes.gene_ids).masked(hidden)
    a, b = encoder(masked, return_attention=True), encoder(other, return_attention=True)
    assert torch.equal(a["tokens"], b["tokens"])
    assert torch.count_nonzero(a["up_attention"][..., 8]) == 0
    assert torch.count_nonzero(a["down_attention"][..., 8]) == 0
    assert not torch.equal(encoder(genes)["tokens"], a["tokens"])


def test_missing_is_not_a_measured_zero_and_all_missing_is_finite():
    _, genes, up, down, symbols = inputs()
    encoder = OriginalGeneAttention(up, down, symbols)
    zero = GeneBatch(torch.zeros_like(genes.values), genes.observed, genes.gene_ids)
    missing = GeneBatch(torch.full_like(genes.values, float("nan")), torch.zeros_like(genes.observed), genes.gene_ids)
    a, b = encoder(zero), encoder(missing)
    assert a["program_mask"].all()
    assert not b["program_mask"].any()
    assert torch.isfinite(b["tokens"]).all() and torch.count_nonzero(b["tokens"]) == 0
    assert torch.count_nonzero(a["tokens"]) > 0


def test_candidate_specific_dual_reads_equal_separate_pair_evaluation():
    model, image, genes, *_ = make()
    out = model.image_only_predict(image)
    ir = torch.tensor([[0, 0, 0], [1, 1, 1]])
    gr = torch.tensor([[0, 1, 2], [1, 2, 3]])
    together = model.paired_scores(out, genes, ir, gr, paired_mode=True, return_details=True)
    for k in range(3):
        one = model.paired_scores(out, genes, ir[:, k:k+1], gr[:, k:k+1], paired_mode=True, return_details=True)
        assert torch.allclose(together["scores"][:, k:k+1], one["scores"], atol=1e-6, rtol=1e-6)
        assert torch.allclose(together["conditioned_gene_tokens"][:, k:k+1], one["conditioned_gene_tokens"], atol=1e-6, rtol=1e-6)
    assert together["condition_is_recomputed_for_each_candidate"] is True
    assert not torch.allclose(together["conditioned_image_tokens"][:, 0], together["conditioned_image_tokens"][:, 1],
                              atol=1e-7, rtol=1e-7)


def test_paired_scores_cannot_be_called_implicitly_or_use_self_negative():
    model, image, genes, *_ = make()
    out = model(image)
    rows = torch.arange(6)[:, None]
    with pytest.raises(ValueError, match="PAIRED_SCORING_MODE_REQUIRED"):
        model.paired_scores(out, genes, rows, rows)
    with pytest.raises(ValueError, match="GENES_REQUIRE_EXPLICIT_PAIRED_MODE"):
        model.encode_gene_programs(genes)
    with pytest.raises(ValueError, match="CONTRASTIVE_SELF_IS_NEGATIVE"):
        bidirectional_contrastive_loss(model, out, genes, rows)


def test_both_direction_query_projections_receive_real_itc_gradient():
    model, image, genes, *_ = make(n=6)
    out = model(image)
    negative = torch.stack((torch.arange(6).roll(1), torch.arange(6).roll(2)), 1)
    loss, audit = bidirectional_contrastive_loss(model, out, genes, negative)
    assert torch.isfinite(loss) and audit["active_image_anchors"] == audit["active_gene_anchors"] == 6
    loss.backward()
    for parameter in (model.gene_to_image.weight, model.image_to_gene.weight,
                      model.window.key.weight, model.gene_encoder.identity_key.weight,
                      model.backbone.project[0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_common_modules_initialization_and_exact_dual_parameter_difference():
    _, _, up, down, symbols = inputs()
    models = {a: SpatialProgramEncoder(a, up, down, symbols, seed=47) for a in ("S_A2", "S_A3", "S_A4", "S_A5")}
    for a in models:
        for name, parameter in models["S_A2"].backbone.state_dict().items():
            assert torch.equal(parameter, models[a].backbone.state_dict()[name])
    for name, parameter in models["S_A4"].state_dict().items():
        assert torch.equal(parameter, models["S_A5"].state_dict()[name])
    assert models["S_A5"].descriptor()["parameters"] - models["S_A4"].descriptor()["parameters"] == 2 * 128 * 128


def test_image_mlp_capacity_match_uses_only_static_parameter_counts():
    _, _, up, down, symbols = inputs()
    info = match_image_mlp_capacity(up, down, symbols)
    assert abs(info["parameters"] - info["target_A5_parameters"]) / info["target_A5_parameters"] <= .01
    assert info["selection_uses_labels"] is False and info["dummy_or_zero_gradient_parameters_added"] is False


def test_graph_padding_never_changes_prediction_and_real_neighbor_can():
    model, image, _, up, down, symbols = make("S_A2")
    mask = image.neighbor_mask.clone(); mask[:, -1] = False
    a = ImageBatch(image.features, image.neighbor_features, image.neighbor_xy_um, mask)
    neighbor = image.neighbor_features.clone(); neighbor[:, -1] = 2000
    b = ImageBatch(image.features, neighbor, image.neighbor_xy_um, mask)
    assert torch.equal(model(a)["program_predictions"], model(b)["program_predictions"])
    neighbor = image.neighbor_features.clone(); neighbor[:, 1] = neighbor[:, 1].roll(1, 0)
    c = ImageBatch(image.features, neighbor, image.neighbor_xy_um, image.neighbor_mask)
    assert not torch.allclose(model(image)["program_predictions"], model(c)["program_predictions"])
    local = SpatialProgramEncoder("S_A1", up, down, symbols)
    assert torch.equal(local(image)["program_predictions"], local(c)["program_predictions"])


def test_masked_gene_utility_never_uses_unmeasured_targets():
    _, genes, *_ = inputs()
    observed = genes.observed.clone(); observed[:, 0] = False
    hidden = fixed_gene_mask(observed, seed=31)
    assert not (hidden & ~observed).any()
    with pytest.raises(ValueError, match="ONLY_MEASURED_GENES"):
        GeneBatch(genes.values, observed, genes.gene_ids).masked(torch.ones_like(observed))


def test_disabled_auxiliary_utilities_have_explicit_support_and_partition_guards():
    from src.spgel.losses import optional_pair_matching_bce, optional_spatial_huber
    logits = torch.tensor([2., -2.], requires_grad=True)
    loss = optional_pair_matching_bce(logits, torch.tensor([1., 0.]), torch.ones(2, dtype=torch.bool))
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1]
    values = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    edge = torch.tensor([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="CROSSES_PARTITION"):
        optional_spatial_huber(values, edge, torch.ones(2), torch.tensor([True, True, False]))
    penalty = optional_spatial_huber(values, edge[:, :1], torch.ones(1), torch.tensor([True, True, False]))
    penalty.backward()
    assert values.grad[2].count_nonzero() == 0 and values.grad[:2].count_nonzero() > 0

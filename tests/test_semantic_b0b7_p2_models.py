"""Synthetic ordinary-attention controls, never patient training."""
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.semantic_b0b7.models import ImageInputs, ModelConfig, OmicsInputs, SemanticRepresentation
from src.semantic_b0b7.p1r_models import RepairSpec
from src.semantic_b0b7.p2_models import CrossAttentionRepresentation
from src.semantic_b0b7.pretext import auxiliary_loss

torch.set_num_threads(2)


def fixture(source="B2_MATCHED"):
    gen = torch.Generator().manual_seed(1334)
    rand = lambda *shape: torch.randn(*shape, generator=gen)
    c = ModelConfig(patch_dim=24, conch_text_dim=24, titan_native_dim=24,
        titan_text_dim=24, titan_score_dim=5, clinical_dim=6, programs=5,
        slots=3, width=12, heads=3, rank=2, dropout=0.)
    image = ImageInputs(rand(6, 8, 24), rand(6, 8, 24), torch.ones(6, 8, dtype=torch.bool),
        rand(6, 24), rand(6, 24), rand(6, 5), rand(6, 6))
    genes = OmicsInputs(rand(6, 5, 3), torch.ones(6, 5, 3, dtype=torch.bool))
    cp, tp, prior = rand(3, 24), rand(3, 24), torch.eye(3, 5)
    spec = RepairSpec(source_arm=source, query_mode="visual_random", score_route="actual")
    model = CrossAttentionRepresentation(cp, tp, prior, config=c, spec=spec, seed=31)
    return model, image, genes, cp, tp, prior


@pytest.mark.parametrize("source", ["B2_MATCHED", "B2_NATIVE"])
def test_all_registered_parameters_are_actually_trained_by_outcome_free_losses(source):
    model, image, genes, *_ = fixture(source)
    out = model(image, genes)
    target = torch.randn(6, 5)
    loss = auxiliary_loss(out, target, torch.ones_like(target, dtype=torch.bool), ~torch.eye(6, dtype=torch.bool))
    loss["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.parameters())
    descriptor = model.architecture_descriptor()
    assert not descriptor["medical_text_query_used"] and not descriptor["biological_relation_prior_used"]
    assert not descriptor["official_paper_reproduction"]


def test_b2_ignores_actual_text_query_identity_and_biological_graph_values_but_keeps_shared_scores():
    first, image, genes, cp, tp, prior = fixture()
    second = CrossAttentionRepresentation(cp * 100 + 9, -tp * 10, prior.flip(0),
        config=first.config, spec=first.spec, seed=31)
    a, b = first(image, genes), second(image, genes)
    assert torch.equal(a["program_prediction"], b["program_prediction"])
    assert torch.equal(a["pair"]["scores"], b["pair"]["scores"])
    changed = replace(image, titan_scores=image.titan_scores * 9 + 7)
    assert not torch.equal(a["program_prediction"], first(changed, genes)["program_prediction"])


@pytest.mark.parametrize("source", ["B2_MATCHED", "B2_NATIVE"])
def test_vectorized_all_candidate_scores_equal_independent_pair_enumeration(source):
    model, image, genes, *_ = fixture(source)
    model.eval()
    outputs = model(image, genes)
    p, g, mask = outputs["visual_tokens"], outputs["gene_tokens"], outputs["gene_mask"]
    expected = torch.empty(6, 6)
    for i in range(6):
        for j in range(6):
            pp, gg = model._conditional(p[i:i+1], g[j:j+1], mask[j:j+1])
            u = F.normalize(model.u(pp), dim=-1).mean(1)
            v = F.normalize(model.v(gg), dim=-1).mean(1)
            expected[i, j] = (u * v).sum() / model.config.temperature
    torch.testing.assert_close(outputs["pair"]["scores"], expected, atol=1e-6, rtol=1e-5)


def test_activation_checkpointing_does_not_change_scores_or_gradients():
    first, image, genes, cp, tp, prior = fixture("B2_NATIVE")
    second = CrossAttentionRepresentation(cp, tp, prior, config=first.config, spec=first.spec, seed=31)
    first.train()
    second.eval() # no dropout in fixture; this disables activation checkpoint
    a, b = first(image, genes), second(image, genes)
    torch.testing.assert_close(a["pair"]["scores"], b["pair"]["scores"])
    a["pair"]["scores"].square().mean().backward()
    b["pair"]["scores"].square().mean().backward()
    for (ka, pa), (kb, pb) in zip(first.named_parameters(), second.named_parameters()):
        assert ka == kb
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, atol=1e-6, rtol=1e-5)


def test_matched_direct_autograd_equals_previous_checkpoint_output_and_gradients(monkeypatch):
    import src.semantic_b0b7.p2_models as operators
    first, image, genes, cp, tp, prior = fixture("B2_MATCHED")
    second = CrossAttentionRepresentation(cp, tp, prior, config=first.config, spec=first.spec, seed=31)
    first.train()
    second.train()
    calls = []
    original = operators.checkpoint
    def audited_checkpoint(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(operators, "checkpoint", audited_checkpoint)
    a = first(image, genes)
    assert calls == [], "MATCHED must use direct autograd, without checkpoint recomputation"
    # Reference replays the previous execution strategy using exactly the same
    # matched operator and precomputed banks; it changes no learned parameter.
    image_outputs = second.image_only(image)
    gene_outputs = second.encode_omics(genes)
    banks = second._factorized_banks(image_outputs["visual_tokens"], gene_outputs["gene_tokens"])
    reference = original(second._factorized_score_block, *banks, gene_outputs["gene_mask"],
                         use_reentrant=False, preserve_rng_state=False) / second.config.temperature
    torch.testing.assert_close(a["pair"]["scores"], reference, atol=0, rtol=0)
    a["pair"]["scores"].square().mean().backward()
    reference.square().mean().backward()
    for (ka, pa), (kb, pb) in zip(first.named_parameters(), second.named_parameters()):
        assert ka == kb and (pa.grad is None) == (pb.grad is None)
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, atol=1e-6, rtol=1e-5)
    descriptor = first.architecture_descriptor()
    assert descriptor["matched_uses_direct_autograd_without_checkpoint_recomputation"]
    assert not descriptor["activation_checkpoint_enabled_during_training"]


def test_independent_main_effects_not_modified_by_conditioned_attention():
    model, image, genes, *_ = fixture()
    a = model(image, genes)
    altered = OmicsInputs(genes.values.flip(0) * 5, genes.measured)
    b = model(image, altered)
    assert torch.equal(a["additive_image"], b["additive_image"])
    assert torch.equal(a["program_prediction"], b["program_prediction"])
    assert not torch.equal(a["interaction_features"], b["interaction_features"])


def test_capacity_match_is_real_but_compute_claim_is_explicitly_unqualified():
    model, _, _, cp, tp, prior = fixture()
    semantic = SemanticRepresentation("B4", cp, tp, prior, config=model.config, seed=31)
    # Dense B2 has no unused unknown-edge scalar; every other parameter lives
    # on a real program/pair path. Same-budget tolerance is much below1%.
    assert semantic.parameter_counts()["total"] - model.parameter_counts()["total"] == 1
    assert model.architecture_descriptor()["FLOP_match_within10percent_requires_real_profile_not_assumed"]
    assert model.attention_heads == model.architecture_descriptor()["attention_heads"] == 1
    assert model.attention_width == 8
    native, *_ = fixture("B2_NATIVE")
    assert native.attention_heads == native.config.heads
    assert native.attention_width == native.config.width


def test_no_measured_gene_evidence_produces_no_conditional_risk_message():
    model, image, genes, *_ = fixture()
    genes.measured[:] = False
    output = model(image, genes)
    mechanism = model.mechanism_from_tokens(output["visual_tokens"], output["gene_tokens"], output["gene_mask"])
    assert torch.equal(mechanism, torch.zeros_like(mechanism))
    assert not output["pair"]["valid"].any()
    assert torch.isfinite(output["pair"]["scores"]).all()


@pytest.mark.parametrize("source", ["B2_MATCHED", "B2_NATIVE"])
@pytest.mark.parametrize("masked", [False, True])
def test_unique_bank_linear_factorization_matches_float64_oracle_and_all_gradients(source, masked):
    model, *_ = fixture(source)
    model.double().eval()
    gen = torch.Generator().manual_seed(574)
    p = torch.randn(4, model.config.slots, model.config.width, generator=gen,
                    dtype=torch.float64, requires_grad=True)
    g = torch.randn(5, model.config.programs, model.config.width, generator=gen,
                    dtype=torch.float64, requires_grad=True)
    mask = torch.ones(5, model.config.programs, dtype=torch.bool)
    if masked:
        mask[0] = False
        mask[1, ::2] = False
        mask[2, 1:] = False
    reference = model._pair_score_block(p, g, mask)
    actual = model._factorized_score_block(*model._factorized_banks(p, g), mask)
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-11)
    probe = torch.randn(reference.shape, generator=gen, dtype=torch.float64)
    variables = [p, g, *model.parameters()]
    reference_gradients = torch.autograd.grad((reference * probe).sum(), variables,
                                              allow_unused=True, retain_graph=True)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), variables, allow_unused=True)
    for expected, observed in zip(reference_gradients, actual_gradients):
        assert (expected is None) == (observed is None)
        if expected is not None:
            torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("source", ["B2_MATCHED", "B2_NATIVE"])
def test_float32_factorization_reordering_error_is_bounded_for_locked_synthetic_probe(source):
    model, *_ = fixture(source)
    model.eval()
    gen = torch.Generator().manual_seed(574)
    p = torch.randn(11, model.config.slots, model.config.width, generator=gen)
    g = torch.randn(13, model.config.programs, model.config.width, generator=gen)
    mask = torch.rand(13, model.config.programs, generator=gen) > .25
    with torch.no_grad():
        expected = model._pair_score_block(p, g, mask)
        actual = model._factorized_score_block(*model._factorized_banks(p, g), mask)
    # Real-arithmetic equality is not a bitwise FP32 claim. This test records
    # only the fixed synthetic numerical qualification, not real-cohort error.
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)

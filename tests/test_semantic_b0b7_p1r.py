"""Synthetic P1-R operator/gate tests; no real-data optimizer updates."""
from argparse import Namespace
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.semantic_b0b7.models import ImageInputs, ModelConfig, OmicsInputs, SemanticRepresentation
from src.semantic_b0b7.pretext import auxiliary_loss
from src.semantic_b0b7.p1r_models import RepairRepresentation, RepairSpec, ScoreMomentSurrogate, frozen_readout_operator

torch.set_num_threads(2)


def fixture():
    gen = torch.Generator().manual_seed(928)
    rand = lambda *shape: torch.randn(*shape, generator=gen)
    c = ModelConfig(patch_dim=24, conch_text_dim=24, titan_native_dim=24,
        titan_text_dim=24, titan_score_dim=5, clinical_dim=6,
        programs=5, slots=3, width=12, heads=3, rank=2, dropout=0.)
    cp, tp = rand(3, 24), rand(3, 24)
    prior = torch.zeros(3, 5)
    prior[0, 0] = prior[1, 1] = prior[2, 2] = 1.
    image = ImageInputs(rand(12, 8, 24), rand(12, 8, 24), torch.ones(12, 8, dtype=torch.bool),
        rand(12, 24), rand(12, 24), rand(12, 5), rand(12, 6))
    genes = OmicsInputs(rand(12, 5, 3), torch.ones(12, 5, 3, dtype=torch.bool))
    return c, cp, tp, prior, image, genes


@pytest.mark.parametrize("arm,query", [("B3", "random"), ("B4", "true")])
def test_unchanged_legacy_spec_has_exact_original_state_and_forward(arm, query):
    c, cp, tp, prior, image, genes = fixture()
    old = SemanticRepresentation(arm, cp, tp, prior, config=c, seed=31)
    new = RepairRepresentation(cp, tp, prior, config=c, seed=31,
        spec=RepairSpec(source_arm=arm, query_mode=query))
    assert set(old.state_dict()) == set(new.state_dict())
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    a, b = old(image, genes), new(image, genes)
    for key in ("visual_tokens", "gene_tokens", "program_prediction", "interaction_features"):
        assert torch.equal(a[key], b[key])
    assert torch.equal(a["pair"]["scores"], b["pair"]["scores"])


def test_textless_does_not_read_text_prototypes_aligned_features_or_explicit_scores():
    c, cp, tp, prior, image, genes = fixture()
    model = RepairRepresentation(None, None, prior, config=c,
        spec=RepairSpec(query_mode="textless", score_route="native_random"))
    a = model(image, genes)
    poisoned = replace(image, patch_aligned=torch.full_like(image.patch_aligned, float("nan")),
                       titan_aligned=torch.full_like(image.titan_aligned, float("nan")),
                       titan_scores=torch.full_like(image.titan_scores, float("nan")))
    b = model(poisoned, genes)
    assert torch.equal(a["program_prediction"], b["program_prediction"])
    assert torch.equal(a["visual_tokens"], b["visual_tokens"])
    truth = SemanticRepresentation("B4", cp, tp, prior, config=c)
    assert model.parameter_counts() == truth.parameter_counts()


def test_textless_registered_parameters_all_have_real_auxiliary_gradients():
    c, _, _, prior, image, genes = fixture()
    model = RepairRepresentation(None, None, prior, config=c,
        spec=RepairSpec(query_mode="textless", score_route="native_random"))
    output = model(image, genes)
    target = torch.randn(12, 5)
    loss = auxiliary_loss(output, target, torch.ones_like(target, dtype=torch.bool), ~torch.eye(12, dtype=torch.bool))
    loss["loss"].backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_score_surrogate_fits_only_train_mean_covariance_and_ignores_validation_scores():
    c, cp, tp, prior, image, genes = fixture()
    surrogate = ScoreMomentSurrogate.fit(image.titan_native[:8], image.titan_scores[:8])
    pred = surrogate(image.titan_native[:8]).double()
    true = image.titan_scores[:8].double()
    torch.testing.assert_close(pred.mean(0), true.mean(0), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(torch.cov(pred.T), torch.cov(true.T), atol=1e-6, rtol=1e-5)
    model = RepairRepresentation(cp, tp, prior, config=c,
        spec=RepairSpec(score_route="train_moment_surrogate"), score_surrogate=surrogate)
    a = model(image, genes)
    changed = replace(image, titan_scores=image.titan_scores * 1000 + 1e6)
    b = model(changed, genes)
    assert torch.equal(a["program_prediction"], b["program_prediction"])
    assert model.parameter_counts() == SemanticRepresentation("B4", cp, tp, prior, config=c).parameter_counts()


@pytest.mark.parametrize("fraction", [.5, .75])
def test_dual_channel_means_match_independent_edge_enumeration(fraction):
    c, cp, tp, prior, image, genes = fixture()
    model = RepairRepresentation(cp, tp, prior, config=c,
        spec=RepairSpec(relation_mode="dual", known_fraction=fraction))
    out = model(image, genes)
    p = torch.nn.functional.normalize(model.u(out["visual_tokens"]), dim=-1)
    g = torch.nn.functional.normalize(model.v(out["gene_tokens"]), dim=-1)
    expected = torch.empty(12, 12)
    for i in range(12):
        for j in range(12):
            edge = p[i] @ g[j].T
            expected[i, j] = (fraction * edge[prior > 0].mean()
                               + (1 - fraction) * edge[prior == 0].mean()) / c.temperature
    torch.testing.assert_close(out["pair"]["scores"], expected)
    assert not isinstance(model.unknown_logit, torch.nn.Parameter)
    weights, info = model.channel_weights(out["gene_mask"])
    torch.testing.assert_close(weights.sum((1, 2)), torch.ones(12))
    torch.testing.assert_close(info["known_effective_fraction"], torch.full((12,), fraction))
    assert info["program_no_known_fallback"][:, 3:].all()


@pytest.mark.parametrize("all_known", [False, True])
def test_empty_relation_channel_falls_back_without_losing_valid_measurements(all_known):
    c, cp, tp, prior, image, genes = fixture()
    prior[:] = 1. if all_known else 0.
    model = RepairRepresentation(cp, tp, prior, config=c,
        spec=RepairSpec(relation_mode="dual", known_fraction=.75))
    out = model(image, genes)
    assert torch.isfinite(out["pair"]["scores"]).all() and out["pair"]["valid"].all()
    weights, diag = model.channel_weights(out["gene_mask"])
    torch.testing.assert_close(weights.sum((1, 2)), torch.ones(12))
    assert (diag["known_effective_fraction"] == (1. if all_known else 0.)).all()
    genes.measured[:] = False
    empty = model(image, genes)
    assert not empty["pair"]["valid"].any() and torch.isfinite(empty["pair"]["scores"]).all()


def test_dual_b3_b4_parameter_budget_equal_and_one_unused_legacy_scalar_removed():
    c, cp, tp, prior, image, genes = fixture()
    a = RepairRepresentation(cp, tp, prior, config=c,
        spec=RepairSpec(source_arm="B3", query_mode="random", relation_mode="dual"))
    b = RepairRepresentation(cp, tp, prior, config=c, spec=RepairSpec(relation_mode="dual"))
    assert a.parameter_counts() == b.parameter_counts()
    legacy = SemanticRepresentation("B4", cp, tp, prior, config=c)
    assert b.parameter_counts()["total"] == legacy.parameter_counts()["total"] - 1


def test_lambda_zero_leaves_pair_only_parameters_untrained_but_image_query_has_gradient():
    c, cp, tp, prior, image, genes = fixture()
    model = RepairRepresentation(cp, tp, prior, config=c, spec=RepairSpec(lambda_pair=0.))
    output = model(image, genes)
    truth = torch.randn(12, 5)
    objective = auxiliary_loss(output, truth, torch.ones_like(truth, dtype=torch.bool),
                               ~torch.eye(12, dtype=torch.bool), lambda_pair=0.)
    objective["loss"].backward()
    assert model.query_projection.weight.grad.abs().sum() > 0
    assert model.u.weight.grad is None or model.u.weight.grad.abs().sum() == 0
    assert model.gene_encoder.weight.grad is None or model.gene_encoder.weight.grad.abs().sum() == 0


def runner():
    spec = importlib.util.spec_from_file_location("p1r_operator_unit_runner", Path(__file__).resolve().parents[1] / "scripts/semantic_b0b7_p1r.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_rule_uses_huber_tolerance_then_hardgap_then_earlier_not_combined_loss():
    r = runner()
    def metrics(huber, gap, aux):
        return {"program": {"patient_equal_masked_huber": huber}, "auxiliary_loss": aux,
                "retrieval_hard": {d: {"paired_gap_mean": gap} for d in ("image_to_gene", "gene_to_image")}}
    rows = [{"epoch": 50, "validation": metrics(.1, .2, .9)},
            {"epoch": 100, "validation": metrics(.10005, .5, .8)},
            {"epoch": 200, "validation": metrics(.101, 1., .1)}]
    assert r.select_checkpoint(rows, .0001)["epoch"] == 100
    assert r.select_checkpoint(rows, .000001)["epoch"] == 50


def test_training_tolerance_depends_only_on_current_training_targets():
    r = runner()
    y = torch.tensor([[0., 0.], [1., 2.], [2., 4.], [3., 6.]])
    mask = torch.ones_like(y, dtype=torch.bool)
    tolerance, reference = r.training_huber_tolerance(y, mask)
    assert tolerance == reference * .01 and reference > 0


def test_invalid_master_lock_and_endpoint_payload_fail_before_optimizer():
    r = runner()
    with pytest.raises(ValueError, match="MASTER_LOCK_SCHEMA"):
        r.lock_binding({"status": "PASS"}, {}, "a" * 64)
    with pytest.raises(ValueError, match="OUTCOME_BEARING"):
        r.validate_scope({"DSS": np.arange(4)}, [0, 1], [2, 3], evaluate_valid=True)


def test_frozen_numpy_readout_preserves_main_effects_and_masks_unmeasured_gene_residual():
    c, cp, tp, prior, image, genes = fixture()
    model = RepairRepresentation(cp, tp, prior, config=c, spec=RepairSpec()).eval().requires_grad_(False)
    genes.measured[:, 3:] = False
    with torch.no_grad():
        out = model(image, genes)
    arrays = {k: v.cpu().numpy() for k, v in out.items() if torch.is_tensor(v)}
    mp = np.zeros_like(arrays["pre_residual_visual_tokens"])
    mg = np.ones_like(arrays["pre_residual_gene_tokens"])
    first = frozen_readout_operator(model, arrays, mp, mg, arm="B7")
    mg[:, 3:] = 100.
    second = frozen_readout_operator(model, arrays, mp, mg, arm="B7")
    np.testing.assert_array_equal(first["mechanism"], second["mechanism"])
    assert first["additive_image"] is arrays["additive_image"]
    assert first["additive_gene"] is arrays["additive_gene"]
    assert first["mechanism"].shape == (12, 2)

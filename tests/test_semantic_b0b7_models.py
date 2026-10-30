"""Synthetic mechanism contracts; no patient data, outcomes, or fit claims."""
from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch

from src.semantic_b0b7.models import (
    ARMS, ImageInputs, ModelConfig, OmicsInputs, SemanticRepresentation,
    masked_softmax, rotate_prototypes,
)
from src.semantic_b0b7.pretext import (
    PairMetadata, auxiliary_loss, matched_derangement, masked_program_loss,
    negative_support, paired_info_nce,
)

torch.set_num_threads(2)


def fixture(arm="B4", *, input_grad=False, dropout=0.):
    gen = torch.Generator().manual_seed(1257)
    c = ModelConfig(patch_dim=24, conch_text_dim=24, titan_native_dim=24,
                    titan_text_dim=24, titan_score_dim=7, clinical_dim=6,
                    programs=5, slots=3, width=12, heads=3, rank=2, dropout=dropout)
    rand = lambda *shape: torch.randn(*shape, generator=gen)
    conch = torch.nn.functional.normalize(rand(3, 24), dim=-1)
    titan = torch.nn.functional.normalize(rand(3, 24), dim=-1)
    relation = torch.zeros(3, 5)
    relation[0, 0] = relation[1, 2] = relation[2, 4] = 1.
    model = SemanticRepresentation(arm, conch, titan, relation, config=c, seed=31)
    patches = rand(6, 8, 24).requires_grad_(input_grad)
    mask = torch.ones(6, 8, dtype=torch.bool)
    mask[0, 6:] = False
    image = ImageInputs(patches, rand(6, 8, 24).requires_grad_(input_grad), mask,
                        rand(6, 24), rand(6, 24), rand(6, 7),
                        rand(6, 6).requires_grad_(input_grad),
                        torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]] * 6))
    genes = OmicsInputs(rand(6, 5, 3).requires_grad_(input_grad),
                        torch.ones(6, 5, 3, dtype=torch.bool))
    return model, image, genes, conch, titan, relation


def metadata():
    return PairMetadata(np.arange(6, dtype=np.int64),
                        np.array([0, 0, 0, 1, 1, 1]),
                        np.zeros(6, dtype=np.int64),
                        np.zeros(6, dtype=np.int64))


@pytest.mark.parametrize("arm", ARMS)
def test_actual_auxiliary_forward_backward_uses_every_trainable_parameter(arm):
    model, image, genes, *_ = fixture(arm, input_grad=True)
    outputs = model(image, genes)
    negatives, _ = negative_support(metadata(), mode="matched")
    target = torch.randn(6, 5)
    loss = auxiliary_loss(outputs, target, torch.ones_like(target, dtype=torch.bool),
                          torch.tensor(negatives), lambda_pair=.1)
    loss["loss"].backward()
    inactive = [name for name, p in model.named_parameters()
                if p.grad is None or not torch.isfinite(p.grad).all()
                or not (p.grad.abs().sum() > 0)]
    assert inactive == []
    for value in (image.patch_values, image.patch_aligned, image.clinical, genes.values):
        assert value.grad is not None and value.grad.abs().sum() > 0
    assert outputs["mechanism_status"] == "PRETEXT_UNRESIDUALIZED"
    assert outputs["clinical_residualization_applied"] is False


def test_b3_b4_equal_capacity_common_initialization_different_frozen_queries():
    a, image, genes, *_ = fixture("B3")
    b, *_ = fixture("B4")
    assert a.parameter_counts() == b.parameter_counts()
    assert {k for k, _ in a.named_parameters()} == {k for k, _ in b.named_parameters()}
    for (ka, va), (kb, vb) in zip(a.named_parameters(), b.named_parameters()):
        assert ka == kb and torch.equal(va, vb)
    assert not torch.allclose(a.conch_prototypes, b.conch_prototypes)
    torch.testing.assert_close(a.conch_prototypes @ a.conch_prototypes.T,
                               b.conch_prototypes @ b.conch_prototypes.T)
    oa, ob = a(image, genes), b(image, genes)
    assert not torch.allclose(oa["patch_attention"], ob["patch_attention"])
    assert not torch.allclose(oa["program_prediction"], ob["program_prediction"])


def test_same_gram_rotation_preserves_norm_spectrum_but_not_row_orthogonality():
    x = torch.randn(4, 11, dtype=torch.float64)
    y = rotate_prototypes(x, 47)
    torch.testing.assert_close(x @ x.T, y @ y.T)
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1))
    torch.testing.assert_close(torch.linalg.svdvals(x), torch.linalg.svdvals(y))
    assert not torch.allclose(y @ y.T, torch.eye(4, dtype=torch.float64))
    assert torch.equal(y, rotate_prototypes(x, 47))


def test_image_only_api_cannot_receive_rna_or_survival_outcome():
    model, image, genes, *_ = fixture()
    parameters = inspect.signature(model.image_only).parameters
    assert set(parameters) == {"x", "view"}
    a = model.image_only(image)["program_prediction"]
    genes.values.fill_(1000.)
    b = model.image_only(image)["program_prediction"]
    assert torch.equal(a, b)
    with pytest.raises(TypeError):
        model.image_only(image, genes=genes)
    with pytest.raises(TypeError):
        model.image_only(image, event=torch.ones(6))


@pytest.mark.parametrize("view,changed", [("clinical", "image"), ("pathology", "clinical")])
def test_p1_single_modality_views_do_not_use_the_excluded_modality(view, changed):
    model, image, *_ = fixture()
    a = model.image_only(image, view=view)["program_prediction"]
    if changed == "image":
        altered = replace(image, patch_values=image.patch_values * 10 + 50,
                          patch_aligned=image.patch_aligned.flip(-1),
                          titan_native=-image.titan_native,
                          titan_aligned=-image.titan_aligned,
                          titan_scores=-image.titan_scores)
    else:
        altered = replace(image, clinical=image.clinical * 10 + 100)
    b = model.image_only(altered, view=view)["program_prediction"]
    assert torch.equal(a, b)


def test_all_masked_rows_and_unmeasured_nan_values_are_not_observed_zero_values():
    model, image, genes, *_ = fixture()
    image.patch_mask[:] = False
    image.patch_values[:] = float("nan")
    image.patch_aligned[:] = float("nan")
    genes.measured[:] = False
    genes.values[:] = float("nan")
    outputs = model(image, genes)
    assert torch.equal(outputs["patch_attention"], torch.zeros_like(outputs["patch_attention"]))
    assert torch.equal(outputs["gene_tokens"], torch.zeros_like(outputs["gene_tokens"]))
    for key in ("program_prediction", "interaction_features", "gene_embedding"):
        assert torch.isfinite(outputs[key]).all()
    pair = paired_info_nce(outputs["pair"]["scores"], outputs["pair"]["valid"],
                           ~torch.eye(6, dtype=torch.bool))
    assert pair["loss"].item() == 0 and pair["image_anchors_used"] == 0
    genes.measured[:] = True
    with pytest.raises(ValueError, match="NONFINITE_MEASURED"):
        model.encode_omics(genes)


def test_multislide_reading_is_equal_slide_average_and_padding_invariant():
    model, image, *_ = fixture()
    one = replace(image, patch_mask=torch.tensor([[True] * 4 + [False] * 4] * 6))
    two = replace(image, patch_mask=torch.tensor([[False] * 4 + [True] * 4] * 6))
    all_valid = replace(image, patch_mask=torch.ones_like(image.patch_mask))
    a = model.image_only(one)["visual_tokens"]
    b = model.image_only(two)["visual_tokens"]
    whole = model.image_only(all_valid)["visual_tokens"]
    torch.testing.assert_close(whole, (a + b) / 2)
    changed = replace(one, patch_values=one.patch_values.clone(),
                      patch_aligned=one.patch_aligned.clone())
    changed.patch_values[:, 4:] = 1e6
    changed.patch_aligned[:, 4:] = 1e6
    assert torch.equal(a, model.image_only(changed)["visual_tokens"])


@pytest.mark.parametrize("arm", ["B5", "B6a", "B7"])
def test_residual_background_is_fail_closed_and_requires_frozen_own_latent_space(arm):
    model, image, genes, *_ = fixture(arm)
    with pytest.raises(ValueError, match="REQUIRES_TRAIN_ONLY"):
        model(image, genes, stage="frozen_readout")
    ov = model(image, genes)
    mu_p = torch.randn_like(ov["visual_tokens"])
    mu_g = torch.randn_like(ov["gene_tokens"])
    with pytest.raises(ValueError, match="FROZEN_EVAL"):
        model(image, genes, stage="frozen_readout", mu_visual=mu_p, mu_gene=mu_g)
    model.eval().requires_grad_(False)
    result = model(image, genes, stage="frozen_readout", mu_visual=mu_p, mu_gene=mu_g)
    assert result["clinical_residualization_applied"] is True
    assert result["mechanism_status"] == "FROZEN_FEATURE_OPERATOR"
    assert torch.equal(result["additive_image"], ov["additive_image"])
    assert torch.equal(result["additive_gene"], ov["additive_gene"])
    with pytest.raises(ValueError, match="COORDINATE_SHAPE"):
        model(image, genes, stage="frozen_readout", mu_visual=mu_p[..., :1], mu_gene=mu_g)


def test_b5_b7_capacity_identical_b7_has_zero_cross_modal_mixed_difference():
    b5, p1, g1, *_ = fixture("B5")
    b7, *_ = fixture("B7")
    assert b5.parameter_counts() == b7.parameter_counts()
    p2 = replace(p1, patch_values=-p1.patch_values,
                 titan_native=-p1.titan_native, titan_scores=-p1.titan_scores)
    g2 = OmicsInputs(g1.values.flip(0) * 2, g1.measured.clone())
    g2.measured[0, 0] = False
    def mixed(model):
        return (model(p1, g1)["interaction_features"] + model(p2, g2)["interaction_features"]
                - model(p1, g2)["interaction_features"] - model(p2, g1)["interaction_features"])
    torch.testing.assert_close(mixed(b7), torch.zeros(6, 2), atol=1e-6, rtol=0.)
    assert mixed(b5).abs().max() > 1e-5


@pytest.mark.parametrize("arm", ["B2_MATCHED_1H", "B2_MATCHED_4H"])
def test_b2_shared_projection_controls_have_exact_active_parameter_budget(arm):
    matched, image, genes, *_ = fixture(arm)
    concept, *_ = fixture("B4")
    native, *_ = fixture("B2")
    assert matched.parameter_counts() == concept.parameter_counts()
    assert native.parameter_counts()["total"] > matched.parameter_counts()["total"]
    assert len(list(matched.g_to_p.parameters())) == 0
    output = matched(image, genes)
    assert output["interaction_features"].shape == (6, 2)


def test_clinical_capacity_control_uses_background_additively_without_subtracting_evidence():
    model, image, genes, *_ = fixture("B5_CLINICAL_CAPACITY_CONTROL")
    base, *_ = fixture("B4")
    residual, *_ = fixture("B5")
    assert model.parameter_counts() == residual.parameter_counts() == base.parameter_counts()
    model.eval().requires_grad_(False)
    base.eval().requires_grad_(False)
    out = base(image, genes)
    mu_p, mu_g = torch.randn_like(out["visual_tokens"]), torch.randn_like(out["gene_tokens"])
    control = model(image, genes, stage="frozen_readout", mu_visual=mu_p, mu_gene=mu_g)
    assert control["clinical_additive_capacity_control_applied"] is True
    assert control["clinical_residualization_applied"] is False
    assert torch.equal(control["pre_residual_visual_tokens"], out["visual_tokens"])
    assert not torch.allclose(control["interaction_features"], out["interaction_features"])
    # The extra feature is exclusively a function of fixed clinical means:
    # changing observed P/G does not change this control increment.
    altered = replace(image, patch_values=-image.patch_values, titan_native=-image.titan_native)
    wrong_genes = OmicsInputs(-genes.values, genes.measured)
    altered_base = base(altered, wrong_genes)
    altered_control = model(altered, wrong_genes, stage="frozen_readout", mu_visual=mu_p, mu_gene=mu_g)
    torch.testing.assert_close(control["interaction_features"] - out["interaction_features"],
                               altered_control["interaction_features"] - altered_base["interaction_features"])


def test_b6a_changes_only_loss_positive_assignment_not_any_model_output_or_target():
    b5, image, genes, *_ = fixture("B5")
    b6, *_ = fixture("B6a")
    out5, out6 = b5(image, genes), b6(image, genes)
    for key in ("program_prediction", "visual_tokens", "gene_tokens", "interaction_features"):
        assert torch.equal(out5[key], out6[key])
    perm, _ = matched_derangement(metadata(), seed=47)
    target, target_mask = torch.randn(6, 5), torch.ones(6, 5, dtype=torch.bool)
    target_copy = target.clone()
    negatives, _ = negative_support(metadata())
    five = auxiliary_loss(out5, target, target_mask, torch.tensor(negatives))
    six = auxiliary_loss(out6, target, target_mask, torch.tensor(negatives), positive=torch.tensor(perm))
    assert torch.equal(five["program"]["loss"], six["program"]["loss"])
    assert not torch.equal(five["pair"]["loss"], six["pair"]["loss"])
    assert torch.equal(target, target_copy)
    assert six["pair"]["positive_is_identity"] is False


def test_matched_derangement_no_fixed_points_and_explicit_fallbacks():
    m = metadata()
    p, audit = matched_derangement(m, seed=31)
    assert not (m.patient[p] == m.patient).any()
    assert np.array_equal(np.sort(p), np.arange(6))
    assert audit["exact_all_three"] == 6 and audit["fallback_anchors"] == 0
    singleton = PairMetadata(np.arange(4), np.arange(4), np.arange(4), np.arange(4))
    p, audit = matched_derangement(singleton, seed=31)
    assert audit["fallback_anchors"] == 4 and not (p == np.arange(4)).any()
    duplicate = PairMetadata(np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64),
                             np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64))
    with pytest.raises(ValueError, match="NO_VALID_WRONG"):
        matched_derangement(duplicate, seed=31)


def test_matched_negatives_without_support_are_excluded_not_replaced_by_easy_negatives():
    m = PairMetadata(np.arange(4), np.arange(4), np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64))
    support, audit = negative_support(m)
    assert not support.any() and audit["anchors_with_negative"] == 0
    random, _ = negative_support(m, mode="random")
    assert random.sum() == 12
    logits = torch.randn(4, 4, requires_grad=True)
    out = paired_info_nce(logits, torch.ones(4, 4, dtype=torch.bool), torch.tensor(support))
    assert out["loss"].item() == 0 and out["image_anchors_used"] == 0
    out["loss"].backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_bijective_pair_loss_matches_independent_manual_row_and_column_ce():
    scores = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(5, 5, dtype=torch.bool)
    negatives = ~torch.eye(5, dtype=torch.bool)
    result = paired_info_nce(scores, valid, negatives)
    independent = (torch.nn.functional.cross_entropy(scores, torch.arange(5))
                   + torch.nn.functional.cross_entropy(scores.T, torch.arange(5))) / 2
    torch.testing.assert_close(result["loss"], independent)
    with pytest.raises(ValueError, match="BIJECTION"):
        paired_info_nce(scores, valid, negatives, positive=torch.zeros(5, dtype=torch.long))


def test_program_huber_equal_patient_weight_and_unmeasured_values_ignored():
    prediction = torch.tensor([[2., 2.], [3., 0.]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True], [True, False]])
    result = masked_program_loss(prediction, target, mask)
    assert result["loss"].item() == 2.0 # (1.5 + 2.5)/2, not 5.5/3
    target[1, 1] = float("nan")
    assert torch.equal(masked_program_loss(prediction, target, mask)["loss"], result["loss"])


def test_masked_softmax_nan_free_and_has_zero_all_masked_gradient():
    x = torch.randn(2, 4, requires_grad=True)
    mask = torch.tensor([[True, False, True, False], [False] * 4])
    output = masked_softmax(x, mask)
    assert output[0].sum() == 1 and output[1].sum() == 0
    output.square().sum().backward()
    assert torch.equal(x.grad[1], torch.zeros(4))


def test_invalid_real_measurement_and_prototype_shapes_fail():
    model, image, genes, cp, tp, prior = fixture()
    with pytest.raises(ValueError, match="SHAPE_MISMATCH"):
        SemanticRepresentation("B4", cp[:2], tp, prior, config=model.config)
    image.patch_values[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NONFINITE_VALID_PATCH"):
        model.image_only(image)

"""Synthetic R6 risk/route contracts: no target patients, endpoints or fits."""
import copy
from dataclasses import replace

import pytest
import torch

from src.bgb_en.r6_omics import GeneValueBank, SparseProgramEncoder, mapping_from_programs, PROGRAM_ORDER
from src.bgb_en.r6_pathology import NativePatchGeometry, source_model_state_sha256
from src.bgb_en.r6_source_model import SourceProgramModel
from src.bgb_en.r6_survival import R6SurvivalModel, TARGET_ARMS, patient_patch_mean


def fixture(arm="B7", *, seed=31):
    torch.manual_seed(6123)
    programs = [dict(id=PROGRAM_ORDER[0], up_genes=["A", "B"], down_genes=["C"]),
                dict(id=PROGRAM_ORDER[1], up_genes=["B", "D"], down_genes=["E"]),
                dict(id=PROGRAM_ORDER[2], up_genes=["F", "G"], down_genes=[]),
                dict(id=PROGRAM_ORDER[3], up_genes=["G", "H"], down_genes=[])]
    mapping = mapping_from_programs(programs)
    bank = GeneValueBank(["outside", "A", "B", "C", "D", "E", "F", "G", "H"], mapping.gene_symbols)
    encoder = SparseProgramEncoder(mapping)
    foundation = SourceProgramModel("S7", gene_reader=encoder, tissue_prior=[.3, .4, .3], init_seed=17)
    kind = ("RANDOM_NO_ST" if arm in ("B2", "B13") else
            "WRONG_DONOR_ST" if arm == "B8" else "MATCHED_ST")
    model = R6SurvivalModel(arm, foundation, source_kind=kind, init_seed=seed)
    raw = torch.randn(3, 9) * 2
    measured = torch.ones_like(raw, dtype=torch.bool)
    measured[1, 3] = False
    measured[2, 7:] = False
    raw[~measured] = float("nan")
    genes = bank(raw, measured)
    states = torch.randn(3, 2, 5, 64)
    mask = torch.tensor([[[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]],
                         [[1, 1, 1, 0, 0], [0, 0, 0, 0, 0]],
                         [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]], dtype=torch.bool)
    states[~mask] = float("nan")
    coords = torch.zeros(3, 2, 5, 2)
    coords[..., 0] = torch.arange(5) * 512
    neighbor = NativePatchGeometry().build(coords, mask)
    prior = torch.randn(3, 2, 5, 4)
    prior[~mask] = float("nan")
    patch = dict(states=states, patch_mask=mask, program_score_prior=prior,
                 diagnostics=dict(source_state_sha256=source_model_state_sha256(model.foundation)))
    offset = torch.tensor([.4, -.6, 1.2])
    hallmark = torch.randn(3, 50, 32)
    hm = torch.ones(3, 50, dtype=torch.bool)
    kwargs = dict(neighborhood=neighbor, hallmark_tokens=hallmark, hallmark_mask=hm)
    if arm == "B9":
        with torch.no_grad():
            context = model.foundation.encode_gene_programs(genes)
        kwargs.update(query_context=context["tokens"].roll(1, 0), query_mask=context["program_mask"].roll(1, 0))
    return model, patch, genes, offset, kwargs, bank, raw, measured


def forward(item, **extra):
    model, patch, genes, offset, kwargs, *_ = item
    return model(patch, genes, offset, **(kwargs | extra))


@pytest.mark.parametrize("arm", TARGET_ARMS)
def test_arms_forward_finite_exact_risk_decomposition_and_source_freeze(arm):
    item = fixture(arm)
    model = item[0]
    output = forward(item)
    assert torch.isfinite(output["risk"]).all()
    assert torch.equal(output["risk"], item[3] + output["delta"])
    assert output["offset"] is item[3]
    assert torch.equal(output["delta"], output["main_delta"] + output["cross_delta"])
    assert torch.equal(output["main_delta"], output["path_main_delta"] +
                       output["gene_main_delta"] + output["hallmark_main_delta"])
    assert output["gene_values"] is item[2].values
    assert output["gene_measured_mask"] is item[2].measured_mask
    assert not output["diagnostics"]["Hallmark_tokens_claimed_ST_aligned"]
    assert not any(p.requires_grad for p in model.foundation.parameters())
    assert model.gene_rereader.gene_encoder is model.foundation.gene_reader.gene_encoder
    model.train()
    assert not model.foundation.training
    if arm in ("B2", "B3"):
        assert torch.count_nonzero(output["cross_delta"]) == 0
    if arm in ("B2", "B3", "B4", "B11"):
        assert torch.count_nonzero(output["delta_G"]) == 0


def test_all_arm_target_initializations_share_bytes_and_reverse_query_budget():
    models = [fixture(arm)[0] for arm in TARGET_ARMS]
    state = models[0].state_dict()
    for model in models[1:]:
        assert state.keys() == model.state_dict().keys()
        for name, value in model.state_dict().items():
            assert torch.equal(value, state[name]), name
    assert models[TARGET_ARMS.index("B5")].parameter_budget()["requires_grad"] == \
        models[TARGET_ARMS.index("B6")].parameter_budget()["requires_grad"]
    assert models[TARGET_ARMS.index("B5")].parameter_budget()["reverse_query_parameters"] == 4096
    assert models[TARGET_ARMS.index("B6")].parameter_budget()["reverse_query_parameters"] == 4096


def test_query_only_wrong_pair_keeps_own_values_main_effects_and_offset_exact():
    item = fixture()
    correct = forward(item)
    context = correct["G0"].roll(1, 0).clone()
    source_mask = correct["own_program_mask"].roll(1, 0).clone()
    source_mask[0, 1] = False
    wrong = forward(item, intervention="wrong_query", query_context=context, query_mask=source_mask)
    for key in ("offset", "main_delta", "patch_values", "gene_values", "gene_measured_mask", "G0"):
        assert torch.equal(correct[key], wrong[key]), key
    assert torch.equal(wrong["query_source_mask"], source_mask)
    assert not wrong["query_source_mask"][0, 1]
    assert torch.count_nonzero(wrong["Econd"][0, 1]) == 0
    assert not torch.equal(correct["conditioned_queries"], wrong["conditioned_queries"])
    assert not wrong["diagnostics"]["RNA_main_effect_replaced"]


@pytest.mark.parametrize("intervention", ["off", "value_off"])
def test_off_or_value_off_has_exact_zero_cross_and_keeps_main(intervention):
    item = fixture()
    normal = forward(item)
    output = forward(item, intervention=intervention)
    for key in ("cross_delta", "delta_E", "delta_G"):
        assert torch.count_nonzero(output[key]) == 0
    assert torch.equal(output["main_delta"], normal["main_delta"])
    assert torch.equal(output["risk"], output["offset"] + output["main_delta"])
    if intervention == "value_off":
        assert torch.equal(output["attention_change"], normal["attention_change"])
    else:
        assert torch.count_nonzero(output["attention_change"]) == 0


def test_mean_is_train_measurement_derived_with_its_source_mask_not_patient_mask():
    item = fixture()
    model, _, genes, *_ = item
    # First two synthetic rows are designated training rows; third has no role
    # in the means. No endpoint labels are created or consulted here.
    train = replace(genes, values=genes.values[:2], measured_mask=genes.measured_mask[:2])
    mean = model.mean_query_context(train)
    own = model.foundation.encode_gene_programs(train)
    expected = (own["tokens"] * own["program_mask"][..., None]).sum(0) / \
        own["program_mask"].sum(0).clamp_min(1)[:, None]
    assert torch.equal(mean.tokens, expected)
    assert not mean.tokens.requires_grad and mean.training_patients == 2
    output = forward(item, intervention="mean_query", mean_context=mean)
    assert torch.equal(output["query_source_mask"], mean.program_mask[None].expand(3, -1))
    assert torch.equal(output["main_delta"], forward(item)["main_delta"])


def test_padding_and_all_missing_are_zero_safe():
    item = fixture()
    model, patch, genes, offset, kwargs, *_ = item
    empty_genes = replace(genes, values=torch.full_like(genes.values, float("nan")),
                          measured_mask=torch.zeros_like(genes.measured_mask))
    output = model(patch, empty_genes, offset, **kwargs)
    assert torch.count_nonzero(output["G0"]) == 0
    assert torch.count_nonzero(output["cross_delta"]) == 0
    assert torch.isfinite(output["risk"]).all()
    empty_patch = dict(patch, states=torch.full_like(patch["states"], float("nan")),
                       patch_mask=torch.zeros_like(patch["patch_mask"]))
    neighbor = torch.full((*patch["patch_mask"].shape, 1), -1, dtype=torch.long)
    output = model(empty_patch, genes, offset, **(kwargs | dict(neighborhood=neighbor)))
    assert torch.count_nonzero(output["patch_values"]) == 0
    assert torch.count_nonzero(output["cross_delta"]) == 0
    assert torch.isfinite(output["risk"]).all()
    for key in ("patch_base_read", "patch_cond_read"):
        assert torch.equal(output[key]["null"], torch.ones_like(output[key]["null"]))


def test_raw_reverse_reencodes_changed_own_gene_with_fixed_path_queries():
    item = fixture()
    model, patch, genes, offset, kwargs, bank, raw, measured = item
    normal = forward(item)
    context, source_mask = normal["G0"].detach().clone(), normal["own_program_mask"].clone()
    before = model(patch, genes, offset, **kwargs, intervention="wrong_query",
                   query_context=context, query_mask=source_mask)
    changed = raw.clone()
    changed[0, 1] += 5.0
    after = model(patch, bank(changed, measured), offset, **kwargs, intervention="wrong_query",
                  query_context=context, query_mask=source_mask)
    assert torch.equal(before["Econd"], after["Econd"])
    assert torch.equal(before["Ebase"], after["Ebase"])
    assert not torch.equal(before["reverse_cond_read"]["tokens"], after["reverse_cond_read"]["tokens"])
    assert after["reverse_cond_read"]["original_gene_values_reencoded_this_call"]
    # Gene A belongs only to iCMS2. Other programs' original gene reads do not change.
    assert torch.equal(before["reverse_cond_read"]["tokens"][:, 1:],
                       after["reverse_cond_read"]["tokens"][:, 1:])


def test_gradients_reach_query_and_native_adapter_but_never_source_weights():
    item = fixture()
    model = item[0]
    output = forward(item)
    # Mathematical pullback, no optimizer and no biological endpoint labels.
    # Only final risk is differentiated: no auxiliary token objective can
    # conceal a disconnected prediction path.
    loss = output["risk"].square().sum()
    loss.backward()
    for parameter in (model.base_queries, model.query_condition[0].weight,
                      model.query_condition[2].weight, model.gene_rereader.query_projection.weight,
                      model.spatial_adapter.down.weight, model.spatial_adapter.up.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert all(parameter.grad is None for parameter in model.foundation.parameters())


def test_linear_readout_and_regularizers_match_locked_formula():
    item = fixture()
    model = item[0]
    output = forward(item)
    p = .025 * model.path_main(output["patch_mean"]).squeeze(-1)
    g = .025 * model.gene_main(output["gene_mean"]).squeeze(-1)
    h = .025 * model.hallmark_main(output["hallmark_mean"]).squeeze(-1)
    assert torch.equal(output["main_delta"], p + g + h)
    assert torch.equal(model.residual_penalty(output),
                       .1 * output["delta"].square().mean() + .1 * output["cross_delta"].square().mean())
    assert torch.equal(model.parameter_regularization(),
                       1e-4 * sum(p.square().sum() for p in model.parameters() if p.requires_grad))
    assert all(head.bias is None for head in (model.path_main, model.gene_main, model.hallmark_main,
                                              model.path_cross, model.gene_cross))


def test_prior_is_added_to_both_reads_without_changing_main_or_trainable_state():
    item = fixture("B6")
    model6, patch, genes, offset, kwargs, *_ = item
    model7 = R6SurvivalModel("B7", model6.foundation, source_kind="MATCHED_ST", init_seed=31)
    output6 = forward(item)
    output7 = model7(patch, genes, offset, **kwargs)
    assert torch.equal(output6["main_delta"], output7["main_delta"])
    expected = .1 * patch["program_score_prior"].permute(0, 1, 3, 2)
    for key in ("patch_base_read", "patch_cond_read"):
        valid = patch["patch_mask"][:, :, None, :] & output7["query_source_mask"][:, None, :, None]
        assert torch.allclose((output7[key]["logits"] - output6[key]["logits"])[valid], expected[valid],
                              atol=3e-7, rtol=0)


def test_patient_patch_mean_weights_wsi_equally_not_by_patch_count():
    x = torch.zeros(1, 2, 3, 64)
    x[:, 0] = 2.
    x[:, 1, 0] = 8.
    mask = torch.tensor([[[1, 1, 1], [1, 0, 0]]], dtype=torch.bool)
    assert torch.equal(patient_patch_mean(x, mask), torch.full((1, 64), 5.))


def test_source_mismatch_or_ambiguous_control_fails_closed():
    item = fixture()
    model, patch, genes, offset, kwargs, *_ = item
    bad = dict(patch, diagnostics=dict(source_state_sha256="0" * 64))
    with pytest.raises(ValueError, match="CACHE_AND_COMPLETE_SOURCE_STATE_DIFFER"):
        model(bad, genes, offset, **kwargs)
    with pytest.raises(ValueError, match="NEEDS_EXPLICIT_SOURCE_CONTEXT"):
        forward(item, intervention="wrong_query")
    with pytest.raises(ValueError, match="WHOLE_OMICS_REQUIRES_OUTER_WRAPPER"):
        forward(item, intervention="whole_omics")
    with pytest.raises(ValueError, match="SOURCE_CONTROL_DOES_NOT_MATCH"):
        R6SurvivalModel("B13", model.foundation, source_kind="MATCHED_ST")

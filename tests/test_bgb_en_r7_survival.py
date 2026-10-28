"""Synthetic R7 tensor contracts only. No real cohorts, endpoints or fitting."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)

from test_bgb_en_r6_survival import fixture as r6_fixture
from src.bgb_en.r6_pathology import NativePatchGeometry, source_model_state_sha256
from src.bgb_en.r6_training import SCALARS
from src.bgb_en.r7_survival import R7SurvivalModel, R7_ARMS, SCALE_NAMES


def fixture(arm="NORM", *, scales=None, seed=31):
    old, patch, genes, offset, kwargs, *_ = r6_fixture("B7", seed=seed)
    model = R7SurvivalModel(arm, old.foundation, init_seed=seed)
    if scales:
        model.set_calibration_scales(**scales)
    own = model.foundation.encode_gene_programs(genes)
    if arm.endswith("_WRONG_RNA"):
        kwargs |= dict(query_context=own["tokens"].roll(1, 0), query_mask=own["program_mask"].roll(1, 0))
    if arm.endswith("_WRONG_PATH"):
        base = model.unconditional_patch_base(patch, genes, neighborhood=kwargs["neighborhood"])
        kwargs |= dict(path_query_context=base["states"].roll(1, 0), path_query_mask=base["program_mask"].roll(1, 0))
    return model, patch, genes, offset, kwargs, old


def forward(item, **extra):
    model, patch, genes, offset, kwargs, _ = item
    return model(patch, genes, offset, **(kwargs | extra))


@pytest.mark.parametrize("arm", R7_ARMS)
def test_finite_own_values_raw_main_and_complete_risk_decomposition(arm):
    item = fixture(arm)
    out = forward(item)
    assert all(torch.isfinite(out[key]).all() for key in SCALARS)
    assert torch.equal(out["risk"], item[3] + out["delta"])
    assert torch.equal(out["delta"], out["main_delta"] + out["cross_delta"])
    old = item[-1](item[1], item[2], item[3], **{k: v for k, v in item[4].items() if k not in
        ("query_context", "query_mask", "path_query_context", "path_query_mask")})
    for key in ("main_delta", "path_main_delta", "gene_main_delta", "hallmark_main_delta", "G0", "patch_values"):
        assert torch.equal(out[key], old[key]), key
    assert out["gene_values"] is item[2].values
    assert out["gene_measured_mask"] is item[2].measured_mask
    assert not any(p.requires_grad for p in item[0].foundation.parameters())
    if arm == "ADD":
        assert torch.count_nonzero(out["cross_delta"]) == 0
    if out["slot_path_contributions"] is not None:
        assert torch.equal(out["slot_path_contributions"].sum(-1), out["path_cross_delta"])
        assert torch.equal(out["slot_gene_contributions"].sum(-1), out["gene_cross_delta"])


def test_raw_is_numerically_identical_to_unchanged_r6_b7_and_ignores_rms():
    item = fixture("RAW", scales={name: .123 for name in SCALE_NAMES})
    actual = forward(item)
    expected = item[-1](item[1], item[2], item[3], **item[4])
    for key in (*SCALARS, "delta_E", "delta_G", "G0", "Ebase", "Econd"):
        assert torch.equal(actual[key], expected[key]), key
    for name, value in item[-1].state_dict().items():
        assert torch.equal(item[0].state_dict()[name], value), name


def test_only_positive_scalar_buffers_and_foundation_bytes_unchanged():
    item = fixture("NORM")
    model = item[0]
    source_sha = source_model_state_sha256(model.foundation)
    before = forward(item)
    model.set_calibration_scales(context_rms=.1, patch_rms=.2, query_rms=.05,
                                 evidence_rms=.3, gene_key_rms=.2, reverse_query_rms=.1)
    after = forward(item)
    for key in ("main_delta", "G0", "patch_values", "offset", "gene_values"):
        assert torch.equal(before[key], after[key]), key
    assert source_model_state_sha256(model.foundation) == source_sha
    assert all(dict(model.named_buffers())[name].ndim == 0 for name in SCALE_NAMES)
    for bad in (0., -1., float("nan"), 1e-7):
        with pytest.raises(ValueError, match="RMS"):
            model.set_calibration_scales(context_rms=bad)


def test_three_stage_calibration_counts_union_genes_and_raw_uses_same_path():
    a, b = fixture("RAW"), fixture("NORM")
    scales = dict(context_rms=.2, patch_rms=.4, query_rms=.1, gene_key_rms=.3, evidence_rms=.5, reverse_query_rms=.2)
    for item in (a, b):
        item[0].set_calibration_scales(**scales)
    for stage in ("raw", "base", "reverse"):
        x = a[0].calibration_batch(a[1], a[2], neighborhood=a[4]["neighborhood"], stage=stage)
        y = b[0].calibration_batch(b[1], b[2], neighborhood=b[4]["neighborhood"], stage=stage)
        assert x["statistics"].keys() == y["statistics"].keys()
        for name in x["statistics"]:
            assert x["statistics"][name]["sum_squares"].dtype == torch.float64
            assert torch.equal(x["statistics"][name]["sum_squares"], y["statistics"][name]["sum_squares"])
            assert torch.equal(x["statistics"][name]["count"], y["statistics"][name]["count"])
        if stage == "raw":
            count = a[2].measured_mask.any(-1).sum() * 64
            assert x["statistics"]["raw_gene_key"]["count"] == count


def test_independent_reverse_can_change_when_forward_delta_is_exact_zero():
    serial, independent = fixture("NORM"), fixture("INDEP")
    for item in (serial, independent):
        with torch.no_grad():
            for p in item[0].query_condition.parameters():
                p.zero_()
            item[0].gene_rereader.query_projection.weight.copy_(torch.eye(64))
        item[0].set_calibration_scales(evidence_rms=.1, reverse_query_rms=.1, gene_key_rms=.1)
    s, i = forward(serial), forward(independent)
    assert torch.count_nonzero(s["delta_E"]) == torch.count_nonzero(i["delta_E"]) == 0
    assert torch.count_nonzero(s["delta_G"]) == 0
    assert torch.count_nonzero(i["delta_G"]) > 0
    assert torch.count_nonzero(i["reverse_base_read"]["query"]) == 0


def test_slot_initial_tie_and_quarter_l2_match_shared_readout():
    shared, slot = fixture("INDEP"), fixture("SLOT")
    a, b = forward(shared), forward(slot)
    assert torch.allclose(a["risk"], b["risk"], atol=1e-7, rtol=0)
    assert torch.allclose(shared[0].parameter_l2_sum(), slot[0].parameter_l2_sum(), atol=2e-6, rtol=1e-6)
    assert torch.equal(slot[0].slot_path_weight, shared[0].path_cross.weight.repeat(4, 1))
    assert not slot[0].path_cross.weight.requires_grad
    assert slot[0].slot_path_weight.requires_grad


def test_gate_one_starts_as_slot_with_unit_gates_and_sync_reverse_p0():
    slot, gate = fixture("SLOT"), fixture("GATE1")
    s, g = forward(slot), forward(gate)
    assert torch.equal(s["risk"], g["risk"])
    assert torch.equal(g["gates"]["path"], torch.ones_like(g["gates"]["path"]))
    assert torch.equal(g["gates"]["gene"], torch.ones_like(g["gates"]["gene"]))
    assert torch.equal(g["state_trace"][0]["PATH_route"], g["Ebase"])
    assert gate[0].parameter_budget()["gate_parameters"] == 258


def test_two_round_foreign_rna_is_same_foreign_initial_plus_own_feedback():
    item = fixture("GATE2_WRONG_RNA", scales={name: .3 for name in SCALE_NAMES})
    out = forward(item)
    first, second = out["state_trace"]
    initial = item[4]["query_context"] / item[0].context_rms
    mask = item[4]["query_mask"]
    expected = torch.where(mask[..., None], initial + (first["G_after"] - out["G0_interaction"]), torch.zeros_like(initial))
    assert torch.equal(second["RNA_route"], expected)
    assert torch.equal(second["PATH_route"], first["P_after"])
    assert out["diagnostics"]["no_foreign_updated_states"]
    assert out["diagnostics"]["RNA_route_is_foreign"]


def test_foreign_path_only_changes_reverse_route_and_all_main_values_stay_own():
    item = fixture("GATE1_WRONG_PATH")
    wrong = forward(item)
    matched = forward(item, intervention="matched", path_query_context=None, path_query_mask=None)
    for key in ("main_delta", "G0", "gene_values", "gene_measured_mask", "patch_values", "offset", "Ebase", "Econd"):
        assert torch.equal(wrong[key], matched[key]), key
    assert torch.equal(wrong["state_trace"][0]["PATH_route"], torch.where(item[4]["path_query_mask"][..., None],
        item[4]["path_query_context"], torch.zeros_like(item[4]["path_query_context"])))
    assert wrong["diagnostics"]["PATH_route_is_foreign"]
    assert not wrong["diagnostics"]["RNA_route_is_foreign"]


def test_off_retains_foreign_context_and_zeroes_only_interaction():
    item = fixture("GATE2_WRONG_RNA")
    ordinary, off = forward(item), forward(item, intervention="off")
    assert torch.equal(ordinary["main_delta"], off["main_delta"])
    assert torch.count_nonzero(off["cross_delta"]) == 0
    assert torch.equal(off["state_trace"][0]["RNA_route_mask"], item[4]["query_mask"])
    assert torch.equal(off["Gfinal"], off["G0_interaction"])
    assert torch.equal(off["Econd"], off["Ebase"])


def test_direct_matched_label_rejects_foreign_or_mean_context():
    rna = fixture("SLOT_WRONG_RNA")
    with pytest.raises(ValueError, match="MATCHED_CANNOT"):
        forward(rna, intervention="matched")
    path = fixture("GATE1_WRONG_PATH")
    with pytest.raises(ValueError, match="MATCHED_CANNOT"):
        forward(path, intervention="matched")
    ordinary = fixture("GATE1")
    mean = ordinary[0].mean_query_context(ordinary[2])
    with pytest.raises(ValueError, match="MATCHED_CANNOT"):
        forward(ordinary, intervention="matched", mean_context=mean)
    assert torch.count_nonzero(forward(rna, intervention="off")["cross_delta"]) == 0


def test_all_masked_returns_original_states_zero_delta_and_no_nan():
    item = fixture("GATE2")
    model, patch, genes, offset, kwargs, old = item
    patch = dict(patch) | dict(patch_mask=torch.zeros_like(patch["patch_mask"]))
    genes = replace(genes, values=torch.zeros_like(genes.values), measured_mask=torch.zeros_like(genes.measured_mask))
    coords = torch.zeros((*patch["patch_mask"].shape, 2))
    kwargs = kwargs | dict(neighborhood=NativePatchGeometry().build(coords, patch["patch_mask"]))
    out = model(patch, genes, offset, **kwargs)
    assert torch.isfinite(out["risk"]).all()
    assert torch.count_nonzero(out["delta_E"]) == torch.count_nonzero(out["delta_G"]) == 0
    assert torch.equal(out["Gfinal"], out["G0_interaction"])
    assert torch.equal(out["Econd"], out["Ebase"])
    assert torch.all(out["patch_cond_read"]["null"] == 1)


def test_nonzero_raw_gene_and_query_gradients_without_frozen_weight_updates():
    item = fixture("GATE2")
    model = item[0]
    out = forward(item)
    loss = out["path_cross_delta"].sum() + out["gene_cross_delta"].sum()
    grads = torch.autograd.grad(loss, (model.query_condition[2].weight, model.gene_rereader.query_projection.weight,
                                      model.slot_path_weight, model.path_state_gate.weight), allow_unused=False)
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(torch.count_nonzero(g) for g in grads)
    assert not any(p.grad is not None for p in model.foundation.parameters())

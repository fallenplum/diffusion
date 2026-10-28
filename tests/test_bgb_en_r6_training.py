"""Small synthetic gradient/cache and wrapper-control tests, no real cohort."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import bgb_en_r6_common as common
from scripts.bgb_en_r6_run import make_references
from src.bgb_en.r6_omics import mapping_from_programs, SparseProgramEncoder, PROGRAM_ORDER
from src.bgb_en.r6_source_model import SourceProgramModel
from src.bgb_en.r6_pathology import source_model_state_sha256
from src.bgb_en.r6_training import full_risk_step, predict, saved_state
from src.phase10_trimodal.math import torch_cox, log_weights


def wrapper(arm="B7"):
    torch.manual_seed(612)
    programs = [dict(id=name, up_genes=[chr(65+i), chr(69+i)], down_genes=[])
                for i, name in enumerate(PROGRAM_ORDER)]
    mapping = mapping_from_programs(programs)
    source = SourceProgramModel("S7", gene_reader=SparseProgramEncoder(mapping),
        tissue_prior=[.3, .3, .4], init_seed=31)
    source.requires_grad_(False).eval()
    n, genes = 6, 9
    feature = dict(rna=np.random.default_rng(9).normal(size=(n, genes)).astype(np.float32),
                   rna_mask=np.ones((n, genes), bool))
    feature["rna_mask"][1, 2] = False
    feature["rna"][~feature["rna_mask"]] = 0.
    prepared = dict(features=feature, base=np.array([.4, -.6, .1, .3, -.2, .5]),
        CP=np.array([.4, -.6, .1, .3, -.2, .5]), RNA=np.linspace(-1., 1., n), beta=.2,
        train=np.arange(4), valid=np.array([4, 5]))
    membership = torch.zeros(50, genes, dtype=torch.bool)
    for k in range(50):
        membership[k, k % genes] = True
        membership[k, (k+2) % genes] = True
    encoded = dict(states=torch.randn(n, 1, 4, 64), patch_mask=torch.ones(n, 1, 4, dtype=torch.bool),
        program_score_prior=torch.randn(n, 1, 4, 4),
        diagnostics=dict(source_state_sha256=source_model_state_sha256(source)))
    neighbor = torch.arange(4)[None, None, :, None].expand(n, 1, 4, 1).clone()
    neighbor = torch.cat((neighbor, neighbor.roll(-1, 2)), -1)
    bank = common.EncodedPatchBank(encoded, neighbor, "cpu")
    model = common.R6PatientModel(arm, prepared, membership, ["outside"] + list(mapping.gene_symbols),
        source, bank, "cpu", source_kind="MATCHED_ST", seed=31)
    model.query_map = np.array([1, 2, 3, 0, 5, 4])
    return model, prepared


def test_cached_update_matches_complete_graph_efron_both_penalties_single_L2():
    a, _ = wrapper()
    b = deepcopy(a)
    idx = torch.arange(4)
    time, event = np.array([1., 2., 2., 4.]), np.array([1, 1, 1, 0])
    oa = torch.optim.SGD([p for p in a.parameters() if p.requires_grad], lr=.03)
    ob = torch.optim.SGD([p for p in b.parameters() if p.requires_grad], lr=.03)
    out = a(idx)
    loss = torch_cox(out["risk"], log_weights(time, event, "cpu"), torch.as_tensor(event))
    loss = loss + .1*out["delta"].square().mean() + .1*out["cross_delta"].square().mean() + a.regularization()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(a.parameters(), 5.)
    oa.step()
    audit = full_risk_step(b, [idx[:2], idx[2:3], idx[3:]], time, event, ob)
    for (name, p), (other_name, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p, q, atol=3e-8, rtol=2e-6, msg=name)
    assert audit["optimizer_updates"] == 1 and audit["full_riskset_patients"] == 4
    assert audit["risk_cache_replay_max"] <= 2e-5
    assert all(p.grad is None for p in b.core.foundation.parameters())


def test_cache_scope_or_nonfinite_regularization_stops_before_update():
    model, _ = wrapper()
    before = saved_state(model)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    time, event = np.array([1., 2., 2., 4.]), np.array([1, 1, 1, 0])
    with pytest.raises(ValueError, match="SCOPE_CHANGED"):
        full_risk_step(model, [torch.tensor([0, 1, 1, 3])], time, event, optimizer)
    model.regularization = lambda: torch.tensor(float("inf"))
    with pytest.raises(RuntimeError, match="NONFINITE_PARAMETER"):
        full_risk_step(model, [torch.arange(4)], time, event, optimizer)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_wrapper_query_controls_preserve_own_evidence_and_source_freeze():
    model, _ = wrapper()
    idx = torch.tensor([0, 1, 2])
    matched = model(idx)
    wrong = model(idx, control="wrong_query")
    for key in ("main_delta", "offset", "gene_values", "gene_measured_mask", "patch_values"):
        assert torch.equal(matched[key], wrong[key]), key
    assert model.hallmark.pool_query.requires_grad is False
    assert all(not p.requires_grad for p in model.hallmark.risk_head.parameters())
    assert all(not p.requires_grad for p in model.core.foundation.parameters())
    model.refresh_training_query_mean(batch_size=2)
    mean = model(idx, control="mean_query")
    assert mean["query_source_mask"].shape == (3, 4)
    assert torch.equal(mean["main_delta"], matched["main_delta"])
    off = model(idx, control="interaction_off")
    assert torch.count_nonzero(off["cross_delta"]) == 0
    whole = model(idx, control="whole_omics")
    expected = model.offset[idx] - model.rna_component[idx] + model.rna_component[torch.tensor([1, 2, 3])]
    assert torch.equal(whole["offset"], expected)
    assert whole["diagnostics"]["RNA_main_effect_replaced"]


def test_wrong_trained_assignment_control_retains_the_wrong_queries():
    model, _ = wrapper("B9")
    idx = torch.tensor([0, 1])
    original = model(idx)
    shuffled = model(idx, control="attention_permutation")
    assert torch.equal(original["query_context_used"], shuffled["query_context_used"])
    assert torch.equal(original["query_source_mask"], shuffled["query_source_mask"])
    assert torch.equal(original["patch_cond_read"]["logits"], shuffled["patch_cond_read"]["logits"])
    assert torch.equal(original["patch_cond_read"]["weights"], shuffled["patch_cond_read"]["weights"])
    assert not torch.equal(original["patch_cond_read"]["states"], shuffled["patch_cond_read"]["states"])
    off = model(idx, control="value_off")
    assert torch.equal(original["query_context_used"], off["query_context_used"])
    assert torch.count_nonzero(off["cross_delta"]) == 0
    own = model(idx, control="matched_query")
    assert not torch.equal(own["query_context_used"], original["query_context_used"])


def test_predict_retains_only_scalar_arrays_not_patch_gene_traces():
    model, _ = wrapper()
    result = predict(model, np.arange(4), batch_size=2)
    assert result["risk"].shape == result["cross_delta"].shape == (4,)
    assert all(value.ndim == 1 and len(value) == 4 for value in result.values())
    assert "patch_cond_read" not in result and "gene_values" not in result


def test_hungarian_pairing_uses_priority_and_is_partition_local_without_events():
    idx = np.arange(8)
    labels = np.array(["a"]*4 + ["b"]*4)
    mapping, audit = common.hungarian_query_mismatch(idx, missing_pattern=labels,
        cancer=["COAD"]*8, stage=["II"]*8, platform=["proxy"]*8, seed=11)
    assert set(mapping[idx]) == set(idx) and np.all(mapping[idx] != idx)
    assert np.array_equal(labels[mapping], labels)
    assert audit["matched_counts"]["RNA_finite_pattern"] == 8
    assert audit["outcomes_accessed"] is False


def test_zero_fit_references_follow_actual_top_level_schema(tmp_path):
    model, prepared = wrapper()
    raw = dict(t=np.arange(1., 7.), e=np.array([1, 1, 0, 1, 1, 0]),
        pfi_valid=np.ones(6, bool), pfi_t=np.arange(1., 7.), pfi_e=np.array([1, 1, 0, 1, 1, 0]))
    args = SimpleNamespace(private=tmp_path, partition=0)
    ctx = dict(raw=raw, lock=dict(contract_sha256="a"*64))
    origin = dict(prepared_receipt_sha256="b"*64)
    make_references(args, ctx, prepared, origin)
    paths = sorted((tmp_path/"references").glob("*.json"))
    assert len(paths) == 6
    for path in paths:
        value = common.read(path)
        assert value["prepared_origin"] == origin and "contract" not in value
        assert value["actual_new_optimizer_fits"] == 0
    hashes = {p.name: common.sha256(p) for p in paths}
    make_references(args, ctx, prepared, origin)
    assert {p.name: common.sha256(p) for p in paths} == hashes

"""Synthetic-only R6 native geometry/source transfer/WSI reading; no fits."""
from dataclasses import replace
import inspect
import math

import pytest
import torch

from src.bgb_en.r6_pathology import (
    FrozenSourceProvenance, NativePatchGeometry, NativeSpatialAdapter, WSIProgramReader,
    attention_assignment_permute, frozen_source_patch_encode, normalize_uni2_patches,
    source_model_state_sha256,
)
from src.bgb_en.r6_source_model import PROGRAMS, SourceProgramModel
from src.bgb_en.r6_omics import GeneEvidenceTransform, SparseProgramEncoder, mapping_from_programs


def random_tensor(shape, seed=62031):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def source_fixture(seed=62031):
    mapping = mapping_from_programs([
        dict(id=name, up_genes=[chr(65 + 2 * k)], down_genes=[chr(66 + 2 * k)])
        for k, name in enumerate(PROGRAMS)])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reader = SparseProgramEncoder(mapping, gene_encoder=GeneEvidenceTransform())
        source = SourceProgramModel("S7", tissue_prior=[.3, .4, .3], gene_reader=reader, init_seed=seed)
    source.requires_grad_(False); source.eval()
    # The receipt digests below are synthetic fixtures, not proof of a real fit.
    provenance = FrozenSourceProvenance("1" * 64, "2" * 64, source_model_state_sha256(source))
    return source, provenance


def patch_fixture():
    x = random_tensor((2, 2, 4, 1536))
    mask = torch.tensor([[[True, True, False, False], [True, True, True, False]],
                         [[True, False, False, False], [False, False, False, False]]])
    x[~mask] = float("nan")
    return x, mask


def test_patch_unit_contract_rejects_bad_valid_data_but_zeros_padding():
    x, mask = patch_fixture()
    normalized = normalize_uni2_patches(x, mask)
    assert normalized.dtype == torch.float32
    torch.testing.assert_close(normalized[mask].norm(dim=-1), torch.ones(int(mask.sum())), atol=2e-7, rtol=0)
    assert torch.equal(normalized[~mask], torch.zeros_like(normalized[~mask]))
    assert torch.isnan(x[~mask]).all()
    for bad in (torch.zeros(1536), torch.full((1536,), float("inf"))):
        changed = x.clone(); changed[0, 0, 0] = bad
        with pytest.raises(ValueError, match="NORM|NONFINITE"):
            normalize_uni2_patches(changed, mask)
    with pytest.raises(ValueError, match="FLOAT32"):
        normalize_uni2_patches(x.double(), mask)


def test_native_geometry_never_crosses_patient_WSI_or_padding_and_caps_budget():
    coords = torch.zeros(2, 2, 20, 2)
    coords[..., 0] = torch.arange(20)
    mask = torch.ones(2, 2, 20, dtype=torch.bool)
    mask[1, 1, 3:] = False; coords[~mask] = float("nan")
    geometry = NativePatchGeometry().build(coords, mask)
    assert geometry.neighbor_indices.shape == (2, 2, 20, 16)
    assert geometry.neighbor_counts[0, 0].eq(16).all()
    assert geometry.neighbor_counts[1, 1, :3].eq(3).all()
    assert geometry.neighbor_indices[~mask].eq(-1).all()
    for b in range(2):
        for s in range(2):
            for n in torch.where(mask[b, s])[0]:
                selected = geometry.neighbor_indices[b, s, n]
                selected = selected[selected >= 0]
                assert n in selected and mask[b, s, selected].all()
                assert torch.linalg.vector_norm(coords[b, s, selected] - coords[b, s, n], dim=-1).max() <= 768
    audit = geometry.public_audit()
    assert audit["cross_patient_or_WSI_edges"] == 0
    assert audit["coordinate_unit"] == "native_level0_coordinate_units_not_micrometers"
    assert audit["radius"] == 768 and not audit["true_coordinates_or_identifiers_exported"]


def test_geometry_cache_isolation_native_radius_and_reorder_equivariance():
    xy = torch.tensor([[[[0., 0.], [512., 0.], [1024., 0.], [0., 512.], [5000., 0.]]]])
    mask = torch.ones(1, 1, 5, dtype=torch.bool)
    builder = NativePatchGeometry(maximum_neighbors=3)
    original = builder(xy, mask)
    assert original.neighbor_indices[0, 0, 4].tolist() == [4, -1, -1]
    original.neighbor_indices.fill_(-1)
    assert builder(xy, mask).neighbor_counts[0, 0, 0] == 3
    permutation = torch.tensor([4, 2, 0, 3, 1])
    actual = builder(xy[:, :, permutation], mask[:, :, permutation])
    reference = builder(xy, mask)
    for new, old in enumerate(permutation):
        got = actual.neighbor_indices[0, 0, new]
        got = permutation[got[got >= 0]]
        want = reference.neighbor_indices[0, 0, old]
        assert torch.equal(got, want[want >= 0])
    with pytest.raises(ValueError, match="HOP"):
        NativePatchGeometry(native_hop=1024)
    with pytest.raises(ValueError, match="BUDGET"):
        NativePatchGeometry(maximum_neighbors=17)
    duplicate = xy.clone(); duplicate[0, 0, 1] = duplicate[0, 0, 0]
    with pytest.raises(ValueError, match="DUPLICATE"):
        builder(duplicate, mask)
    bad = xy.clone(); bad[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NONFINITE"):
        builder(bad, mask)


def test_complete_frozen_source_encode_preserves_every_axis_and_exact_score_prior():
    source, provenance = source_fixture()
    x, mask = patch_fixture()
    out = frozen_source_patch_encode(source, x, mask, source_provenance=provenance, batch=2)
    assert out["states"].shape == (2, 2, 4, 64)
    assert out["program_key_prior"].shape == (2, 2, 4, 4, 64)
    normalized = normalize_uni2_patches(x, mask)
    with torch.no_grad():
        ref = source(normalized[mask])
    for name, ref_name in (("states", "embedding"), ("tissue_probabilities", "tissue_probs"),
                           ("program_key_prior", "queries"), ("program_score_prior", "program_predictions")):
        torch.testing.assert_close(out[name][mask], ref[ref_name], atol=3e-6, rtol=2e-5)
        assert out[name][~mask].eq(0).all() and not out[name].requires_grad
    replay = (out["states"][mask][:, None, :] * out["program_key_prior"][mask]).sum(-1) / math.sqrt(64)
    torch.testing.assert_close(replay + source.program_bias, out["program_score_prior"][mask], atol=1e-7, rtol=1e-5)
    assert out["diagnostics"]["only_valid_source_inputs"]
    assert out["diagnostics"]["source_fit_success_is_not_inferred_from_digest"]
    assert not {"tissue_y", "program_targets", "DSS", "gene_values"} & set(inspect.signature(frozen_source_patch_encode).parameters)


def test_source_feature_gradient_is_enabled_only_by_explicit_flag_and_weights_stay_frozen():
    source, provenance = source_fixture()
    x, mask = patch_fixture(); x.requires_grad_(True)
    with torch.no_grad():
        out = frozen_source_patch_encode(source, x, mask, source_provenance=provenance, enable_grad=True, batch=3)
    out["states"].square().sum().backward()
    assert x.grad[mask].abs().sum() > 0 and x.grad[~mask].eq(0).all()
    assert torch.isfinite(x.grad).all() and all(p.grad is None for p in source.parameters())
    assert not any(p.requires_grad for p in source.parameters())
    with pytest.raises(ValueError, match="SHA_MISMATCH"):
        frozen_source_patch_encode(source, x, mask,
            source_provenance=replace(provenance, complete_source_state_sha256="0" * 64))
    source.program_queries.requires_grad_(True)
    with pytest.raises(ValueError, match="FROZEN_AND_EVAL"):
        frozen_source_patch_encode(source, x, mask, source_provenance=provenance)


def test_source_guard_rejects_projection_only_or_mutated_queries_and_empty_input_is_zero():
    source, provenance = source_fixture()
    x, mask = patch_fixture()
    empty = torch.zeros_like(mask)
    out = frozen_source_patch_encode(source, x, empty, source_provenance=provenance)
    assert out["states"].eq(0).all() and out["program_score_prior"].eq(0).all()
    with pytest.raises(ValueError, match="TRANSFER_CONTRACT"):
        replace(provenance, projection_only_transfer=True)
    with pytest.raises(ValueError, match="EXACT_SHA256"):
        replace(provenance, checkpoint_sha256="not_a_sha")
    with torch.no_grad():
        source.program_queries[0, 0] += .5
    with pytest.raises(ValueError, match="SHA_MISMATCH"):
        frozen_source_patch_encode(source, x, mask, source_provenance=provenance)


def test_native_adapter_has_low_rank_budget_preserves_padding_and_no_neighbor_is_zero():
    z = random_tensor((1, 2, 4, 64)).requires_grad_(True)
    mask = torch.tensor([[[True, True, True, False], [True, False, False, False]]])
    graph = torch.full((1, 2, 4, 3), -1, dtype=torch.long)
    graph[0, 0, :3, :2] = torch.tensor([[0, 1], [1, 2], [2, 0]])
    adapter = NativeSpatialAdapter()
    out = adapter(z, mask, graph)
    assert sum(p.numel() for p in adapter.parameters()) == 512
    assert out["delta"][0, 1].eq(0).all() and out["states"][~mask].eq(0).all()
    assert not out["delta"][0, 0, :3].eq(0).all()
    out["states"].square().sum().backward()
    assert adapter.down.weight.grad.abs().sum() > 0 and adapter.up.weight.grad.abs().sum() > 0
    assert z.grad[~mask].eq(0).all()
    bad = graph.clone(); bad[0, 0, 0, 1] = 3
    with pytest.raises(ValueError, match="POINTS_TO_PADDING"):
        adapter(z, mask, bad)
    bad = graph.clone(); bad[0, 0, 3, 0] = 0
    with pytest.raises(ValueError, match="PADDING_CENTER"):
        adapter(z, mask, bad)


def test_geometry_adapter_and_readout_are_patch_permutation_equivariant():
    coords = torch.tensor([[[[0., 0.], [512., 0.], [1024., 0.], [512., 512.]]]])
    z = random_tensor((1, 1, 4, 64)); mask = torch.ones(1, 1, 4, dtype=torch.bool)
    query = random_tensor((1, 4, 64), seed=62047); pmask = torch.ones(1, 4, dtype=torch.bool)
    geometry, adapter, reader = NativePatchGeometry(), NativeSpatialAdapter(), WSIProgramReader()
    initial = adapter(z, mask, geometry(coords, mask))["states"]
    expected = reader(query, initial, mask, pmask)
    order = torch.tensor([3, 1, 0, 2])
    changed = adapter(z[:, :, order], mask[:, :, order], geometry(coords[:, :, order], mask[:, :, order]))["states"]
    actual = reader(query, changed, mask[:, :, order], pmask)
    torch.testing.assert_close(changed, initial[:, :, order], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual["states"], expected["states"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual["weights"], expected["weights"][..., order], atol=1e-7, rtol=1e-6)


def test_reader_includes_null_and_equal_weights_each_valid_WSI_not_all_patches():
    z = torch.zeros(1, 3, 3, 64)
    mask = torch.tensor([[[True, False, False], [True, True, True], [False, False, False]]])
    z[0, 0, 0] = 1.; z[0, 1] = 10.; z[~mask] = float("nan")
    out = WSIProgramReader()(torch.zeros(1, 2, 64), z, mask, torch.tensor([[True, False]]))
    # Each WSI has its own null: (.5*1 + 3/4*10)/2 = 4, not one pooled-patch read.
    torch.testing.assert_close(out["states"][0, 0], torch.full((64,), 4.))
    assert out["valid_WSI_count"].tolist() == [[2, 0]]
    assert out["states"][0, 1].eq(0).all()
    torch.testing.assert_close(out["weights"].sum(-1) + out["null"], torch.ones(1, 3, 2))
    assert out["weights"][0, 2].eq(0).all() and out["null"][0, 2].eq(1).all()
    assert not out["diagnostics"]["null_is_calibrated_medical_confidence"]
    order = torch.tensor([2, 0, 1])
    reordered = WSIProgramReader()(torch.zeros(1, 2, 64), z[:, order], mask[:, order], torch.tensor([[True, False]]))
    assert torch.equal(reordered["states"], out["states"])
    assert torch.equal(reordered["weights"], out["weights"][:, order])


def test_reader_empty_program_or_patch_is_exact_zero_and_has_no_nan():
    reader = WSIProgramReader()
    for n in (0, 3):
        z = torch.full((2, 2, n, 64), float("nan"))
        mask = torch.zeros(2, 2, n, dtype=torch.bool)
        query = torch.full((2, 4, 64), float("nan"))
        out = reader(query, z, mask, torch.zeros(2, 4, dtype=torch.bool))
        for name in ("states", "per_WSI", "weights", "logits"):
            assert torch.isfinite(out[name]).all() and out[name].eq(0).all()
        assert out["null"].eq(1).all() and not out["program_mask"].any()
        geometry = NativePatchGeometry()(torch.empty(2, 2, n, 2), mask)
        assert NativeSpatialAdapter()(z, mask, geometry)["states"].eq(0).all()


def test_exact_source_prior_includes_bias_and_key_interface_never_double_counts():
    source, provenance = source_fixture()
    with torch.no_grad():
        source.program_bias[:] = torch.tensor([.2, -.1, .3, .8])
    provenance = replace(provenance, complete_source_state_sha256=source_model_state_sha256(source))
    x, mask = patch_fixture()
    encoded = frozen_source_patch_encode(source, x, mask, source_provenance=provenance)
    query, pmask = random_tensor((2, 4, 64), seed=62047), torch.ones(2, 4, dtype=torch.bool)
    reader = WSIProgramReader()
    base = reader(query, encoded["states"], mask, pmask)
    score = reader(query, encoded["states"], mask, pmask, use_source_prior=True,
                   program_score_prior=encoded["program_score_prior"])
    keys = reader(query, encoded["states"], mask, pmask, use_source_prior=True,
                  program_key_prior=encoded["program_key_prior"], program_bias_prior=encoded["program_bias_prior"],
                  source_states_prior=encoded["states"])
    torch.testing.assert_close(score["states"], keys["states"], atol=1e-7, rtol=1e-6)
    observed = mask[:, :, None, :].expand_as(score["logits"])
    expected = .1 * encoded["program_score_prior"].permute(0, 1, 3, 2)
    torch.testing.assert_close((score["logits"] - base["logits"])[observed], expected[observed], atol=1e-7, rtol=1e-5)
    with pytest.raises(ValueError, match="EXACTLY_ONE"):
        reader(query, encoded["states"], mask, pmask, use_source_prior=True,
               program_score_prior=encoded["program_score_prior"], program_key_prior=encoded["program_key_prior"])
    with pytest.raises(ValueError, match="NOT_ENABLED"):
        reader(query, encoded["states"], mask, pmask, program_score_prior=encoded["program_score_prior"])


def test_assignment_shuffle_preserves_weights_and_only_reassigns_own_WSI_values():
    z = random_tensor((2, 2, 4, 64)); mask = torch.ones(2, 2, 4, dtype=torch.bool)
    mask[1, 1, 2:] = False; z[~mask] = float("nan")
    query, pmask = random_tensor((2, 4, 64), seed=62047), torch.ones(2, 4, dtype=torch.bool)
    reader = WSIProgramReader()
    normal = reader(query, z, mask, pmask)
    changed = reader(query, z, mask, pmask, assignment_shuffle_seed=831)
    assert torch.equal(changed["weights"], normal["weights"]) and torch.equal(changed["logits"], normal["logits"])
    assert not torch.allclose(changed["states"], normal["states"])
    permuted = attention_assignment_permute(z, mask, seed=831)
    assert torch.equal(changed["value_assignment"], permuted["assignment"])
    assert torch.equal(permuted["assignment"], attention_assignment_permute(z, mask, seed=831)["assignment"])
    for b in range(2):
        for s in range(2):
            ids = torch.where(mask[b, s])[0]
            targets = permuted["assignment"][b, s, ids]
            assert torch.equal(targets.sort().values, ids)
            assert targets.ne(ids).all()
            torch.testing.assert_close(permuted["values"][b, s, ids], z[b, s, targets])
    assert permuted["values"][~mask].eq(0).all()


def test_reader_query_only_changes_reading_and_has_valid_feature_and_query_gradients():
    z = random_tensor((2, 1, 4, 64)).requires_grad_(True)
    query = random_tensor((2, 4, 64), seed=62047).requires_grad_(True)
    mask, pmask = torch.ones(2, 1, 4, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool)
    reader = WSIProgramReader()
    out = reader(query, z, mask, pmask)
    wrong = reader(query.flip(0), z, mask, pmask)
    assert not torch.allclose(out["weights"], wrong["weights"])
    assert torch.equal(z.detach(), random_tensor((2, 1, 4, 64)))
    out["states"].square().sum().backward()
    assert query.grad.abs().sum() > 0 and z.grad.abs().sum() > 0
    assert torch.isfinite(query.grad).all() and torch.isfinite(z.grad).all()
    replay = reader(query.detach(), z.detach(), mask, pmask)
    assert torch.equal(replay["states"], out["states"].detach())


def test_pathology_module_does_not_load_cohorts_outcomes_or_train_anything():
    import src.bgb_en.r6_pathology as module
    text = inspect.getsource(module)
    assert "torch.load(" not in text and "optimizer" not in text
    assert not any(name in text for name in ("r5_model", "Efron", "Cox", "patient_ids"))
    assert not list(WSIProgramReader().parameters())

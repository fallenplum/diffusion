"""Source trainer exact-gradient/mapping/gate tests; no optimizer fits."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bgb_en_r6_pretrain.py"
SPEC = importlib.util.spec_from_file_location("bgb_en_r6_pretrain_test_module", SCRIPT)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class ToyDataset(SimpleNamespace):
    def __len__(self):
        return len(self.features_unit)


def source_fixture():
    from src.bgb_en.r6_omics import mapping_from_programs
    rng = np.random.default_rng(61031)
    mapping = mapping_from_programs([dict(id=name, up_genes=[str(i)], down_genes=[str(i + 4)])
                                    for i, name in enumerate(p.PROGRAMS)])
    n = 144
    donor = np.repeat(np.arange(6), 24)
    section = donor.copy()
    graph = np.c_[np.arange(n), np.concatenate([np.roll(np.arange(i * 24, (i + 1) * 24), 1) for i in range(6)])]
    x = rng.normal(size=(n, 1536)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    data = ToyDataset(features_unit=x, donor_codes=donor, section_codes=section,
        program_targets=rng.normal(size=(n, 4)), tissue_y=np.tile(np.repeat(np.arange(3), 8), 6),
        log1p_cp10k=rng.normal(size=(n, 8)), measured_mask=np.ones((n, 8), bool),
        library_size=np.tile(np.linspace(100, 10000, 24), 6),
        gene_symbols=np.array(mapping.gene_symbols), neighbors=graph)
    return data, mapping


class ToySource(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(5, 4, bias=True).double()
        self.gene = nn.Linear(3, 4, bias=False).double()

    def forward(self, visual, gene):
        return dict(visual=self.visual(visual), gene=self.gene(gene))


def test_two_pass_gradient_cache_equals_full_objective_including_all_negative_keys_and_single_L2():
    torch.manual_seed(61031)
    full = ToySource()
    cached = deepcopy(full)
    x = torch.randn(11, 5, dtype=torch.float64)
    g = torch.randn(11, 3, dtype=torch.float64)
    y = torch.randn(11, 4, dtype=torch.float64)

    def objective(outputs):
        v, mol = outputs["visual"], outputs["gene"]
        # The contrastive denominator connects every microbatch's molecular
        # keys to other microbatches' visual queries; detaching negative keys
        # would fail the equality asserted below.
        score = F.normalize(v, dim=-1) @ F.normalize(mol, dim=-1).T / .2
        contrastive = F.cross_entropy(score, torch.arange(11))
        contrast = F.huber_loss(v[:-1] - v[1:], y[:-1] - y[1:])
        return contrastive + contrast, dict(contrastive=contrastive, contrast=contrast)

    value, _ = objective(full(x, g))
    penalty = .0001 * sum(param.square().sum() for param in full.parameters())
    (value + penalty).backward()
    audit = p.deterministic_output_gradient_cache_backward(cached,
        lambda start, end: cached(x[start:end], g[start:end]), len(x), objective,
        microbatch=3, parameter_L2=.0001, tolerance=1e-12)
    assert audit["microbatches"] == 4 and audit["complete_source_updates_represented"] == 1
    assert audit["negative_key_gradients_included"] and audit["replay_max_abs_error"] < 1e-12
    assert abs(audit["parameter_L2"] - penalty.detach().item()) < 1e-15
    for reference, actual in zip(full.parameters(), cached.parameters()):
        torch.testing.assert_close(reference.grad, actual.grad, atol=2e-14, rtol=2e-13)


def test_cache_does_not_silently_accept_stochastic_output_replay():
    source = nn.Linear(2, 1)
    x = torch.randn(5, 2)
    calls = 0

    def changing(start, end):
        nonlocal calls
        calls += 1
        return dict(value=source(x[start:end]) + calls)

    with pytest.raises(ValueError, match="OUTPUT_REPLAY_FAILED"):
        p.deterministic_output_gradient_cache_backward(source, changing, 5,
            lambda out: (out["value"].square().mean(), {}), microbatch=2)
    with pytest.raises(ValueError, match="DETERMINISTIC_FORWARD"):
        p.deterministic_output_gradient_cache_backward(nn.Sequential(nn.Linear(2, 2), nn.Dropout()),
            changing, 5, lambda out: (out["value"].sum(), {}), microbatch=2)


def test_real_S7_source_cache_matches_full_shared_gene_stem_gradient():
    from src.bgb_en.r6_omics import mapping_from_programs, GeneValueBank, GeneEvidenceTransform, SparseProgramEncoder
    from src.bgb_en.r6_source_model import SourceProgramModel, PROGRAMS, source_loss
    torch.manual_seed(61031)
    mapping = mapping_from_programs([dict(id=name, up_genes=[str(i)], down_genes=[str(i + 4)])
                                    for i, name in enumerate(PROGRAMS)])
    bank = GeneValueBank(mapping.gene_symbols, mapping.gene_symbols)
    reader = SparseProgramEncoder(mapping, gene_encoder=GeneEvidenceTransform())
    full = SourceProgramModel("S7", tissue_prior=[.3, .4, .3], gene_reader=reader, init_seed=31)
    cached = deepcopy(full)
    x = F.normalize(torch.randn(7, 1536), dim=-1)
    raw = torch.randn(7, 8)
    mask = torch.ones_like(raw, dtype=torch.bool)
    target = torch.randn(7, 4)
    tissue = torch.arange(7) % 3
    negative = torch.stack(((torch.arange(7) + 1) % 7, (torch.arange(7) + 3) % 7), -1)

    def forward(source, start, end):
        out = source(x[start:end])
        gene = source.encode_gene_programs(bank(raw[start:end], mask[start:end]))
        return {"program_predictions": out["program_predictions"], "tissue_logits": out["tissue_logits"],
                "visual_program_embeddings": out["visual_program_embeddings"], "gene_tokens": gene["tokens"]}

    def objective(source, out):
        return source_loss(source, out, target, tissue, gene_programs=dict(tokens=out["gene_tokens"],
            program_mask=torch.ones(7, 4, dtype=torch.bool)), negative_index=negative)

    loss, _ = objective(full, forward(full, 0, 7))
    (loss + .0001 * sum(param.square().sum() for param in full.parameters() if param.requires_grad)).backward()
    audit = p.deterministic_output_gradient_cache_backward(cached, lambda a, b: forward(cached, a, b),
        7, lambda out: objective(cached, out), microbatch=3, parameter_L2=.0001)
    assert audit["replay_max_abs_error"] < 2e-6
    for (name, reference), (_, actual) in zip(full.named_parameters(), cached.named_parameters()):
        if reference.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(reference.grad, actual.grad, atol=3e-7, rtol=2e-4, msg=name)


def test_within_source_joint_shuffle_preserves_section_tissue_depth_and_mask_rows():
    section = np.repeat(np.arange(2), 24)
    tissue = np.tile(np.repeat(np.arange(3), 8), 2)
    library = np.tile(np.repeat([100., 200., 300., 400.], 6), 2)
    rows = np.arange(24)
    mapping, audit = p.within_section_tissue_depth_permutation(section, tissue, library, rows, seed=31)
    assert np.array_equal(mapping[24:], np.arange(24, 48))
    assert np.array_equal(np.sort(mapping[rows]), rows)
    assert np.array_equal(section[mapping[rows]], section[rows])
    assert np.array_equal(tissue[mapping[rows]], tissue[rows])
    assert audit["changed_rows"] > 0 and audit["bins_fitted_on_training_section_only"]
    library[24:] = 1e20
    changed, _ = p.within_section_tissue_depth_permutation(section, tissue, library, rows, seed=31)
    assert np.array_equal(mapping, changed)
    y = np.arange(48 * 4).reshape(48, 4)
    valid = np.ones_like(y, bool); valid[2, 1] = False
    assert p.joint_permutation_audit(y, valid, mapping, rows)["unweighted_joint_vector_and_mask_multiset_preserved"]


def test_wrong_neighborhood_uses_only_section_geometry_and_preserves_degree_and_self_edges():
    graph = np.array([[0, 1, -1], [1, 0, 2], [2, 1, 3], [3, 2, -1],
                      [4, 5, -1], [5, 4, 6], [6, 5, 7], [7, 6, -1]])
    section = np.repeat([0, 1], 4)
    sham, audit = p.geometry_sham_graph(graph, section, seed=31)
    assert audit["cross_section_edges"] == 0 and not audit["tissue_or_depth_labels_used"]
    assert not audit["composition_matched_claimed"] and not audit["physically_real_300um_neighbors_claimed"]
    assert np.array_equal(np.sort((graph >= 0).sum(1)), np.sort((sham >= 0).sum(1)))
    assert all(row in sham[row] for row in range(len(sham)))
    assert np.all(section[np.maximum(sham, 0)][sham >= 0] == np.broadcast_to(section[:, None], sham.shape)[sham >= 0])
    # A section-preserving node relabel does not use tissue truth, source
    # expression, measured depth or regression targets at either inference.
    assert not np.array_equal(graph, sham)


def test_alignment_negatives_are_same_donor_section_tissue_nearest_depth_no_self_or_replacement():
    donor = np.repeat([0, 1], 24)
    section = np.repeat(np.arange(4), 12)
    tissue = np.tile(np.repeat([0, 1], 6), 4)
    library = np.linspace(100, 2000, 48)
    indices = np.arange(36)
    negatives, audit = p.hard_negative_indices(donor, section, tissue, library, indices)
    assert audit["eligible_spots"] == 36 and audit["complete_candidate_spots"] == 0
    assert not np.any(negatives == np.arange(36)[:, None])
    for i, row in enumerate(negatives):
        chosen = row[row >= 0]
        assert len(set(chosen)) == len(chosen) == 5
        assert np.all(donor[chosen] == donor[i])
        assert np.all(section[chosen] == section[i])
        assert np.all(tissue[chosen] == tissue[i])
        depth_gaps = np.abs(np.log1p(library[chosen]) - np.log1p(library[i]))
        assert np.all(np.diff(depth_gaps) >= 0)


def test_source_metrics_use_true_targets_and_disclose_unidentifiable_subgroups():
    truth = np.arange(32, dtype=float).reshape(8, 4) / 7
    tissue = np.arange(8) % 3
    logits = np.zeros((8, 3)); logits[np.arange(8), tissue] = 20
    out = dict(program_predictions=truth.copy(), tissue_logits=logits, embedding=np.ones((8, 64)))
    metrics = p.evaluate_source_outputs(out, truth, tissue, np.arange(1, 9) * 100, class_weights=np.ones(3))
    assert metrics["program_Huber"] == 0 and metrics["mean_program_Spearman"] == 1
    assert metrics["tissue_BA"] == 1 and metrics["tissue_present_classes"] == 3
    assert metrics["embedding_norm_log_library_Spearman"] is None
    out["program_predictions"] *= -1
    wrong = p.evaluate_source_outputs(out, truth, tissue, np.arange(1, 9), class_weights=np.ones(3))
    assert wrong["mean_program_Spearman"] == -1 and wrong["program_Huber"] > 0


def test_launch_gate_and_immutable_code_lock_fail_closed(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps(dict(r6_formal_training_allowed=False)), encoding="utf-8")
    with pytest.raises(ValueError, match="NOT_PASSED"):
        p.verify_launch_gate(gate, p.sha256(gate))
    gate.write_text(json.dumps(dict(r6_formal_training_allowed=True)), encoding="utf-8")
    assert p.verify_launch_gate(gate, p.sha256(gate))["r6_formal_training_allowed"]
    with pytest.raises(ValueError, match="SHA_CHANGED"):
        p.verify_launch_gate(gate, "0" * 64)
    config = dict(schema="bgb_en_r6_source_pretrain_lock_v1", status="LOCKED_BEFORE_SOURCE_FITS",
        source_code_sha256=p.current_code_hashes(), fixed_defaults=dict(p.FIXED_DEFAULTS),
        source_cache_manifest_sha256="0" * 64, fresh_certificate_sha256="1" * 64,
        cells=[dict(arm="S7", control="MATCHED", heldout_donor=-1, seed=31)])
    assert p.validate_lock(config)["epochs"] == 80
    config["fixed_defaults"]["epochs"] = 81
    with pytest.raises(ValueError, match="TRAINING_CONTRACT_CHANGED"):
        p.validate_lock(config)
    config["fixed_defaults"]["epochs"] = 80
    config["source_code_sha256"]["src/bgb_en/r6_omics.py"] = "0" * 64
    with pytest.raises(ValueError, match="SOURCE_CODE_SHA_GUARD"):
        p.validate_lock(config)


def test_partial_or_error_cell_is_preserved_and_never_automatically_retrained(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "validate_lock", lambda _: p.FIXED_DEFAULTS)
    cell = dict(arm="S4", control="MATCHED", heldout_donor=0, seed=31)
    directory = tmp_path / "source_cells/S4_MATCHED_h0_seed31"
    directory.mkdir(parents=True)
    (directory / "ERROR.json").write_text('{"sentinel":"preserved"}', encoding="utf-8")
    with pytest.raises(ValueError, match="INCOMPLETE_SOURCE_CELL_PRESERVED"):
        p.fit_cell(dict(cells=[cell]), 0, None, None, tmp_path, device="cpu",
                   launch_gate=dict(r6_formal_training_allowed=True), resume=True)
    assert (directory / "ERROR.json").read_text(encoding="utf-8") == '{"sentinel":"preserved"}'


def test_only_validated_complete_receipt_can_be_reused_without_a_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "validate_lock", lambda _: p.FIXED_DEFAULTS)
    cell = dict(arm="S4", control="MATCHED", heldout_donor=0, seed=31)
    config = dict(cells=[cell])
    directory = tmp_path / "source_cells/S4_MATCHED_h0_seed31"
    directory.mkdir(parents=True)
    artifact = directory / "artifact.private.bin"; artifact.write_bytes(b"immutable")
    receipt = dict(status="COMPLETE_R6_SOURCE_FIT", config_sha256=p.canonical_hash(config),
                   source_code_sha256=p.current_code_hashes(), cell=cell,
                   epochs=80, optimizer_updates=80,
                   artifact_sha256={artifact.name: p.sha256(artifact)})
    (directory / "COMPLETE.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="VERIFIED_RESUME"):
        p.fit_cell(config, 0, None, None, tmp_path, device="cpu", launch_gate=dict(r6_formal_training_allowed=True))
    assert p.fit_cell(config, 0, None, None, tmp_path, device="cpu", launch_gate=dict(r6_formal_training_allowed=True), resume=True) == receipt
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="ARTIFACT_CHANGED"):
        p.fit_cell(config, 0, None, None, tmp_path, device="cpu", launch_gate=dict(r6_formal_training_allowed=True), resume=True)


def test_full_source_cells_have_no_fake_heldout_and_sham_types_are_explicit():
    p.validate_cell(dict(arm="S7", control="WRONG_DONOR", heldout_donor=-1, seed=31))
    p.validate_cell(dict(arm="S7", control="WITHIN_SECTION_TISSUE_DEPTH", heldout_donor=-1, seed=31))
    with pytest.raises(ValueError, match="FIXED_S7_ONLY"):
        p.validate_cell(dict(arm="S4", control="MATCHED", heldout_donor=-1, seed=31))
    with pytest.raises(ValueError, match="NOT_PREDEFINED"):
        p.validate_cell(dict(arm="S6", control="WRONG_DONOR", heldout_donor=0, seed=31))


@pytest.mark.parametrize("control", ["WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH", "WRONG_SPOT_ALIGNMENT"])
def test_S7_shams_route_actual_joint_gene_rows_without_label_or_positive_pair_bypass(control):
    dataset, mapping = source_fixture()
    cell = dict(arm="S7", control=control, heldout_donor=-1, seed=31)
    data = p.SourceFitData(dataset, cell, mapping, "cpu")
    source = p._model_for_cell(cell, data, mapping, "cpu")
    rows = np.arange(6)
    out = data.chunk(source, rows, training_alignment=True)
    expected_gene = source.encode_gene_programs(data.gene_bank(
        torch.as_tensor(data.gene_z[data.control_mapping[rows]]),
        torch.as_tensor(dataset.measured_mask[data.control_mapping[rows]])))
    assert torch.equal(out["gene_tokens"], expected_gene["tokens"])
    assert not np.array_equal(data.control_mapping, np.arange(len(dataset)))
    true_image = source(torch.as_tensor(dataset.features_unit[rows]))["program_predictions"]
    assert torch.equal(out["program_predictions"], true_image)
    all_outputs = p._forward_small(source, data, 0, len(dataset))
    _, parts = data.objective(source)(all_outputs)
    target_rows = np.arange(len(dataset)) if control == "WRONG_SPOT_ALIGNMENT" else data.control_mapping
    target = torch.as_tensor(data.target_z[target_rows], dtype=torch.float32)
    expected_huber = F.huber_loss(all_outputs["program_predictions"], target)
    torch.testing.assert_close(parts["program_huber"], expected_huber)
    assert parts["joint_target_and_gene_sham"] == (control != "WRONG_SPOT_ALIGNMENT")
    assert parts["program_target_is_wrong_donor"] == (control == "WRONG_DONOR")
    if control == "WRONG_DONOR":
        assert np.all(dataset.donor_codes[data.control_mapping] != dataset.donor_codes)
    assert len(data.heldout) == 0


def test_random_reference_checkpoint_snapshot_and_independent_replay_have_zero_updates(tmp_path):
    dataset, mapping = source_fixture()
    cell = dict(arm="S0", control="MATCHED", heldout_donor=0, seed=31)
    config = dict(schema="bgb_en_r6_source_pretrain_lock_v1", status="LOCKED_BEFORE_SOURCE_FITS",
        source_code_sha256=p.current_code_hashes(), fixed_defaults=dict(p.FIXED_DEFAULTS),
        source_cache_manifest_sha256="0" * 64, fresh_certificate_sha256="1" * 64, cells=[cell])
    directory = tmp_path / "source_cells/S0_MATCHED_h0_seed31"
    directory.mkdir(parents=True)
    data = p.SourceFitData(dataset, cell, mapping, "cpu")
    source = p._model_for_cell(cell, data, mapping, "cpu")
    assert not any(param.requires_grad for param in source.parameters())
    snapshot = p.snapshot(source, None, data, directory, 80, config_sha=p.canonical_hash(config),
        source_hashes=p.current_code_hashes(), microbatch=128, launch_gate=None)
    assert snapshot["optimizer_updates"] == 0
    assert snapshot["metrics"]["heldout"]["spots"] == 24
    receipt = dict(status="COMPLETE_R6_RANDOM_REFERENCE_NO_OPTIMIZER", cell=cell,
        config_sha256=p.canonical_hash(config), source_code_sha256=p.current_code_hashes(),
        epochs=80, optimizer_updates=0,
        artifact_sha256=snapshot["artifact_sha256"], metrics=snapshot["metrics"])
    (directory / "COMPLETE.json").write_text(json.dumps(receipt), encoding="utf-8")
    replay = p.replay_cell(config, 0, dataset, mapping, tmp_path, device="cpu")
    assert replay["status"] == "PASS_COMPLETE_R6_SOURCE_CHECKPOINT_REPLAY"
    assert replay["optimizer_updates"] == replay["new_optimizer_fits"] == 0
    assert replay["image_only_heldout_program_prediction"]
    assert replay["original_fit_artifacts_modified"] is False
    assert max(replay["forward_max_abs_errors"]["heldout"].values()) == 0


def test_wrong_seed_self_consistent_bundle_cannot_pass_source_replay(tmp_path):
    dataset, mapping = source_fixture()
    first = dict(arm="S0", control="MATCHED", heldout_donor=0, seed=31)
    second = dict(arm="S0", control="MATCHED", heldout_donor=0, seed=47)
    config = dict(schema="bgb_en_r6_source_pretrain_lock_v1", status="LOCKED_BEFORE_SOURCE_FITS",
        source_code_sha256=p.current_code_hashes(), fixed_defaults=dict(p.FIXED_DEFAULTS),
        source_cache_manifest_sha256="0" * 64, fresh_certificate_sha256="1" * 64, cells=[first, second])
    directory = tmp_path / "source_cells/S0_MATCHED_h0_seed31"
    directory.mkdir(parents=True)
    data = p.SourceFitData(dataset, first, mapping, "cpu")
    source = p._model_for_cell(first, data, mapping, "cpu")
    snapshot = p.snapshot(source, None, data, directory, 80, config_sha=p.canonical_hash(config),
        source_hashes=p.current_code_hashes(), microbatch=128, launch_gate=None)
    receipt = dict(status="COMPLETE_R6_RANDOM_REFERENCE_NO_OPTIMIZER", cell=first,
        epochs=80, optimizer_updates=0, config_sha256=p.canonical_hash(config),
        source_code_sha256=p.current_code_hashes(), artifact_sha256=snapshot["artifact_sha256"], metrics=snapshot["metrics"])
    (directory / "COMPLETE.json").write_text(json.dumps(receipt), encoding="utf-8")
    copied = tmp_path / "source_cells/S0_MATCHED_h0_seed47"
    shutil.copytree(directory, copied)
    with pytest.raises(ValueError, match="RECEIPT_LINEAGE_MISMATCH"):
        p.replay_cell(config, 1, dataset, mapping, tmp_path, device="cpu")
    # Forging the outer receipt's identity does not fix the inner checkpoint.
    receipt["cell"] = second
    (copied / "COMPLETE.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="STATE_LINEAGE_OR_FEATURE_CONTRACT"):
        p.replay_cell(config, 1, dataset, mapping, tmp_path, device="cpu")

"""Finite synthetic execution/receipt tests; no real patient cache or training.

Fake report receipts below test parser/selection logic only. They never assert
that a real optimizer, checkpoint forward, or scientific experiment occurred.
"""
from argparse import Namespace
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import Ridge
import torch

from src.semantic_b0b7.pretext import PairMetadata, negative_support, paired_info_nce


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("semantic_b0b7_p1_synthetic_review",
                                            ROOT / "scripts/semantic_b0b7_p1.py")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def json_write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def fake_gate(tmp_path):
    # This is deliberately not a model/cache; rejection must precede torch.load.
    input_path = tmp_path / "synthetic_not_a_patient_cache.pt"
    input_path.write_bytes(b"synthetic-gate-binding-test-only")
    sha = RUN.sha_file(input_path)
    gate = {"schema": "semantic_b0b7_p0_gate_v1", "status": "PASS",
            "input_sha256": sha, "frozen_sources": RUN.source_hashes(),
            "hard_gates": {name: {"status": "PASS"} for name in RUN.HARD_GATES},
            "probe_evidence": {"optimizer_steps": 0, "input_sha256": sha,
                               "actual_forward_backward": True, "gradients_finite": True}}
    path = tmp_path / "synthetic_gate.json"
    json_write(path, gate)
    return Namespace(inputs=str(input_path), expected_input_sha256=sha, gate=str(path)), gate


def forbid_load(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("UNQUALIFIED_INPUT_WAS_LOADED")
    monkeypatch.setattr(RUN.torch, "load", forbidden)


def test_input_sha_gate_precedes_pickle_or_large_tensor_loading(tmp_path, monkeypatch):
    args, _ = fake_gate(tmp_path)
    Path(args.inputs).write_bytes(b"tampered")
    forbid_load(monkeypatch)
    with pytest.raises(ValueError, match="INPUT_SHA_MISMATCH"):
        RUN.load_gate_and_inputs(args)


def test_source_hash_mismatch_precedes_data_loading(tmp_path, monkeypatch):
    args, gate = fake_gate(tmp_path)
    gate["frozen_sources"][RUN.SOURCE_PATHS[0]] = "0" * 64
    json_write(Path(args.gate), gate)
    forbid_load(monkeypatch)
    with pytest.raises(ValueError, match="P0_GATE_SOURCE_BINDING"):
        RUN.load_gate_and_inputs(args)


@pytest.mark.parametrize("gate_name", RUN.HARD_GATES)
def test_each_real_hard_gate_required(tmp_path, monkeypatch, gate_name):
    args, gate = fake_gate(tmp_path)
    gate["hard_gates"][gate_name]["status"] = "PENDING"
    json_write(Path(args.gate), gate)
    forbid_load(monkeypatch)
    with pytest.raises(ValueError, match="MISSING_OR_FAILED_P0_HARD_GATE"):
        RUN.load_gate_and_inputs(args)


def test_zero_optimizer_probe_is_required_before_loading(tmp_path, monkeypatch):
    args, gate = fake_gate(tmp_path)
    gate["probe_evidence"]["optimizer_steps"] = 1
    json_write(Path(args.gate), gate)
    forbid_load(monkeypatch)
    with pytest.raises(ValueError, match="P0_REAL_PROBE_EVIDENCE_REQUIRED"):
        RUN.load_gate_and_inputs(args)


def test_profile_must_match_exact_current_sources_and_input(tmp_path):
    binding = {"frozen_sources": {"source": "a"}, "input_sha256": "b", "gate_sha256": "c"}
    receipt = {"schema": "semantic_b0b7_p1_actual_profile_v1", "status": "PASS", **binding,
               "optimizer_steps": 0, "all_parameter_gradients_finite_nonzero": True,
               "runtime": RUN.configure_runtime()}
    path = tmp_path / "synthetic_profile.json"
    args = Namespace(profile=str(path))
    json_write(path, receipt)
    assert RUN.validate_profile(args, binding) == RUN.sha_file(path)
    receipt["frozen_sources"] = {"source": "changed"}
    json_write(path, receipt)
    with pytest.raises(ValueError, match="PROFILE_NOT_AN_ACTUAL"):
        RUN.validate_profile(args, binding)


def test_800_cannot_bypass_group_extension_with_a_fresh_fit(monkeypatch):
    monkeypatch.setattr(RUN, "load_gate_and_inputs", lambda args: (_ for _ in ()).throw(AssertionError("loaded")))
    args = Namespace(arm="B4", seed=31, max_epochs=800, resume=False)
    with pytest.raises(ValueError, match="800|EXTENSION|RESUME"):
        RUN.cell(args)


def test_400_cannot_be_restarted_as_an_extension(monkeypatch):
    monkeypatch.setattr(RUN, "load_gate_and_inputs", lambda args: (_ for _ in ()).throw(AssertionError("loaded")))
    args = Namespace(arm="B4", seed=31, max_epochs=400, resume=True)
    with pytest.raises(ValueError, match="400|800|RESUME"):
        RUN.cell(args)


def test_masked_huber_is_patient_equal_and_nan_targets_not_zero_truth():
    pred = np.array([[1, 99], [2, 4], [3, -4]], float)
    target = np.array([[0, np.nan], [1, 4], [5, -3]], float)
    mask = np.array([[1, 0], [1, 1], [1, 1]], bool)
    report = RUN.program_metrics(pred, target, mask)
    assert report["patient_equal_masked_huber"] == pytest.approx((.5 + .25 + 1.) / 3)
    assert report["measured_target_cells"] == 5
    assert report["valid_pearson_programs"] == 1
    assert report["per_program"][1]["pearson"] is None
    assert report["per_program"][0]["pearson"] == pytest.approx(np.corrcoef(pred[:, 0], target[:, 0])[0, 1])


def test_retrieval_tie_statistics_and_zero_negative_support():
    scores = np.array([[2., 2., 0.], [2., 2., 0.], [1., 0., 2.]])
    valid = np.ones((3, 3), bool)
    negative = ~np.eye(3, dtype=bool)
    report = RUN.retrieval_metrics(scores, valid, negative)
    assert report["image_to_gene"]["anchors_covered"] == 3
    assert report["image_to_gene"]["top1_tie_averaged"] == pytest.approx((.5 + .5 + 1) / 3)
    empty = RUN.retrieval_metrics(scores, valid, np.zeros((3, 3), bool))
    assert empty["image_to_gene"]["anchors_covered"] == 0
    assert empty["image_to_gene"]["paired_gap_mean"] is None


def test_dual_ridge_matches_sklearn_sum_squares_not_n_scaled_penalty():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(12, 37))
    x[:, -1] = 1
    valid_x = rng.normal(size=(4, 37)); valid_x[:, -1] = 1
    y = rng.normal(size=(12, 3))
    mask = np.ones_like(y, dtype=bool)
    mask[-3:, 1:] = False
    alpha = 1.7
    predicted, state = RUN.fit_masked_ridge(x, valid_x, y, mask, alpha=alpha)
    center, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1
    train_z, valid_z = (x-center)/scale, (valid_x-center)/scale
    for k in range(3):
        oracle = Ridge(alpha=alpha, fit_intercept=True, solver="svd").fit(train_z[mask[:, k]], y[mask[:, k], k])
        assert np.allclose(predicted[:, k], oracle.predict(valid_z), rtol=2e-12, atol=2e-12)
    assert state["linear_system_scalar_rhs"] == 3
    assert state["linear_solver_calls_factoring_shared_masks"] == 2


def test_unmeasured_targets_do_not_affect_ridge_and_eval_has_no_fit_role():
    rng = np.random.default_rng(91)
    x, valid_x, y = rng.normal(size=(9, 5)), rng.normal(size=(2, 5)), rng.normal(size=(9, 2))
    mask = np.ones_like(y, dtype=bool); mask[::2, 1] = False
    first, state = RUN.fit_masked_ridge(x, valid_x, y, mask)
    altered = y.copy(); altered[~mask] = 1e12
    second, other = RUN.fit_masked_ridge(x, valid_x, altered, mask)
    assert np.array_equal(first, second)
    assert np.array_equal(state["standardize_center"], other["standardize_center"])
    third, changed_eval = RUN.fit_masked_ridge(x, valid_x+1e6, y, mask)
    assert np.array_equal(state["standardize_center"], changed_eval["standardize_center"])
    assert np.array_equal(state["standardize_scale"], changed_eval["standardize_scale"])
    for a, b in zip(state["program_fits"], changed_eval["program_fits"]):
        assert np.array_equal(a["coefficient"], b["coefficient"])
    assert not np.array_equal(first, third)


def test_ridge_refuses_unsupported_program_instead_of_scoring_filled_targets():
    with pytest.raises(ValueError, match="INSUFFICIENT_MEASURED_PROGRAM_FOR_RIDGE"):
        RUN.fit_masked_ridge(np.eye(3), np.ones((1, 3)), np.ones((3, 2)),
                             np.array([[1, 0], [1, 0], [1, 0]], bool))


def test_first_round_constants_match_lock_recipe():
    recipe = json.loads((ROOT / "configs/semantic_b0b7/p0_p1_v1.json").read_text(encoding="utf-8"))
    assert recipe["scope"] == "P0_P1_AND_MINIMAL_REAL_PROFILE_ONLY"
    assert recipe["formal_DSS_training_authorized_this_round"] is False
    assert list(RUN.ARMS) == recipe["P1_matrix"]["neural_arms"]
    assert list(RUN.SEEDS) == recipe["P1_scope"]["seeds"]
    assert list(RUN.CHECKPOINTS) == recipe["training"]["checkpoints"]


def test_report_refuses_a_completed_receipt_from_other_sources(tmp_path, monkeypatch):
    binding = {"input_sha256": "synthetic-input", "frozen_sources": {"source": "new"}}
    monkeypatch.setattr(RUN, "load_gate_and_inputs", lambda args: ({}, binding))
    path = tmp_path / "cells" / (RUN.ARMS[0] + "_seed" + str(RUN.SEEDS[0])) / "cell.private.json"
    path.parent.mkdir(parents=True)
    json_write(path, {"status": "COMPLETE", "arm": RUN.ARMS[0], "seed": RUN.SEEDS[0],
                      "input_sha256": "synthetic-input", "frozen_sources": {"source": "old"},
                      "optimizer_steps": 400})
    with pytest.raises(ValueError, match="CELL_COMPLETION_BINDING_MISMATCH"):
        RUN.report(Namespace(private_root=str(tmp_path)))


@pytest.mark.parametrize("n,expected_negative,expected_anchors", [(2, 0, 0), (3, 1, 3)])
def test_common_support_eliminates_two_person_wrong_positive_coverage_confound(n, expected_negative, expected_anchors):
    metadata = PairMetadata(np.arange(n), np.zeros(n, int), np.zeros(n, int), np.zeros(n, int))
    common, wrong, audit = RUN.common_negative_support(metadata, seed=31)
    assert not np.any(wrong == np.arange(n))
    assert np.all(common.sum(1) == expected_negative)
    assert not np.diag(common).any() and not common[np.arange(n), wrong].any()
    assert audit["after"]["anchors_with_negative"] == expected_anchors
    valid = torch.ones((n, n), dtype=torch.bool)
    scores = torch.arange(n*n, dtype=torch.float32).reshape(n, n).requires_grad_()
    negative = torch.tensor(common)
    identity_loss = paired_info_nce(scores, valid, negative)
    wrong_loss = paired_info_nce(scores, valid, negative, positive=torch.tensor(wrong))
    first = RUN.actual_pair_support(valid, negative)
    second = RUN.actual_pair_support(valid, negative, positive=torch.tensor(wrong))
    assert first == second
    assert identity_loss["image_anchors_used"] == wrong_loss["image_anchors_used"] == expected_anchors
    assert identity_loss["gene_anchors_used"] == wrong_loss["gene_anchors_used"] == expected_anchors
    assert torch.isfinite(identity_loss["loss"]) and torch.isfinite(wrong_loss["loss"])
    (identity_loss["loss"] + wrong_loss["loss"]).backward()
    assert torch.isfinite(scores.grad).all()


def test_common_train_support_does_not_mutate_validation_candidate_bank():
    stage = np.array([0, 0, 1, 1, 1])
    metadata = PairMetadata(np.arange(5), stage, np.zeros(5, int), np.zeros(5, int))
    original, audit = negative_support(metadata, mode="matched")
    train = (None, None, None, None, metadata, torch.tensor(original), None, audit, {})
    changed, wrong, common_audit = RUN.train_common_support(train, 47)
    assert np.array_equal(train[5].numpy(), original)
    assert not np.array_equal(changed[5].numpy(), original)
    assert common_audit["after"]["anchors_with_negative"] == 3
    assert common_audit["after"]["candidate_negative_pairs"] == 3
    assert np.array_equal(negative_support(metadata, mode="matched")[0], original)
    assert changed[4] is metadata and changed[7] is common_audit


def test_checkpoint_replay_context_preserves_cpu_rng_and_next_draws():
    torch.manual_seed(9147)
    before = torch.get_rng_state().clone()
    reference = torch.Generator().set_state(before)
    expected = torch.rand(13, generator=reference)
    with RUN.preserve_replay_rng("cpu"):
        torch.manual_seed(31)
        torch.rand(79)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(torch.rand(13), expected)


def test_checkpoint_replay_protects_all_simulated_cuda_generators(monkeypatch):
    # Actual torch fork_rng executes unchanged; only two CUDA RNG backends are
    # simulated to verify that every visible device, not just cuda:0, is saved.
    generators = [torch.Generator().manual_seed(1021), torch.Generator().manual_seed(1022)]
    before = [g.get_state().clone() for g in generators]
    expected = [torch.rand(13, generator=torch.Generator().set_state(s)) for s in before]
    cpu_before = torch.get_rng_state().clone()
    cpu_expected = torch.rand(13, generator=torch.Generator().set_state(cpu_before))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device=0: generators[int(device)].get_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device=0: generators[int(device)].set_state(state))
    def reset_all(seed):
        for g in generators:
            g.manual_seed(seed)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", reset_all)
    with RUN.preserve_replay_rng("cuda:0"):
        # This reproduces the exact problematic constructor nesting: its CPU
        # fork alone does NOT protect global CUDA RNGs from manual_seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
        for g in generators:
            torch.rand(29, generator=g)
        torch.rand(17)
        assert any(not torch.equal(g.get_state(), s) for g, s in zip(generators, before))
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.rand(13), cpu_expected)
    for g, s, want in zip(generators, before, expected):
        assert torch.equal(g.get_state(), s)
        assert torch.equal(torch.rand(13, generator=g), want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA unavailable; independent two-device simulation still runs")
def test_actual_cuda_replay_context_keeps_continuous_rng_stream():
    # Tiny synthetic random tensors only; no model, patient, or optimizer fit.
    torch.manual_seed(519)
    device = torch.device("cuda:0")
    cpu_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(device).clone()
    expected = torch.rand(13, device=device).cpu()
    torch.cuda.set_rng_state(cuda_before, device=device)
    with RUN.preserve_replay_rng(device):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
        torch.rand(29, device=device)
        torch.rand(17)
    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert torch.equal(torch.cuda.get_rng_state(device), cuda_before)
    assert torch.equal(torch.rand(13, device=device).cpu(), expected)

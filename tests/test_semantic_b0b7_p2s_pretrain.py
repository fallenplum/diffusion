"""Synthetic continuation/gate tests; no real cohort or endpoint is accessed."""
from copy import deepcopy
from dataclasses import asdict
import itertools
from pathlib import Path

import pytest
import torch

from scripts import semantic_b0b7_p2s_pretrain as runner
from src.semantic_b0b7.p1r_models import RepairSpec


def locked_fixture(tmp_path):
    old = "src/frozen_scientific_source.py"
    for name in (old, runner.SCRIPT, runner.TEST):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen synthetic source\n", encoding="utf-8")
    sources = {name: runner.file_sha(tmp_path / name) for name in (old, runner.SCRIPT, runner.TEST)}
    cells = []
    for split, family, seed, weight in itertools.product(
            ("outer1_inner0", "outer1_inner1", "outer1_inner2"), ("MAIN", "ATTN_MATCHED"),
            (31, 47, 83), (.1, .03)):
        cells.append({"split": split, "family": family, "seed": seed, "lambda_pair": weight,
            **{key: "1" * 64 for key in ("input_sha256", "parent_receipt_sha256",
                                        "parent_binding_sha256", "parent_state400_sha256")}})
    master = {"frozen_sources": {old: sources[old]}}
    lock = {"schema": "semantic_b0b7_p2s_lock_v1", "status": "LOCKED_BEFORE_P2S_OPTIMIZER",
        "parent_master_sha256": runner.P2_SHA,
        "parent_git_commit": "dab64dff52b1170f9a11ceb0833d9d49424fa63a",
        "training": deepcopy(runner.TRAINING), "frozen_sources": sources, "cells": cells}
    return lock, master


def test_exact_36_cell_matrix_and_parent_source_closure(tmp_path):
    lock, master = locked_fixture(tmp_path)
    cells = runner.validate_diagnostic_lock(lock, master, lock_sha="2" * 64, root=tmp_path)
    assert len(cells) == 36
    assert sum(k[3] == .1 for k in cells) == 18
    assert sum(k[3] == .03 for k in cells) == 18


@pytest.mark.parametrize("mutation", ["duplicate", "outer_test", "native", "new_seed", "new_lambda",
    "missing_parent_state", "source_rebinding", "source_changed", "missing_runner", "no_prelock",
    "new_schedule", "new_parent", "new_commit"])
def test_unlocked_scope_or_source_change_is_rejected(tmp_path, mutation):
    lock, master = locked_fixture(tmp_path)
    if mutation == "duplicate":
        lock["cells"][-1] = deepcopy(lock["cells"][0])
    elif mutation == "outer_test":
        lock["cells"][0]["split"] = "outer1_final"
    elif mutation == "native":
        lock["cells"][0]["family"] = "ATTN_NATIVE"
    elif mutation == "new_seed":
        lock["cells"][0]["seed"] = 999
    elif mutation == "new_lambda":
        lock["cells"][0]["lambda_pair"] = .01
    elif mutation == "missing_parent_state":
        del lock["cells"][0]["parent_state400_sha256"]
    elif mutation == "source_rebinding":
        lock["frozen_sources"][next(iter(master["frozen_sources"]))] = "0" * 64
    elif mutation == "source_changed":
        (tmp_path / runner.SCRIPT).write_text("# changed\n", encoding="utf-8")
    elif mutation == "missing_runner":
        del lock["frozen_sources"][runner.SCRIPT]
    elif mutation == "no_prelock":
        lock["status"] = "PLANNED"
    elif mutation == "new_schedule":
        lock["training"]["maximum_epoch"] = 1600
    elif mutation == "new_parent":
        lock["parent_master_sha256"] = "0" * 64
    else:
        lock["parent_git_commit"] = "0" * 40
    with pytest.raises(ValueError):
        runner.validate_diagnostic_lock(lock, master, lock_sha="2" * 64, root=tmp_path)


def fake_adam(epoch=400):
    return {"param_groups": [{"params": [0], "lr": .0003, "betas": (.9, .999),
        "weight_decay": 0, "eps": 1e-8, "amsgrad": False}],
        "state": {0: {"step": torch.tensor(float(epoch)), "exp_avg": torch.ones(2),
                      "exp_avg_sq": torch.ones(2)}}}


def parent_snapshot():
    spec = RepairSpec(source_arm="B4", relation_mode="dual", lambda_pair=.1)
    saved = {"schema": "semantic_b0b7_p1r_state_v1", "epoch": 400,
        "input_sha256": "i" * 64, "seed": 31, "spec": asdict(spec),
        "train_rows_sha256": "t" * 64, "valid_rows_sha256": "v" * 64,
        "evaluate_valid": True, "fixed_epoch": None, "DSS_OS_PFI_accessed": False,
        "expected_replay": {"prediction": torch.zeros(2, 3), "pair_scores": torch.zeros(2, 2),
                            "pair_valid": torch.ones(2, 2, dtype=torch.bool)},
        "cpu_rng": torch.get_rng_state().clone(), "cuda_rng": torch.zeros(16, dtype=torch.uint8),
        "optimizer": fake_adam()}
    receipt = {"DSS_OS_PFI_accessed": False, "optimizer_steps": 400,
               "history": [{"epoch": i} for i in range(1, 401)]}
    return saved, receipt, spec


def validate_snapshot(saved, receipt, spec):
    return runner.validate_parent_snapshot(saved, receipt, spec=spec, input_sha256="i" * 64,
        seed=31, train_hash="t" * 64, valid_hash="v" * 64)


def test_parent400_contains_complete_adam_rng_and_scope():
    saved, receipt, spec = parent_snapshot()
    validate_snapshot(saved, receipt, spec)


@pytest.mark.parametrize("mutation", ["adam_step", "adam_missing", "adam_lr", "adam_nan",
    "cpu_rng", "cuda_rng", "input", "DSS", "history", "expected_replay", "wrong_spec"])
def test_inexact_parent_state_fails_closed(mutation):
    saved, receipt, spec = parent_snapshot()
    if mutation == "adam_step":
        saved["optimizer"]["state"][0]["step"] = torch.tensor(399.)
    elif mutation == "adam_missing":
        saved["optimizer"]["state"] = {}
    elif mutation == "adam_lr":
        saved["optimizer"]["param_groups"][0]["lr"] = .001
    elif mutation == "adam_nan":
        saved["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
    elif mutation == "cpu_rng":
        saved["cpu_rng"] = None
    elif mutation == "cuda_rng":
        saved["cuda_rng"] = None
    elif mutation == "input":
        saved["input_sha256"] = "z" * 64
    elif mutation == "DSS":
        saved["DSS_OS_PFI_accessed"] = True
    elif mutation == "history":
        receipt["history"][2]["epoch"] = 9
    elif mutation == "expected_replay":
        del saved["expected_replay"]["pair_valid"]
    else:
        saved["spec"]["lambda_pair"] = .03
    with pytest.raises(ValueError):
        validate_snapshot(saved, receipt, spec)


def assert_state_equal(first, second):
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_state_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_state_equal(a, b)
    else:
        assert first == second


def test_adam_and_dropout_continuation_is_exact_after_reconstruction():
    """Six tiny synthetic updates equal 3 + save/rebuild/restore + 3 exactly."""
    def create():
        return torch.nn.Sequential(torch.nn.Linear(5, 9), torch.nn.GELU(),
            torch.nn.Dropout(.4), torch.nn.Linear(9, 2))
    def update(model, optimizer):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 20
        loss = (model(x) - .3).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone()
    torch.manual_seed(712)
    model = create()
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    for _ in range(3):
        update(model, opt)
    saved = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(opt.state_dict()),
             "cpu_rng": torch.get_rng_state().clone(), "cuda_rng": None}
    continuous_losses = [update(model, opt) for _ in range(3)]
    continuous_rng = torch.get_rng_state().clone()
    torch.manual_seed(999)  # reconstruction is intentionally RNG destructive.
    resumed = create()
    resumed.load_state_dict(saved["model"])
    restored_opt = runner.restore_adam_rng(resumed, saved, torch.device("cpu"))
    resumed_losses = [update(resumed, restored_opt) for _ in range(3)]
    assert all(torch.equal(a, b) for a, b in zip(continuous_losses, resumed_losses))
    assert_state_equal(model.state_dict(), resumed.state_dict())
    assert_state_equal(opt.state_dict(), restored_opt.state_dict())
    assert torch.equal(torch.get_rng_state(), continuous_rng)


class ToyReader(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)
        self.dropout = torch.nn.Dropout(.25)

    def forward(self, image, genes):
        prediction = self.linear(self.dropout(image))
        return {"program_prediction": prediction,
            "pair": {"scores": prediction @ prediction.T,
                     "valid": torch.ones(len(image), len(image), dtype=torch.bool)}}


def test_actual_checkpoint_rebuild_replay_preserves_training_rng(tmp_path, monkeypatch):
    model = ToyReader()
    batch = (torch.ones(4, 4), None)
    saved = {"model": model.state_dict(), "expected_replay": runner.replay_values(model, batch)}
    path = tmp_path / "state.private.pt"
    torch.save(saved, path)
    def factory(*unused_args, saved, **unused_kwargs):
        torch.manual_seed(998)
        result = ToyReader()
        result.load_state_dict(saved["model"])
        return result
    monkeypatch.setattr(runner.phase, "make_model", factory)
    before = torch.get_rng_state().clone()
    result = runner.checked_replay({}, None, 31, torch.arange(2), torch.device("cpu"), path, batch)
    assert result["actual_replay"] is True and result["replay_preserved_training_rng"] is True
    assert torch.equal(before, torch.get_rng_state())


def test_changed_checkpoint_prediction_is_not_a_replay_pass():
    saved = {"expected_replay": {"prediction": torch.ones(2), "pair_scores": torch.ones(2),
                                 "pair_valid": torch.ones(2, dtype=torch.bool)}}
    actual = deepcopy(saved["expected_replay"])
    actual["prediction"][0] += .001
    with pytest.raises(ValueError, match="ACTUAL_REPLAY_MISMATCH"):
        runner.assert_replay(saved, actual)


def test_diagnostic_budget_and_checkpoint_scope_are_fixed():
    # 18 continuations and 18 independent new trajectories, not 36 new 800 runs.
    assert 18 * 400 + 18 * 800 == 21600
    assert runner.TRAINING["checkpoint_epochs"] == [400, 800]
    assert runner.TRAINING["use_DSS"] is False
    assert runner.TRAINING["eval_only_at_fixed_checkpoints"] is True


@pytest.mark.parametrize("threads,device", [(1, "cuda"), (4, "cuda"), (2, "cpu")])
def test_runtime_rejects_changed_formal_execution_before_cuda(threads, device, monkeypatch):
    monkeypatch.setattr(runner, "STARTUP_THREAD_ENVIRONMENT", {
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    with pytest.raises(ValueError, match="FORMAL_GPU_THREADS_CHANGED"):
        runner.runtime_guard(threads, device)


def test_postimport_thread_environment_cannot_fake_formal_startup(monkeypatch):
    monkeypatch.setattr(runner, "STARTUP_THREAD_ENVIRONMENT", {
        "OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None, "OPENBLAS_NUM_THREADS": None})
    for key in runner.STARTUP_THREAD_ENVIRONMENT:
        monkeypatch.setenv(key, "2")
    with pytest.raises(ValueError, match="BEFORE_NUMPY_IMPORT"):
        runner.runtime_guard(2, "cuda")

"""Actual tiny synthetic optimizer/state replay checks, not real source fits."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_spgel_contracts import source
from src.spgel.contracts import SourceStatistics, sha256
from src.spgel import training


def synthetic_output(monkeypatch, tmp_path):
    # Only synthetic fixture tensors are persisted under pytest's workspace temp.
    # Real private-path rejection is checked separately, without this override.
    monkeypatch.setattr(training, "private_directory", lambda p: Path(p))
    return tmp_path


def config():
    return training.read_json(Path(__file__).parents[1] / "configs/spgel/pilot_v1.json")


def test_tissue_only_convex_solution_has_real_stationary_gradient():
    data = source(); train, held = data.split(0)
    stats = SourceStatistics.fit(data, train, heldout_donors=[0])
    fit = training.fit_tissue_classifier(data, train, stats, config(), 31)
    assert fit["gradient_infinity_norm"] <= 1e-6
    output, count = training.collect_images(None, data, held, allowed_rows=held, classifier=fit)
    assert count == 0 and output["program_predictions"].shape == (len(held), 4)
    # Changing heldout truth cannot change the loaded image-only classifier prediction.
    other = copy.deepcopy(data); other.tissue[held] = (other.tissue[held] + 1) % 3
    again, _ = training.collect_images(None, other, held, allowed_rows=held, classifier=fit)
    assert np.array_equal(output["program_predictions"], again["program_predictions"])


def test_actual_small_40_epoch_fit_and_three_saved_states_replay(monkeypatch, tmp_path):
    data = source(); recipe = config()
    monkeypatch.setattr(training, "validate_lock", lambda *a: {"contract_sha256": "f" * 64})
    config_path = Path(__file__).parents[1] / "configs/spgel/pilot_v1.json"
    p = synthetic_output(monkeypatch, tmp_path)
    if True:
        result = training.fit_cell(data, recipe, config_path, p, arm="S_A5", donor=0, seed=31, device="cpu")
        assert result["optimizer_updates"] == 40 and result["epochs"] == 40
        assert [x["label"] for x in result["snapshots"]] == ["epoch10", "epoch20", "epoch40"]
        assert result["snapshots"][0]["heldout"]["task_B_common"]["status"] == "NOT_EVALUATED_FOR_TASKB"
        assert result["snapshots"][-1]["heldout"]["task_B_common"]["status"] == "COMPLETE"
        assert result["snapshots"][-1]["heldout"]["task_B_native"]["status"] == "COMPLETE"
        replay = training.replay_cell(data, recipe, config_path, p, arm="S_A5", donor=0, seed=31, device="cpu")
        assert replay["status"] == "PASS" and len(replay["snapshots"]) == 3
        assert replay["actual_saved_state_loaded"] and replay["new_optimizer_fits"] == 0
        assert max(x["maximum_prediction_error"] for x in replay["snapshots"]) == 0
        directory = p / "pilot/cells/S_A5_d0_seed31"
        checkpoint = directory / "checkpoint_epoch20.pt"
        state = torch.load(checkpoint, weights_only=False)
        state["state_dict"]["slot_bias"] += 1.
        torch.save(state, checkpoint)
        metadata = training.read_json(directory / "COMPLETE.json")
        metadata["snapshots"][1]["checkpoint_sha256"] = sha256(checkpoint)
        (directory / "COMPLETE.json").write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(ValueError, match="SOURCE_REPLAY_PREDICTION_TOLERANCE_FAILED"):
            training.replay_cell(data, recipe, config_path, p, arm="S_A5", donor=0, seed=31, device="cpu")


def test_actual_convex_fit_replay_is_one_solver_state_not_fake_epochs(monkeypatch, tmp_path):
    data = source(); recipe = config()
    monkeypatch.setattr(training, "validate_lock", lambda *a: {"contract_sha256": "e" * 64})
    config_path = Path(__file__).parents[1] / "configs/spgel/pilot_v1.json"
    p = synthetic_output(monkeypatch, tmp_path)
    if True:
        fit = training.fit_cell(data, recipe, config_path, p, arm="S_A0", donor=0, seed=31, device="cpu")
        assert fit["optimizer_updates"] == 0 and fit["convex_optimizer_fits"] == 1
        assert fit["neural_optimizer_fits"] == 0 and fit["epochs"] is None
        assert [x["label"] for x in fit["snapshots"]] == ["solver_final"]
        replay = training.replay_cell(data, recipe, config_path, p, arm="S_A0", donor=0, seed=31)
        assert replay["status"] == "PASS" and len(replay["snapshots"]) == 1
        with pytest.raises(ValueError, match="EXISTING_CELL_NO_AUTOMATIC_RETRY"):
            training.fit_cell(data, recipe, config_path, p, arm="S_A0", donor=0, seed=31)


def test_full_candidate_retrieval_is_not_filtered_by_heldout_tissue_or_depth():
    data = source(); train, held = data.split(0)
    stats = SourceStatistics.fit(data, train, heldout_donors=[0])
    prediction = stats.targets_z(data, held)
    out = dict(program_predictions=prediction)
    a, matrices_a, _ = training.evaluate_retrieval(None, out, data, held, stats, native=False)
    other = copy.deepcopy(data); other.tissue[held] = 0; other.library[held] *= 999
    b, matrices_b, _ = training.evaluate_retrieval(None, out, other, held, stats, native=False)
    assert a == b and all(np.array_equal(matrices_a[k], matrices_b[k]) for k in matrices_a)
    assert all(x["candidates_count"] == 3 for x in a["per_section"])


def test_exact_full_program_macro_requires_all_four_defined():
    mean, sd = np.zeros(4), np.ones(4)
    target = np.tile(np.arange(12)[:, None], (1, 4)).astype(float)
    prediction = target.copy(); prediction[:, 0] = 1.
    metrics = training.image_prediction_metrics(prediction, target, np.ones_like(target, bool), mean, sd)
    assert metrics["defined_program_count"] == 3 and metrics["primary_donor_four_program_Pearson"] is None


def test_recipe_rejects_hidden_loss_or_heldout_epoch_selection():
    recipe = config(); training.validate_recipe(recipe)
    bad = copy.deepcopy(recipe); bad["loss"]["masked_gene_weight"] = .1
    with pytest.raises(ValueError, match="PILOT_LOSS_PACKAGE_CHANGED"):
        training.validate_recipe(bad)
    bad = copy.deepcopy(recipe); bad["training"]["model_or_epoch_selection_from_heldout"] = True
    with pytest.raises(ValueError, match="TRAINING_BUDGET_OR_OPTIMIZER_CHANGED"):
        training.validate_recipe(bad)


def test_real_private_outputs_cannot_be_created_inside_science_workspace():
    with pytest.raises(ValueError, match="SOURCE_PRIVATE_OUTPUT_INSIDE_GIT_WORKSPACE"):
        training.private_directory(training.ROOT / "tmp" / "forbidden_real_private_outputs")

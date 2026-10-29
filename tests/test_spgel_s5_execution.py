"""S5 synthetic execution/loss attribution; never real donor training.

Two tiny fixture fits exercise the actual 40-update optimizer and six saved-state
replays. Only the provenance lock is mocked for synthetic inputs. Temp fixtures
must live in the explicitly selected workspace test directory, not C: temp.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_spgel_contracts import source as synthetic_source
from src.spgel.contracts import GeneBatch, SourceStatistics, array_hash, sha256
from src.spgel import training as inherited
from src.spgel_s5 import execution
from src.spgel_s5.losses import objective, paired_loss
from src.spgel_s5.sampling import make_wrong_pairs


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/spgel_s5/mechanism_v1.json"
torch.set_num_threads(2)


class PairRecorder:
    """Index/CE probe, not a research model or learned performance baseline."""
    def __init__(self, arm):
        self.arm = arm
        self.calls = []

    def gene_encoder(self, genes):
        return {"probe": genes.values}

    def paired_scores(self, outputs, genes, ir, gr, **kwargs):
        self.calls.append((ir.detach().clone(), gr.detach().clone()))
        scores = -(ir.to(torch.float32) - gr.to(torch.float32)).square() / 10
        return dict(scores=scores, valid=torch.ones_like(ir, dtype=torch.bool))


def loss_fixture():
    b = 4
    output = dict(program_predictions=torch.arange(b*4, dtype=torch.float32).reshape(b, 4)/10,
                  tissue_logits=torch.arange(b*3, dtype=torch.float32).reshape(b, 3)/10)
    genes = GeneBatch(torch.arange(8*6, dtype=torch.float32).reshape(8, 6),
                      torch.ones((8, 6), dtype=torch.bool), torch.arange(6))
    negative = torch.full((b, 8), -1, dtype=torch.long)
    negative[:, 0] = torch.arange(b).roll(1)
    negative[:, 1] = torch.arange(b).roll(2)
    return output, genes, negative


@pytest.mark.parametrize("wrong", (False, True))
def test_both_anchor_losses_use_correct_positive_bank_and_original_negative_bank(wrong):
    output, genes, negative = loss_fixture()
    model = PairRecorder("M5" if wrong else "M2")
    loss, audit = paired_loss(model, output, genes, negative, negative,
                              torch.ones(4, dtype=torch.bool), wrong=wrong)
    assert torch.isfinite(loss) and len(model.calls) == 2
    own = torch.arange(4)
    ir, gr = model.calls[0]
    ri, rg = model.calls[1]
    expected = own + (4 if wrong else 0)
    assert torch.equal(ir[:, 0], own) and torch.equal(gr[:, 0], expected)
    assert torch.equal(ri[:, 0], own) and torch.equal(rg[:, 0], expected)
    assert torch.equal(gr[:, 1:3], negative[:, :2])
    assert torch.equal(ri[:, 1:3], negative[:, :2])
    assert (gr[:, 1:] < 4).all() and (ri[:, 1:] < 4).all()
    assert audit["active_image_anchors"] == audit["active_gene_anchors"] == 4
    assert audit["two_anchor_losses_retained"] and audit["wrong_positive_gene_bank"] == wrong


def test_wrong_pairing_changes_only_itc_not_main_image_or_tissue_targets():
    output, genes, negative = loss_fixture()
    target = torch.flip(output["program_predictions"], dims=(0,))
    mask = torch.ones_like(target, dtype=torch.bool)
    labels, weights = torch.tensor([0, 1, 2, 1]), torch.ones(3)
    active = torch.ones(4, dtype=torch.bool)
    before = target.clone(), labels.clone(), genes.values.clone()
    _, matched = objective(PairRecorder("M2"), output, target, mask, labels,
                            weights, genes, negative, negative, active)
    _, wrong = objective(PairRecorder("M5"), output, target, mask, labels,
                          weights, genes, negative, negative, active)
    assert torch.equal(matched["prediction_Huber"], wrong["prediction_Huber"])
    assert torch.equal(matched["tissue_CE"], wrong["tissue_CE"])
    assert not torch.equal(matched["ITC"], wrong["ITC"])
    assert torch.equal(target, before[0]) and torch.equal(labels, before[1])
    assert torch.equal(genes.values, before[2])
    assert wrong["regression_targets_mismatched"] is False


def test_unsupported_anchors_do_not_become_fallback_self_positives():
    output, genes, negative = loss_fixture()
    output["program_predictions"].requires_grad_(True)
    loss, audit = paired_loss(PairRecorder("M5"), output, genes,
        negative, negative, torch.zeros(4, dtype=torch.bool), wrong=True)
    assert loss.item() == 0
    assert audit["active_image_anchors"] == audit["active_gene_anchors"] == 0
    loss.backward()
    assert output["program_predictions"].grad.count_nonzero() == 0


def test_missing_second_bank_fails_closed():
    output, genes, negative = loss_fixture()
    smaller = GeneBatch(genes.values[:4], genes.observed[:4], genes.gene_ids)
    with pytest.raises(ValueError, match="ORIGINAL_PLUS_WRONG_GENE_BANK"):
        paired_loss(PairRecorder("M2"), output, smaller, negative, negative,
                    torch.ones(4, dtype=torch.bool))


def test_budget_is_84_real_fits_72800_minibatch_updates_and_252_unique_states():
    config = execution.recipe(CONFIG)
    old = inherited.read_json(ROOT/"reports/spgel/SOURCE_RUN_LOCK.json")
    updates = sum(40 * int(np.ceil(x["train_rows"]/512)) for x in old["measured_budget"]["folds"])
    assert 7*6*2 == config["budget"]["new_neural_fits"] == 84
    assert updates*7*2 == config["budget"]["optimizer_updates"] == 72800
    assert 84*len(config["training"]["save_epochs"]) == config["budget"]["saved_state_replays"] == 252
    bad = deepcopy(config);bad["loss"]["bidirectional_ITC"] = .2
    # Recipe/source hashes alone must not claim a different effective loss than
    # the .1 coefficient actually used by objective().
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution, "read_json", lambda p: bad)
        with pytest.raises(ValueError, match="LOSS|loss"):
            execution.recipe(CONFIG)


def synthetic_contract(monkeypatch, tmp_path, donor=0, seed=31):
    data = synthetic_source()
    train, held = data.split(donor)
    stats = SourceStatistics.fit(data, train, heldout_donors=[donor])
    config = inherited.read_json(CONFIG)
    # This override concerns only synthetic fixture output eligibility. It does
    # not make production lock creation permit in-repository private outputs.
    fake_root = tmp_path/"synthetic_science_root"
    monkeypatch.setattr(inherited, "ROOT", fake_root)
    private = tmp_path/"synthetic_private"
    assert not private.is_relative_to(fake_root)
    (private/"qualification").mkdir(parents=True)
    wrong = make_wrong_pairs(data, train, stats, seed)
    assert wrong["active"][train].any()
    path = execution._mapping_path(private, donor, seed)
    with path.open("xb") as stream:
        np.savez_compressed(stream, partner_global_rows=wrong["partner_global_rows"],
            active=wrong["active"], up=data.up, down=data.down,
            gene_permutation=np.arange(len(data.gene_symbols)))
    partition = dict(donor=donor, seed=seed, train_rows_sha256=array_hash(train),
        heldout_rows_sha256=array_hash(held), mapping_sha256=sha256(path), updates_per_fit=40)
    lock = dict(contract_sha256="f"*64, partitions=[partition])
    monkeypatch.setattr(execution, "validate_lock", lambda *a: (lock, config))
    return data, stats, private, lock


def test_batch_builds_identical_original_plus_wrong_input_bank_for_every_arm(monkeypatch, tmp_path):
    data, stats, private, lock = synthetic_contract(monkeypatch, tmp_path)
    train, _ = data.split(0)
    maps, _ = execution.mapping(private, lock, 0, 31)
    genes, ni, ng, active, audit = execution._batch(data, stats, train, maps, 31, "cpu")
    expected = stats.genes_z(data, np.concatenate((train, maps["partner_global_rows"][train])))
    assert len(genes.values) == 2*len(train)
    assert torch.equal(genes.values, expected.values)
    assert torch.equal(genes.observed, expected.observed)
    assert ni.shape == ng.shape == (len(train), 8) and torch.equal(ni, ng)
    assert active.dtype == torch.bool and active.any()
    assert ((ni >= -1) & (ni < len(train))).all()
    assert audit["separate_directional_CE_required"]


def test_M6_overrides_only_model_membership_and_keeps_real_source_truth(monkeypatch, tmp_path):
    data, stats, _, _ = synthetic_contract(monkeypatch, tmp_path)
    target, up, down = data.targets.copy(), data.up.copy(), data.down.copy()
    permutation = np.roll(np.arange(len(data.gene_symbols)), 1)
    maps = dict(up=data.up[:, permutation].copy(), down=data.down[:, permutation].copy())
    model = execution._model(data, stats, "M6", 31, maps, "cpu")
    assert np.array_equal(model.gene_encoder.up.numpy(), maps["up"])
    assert np.array_equal(model.gene_encoder.down.numpy(), maps["down"])
    assert np.array_equal(data.targets, target)
    assert np.array_equal(data.up, up) and np.array_equal(data.down, down)


@pytest.mark.parametrize("arm", ("M2", "M5"))
def test_two_actual_synthetic40_fits_six_saved_states_replay_and_tamper_rejected(
        monkeypatch, tmp_path, arm):
    data, _, private, _ = synthetic_contract(monkeypatch, tmp_path)
    done = execution.fit_cell(data, CONFIG, private, arm=arm, donor=0, seed=31)
    assert done["actual_neural_fits"] == 1 and done["actual_optimizer_updates"] == 40
    assert done["actual_saved_states"] == 3
    assert [x["epoch"] for x in done["snapshots"]] == [10, 20, 40]
    assert done["wrong_pairing_changes_main_targets"] is False
    assert done["image_only_inference_uses_RNA"] is False
    assert done["snapshots"][0]["heldout"]["task_B_native"]["status"] == "NOT_EVALUATED_FOR_TASKB"
    assert done["snapshots"][-1]["heldout"]["task_B_native"]["status"] == "COMPLETE"
    actual = execution.replay_cell(data, CONFIG, private, arm=arm, donor=0, seed=31)
    assert actual["status"] == "PASS" and len(actual["states"]) == 3
    assert [x["epoch"] for x in actual["states"]] == [10, 20, 40]
    assert all(x["actual_saved_state_loaded"] for x in actual["states"])
    assert all(x["actual_image_and_native_pair_forward_replayed"] for x in actual["states"])
    assert max(x["maximum_prediction_error"] for x in actual["states"]) == 0
    assert actual["new_optimizer_fits"] == actual["new_optimizer_updates"] == 0
    with pytest.raises(ValueError, match="EXISTING_CELL_NO_AUTOMATIC_RETRY"):
        execution.fit_cell(data, CONFIG, private, arm=arm, donor=0, seed=31)
    folder = private/"cells"/execution.unit_name(arm, 0, 31)
    complete = folder/"COMPLETE.json"
    original = complete.read_bytes()
    # Hash-only agreement must not allow replaying epoch10 three times and
    # reporting three distinct mandatory states.
    altered = deepcopy(done);altered["snapshots"] = [deepcopy(done["snapshots"][0]) for _ in range(3)]
    complete.write_text(__import__("json").dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="S5_"):
        execution.replay_cell(data, CONFIG, private, arm=arm, donor=0, seed=31)
    complete.write_bytes(original)
    checkpoint = folder/"checkpoint_epoch20.pt"
    stored = torch.load(checkpoint, weights_only=False)
    stored["state_dict"]["slot_bias"] += 1.
    torch.save(stored, checkpoint)
    altered = inherited.read_json(complete)
    altered["snapshots"][1]["checkpoint_sha256"] = sha256(checkpoint)
    complete.write_text(__import__("json").dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="ACTUAL_STATE_REPLAY_FAILED"):
        execution.replay_cell(data, CONFIG, private, arm=arm, donor=0, seed=31)


def test_unmocked_private_path_guard_rejects_science_workspace():
    with pytest.raises(ValueError, match="SOURCE_PRIVATE_OUTPUT_INSIDE_GIT_WORKSPACE"):
        inherited.private_directory(inherited.ROOT/"tmp/forbidden_real_S5_private_outputs")

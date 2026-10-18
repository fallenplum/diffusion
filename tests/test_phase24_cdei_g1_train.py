"""Synthetic joint-training qualification; no cohort data or persistent fits."""
import copy
import inspect

import numpy as np
import pytest
import torch

from src.phase23_cdei.model import module_fingerprint
from src.phase23_cdei.warmup import _seed
from src.phase24_cdei_g1 import train
from src.phase24_cdei_g1.model import G1Model


class _Synthetic:
    """35 fit patients force a second checkpoint chunk; one has two WSIs."""
    def __init__(self, nfit=35, clinical_dim=33):
        generator = torch.Generator().manual_seed(240108)
        patients = nfit + 4
        self.fit = np.arange(nfit, dtype=np.int64)[::-1].copy()
        self.valid = np.arange(nfit, patients, dtype=np.int64)
        self.owner = np.concatenate((np.arange(patients), np.array([0]))).astype(np.int64)
        self.visual = torch.randn(len(self.owner), 3, 1536, generator=generator)
        self.mask = torch.ones(len(self.owner), 3, dtype=torch.bool)
        self.mask[:, -1] = False
        self.visual[:, -1] = float("nan")
        member = torch.zeros(8, 512, dtype=torch.bool)
        for k in range(8):
            member[k, 64*k:64*(k+1)] = True
        self.ft = {
            "membership": member,
            "C": torch.randn(patients, clinical_dim, generator=generator),
            "R": torch.randn(patients, 512, generator=generator),
            "G": torch.rand(patients, 512, generator=generator),
            "L": torch.rand(patients, 512, generator=generator),
            "RM": torch.ones(patients, 512, dtype=torch.bool),
            "CM": torch.ones(patients, 512, dtype=torch.bool),
            "ANCHOR": torch.linspace(-1., 1., patients),
        }
        self.anchor_sd_fit = float(self.ft["ANCHOR"][self.fit].double().std(unbiased=False))
        self.wrong_o, self.wrong_c = np.arange(patients), np.arange(patients)
        for part in (self.fit, self.valid):
            self.wrong_o[part] = np.roll(part, 1)
            self.wrong_c[part] = np.roll(part, -1)
        self._times = np.arange(patients, dtype=np.float64) % 9 + 1.
        self._events = (np.arange(patients) % 3 != 0).astype(float)
        self.label_calls = 0

    @property
    def ep(self):
        raise AssertionError("direct endpoint access is forbidden")

    def selection_labels(self):
        raise AssertionError("training cannot select using heldout endpoints")

    def fit_labels(self):
        self.label_calls += 1
        return {"time": self._times[self.fit].copy(), "event": self._events[self.fit].copy()}


@pytest.fixture(scope="module")
def trained_pairs():
    results = {}
    for arm in train.ARMS:
        ordinary, poisoned = _Synthetic(), _Synthetic()
        poisoned._times[poisoned.valid] = [1.e6, 2.e6, 3.e6, 4.e6]
        poisoned._events[poisoned.valid] = 1. - poisoned._events[poisoned.valid]
        anchors = ordinary.ft["ANCHOR"].clone(), poisoned.ft["ANCHOR"].clone()
        callback = []
        one = train.train_arm(ordinary, arm, 31, 2, epochs=2,
                              step_callback=lambda *value: callback.append(value))
        two = train.train_arm(poisoned, arm, 31, 2, epochs=2)
        results[arm] = ordinary, poisoned, one, two, anchors, callback
    return results


@pytest.mark.parametrize("arm", train.ARMS)
def test_joint_training_uses_fit_labels_once_and_preserves_all_anchors(trained_pairs, arm):
    ds, other, (model, audit), (_, second), anchors, callback = trained_pairs[arm]
    assert ds.label_calls == other.label_calls == 1
    assert torch.equal(ds.ft["ANCHOR"], anchors[0])
    assert torch.equal(other.ft["ANCHOR"], anchors[1])
    assert callback == [(arm, 1), (arm, 2)]
    assert not model.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    for record in (audit, second):
        assert record["actual_epochs"] == record["optimizer_steps"] == 2
        assert record["fixed_final_epoch_no_validation_selection"] is True
        assert record["heldout_endpoint_reads"] == 0
        assert record["anchor_unchanged"] is True
        assert record["reference_reencoded_each_forward"] is True
        assert record["reference_gradient_detached"] is False
        assert record["initial_backbone_sha256"] != record["final_backbone_sha256"]
        for row in record["history"]:
            assert np.isfinite(row["training_loss"]) and np.isfinite(row["fit_loss"])
            assert np.isfinite(row["grad_norm"]) and row["grad_norm"] > 0
            for group in ("kernel", "clinical_nonintercept", "RNA", "CNA_gain", "CNA_loss",
                          "relative_head", "output_head"):
                assert np.isfinite(row["gradients"][group]) and row["gradients"][group] > 0, group
            assert row["gradients"]["extra_head"] > 0 if arm != "REL" else row["gradients"]["extra_head"] == 0


@pytest.mark.parametrize("arm", train.ARMS)
def test_heldout_endpoint_poison_cannot_change_any_parameter_or_prediction(trained_pairs, arm):
    ds, other, (left, la), (right, ra), _, _ = trained_pairs[arm]
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], atol=0, rtol=0)
    assert module_fingerprint(left) == module_fingerprint(right)
    for lrow, rrow in zip(la["history"], ra["history"]):
        for key in lrow:
            if key != "epoch_seconds":
                assert lrow[key] == rrow[key]
    lp, rp = train.predict_arm(left, ds), train.predict_arm(right, other)
    np.testing.assert_array_equal(lp["fit_risk"], rp["fit_risk"])
    np.testing.assert_array_equal(lp["risk"], rp["risk"])
    if arm == "LOCAL":
        assert set(lp["sensitivities"]) == {"OFF", "NO_C", "WRONG_C", "WRONG_O"}
        for key in lp["sensitivities"]:
            np.testing.assert_array_equal(lp["sensitivities"][key], rp["sensitivities"][key])
    assert ds.label_calls == other.label_calls == 1, "prediction must not read endpoints"


def test_actual_training_receipts_share_common_initialization(trained_pairs):
    fingerprints = {entry[2][1]["initial_backbone_sha256"] for entry in trained_pairs.values()}
    assert len(fingerprints) == 1
    ds = _Synthetic()
    models = []
    for arm in train.ARMS:
        _seed(31, 2)
        models.append(G1Model(33, ds.ft["membership"], arm=arm))
    for model in models[1:]:
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
    local, ctrl = models[2:]
    assert sum(p.numel() for p in local.parameters()) == sum(p.numel() for p in ctrl.parameters())
    for key, value in local.state_dict().items():
        torch.testing.assert_close(value, ctrl.state_dict()[key], atol=0, rtol=0)


def test_complete_efron_risk_set_and_fit_order_are_preserved(monkeypatch):
    ds = _Synthetic()
    sizes, actual = [], train.full_risk_efron

    def checked(risk, times, events):
        sizes.append(len(risk))
        assert len(risk) == len(ds.fit) == 35
        np.testing.assert_array_equal(times, ds._times[ds.fit])
        np.testing.assert_array_equal(events, ds._events[ds.fit])
        return actual(risk, times, events)

    monkeypatch.setattr(train, "full_risk_efron", checked)
    train.train_arm(ds, "LOCAL", 47, 1, epochs=2)
    assert sizes == [35] * 5  # Anchor once; training/eval each epoch.
    assert ds.label_calls == 1


@pytest.mark.parametrize("arm", train.ARMS)
def test_checkpoint_backward_matches_direct_forward_across_owner_chunks(monkeypatch, arm):
    ds = _Synthetic()
    _seed(83, 3)
    checkpointed = G1Model(33, ds.ft["membership"], arm=arm).train()
    direct = copy.deepcopy(checkpointed).train()
    observed_chunks, actual_checkpoint = [], train.checkpoint

    def tracked(fn, *args, **kwargs):
        assert kwargs == {"use_reentrant": False, "preserve_rng_state": True}
        observed_chunks.append(np.array(fn.__kwdefaults__["owner_rows"], copy=True))
        return actual_checkpoint(fn, *args, **kwargs)

    monkeypatch.setattr(train, "checkpoint", tracked)
    torch.manual_seed(900)
    crisk, _, _ = train._forward(checkpointed, ds, ds.fit, train.reference_raw(ds), training=True)
    closs = train.full_risk_efron(crisk, ds._times[ds.fit], ds._events[ds.fit])
    closs.backward()
    assert len(observed_chunks) == 2
    assert len(observed_chunks[0]) != len(observed_chunks[1])
    assert 0 not in observed_chunks[0] and np.count_nonzero(observed_chunks[1] == 0) == 2

    monkeypatch.setattr(train, "checkpoint", lambda fn, *args, **kwargs: fn(*args))
    torch.manual_seed(900)
    drisk, _, _ = train._forward(direct, ds, ds.fit, train.reference_raw(ds), training=True)
    dloss = train.full_risk_efron(drisk, ds._times[ds.fit], ds._events[ds.fit])
    dloss.backward()
    torch.testing.assert_close(crisk, drisk, atol=0, rtol=0)
    torch.testing.assert_close(closs, dloss, atol=0, rtol=0)
    for (name, cp), (other_name, dp) in zip(checkpointed.named_parameters(), direct.named_parameters()):
        assert name == other_name
        assert (cp.grad is None) == (dp.grad is None)
        if cp.grad is not None:
            torch.testing.assert_close(cp.grad, dp.grad, atol=2e-7, rtol=2e-5, msg=name)


def test_equal_wsi_pooling_precedes_single_anchor_addition():
    class Toy(torch.nn.Module):
        arm = "REL"

        def forward(self, visual, *unused, **kwargs):
            return {"risk_delta_unit": visual[:, 0, 0]}

    ds = _Synthetic(nfit=2)
    ds.visual.zero_()
    ds.visual[0, 0, 0], ds.visual[1, 0, 0], ds.visual[-1, 0, 0] = .2, .6, .8
    ds.anchor_sd_fit = 2.
    fit = np.array([0, 1])
    risk, correction, _ = train._forward(Toy(), ds, fit, train.reference_raw(ds))
    expected = torch.tensor([.5*(float(ds.visual[0, 0, 0])+float(ds.visual[-1, 0, 0])),
                             float(ds.visual[1, 0, 0])], dtype=torch.float64)*2
    torch.testing.assert_close(correction, expected, atol=0, rtol=0)
    torch.testing.assert_close(risk, ds.ft["ANCHOR"][fit].double()+expected, atol=0, rtol=0)
    assert ds.label_calls == 0


@pytest.mark.parametrize("clinical_dim", [33, 41])
def test_fit_local_clinical_width_is_preserved(clinical_dim):
    ds = _Synthetic(nfit=6, clinical_dim=clinical_dim)
    model, audit = train.train_arm(ds, "LOCAL", 31, 1, epochs=1)
    assert model.backbone.clinical_dim == clinical_dim
    assert model.backbone.compatibility.c.weight.shape == (4, clinical_dim+1)
    assert audit["history"][0]["gradients"]["clinical_nonintercept"] > 0
    assert ds.label_calls == 1


def test_training_function_has_no_endpoint_selection_or_persistent_write():
    assert inspect.signature(train.train_arm).parameters["epochs"].default == 80
    source = inspect.getsource(train.train_arm)
    assert "selection_labels" not in source and "ds.ep" not in source
    assert "torch.save" not in source and "np.save" not in source
    assert "best_epoch" not in source and "best_state" not in source

"""Synthetic fit-only warmup verification; no real patient data or saved fits."""
import inspect

import numpy as np
import pytest
import torch

from src.phase23_cdei import warmup
from src.phase23_cdei.model import module_fingerprint


class _Synthetic:
    def __init__(self, nfit=35, clinical_dim=41):
        generator = torch.Generator().manual_seed(108)
        patients = nfit + 3
        self.fit = np.arange(nfit, dtype=np.int64)[::-1].copy()
        self.valid = np.arange(nfit, patients, dtype=np.int64)
        # Patient zero has two WSIs, so equality of per-patient risk-set rows
        # cannot be mistaken for simply treating every WSI as a patient.
        self.owner = np.concatenate([np.arange(patients, dtype=np.int64), np.array([0])])
        self.visual = torch.randn((len(self.owner), 3, 1536), generator=generator)
        self.mask = torch.ones((len(self.owner), 3), dtype=torch.bool)
        self.mask[:, -1] = False
        self.visual[:, -1] = float("nan")
        member = torch.zeros((8, 512), dtype=torch.bool)
        for k in range(8):
            member[k, k * 64:(k + 1) * 64] = True
        self.ft = {"membership": member,
                   "C": torch.randn((patients, clinical_dim), generator=generator),
                   "R": torch.randn((patients, 512), generator=generator),
                   "G": torch.randn((patients, 512), generator=generator).relu(),
                   "L": torch.randn((patients, 512), generator=generator).relu(),
                   "RM": torch.ones((patients, 512), dtype=torch.bool),
                   "CM": torch.ones((patients, 512), dtype=torch.bool),
                   "ANCHOR": torch.linspace(-1., 1., patients)}
        self.anchor_sd_fit = float(self.ft["ANCHOR"][self.fit].double().std(unbiased=False))
        self._times = np.arange(patients, dtype=np.float64) % 9 + 1.
        self._events = (np.arange(patients) % 3 != 0).astype(float)
        self.label_calls = 0

    @property
    def ep(self):
        raise AssertionError("direct endpoint access is forbidden")

    def fit_labels(self):
        self.label_calls += 1
        return {"time": self._times[self.fit].copy(), "event": self._events[self.fit].copy()}


def _assert_matching_fits(left, right):
    lb, lh, la = left
    rb, rh, ra = right
    assert module_fingerprint(lb) == module_fingerprint(rb)
    assert lh.keys() == rh.keys()
    assert all(torch.equal(lh[key], rh[key]) for key in lh)
    for one, two in zip(la["history"], ra["history"]):
        for key in one:
            if key != "epoch_seconds":
                assert one[key] == two[key]


def test_fixed_complete_patient_loss_gradients_anchor_and_frozen_return(monkeypatch):
    ds = _Synthetic()
    anchor_before = ds.ft["ANCHOR"].clone()
    calls, loss_sizes, checkpoints = [], [], []
    original_loss = warmup.full_risk_efron
    original_checkpoint = warmup.checkpoint

    def checked_loss(risk, times, events):
        loss_sizes.append(len(risk))
        assert len(risk) == len(ds.fit) == len(times) == len(events)
        np.testing.assert_array_equal(times, ds._times[ds.fit])
        np.testing.assert_array_equal(events, ds._events[ds.fit])
        return original_loss(risk, times, events)

    def checked_checkpoint(fn, *args, **kwargs):
        checkpoints.append(kwargs)
        assert kwargs == {"use_reentrant": False, "preserve_rng_state": True}
        return original_checkpoint(fn, *args, **kwargs)

    monkeypatch.setattr(warmup, "full_risk_efron", checked_loss)
    monkeypatch.setattr(warmup, "checkpoint", checked_checkpoint)
    backbone, head, audit = warmup.train_kernel(ds, 31, 1, epochs=2,
                                               step_callback=lambda *x: calls.append(x))
    assert ds.label_calls == 1
    assert loss_sizes == [35] * 5
    assert len(checkpoints) == 4  # Two patient chunks, two optimizer steps.
    assert calls == [("kernel_warmup", 1), ("kernel_warmup", 2)]
    assert torch.equal(ds.ft["ANCHOR"], anchor_before)
    assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
    assert all(p.grad is None for p in backbone.parameters())
    assert all(not value.requires_grad and value.device.type == "cpu" for value in head.values())
    assert audit["optimizer_steps"] == audit["actual_epochs"] == 2
    assert audit["requested_epochs"] == 2 and audit["budget_mode"] == "QUALIFICATION_ONLY_NOT_FORMAL"
    assert audit["main_effect_trainable_parameters"] == 0
    assert audit["full_risk_calls"] == 5 and audit["heldout_endpoint_reads"] == 0
    assert audit["fit_WSIs"] == 36
    for row in audit["history"]:
        assert np.isfinite(row["training_loss"]) and np.isfinite(row["fit_loss"])
        assert np.isfinite(row["grad_norm"]) and row["grad_norm"] > 0
        for group in ("q", "k", "path_factor", "omics_factor", "clinical_nonintercept",
                      "molecular", "RNA_filters", "CNA_gain_filters", "CNA_loss_filters"):
            assert np.isfinite(row["gradients"][group]) and row["gradients"][group] > 0


def test_same_seed_reproducible_and_changed_heldout_endpoints_cannot_change_updates():
    ds1 = _Synthetic(nfit=8)
    ds2 = _Synthetic(nfit=8)
    ds3 = _Synthetic(nfit=8)
    ds3._times[ds3.valid] = [100000., 200000., 300000.]
    ds3._events[ds3.valid] = 1. - ds3._events[ds3.valid]
    one = warmup.train_kernel(ds1, 47, 3, epochs=2)
    two = warmup.train_kernel(ds2, 47, 3, epochs=2)
    mutated = warmup.train_kernel(ds3, 47, 3, epochs=2)
    _assert_matching_fits(one, two)
    _assert_matching_fits(one, mutated)
    assert ds1.label_calls == ds2.label_calls == ds3.label_calls == 1


def test_equal_wsi_pooling_occurs_before_exact_anchor_addition():
    class ToyBackbone(torch.nn.Module):
        def forward(self, visual, *unused):
            return visual[:, 0, :1], {}

    class ToyHead(torch.nn.Module):
        def forward(self, relative):
            return relative

    ds = _Synthetic(nfit=2)
    # Patient 0 has WSI responses tanh(0), tanh(2); patient 1 tanh(1).
    ds.visual.zero_()
    ds.visual[0, 0, 0], ds.visual[1, 0, 0], ds.visual[-1, 0, 0] = 0., 1., 2.
    fit = np.array([0, 1])
    anchor = torch.tensor([.3, -.4], dtype=torch.float64)
    risk, correction = warmup._fit_forward(ToyBackbone(), ToyHead(), ds, fit, ds.owner,
                                           anchor, 2., training=False)
    expected = torch.tensor([torch.tanh(torch.tensor(2.)).item(),
                             2 * torch.tanh(torch.tensor(1.)).item()], dtype=torch.float64)
    torch.testing.assert_close(correction, expected)
    torch.testing.assert_close(risk, anchor + expected)


def test_default_budget_and_source_have_no_endpoint_or_selection_path():
    signature = inspect.signature(warmup.train_kernel)
    assert signature.parameters["epochs"].default == 80
    source = inspect.getsource(warmup)
    assert "ds.ep" not in source and "ds.valid" not in source
    assert "torch.save" not in source and "np.save" not in source
    assert "best_state" not in source and "best_epoch" not in source


@pytest.mark.parametrize("clinical_dim", [33, 41])
def test_fit_local_clinical_onehot_dimension_is_not_globally_fixed(clinical_dim):
    ds = _Synthetic(nfit=6, clinical_dim=clinical_dim)
    backbone, _, audit = warmup.train_kernel(ds, 31, 1, epochs=2)
    assert backbone.clinical_dim == clinical_dim
    assert backbone.compatibility.c.weight.shape == (4, clinical_dim + 1)
    assert audit["optimizer_steps"] == 2 and audit["heldout_endpoint_reads"] == 0
    assert all(row["gradients"]["clinical_nonintercept"] > 0 for row in audit["history"])


@pytest.mark.parametrize("epochs", [0, -1, 81, 1.5, True])
def test_invalid_budget_fails_before_training(epochs):
    with pytest.raises(ValueError, match="epochs"):
        warmup.train_kernel(_Synthetic(nfit=3), 31, 1, epochs=epochs)


@pytest.mark.parametrize("fault", ["no_events", "duplicate_fit", "bad_mask", "bad_membership", "zero_scale"])
def test_invalid_fit_contract_fails_closed(fault):
    ds = _Synthetic(nfit=4)
    if fault == "no_events":
        ds._events[:] = 0
    elif fault == "duplicate_fit":
        ds.fit[0] = ds.fit[1]
    elif fault == "bad_mask":
        ds.ft["RM"] = ds.ft["RM"].float()
    elif fault == "bad_membership":
        ds.ft["membership"][0] = False
    else:
        ds.anchor_sd_fit = 0.
    with pytest.raises((ValueError, TypeError)):
        warmup.train_kernel(ds, 31, 1, epochs=2)

"""Synthetic end-to-end frozen-feature/readout qualification, without files."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from src.phase23_cdei.model import CompatibilityBackbone, FrozenDualEvidence, module_fingerprint
from src.phase23_cdei.reference import FrozenPatientReferenceBank
from src.phase23_cdei.train import ARMS, extract_and_fit


class _SyntheticData:
    def __init__(self):
        generator = torch.Generator().manual_seed(230424)
        self.fit = np.array([5, 0, 7, 2, 6, 1, 4, 3], dtype=np.int64)
        self.valid = np.array([10, 8, 9], dtype=np.int64)
        # Deliberately different feature, patient, fit and WSI row orders.
        self.active = np.array([10, 3, 5, 0, 8, 7, 2, 6, 1, 9, 4], dtype=np.int64)
        self.owner = np.array([3, 1, 1, 9, 0, 5, 2, 10, 7, 8, 6, 4, 4], dtype=np.int64)
        self.role, self.seed, self.fold = "inner", 31, 1
        self.audit = {"feature_cache_sha256": "synthetic-not-a-real-cache"}
        member = torch.zeros((8, 512), dtype=torch.bool)
        for k in range(8):
            member[k, k * 64:(k + 1) * 64] = True
        self.ft = {"membership": member,
                   "C": torch.randn((11, 41), generator=generator),
                   "R": torch.randn((11, 512), generator=generator),
                   "G": torch.rand((11, 512), generator=generator),
                   "L": torch.rand((11, 512), generator=generator),
                   "RM": torch.ones((11, 512), dtype=torch.bool),
                   "CM": torch.ones((11, 512), dtype=torch.bool),
                   "ANCHOR": torch.linspace(-.7, 1.3, 11)}
        self.visual = torch.randn((len(self.owner), 4, 1536), generator=generator)
        self.mask = torch.ones((len(self.owner), 4), dtype=torch.bool)
        self.mask[:, -1] = False
        self.visual[:, -1] = float("nan")
        self.wrong_o = np.arange(11, dtype=np.int64)
        self.wrong_c = np.arange(11, dtype=np.int64)
        for partition in (self.fit, self.valid):
            self.wrong_o[partition] = np.roll(partition, 1)
            self.wrong_c[partition] = np.roll(partition, -1)
        self.inherited_simple = {"models": {"J_CS": {"risk": np.array([.2, -.1, .3])}}}
        self._times = np.array([1., 4., 2., 5., 3., 3., 7., 6., 4., 8., 2.])
        self._events = np.array([1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 1.])
        self.fit_label_calls = 0

    @property
    def ep(self):
        raise AssertionError("direct endpoint access forbidden")

    def selection_labels(self):
        raise AssertionError("heldout labels forbidden in extraction/readout fitting")

    def fit_labels(self):
        self.fit_label_calls += 1
        return {"time": self._times[self.fit].copy(), "event": self._events[self.fit].copy()}


@pytest.fixture(scope="module")
def fitted():
    torch.set_num_threads(2)
    torch.manual_seed(230525)
    ds = _SyntheticData()
    backbone = CompatibilityBackbone(41, ds.ft["membership"])
    backbone.requires_grad_(False).eval()
    fingerprint = module_fingerprint(backbone)
    payload, state = extract_and_fit(backbone, ds, "synthetic-source")
    assert module_fingerprint(backbone) == fingerprint
    return ds, backbone, payload, state


def test_all_eight_arms_share_exact_relative_blocks_and_fixed_capacity(fitted):
    ds, _, payload, _ = fitted
    assert tuple(payload["arms"]) == ARMS
    assert len(ARMS) == 8
    assert ds.fit_label_calls == 1
    relative = payload["arms"]["REL"]
    for arm, value in payload["arms"].items():
        dimensions = 64 if arm == "REL" else 66
        assert value["fit_features"].shape == (len(ds.fit), dimensions)
        assert value["valid_features"].shape == (len(ds.valid), dimensions)
        np.testing.assert_array_equal(value["fit_features"][:, :64], relative["fit_features"])
        np.testing.assert_array_equal(value["valid_features"][:, :64], relative["valid_features"])
        assert value["weights"].shape == (dimensions,)
        diagnostic = value["diagnostic"]
        assert diagnostic["status"] == "CONVERGED" and diagnostic["penalty"] == 10.
        assert diagnostic["parameters"] == dimensions and diagnostic["gradient_max"] <= 1e-7
        assert not diagnostic["offset_trainable"] and not diagnostic["intercept"]
    assert payload["feature_audit"]["readout_solver_calls"] == 8


def test_scaler_fits_exactly_training_rows_not_combined_rows(fitted):
    ds, _, payload, _ = fitted
    differs_from_pooled_mean = False
    for value in payload["arms"].values():
        x, v, scaler = value["fit_features"], value["valid_features"], value["scaler"]
        raw_sd = x.std(0, ddof=0)
        np.testing.assert_array_equal(scaler["mean"], x.mean(0))
        np.testing.assert_array_equal(scaler["scale"], np.where(raw_sd < 1e-8, 1., raw_sd))
        np.testing.assert_array_equal(scaler["constant"], raw_sd < 1e-8)
        assert scaler["fit_n"] == len(ds.fit)
        differs_from_pooled_mean |= not np.allclose(scaler["mean"], np.r_[x, v].mean(0), rtol=1e-7, atol=1e-10)
    assert differs_from_pooled_mean


def test_same_dual_head_interventions_reuse_dual_scaler_and_weights(fitted):
    _, _, payload, _ = fitted
    dual = payload["arms"]["DUAL"]
    mean, scale, w = dual["scaler"]["mean"], dual["scaler"]["scale"], dual["weights"]
    standardized = (dual["valid_features"] - mean) / scale
    standardized[:, -2:] = 0.
    np.testing.assert_array_equal(payload["sensitivities"]["DUAL_ABS_OFF"],
                                  payload["anchor_valid"] + standardized @ w)
    for arm in ("WRONG_O", "WRONG_C", "NO_C"):
        # Deliberately do NOT use that separately fitted control's weights.
        x = (payload["arms"][arm]["valid_features"] - mean) / scale
        expected = payload["anchor_valid"] + x @ w
        np.testing.assert_array_equal(payload["sensitivities"]["DUAL_" + arm], expected)


def test_exact_frozen_anchor_and_equal_wsi_patient_pooling(fitted):
    ds, backbone, payload, _ = fitted
    anchor = ds.ft["ANCHOR"].double().numpy()
    np.testing.assert_array_equal(payload["anchor_fit"], anchor[ds.fit])
    np.testing.assert_array_equal(payload["anchor_valid"], anchor[ds.valid])
    for value in payload["arms"].values():
        for partition in ("fit", "valid"):
            x = (value[partition + "_features"] - value["scaler"]["mean"]) / value["scaler"]["scale"]
            risk = value["fit_risk"] if partition == "fit" else value["risk"]
            np.testing.assert_array_equal(risk, payload["anchor_" + partition] + x @ value["weights"])
    with torch.no_grad():
        ft = ds.ft
        tokens, mask = backbone.encode_omics(*(ft[k] for k in ("R", "G", "L", "RM", "CM")))
        bank = FrozenPatientReferenceBank(tokens, mask, ft["C"], ds.fit,
                    provenance={"kernel_sha256": module_fingerprint(backbone)})
        extractor = FrozenDualEvidence(backbone, bank)
        own = ds.owner
        other = ds.wrong_o[own]
        raw = extractor.extract(ds.visual, ds.mask, ft["C"][own],
                    *(ft[k][own] for k in ("R", "G", "L", "RM", "CM")), own,
                    reference_exclusions=np.stack((own, other), -1),
                    absolute_token_mask=mask[own] & mask[other])
    for partition in ("fit", "valid"):
        patients = getattr(ds, partition)
        manual_relative = np.stack([raw["relative"][own == p].double().numpy().mean(0) for p in patients])
        manual_absolute = np.stack([raw["absolute"][own == p].numpy().mean(0) for p in patients])
        # Relative pooling occurs in the original float32 feature dtype before
        # the readout casts to float64; compare within that documented precision.
        np.testing.assert_allclose(payload["arms"]["REL"][partition + "_features"],
                                   manual_relative, rtol=1e-6, atol=2e-8)
        np.testing.assert_allclose(payload["arms"]["DUAL"][partition + "_features"][:, -2:],
                                   manual_absolute, rtol=1e-12, atol=1e-12)
    assert payload["feature_audit"]["equal_WSI_patient_pooling"]


def test_heldout_label_mutation_cannot_change_fitted_state_or_predictions(fitted):
    ds, backbone, payload, _ = fitted
    changed = copy.deepcopy(ds)
    changed.fit_label_calls = 0
    changed._times[changed.valid] = [1e6, 2e6, 3e6]
    changed._events[changed.valid] = 1. - changed._events[changed.valid]
    other, _ = extract_and_fit(backbone, changed, "synthetic-source")
    assert changed.fit_label_calls == 1
    for arm in ARMS:
        for key in ("weights", "risk", "fit_risk", "fit_features", "valid_features"):
            np.testing.assert_array_equal(other["arms"][arm][key], payload["arms"][arm][key])
    for key in payload["sensitivities"]:
        np.testing.assert_array_equal(other["sensitivities"][key], payload["sensitivities"][key])


def test_heldout_feature_changes_do_not_enter_scaler_pca_reference_or_readout_fit(fitted):
    ds, backbone, payload, state = fitted
    changed = copy.deepcopy(ds)
    changed.fit_label_calls = 0
    changed.ft["C"][changed.valid] += 10.
    changed.ft["R"][changed.valid] *= -3.
    changed.visual[np.isin(changed.owner, changed.valid)] *= 2.
    other, other_state = extract_and_fit(backbone, changed, "synthetic-source")
    for arm in ARMS:
        original, altered = payload["arms"][arm], other["arms"][arm]
        for key in ("fit_features", "weights", "fit_risk"):
            np.testing.assert_array_equal(original[key], altered[key])
        for key in ("mean", "scale", "constant"):
            np.testing.assert_array_equal(original["scaler"][key], altered["scaler"][key])
    assert payload["feature_audit"]["additive_control"] == other["feature_audit"]["additive_control"]
    for key in state:
        if isinstance(state[key], torch.Tensor):
            torch.testing.assert_close(state[key], other_state[key], rtol=0, atol=0)
        else:
            assert state[key] == other_state[key]
    assert not np.array_equal(payload["arms"]["DUAL"]["valid_features"],
                              other["arms"]["DUAL"]["valid_features"])


def test_extraction_only_does_not_read_any_endpoint(fitted):
    ds, backbone, _, _ = fitted
    changed = copy.deepcopy(ds)
    changed.fit_label_calls = 0
    def forbidden():
        raise AssertionError("feature-only path read an endpoint")
    changed.fit_labels = forbidden
    payload, state = extract_and_fit(backbone, changed, "synthetic-source", fit_readouts=False)
    assert payload["arms"] == {} and payload["sensitivities"] == {}
    assert payload["feature_audit"]["selection_label_reads"] == 0
    assert changed.fit_label_calls == 0
    np.testing.assert_array_equal(state["_fit_indices"].numpy(), changed.fit)


def test_integration_allows_fit_local_clinical_dimension_33():
    torch.set_num_threads(2)
    torch.manual_seed(230526)
    ds = _SyntheticData()
    ds.ft["C"] = ds.ft["C"][:, :33].clone()
    backbone = CompatibilityBackbone(ds.ft["C"].shape[1], ds.ft["membership"])
    backbone.requires_grad_(False).eval()
    payload, _ = extract_and_fit(backbone, ds, "synthetic-variable-clinical-dimension")
    assert len(payload["arms"]) == 8 and ds.fit_label_calls == 1
    assert all(x["diagnostic"]["status"] == "CONVERGED" for x in payload["arms"].values())

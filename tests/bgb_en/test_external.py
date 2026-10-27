"""Synthetic source-pretraining qualification; no real source fit is run."""
import numpy as np
import pytest
import torch

from src.bgb_en.external import (INPUT_NORMALIZATION, ExternalProgramModel, balanced_class_weights,
    fit_target_scaler, initialize_pathology_project, natural_weights, source_loss,
    source_partitions, normalize_source_features)
from src.bgb_en.pathology import PathologyEncoder


def test_hierarchical_source_weight_and_class_balance():
    donor = np.array(["a"]*4+["b"]*6)
    section = np.array(["a1"]*3+["a2"]+["b1"]*6)
    weight = natural_weights(donor, section)
    assert np.isclose(weight[donor == "a"].sum(), .5)
    assert np.isclose(weight[donor == "b"].sum(), .5)
    assert np.isclose(weight[section == "a1"].sum(), .25)
    assert np.isclose(weight[section == "a2"].sum(), .25)
    tissue = np.array([0, 0, 1, 2, 0, 0, 1, 1, 2, 2])
    cw = balanced_class_weights(tissue, weight)
    for label in range(3):
        assert np.isclose(np.sum(weight[tissue == label]*cw[label]), 1/3)


def test_target_statistics_are_source_fit_only_and_mask_no_substitution():
    rng = np.random.default_rng(7)
    y = rng.normal(size=(15, 4))
    fit = np.arange(10)
    weight = natural_weights(np.repeat(["a", "b"], 5), np.repeat(["a1", "b1"], 5))
    original = fit_target_scaler(y[fit], weight)
    y[10:] = 1e20
    unchanged = fit_target_scaler(y[fit], weight)
    assert np.array_equal(original["mean"], unchanged["mean"])
    assert np.array_equal(original["scale"], unchanged["scale"])
    with pytest.raises(ValueError, match="constant"):
        fit_target_scaler(np.ones((10, 4)), weight)


def test_projection_matches_uni2_p1_and_gradients_are_active():
    source = ExternalProgramModel()
    target = PathologyEncoder("P1", input_dim=1536)
    assert source.project.state_dict().keys() == target.project.state_dict().keys()
    for key in source.project.state_dict():
        assert source.project.state_dict()[key].shape == target.project.state_dict()[key].shape
    x = torch.randn(12, 1536)
    p, t = source(x)
    loss, pieces = source_loss(p, t, torch.randn(12, 4), torch.arange(12)%3,
                              torch.full((12,), 1/12), torch.ones(3))
    torch.testing.assert_close(loss, pieces["program_huber"]+.1*pieces["tissue_CE"])
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in source.parameters())


def test_seven_partitions_are_donor_disjoint_and_full_fit_last():
    donors = np.repeat(np.array(list("abcdef")), 4)
    data = dict(donor=donors, X=np.zeros((24, 1536)))
    partitions = source_partitions(data)
    assert len(partitions) == 7
    for partition in partitions[:6]:
        assert len(partition["train"]) == 20 and len(partition["valid"]) == 4
        assert not set(donors[partition["train"]]) & set(donors[partition["valid"]])
    assert np.array_equal(partitions[6]["train"], np.arange(24))
    assert len(partitions[6]["valid"]) == 0


def test_project_transfer_preserves_heads_and_rejects_feature_space_mismatch(tmp_path):
    source = ExternalProgramModel()
    target = PathologyEncoder("P1", input_dim=1536)
    heads = {key: value.clone() for key, value in target.state_dict().items() if not key.startswith("project.")}
    path = tmp_path/"source.pt"
    torch.save(dict(schema="bgb_en_external_projection_transfer_v2_unitnorm", role="full_source_transfer", epochs=20,
                    source_donors=6,
                    input_normalization=INPUT_NORMALIZATION,
                    project_state_dict=source.project.state_dict(), source_data_sha256="0"*64,
                    TCGA_DSS_read=False, original_ST_test_read=False), path)
    audit = initialize_pathology_project(target, path)
    assert audit["target_heads_replaced"] is False
    for key, value in source.project.state_dict().items():
        assert torch.equal(target.project.state_dict()[key], value)
    for key, value in heads.items():
        assert torch.equal(target.state_dict()[key], value)
    with pytest.raises(ValueError, match="feature space"):
        initialize_pathology_project(PathologyEncoder("P1", input_dim=768), path)


def test_source_normalization_is_exact_float32_row_l2_and_raw_readonly():
    rng = np.random.default_rng(92)
    raw = rng.normal(size=(20, 1536)).astype(np.float32)*.4
    before = raw.copy()
    normalized, audit = normalize_source_features(raw)
    expected = raw/np.linalg.norm(raw, axis=1)[:, None]
    assert np.array_equal(normalized, expected)
    assert np.array_equal(raw, before)
    assert normalized.dtype == np.float32 and not normalized.flags.writeable
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1, atol=2e-6, rtol=0)
    assert audit["raw_array_modified"] is False and audit["raw_median_L2"] > 10
    with pytest.raises(ValueError, match="zero"):
        normalize_source_features(np.zeros((2, 1536), np.float32))


def test_legacy_raw_projection_is_rejected_by_unitnorm_transfer(tmp_path):
    path = tmp_path/"old.pt"
    torch.save(dict(schema="bgb_en_external_projection_transfer_v1", role="full_source_transfer",
                    epochs=20, source_donors=6, project_state_dict=ExternalProgramModel().project.state_dict(),
                    source_data_sha256="0"*64, TCGA_DSS_read=False, original_ST_test_read=False), path)
    with pytest.raises(ValueError, match="schema"):
        initialize_pathology_project(PathologyEncoder("P1", input_dim=1536), path)

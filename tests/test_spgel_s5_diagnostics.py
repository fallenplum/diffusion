"""Synthetic frozen interventions; forbidden label metadata raises on access."""
import json

import numpy as np
import pytest
import torch

from src.spgel.contracts import SourceStatistics, SpatialSource
from src.spgel_s5.diagnostics import paired_diagnostics
from src.spgel_s5.models import ARMS, S5ProgramEncoder

torch.set_num_threads(2)


class Source:
    def __init__(self, n=12):
        rng = np.random.default_rng(509)
        self.features = rng.normal(size=(n, 1536)).astype(np.float32)
        self.features /= np.linalg.norm(self.features, axis=1, keepdims=True)
        self.genes = rng.uniform(.1, 3., size=(n, 24)).astype(np.float64)
        self.gene_mask = np.ones_like(self.genes, bool)
        self.xy_um = np.c_[np.arange(n)*10., np.zeros(n)]
        nearest = np.minimum(np.arange(n)+1, n-1)
        nearest[-1] = n-2
        self.neighbors = np.c_[np.arange(n), nearest].astype(np.int64)
        self.last_image_rows = None
        self.image_batch_calls = 0

    def __getattr__(self, name):
        if name in ("targets", "target_mask", "tissue", "library", "library_size", "donor", "section"):
            raise AssertionError("FORBIDDEN_LABEL_METADATA_READ:" + name)
        raise AttributeError(name)

    def image_batch(self, rows, *, allowed_rows, device="cpu"):
        self.last_image_rows = np.array(rows, copy=True)
        self.image_batch_calls += 1
        return SpatialSource.image_batch(self, rows, allowed_rows=allowed_rows, device=device)


class Stats:
    gene_mean = np.zeros(24)
    gene_sd = np.ones(24)
    genes_z = SourceStatistics.genes_z


def model(arm, *, n=12):
    data = Source(n)
    up, down = np.zeros((4, 24), bool), np.zeros((4, 24), bool)
    for k in range(4):
        up[k, (np.arange(6)+k*3) % 24] = True
        down[k, (np.arange(6)+k*3+8) % 24] = True
    symbols = [f"DIAGNOSTIC_FIXTURE_GENE_{i}" for i in range(24)]
    return S5ProgramEncoder(arm, up, down, symbols), data


@pytest.mark.parametrize("arm", ARMS)
def test_intervention_separates_self_capacity_and_cross_conditioning(arm):
    encoder, data = model(arm)
    result = paired_diagnostics(encoder, data, np.arange(12), Stats())
    assert result["status"] == "COMPLETE_THREE_FROZEN_TASK_B_RUNS"
    assert result["probe_count"] == 12 and result["paired_score_calls"] == 3
    assert result["model_state_unchanged"] and result["existing_gradients_unchanged"]
    gi = result["gene_to_image"]["changed_gene_context"]["RMS"]
    ig = result["image_to_gene"]["changed_image_context"]["RMS"]
    if arm in ("M0", "M1", "M4"):
        assert gi == 0
        assert result["gene_to_image"]["changed_gene_attention"]["maximum_absolute_difference"] == 0
    else:
        assert gi > 0
    if arm in ("M0", "M1", "M3"):
        assert ig == 0
        assert result["image_to_gene"]["changed_image_up_attention"]["maximum_absolute_difference"] == 0
        assert result["image_to_gene"]["changed_image_down_attention"]["maximum_absolute_difference"] == 0
    else:
        assert ig > 0
    if arm in ("M0", "M4"):
        assert result["gene_to_image"]["base_to_conditioned"]["RMS"] == 0
    if arm in ("M0", "M3"):
        assert result["image_to_gene"]["base_to_conditioned"]["RMS"] == 0
    if arm == "M1":
        assert result["gene_to_image"]["base_to_conditioned"]["RMS"] > 0
        assert result["image_to_gene"]["base_to_conditioned"]["RMS"] > 0
        # Paired score changes because one source of evidence is replaced;
        # it must not be conflated with a cross-condition token change.
        assert result["paired_score_change"]["rolled_genes"]["RMS"] > 0
    assert result["optimizer_instances"] == result["optimizer_updates"] == result["backward_calls"] == 0
    json.dumps(result, allow_nan=False)


def test_fixed_128_row_selection_is_value_blind_and_preserves_source_arrays():
    encoder, data = model("M2", n=137)
    rows = np.arange(137, dtype=np.int64)
    features, genes, mask = data.features.copy(), data.genes.copy(), data.gene_mask.copy()
    a = paired_diagnostics(encoder, data, rows, Stats())
    expected = rows[np.linspace(0, 136, 128, dtype=np.int64)]
    assert np.array_equal(data.last_image_rows, expected) and data.image_batch_calls == 1
    assert a["probe_count"] == 128
    assert np.array_equal(features, data.features) and np.array_equal(genes, data.genes)
    assert np.array_equal(mask, data.gene_mask)
    data.genes *= 200
    b = paired_diagnostics(encoder, data, rows, Stats())
    assert a["probe_rows_sha256"] == b["probe_rows_sha256"]
    assert a["rolled_local_mapping_sha256"] == b["rolled_local_mapping_sha256"]


def test_training_modes_rng_and_existing_gradients_are_unchanged():
    encoder, data = model("M2")
    encoder.train(); encoder.gene_encoder.eval()
    encoder.gene_to_image.weight.grad = torch.ones_like(encoder.gene_to_image.weight)
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    modes = {name: module.training for name, module in encoder.named_modules()}
    rng = torch.random.get_rng_state().clone()
    result = paired_diagnostics(encoder, data, np.arange(12), Stats())
    assert all(torch.equal(value, encoder.state_dict()[name]) for name, value in before.items())
    assert modes == {name: module.training for name, module in encoder.named_modules()}
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(encoder.gene_to_image.weight.grad, torch.ones_like(encoder.gene_to_image.weight))
    assert result["training_modes_restored"]


def test_all_missing_program_evidence_keeps_undefined_metrics_null():
    encoder, data = model("M2")
    data.gene_mask[:] = False
    data.genes[:] = np.nan
    result = paired_diagnostics(encoder, data, np.arange(12), Stats())
    for direction in ("gene_to_image", "image_to_gene"):
        assert result[direction]["base_to_conditioned"]["valid_spot_programs"] == 0
        assert result[direction]["base_to_conditioned"]["RMS"] is None
    assert result["paired_score_change"]["rolled_genes"]["valid_pairs"] == 0
    assert result["paired_score_change"]["rolled_genes"]["RMS"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("rows", [np.array([0]), np.array([0, 0]), np.array([-1, 2]), np.array([0., 1.])])
def test_invalid_probe_scope_fails_before_model_or_source_read(rows):
    encoder, data = model("M2")
    with pytest.raises(ValueError, match="REQUIRES_DISTINCT_PARTITION_ROWS"):
        paired_diagnostics(encoder, data, rows, Stats())
    assert data.image_batch_calls == 0

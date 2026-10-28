"""Row-byte randomized gate math and source-cache isolation; no optimizer fits."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from src.bgb_en.r6_source_model import (SourceProgramModel, fixed_feature_gate_permutation,
    gate_probability_audit, GATE_PERMUTATIONS, RANDOM_GATE_ALGORITHM, source_loss)
from src.bgb_en.r6_omics import mapping_from_programs, PROGRAM_ORDER

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("r6_pretrain_gate_tests", ROOT / "scripts/bgb_en_r6_pretrain.py")
pretrain = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(pretrain)
REPORT_SPEC = importlib.util.spec_from_file_location("r6_report_gate_tests", ROOT / "scripts/bgb_en_r6_report_source.py")
reporter = importlib.util.module_from_spec(REPORT_SPEC); REPORT_SPEC.loader.exec_module(reporter)


def features(n=64):
    return F.normalize(torch.randn(n, 1536, generator=torch.Generator().manual_seed(61031)), dim=-1)


def model(mode):
    return SourceProgramModel("S4", tissue_prior=[.3, .4, .3], gate_mode=mode, init_seed=31)


def test_original_float32_row_bytes_alone_determine_reorder_and_split_invariant_permutation():
    x = features()
    actual = fixed_feature_gate_permutation(x)
    expected = [GATE_PERMUTATIONS[int.from_bytes(hashlib.sha256(row.numpy().astype("<f4", copy=False).tobytes()).digest()[:8], "little") % 6]
                for row in x]
    assert torch.equal(actual, torch.tensor(expected))
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(31))
    assert torch.equal(actual[order], fixed_feature_gate_permutation(x[order]))
    assert torch.equal(actual, torch.cat([fixed_feature_gate_permutation(part) for part in x.split(7)]))
    assert torch.equal(fixed_feature_gate_permutation(x[:1].repeat(3, 1)), actual[:1].repeat(3, 1))
    assert torch.equal(actual, fixed_feature_gate_permutation(x.numpy()))
    assert len(actual.unique(dim=0)) > 1  # Synthetic QA, not a real-cohort gate.


def test_random_gate_preserves_own_probability_values_entropy_and_gradient_without_matching_other_models():
    x = features()
    source = model("random")
    out = source(x)
    p, q = out["tissue_probs"], out["program_gate_probs"]
    assert torch.equal(p.sort(-1).values, q.sort(-1).values)
    torch.testing.assert_close(-(p * p.log()).sum(-1), -(q * q.log()).sum(-1), atol=2e-7, rtol=0)
    audit = gate_probability_audit(p, q, out["gate_permutation"], algorithm=RANDOM_GATE_ALGORITHM)
    assert audit["row_probability_multiset_exact"] and audit["used_probabilities_follow_reported_permutation"]
    assert audit["row_probability_multiset_max_abs_difference"] == 0 and audit["row_entropy_max_abs_difference"] < 1e-14
    assert sum(audit["permutation_counts"].values()) == len(x)
    assert audit["identity_rate"] == audit["permutation_counts"]["012"] / len(x)
    assert not audit["between_model_distributions_exactly_matched_claimed"]
    assert not audit["confidence_information_removed_claimed"]
    assert not audit["permutation_uses_labels"] and not audit["permutation_uses_patient_or_spot_ids"]
    # Training may change each model's probability distribution; the control
    # matches only the original/used pair in that very model and forward.
    changed = deepcopy(source)
    with torch.no_grad():
        changed.tissue_head.bias.add_(torch.tensor([.8, -.4, -.4]))
    later = changed(x)
    assert not torch.equal(p, later["tissue_probs"])
    assert torch.equal(later["tissue_probs"].sort(-1).values, later["program_gate_probs"].sort(-1).values)
    assert torch.equal(out["gate_permutation"], later["gate_permutation"])


@pytest.mark.parametrize("mode", ["predicted", "random"])
def test_program_only_gradient_reaches_tissue_head_in_both_control_arms(mode):
    source = model(mode)
    x = features(19).requires_grad_(True)
    target = torch.randn(19, 4, generator=torch.Generator().manual_seed(47))
    F.huber_loss(source(x)["program_predictions"], target).backward()
    assert source.tissue_head.weight.grad is not None and source.tissue_head.weight.grad.abs().sum() > 0
    assert source.condition_gate.weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_precomputed_indices_equal_fallback_and_cannot_enter_predicted_or_invalid_gate():
    x = features(13)
    order = fixed_feature_gate_permutation(x)
    source = model("random")
    a, b = source(x), source(x, gate_permutation=order)
    for key in ("program_predictions", "tissue_logits", "program_gate_probs", "queries", "gate_permutation"):
        assert torch.equal(a[key], b[key])
    with pytest.raises(ValueError, match="ONLY_ALLOWED_FOR_RANDOM"):
        model("predicted")(x, gate_permutation=order)
    with pytest.raises(ValueError, match="BAD_FIXED_CLASS"):
        source(x, gate_permutation=torch.zeros_like(order))
    with pytest.raises(ValueError, match="FLOAT32"):
        fixed_feature_gate_permutation(x.double())
    with pytest.raises(ValueError, match="UNIT_L2"):
        fixed_feature_gate_permutation(x * 2)


def test_both_arms_keep_identical_active_capacity_initial_weights_and_no_gaussian_gate_state():
    left, right = model("predicted"), model("random")
    assert left.parameter_budget() == right.parameter_budget()
    for key, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[key])
    assert "random_gate_matrix" not in right.state_dict()
    assert right.descriptor()["random_gate_algorithm"] == RANDOM_GATE_ALGORITHM
    assert not right.descriptor()["legacy_Gaussian_random_gate_used"]
    assert right.descriptor()["random_gate_retains_confidence_information"]


def test_full_objective_and_replayed_microbatches_have_same_random_gate_gradient():
    full, cached = model("random"), model("random")
    x = features(17)
    order = fixed_feature_gate_permutation(x)
    target = torch.randn(17, 4, generator=torch.Generator().manual_seed(83))
    tissue = torch.arange(17) % 3
    def forward(source, a, b):
        out = source(x[a:b], gate_permutation=order[a:b])
        return {key: out[key] for key in ("program_predictions", "tissue_logits")}
    def objective(source, output):
        return source_loss(source, output, target, tissue)
    loss, _ = objective(full, forward(full, 0, len(x)))
    (loss + .0001 * sum(p.square().sum() for p in full.parameters() if p.requires_grad)).backward()
    audit = pretrain.deterministic_output_gradient_cache_backward(cached,
        lambda a, b: forward(cached, a, b), len(x), lambda output: objective(cached, output), microbatch=5, parameter_L2=.0001)
    assert audit["replay_max_abs_error"] <= 2e-6
    for (name, expected), (_, actual) in zip(full.named_parameters(), cached.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(expected.grad, actual.grad, atol=3e-7, rtol=2e-4, msg=name)


def test_gate_metrics_retain_per_donor_counts_and_source_report_drops_unknown_rows():
    source = model("random")
    out = {key: value.detach().numpy() for key, value in source(features(24)).items() if torch.is_tensor(value)}
    y = np.arange(96, dtype=float).reshape(24, 4) / 20
    donor = np.repeat([0, 1], 12)
    metrics = pretrain.evaluate_source_outputs(out, y, np.arange(24) % 3, np.linspace(100, 200, 24),
        class_weights=np.ones(3), donor_codes=donor, gate_algorithm=RANDOM_GATE_ALGORITHM)
    assert metrics["gate_diagnostics"]["rows"] == 24
    assert [r["diagnostics"]["rows"] for r in metrics["gate_diagnostics_by_donor"]] == [12, 12]
    safe = reporter.cell_gate_metadata(dict(train=metrics), dict(heldout_donor=2, control="RANDOM_GATE"))
    assert safe["train"]["total"]["row_probability_multiset_exact"]
    assert len(safe["train"]["by_donor"]) == 2
    with pytest.raises(ValueError, match="TRAIN_INCLUDED_HELDOUT_DONOR"):
        reporter.cell_gate_metadata(dict(train=metrics), dict(heldout_donor=0, control="RANDOM_GATE"))
    injected = deepcopy(metrics["gate_diagnostics"])
    injected["raw_patient_probability_rows"] = "PRIVATE_ROWS_DO_NOT_PUBLISH"
    assert "PRIVATE_ROWS_DO_NOT_PUBLISH" not in json.dumps(reporter.gate_metadata(injected))
    injected["permutation_uses_labels"] = True
    with pytest.raises(ValueError, match="SCOPE_OR_PRIVACY"):
        reporter.gate_metadata(injected)


def test_source_fit_data_precomputes_tiny_indices_and_never_rehashes_training_chunks(monkeypatch):
    class Source(SimpleNamespace):
        def __len__(self):
            return len(self.features_unit)
    mapping = mapping_from_programs([dict(id=name, up_genes=[str(i)], down_genes=[str(i + 4)])
                                    for i, name in enumerate(PROGRAM_ORDER)])
    n = 72
    donor = np.repeat(np.arange(6), 12)
    dataset = Source(features_unit=features(n).numpy(), donor_codes=donor, section_codes=donor.copy(),
        program_targets=np.arange(n * 4, dtype=np.float64).reshape(n, 4) / 20,
        tissue_y=np.arange(n, dtype=np.int64) % 3, library_size=np.arange(n, dtype=float) + 100,
        neighbors=np.arange(n, dtype=np.int64)[:, None], gene_symbols=np.asarray(mapping.gene_symbols),
        log1p_cp10k=np.zeros((n, 8)), measured_mask=np.ones((n, 8), bool))
    cell = dict(arm="S4", control="RANDOM_GATE", heldout_donor=0, seed=31)
    data = pretrain.SourceFitData(dataset, cell, mapping, "cpu")
    source = pretrain._model_for_cell(cell, data, mapping, "cpu")
    assert data.gate_permutation.shape == (n, 3) and data.gate_permutation.numel() * data.gate_permutation.element_size() == n * 3 * 8
    rows = np.array([50, 13, 35, 22])
    expected = source(torch.as_tensor(dataset.features_unit[rows]))
    def forbidden(_):
        raise AssertionError("a training chunk must use the precomputed indices")
    import src.bgb_en.r6_source_model as source_module
    monkeypatch.setattr(source_module, "fixed_feature_gate_permutation", forbidden)
    actual = data.chunk(source, rows)
    assert torch.equal(expected["gate_permutation"], actual["gate_permutation"])
    assert torch.equal(expected["program_predictions"], actual["program_predictions"])
    outputs = pretrain.collect_outputs(source, data, data.train, microbatch=7)
    assert outputs["gate_permutation"].shape == outputs["tissue_probs"].shape == outputs["program_gate_probs"].shape == (60, 3)
    np.testing.assert_array_equal(np.sort(outputs["tissue_probs"], 1), np.sort(outputs["program_gate_probs"], 1))

"""Synthetic qualification only; no cohort, endpoint or real checkpoint IO."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.phase26_multiquery import model as old_model
from src.phase27_readout import features as f


@pytest.fixture
def example():
    torch.set_num_threads(2)
    torch.manual_seed(270018)
    patch = torch.nn.functional.normalize(torch.randn(3, 7, 768), dim=-1)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [1, 0, 1, 0, 1, 0, 1]], dtype=torch.bool)
    patch[~mask] = float("nan")
    return patch, mask, torch.randn(3, 512), torch.randn(3, 5)


def active_model(arm):
    torch.manual_seed(270100)
    model = old_model.MultiQuery(5, arm).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.head[-1].weight, std=.1)
        torch.nn.init.normal_(model.clinical_up.weight, std=.15)
    return model


def test_permutation_is_deterministic_mask_safe_and_zero_self(example):
    mask = example[1]
    mapping, audit = f.patch_permutation(mask, np.array([9, 15, 20]), 31, 1)
    again, _ = f.patch_permutation(mask, np.array([9, 15, 20]), 31, 1)
    assert torch.equal(mapping, again)
    assert audit == {"WSI_count": 3, "deranged_WSI_count": 2, "single_patch_unpermuted_WSI_count": 1}
    for keep, row in zip(mask, mapping):
        valid = keep.nonzero().flatten()
        assert torch.equal(row[valid].sort().values, valid)
        if len(valid) > 1: assert (row[valid] != valid).all()
        assert torch.equal(row[~keep], (~keep).nonzero().flatten())
    assert torch.equal(mask.gather(1, mapping), mask)
    # Same explicit seed formula, independent of batching/order.
    one, _ = f.patch_permutation(mask[[2]], np.array([20]), 31, 1)
    assert torch.equal(one[0], mapping[2])


@pytest.mark.parametrize("arm", ["ATTN", "COMP"])
def test_frozen_two_round_routing_keeps_distribution_and_values(example, arm):
    model = active_model(arm)
    inputs = [x.clone() for x in example]
    weights = copy.deepcopy(model.state_dict())
    original_function = old_model.attention_weights
    with torch.no_grad():
        matched = model(*example, return_details=True)
        changed, audit = f.permuted_forward(model, example, matched, np.array([9, 15, 20]), 31, 1)
    assert old_model.attention_weights is original_function
    assert audit["routing_calls"] == 2 and audit["probability_distribution_claim"]
    assert max(audit["maximum_distribution_error"].values()) <= 1e-6
    assert audit["maximum_distribution_error"]["row_weight_multiset"] == 0
    assert torch.equal(changed["key"], matched["key"]) and torch.equal(changed["value"], matched["value"])
    assert torch.equal(changed["initial_slots"], matched["initial_slots"])
    assert not torch.equal(changed["slots"][[0, 2]], matched["slots"][[0, 2]])
    assert torch.equal(changed["slots"][1], matched["slots"][1])
    mapping, _ = f.patch_permutation(example[1], np.array([9, 15, 20]), 31, 1)
    for a, b in zip(matched["rounds"], changed["rounds"]):
        torch.testing.assert_close(b["attention"], f._column_gather(a["attention"], mapping), atol=0, rtol=0)
        torch.testing.assert_close(b["message"], b["attention"]@matched["value"], atol=0, rtol=0)
    assert not torch.equal(changed["rounds"][1]["input_slots"][[0, 2]], matched["rounds"][1]["input_slots"][[0, 2]])
    for x, before in zip(example, inputs): assert f._same_bits(x, before)
    for key, value in model.state_dict().items(): assert torch.equal(value, weights[key])


def test_query_permutation_is_disabled_and_has_no_distribution_claim(example):
    model = active_model("QUERY")
    with torch.no_grad(): matched = model(*example, return_details=True)
    changed, audit = f.permuted_forward(model, example, matched, np.array([9, 15, 20]), 31, 1)
    assert changed is matched and not audit["probability_distribution_claim"]
    assert audit["routing_calls"] == 0 and audit["maximum_distribution_error"] == {}
    assert not changed["attention"].any()


def test_override_restores_module_after_exception_and_rejects_nesting(example):
    model = active_model("COMP")
    with torch.no_grad(): matched = model(*example, return_details=True)
    mapping, _ = f.patch_permutation(example[1], np.array([9, 15, 20]), 31, 1)
    original = old_model.attention_weights
    with pytest.raises(RuntimeError, match="injected"):
        with f.frozen_routing(matched, example[1], mapping, "COMP"):
            raise RuntimeError("injected")
    assert old_model.attention_weights is original
    with pytest.raises(ValueError, match="nested"):
        with f.frozen_routing(matched, example[1], mapping, "COMP"):
            with f.frozen_routing(matched, example[1], mapping, "COMP"):
                pass
    assert old_model.attention_weights is original
    # The lock is released after exceptions, not poisoned for the next replay.
    with torch.no_grad(): f.permuted_forward(model, example, matched, np.array([9, 15, 20]), 31, 1)


def test_invalid_permutation_inputs_fail_closed(example):
    with pytest.raises(ValueError): f.patch_permutation(torch.zeros(1, 2, dtype=torch.bool), [0], 31, 1)
    with pytest.raises(ValueError): f.patch_permutation(example[1], [1, 1, 2], 31, 1)
    with pytest.raises(ValueError): f.patch_permutation(example[1].float(), [1, 2, 3], 31, 1)
    with pytest.raises(ValueError): f.patch_permutation(example[1], [-1, 2, 3], 31, 1)


def synthetic_cell():
    torch.manual_seed(270200)
    # Nonmonotone patient order and interleaved slides exercise equal-WSI means.
    owner = np.array([2, 0, 3, 1, 0, 2])
    visual = torch.nn.functional.normalize(torch.randn(6, 5, 768), dim=-1)
    mask = torch.ones(6, 5, dtype=torch.bool); mask[:, -1] = False
    visual[~mask] = float("nan")
    ds = SimpleNamespace(fit=np.array([1, 0]), valid=np.array([3, 2]), owner=owner,
                         visual=visual, mask=mask, R=torch.randn(4, 512), C=torch.randn(4, 5),
                         base=torch.tensor([.1, -.3, .5, .7], dtype=torch.float64),
                         wrong_rna=np.array([1, 0, 3, 2]))
    record = {"source": "synthetic-not-formal", "stage": "prototype", "seed": 31, "fold": 1,
              "fit": ds.fit.copy(), "valid": ds.valid.copy(), "arms": {}}
    # Independent manual fixture, matching the historic grouping and risk formula.
    expected = {}
    with torch.no_grad():
        for arm in f.ARMS:
            model = active_model(arm)
            saved = {"clinical_dim": 5, "model_state": copy.deepcopy(model.state_dict()), "sensitivities": {}}
            expected[arm] = {}
            for partition in ("fit", "valid"):
                patients = getattr(ds, partition)
                slides = np.flatnonzero(np.isin(ds.owner, patients)); owners = ds.owner[slides]
                args = (ds.visual[slides], ds.mask[slides], ds.R[owners], ds.C[owners])
                out = model(*args, return_details=True)
                features = torch.stack([out["slots"][owners == p].double().mean(0).flatten() for p in patients]).numpy()
                units = torch.stack([out["unit"][owners == p].double().mean() for p in patients]).numpy()
                saved["fit_risk" if partition == "fit" else "risk"] = ds.base[patients].numpy()+.1*units
                expected[arm][partition] = features
                if partition == "valid":
                    for control, options in (("RNA_MEAN_QUERY", {"query_rna": ds.R[ds.fit].mean(0).expand(len(owners), -1)}),
                            ("WRONG_RNA_QUERY", {"query_rna": ds.R[ds.wrong_rna[owners]]}),
                            ("VALUE_OFF", {"value_off": True})):
                        altered = model(*args, **options)["unit"]
                        unit = torch.stack([altered[owners == p].double().mean() for p in patients]).numpy()
                        saved["sensitivities"][control] = ds.base[patients].numpy()+.1*unit
            record["arms"][arm] = saved
    return ds, record, expected


def test_complete_extraction_preserves_slots_equal_WSI_aggregation_and_old_risks():
    ds, record, expected = synthetic_cell()
    input_snapshots = {k: v.clone() for k, v in vars(ds).items() if isinstance(v, torch.Tensor)}
    checkpoint = copy.deepcopy(record)
    result = f.extract_cell(ds, record, 31, 1, "cpu")
    audit = result["audit"]
    assert audit["status"] == "PASS_FROZEN_SLOT_EXTRACTION"
    assert audit["old_risk_vectors_replayed"] == 15 and audit["maximum_old_risk_error"] == 0
    assert audit["endpoint_API_reads"] == audit["new_predictor_fits"] == audit["new_neural_training_steps"] == 0
    for arm in f.ARMS:
        fit, valid = result["fit_features"][arm], result["valid_features"][arm]
        assert fit.dtype == np.float64 and fit.shape == (2, 256)
        assert np.array_equal(fit, expected[arm]["fit"])
        assert np.array_equal(valid["MATCHED"], expected[arm]["valid"])
        assert set(valid) == set(f.CONTROLS)
        assert audit["arms"][arm]["model_state_unchanged"]
        for key, value in record["arms"][arm]["model_state"].items():
            assert torch.equal(value, checkpoint["arms"][arm]["model_state"][key])
    for key, value in input_snapshots.items(): assert f._same_bits(getattr(ds, key), value)
    for control in f.CONTROLS:
        assert np.array_equal(result["valid_features"]["QUERY"][control], result["valid_features"]["QUERY"]["MATCHED"])
    assert not np.array_equal(result["valid_features"]["COMP"]["MATCHED"], result["valid_features"]["COMP"]["PATCH_PERMUTE"])


def test_old_risk_mismatch_does_not_adapt_tolerance_or_return_features():
    ds, record, _ = synthetic_cell()
    record["arms"]["QUERY"]["fit_risk"][0] += 2e-7
    with pytest.raises(ValueError, match="fixed tolerance"):
        f.extract_cell(ds, record, 31, 1, "cpu")


def test_batching_is_32_patients_not_32_WSI():
    class BatchRecorder:
        arm = "QUERY"
        def __init__(self): self.calls = []
        def __call__(self, patch, mask, rna, clinical, return_details):
            self.calls.append(len(patch))
            slots = rna[:, :256].reshape(-1, 8, 32)
            return {"slots": slots, "unit": clinical[:, 0].tanh()}
    model = BatchRecorder()
    owner = np.repeat(np.arange(35), 2)
    ds = SimpleNamespace(owner=owner, visual=torch.zeros(70, 2, 768),
                         mask=torch.ones(70, 2, dtype=torch.bool),
                         R=torch.arange(35*512, dtype=torch.float32).reshape(35, 512),
                         C=torch.zeros(35, 1), base=torch.zeros(35, dtype=torch.float64))
    values, risks, audit = f._extract_partition(model, ds, np.arange(35), 31, 1, ("MATCHED",))
    assert model.calls == [64, 6] and audit["patient_batches"] == 2
    assert np.array_equal(values["MATCHED"], ds.R[:, :256].double().numpy())
    assert np.array_equal(risks["MATCHED"], np.zeros(35))


def test_partition_and_stage_mismatch_fail_before_forward():
    ds, record, _ = synthetic_cell()
    record["valid"] = record["valid"][::-1]
    with pytest.raises(ValueError, match="partition"):
        f.extract_cell(ds, record, 31, 1, "cpu")
    record["stage"] = "outer"
    with pytest.raises(ValueError, match="stage"):
        f.extract_cell(ds, record, 31, 1, "cpu")

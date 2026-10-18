"""Endpoint-free extraction from the locked, frozen Phase26 encoders.

The spatial control freezes both matched routing matrices.  Its second round
does not recompute routing after the first round has changed.  Consequently it
is a correspondence diagnostic, not a natural query intervention.  Arrays in
the return value are private patient-level features; only ``audit`` is public.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

import numpy as np
import torch

from src.phase26_multiquery import model as old_model


ARMS = ("QUERY", "ATTN", "COMP")
CONTROLS = ("MATCHED", "RNA_MEAN_QUERY", "WRONG_RNA_QUERY", "VALUE_OFF", "PATCH_PERMUTE")
OLD_RISK_TOLERANCE = 1e-7
DISTRIBUTION_TOLERANCE = 1e-6
PERMUTATION_SEED_BASE = 270117
_ROUTING_LOCK = threading.Lock()


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _same_bits(left, right):
    """Compare padded NaNs without changing their values or NaN bit patterns."""
    return (left.shape == right.shape and left.dtype == right.dtype
            and torch.equal(left.detach().contiguous().view(torch.uint8),
                            right.detach().contiguous().view(torch.uint8)))


def patch_permutation(mask, slide_indices, seed, fold):
    """Return a target-to-source valid-column derangement, padding unchanged.

    A random cycle followed by one cyclic shift has no fixed point for N>=2.
    The active-view slide index fixes the same mapping for all slots/rounds/arms.
    Single-Patch slides have no derangement and are explicitly counted.
    """
    _require(isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.dtype == torch.bool,
             "permutation needs a Boolean WSI/Patch mask")
    indices = np.asarray(slide_indices)
    _require(indices.shape == (len(mask),) and np.issubdtype(indices.dtype, np.integer)
             and np.all(indices >= 0) and len(np.unique(indices)) == len(indices),
             "invalid active-view slide indices")
    observed = mask.detach().cpu().numpy()
    mapping = np.broadcast_to(np.arange(mask.shape[1]), mask.shape).copy()
    single = changed = 0
    for row, (keep, index) in enumerate(zip(observed, indices)):
        valid = np.flatnonzero(keep)
        _require(len(valid) > 0, "empty WSI cannot be spatially permuted")
        if len(valid) == 1:
            single += 1
            continue
        rng = np.random.default_rng(PERMUTATION_SEED_BASE + int(fold)*1000
                                    + int(seed)*10 + int(index))
        cycle = rng.permutation(valid)
        mapping[row, cycle] = np.roll(cycle, 1)
        _require(np.all(mapping[row, valid] != valid)
                 and np.array_equal(np.sort(mapping[row, valid]), valid),
                 "spatial permutation is not a zero-self bijection")
        changed += 1
    return torch.as_tensor(mapping, dtype=torch.long, device=mask.device), {
        "WSI_count": len(mask), "deranged_WSI_count": changed,
        "single_patch_unpermuted_WSI_count": single,
    }


def _column_gather(array, mapping):
    return array.gather(-1, mapping[:, None, :].expand_as(array))


def _distribution_errors(before, after, mask):
    """Double-precision invariants; no medical or top-k tie claims are made."""
    a, b = before["attention"].double(), after["attention"].double()
    aa, bb = before["assignment"].double(), after["assignment"].double()
    valid = mask[:, None, :]
    _require(not a.masked_select(~valid.expand_as(a)).any()
             and not b.masked_select(~valid.expand_as(b)).any(), "nonzero padding weight")
    entropy = lambda x: -(x*x.clamp_min(torch.finfo(torch.float64).tiny).log()).sum(-1)
    ha, hb = entropy(a), entropy(b)
    count = mask.sum(-1).double()[:, None]
    error = lambda x, y: float((x-y).abs().max())
    return {
        "row_weight_multiset": error(a.sort(-1).values, b.sort(-1).values),
        "row_mass": error(a.sum(-1), b.sum(-1)),
        "entropy": error(ha, hb),
        "effective_patch_count": error(ha.exp(), hb.exp()),
        "soft_slot_overlap_gram": error(a@a.transpose(-1, -2), b@b.transpose(-1, -2)),
        "assignment_row_fraction": error(aa.sum(-1)/count, bb.sum(-1)/count),
        "slot_mass": error(before["slot_mass"].double(), after["slot_mass"].double()),
    }


@contextmanager
def frozen_routing(matched, mask, mapping, arm):
    """Scoped override of the old module function, restored even on failure.

    Only the routing function is replaced. Patch encodings and Value rows remain
    untouched. Concurrent/nested routing overrides in one process fail closed.
    """
    _require(arm in ("ATTN", "COMP") and len(matched["rounds"]) == 2,
             "frozen routing requires an enabled two-round operator")
    _require(mapping.shape == mask.shape and mapping.device == mask.device
             and mapping.dtype == torch.long, "invalid permutation shape/device")
    _require(torch.equal(mask.gather(-1, mapping), mask), "permutation changed observed support")
    _require(_ROUTING_LOCK.acquire(blocking=False), "concurrent or nested routing override")
    original = old_model.attention_weights
    state = {"calls": 0}

    def override(scores, current_mask, operator):
        index = state["calls"]
        _require(index < 2 and operator == arm and torch.equal(current_mask, mask),
                 "frozen routing call count/operator/Patch support changed")
        saved = matched["rounds"][index]
        _require(scores.shape == saved["attention"].shape
                 and scores.device == saved["attention"].device
                 and scores.dtype == saved["attention"].dtype,
                 "frozen routing changed score shape/device/dtype")
        _require(torch.isfinite(scores[mask[:, None, :].expand_as(scores)]).all(),
                 "nonfinite counterfactual compatibility")
        state["calls"] += 1
        return (_column_gather(saved["attention"], mapping),
                _column_gather(saved["assignment"], mapping), saved["slot_mass"])

    try:
        old_model.attention_weights = override
        yield state
        _require(state["calls"] == 2, "frozen routing did not replay exactly two rounds")
    finally:
        old_model.attention_weights = original
        _ROUTING_LOCK.release()


@torch.no_grad()
def permuted_forward(model, args, matched, slide_indices, seed, fold):
    """Replay cached matched routing after the locked valid-column permutation."""
    mask = args[1]
    mapping, counts = patch_permutation(mask, slide_indices, seed, fold)
    if model.arm == "QUERY":
        _require(not matched["patch_message_enabled"] and not matched["attention"].any(),
                 "QUERY unexpectedly enabled Patch routing")
        return matched, dict(counts, routing_calls=0, probability_distribution_claim=False,
                             maximum_distribution_error={}, values_unchanged=True)
    with frozen_routing(matched, mask, mapping, model.arm) as state:
        output = model(*args, return_details=True)
    _require(_same_bits(output["value"], matched["value"])
             and _same_bits(output["key"], matched["key"]), "permutation altered K/V rows")
    _require(_same_bits(output["initial_slots"], matched["initial_slots"]),
             "permutation altered own RNA residual")
    worst = {}
    for left, right in zip(matched["rounds"], output["rounds"]):
        _require(_same_bits(right["attention"], _column_gather(left["attention"], mapping))
                 and _same_bits(right["assignment"], _column_gather(left["assignment"], mapping)),
                 "counterfactual routing did not use frozen matched columns")
        for key, value in _distribution_errors(left, right, mask).items():
            worst[key] = max(worst.get(key, 0.), value)
    _require(max(worst.values()) <= DISTRIBUTION_TOLERANCE,
             "frozen permutation failed distribution invariants")
    return output, dict(counts, routing_calls=state["calls"], probability_distribution_claim=True,
                        maximum_distribution_error=worst, values_unchanged=True)


def _patient_arrays(output, owners, patients, base):
    feature, units = [], []
    for patient in patients:
        keep = torch.as_tensor(owners == patient, device=output["slots"].device)
        _require(keep.any(), "patient has no WSI")
        # Preserve each slot: first equal-WSI mean, then flatten (never mean slots).
        feature.append(output["slots"][keep].double().mean(0).flatten())
        units.append(output["unit"][keep].double().mean())
    features = torch.stack(feature)
    risk = base[patients].detach().double()+.1*torch.stack(units)
    _require(features.shape == (len(patients), 256) and torch.isfinite(features).all()
             and torch.isfinite(risk).all(), "nonfinite or wrong-shaped patient output")
    return features.cpu().numpy(), risk.cpu().numpy()


def _max_error(actual, expected, name):
    expected = np.asarray(expected, dtype=np.float64)
    _require(actual.shape == expected.shape and np.isfinite(expected).all(),
             "invalid old risk vector: "+name)
    error = float(np.max(np.abs(actual-expected)))
    _require(error <= OLD_RISK_TOLERANCE, "old risk replay exceeded fixed tolerance: "+name)
    return error


@torch.no_grad()
def _extract_partition(model, ds, patients, seed, fold, controls):
    buffers = {key: [] for key in controls}
    risks = {key: [] for key in controls}
    audit = {"patient_batches": 0, "WSI_count": 0, "deranged_WSI_count": 0,
             "single_patch_unpermuted_WSI_count": 0, "routing_calls": 0,
             "maximum_distribution_error": {}, "inputs_unchanged": True,
             "values_unchanged": True, "own_RNA_initial_slots_unchanged": True,
             "probability_distribution_claim": model.arm != "QUERY"}
    patients = np.asarray(patients, dtype=int)
    for start in range(0, len(patients), 32):
        current = patients[start:start+32]
        slides = np.flatnonzero(np.isin(ds.owner, current))
        owners = np.asarray(ds.owner)[slides]
        _require(len(slides) and np.isin(current, owners).all(), "missing patient WSI")
        args = (ds.visual[slides], ds.mask[slides], ds.R[owners], ds.C[owners])
        snapshots = [value.detach().clone() for value in args]
        matched = model(*args, return_details=True)
        audit["patient_batches"] += 1
        for control in controls:
            if control == "MATCHED":
                output = matched
            elif control == "PATCH_PERMUTE":
                output, spatial = permuted_forward(model, args, matched, slides, seed, fold)
                for key in ("WSI_count", "deranged_WSI_count", "single_patch_unpermuted_WSI_count", "routing_calls"):
                    audit[key] += spatial[key]
                for key, value in spatial["maximum_distribution_error"].items():
                    audit["maximum_distribution_error"][key] = max(
                        audit["maximum_distribution_error"].get(key, 0.), value)
            else:
                options = {}
                if control == "RNA_MEAN_QUERY":
                    options["query_rna"] = ds.R[np.asarray(ds.fit)].mean(0).expand(len(owners), -1)
                elif control == "WRONG_RNA_QUERY":
                    options["query_rna"] = ds.R[np.asarray(ds.wrong_rna)[owners]]
                elif control == "VALUE_OFF":
                    options["value_off"] = True
                else:
                    raise ValueError("unknown feature control")
                output = model(*args, **options, return_details=True)
                for field in ("initial_slots", "key", "value"):
                    _require(_same_bits(output[field], matched[field]),
                             control+" changed own RNA residual or Patch K/V")
            values, risk = _patient_arrays(output, owners, current, ds.base)
            buffers[control].append(values)
            risks[control].append(risk)
        _require(all(_same_bits(value, snapshot) for value, snapshot in zip(args, snapshots)),
                 "feature extraction mutated its input")
    return ({key: np.concatenate(value) for key, value in buffers.items()},
            {key: np.concatenate(value) for key, value in risks.items()}, audit)


@torch.no_grad()
def extract_cell(ds, record, seed, fold, device):
    """Extract private slot features and return only anonymous audit summaries.

    The caller verifies checkpoint/DONE/source receipts and supplies an
    endpoint-free feature view. This module never imports endpoint or fitting
    APIs, and uses the original 32-patient batching for old risk replay parity.
    """
    _require(int(record["seed"]) == int(seed) and int(record["fold"]) == int(fold)
             and record["stage"] == "prototype", "old cell identity/stage mismatch")
    fit, valid = np.asarray(ds.fit, int), np.asarray(ds.valid, int)
    _require(len(fit) and len(valid) and len(np.unique(fit)) == len(fit)
             and len(np.unique(valid)) == len(valid) and not np.intersect1d(fit, valid).size
             and np.array_equal(record["fit"], fit) and np.array_equal(record["valid"], valid),
             "feature partition mismatch")
    _require(set(record["arms"]) == set(ARMS), "incomplete frozen arm set")
    requested_device = torch.device(device)
    _require(ds.visual.device.type == requested_device.type
             and (requested_device.index is None or ds.visual.device.index == requested_device.index),
             "feature view/device mismatch")
    fit_features, valid_features, rows = {}, {}, {}
    for arm in ARMS:
        saved = record["arms"][arm]
        _require(int(saved["clinical_dim"]) == ds.C.shape[1], "old Clinical width mismatch")
        model = old_model.MultiQuery(ds.C.shape[1], arm).to(ds.visual.device)
        model.load_state_dict(saved["model_state"], strict=True)
        model.requires_grad_(False).eval()
        weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        ff, fr, fa = _extract_partition(model, ds, fit, seed, fold, ("MATCHED",))
        vf, vr, va = _extract_partition(model, ds, valid, seed, fold, CONTROLS)
        errors = {"FIT_MATCHED": _max_error(fr["MATCHED"], saved["fit_risk"], arm+" fit"),
                  "VALID_MATCHED": _max_error(vr["MATCHED"], saved["risk"], arm+" valid")}
        for control in ("RNA_MEAN_QUERY", "WRONG_RNA_QUERY", "VALUE_OFF"):
            errors[control] = _max_error(vr[control], saved["sensitivities"][control], arm+" "+control)
        _require(all(_same_bits(value, weights[key]) for key, value in model.state_dict().items()),
                 "frozen model state changed during extraction")
        if arm == "QUERY":
            _require(all(np.array_equal(vf[key], vf["MATCHED"]) for key in CONTROLS)
                     and all(np.array_equal(vr[key], vr["MATCHED"]) for key in CONTROLS),
                     "QUERY control unexpectedly changed disabled routing")
        fit_features[arm], valid_features[arm] = ff["MATCHED"], vf
        rows[arm] = {"old_risk_vectors_replayed": len(errors), "old_risk_errors": errors,
                     "fit": fa, "valid": va, "model_state_unchanged": True,
                     "QUERY_controls_exact_matched": arm == "QUERY"}
        del model
    return {"fit_features": fit_features, "valid_features": valid_features, "audit": {
        "status": "PASS_FROZEN_SLOT_EXTRACTION", "stage": "prototype", "source": record["source"],
        "seed": int(seed), "fold": int(fold), "fit_patients": len(fit), "valid_patients": len(valid),
        "feature_width": 256, "slot_shape": [8, 32], "aggregation": "equal_WSI_mean_then_flatten",
        "patient_batch_size": 32, "old_risk_tolerance": OLD_RISK_TOLERANCE,
        "distribution_tolerance": DISTRIBUTION_TOLERANCE,
        "old_risk_vectors_replayed": sum(row["old_risk_vectors_replayed"] for row in rows.values()),
        "maximum_old_risk_error": max(max(row["old_risk_errors"].values()) for row in rows.values()),
        "new_neural_training_steps": 0, "new_predictor_fits": 0, "endpoint_API_reads": 0,
        "permutation_seed": "270117 + fold*1000 + seed*10 + active_view_slide_index",
        "spatial_control": "both_matched_round_weights_frozen;Value_rows_unchanged",
        "overlap_definition": "soft_slot_overlap_Gram;no_topk_tie_or_medical_truth_claim",
        "fit_representation_crossfitted": False, "arms": rows}}

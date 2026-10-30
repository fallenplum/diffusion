"""Outcome-free partition builders; immutable FM inference may be shared.

Only partition-local clinical and measured-gene statistics are learned here.
Endpoint-bearing mappings are refused, including otherwise unused endpoints.
Patient tensors and preprocessing objects are private execution artifacts.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch

from .data import ProgramTransform, array_sha, clinical_design, fit_rows, read_membership


OUTCOMES = frozenset({"t", "e", "valid", "pfi_t", "pfi_e", "pfi_valid", "DSS", "OS", "PFI",
                      "time", "event", "risk", "offset"})
FROZEN_IMAGE_FIELDS = ("patch_values", "patch_aligned", "patch_mask", "patch_slide",
                       "titan_native", "titan_aligned", "titan_scores")


def require_no_outcomes(raw):
    if OUTCOMES.intersection(raw) or raw.get("audit", {}).get("endpoint_read", False):
        raise ValueError("ENDPOINT_BEARING_MAPPING_FORBIDDEN_IN_PHASE_A")
    for key in ("clinical_raw",):
        if OUTCOMES.intersection(raw.get(key, {})):
            raise ValueError("NESTED_ENDPOINT_FIELD_FORBIDDEN")


def molecular_memberships(raw, gmt):
    require_no_outcomes(raw)
    text = Path(gmt).read_text(encoding="utf-8-sig")
    names, rna = read_membership(text, raw["rna_symbols"])
    names2, cna = read_membership(text, raw["cna_symbols"])
    if names != names2:
        raise ValueError("MOLECULAR_PROGRAM_ORDER_MISMATCH")
    return names, rna, cna


def build_partition_inputs(raw, frozen, fit, evaluation, memberships, *, split_name):
    """Return one independent fit/evaluation partition with no outcome access.

    Frozen payload carries canonical ``global_rows`` and seven image tensors.
    The original P1 cache (409 rows) and all-patient FM-only cache both work.
    Extra rows not in this partition are never placed in the returned payload.
    """
    require_no_outcomes(raw)
    n = len(raw["folds"])
    tr = fit_rows(np.asarray(fit, dtype=np.int64), n)
    ev = np.asarray(evaluation, dtype=np.int64)
    if ev.ndim != 1 or len(np.unique(ev)) != len(ev) or (ev < 0).any() or (ev >= n).any():
        raise ValueError("INVALID_EVALUATION_ROWS")
    if np.intersect1d(tr, ev).size:
        raise ValueError("FIT_EVALUATION_OVERLAP")
    rows = np.r_[tr, ev]
    registry = np.asarray(frozen["global_rows"], np.int64)
    if len(np.unique(registry)) != len(registry):
        raise ValueError("DUPLICATE_FROZEN_IMAGE_ROWS")
    reverse = {int(g): i for i, g in enumerate(registry)}
    if any(int(g) not in reverse for g in rows):
        raise ValueError("FROZEN_IMAGE_SCOPE_INSUFFICIENT")
    take = torch.tensor([reverse[int(g)] for g in rows], dtype=torch.long)
    names, rm, cm = memberships
    if list(names) != list(frozen["program_names"]):
        raise ValueError("FROZEN_TEXT_RELATION_PROGRAM_ORDER_CHANGED")
    rna = np.where(raw["rna_measured"], raw["rna"], np.nan)
    cna = np.where(raw["cna_measured"], raw["cna"], np.nan)
    transform = ProgramTransform.fit(rna, cna, tr, rm, cm, min_coverage=.5, min_genes=5)
    molecular = transform.transform(rna[rows], cna[rows])
    clinical, clinical_state = clinical_design(raw["clinical_raw"], tr)
    image = {}
    for key in FROZEN_IMAGE_FIELDS:
        value = frozen["image"][key]
        image[key] = value[take] if value is not None else None
    image["clinical"] = torch.from_numpy(clinical[rows].astype(np.float32))
    from scripts.semantic_b0b7_prepare import coarse_stage
    clinical_raw = {k: np.asarray(v)[rows].astype(str) for k, v in raw["clinical_raw"].items()}
    payload = {"image": image, "omics": {
        "values": torch.from_numpy(molecular["values"]),
        "measured": torch.from_numpy(molecular["measured"])},
        "target": torch.from_numpy(molecular["values"][..., 0].copy()),
        "target_mask": torch.from_numpy(molecular["measured"][..., 0].copy()),
        "coverage": torch.from_numpy(molecular["coverage"]),
        "train_indices": torch.arange(len(tr)),
        "valid_indices": torch.arange(len(tr), len(rows)),
        "global_rows": torch.from_numpy(rows),
        "metadata": {"patient": rows.copy(),
            "stage": coarse_stage(clinical_raw["AJCC_PATHOLOGIC_TUMOR_STAGE"]),
            "cohort": np.asarray(raw["cancer"])[rows].astype(str),
            "missing_pattern": np.asarray(raw["platform"])[rows].astype(np.int64)},
        "clinical_raw": clinical_raw, "program_names": list(names),
        "provenance": {"schema": "semantic_partition_outcome_free_v1", "split_name": str(split_name),
            "train_n": len(tr), "evaluation_n": len(ev), "fit_rows_sha256": array_sha(tr),
            "evaluation_rows_sha256": array_sha(ev), "molecular_fit_hash": transform.rna_scale.train_hash,
            "clinical_fit_hash": clinical_state["fit_hash"], "outcomes_loaded": False,
            "frozen_images_train_fitted": False,
            "raw_container_sha256": raw["audit"]["cohort_sha256"],
            "parent_frozen_input_sha256": frozen.get("frozen_input_sha256")}}
    for key in ("conch_prototypes", "titan_prototypes", "relation_prior", "shuffled_relation_prior"):
        payload[key] = frozen[key].clone()
    return payload, {"program_transform": transform, "clinical_state": clinical_state,
                     "train_global_rows": tr.copy(), "evaluation_global_rows": ev.copy()}


def payload_tensor_hashes(payload):
    """Small anonymous receipts; do not publish original arrays/identifiers."""
    result = {}
    for section in ("image", "omics"):
        for key, value in payload[section].items():
            if value is not None:
                result[section + "." + key] = array_sha(value.detach().cpu().numpy())
    for key in ("target", "target_mask", "train_indices", "valid_indices"):
        result[key] = array_sha(payload[key].detach().cpu().numpy())
    return result

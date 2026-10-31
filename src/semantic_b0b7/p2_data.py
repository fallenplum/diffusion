"""Explicit DSS-only endpoints and immutable nested split planning for P2.

The Phase-A reader remains separate and refuses every endpoint-bearing mapping.
No PFI/OS value is loaded, selected, scored or passed to pretraining.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .p2_anchor import array_sha


ENDPOINT_SHA256 = "91b5d15aa32489c825a8341aed8d9733281dc62386151c8c6246b6312975fd4e"
GROUPS_SHA256 = "5b3d407cb6f7ce2e833393f3917e32472525e708dc2addb955c942a48f259df9"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def read_dss_only(path, canonical_ids):
    if file_sha(path) != ENDPOINT_SHA256:
        raise ValueError("PINNED_ENDPOINT_CONTAINER_CHANGED")
    with np.load(path, allow_pickle=False) as archive:
        patient_hash = str(archive["patient_hash"])
        if patient_hash != hashlib.sha256(np.ascontiguousarray(canonical_ids).tobytes()).hexdigest():
            raise ValueError("DSS_CANONICAL_PATIENT_ORDER_CHANGED")
        time = np.array(archive["DSS_time"], np.float64)
        event = np.array(archive["DSS_event"], np.float64)
        valid = np.array(archive["DSS_valid"], bool)
    if (time.shape != (512,) or event.shape != (512,) or valid.shape != (512,) or not valid.all()
            or not np.isfinite(time).all() or (time < 0).any()
            or not np.isin(event, (0, 1)).all() or event.sum() != 65):
        raise ValueError("DSS_CANONICAL_ELIGIBILITY_CHANGED")
    return {"t": time, "e": event, "valid": valid}, {
        "endpoint_sha256": ENDPOINT_SHA256, "arrays_read": ["patient_hash", "DSS_time", "DSS_event", "DSS_valid"],
        "DSS_events": 65, "patients": 512, "time_unit": "months", "PFI_OS_read": False}


def attach_legacy_union(raw, groups_path):
    if file_sha(groups_path) != GROUPS_SHA256:
        raise ValueError("LEGACY_EIGHT_PROGRAM_MEMBERS_CHANGED")
    groups = json.loads(Path(groups_path).read_text(encoding="utf-8-sig"))
    if len(groups) != 8:
        raise ValueError("EXPECTED_EIGHT_LEGACY_GROUPS")
    universe = set().union(*(set(v) for v in groups.values()))
    mask = np.asarray([str(g) in universe for g in raw["rna_symbols"]], bool)
    result = dict(raw)
    result["_direct_gene_universe"] = mask
    return result, {"legacy_group_sha256": GROUPS_SHA256, "external_union_genes": len(universe),
                    "canonical_union_genes": int(mask.sum()), "legacy_eight_program_union_sha256": array_sha(mask)}


def nested_split_plan(folds, event, *, seed=27001):
    f, e = np.asarray(folds), np.asarray(event)
    if f.shape != (512,) or e.shape != f.shape or set(f) != {1, 2, 3, 4, 5}:
        raise ValueError("CANONICAL_SPLIT_SCOPE_CHANGED")
    plan = {}
    for outer in range(1, 6):
        train, test = np.flatnonzero(f != outer), np.flatnonzero(f == outer)
        split = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed + outer)
        for inner, (a, b) in enumerate(split.split(train, e[train])):
            fit, ev = train[a], train[b]
            plan[f"outer{outer}_inner{inner}"] = {
                "outer": outer, "inner": inner, "role": "inner", "fit": fit.tolist(), "evaluation": ev.tolist(),
                "fit_sha256": array_sha(fit), "evaluation_sha256": array_sha(ev),
                "split_uses_only_outer_train_events": True, "teacher_seed": 27011 + outer * 100 + inner}
        plan[f"outer{outer}_final"] = {
            "outer": outer, "inner": None, "role": "outer_final", "fit": train.tolist(), "evaluation": test.tolist(),
            "fit_sha256": array_sha(train), "evaluation_sha256": array_sha(test),
            "teacher_seed": 27011 + outer * 100 + 31}
    return plan

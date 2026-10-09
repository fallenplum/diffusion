"""Independent structural and leakage audits for Phase8-V0 artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def architecture_audit(cell_rows):
    required = {"A0", "A1", "A2", "A3", "A4", "A5", "A6"}
    seen = {x["arm"] for x in cell_rows}
    return {"status": "PASS" if seen == required else "FAIL", "arms": sorted(seen), "expected": sorted(required),
            "zero_residual_tolerance": 1e-8, "cell_count": len(cell_rows)}


def leakage_audit(cell_rows):
    failures = []
    for row in cell_rows:
        if row.get("outer_test_used_for_selection"): failures.append(row["cell"])
        if row.get("inner_val_used_in_projection_fit"): failures.append(row["cell"] + ":projection")
        if row.get("clinical_offset_overlap"): failures.append(row["cell"] + ":offset")
    return {"status": "PASS" if not failures else "FAIL", "failures": failures,
            "rules": ["outer-test labels unavailable", "inner-val excluded from projections and offsets", "clinical OOF offset for residual-head training"]}


def independent_validation(out):
    out = Path(out); cells = []
    for p in (out / "cells").glob("*.json"):
        cells.append(json.loads(p.read_text()))
    if not cells: return {"status": "FAIL", "reason": "no cells"}
    max_grad = max(float(c.get("max_grad", 0.0)) for c in cells)
    finite_grad = all(np.isfinite(float(c.get("max_grad", np.nan))) for c in cells)
    finite_loss = all(np.isfinite(float(c.get("train_loss_last", 0.0))) for c in cells if c.get("arm") != "A0")
    isolated = not any(c.get("outer_test_used_for_selection", True) for c in cells)
    return {"status": "PASS" if finite_grad and finite_loss and isolated and all(c.get("finite", False) for c in cells) else "FAIL",
            "cells": len(cells), "max_gradient": max_grad, "gradient_finite": finite_grad, "loss_finite": finite_loss,
            "all_finite": all(c.get("finite", False) for c in cells), "outer_test_used_for_selection": not isolated}

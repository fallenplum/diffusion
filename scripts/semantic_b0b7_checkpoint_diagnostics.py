"""Read saved aggregate validation tensors for scale/route diagnostics; no fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--expected-input-sha256", required=True)
    p.add_argument("--aggregate", required=True, type=Path)
    p.add_argument("--private-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    assert sha(args.inputs) == args.expected_input_sha256 and not args.output.exists()
    aggregate = json.loads(args.aggregate.read_text())
    assert aggregate["input_sha256"] == args.expected_input_sha256
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
    valid = np.asarray(inputs["valid_indices"], int)
    truth = np.asarray(inputs["target"])[valid]
    measured = np.asarray(inputs["target_mask"])[valid]
    rows = []
    for cell in aggregate["cells"]:
        root = args.private_root / "cells" / (cell["arm"] + "_seed" + str(cell["seed"]))
        receipt = json.loads((root / "cell.private.json").read_text())
        selected = cell["selected_epoch"]
        info = next(x for x in receipt["states"] if x["epoch"] == selected)
        path = root / info["file"]
        assert sha(path) == info["sha256"]
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        assert snapshot["input_sha256"] == args.expected_input_sha256
        pred = snapshot["expected_validation"]["prediction"].numpy()
        ratios = []
        for k in range(50):
            keep = measured[:, k]
            sd = truth[keep, k].std()
            if keep.sum() >= 3 and sd > 1e-10:
                ratios.append(float(pred[keep, k].std() / sd))
        state = snapshot["model"]
        prior = state["relation_prior"].numpy()
        unknown = float(torch.sigmoid(state["unknown_logit"]))
        known_mass = float(prior.sum())
        unknown_mass = float((prior == 0).sum() * unknown)
        by_epoch = {x["epoch"]: x for x in receipt["history"]}
        rows.append({"arm": cell["arm"], "seed": cell["seed"], "selected_epoch": selected,
                     "selected_state_sha256": info["sha256"],
                     "median_prediction_to_target_std_ratio": float(np.median(ratios)),
                     "min_prediction_to_target_std_ratio": min(ratios),
                     "max_prediction_to_target_std_ratio": max(ratios),
                     "unknown_link_weight": unknown, "known_prior_mass": known_mass,
                     "unknown_total_mass_fraction": unknown_mass / (known_mass + unknown_mass),
                     "train_aux200": by_epoch[200]["train_auxiliary_loss"],
                     "train_aux400": by_epoch[400]["train_auxiliary_loss"],
                     "validation_aux200": by_epoch[200]["validation"]["auxiliary_loss"],
                     "validation_aux400": by_epoch[400]["validation"]["auxiliary_loss"],
                     "selected_validation_program_loss": cell["selected_validation"]["program_loss"],
                     "selected_validation_weighted_pair_loss": 0.1 * cell["selected_validation"]["pair_hard_loss"]})
    result = {"schema": "semantic_b0b7_saved_state_scale_route_audit_v1", "status": "COMPLETE_DESCRIPTIVE_ONLY",
              "input_sha256": args.expected_input_sha256, "source_aggregate_sha256": sha(args.aggregate),
              "source_sha256": sha(__file__), "rows": rows, "model_forwards": 0,
              "optimizer_updates": 0, "checkpoint_selection_changed": False,
              "limitations": ["route mass is not outcome contribution", "prediction scale is not proof of a unique failure cause", "ridge and neural decoders have different feature dimensions and regularization", "no DSS claims"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "cells": len(rows), "B4": [x for x in rows if x["arm"] == "B4"]}))


if __name__ == "__main__":
    main()

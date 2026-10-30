"""Real-data partition reproduction and held-out perturbation checks (zero fits)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_b0b7_prepare import legacy_data_module, file_sha, read_json
from src.semantic_b0b7.p1r_data import build_partition_inputs, molecular_memberships
from src.semantic_b0b7.data import array_sha


def main():
    p = argparse.ArgumentParser()
    for name in ("assets", "legacy-root", "parent-input", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    assets = read_json(a.assets)
    legacy, _ = legacy_data_module(a.legacy_root)
    raw = legacy.load_raw(assets, include_endpoints=False)
    parent = torch.load(a.parent_input, map_location="cpu", weights_only=False)
    rows = parent["global_rows"].numpy()
    train = rows[parent["train_indices"].numpy()]
    valid = rows[parent["valid_indices"].numpy()]
    memberships = molecular_memberships(raw, assets["gmt"])
    one, st = build_partition_inputs(raw, parent, train, valid, memberships, split_name="parent-reproduction")
    local = {int(g): i for i, g in enumerate(rows)}
    order = torch.tensor([local[int(g)] for g in one["global_rows"]])
    errors = {}
    for section in ("image", "omics"):
        for k, v in one[section].items():
            old = parent[section][k][order]
            err = float((v.float() - old.float()).abs().max())
            errors[section + "." + k] = err
    for k in ("target", "target_mask", "coverage"):
        errors[k] = float((one[k].float() - parent[k][order].float()).abs().max())
    # Repacking rows before FP32 program GEMM may change its final bit.
    # This is a numerical equivalence check, not a byte-identity claim.
    tolerance = 2e-7
    if max(errors.values()) > tolerance:
        raise ValueError("PARTITION_REPRODUCTION_NONEXACT:" + json.dumps(errors))
    # Mutate only held-out raw measurements/clinical. No endpoint is opened.
    changed = dict(raw)
    changed["rna"] = raw["rna"].copy()
    changed["cna"] = raw["cna"].copy()
    changed["rna"][valid] = 12345
    changed["cna"][valid] = -12345
    changed["clinical_raw"] = {k: v.copy() for k, v in raw["clinical_raw"].items()}
    changed["clinical_raw"]["AGE"][valid] = "999"
    two, st2 = build_partition_inputs(changed, parent, train, valid, memberships, split_name="perturbation")
    for section in ("image", "omics"):
        for k, v in one[section].items():
            if not torch.equal(v[:len(train)], two[section][k][:len(train)]):
                raise ValueError("HELDOUT_COVARIATE_CHANGED_FIT_INPUT")
    report = {"status": "PASS_REAL_NUMERICAL_PARTITION_EQUIVALENCE_AND_HELDOUT_STATISTICS_ISOLATION",
              "source_sha256": file_sha(__file__),
              "partition_builder_sha256": file_sha(ROOT / "src/semantic_b0b7/p1r_data.py"),
              "parent_input_sha256": file_sha(a.parent_input), "patients": len(rows),
              "fit_n": len(train), "evaluation_n": len(valid), "maximum_errors": errors,
              "comparison_absolute_tolerance": tolerance,
              "repacked_molecular_float32_bitwise_equal": max(errors.values()) == 0,
              "fit_rows_sha256": array_sha(train), "valid_rows_sha256": array_sha(valid),
              "new_neural_fits": 0, "optimizer_updates": 0, "outcomes_read": False,
              "heldout_mutation_changed_fit_tensors": False}
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()

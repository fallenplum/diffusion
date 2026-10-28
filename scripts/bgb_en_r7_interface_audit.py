"""Read-only R6 initial/20/40/80 interface audit on fixed training probes.

No optimizer, validation scores, held-out selection, or raw tensor export.
This is evidence for R7's preregistered scale question, not a causal repair.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_r6_common as legacy
from src.bgb_en.r6_training import saved_state, state_hash


def masked_scale(value, mask):
    if tuple(mask.shape) != tuple(value.shape[:-1]) or mask.dtype != torch.bool:
        raise ValueError("R7_AUDIT_INVALID_MASK")
    x = value.detach()[mask].double()
    if not len(x):
        return {"valid_vectors": 0, "coordinate_RMS": None, "mean_L2": None,
                "median_L2": None, "maximum_L2": None, "zero_vectors": 0}
    if not torch.isfinite(x).all():
        raise ValueError("R7_AUDIT_NONFINITE_VALID_STATE")
    norms = x.norm(dim=-1)
    return dict(valid_vectors=len(x), coordinate_RMS=float(x.square().mean().sqrt()),
                mean_L2=float(norms.mean()), median_L2=float(torch.quantile(norms, .5)),
                maximum_L2=float(norms.max()), zero_vectors=int((norms == 0).sum()))


def row_audit(model, indices):
    model.eval()
    with torch.no_grad():
        idx = torch.as_tensor(indices, device=model.device)
        row = model(idx, control="matched_query")
        genes = model.genes(idx)
        encoded = model.core.foundation.gene_reader.gene_encoder(genes)
        own = row["own_program_mask"]
        evidence = row["patch_base_read"]["program_mask"]
        fields = {
            "RNA_G0": (row["G0"], own),
            "Patch_values": (row["patch_values"], row["native_adapter"]["patch_mask"]
                             if "patch_mask" in row["native_adapter"] else
                             model.patch_bank.select(idx)[0]["patch_mask"]),
            "Base_query": (row["base_queries"], row["query_source_mask"]),
            "Conditional_query": (row["conditioned_queries"], row["query_source_mask"]),
            "Query_change": (row["conditioned_queries"] - row["base_queries"], row["query_source_mask"]),
            "Ebase": (row["Ebase"], evidence),
            "Econd": (row["Econd"], evidence),
            "delta_E": (row["delta_E"], evidence),
            "delta_G": (row["delta_G"], own),
            "Gene_keys": (encoded["keys"], encoded["gene_mask"]),
            "Gene_values": (encoded["values"], encoded["gene_mask"]),
        }
        reverse = row["reverse_cond_read"]
        if reverse is not None:
            fields["Reverse_query"] = (reverse["query"], reverse["program_mask"])
            fields["Reverse_evidence"] = (reverse["tokens"], reverse["program_mask"])
        stats = {name: masked_scale(v, m) for name, (v, m) in fields.items()}
        stats["cross_risk_SD"] = float(row["cross_delta"].std(unbiased=False))
        stats["main_risk_SD"] = float(row["main_delta"].std(unbiased=False))
        return stats


def main():
    legacy.configure_deterministic_runtime()
    parser = legacy.parser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ctx = legacy.context(args)
    qsha = legacy.validate_qualification(args, ctx)
    baseline = {str(p.relative_to(ROOT)): legacy.sha256(p)
                for group in ("bgb_en", "bgb_en_r5", "bgb_en_r6")
                for p in (ROOT / "reports" / group).rglob("*") if p.is_file()}
    rows, unique = [], set()
    for part in legacy.PARTITIONS:
        args.partition = part
        prepared, origin = legacy.qualified_prepared(args, ctx, part)
        selected = np.asarray(prepared["train"])[:8]
        unique.update(int(i) for i in selected)
        for seed in legacy.SEEDS:
            directory = args.private / "cells" / f"B7_p{part}_seed{seed}"
            complete = legacy.read(directory / "COMPLETE.json")
            artifact = directory / "model.pt"
            if legacy.sha256(artifact) != complete["artifact_sha256"]:
                raise ValueError("R7_AUDIT_R6_ARTIFACT_CHANGED")
            payload = torch.load(artifact, map_location="cpu", weights_only=False)
            model, _ = legacy.construct(args, ctx, prepared, "B7", seed)
            if state_hash(saved_state(model)) != payload["initialization"]["full_state_sha256"]:
                raise ValueError("R7_AUDIT_R6_INITIALIZATION_CHANGED")
            initial = row_audit(model, selected)
            rows.append(dict(partition=part, seed=seed, epoch=0, quantities=initial))
            for epoch in (20, 40, 80):
                model.load_state_dict(payload["snapshots"][epoch]["state_dict"], strict=True)
                rows.append(dict(partition=part, seed=seed, epoch=epoch,
                                 quantities=row_audit(model, selected)))
            del payload, model
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
    after = {str(p.relative_to(ROOT)): legacy.sha256(p)
             for group in ("bgb_en", "bgb_en_r5", "bgb_en_r6")
             for p in (ROOT / "reports" / group).rglob("*") if p.is_file()}
    if after != baseline:
        raise ValueError("R7_AUDIT_HISTORICAL_REPORT_CHANGED")
    result = dict(schema="bgb_en_r7_readonly_r6_interface_audit_v1",
                  status="PASS_9_R6_CELLS_36_STATE_FORWARDS", rows=rows,
                  rows_count=len(rows), actual_model_forwards=len(rows),
                  patients_per_partition=8, unique_training_probe_patients=len(unique),
                  repeated_probes_not_independent_cohorts=True, validation_scores_computed=False,
                  labels_used_for_probe_selection=False, new_optimizer_fits=0,
                  new_backward_calls=0, inherited_contract_sha256=ctx["lock"]["contract_sha256"],
                  qualification_sha256=qsha,
                  scope="fixed_first_eight_training_rows_per_partition_reused_across_seeds_and_states",
                  coordinate_RMS="sqrt_mean_valid_float32_elements_squared_accumulated_float64",
                  per_patient_values_coordinates_or_identifiers_exported=False,
                  interpretation="scale_trajectory_is_descriptive_not_causal_and_not_a_budget_selection")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ("status", "actual_model_forwards", "new_optimizer_fits")}))


if __name__ == "__main__":
    main()

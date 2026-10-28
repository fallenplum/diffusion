"""Real-data, zero-optimizer R7 qualification before the formal matrix.

Only current training patients determine interface scales and probes. Nine
unsupervised calibration receipts are separate from neural/teacher fitting.
No qualification model or optimizer state is saved.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_r7_common as common
from scripts import bgb_en_r6_common as legacy
from scripts.bgb_en_r6_replay import upstream_replay
from src.bgb_en.r6_pathology import source_model_state_sha256
from src.bgb_en.r6_training import SCALARS


def checks(model, indices, controls):
    model.eval()
    idx = torch.as_tensor(indices, device=model.device)
    with torch.no_grad():
        matched = model(idx, control="matched_query")
        again = model(idx, control="matched_query")
        error = max(float((matched[k] - again[k]).abs().max()) for k in SCALARS)
        common.require(error == 0, "R7_QUALIFICATION_REPEATED_FORWARD_CHANGED")
        invariants, responses = {}, {}
        for tag in controls:
            out = model(idx, control=tag)
            if tag in ("wrong_query", "wrong_path", "mean_query", "interaction_off"):
                preserved = all(torch.equal(matched[key], out[key]) for key in
                    ("patch_values", "gene_values", "gene_measured_mask", "main_delta", "offset"))
                common.require(preserved, "R7_QUALIFICATION_QUERY_REPLACED_OWN_EVIDENCE_OR_MAIN")
                invariants[tag] = preserved
            responses[tag] = dict(risk_max_absolute_change=float((matched["risk"] - out["risk"]).abs().max()),
                                  frozen_not_retrained=True)
        off = model(idx, control="interaction_off")
        common.require(torch.count_nonzero(off["cross_delta"]) == 0,
                       "R7_QUALIFICATION_OFF_NOT_EXACT_ADDITIVE")
    model.zero_grad(set_to_none=True)
    before = source_model_state_sha256(model.core.foundation)
    out = model(idx, control="matched_query")
    probe = torch.linspace(-1, 1, len(idx), device=model.device)
    (out["risk"] * probe).sum().backward()
    frozen = [p.grad for p in model.core.foundation.parameters()]
    common.require(all(g is None for g in frozen), "R7_QUALIFICATION_SOURCE_RECEIVED_TARGET_GRADIENT")
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            key = ".".join(name.split(".")[:2])
            group = groups.setdefault(key, dict(parameters=0, gradient_L2_squared=0., finite=True))
            group["parameters"] += parameter.numel()
            if parameter.grad is not None:
                group["finite"] &= bool(torch.isfinite(parameter.grad).all())
                group["gradient_L2_squared"] += float(parameter.grad.double().square().sum())
    common.require(all(g["finite"] for g in groups.values()), "R7_QUALIFICATION_NONFINITE_GRADIENT")
    model.zero_grad(set_to_none=True)
    common.require(before == source_model_state_sha256(model.core.foundation),
                   "R7_QUALIFICATION_MUTATED_SOURCE_STATE")
    return dict(repeated_forward_max_error=error, own_evidence_invariants=invariants,
                frozen_responses=responses, gradient_groups=groups,
                derivative_probe="fixed_training_row_linear_risk_function_no_outcomes",
                explicit_backward_passes=1, optimizer_updates=0,
                complete_model_forward_calls=len(controls) + 4,
                active_parameters=model.parameter_budget())


def main():
    common.configure_deterministic_runtime()
    torch.set_num_threads(2)
    args = common.parser(__doc__).parse_args()
    common.require(not (args.private / "QUALIFICATION.json").exists(),
                   "R7_EXISTING_QUALIFICATION_NOT_OVERWRITTEN")
    args.private.mkdir(parents=True, exist_ok=True)
    historical = common.public_snapshot_guard()
    try:
        ctx = common.context(args, locked=False)
        prepared, origins, upstream = {}, {}, []
        calibration_rows = []
        for part in common.PARTITIONS:
            args.partition = part
            prepared[part], origins[part] = common.qualified_prepared(args, ctx, part)
            upstream.append(dict(partition=part, **upstream_replay(ctx["r6_ctx"], prepared[part])))
            for seed in common.SEEDS:
                calibration = common.create_calibration_cache(args, ctx, prepared[part], seed)
                calibration_rows.append(dict(partition=part, seed=seed, calibration=calibration))
        upstream_audit = dict(status="PASS_R7_REUSED_UPSTREAM_39_COX_AND_3_BETA",
            rows=upstream, teacher_refits=0, source_refits=0, new_optimizer_fits=0,
            inherited_R6_contract_sha256=ctx["upstream_r6"]["contract_sha256"])
        common.write(args.private / "UPSTREAM_REPLAY_AUDIT.json", upstream_audit)
        args.partition = 0
        seed = 31
        selected = np.asarray(prepared[0]["train"])[:8]
        records = []
        raw_error = None
        common_init = None
        for arm in common.TARGET_ARMS:
            model, origin = common.construct(args, ctx, prepared[0], arm, seed)
            # Common main initialization, not a claim that trained main effects
            # remain equal after separate optimizers have learned.
            main = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                    if name.startswith(("hallmark.", "core.path_main.", "core.gene_main.",
                                        "core.hallmark_main.", "core.spatial_adapter."))}
            if common_init is None:
                common_init = main
            common.require(set(common_init) == set(main) and
                all(torch.equal(main[n], common_init[n]) for n in main),
                "R7_QUALIFICATION_COMMON_MAIN_INITIALIZATION_CHANGED")
            detail = checks(model, selected, common.frozen_controls_for_arm(arm))
            if arm == "RAW":
                ctx["r6_args"].partition = 0
                previous, _ = legacy.construct(ctx["r6_args"], ctx["r6_ctx"], prepared[0], "B7", seed)
                previous.eval()
                idx = torch.as_tensor(selected, device=args.device)
                with torch.no_grad():
                    a = model(idx, control="matched_query")
                    b = previous(idx, control="matched_query")
                raw_error = max(float((a[k] - b[k]).abs().max()) for k in SCALARS)
                common.require(raw_error == 0, "R7_RAW_NOT_EXACT_R6_INITIAL_FORWARD")
                del previous, a, b
            records.append(dict(arm=arm, **detail))
            del model
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        # A source LOCK is created only after all zero-optimizer eligibility
        # checks pass. It precedes every formal cell and binds all nine RMS
        # receipts, the final configuration and scientific source bytes.
        lock = common.create_lock(args, ctx)
        ctx["lock"] = lock
        common.require(historical == common.public_snapshot_guard(),
                       "R7_QUALIFICATION_HISTORICAL_REPORT_CHANGED")
        value = dict(schema="bgb_en_r7_real_data_qualification_v1",
            status="PASS_R7_REAL_DATA_AND_READONLY_R6_TRANSFER",
            contract_sha256=lock["contract_sha256"], records=records,
            baseline_main_initialization_equal=True, RAW_vs_R6_SCALARS_max_error=raw_error,
            upstream_replay_audit_sha256=common.sha256(args.private / "UPSTREAM_REPLAY_AUDIT.json"),
            calibrations=calibration_rows, distinct_train_RMS_calibrations=9,
            scalar_scales_per_calibration=6, calibration_fits_are_unsupervised_not_neural_fits=True,
            source_refits=0, teacher_refits=0, scalar_beta_refits=0,
            optimizer_instances=0, optimizer_updates=0, qualification_weights_saved=False,
            explicit_backward_passes=len(records), probe_patients=8,
            complete_model_forward_calls=sum(r["complete_model_forward_calls"] for r in records) + 2,
            scope="fixed_first_eight_inner0_training_patients_no_validation_selection",
            new_survival_performance_estimate=False, outer_test_access=False)
        common.write(args.private / "QUALIFICATION.json", value)
        # Publish an explicit summary only; never copy source origins carrying
        # private paths, train hashes/indices or raw calibration arrays.
        public = {k: v for k, v in value.items() if k not in ("calibrations",)}
        public["calibration_scope"] = "nine_partition_seed_training_only_global_RMS_sets"
        common.write(ROOT / "reports/bgb_en_r7/R7_0_REAL_DATA_QUALIFICATION.json", public)
        common.write(ROOT / "reports/bgb_en_r7/PROTOCOL_LOCK.json", lock)
        print(json.dumps(dict(status=value["status"], contract_sha256=lock["contract_sha256"],
                              optimizer_updates=0, formal_fits=0)))
    except BaseException as exc:
        failure = dict(status="R7_QUALIFICATION_FAILED_NO_FORMAL_TRAINING",
                       error_type=type(exc).__name__, optimizer_updates=0)
        common.write(args.private / "QUALIFICATION_FAILURE.json", failure)
        (args.private / "QUALIFICATION_FAILURE_PRIVATE.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

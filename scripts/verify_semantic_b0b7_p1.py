"""Independent JSON-arithmetic and byte-hash review of completed P1 artifacts.

No torch import, checkpoint deserialization, model forward, training, or raw
patient-array loading occurs. A checkpoint hash validates the producer's
replay record binding; it is not a second forward replay by this reviewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ("B3", "B4", "B6a", "B4_MAP_SHUFFLE")
SEEDS = (31, 47, 83)
EPOCHS = (50, 100, 200, 400)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identical(a, b, message):
    if json.dumps(a, sort_keys=True, allow_nan=False) != json.dumps(b, sort_keys=True, allow_nan=False):
        raise ValueError(message)


class Arithmetic:
    def __init__(self):
        self.scalar_checks = 0
        self.max_abs_error = 0.
        self.program_reports = 0
        self.support_reports = 0

    def close(self, actual, expected, *, tolerance=1e-12, label):
        self.scalar_checks += 1
        if actual is None or expected is None:
            if actual is not None or expected is not None:
                raise ValueError("NULL_SUPPORT_MISMATCH:" + label)
            return
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError("NONFINITE_ARITHMETIC:" + label)
        error = abs(float(actual)-float(expected))
        self.max_abs_error = max(self.max_abs_error, error)
        if error > tolerance:
            raise ValueError("ARITHMETIC_MISMATCH:" + label)

    def program(self, metric):
        self.program_reports += 1
        programs = metric["per_program"]
        if len(programs) != 50 or {x["program_index"] for x in programs} != set(range(50)):
            raise ValueError("PROGRAM_INDEX_OR_SUPPORT_CHANGED")
        n = metric["patients"]
        supported = []
        for field in ("pearson", "spearman"):
            values = []
            for row in programs:
                value = row[field]
                if row["patients"] < 0 or row["patients"] > n:
                    raise ValueError("PROGRAM_PATIENT_COUNT_INVALID")
                if value is not None:
                    if row["patients"] < 3 or not math.isfinite(value) or abs(value) > 1+1e-12:
                        raise ValueError("CORRELATION_WITHOUT_SUPPORT")
                    values.append(value)
            expected = math.fsum(values)/len(values) if values else None
            self.close(metric["macro_program_"+field], expected, label="macro_"+field)
            if field == "pearson":
                supported = values
        self.close(metric["valid_pearson_programs"], len(supported), label="supported_programs")
        total = sum(x["patients"] for x in programs)
        self.close(metric["measured_target_cells"], total, label="program_cell_coverage")
        self.close(metric["coverage_fraction"], total/(n*50) if n else 0., tolerance=1e-12, label="program_coverage_fraction")
        if not 0 <= metric["patients_with_measurement"] <= n:
            raise ValueError("PATIENT_MEASUREMENT_COVERAGE_INVALID")
        h = metric["patient_equal_masked_huber"]
        if h is not None and (not math.isfinite(h) or h < 0):
            raise ValueError("MASKED_HUBER_INVALID")

    def support(self, support):
        self.support_reports += 1
        pairs = support["candidate_negative_pairs"]
        for axis, key in (("image_to_gene", "image_anchors_used"), ("gene_to_image", "gene_anchors_used")):
            row, used = support[axis], support[key]
            self.close(row["anchors_used"], used, label="direction_anchor_count")
            if used:
                self.close(row["negative_count_mean_on_used"], pairs/used, label="mean_actual_negatives")
                self.close(row["denominator_candidates_mean_on_used"], pairs/used+1, label="actual_denominator_candidates")
                if not 1 <= row["negative_count_min_on_used"] <= row["negative_count_max_on_used"]:
                    raise ValueError("ACTUAL_NEGATIVE_RANGE_INVALID")
            elif any(row[k] is not None for k in ("negative_count_mean_on_used", "negative_count_min_on_used", "negative_count_max_on_used", "denominator_candidates_mean_on_used")):
                raise ValueError("EMPTY_DIRECTION_SUPPORT_MUST_REMAIN_NULL")

    def validation(self, metric):
        self.program(metric["program"])
        self.support(metric["actual_pair_support"])
        self.close(metric["auxiliary_loss"], metric["program_loss"]+.1*metric["pair_hard_loss"],
                   tolerance=5e-7, label="validation_float32_auxiliary_sum")
        self.close(metric["program_loss"], metric["program"]["patient_equal_masked_huber"],
                   tolerance=5e-7, label="validation_huber_metric_loss_consistency")
        for retrieval in (metric["retrieval_hard"], metric["retrieval_random"]):
            if not retrieval["positive_identity_at_evaluation"] or not retrieval["validation_gene_bank_used_only_for_retrieval_scoring"]:
                raise ValueError("VALIDATION_RETRIEVAL_SCOPE_CHANGED")
            for axis in ("image_to_gene", "gene_to_image"):
                row = retrieval[axis]
                self.close(row["coverage"], row["anchors_covered"]/row["anchors"], label="retrieval_coverage")
                for key in ("top1_tie_averaged", "mean_reciprocal_midrank"):
                    value = row[key]
                    if row["anchors_covered"] == 0 and value is not None:
                        raise ValueError("UNSUPPORTED_RETRIEVAL_MUST_STAY_NULL")
                    if value is not None and not 0 <= value <= 1:
                        raise ValueError("RETRIEVAL_RANGE_INVALID")
        for groups in metric["strata"].values():
            self.close(sum(g["program"]["patients"] for g in groups.values()), metric["program"]["patients"], label="stratum_patient_coverage")
            for group in groups.values():
                self.program(group["program"])


def main(args):
    base, repository = Path(args.private_root), Path(args.repository_root)
    report_path = base/"P1_REPORT_400.private.json"
    gate_path, profile_path = base/"P0_P1_GATE_v2.json", base/"PROFILE_v2.json"
    extension_path, queue_path = base/"P1_EXTENSION_GATE_400.private.json", base/"RUN_STATE_400.private.json"
    report, gate, profile, extension, queue = [read(x) for x in (report_path, gate_path, profile_path, extension_path, queue_path)]
    if report["schema"] != "semantic_b0b7_p1_public_aggregate_v1" or report["status"] != "P1_EVALUATED":
        raise ValueError("PRODUCER_REPORT_SCHEMA_OR_STATUS")
    binding = {key: report[key] for key in ("input_sha256", "gate_sha256", "frozen_sources")}
    if digest(args.inputs) != binding["input_sha256"] or digest(gate_path) != binding["gate_sha256"]:
        raise ValueError("INPUT_OR_GATE_BYTE_HASH_CHANGED")
    if gate["status"] != "PASS" or gate["P2_DSS_authorized"] or gate["scientific_mechanism_supported"]:
        raise ValueError("P0_GATE_SCOPE_INVALID")
    identical(gate["frozen_sources"], binding["frozen_sources"], "GATE_SOURCE_SET_CHANGED")
    for rel, expected in binding["frozen_sources"].items():
        path = repository/rel
        if not path.resolve().is_relative_to(repository.resolve()) or digest(path) != expected:
            raise ValueError("FROZEN_SOURCE_BYTE_HASH_CHANGED")
    for key, value in binding.items():
        identical(profile[key], value, "PROFILE_BINDING_CHANGED")
        identical(extension[key], value, "EXTENSION_BINDING_CHANGED")
    if (profile["status"] != "PASS" or profile["optimizer_steps"] != 0
            or not profile["actual_forward_backward"] or not profile["all_parameter_gradients_finite_nonzero"]):
        raise ValueError("PROFILE_IS_NOT_A_ZERO_OPTIMIZER_QUALIFICATION")
    if queue["active"] or queue["failed"] or len(queue["completed"]) != 12 or queue["DSS_fits"] != 0:
        raise ValueError("QUEUE_NOT_COMPLETE_WITHOUT_FAILURE")
    if any(report[x] for x in ("P2_started", "dss_models_fitted", "dss_scores_selected", "all_twelve_extended", "budget_limited")):
        raise ValueError("UNEXPECTED_P2_OR_EXTENSION_STATUS")
    cells = report["cells"]
    if {(x["arm"],x["seed"]) for x in cells} != {(a,s) for a in ARMS for s in SEEDS} or len(cells) != 12:
        raise ValueError("TWELVE_UNIQUE_NEURAL_CELLS_REQUIRED")
    audit = Arithmetic()
    state_hashes, receipt_hashes, selected = [], [], []
    common_support = {}
    updates, votes, scalar_gradient_checks = 0, 0, 0
    for public in cells:
        arm, seed = public["arm"], public["seed"]
        directory = base/"P1_fits_v1/cells"/(arm+"_seed"+str(seed))
        path = directory/"cell.private.json"
        receipt = read(path)
        if (receipt["schema"] != "semantic_b0b7_p1_cell_v1" or receipt["status"] != "COMPLETE"
                or receipt["arm"] != arm or receipt["seed"] != seed or receipt["optimizer_steps"] != 400
                or receipt["phase"] != "P1_ONLY" or receipt["dss_or_other_outcomes_loaded"]):
            raise ValueError("NEURAL_RECEIPT_SCOPE_OR_IDENTITY")
        for key, value in binding.items():
            identical(receipt[key], value, "NEURAL_RECEIPT_BINDING_CHANGED")
        if receipt["profile_sha256"] != digest(profile_path) or digest(path) != public["private_receipt_sha256"]:
            raise ValueError("NEURAL_RECEIPT_OR_PROFILE_HASH_CHANGED")
        identical(receipt["runtime"], profile["runtime"], "RUNTIME_DETERMINISM_CHANGED")
        receipt_hashes.append({"arm":arm,"seed":seed,"sha256":digest(path)})
        states = receipt["states"]
        if len(states) != 4 or {x["epoch"] for x in states} != set(EPOCHS):
            raise ValueError("FOUR_SAVED_CHECKPOINTS_REQUIRED")
        for state in states:
            artifact = directory/state["file"]
            if (artifact.resolve().parent != directory.resolve() or digest(artifact) != state["sha256"]
                    or not state["actual_replay"] or state["replay_max_error"] != 0):
                raise ValueError("CHECKPOINT_HASH_OR_PRODUCER_REPLAY_RECORD_CHANGED")
            state_hashes.append({"arm":arm,"seed":seed,"epoch":state["epoch"],"sha256":state["sha256"]})
        history = receipt["history"]
        if len(history) != 400 or [x["epoch"] for x in history] != list(range(1,401)):
            raise ValueError("EXACT_400_UPDATES_REQUIRED")
        if sum(x["optimizer_updates_this_invocation"] for x in receipt["invocations"]) != 400:
            raise ValueError("INVOCATION_UPDATE_LEDGER_CHANGED")
        supports = []
        checkpoints = {}
        for row in history:
            audit.close(row["optimizer_steps"], row["epoch"], label="actual_update_index")
            audit.close(row["train_auxiliary_loss"], row["train_program_loss"]+.1*row["train_pair_loss"],
                        tolerance=5e-7, label="training_float32_auxiliary_sum")
            audit.support(row["actual_pair_support"])
            supports.append({"epoch":row["epoch"], **row["actual_pair_support"]})
            if set(row["gradient_norms"]) != set(row["weight_l2"]):
                raise ValueError("PARAMETER_TRACE_SUPPORT_CHANGED")
            for numbers in (row["gradient_norms"],row["weight_l2"]):
                for value in numbers.values():
                    scalar_gradient_checks += 1
                    if value is None or not math.isfinite(value) or value < 0:
                        raise ValueError("MISSING_OR_NONFINITE_PARAMETER_TRACE")
            if "validation" in row:
                checkpoints[row["epoch"]] = row["validation"]
                audit.validation(row["validation"])
        if set(checkpoints) != set(EPOCHS):
            raise ValueError("CHECKPOINT_VALIDATION_SET_CHANGED")
        expected_epoch = min(checkpoints,key=lambda x:(checkpoints[x]["auxiliary_loss"],x))
        if public["selected_epoch"] != expected_epoch or public["selection_rule"] != "validation_Laux_then_lower_epoch_tie":
            raise ValueError("AUXILIARY_CHECKPOINT_SELECTION_RULE_CHANGED")
        identical(public["selected_validation"],checkpoints[expected_epoch],"SELECTED_VALIDATION_CHANGED")
        identical(public["checkpoint_validation"],{str(k):v for k,v in checkpoints.items()},"PUBLIC_CHECKPOINT_VALIDATION_CHANGED")
        identical(public["actual_train_pair_support_by_epoch"],supports,"PUBLIC_ACTUAL_SUPPORT_CHANGED")
        if seed in common_support:
            identical(supports,common_support[seed],"EFFECTIVE_NEGATIVE_SUPPORT_DIFFERS_BETWEEN_ARMS")
        else:
            common_support[seed]=supports
        if arm == "B6a":
            if receipt["pairing_audit"]["self_or_same_patient_pairs"] != 0:
                raise ValueError("WRONG_POSITIVE_HAS_IDENTITY_MATCH")
        elif not receipt["pairing_audit"].get("identity_positives"):
            raise ValueError("IDENTITY_ARM_PAIR_ASSIGNMENT_CHANGED")
        votes += int(checkpoints[400]["auxiliary_loss"] < checkpoints[200]["auxiliary_loss"])
        updates += receipt["optimizer_steps"]
        selected.append({"arm":arm,"seed":seed,"selected_epoch":expected_epoch,
                         "validation_Laux":checkpoints[expected_epoch]["auxiliary_loss"],
                         "program_macro_Pearson":checkpoints[expected_epoch]["program"]["macro_program_pearson"],
                         "program_masked_Huber":checkpoints[expected_epoch]["program"]["patient_equal_masked_huber"]})
    base_dir=base/"P1_fits_v1/baselines"
    baseline_path=base_dir/"baselines.private.json"
    baseline=read(baseline_path)
    for key,value in binding.items():
        identical(baseline[key],value,"BASELINE_BINDING_CHANGED")
    if (baseline["decoder_models_fitted"]!=3 or baseline["scalar_program_linear_system_solves"]!=150
            or baseline["median_optimizer_fits"]!=0 or baseline["dss_or_other_outcomes_loaded"]
            or digest(base_dir/"ridge_states.private.pt")!=baseline["private_artifact_sha256"]):
        raise ValueError("RIDGE_COST_OR_ARTIFACT_CHANGED")
    identical(report["baseline_metrics"],baseline["metrics"],"PUBLIC_BASELINE_METRICS_CHANGED")
    for metric in baseline["metrics"].values():
        audit.program(metric)
    if (votes != extension["improved200to400_cells"] or votes != 0 or extension["extension_required"]
            or extension["complete_cells"]!=12 or report["common_extension"]!=extension):
        raise ValueError("COMMON_800_EXTENSION_RULE_CHANGED")
    recorded_400={(x["arm"],x["seed"]):x["sha256"] for x in extension["completed400_receipts"]}
    if recorded_400!={(x["arm"],x["seed"]):x["sha256"] for x in receipt_hashes}:
        raise ValueError("COMMON_400_RECEIPT_HASH_SET_CHANGED")
    if report["optimizer_steps"]!=updates or updates!=4800 or report["actual_checkpoint_replays"]!=len(state_hashes) or len(state_hashes)!=48:
        raise ValueError("COST_OR_REPLAY_LEDGER_CHANGED")
    result={"schema":"semantic_b0b7_independent_p1_report_arithmetic_hash_audit_v1",
            "date":"2026-10-31","status":"PASS_INDEPENDENT_REPORT_ARITHMETIC_AND_BYTE_HASH_AUDIT",
            "scope":"COMPLETED_P1_ONLY_NOT_A_SECOND_FORWARD_REPLAY_OR_NEW_TRAINING",
            "producer_report_sha256":digest(report_path),"input_sha256":binding["input_sha256"],
            "gate_sha256":digest(gate_path),"profile_sha256":digest(profile_path),
            "extension_gate_sha256":digest(extension_path),"queue_sha256":digest(queue_path),
            "frozen_sources":binding["frozen_sources"],"frozen_source_files_verified":len(binding["frozen_sources"]),
            "reviewer_source_sha256":digest(__file__),"neural_cell_receipts_verified":len(cells),
            "checkpoint_files_sha256_verified":len(state_hashes),"producer_actual_replays_recorded":48,
            "producer_replay_max_error":0.,"producer_optimizer_updates_verified":updates,
            "ridge_decoder_families":3,"scalar_program_linear_system_rhs":150,
            "ridge_factorizations_with_shared_measured_masks":baseline["linear_solver_calls_factoring_shared_masks"],
            "recorded_failures":len(queue["failed"]),"improved200to400_cells":votes,
            "800_extension_required":False,"all_twelve_common_support_verified":True,
            "source_input_gate_profile_bindings_verified":True,
            "B6a_own_targets_contract":"same full input byte hash and static frozen runner target/mask path; only positive assignment changes; no new per-step target-array reconstruction by reviewer",
            "selected_checkpoints":selected,"receipt_sha256":receipt_hashes,"checkpoint_sha256":state_hashes,
            "arithmetic":{"scalar_checks":audit.scalar_checks,"max_abs_error":audit.max_abs_error,
                          "program_metric_reports":audit.program_reports,"actual_support_reports":audit.support_reports,
                          "finite_parameter_trace_values":scalar_gradient_checks,
                          "aggregate_correlation_tolerance":1e-12,"float32_loss_sum_tolerance":5e-7},
            "independent_model_forwards":0,"independent_optimizer_updates":0,
            "checkpoint_arrays_deserialized":False,"raw_patient_arrays_loaded":False,
            "checkpoint_sha_streaming_reads_only":True,"patient_level_files_downloaded":False,
            "DSS_models_fitted":0,"DSS_scores_selected":0,"P2_started":False,
            "old95_P0_risk_replays_in_this_cost_ledger":False,"P0_zero_step_profiles_counted_as_training":False,
            "limits":["This verifies hashes, producer replay records, and report arithmetic; it is not another set of actual checkpoint forwards.",
                      "Macro program Pearson/Spearman and coverage are independently reconstructed from persisted per-program scalars; raw prediction correlations and patient-level Huber are not recomputed from arrays.",
                      "A positive matched retrieval gap alone does not establish patient-specific information; compare against exact per-anchor candidate-size chance, not 1/(mean negative count+1).",
                      "No400to800 extension under the locked rule is not proof that every parameter or optimizer converged.",
                      "Inner validation selected checkpoints; this is auxiliary development evidence, not independent external or DSS evidence."]}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status":result["status"],"neural_receipts":len(cells),"checkpoint_hashes":len(state_hashes),
                      "optimizer_updates":updates,"arithmetic":result["arithmetic"],"sha256":digest(output)},sort_keys=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-root",required=True)
    parser.add_argument("--repository-root",required=True)
    parser.add_argument("--inputs",required=True)
    parser.add_argument("--output",required=True)
    main(parser.parse_args())

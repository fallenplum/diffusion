"""S5-A complete-matrix metadata report; no models/arrays are deserialized.

This is a separate reporting program, not one of the 13 frozen training files.
Primary comparisons and the 2000-donor-bootstrap recipe are fixed in code before
formal outcomes are read. Nothing here selects an epoch or a research model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import spgel_source_report as old

require, read_json, canonical = old.require, old.read_metadata, old.canonical_sha256
ARMS, SEEDS = tuple(f"M{i}" for i in range(7)), (31, 47)
SIZES = (1984, 4119, 1091, 1827, 932, 3058)
PARAMETERS = dict(M0=300435, M1=333203, M2=333203, M3=316819,
                  M4=316819, M5=333203, M6=333203)
PRIMARY = ("M1", "M5")
REFERENCES = ("M1", "M5", "M0", "M3", "M4", "M6")
BOOTSTRAPS, BOOTSTRAP_SEED = 2000, 510531
PROGRAMS = ("iCMS2", "iCMS3", "FAP_fibroblast_associated_top25", "MARCO_macrophage_associated_top25")
SOURCE_FILES = (
    "src/spgel_s5/__init__.py", "src/spgel_s5/models.py", "src/spgel_s5/sampling.py",
    "src/spgel_s5/losses.py", "src/spgel_s5/execution.py", "src/spgel_s5/diagnostics.py",
    "configs/spgel_s5/mechanism_v1.json", "scripts/spgel_s5.py", "scripts/spgel_s5_schedule.py",
    "tests/test_spgel_s5_models.py", "tests/test_spgel_s5_sampling.py",
    "tests/test_spgel_s5_execution.py", "tests/test_spgel_s5_diagnostics.py")
NAMES = ("S5A_RESULTS.json", "S5A_RESULTS.md", "S5A_PER_DONOR.csv", "S5A_PROGRAM_METRICS.csv",
         "S5A_PAIRED_COMPARISONS.csv", "S5A_DECISION.json", "S5A_ACTUAL_LEDGER.json",
         "S5A_TRAINING_DIAGNOSTICS.json")


def sha(path):
    """Stream bytes only: .pt/.npz contents never enter a decoder."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "S5_REPORT_ARTIFACT_MISSING_OR_SYMLINK")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            result.update(block)
    return result.hexdigest()


def number(value, code="S5_INVALID_FINITE_NUMBER", **kwargs):
    return old.finite_number(value, code, **kwargs)


def strict_mean(values):
    values = list(values)
    result = old.mean_defined(values)
    result["mean"] = result["mean"] if result["defined_count"] == len(values) else None
    result["requires_all_registered_units_defined"] = True
    return result


def check_config(config):
    require(config.get("schema") == "spgel_s5_mechanism_recipe_v1"
        and config.get("arms") == list(ARMS) and config.get("seeds") == list(SEEDS)
        and config.get("heldout_donors") == list(range(6)), "S5_REPORT_REGISTERED_MATRIX_CHANGED")
    require(config.get("budget") == dict(new_neural_fits=84, optimizer_updates=72800,
        saved_state_replays=252, automatic_retry_fits=0, M7_M8_fits_in_this_lock=0),
        "S5_REPORT_REGISTERED_BUDGET_CHANGED")
    require(config.get("training") == dict(epochs=40, batch_size=512, optimizer="Adam",
        learning_rate=.001, weight_decay=.0001, gradient_clip_norm=5., save_epochs=[10,20,40],
        primary_epoch=40, drop_last=False, heldout_model_or_epoch_selection=False),
        "S5_REPORT_FIXED_EPOCH_OR_TRAINING_CHANGED")
    require(config.get("loss") == dict(program_Huber=1., tissue_CE=.1, bidirectional_ITC=.1,
        temperature=.1, both_anchor_directions_in_all_arms=True), "S5_REPORT_LOSS_CHANGED")
    require(config.get("scope", {}).get("DSS_PFI_TCGA_or_R8_loaded") is False
        and config["scope"].get("old_checkpoint_bitwise_training_protocol_identity_claimed") is False
        and config["evaluation"].get("new_independent_cohort_claimed") is False,
        "S5_REPORT_SCOPE_CHANGED")


def source_assets(assets_path, binding):
    assets = read_json(assets_path)
    require(set(assets) == {"cache_root", "fresh_source_receipt"}, "S5_REPORT_SOURCE_ONLY_ASSETS")
    root, receipt = Path(assets["cache_root"]), Path(assets["fresh_source_receipt"])
    paths = {"manifest": root/"manifest.private.json", "arrays": root/"source_arrays.private.npz",
             "counts": root/"gene_counts.private.npz", "qualification": receipt}
    expected = dict(manifest=binding["cache_manifest_sha256"], arrays=binding["source_arrays_sha256"],
                    counts=binding["gene_counts_sha256"], qualification=binding["fresh_requalification_sha256"])
    require({key:sha(path) for key,path in paths.items()} == expected, "S5_REPORT_SOURCE_BYTES_CHANGED")
    fresh = read_json(receipt)
    require(fresh.get("status") == "PASS_FRESH_RAW_FULL_LIBRARY_AND_EXISTING_CACHE_REPLAY"
        and fresh.get("source_cache_manifest_sha256") == expected["manifest"]
        and fresh.get("DSS_or_PFI_read") is False and fresh.get("original_ST_test_read") is False
        and fresh.get("source_files_mutated") is False, "S5_REPORT_SOURCE_QUALIFICATION_CHANGED")
    return dict(current_source_file_hashes=expected, asset_pointer_sha256=sha(assets_path),
                source_arrays_deserialized=False)


def validate_lock(private, config_path, assets_path, *, root=ROOT):
    private, root = Path(private), Path(root)
    config, lock = read_json(config_path), read_json(private/"S5_RUN_LOCK.json")
    check_config(config)
    require(lock.get("schema") == "spgel_s5_run_lock_v1"
        and lock.get("status") == "LOCKED_BEFORE_84_NEW_FITS"
        and lock.get("config_sha256") == sha(config_path)
        and lock.get("contract_sha256") == canonical({k:v for k,v in lock.items() if k!="contract_sha256"})
        and lock.get("budget") == config["budget"] and lock.get("no_survival_or_R8_inputs") is True
        and lock.get("source_development_donors_not_new_external_confirmation") is True,
        "S5_REPORT_LOCK_INVALID")
    require(set(lock.get("sources", {})) == set(SOURCE_FILES), "S5_REPORT_EXACT_13_SOURCES_REQUIRED")
    for name, expected in lock["sources"].items():
        require(sha(root/name) == expected, "S5_REPORT_FROZEN_SOURCE_CHANGED:"+name)
    historical = read_json(root/"reports/spgel/SOURCE_RUN_LOCK.json")
    reference = read_json(root/config["reference_source_recipe"])
    require(lock.get("inherited_sources") == historical["sources"]
        and lock.get("reference_source_binding") == reference["source_binding"]
        and tuple(reference["program_order"]) == PROGRAMS, "S5_REPORT_HISTORICAL_BINDING_CHANGED")
    for name, expected in lock["inherited_sources"].items():
        require(sha(root/name) == expected, "S5_REPORT_HISTORICAL_SCIENCE_CHANGED:"+name)
    require(sha(private/"QUALIFICATION.private.json") == lock["qualification_sha256"],
            "S5_REPORT_REAL_QUALIFICATION_CHANGED")
    qualification = read_json(private/"QUALIFICATION.private.json")
    require(qualification.get("status") == "PASS_REAL_DATA_IMAGE_ONLY_AND_ALL_EFFECTIVE_PAIRED_PATHS"
        and qualification.get("new_optimizer_fits") == qualification.get("optimizer_instances")
            == qualification.get("optimizer_updates") == 0, "S5_REPORT_ZERO_FIT_QUALIFICATION_REQUIRED")
    profiles = {x["arm"]:x for x in qualification["profiles"]}
    require(len(qualification["profiles"]) == len(profiles) == 7 and set(profiles) == set(ARMS),
            "S5_REPORT_ALL_7_QUALIFIED_ARMS_REQUIRED")
    for arm, profile in profiles.items():
        require(profile["parameters"] == PARAMETERS[arm] and profile["state_unchanged"] is True
            and profile["optimizer_instances"] == profile["optimizer_updates"] == 0,
            "S5_REPORT_QUALIFIED_EFFECTIVE_CAPACITY_CHANGED")
        for direction in ("gene_to_image", "image_to_gene"):
            value = profile["query_gradient_norms"][direction]
            disabled = arm == "M0" or (arm == "M3" and direction == "image_to_gene") or (
                arm == "M4" and direction == "gene_to_image")
            require(value is None if disabled else number(value, minimum=0) > 0,
                    "S5_REPORT_QUALIFIED_QUERY_PATH_GRADIENT_CHANGED")
    partitions = {}
    for part in lock["partitions"]:
        key = part["donor"], part["seed"]
        require(key not in partitions and key in {(d,s) for d in range(6) for s in SEEDS},
                "S5_REPORT_DUPLICATE_OR_UNKNOWN_PARTITION")
        donor, seed = key
        require(part["train_rows"] == 13011-SIZES[donor] and part["heldout_rows"] == SIZES[donor]
            and part["updates_per_fit"] == 40*math.ceil(part["train_rows"]/512),
            "S5_REPORT_PARTITION_ROW_OR_UPDATE_BUDGET_CHANGED")
        require(sha(private/"qualification"/f"donor{donor}_seed{seed}.private.npz") == part["mapping_sha256"],
                "S5_REPORT_TRAIN_MAPPING_CHANGED")
        pair, member = part["pairing_audit"], part["membership_audit"]
        require(pair["train_rows_sha256"] == part["train_rows_sha256"]
            and pair["training_rows"] == part["train_rows"] and pair["seed"] == seed
            and pair["all_arms_share_supported_anchors"] is True
            and pair["regression_and_tissue_truth_unchanged"] is True
            and pair["same_tissue_and_train_depth_quartile"] is True
            and pair["heldout_metadata_used"] is False
            and pair["candidate_gene_values_or_program_targets_used"] is False
            and sum(x["training_rows"] for x in pair["groups"]) == part["train_rows"]
            and sum(x["ITC_active_rows"] for x in pair["groups"]) == pair["ITC_active_rows"],
            "S5_REPORT_MATCHED_PAIR_SUPPORT_CONTRACT_CHANGED")
        require(member["train_rows_sha256"] == part["train_rows_sha256"]
            and member["seed"] == seed and member["heldout_measurements_used"] is False
            and member["source_arrays_mutated"] is False
            and member["values_gene_IDs_outputs_and_medical_truth_columns_permuted"] is False
            and member["signed_row_sizes_exactly_preserved"] is True
            and member["all_signed_pairwise_overlaps_exactly_preserved"] is True
            and member["permutation_keeps_coverage_mean_sd_bins"] is True
            and member["changed_signed_membership_cells"] > 0,
            "S5_REPORT_MEMBERSHIP_NULL_IS_NOT_TARGET_RELABELING")
        partitions[key] = part
    require(len(partitions) == 12, "S5_REPORT_EXACT_12_PARTITIONS_REQUIRED")
    return config, lock, partitions, source_assets(assets_path, reference["source_binding"])


def name(key):
    arm, donor, seed = key
    return f"{arm}_d{donor}_seed{seed}"


def validate_cell(cell, replay, key, part, lock, folder, complete_sha):
    arm, donor, seed = key
    require(type(cell.get("donor")) is int and type(cell.get("seed")) is int
        and (cell.get("arm"), cell.get("donor"), cell.get("seed")) == key
        and cell.get("schema") == "spgel_s5_completed_cell_v1"
        and cell.get("status") == "COMPLETE_ONE_LOCKED_S5_FIT", "S5_REPORT_COMPLETE_IDENTITY_CHANGED")
    require(cell.get("contract_sha256") == lock["contract_sha256"]
        and cell.get("config_sha256") == lock["config_sha256"]
        and cell.get("mapping_sha256") == part["mapping_sha256"]
        and cell.get("train_rows_sha256") == part["train_rows_sha256"]
        and cell.get("heldout_rows_sha256") == part["heldout_rows_sha256"]
        and cell.get("train_rows") == part["train_rows"] and cell.get("heldout_rows") == part["heldout_rows"],
        "S5_REPORT_CELL_LINEAGE_CHANGED")
    require(cell.get("actual_neural_fits") == 1 and cell.get("actual_optimizer_updates") == part["updates_per_fit"]
        and cell.get("actual_saved_states") == 3 and cell.get("automatic_retry_fits") == 0
        and cell.get("wrong_pairing_changes_main_targets") is False
        and cell.get("image_only_inference_uses_RNA") is False and cell.get("new_external_confirmation") is False,
        "S5_REPORT_FIT_COUNT_INPUT_OR_TARGET_CONTRACT_CHANGED")
    descriptor = cell["descriptor"]
    require(descriptor.get("arm") == arm and descriptor["parameters"] == descriptor["registered_parameters"]
        == PARAMETERS[arm] and descriptor["image_only_prediction_uses_conditional_reread"] is False
        and descriptor["two_anchor_contrastive_losses_preserved"] is True,
        "S5_REPORT_ACTUAL_EFFECTIVE_PARAMETERS_OR_INPUT_BUDGET_CHANGED")
    require(sha(folder/"predictions.private.npz") == cell["predictions_sha256"], "S5_REPORT_PREDICTION_BYTES_CHANGED")
    require(replay.get("schema") == "spgel_s5_cell_replay_v1" and replay.get("status") == "PASS"
        and (replay.get("arm"), replay.get("donor"), replay.get("seed")) == key
        and replay.get("contract_sha256") == lock["contract_sha256"]
        and replay.get("COMPLETE_sha256") == complete_sha
        and replay.get("predictions_sha256") == cell["predictions_sha256"]
        and replay.get("new_optimizer_fits") == replay.get("new_optimizer_updates") == 0,
        "S5_REPORT_ACTUAL_REPLAY_BINDING_CHANGED")
    states, repeats = cell["snapshots"], replay["states"]
    require([(x["label"],x["epoch"]) for x in states] == [("epoch10",10),("epoch20",20),("epoch40",40)]
        and [x["epoch"] for x in repeats] == [10,20,40], "S5_REPORT_EXACT_3_DISTINCT_STATES_REQUIRED")
    for state, repeat in zip(states, repeats):
        require(sha(folder/f"checkpoint_{state['label']}.pt") == state["checkpoint_sha256"]
            == repeat["checkpoint_sha256"] and repeat.get("actual_saved_state_loaded") is True
            and repeat.get("actual_image_and_native_pair_forward_replayed") is True
            and repeat.get("new_optimizer_fits") == repeat.get("new_optimizer_updates") == 0,
            "S5_REPORT_ACTUAL_SAVED_STATE_FORWARD_REQUIRED")
        number(repeat["maximum_prediction_error"], minimum=0, maximum=2e-6)
        number(repeat["maximum_metric_error"], minimum=0, maximum=1e-8)
    require([x["epoch"] for x in cell["learning_curve"]] == list(range(1,41)), "S5_REPORT_FULL_40_EPOCH_CURVE_REQUIRED")
    for epoch, curve in enumerate(cell["learning_curve"],1):
        require(curve["optimizer_updates"] == epoch*math.ceil(part["train_rows"]/512)
            and curve["paired_support"]["batch_rows"] == part["train_rows"], "S5_REPORT_UPDATE_OR_BATCH_SUPPORT_CHANGED")
        require(set(curve["losses"])=={"total","prediction_Huber","tissue_CE","ITC"}
            and set(curve["query_gradient_norms_mean"])=={"gene_to_image","image_to_gene"},
            "S5_REPORT_ONLY_REGISTERED_LOSS_AND_GRADIENT_METRICS")
        for value in curve["losses"].values():
            number(value, minimum=0)
        for value in curve["query_gradient_norms_mean"].values():
            number(value, minimum=0)
    number(cell["elapsed_seconds"], minimum=0);number(cell["peak_torch_GPU_bytes"], minimum=0)


def load_matrix(private, config_path, assets_path, *, root=ROOT):
    private = Path(private)
    config, lock, parts, source = validate_lock(private, config_path, assets_path, root=root)
    expected = {(a,d,s) for a in ARMS for d in range(6) for s in SEEDS}
    require({p.name for p in (private/"cells").iterdir() if p.is_dir()} == {name(k) for k in expected},
            "S5_REPORT_EXACT_84_CELLS_REQUIRED_NO_PARTIAL_MEANS")
    require({p.name for p in (private/"replay").glob("*.json")} == {name(k)+".json" for k in expected},
            "S5_REPORT_EXACT_84_REPLAY_UNITS_REQUIRED")
    cells, provenance, prediction_error, metric_error = {}, {}, 0., 0.
    for key in sorted(expected):
        folder = private/"cells"/name(key)
        path, rp = folder/"COMPLETE.json", private/"replay"/(name(key)+".json")
        cell, replay = read_json(path), read_json(rp)
        validate_cell(cell,replay,key,parts[key[1:]],lock,folder,sha(path))
        cells[key] = cell
        provenance[name(key)] = dict(COMPLETE_sha256=sha(path), replay_sha256=sha(rp),
            predictions_sha256=cell["predictions_sha256"],
            checkpoint_sha256={x["label"]:x["checkpoint_sha256"] for x in cell["snapshots"]})
        prediction_error = max(prediction_error,max(x["maximum_prediction_error"] for x in replay["states"]))
        metric_error = max(metric_error,max(x["maximum_metric_error"] for x in replay["states"]))
    for donor in range(6):
        for seed in SEEDS:
            base = cells[("M2",donor,seed)]
            initial = base["initial_state_tensor_sha256"]
            for arm in ARMS:
                cell = cells[(arm,donor,seed)]
                expected_initial = {k:v for k,v in initial.items()
                    if not (arm in ("M0","M4") and k=="gene_to_image.weight")
                    and not (arm in ("M0","M3") and k=="image_to_gene.weight")}
                got = cell["initial_state_tensor_sha256"]
                require(set(got)==set(expected_initial), "S5_REPORT_INITIAL_PARAMETER_SET_CHANGED")
                for key,value in got.items():
                    if arm!="M6" or key not in ("gene_encoder.up","gene_encoder.down"):
                        require(value==expected_initial[key], "S5_REPORT_COMMON_INITIALIZATION_CHANGED")
                if arm=="M6":
                    require(any(got[k]!=initial[k] for k in ("gene_encoder.up","gene_encoder.down")),
                            "S5_REPORT_MEMBERSHIP_NULL_DID_NOT_CHANGE_MODEL")
                require([(x["paired_support"],x["negative_candidate_sha256"]) for x in cell["learning_curve"]]
                    == [(x["paired_support"],x["negative_candidate_sha256"]) for x in base["learning_curve"]],
                    "S5_REPORT_ALL_ARMS_MUST_SHARE_ACTUAL_ANCHORS_AND_NEGATIVES")
    ledger = dict(schema="spgel_s5_actual_metadata_execution_ledger_v1", actual_neural_fits=84,
        actual_optimizer_updates=sum(c["actual_optimizer_updates"] for c in cells.values()),
        actual_saved_state_replays=252, mapping_partitions=12, new_fits_or_forwards_by_reporter=0,
        checkpoint_or_array_deserializations_by_reporter=0, current_checkpoint_files_hashed=252,
        current_prediction_files_hashed=84, maximum_prediction_replay_error=prediction_error,
        maximum_metric_replay_error=metric_error, sum_fit_elapsed_seconds=sum(c["elapsed_seconds"] for c in cells.values()),
        sum_fit_elapsed_is_not_parallel_wall_time=True,
        maximum_individual_fit_peak_GPU_bytes=max(c["peak_torch_GPU_bytes"] for c in cells.values()),
        individual_peak_is_not_aggregate_parallel_GPU_use=True, all_arms_actual_support_and_negative_hashes_equal=True,
        common_initialization_actual_equal=True, capacity_M1_M2_effective_parameters=333203,
        artifacts=provenance, source=source, lock_sha256=sha(private/"S5_RUN_LOCK.json"),
        qualification_sha256=lock["qualification_sha256"], config_sha256=sha(config_path),
        contract_sha256=lock["contract_sha256"], reporter_sha256=sha(Path(__file__)))
    require(ledger["actual_optimizer_updates"] == 72800, "S5_REPORT_EXACT_72800_UPDATES_REQUIRED")
    return config, lock, cells, ledger


def _diagnostic_token(row, probe):
    require(type(row.get("valid_spot_programs")) is int
        and 0<=row["valid_spot_programs"]<=4*probe
        and len(row.get("per_program",[]))==4, "S5_REPORT_DIAGNOSTIC_SUPPORT")
    per=[]
    for k,item in enumerate(row["per_program"]):
        require(item.get("program_index")==k and type(item.get("valid_spots")) is int
            and 0<=item["valid_spots"]<=probe, "S5_REPORT_DIAGNOSTIC_PROGRAM_SUPPORT")
        value=item.get("RMS")
        require((value is None)==(item["valid_spots"]==0), "S5_REPORT_DIAGNOSTIC_NULL_SUPPORT")
        if value is not None:number(value,minimum=0)
        per.append(dict(program_index=k,valid_spots=item["valid_spots"],RMS=value))
    count=sum(x["valid_spots"] for x in per)
    require(count==row["valid_spot_programs"] and (row.get("RMS") is None)==(count==0),
        "S5_REPORT_DIAGNOSTIC_MACRO_SUPPORT")
    if count:
        expected=math.sqrt(sum(x["valid_spots"]*x["RMS"]**2 for x in per if x["valid_spots"])/count)
        require(abs(number(row["RMS"],minimum=0)-expected)<=1e-9*max(1.,expected),
            "S5_REPORT_DIAGNOSTIC_RMS_NOT_REPRODUCED")
    return dict(RMS=row.get("RMS"),valid_spot_programs=count,per_program=per)


def _diagnostic_attention(row, probe):
    per=[]
    require(len(row.get("per_program",[]))==4,"S5_REPORT_DIAGNOSTIC_ATTENTION_PROGRAMS")
    for k,item in enumerate(row["per_program"]):
        require(item.get("program_index")==k and type(item.get("valid_spots")) is int
            and 0<=item["valid_spots"]<=probe,"S5_REPORT_DIAGNOSTIC_ATTENTION_SUPPORT")
        value=item.get("mean_row_L1")
        require((value is None)==(item["valid_spots"]==0),"S5_REPORT_DIAGNOSTIC_ATTENTION_NULL_SUPPORT")
        if value is not None:number(value,minimum=0,maximum=2.+1e-6)
        per.append(dict(program_index=k,valid_spots=item["valid_spots"],mean_row_L1=value))
    count=sum(x["valid_spots"] for x in per)
    require(row.get("valid_spot_programs")==count and (row.get("mean_row_L1") is None)==(count==0),
            "S5_REPORT_DIAGNOSTIC_ATTENTION_MACRO_SUPPORT")
    if count:
        expected=sum(x["valid_spots"]*x["mean_row_L1"] for x in per if x["valid_spots"])/count
        require(abs(number(row["mean_row_L1"],minimum=0,maximum=2.+1e-6)-expected)<=1e-9,
            "S5_REPORT_DIAGNOSTIC_ATTENTION_MACRO_CHANGED")
        number(row["maximum_absolute_difference"],minimum=0,maximum=1.+1e-6)
    else:require(row.get("maximum_absolute_difference") is None,"S5_REPORT_DIAGNOSTIC_ATTENTION_UNDEFINED")
    return dict(mean_row_L1=row.get("mean_row_L1"),valid_spot_programs=count,
        maximum_absolute_difference=row.get("maximum_absolute_difference"),per_program=per)


def diagnostic_summary(raw, arm):
    require(raw.get("status") == "COMPLETE_THREE_FROZEN_TASK_B_RUNS"
        and 2<=raw.get("probe_count",0)<=128 and raw.get("paired_score_calls")==3
        and raw.get("optimizer_instances")==raw.get("optimizer_updates")==raw.get("backward_calls")==0
        and raw.get("target_tissue_library_or_survival_reads") is False
        and raw.get("model_state_unchanged") is raw.get("existing_gradients_unchanged")
            is raw.get("training_modes_restored") is True, "S5_REPORT_FROZEN_DIAGNOSTIC_CONTRACT_CHANGED")
    old.digest(raw.get("model_state_sha256"),"S5_REPORT_DIAGNOSTIC_STATE_SHA")
    result = {}
    for direction, field in (("gene_to_image","changed_gene_context"),("image_to_gene","changed_image_context")):
        part = raw[direction]
        result[direction] = {key:_diagnostic_token(part[key],raw["probe_count"])
            for key in ("base_to_conditioned",field)}
        attention=("base_attention_change","changed_gene_attention") if direction=="gene_to_image" else (
            "base_up_attention_change","base_down_attention_change","changed_image_up_attention","changed_image_down_attention")
        result[direction].update({key:_diagnostic_attention(part[key],raw["probe_count"]) for key in attention})
        closed = arm in ("M0","M1") or (arm=="M3" and direction=="image_to_gene") or (arm=="M4" and direction=="gene_to_image")
        if closed:
            require(all(x is None or x==0 for x in [part[field]["RMS"],*[p["RMS"] for p in part[field]["per_program"]]]),
                    "S5_REPORT_CLOSED_CROSS_CONTEXT_PATH_NONZERO")
    scores={}
    for label in ("rolled_genes","rolled_images"):
        item=raw["paired_score_change"][label];count=item["valid_pairs"]
        require(type(count) is int and 0<=count<=raw["probe_count"],"S5_REPORT_DIAGNOSTIC_SCORE_SUPPORT")
        selected=dict(valid_pairs=count)
        for metric in ("mean_difference","mean_absolute_difference","RMS"):
            value=item[metric]
            require((value is None)==(count==0),"S5_REPORT_DIAGNOSTIC_SCORE_NULL_SUPPORT")
            if value is not None:number(value,minimum=-2. if metric=="mean_difference" else 0.,maximum=2.+1e-6)
            selected[metric]=value
        scores[label]=selected
    result.update(paired_score_change=scores, probe_count=raw["probe_count"],
        model_state_sha256=raw["model_state_sha256"], cross_context_differences_are_not_performance_evidence=True,
        rolled_pairing_is_not_matched_tissue_depth_training_control=True)
    return result


def paired_bootstrap(values):
    """2 seeds stay together whenever a donor is resampled; no spot bootstrap."""
    require(len(values)==2 and all(len(row)==6 for row in values), "S5_REPORT_BOOTSTRAP_2_BY_6_REQUIRED")
    seed_summary = {str(seed):strict_mean(row) for seed,row in zip(SEEDS,values)}
    point = strict_mean(seed_summary[str(seed)]["mean"] for seed in SEEDS)
    result = dict(seed_summary=seed_summary, two_seed_mean=point,
        unit="six_historical_donors_with_both_seeds_kept_together", bootstrap_replicates=BOOTSTRAPS,
        bootstrap_seed=BOOTSTRAP_SEED, CI95=None, nominal_seeds_are_not_independent_patients=True)
    if point["mean"] is None:
        result["status"]="UNDEFINED_REQUIRES_ALL_SIX_DONORS_AND_TWO_SEEDS"
        return result
    matrix = np.asarray(values,dtype=float)
    number(float(matrix.mean()))
    donors = matrix.mean(0)
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0,6,size=(BOOTSTRAPS,6))
    low,high = np.quantile(donors[samples].mean(1),[.025,.975])
    result.update(status="DEFINED_FIXED_DONOR_PAIRED_BOOTSTRAP",CI95=[float(low),float(high)],
        bootstrap_donor_seed_means=[float(x) for x in donors],
        CI_is_development_uncertainty_not_external_confirmation=True)
    return result


def build_report(config, lock, cells, ledger):
    rows, programs, diagnostics, curves = [], [], [], []
    for (arm,donor,seed),cell in sorted(cells.items()):
        for state in cell["snapshots"]:
            train = old.task_a_summary(state["train"]["task_A"],rows=cell["train_rows"])
            held = old.task_a_summary(state["heldout"]["task_A"],rows=cell["heldout_rows"])
            require(set(state["losses"])=={"total","prediction_Huber","tissue_CE","ITC"},
                    "S5_REPORT_SNAPSHOT_LOSS_SCHEMA")
            losses={k:number(v,minimum=0) for k,v in state["losses"].items()}
            diagnostics.append(dict(arm=arm,donor=donor,seed=seed,epoch=state["epoch"],
                train_Pearson=train["primary_donor_four_program_Pearson"],
                heldout_Pearson=held["primary_donor_four_program_Pearson"],losses=losses,
                primary=state["epoch"]==40, epoch_selection_performed=False))
            if state["epoch"]!=40:
                require(all(state["heldout"][key].get("status")=="NOT_EVALUATED_FOR_TASKB"
                    for key in ("task_B_common","task_B_native")), "S5_REPORT_EARLY_RETRIEVAL_NOT_REGISTERED")
        final = cell["snapshots"][-1]["heldout"]
        a = old.task_a_summary(final["task_A"],rows=cell["heldout_rows"])
        c = old.task_c_summary(final["task_C"],rows=cell["heldout_rows"])
        common = old.task_b_summary(final["task_B_common"],native_applicable=True)
        native = old.task_b_summary(final["task_B_native"],native_applicable=True)
        path = diagnostic_summary(cell["snapshots"][-1]["paired_path_diagnostics"],arm)
        require(path["probe_count"]==min(128,cell["heldout_rows"])
            and cell["snapshots"][-1]["paired_path_diagnostics"]["partition_rows_sha256"]==cell["heldout_rows_sha256"],
            "S5_REPORT_DIAGNOSTIC_HELDOUT_SCOPE_CHANGED")
        row = dict(arm=arm,donor=donor,seed=seed,heldout_spots=cell["heldout_rows"],parameters=PARAMETERS[arm],
            optimizer_updates=cell["actual_optimizer_updates"], primary_epoch=40,
            task_A_Pearson=a["primary_donor_four_program_Pearson"], task_A_defined_programs=a["native"]["defined_programs"],
            task_A_native_MSE=a["native"]["Mean_MSE"]["mean"],
            task_A_train_Z_MSE=a["train_Z"]["Mean_MSE"]["mean"],
            task_A_train_Z_Huber=a["train_Z"]["Mean_Huber_delta1"]["mean"],
            task_C_AUPRC=c["primary_four_program_AUPRC"],task_C_defined_AUPRC_programs=c["AUPRC"]["defined_count"],
            native_retrieval=native,common_retrieval=common,paired_path_diagnostics=path)
        rows.append(row)
        for k in range(4):
            pr,z,loc = a["native"]["per_output"][k],a["train_Z"]["per_output"][k],c["per_output"][k]
            programs.append(dict(arm=arm,donor=donor,seed=seed,program_index=k,program=PROGRAMS[k],
                observed_spots=pr["observed_count"],Pearson=pr["Pearson"],Spearman=pr["Spearman"],
                native_MAE=pr["MAE"],native_MSE=pr["MSE"],train_Z_MSE=z["MSE"],train_Z_Huber=z["Huber_delta1"],
                AUPRC=loc["AUPRC"],high_activity_spots=loc["high_activity_count"],
                high_activity_threshold_from_train=loc["high_activity_threshold_from_train"]))
        curve = cell["learning_curve"]
        curves.append(dict(arm=arm,donor=donor,seed=seed,epochs=40,
            loss_by_epoch={k:[x["losses"][k] for x in curve] for k in ("total","prediction_Huber","tissue_CE","ITC")},
            query_gradient_norm_by_epoch={k:[x["query_gradient_norms_mean"][k] for x in curve]
                for k in ("gene_to_image","image_to_gene")},
            last10_total_loss_relative_change=(curve[-1]["losses"]["total"]-curve[-10]["losses"]["total"])
                /max(abs(curve[-10]["losses"]["total"]),1e-12),
            sufficient_optimization_claimed=False))
    require(len(rows)==84 and len(programs)==336 and len(diagnostics)==252, "S5_REPORT_ALL_METRIC_UNITS_REQUIRED")
    summary = []
    for arm in ARMS:
        per_seed=[]
        for seed in SEEDS:
            selected=[x for x in rows if x["arm"]==arm and x["seed"]==seed]
            require(len(selected)==6 and sum(x["heldout_spots"] for x in selected)==13011,
                    "S5_REPORT_EQUAL_SIX_DONOR_SUPPORT_REQUIRED")
            item=dict(seed=seed,task_A_Pearson=strict_mean(x["task_A_Pearson"] for x in selected),
                task_C_AUPRC=strict_mean(x["task_C_AUPRC"] for x in selected))
            for mode in ("native_retrieval","common_retrieval"):
                item[mode]={direction:{metric:strict_mean(x[mode][direction][metric] for x in selected)
                    for metric in ("Recall_at_1","Recall_at_5","pair_AUC","median_rank")}
                    for direction in ("image_to_gene","gene_to_image")}
            per_seed.append(item)
        summary.append(dict(arm=arm,per_seed=per_seed,
            task_A_two_seed_mean=strict_mean(x["task_A_Pearson"]["mean"] for x in per_seed),
            task_C_two_seed_mean=strict_mean(x["task_C_AUPRC"]["mean"] for x in per_seed)))
    lookup={(x["arm"],x["donor"],x["seed"]):x for x in rows}
    metrics=[("TASK_A","four_program_Pearson",lambda x:x["task_A_Pearson"]),
             ("TASK_C","strict_four_program_AUPRC",lambda x:x["task_C_AUPRC"])]
    metrics += [("TASK_B_NATIVE",direction+"_"+metric,
        lambda x,d=direction,m=metric:x["native_retrieval"][d][m])
        for direction in ("image_to_gene","gene_to_image") for metric in ("Recall_at_1","Recall_at_5","pair_AUC")]
    paired=[]
    for ref in REFERENCES:
        for task,metric,extract in metrics:
            values=[]
            for seed in SEEDS:
                delta=[]
                for donor in range(6):
                    left,right=extract(lookup[("M2",donor,seed)]),extract(lookup[(ref,donor,seed)])
                    delta.append(None if left is None or right is None else left-right)
                values.append(delta)
            paired.append(dict(candidate="M2",reference=ref,comparison_role="primary" if ref in PRIMARY and task=="TASK_A" else "secondary",
                task=task,metric=metric,donor_differences_by_seed={str(s):v for s,v in zip(SEEDS,values)},
                uncertainty=paired_bootstrap(values)))
    primary=[x for x in paired if x["comparison_role"]=="primary"]
    require(len(primary)==2,"S5_REPORT_EXACT_TWO_PRIMARY_COMPARISONS_REQUIRED")
    claims=[dict(reference=x["reference"],mean_delta=x["uncertainty"]["two_seed_mean"]["mean"],
        both_seed_means_positive=all(x["uncertainty"]["seed_summary"][str(s)]["mean"] is not None
            and x["uncertainty"]["seed_summary"][str(s)]["mean"]>0 for s in SEEDS),
        donor_bootstrap_CI95=x["uncertainty"]["CI95"],
        donor_bootstrap_lower_positive=x["uncertainty"]["CI95"] is not None and x["uncertainty"]["CI95"][0]>0)
        for x in primary]
    return dict(schema="spgel_s5_attribution_complete_report_v1",status="COMPLETE_84_FITS_252_ACTUAL_REPLAYS",
        contract_sha256=lock["contract_sha256"], config_sha256=lock["config_sha256"],
        fixed_primary_comparisons=["M2-M1","M2-M5"], primary_epoch=40, program_order=list(PROGRAMS),
        aggregation="four_program_strict_macro_then_equal_six_donors_per_seed_then_equal_two_seeds",
        undefined_metrics_are_null_not_imputed=True,seeds_are_not_independent_patients=True,
        historical_development_source_only=True,new_external_confirmation=False,
        cell_source_or_histology_truth_claimed=False,survival_or_R8_information_used=False,
        fixed40_epochs_prove_convergence=False,old_A4_A5_architecture_is_not_old_training_protocol_identity=True,
        epoch_selection_or_architecture_expansion_performed=False,ledger=ledger,
        arm_summary=summary,donor_rows=rows,program_rows=programs,checkpoint_diagnostics=diagnostics,
        training_diagnostics=curves,paired_comparisons=paired,primary_evidence=claims,
        decision=dict(status="DEVELOPMENT_ATTRIBUTION_REPORTED_REQUIRES_FORMAL_BENCHMARK_AND_INDEPENDENT_CONFIRMATION",
            primary_evidence=claims,complete_S5B_or_external_confirmation=False,
            diagnosis_sensitivity_does_not_automatically_prove_performance=True,
            capacity_control_M1_uses_effective_self_context_reread=True,
            M5_changes_ITC_positive_pair_only=True,M6_changes_model_membership_not_truth=True,
            model_selection_or_idea_wide_no_go_authorized=False))


def write_csv(path, rows):
    require(rows, "S5_REPORT_EMPTY_PUBLIC_CSV")
    with path.open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def encoded_json(value):
    return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode("utf-8")


def restore_report(path):
    """Restore all 40-epoch curves and verify the original full-report bytes."""
    path=Path(path);main=read_json(path)
    lineage=main.pop("report_serialization")
    require(lineage.get("schema")=="spgel_s5_lossless_training_component_v1",
            "S5_REPORT_SERIALIZATION_CONTRACT_CHANGED")
    reference=main["training_diagnostics"]
    require(reference.get("file")=="S5A_TRAINING_DIAGNOSTICS.json"
        and reference.get("fit_units")==84 and reference.get("epochs_per_fit")==40,
        "S5_REPORT_TRAINING_COMPONENT_NAME_OR_SUPPORT_CHANGED")
    side_path=path.parent/reference["file"]
    require(sha(side_path)==reference["sha256"],"S5_REPORT_TRAINING_COMPONENT_BYTES_CHANGED")
    side=read_json(side_path)
    require(side.get("schema")=="spgel_s5_complete_training_curves_v1"
        and side.get("contract_sha256")==main["contract_sha256"] and len(side.get("rows",[]))==84,
        "S5_REPORT_TRAINING_COMPONENT_LINEAGE_CHANGED")
    expected={(a,d,s) for a in ARMS for d in range(6) for s in SEEDS}
    require({(r["arm"],r["donor"],r["seed"]) for r in side["rows"]}==expected
        and all(r["epochs"]==40 for r in side["rows"]),"S5_REPORT_TRAINING_COMPONENT_MATRIX_CHANGED")
    main["training_diagnostics"]=side["rows"]
    require(canonical(main)==lineage["original_canonical_sha256"]
        and hashlib.sha256(encoded_json(main)).hexdigest()==lineage["original_full_report_LF_SHA256"],
        "S5_REPORT_FULL_REPORT_RECONSTRUCTION_FAILED")
    return main


def write_report(output,result):
    output=Path(output)
    require(result.get("status")=="COMPLETE_84_FITS_252_ACTUAL_REPLAYS"
        and result["ledger"]["actual_neural_fits"]==84 and result["ledger"]["actual_saved_state_replays"]==252,
        "S5_REPORT_NO_PARTIAL_PUBLIC_REPORT")
    # Serialization validation precedes all output file creation.
    full_raw=encoded_json(result)
    output.mkdir(parents=True,exist_ok=True)
    require(all(not (output/n).exists() for n in NAMES), "S5_REPORT_EXISTING_RESULTS_PRESERVED")
    donor=[]
    for row in result["donor_rows"]:
        entry={k:row[k] for k in ("arm","donor","seed","heldout_spots","parameters","optimizer_updates",
            "primary_epoch","task_A_Pearson","task_A_defined_programs","task_A_native_MSE",
            "task_A_train_Z_MSE","task_A_train_Z_Huber","task_C_AUPRC","task_C_defined_AUPRC_programs")}
        for mode in ("native_retrieval","common_retrieval"):
            for direction in ("image_to_gene","gene_to_image"):
                for metric in ("Recall_at_1","Recall_at_5","pair_AUC","median_rank"):
                    entry[mode+"_"+direction+"_"+metric]=row[mode][direction][metric]
        for direction,key in (("gene_to_image","changed_gene_context"),("image_to_gene","changed_image_context")):
            entry[direction+"_context_RMS"]=row["paired_path_diagnostics"][direction][key]["RMS"]
        donor.append(entry)
    pairs=[]
    for row in result["paired_comparisons"]:
        for seed,values in row["donor_differences_by_seed"].items():
            for code,value in enumerate(values):
                pairs.append(dict(candidate=row["candidate"],reference=row["reference"],comparison_role=row["comparison_role"],
                    task=row["task"],metric=row["metric"],seed=int(seed),donor=code,delta=value))
    def dump(name,value):
        with (output/name).open("xb") as stream:stream.write(encoded_json(value))
    side=dict(schema="spgel_s5_complete_training_curves_v1",contract_sha256=result["contract_sha256"],
              rows=result["training_diagnostics"])
    public=dict(result)
    public["training_diagnostics"]=dict(file="S5A_TRAINING_DIAGNOSTICS.json",
        sha256=hashlib.sha256(encoded_json(side)).hexdigest(),fit_units=84,epochs_per_fit=40)
    public["report_serialization"]=dict(schema="spgel_s5_lossless_training_component_v1",
        original_canonical_sha256=canonical(result),
        original_full_report_LF_SHA256=hashlib.sha256(full_raw).hexdigest(),
        metrics_or_curve_values_removed=False,reconstruction="scripts.spgel_s5_report.restore_report")
    # All JSON encodings and their independent file limits are checked before
    # any result file is created. Every aggregate curve remains in the sidecar.
    json_outputs={"S5A_RESULTS.json":public,"S5A_TRAINING_DIAGNOSTICS.json":side,
        "S5A_DECISION.json":result["decision"],"S5A_ACTUAL_LEDGER.json":result["ledger"]}
    require(all(len(encoded_json(value))<=2*1024**2 for value in json_outputs.values()),
            "S5_REPORT_PUBLIC_JSON_EXCEEDS_TWO_MIB_REQUIRES_NEW_LOSSLESS_COMPONENT")
    for filename,value in json_outputs.items():dump(filename,value)
    write_csv(output/"S5A_PER_DONOR.csv",donor)
    write_csv(output/"S5A_PROGRAM_METRICS.csv",result["program_rows"])
    write_csv(output/"S5A_PAIRED_COMPARISONS.csv",pairs)
    display=lambda v:"null" if v is None else f"{v:.6f}"
    lines=["# S5-A 机制归因：固定六供体、两个seed", "",
        "84次真实拟合、72,800次minibatch更新、252个保存状态实际回放均通过；报告过程没有加载模型/数组或新增前向/拟合。", "",
        "主指标先对每位供体取四程序严格宏平均，再对六供体等权平均；seed31/47分别报告，最后等权平均。缺失指标保留null，不按spot或可定义程序筛选赢家。", "",
        "| 模型 | seed31 PCC | seed47 PCC | 两seed均值 | 有效参数 |", "|---|---:|---:|---:|---:|"]
    for item in result["arm_summary"]:
        lines.append("| "+" | ".join([item["arm"],*[display(x["task_A_Pearson"]["mean"]) for x in item["per_seed"]],
            display(item["task_A_two_seed_mean"]["mean"]),str(PARAMETERS[item["arm"]])])+" |")
    lines += ["", "## 预先固定的直接配对比较", "",
        "主比较为M2−M1（跨模态条件 vs 等容量自适配）与M2−M5（真实 vs 同组织/近深度错误辅助配对）。以下95%区间固定抽样2000次，以六供体为单位并保留两seed共同抽样；不是12名独立患者。", ""]
    for item in result["primary_evidence"]:
        interval=item["donor_bootstrap_CI95"]
        lines.append(f"- M2−{item['reference']}：均值 {display(item['mean_delta'])}；CI95 "+(
            "null（缺少完整六供体×两seed支持）" if interval is None else f"[{display(interval[0])}, {display(interval[1])}]")+"。")
    lines += ["", "## 机制和解释边界", "",
        "M1/M2均为333,203个实际参与训练的参数；原A4/M0为300,435。M3/M4为316,819。所有臂使用同一ITC支持、负候选及本人回归/组织真值；M5只改辅助正配对，M6只改模型成员而不重命名真目标。", "",
        "A5的条件重读用于训练配对辅助目标与TASK-B；TASK-A/C推理仅用图像。冻结干预的token/注意力响应不自动构成预测收益。两个native检索方向的R@1、R@5和pair-AUC单独报告，不能混成一项排名。", "",
        "组织/深度匹配是粗分层的控制，不能排除全部来源、细胞组成或批次混杂。候选区域注意力不是组织学真值；历史六供体的交叉验证不能称为新的独立外部确认。固定40轮不是收敛证明；结果不授权事后换轮次或扩大结构。", "",
        "完整40轮训练损失与路径梯度保留在S5A_TRAINING_DIAGNOSTICS.json；主JSON以SHA引用。restore_report可恢复完整报告并核验其canonical及原始LF字节SHA，未删任何指标或曲线值。", "",
        "本阶段只完成S5-A归因；S5-B官方50基因强模型、严格SOTA复现和独立来源验证另立数据/预算合同，不能由本结果代替。", ""]
    with (output/"S5A_RESULTS.md").open("x",encoding="utf-8",newline="\n") as stream:stream.write("\n".join(lines))
    return dict(status=result["status"],public_files=len(NAMES),per_donor_rows=len(donor),
                per_program_rows=len(result["program_rows"]),paired_rows=len(pairs),new_fits=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private",type=Path,required=True)
    parser.add_argument("--assets",type=Path,required=True,help="Existing source-only pointer; hashes bytes, never loads arrays")
    parser.add_argument("--config",type=Path,default=ROOT/"configs/spgel_s5/mechanism_v1.json")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/spgel_s5")
    args=parser.parse_args()
    config,lock,cells,ledger=load_matrix(args.private,args.config,args.assets)
    result=build_report(config,lock,cells,ledger)
    print(json.dumps(write_report(args.output,result),allow_nan=False))


if __name__=="__main__":main()

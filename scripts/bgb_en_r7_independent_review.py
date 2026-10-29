"""Postfit R7 anonymous arithmetic review, independent of training/report code.

JSON/CSV only. Export reads PASS replay and optional COMPLETE/reference metadata,
never model/checkpoint/array/label files. Review never repeats model forwards.
The 99/297 claims remain upstream replay-certificate evidence, not new replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("ADD", "RAW", "NORM", "INDEP", "SLOT", "GATE1", "GATE2", "SLOT_WRONG_RNA",
    "GATE1_WRONG_RNA", "GATE1_WRONG_PATH", "GATE2_WRONG_RNA")
ALL_ARMS = ("B0", "B1", *ARMS)
PARTITIONS, SEEDS = (0, 1, 2), (31, 47, 83)
COMPARISONS = (("NORM", "RAW", "R7.0_joint_scale_interfaces"),
    ("INDEP", "NORM", "R7.1_independent_reverse"),
    ("SLOT", "INDEP", "R7.2_program_preserving_linear_readout"),
    ("GATE1", "SLOT", "R7.3_single_round_state_gate"),
    ("GATE2", "GATE1", "R7.3_shared_parameter_second_round"),
    ("GATE1", "ADD", "R7.4_interaction_vs_same_main_effects_retrained"),
    ("GATE1", "B1", "R7.4_increment_over_frozen_CP_scalar"),
    ("GATE1", "GATE1_WRONG_RNA", "R7.4_own_initial_molecular_context"),
    ("GATE1", "GATE1_WRONG_PATH", "R7.4_own_initial_pathology_reverse_context"),
    ("SLOT", "SLOT_WRONG_RNA", "R7.4_slot_initial_molecular_context"),
    ("GATE2", "GATE2_WRONG_RNA", "R7.4_two_round_initial_molecular_context"))
FIELDS = ("Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den")
CREDITS = ("repair", "damage", "Uno_repair", "Uno_damage")
DIRECT_NAME, SECONDARY_NAME, EXPORT_NAME = "R7_DIRECT_PAIR_CREDITS.csv", "R7_SECONDARY_FOLD_METRICS.csv", "R7_POSTFIT_EXPORT_AUDIT.json"
REVIEW_NAMES = ("R7_INDEPENDENT_REVIEW.json", "R7_INDEPENDENT_REVIEW.md")
RAW_KEYS = frozenset(("patient_ids", "patient_id", "barcodes", "barcode", "coordinates", "coords",
    "risk_vector", "OOF_predictions", "RNA_matrix", "CNA_matrix", "state_dict", "optimizer_state",
    "patient_indices", "train_indices", "valid_indices", "test_indices"))


def require(value, code):
    if not value:
        raise ValueError(code)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sha(content):
    return hashlib.sha256(content).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def equal(actual, expected, code):
    require((actual is None and expected is None) or
        (finite(actual) and finite(expected) and abs(actual - expected) <= 1e-10), code)


def numeric(value, *, nullable=False):
    if nullable and value in (None, "", "None"):
        return None
    try:
        number = float(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        raise ValueError("NONFINITE_OR_INVALID_AGGREGATE_NUMBER") from None
    require(finite(number), "NONFINITE_OR_INVALID_AGGREGATE_NUMBER")
    return float(number)


def flag(value, code):
    require(type(value) is bool or value in ("True", "False"), code)
    return value is True or value == "True"


def anonymous(value):
    if isinstance(value, dict):
        require(not RAW_KEYS.intersection(value), "RAW_IDENTIFIER_OR_ARRAY_FIELD_IN_ANONYMOUS_INPUT")
        for item in value.values():
            anonymous(item)
    elif isinstance(value, list):
        for item in value:
            anonymous(item)


def exact(rows, fields, expected, code):
    require(isinstance(rows, list) and all(isinstance(r, dict) for r in rows), code)
    keys = [tuple(r.get(k) for k in fields) for r in rows]
    require(len(keys) == len(expected) and len(set(keys)) == len(keys) and set(keys) == set(expected), code)
    return dict(zip(keys, rows))


class Metadata:
    def __init__(self):
        self.sources = {}

    def content(self, path, label):
        path = Path(path)
        require(path.suffix in (".json", ".csv") and path.is_file() and not path.is_symlink(), "METADATA_ONLY_SOURCE_REQUIRED")
        before = path.stat(); content = path.read_bytes(); after = path.stat()
        require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), "SOURCE_CHANGED_DURING_REVIEW")
        require(len(content) <= 32 * 1024**2, "METADATA_SOURCE_TOO_LARGE")
        entry = dict(source=label, bytes=len(content), sha256=sha(content))
        require(label not in self.sources or self.sources[label] == entry, "SOURCE_LABEL_CHANGED")
        self.sources[label] = entry
        return content

    def json(self, path, label, *, public=False):
        def reject(_):
            raise ValueError("NONFINITE_JSON_AGGREGATE")
        value = json.loads(self.content(path,label).decode("utf-8-sig"), parse_constant=reject)
        require(isinstance(value, dict), "METADATA_JSON_OBJECT_REQUIRED")
        if public:
            anonymous(value)
        return value

    def csv(self, path, label):
        rows = list(csv.DictReader(io.StringIO(self.content(path,label).decode("utf-8-sig"))))
        require(rows and all(None not in r for r in rows) and not RAW_KEYS.intersection(rows[0]), "INVALID_ANONYMOUS_CSV")
        return rows


def metrics(raw):
    result = {}
    for metric, numerator, denominator in (("Harrell","Harrell_num","Harrell_den"), ("Uno60","Uno_num","Uno_den")):
        n, d = numeric(raw.get(numerator)), numeric(raw.get(denominator))
        require(0 <= n <= d + 1e-10 and d >= 0, "CONCORDANCE_COMPONENT_RANGE_INVALID")
        ratio = n / d if d > 0 else None
        equal(numeric(raw.get(metric),nullable=True), ratio, "CONCORDANCE_RATIO_MISMATCH")
        result.update({metric:ratio, numerator:n, denominator:d})
    return result


def parse_identity(row):
    result = dict(row)
    for key in ("partition", "seed"):
        value = numeric(row.get(key)); require(value.is_integer(), "NONINTEGER_MODEL_IDENTITY")
        result[key] = int(value)
    return result


def metric_grid(rows):
    rows = [parse_identity(r) for r in rows]
    expected = {(a,p,s) for a in ALL_ARMS for p in PARTITIONS for s in SEEDS}
    indexed = exact(rows,("arm","partition","seed"),expected,"EXACT_117_ANONYMOUS_METRIC_ROWS_REQUIRED")
    return {key:metrics(row) for key,row in indexed.items()}


def aggregate(grid):
    result = {}
    for arm in ALL_ARMS:
        seeds = []
        for seed in SEEDS:
            cells = [grid[(arm,p,seed)] for p in PARTITIONS]
            values = {k:math.fsum(c[k] for c in cells) for k in ("Harrell_num","Harrell_den","Uno_num","Uno_den")}
            values.update(seed=seed, Harrell=values["Harrell_num"] / values["Harrell_den"] if values["Harrell_den"] else None,
                Uno60=values["Uno_num"] / values["Uno_den"] if values["Uno_den"] else None)
            seeds.append(values)
        result[arm] = dict(arm=arm, per_seed=seeds)
        for name in ("Harrell","Uno60"):
            result[arm][name] = statistics.fmean(r[name] for r in seeds) if all(r[name] is not None for r in seeds) else None
    return result


def validate_replay_certificate(replay):
    require(replay.get("status") == "PASS_COMPLETE_R7_ALL_CHECKPOINT_REPLAY"
        and all(replay.get(k) == 99 for k in ("expected_units","available_units","passed_units"))
        and replay.get("failed_units") == 0 and replay.get("expected_checkpoints") == replay.get("passed_checkpoints") == 297
        and replay.get("missing_or_unreplayed_units") == []
        and replay.get("new_optimizer_fits") == replay.get("new_preprocessing_fits") == 0
        and digest(replay.get("contract_sha256")) and digest(replay.get("qualification_sha256")), "COMPLETE_UPSTREAM_99_297_REPLAY_CERTIFICATE_REQUIRED")
    rows = exact(replay.get("units"),("arm","partition","seed"),
        {(a,p,s) for a in ARMS for p in PARTITIONS for s in SEEDS},"UPSTREAM_REPLAY_99_IDENTITIES_CHANGED")
    for (arm,part,seed), row in rows.items():
        require(row.get("unit") == f"{arm}_p{part}_seed{seed}" and row.get("status") == "PASS"
            and row.get("actual_checkpoint_count") == 3 and row.get("all_checkpoints_actual_forward_replayed") is True
            and row.get("source_frozen_weight_hash_replayed") is True and row.get("saved_query_mapping_recomputed") is True
            and row.get("new_optimizer_updates") == row.get("changed_pair_credits") == 0
            and row.get("contract_sha256") == replay["contract_sha256"]
            and row.get("qualification_sha256") == replay["qualification_sha256"]
            and digest(row.get("completion_sha256")) and digest(row.get("artifact_sha256")), "UPSTREAM_ACTUAL_UNIT_REPLAY_NOT_VERIFIED")
        checkpoints = row.get("checkpoints",[])
        require([c.get("epoch") for c in checkpoints] == [20,40,80], "UPSTREAM_THREE_CHECKPOINT_EPOCHS_REQUIRED")
        for checkpoint in checkpoints:
            require(checkpoint.get("status") == "PASS" and checkpoint.get("actual_state_dict_forward_replayed") is True
                and checkpoint.get("new_optimizer_updates") == checkpoint.get("changed_pair_credits") == 0
                and finite(checkpoint.get("maximum_risk_error")) and 0 <= checkpoint["maximum_risk_error"] <= 2e-6
                and digest(checkpoint.get("checkpoint_sha256")), "UPSTREAM_ACTUAL_CHECKPOINT_REPLAY_NOT_VERIFIED")
    references = exact(replay.get("zero_fit_baseline_reference_replays"),("arm","partition","seed"),
        {(a,p,s) for a in ("B0","B1") for p in PARTITIONS for s in SEEDS},"EXACT_18_UPSTREAM_REFERENCE_RECORDS_REQUIRED")
    require(all(r.get("status") == "PASS" and r.get("independent_fixed_anchor_metrics_recomputed") is True
        and r.get("new_optimizer_fits") == 0 and digest(r.get("receipt_sha256")) for r in references.values()), "UPSTREAM_REFERENCE_REPLAY_CHANGED")
    return rows,references


def direct_rows(replay):
    expected = {(a,b,p,s) for a,b,_ in COMPARISONS for p in PARTITIONS for s in SEEDS}
    indexed = exact(replay.get("paired_retrained_controls"),("candidate","reference","partition","seed"),
        expected,"EXACT_99_DIRECT_PAIR_CREDIT_RECORDS_REQUIRED")
    output = []
    for key,row in sorted(indexed.items()):
        require(row.get("direct_risk_pair_credits") is True and row.get("primary_checkpoint_epoch") == 80,
            "DIRECT_CREDITS_MUST_BE_REPLAYED_EPOCH80")
        score = metrics(row.get("candidate_vs_reference",{}))
        credit = {}
        for label, denominator in (("repair","Harrell_den"),("damage","Harrell_den"),("Uno_repair","Uno_den"),("Uno_damage","Uno_den")):
            value = numeric(row["candidate_vs_reference"].get(label),nullable=True)
            require(value is None if score[denominator] == 0 else value is not None and 0 <= value <= 1,
                "NORMALIZED_PAIR_CREDIT_RANGE_OR_NULL_INVALID")
            credit[label] = value
        output.append(dict(candidate=key[0],reference=key[1],partition=key[2],seed=key[3], **score, **credit,
            direct_risk_pair_credits=True, primary_checkpoint_epoch=80))
    return output


def safe_output(output,names):
    output = Path(output).resolve()
    require(not any(output == p.resolve() or output.is_relative_to(p.resolve()) for p in
        (ROOT/"reports/bgb_en",ROOT/"reports/bgb_en_r5",ROOT/"reports/bgb_en_r6")), "HISTORICAL_REPORTS_PROTECTED")
    require(not any((output/name).exists() for name in names),"APPEND_ONLY_OUTPUT_EXISTS")
    output.mkdir(parents=True,exist_ok=True)
    return output


def write_csv(path,rows,fields):
    with path.open("x",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fields);writer.writeheader();writer.writerows(rows)
    content = path.read_bytes()
    return dict(file=path.name,bytes=len(content),sha256=sha(content),rows=len(rows))


def write_json(path,value):
    with path.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n")


def export_direct(replay_path,output,*,private_root=None):
    reader = Metadata(); replay = reader.json(replay_path,"upstream_replay_certificate")
    units,references = validate_replay_certificate(replay)
    rows = direct_rows(replay);secondary=[]
    if private_root is not None:
        private = Path(private_root)
        for (arm,part,seed),record in sorted({**units,**references}.items()):
            unit = f"{arm}_p{part}_seed{seed}"; baseline = arm in ("B0","B1")
            path = private/"references"/(unit+".json") if baseline else private/"cells"/unit/"COMPLETE.json"
            label = "reference_"+unit if baseline else "completion_"+unit
            cell = reader.json(path,label)
            expected_sha = record["receipt_sha256"] if baseline else record["completion_sha256"]
            require(reader.sources[label]["sha256"] == expected_sha and
                (cell.get("arm"),cell.get("partition"),cell.get("seed")) == (arm,part,seed),"BOUND_COMPLETION_OR_REFERENCE_METADATA_CHANGED")
            if baseline:
                require(cell.get("status") == "R7_BASELINE_REFERENCE" and cell.get("actual_new_optimizer_fits") == cell.get("teacher_refits") == 0
                    and cell.get("contract_sha256") == replay["contract_sha256"],"SECONDARY_REFERENCE_CONTRACT_CHANGED")
            else:
                require(cell.get("status") == "COMPLETE_R7_TARGET_FIT" and cell.get("epochs") == cell.get("optimizer_updates") == 80
                    and cell.get("actual_new_optimizer_fits") == 1 and cell.get("teacher_refits") == cell.get("source_training_updates") == 0
                    and cell.get("outer_test_used") is False and cell.get("contract",{}).get("contract_sha256") == replay["contract_sha256"],
                    "SECONDARY_COMPLETE_CONTRACT_CHANGED")
            secondary.append(dict(arm=arm,partition=part,seed=seed, **metrics(cell.get("metrics_vs",{}).get("PFI_frozen_DSS_risk",{}))))
    names = [DIRECT_NAME,EXPORT_NAME]+([SECONDARY_NAME] if secondary else [])
    output = safe_output(output,names)
    artifacts = [write_csv(output/DIRECT_NAME,rows,("candidate","reference","partition","seed",*FIELDS,*CREDITS,
        "direct_risk_pair_credits","primary_checkpoint_epoch"))]
    if secondary:
        require(len(secondary) == 117,"EXACT_117_SECONDARY_ROWS_REQUIRED")
        artifacts.append(write_csv(output/SECONDARY_NAME,secondary,("arm","partition","seed",*FIELDS)))
    audit = dict(schema="bgb_en_r7_postfit_metadata_export_v1",status="PASS_ANONYMOUS_METADATA_EXPORT",
        generator=dict(file="scripts/bgb_en_r7_independent_review.py",sha256=sha(Path(__file__).read_bytes())),
        contract_sha256=replay["contract_sha256"],source_replay_sha256=reader.sources["upstream_replay_certificate"]["sha256"],
        validated_upstream_certificate=dict(neural_units=99,checkpoint_records=297,zero_fit_reference_records=18,
            actual_forward_flags_checked=True,independent_model_forwards_performed_here=0),
        direct_rows=99,secondary_rows=len(secondary),secondary_endpoint="PFI_frozen_DSS_risk" if secondary else "NOT_EXPORTED",
        source_metadata=list(reader.sources.values()),artifacts=artifacts,
        runtime=dict(optimizer_fits=0,optimizer_updates=0,model_forwards=0,backward_calls=0,
            model_checkpoint_array_label_files_read=False,only_JSON_metadata_read=True),
        limits=dict(upstream_PASS_metadata_is_not_new_model_replay=True,not_independent_external_validation=True))
    write_json(output/EXPORT_NAME,audit)
    return audit


def check_summaries(recomputed,declared,label):
    require(isinstance(declared,dict) and set(declared) == set(ALL_ARMS),"EXACT_13_REPORTED_ARM_SUMMARIES_REQUIRED")
    for arm,actual in recomputed.items():
        row = declared[arm]
        require(row.get("arm") == arm and row.get("complete_cells") == row.get("expected_cells") == 9
            and row.get("status") == "COMPLETE_NINE_CELLS" and row.get("cross_partition_risk_ranking") is False,
            label+"_INCOMPLETE_OR_INVALID_SUMMARY")
        seeds = exact(row.get("per_seed"),("seed",),{(s,) for s in SEEDS},"THREE_REPORTED_SEEDS_REQUIRED")
        for seed in actual["per_seed"]:
            for name in FIELDS:
                equal(seeds[(seed["seed"],)].get(name),seed[name],label+"_SEED_COMPONENT_AGGREGATION_MISMATCH")
        for name in ("Harrell","Uno60"):
            equal(row.get(name),actual[name],label+"_SEED_MEAN_AGGREGATION_MISMATCH")


def paired(grid,summaries,rows):
    parsed = [parse_identity(r) for r in rows]
    expected = {(a,b,p,s) for a,b,_ in COMPARISONS for p in PARTITIONS for s in SEEDS}
    indexed = exact(parsed,("candidate","reference","partition","seed"),expected,"EXACT_99_EXPORTED_DIRECT_PAIR_ROWS_REQUIRED")
    computed = []
    for candidate,reference,question in COMPARISONS:
        per_seed=[]
        for seed in SEEDS:
            values=[]
            for part in PARTITIONS:
                row=indexed[(candidate,reference,part,seed)];score=metrics(row)
                require(flag(row.get("direct_risk_pair_credits"),"DIRECT_PAIR_FLAG_INVALID")
                    and numeric(row.get("primary_checkpoint_epoch")) == 80,"DIRECT_PAIR_REPLAY_SCOPE_CHANGED")
                a,b=grid[(candidate,part,seed)],grid[(reference,part,seed)]
                for name in FIELDS:
                    equal(score[name],a[name],"DIRECT_PAIR_CANDIDATE_COMPONENTS_CHANGED")
                for name in ("Harrell_den","Uno_den"):
                    equal(a[name],b[name],"PAIRED_ARMS_HAVE_DIFFERENT_COMPARABLE_PAIR_SUPPORT")
                credit={}
                for name,den in (("repair","Harrell_den"),("damage","Harrell_den"),("Uno_repair","Uno_den"),("Uno_damage","Uno_den")):
                    value=numeric(row.get(name),nullable=True)
                    require(value is None if score[den] == 0 else value is not None and 0 <= value <= 1,"DIRECT_PAIR_CREDIT_RANGE_OR_NULL_CHANGED")
                    credit[name]=value
                for metric,r,d in (("Harrell","repair","damage"),("Uno60","Uno_repair","Uno_damage")):
                    delta=None if a[metric] is None or b[metric] is None else a[metric]-b[metric]
                    equal(None if credit[r] is None else credit[r]-credit[d],delta,"DIRECT_PARTITION_REPAIR_DAMAGE_IDENTITY_FAILED")
                values.append(dict(score,**credit))
            delta={}
            sa=next(x for x in summaries[candidate]["per_seed"] if x["seed"] == seed)
            sb=next(x for x in summaries[reference]["per_seed"] if x["seed"] == seed)
            for metric in ("Harrell","Uno60"):
                delta["delta_"+metric]=None if sa[metric] is None or sb[metric] is None else sa[metric]-sb[metric]
            credit={}
            for name,den in (("repair","Harrell_den"),("damage","Harrell_den"),("Uno_repair","Uno_den"),("Uno_damage","Uno_den")):
                denominator=math.fsum(v[den] for v in values)
                credit[name]=math.fsum(v[name]*v[den] for v in values if v[den]>0)/denominator if denominator else None
            equal(None if credit["repair"] is None else credit["repair"]-credit["damage"],delta["delta_Harrell"],"DIRECT_SEED_HARRELL_IDENTITY_FAILED")
            equal(None if credit["Uno_repair"] is None else credit["Uno_repair"]-credit["Uno_damage"],delta["delta_Uno60"],"DIRECT_SEED_UNO_IDENTITY_FAILED")
            per_seed.append(dict(seed=seed,**delta,**credit))
        result=dict(candidate=candidate,reference=reference,question=question,per_seed=per_seed)
        for name in ("delta_Harrell","delta_Uno60",*CREDITS):
            result[name]=statistics.fmean(r[name] for r in per_seed) if all(r[name] is not None for r in per_seed) else None
        h,u=result["delta_Harrell"],result["delta_Uno60"];known=h is not None and u is not None
        result.update(metric_status="DEFINED" if known else "INCONCLUSIVE_UNDEFINED_METRIC",
            joint_directional_support=known and h>0 and u>0,mixed_metric_direction=known and ((h>0)!=(u>0)),
            small_delta_is_weak_evidence=known and (abs(h)<.003 or abs(u)<.003),
            positive_both_seed_count=sum(r["delta_Harrell"] is not None and r["delta_Uno60"] is not None
                and r["delta_Harrell"]>0 and r["delta_Uno60"]>0 for r in per_seed))
        computed.append(result)
    return computed


def check_pairs(computed,declared,csv_rows):
    expected={(a,b) for a,b,_ in COMPARISONS}
    values=exact(declared,("candidate","reference"),expected,"EXACT_11_REPORTED_PAIR_COMPARISONS_REQUIRED")
    tabular=exact(csv_rows,("candidate","reference"),expected,"EXACT_11_PAIRED_CSV_ROWS_REQUIRED")
    for actual in computed:
        key=actual["candidate"],actual["reference"]
        for row,is_csv in ((values[key],False),(tabular[key],True)):
            require(row.get("question") == actual["question"] and row.get("metric_status") == actual["metric_status"],"PAIRED_QUESTION_OR_METRIC_STATUS_CHANGED")
            for name in ("delta_Harrell","delta_Uno60",*CREDITS):
                equal(numeric(row.get(name),nullable=True),actual[name],"PAIRED_REPORTED_AGGREGATE_CHANGED")
            for name in ("joint_directional_support","mixed_metric_direction","small_delta_is_weak_evidence"):
                require(flag(row.get(name),"PAIRED_DIRECTION_FLAG_INVALID") == actual[name],"PAIRED_DIRECTION_FLAG_CHANGED")
            require(numeric(row.get("positive_both_seed_count")) == actual["positive_both_seed_count"],"POSITIVE_SEED_COUNT_CHANGED")
        seeds=exact(values[key].get("per_seed"),("seed",),{(s,) for s in SEEDS},"PAIRED_REPORTED_THREE_SEEDS_REQUIRED")
        for seed in actual["per_seed"]:
            for name in ("delta_Harrell","delta_Uno60",*CREDITS):
                equal(seeds[(seed["seed"],)].get(name),seed[name],"PAIRED_REPORTED_SEED_AGGREGATE_CHANGED")


def gate_status(comparisons):
    refs={"ADD","B1","GATE1_WRONG_RNA","GATE1_WRONG_PATH"}
    gate=[r for r in comparisons if r["candidate"] == "GATE1" and r["reference"] in refs]
    require(len(gate) == 4,"FIXED_GATE1_FOUR_COMPARISONS_REQUIRED")
    return "INCONCLUSIVE_UNDEFINED_METRIC" if any(r["metric_status"] != "DEFINED" for r in gate) else \
        "DIRECTIONAL_PATIENT_CONTEXT_INCREMENT" if all(r["joint_directional_support"] for r in gate) else "CURRENT_GATE1_NOT_SUPPORTED"


def review(report_dir,output,*,export_dir=None):
    directory=Path(report_dir).resolve();exports=directory if export_dir is None else Path(export_dir).resolve()
    reader=Metadata(); result=reader.json(directory/"R7_RESULTS.json","R7_RESULTS.json",public=True)
    require(result.get("schema") == "bgb_en_r7_complete_report_v1" and result.get("status") == "COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS"
        and all(digest(result.get(k)) for k in ("contract_sha256","config_sha256","qualification_sha256","replay_sha256")),"COMPLETE_R7_ANONYMOUS_REPORT_REQUIRED")
    budget=dict(formal_neural_fits=99,formal_optimizer_updates=7920,source_refits=0,teacher_or_beta_refits=0,
        unsupervised_train_RMS_calibrations=9,zero_fit_baseline_reference_rows=18,diagnostics_optimizer_updates=0)
    require(result.get("budget") == budget and result.get("no_cross_partition_risk_ranking") is True
        and result.get("seed_means_are_not_independent_cohorts") is True,"R7_DECLARED_BUDGET_OR_AGGREGATION_CHANGED")
    grid=metric_grid(reader.csv(directory/"R7_FOLD_METRICS.csv","R7_FOLD_METRICS.csv"));summaries=aggregate(grid)
    check_summaries(summaries,result.get("primary_DSS"),"DSS")
    audit=reader.json(exports/EXPORT_NAME,EXPORT_NAME,public=True)
    require(audit.get("schema") == "bgb_en_r7_postfit_metadata_export_v1" and audit.get("status") == "PASS_ANONYMOUS_METADATA_EXPORT"
        and audit.get("contract_sha256") == result["contract_sha256"] and audit.get("source_replay_sha256") == result["replay_sha256"]
        and audit.get("direct_rows") == 99,"DIRECT_EXPORT_SOURCE_REPLAY_BINDING_CHANGED")
    require(audit.get("generator") == dict(file="scripts/bgb_en_r7_independent_review.py",sha256=sha(Path(__file__).read_bytes())),"POSTFIT_EXPORT_GENERATOR_CHANGED")
    require(audit.get("validated_upstream_certificate") == dict(neural_units=99,checkpoint_records=297,zero_fit_reference_records=18,
        actual_forward_flags_checked=True,independent_model_forwards_performed_here=0),"UPSTREAM_REPLAY_METADATA_COUNTS_CHANGED")
    require(audit.get("runtime") == dict(optimizer_fits=0,optimizer_updates=0,model_forwards=0,backward_calls=0,
        model_checkpoint_array_label_files_read=False,only_JSON_metadata_read=True),"POSTFIT_EXPORT_RUNTIME_SCOPE_CHANGED")
    source_labels={("upstream_replay_certificate",)}
    if audit.get("secondary_rows") == 117:
        source_labels|={(f"completion_{a}_p{p}_seed{s}",) for a in ARMS for p in PARTITIONS for s in SEEDS}
        source_labels|={(f"reference_{a}_p{p}_seed{s}",) for a in ("B0","B1") for p in PARTITIONS for s in SEEDS}
    bound_sources=exact(audit.get("source_metadata"),("source",),source_labels,"POSTFIT_EXPORT_METADATA_SOURCE_INVENTORY_CHANGED")
    require(all(digest(s.get("sha256")) and type(s.get("bytes")) is int and s["bytes"]>0 for s in bound_sources.values())
        and bound_sources[("upstream_replay_certificate",)]["sha256"] == result["replay_sha256"],"POSTFIT_EXPORT_METADATA_SOURCE_SHA_CHANGED")
    artifacts=exact(audit.get("artifacts"),("file",),{(DIRECT_NAME,)}|({(SECONDARY_NAME,)} if audit.get("secondary_rows") == 117 else set()),"EXACT_POSTFIT_EXPORT_ARTIFACTS_REQUIRED")
    def exported(name):
        rows=reader.csv(exports/name,name);source=reader.sources[name];bound=artifacts[(name,)]
        require(bound.get("sha256") == source["sha256"] and bound.get("bytes") == source["bytes"] and bound.get("rows") == len(rows),"EXPORTED_ANONYMOUS_CSV_BYTES_CHANGED")
        return rows
    comparisons=paired(grid,summaries,exported(DIRECT_NAME))
    check_pairs(comparisons,result.get("paired_comparisons"),reader.csv(directory/"R7_PAIRED_COMPARISONS.csv","R7_PAIRED_COMPARISONS.csv"))
    secondary_status="NOT_EXPORTED_NOT_INDEPENDENTLY_RECOMPUTED"
    if audit.get("secondary_rows") == 117:
        require(audit.get("secondary_endpoint") == "PFI_frozen_DSS_risk","SECONDARY_ENDPOINT_CHANGED")
        secondary=aggregate(metric_grid(exported(SECONDARY_NAME)))
        check_summaries(secondary,result.get("secondary_PFI_frozen_DSS_risk"),"PFI")
        secondary_status="PASS_117_FROZEN_DSS_RISK_PFI_ROWS"
    else:
        require(audit.get("secondary_rows") == 0,"SECONDARY_EXPORT_ROW_COUNT_INVALID")
    decision=result.get("decision",{});status=gate_status(comparisons)
    require(decision.get("status") == status and decision.get("primary_candidate") == "GATE1"
        and decision.get("GATE2_selected_from_validation") is False and decision.get("complete_formal_fits") == 99
        and decision.get("complete_actual_checkpoint_replays") == 297 and decision.get("outer_results") == "NOT_RUN"
        and decision.get("CI_or_pvalues_computed") is False and decision.get("PFI_used_for_selection") is False,
        "FIXED_GATE1_DECISION_OR_SCOPE_CHANGED")
    gate_keys={("GATE1",r) for r in ("ADD","B1","GATE1_WRONG_RNA","GATE1_WRONG_PATH")}
    gate=exact(decision.get("gate_comparisons"),("candidate","reference"),gate_keys,"FIXED_GATE1_COMPARISON_SET_CHANGED")
    declared={(r["candidate"],r["reference"]):r for r in result["paired_comparisons"]}
    require(all(gate[key] == declared[key] for key in gate_keys),"DECISION_GATE_COMPARISON_COPY_CHANGED")
    expected={(a,p) for a in ARMS for p in PARTITIONS};index=result.get("diagnostic_index",{})
    require(index.get("logical_cells") == 99 and index.get("epochs_per_cell") == [20,40,60,80],"DIAGNOSTIC_INDEX_SCOPE_CHANGED")
    descriptors=exact(index.get("parts"),("file",),{(f"R7_DIAGNOSTICS_{a}_p{p}.json",) for a,p in expected},"EXACT_33_DIAGNOSTIC_SHARDS_REQUIRED")
    for arm,part in sorted(expected):
        name=f"R7_DIAGNOSTICS_{arm}_p{part}.json";bound=descriptors[(name,)]
        shard=reader.json(directory/name,name,public=True);source=reader.sources[name]
        require(bound.get("rows") == 3 and bound.get("bytes") == source["bytes"] <= 2*1024**2 and bound.get("sha256") == source["sha256"],"DIAGNOSTIC_SHARD_BYTES_OR_SHA_CHANGED")
        require(shard.get("schema") == "bgb_en_r7_diagnostic_shard_v1","DIAGNOSTIC_SHARD_SCHEMA_CHANGED")
        cells=exact(shard.get("rows"),("arm","partition","seed"),{(arm,part,s) for s in SEEDS},"DIAGNOSTIC_SHARD_THREE_MODEL_ROWS_REQUIRED")
        require(all(set(c.get("checkpoint_diagnostics",{})) == {"20","40","60","80"} for c in cells.values()),"DIAGNOSTIC_FOUR_EPOCHS_REQUIRED")
    output=safe_output(output,REVIEW_NAMES)
    review_result=dict(schema="bgb_en_r7_independent_arithmetic_review_v1",status="PASS_ANONYMOUS_ARITHMETIC_AND_HASH_BINDINGS",
        generator=dict(file="scripts/bgb_en_r7_independent_review.py",sha256=sha(Path(__file__).read_bytes())),
        contract_sha256=result["contract_sha256"],source_files=list(reader.sources.values()),
        verified_counts=dict(arms=13,primary_fold_rows=117,paired_comparisons=11,direct_partition_seed_rows=99,
            diagnostic_shards=33,diagnostic_model_rows=99),recomputed_primary_DSS=summaries,recomputed_pair_comparisons=comparisons,
        fixed_gate1_status=status,secondary_review=secondary_status,declared_upstream_budget=budget,
        upstream_replay_metadata=dict(neural_units=99,actual_checkpoint_records=297,reference_records=18,
            source_replay_sha256=result["replay_sha256"],actual_model_replay_performed_by_this_script=False),
        runtime=dict(optimizer_fits=0,optimizer_updates=0,model_forwards=0,backward_calls=0,
            model_checkpoint_array_label_files_read=False,patient_risk_reconstruction=False),
        limits=dict(arithmetic_checks_do_not_replay_weights=True,upstream_PASS_flags_are_not_new_actual_forward_evidence=True,
            seed_means_are_not_independent_cohorts=True,source_TCGA_patient_exclusivity="UNVERIFIED",no_new_CI_pvalues_or_selection=True))
    write_json(output/REVIEW_NAMES[0],review_result)
    lines=["# R7 独立匿名算术复核","",f"状态：**{review_result['status']}**。固定 GATE1 判定：`{status}`。",
        "","核对13模型×9单元=117行；每seed先合并3分区的Harrell/Uno分子分母，再平均3seeds，未拼接患者风险。",
        "11项比较×9直接配对记录分别按Harrell/Uno分母加权；逐分区和逐seed核验repair−damage=指标差。None保留为无定义并给出INCONCLUSIVE，未替换成0。",
        f"33份诊断分片的字节数、SHA、模型身份和4个诊断epoch齐全。次要终点：`{secondary_status}`。",
        "","预算99拟合/7920更新、18条零拟合引用及297份actual-checkpoint记录来自上游PASS收据，本脚本验证其匿名metadata绑定；**本脚本没有重新执行297次模型回放**。合成测试fixture也不构成正式拟合或实际回放证据。",
        "","优势是独立算术、逐分区直接credit和原字节绑定；局限是不能仅凭匿名统计重新确认权重的数值前向、患者无重叠或生物学真值。没有新训练、数组/权重/标签读取、风险重建、CI/p值或模型选择。"]
    with (output/REVIEW_NAMES[1]).open("x",encoding="utf-8") as stream:stream.write("\n".join(lines)+"\n")
    return review_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("export-direct","review"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--replay-audit",type=Path)
    parser.add_argument("--private-root",type=Path)
    parser.add_argument("--report-dir",type=Path)
    parser.add_argument("--export-dir",type=Path)
    args=parser.parse_args()
    if args.mode == "export-direct":
        require(args.replay_audit is not None,"REPLAY_AUDIT_ARGUMENT_REQUIRED")
        value=export_direct(args.replay_audit,args.output,private_root=args.private_root)
    else:
        require(args.report_dir is not None,"REPORT_DIRECTORY_ARGUMENT_REQUIRED")
        value=review(args.report_dir,args.output,export_dir=args.export_dir)
    print(json.dumps(dict(status=value["status"],optimizer_fits=0,model_forwards=0)))


if __name__ == "__main__":main()

"""Seven fixed R6 diagnostic plots from anonymous JSON/CSV reports only.

No checkpoints, arrays, patients, labels, model calls, fitting or selection.
One axes per figure; SVG/PNG pairs are saved from the same matplotlib figure.
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
import textwrap
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (("S0", "MATCHED"), ("S1", "MATCHED"), ("S2", "MATCHED"),
    ("S3", "MATCHED"), ("S4", "MATCHED"), ("S5", "WRONG_DONOR"),
    ("S4", "WITHIN_SECTION_TISSUE_DEPTH"), ("S4", "RANDOM_GATE"),
    ("S6", "MATCHED"), ("S6", "WRONG_NEIGHBOR"), ("S7", "MATCHED"),
    ("S7", "WRONG_SPOT_ALIGNMENT"))
ARMS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B11", "B13")
TARGET_ARMS = ("B0", "B1", *ARMS)
SEEDS, PARTITIONS = (31, 47, 83), (0, 1, 2)
PROGRAMS = ("iCMS2", "iCMS3", "FAP_fibroblast_associated_top25", "MARCO_macrophage_associated_top25")
REPRESENTATIONS = ("FIXED_HALLMARK_SUM50", "FIXED_HALLMARK_MEAN50", "ST_GENE_UNION_RAW760",
    "ST_SIGNED_PROGRAM_MEAN4", "FROZEN_S7_FLAT256", "FROZEN_S7_MEAN64",
    "LEGACY_RNA512_RIDGE", "LEGACY_FULL_RNA20511_RIDGE", "LEGACY_RNA512_PCA256_UNWHITENED_GENE_L2")
BASENAMES = ("source_huber", "source_within_tissue_rho", "retention_Harrell",
    "target_Harrell_delta_B1", "target_Uno60_delta_B1", "risk_branch_SD", "query_attention_TV")
MANIFEST_NAME, README_NAME = "R6_FIGURE_BINDINGS.json", "FIGURE_README.md"
MAX_JSON_BYTES = 2 * 1024**2
COLORS = ("#3B6EA8", "#CB6D34", "#4C916F", "#9465A7", "#A59439", "#70858D")
SVG_HASHSALT = "bgb_en_r6_anonymous_diagnostics_v1"
X_UNITS = dict(source_huber="source-model program Huber loss",
    source_within_tissue_rho="Spearman correlation coefficient (dimensionless)",
    retention_Harrell="Harrell concordance probability (dimensionless)",
    target_Harrell_delta_B1="difference in Harrell concordance (dimensionless)",
    target_Uno60_delta_B1="difference in Uno60 concordance (dimensionless)",
    risk_branch_SD="standard deviation of Cox log-risk correction (dimensionless)",
    query_attention_TV="total variation distance including null mass (dimensionless; range 0 to 1)")


def require(value, code):
    if not value:
        raise ValueError(code)


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def canonical(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def scalar(value, code, *, low=None, high=None, nullable=False):
    if value is None and nullable:
        return None
    require(finite(value) and (low is None or value >= low) and (high is None or value <= high), code)
    return float(value)


def strict_mean(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def equal_number(left, right, code):
    require((left is None and right is None) or (finite(left) and finite(right) and abs(left - right) <= 1e-10), code)


def exact_rows(rows, fields, expected, code):
    require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), code)
    keys = [tuple(row.get(k) for k in fields) for row in rows]
    require(len(keys) == len(expected) and len(set(keys)) == len(keys) and set(keys) == set(expected), code)
    return dict(zip(keys, rows))


class Reports:
    """Only explicit anonymous report filenames; no discovery of private files."""
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        require(self.directory.is_dir(), "ANONYMOUS_REPORT_DIRECTORY_REQUIRED")
        self.sources = {}

    def bytes(self, name):
        require(isinstance(name, str) and Path(name).name == name and "/" not in name and "\\" not in name,
            "FIXED_REPORT_BASENAME_REQUIRED")
        path = self.directory / name
        require(path.is_file() and not path.is_symlink() and path.resolve().parent == self.directory,
            "ANONYMOUS_REPORT_FILE_ABSENT_OR_UNSAFE")
        require(path.stat().st_size <= MAX_JSON_BYTES, "ANONYMOUS_REPORT_EXCEEDS_2MIB_USE_PRODUCER_SHARDS")
        before = path.stat()
        content = path.read_bytes()
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "REPORT_CHANGED_DURING_READ")
        record = dict(file=name, bytes=len(content), sha256=sha_bytes(content))
        require(name not in self.sources or self.sources[name] == record, "REPORT_CHANGED_BETWEEN_READS")
        self.sources[name] = record
        return content

    def json(self, name):
        def invalid_constant(_):
            raise ValueError("NONFINITE_REPORT_JSON")
        value = json.loads(self.bytes(name).decode("utf-8-sig"), parse_constant=invalid_constant)
        require(isinstance(value, dict), "REPORT_JSON_OBJECT_REQUIRED")
        return value

    def csv(self, name):
        return list(csv.DictReader(io.StringIO(self.bytes(name).decode("utf-8-sig"))))

    def unchanged(self):
        for name in list(self.sources):
            self.bytes(name)


def csv_float(value, *, nullable=False):
    if value in (None, "", "None") and nullable:
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError("SOURCE_CSV_NUMBER_INVALID") from None
    require(math.isfinite(result), "SOURCE_CSV_NUMBER_NONFINITE")
    return result


def components(row):
    result = {}
    for label, numerator, denominator in (("Harrell", "Harrell_num", "Harrell_den"), ("Uno60", "Uno_num", "Uno_den")):
        n = scalar(row.get(numerator), "CONCORDANCE_COMPONENT_INVALID", low=0)
        d = scalar(row.get(denominator), "CONCORDANCE_COMPONENT_INVALID", low=0)
        require(n <= d + 1e-10, "CONCORDANCE_NUMERATOR_EXCEEDS_DENOMINATOR")
        value = n / d if d else None
        equal_number(row.get(label), value, "CONCORDANCE_RATIO_CHANGED")
        result.update({label: value, numerator: n, denominator: d})
    return result


def group(name, color, values, *, marker="o", offset=0., means=None):
    return dict(name=name, color=color, values=values, marker=marker, offset=offset, means=means)


def plan(basename, title, xlabel, labels, groups, means, note, sources, table, aggregation, point_unit, *, zero=False):
    return dict(basename=basename, title=title, xlabel=xlabel, labels=labels, groups=groups,
        means=means, note=note, sources=list(sources), source_table=table, aggregation=aggregation,
        point_unit=point_unit, zero_line=zero)


def source_plans(reports):
    report_name, csv_name = "R6_1_SOURCE_SUPERVISION.json", "R6_SOURCE_CELL_METRICS.csv"
    report = reports.json(report_name)
    require(report.get("schema") == "bgb_en_r6_source_supervision_report_v1"
        and report.get("status") == "COMPLETE_147_SOURCE_UNITS_AND_REPLAY", "SOURCE_MATRIX_NOT_COMPLETE")
    families = exact_rows(report.get("complete_family_summaries"), ("arm", "control"), set(FAMILIES), "SOURCE_TWELVE_FAMILIES_REQUIRED")
    rows = reports.csv(csv_name)
    cells = {}
    expected = {(a, c, d, s) for a, c in FAMILIES for d in range(6) for s in (31, 47)}
    for row in rows:
        try:
            key = row["arm"], row["control"], int(row["heldout_donor"]), int(row["seed"])
            subkey = row["program"], int(row["tissue_class"])
        except (KeyError, ValueError, TypeError):
            raise ValueError("SOURCE_METRIC_IDENTITY_INVALID") from None
        require(key in expected and row.get("status") == "PASS_COMPLETE_AND_REPLAYED"
            and subkey[0] in PROGRAMS and subkey[1] in (0, 1, 2), "SOURCE_METRIC_IDENTITY_INVALID")
        cell = cells.setdefault(key, {})
        require(subkey not in cell, "SOURCE_DUPLICATE_PROGRAM_TISSUE_ROW")
        cell[subkey] = row
    require(set(cells) == expected, "SOURCE_144_LODO_CELLS_REQUIRED")
    scores = {}
    for key, cell in cells.items():
        require(set(cell) == {(p, t) for p in PROGRAMS for t in range(3)}, "SOURCE_FOUR_PROGRAM_THREE_TISSUE_GRID_REQUIRED")
        huber, rho = [], []
        for program in PROGRAMS:
            values = [csv_float(cell[(program, t)]["program_global_Huber"]) for t in range(3)]
            require(values[0] >= 0 and all(x == values[0] for x in values), "SOURCE_GLOBAL_HUBER_REPETITIONS_CHANGED")
            huber.append(values[0])
        for row in cell.values():
            require(row.get("eligible_for_within_tissue_rho") in ("True", "False"), "SOURCE_RHO_ELIGIBILITY_UNRESOLVED")
            if row["eligible_for_within_tissue_rho"] == "True":
                value = csv_float(row.get("Spearman"), nullable=True)
                require(value is None or -1 <= value <= 1, "SOURCE_SPEARMAN_RANGE_INVALID")
                rho.append(value)
        scores[key] = dict(program_Huber=statistics.mean(huber), within_tissue_Spearman=strict_mean(rho))
    labels = [a + " / " + {"MATCHED": "matched", "WRONG_DONOR": "wrong donor",
        "WITHIN_SECTION_TISSUE_DEPTH": "within-tissue depth", "RANDOM_GATE": "random gate",
        "WRONG_NEIGHBOR": "wrong neighbor", "WRONG_SPOT_ALIGNMENT": "wrong spot"}[c] for a, c in FAMILIES]
    plans = []
    for metric, basename, title, xlabel in (("program_Huber", "source_huber", "Source program prediction", "Mean program Huber loss (lower is better)"),
        ("within_tissue_Spearman", "source_within_tissue_rho", "Source within-tissue association", "Mean within-tissue Spearman rho")):
        donor_points = []
        means = []
        for arm, control in FAMILIES:
            summary = families[(arm, control)]
            require(summary.get("status") == "COMPLETE_12_LODO_UNITS" and summary.get("complete_replayed_units") == 12,
                "SOURCE_FAMILY_NOT_COMPLETE")
            values = [strict_mean([scores[(arm, control, donor, seed)][metric] for seed in (31, 47)]) for donor in range(6)]
            aggregate = strict_mean(values)
            equal_number(summary.get(metric), aggregate, "SOURCE_FAMILY_AGGREGATION_CHANGED")
            donor_points.append(values); means.append(aggregate)
        groups = [group("Donor " + str(d + 1), COLORS[d], [[row[d]] for row in donor_points]) for d in range(6)]
        plans.append(plan(basename, title, xlabel, labels, groups, means,
            "Colored points: two-seed mean for each held-out donor. Black diamond: equal-donor mean. Six donors are reused across families; no CI or p-values.",
            (report_name, csv_name), "complete_family_summaries + R6_SOURCE_CELL_METRICS",
            "four global program Hubers / eligible tissue-program rho; equal two seeds within donor, then equal six donors",
            "one held-out-donor mean, not 12 independent seed fits", zero=metric == "within_tissue_Spearman"))
    return plans


def retention_plan(reports):
    name = "R6_2_OMICS_RETENTION.json"; report = reports.json(name)
    require(report.get("schema") == "bgb_en_r6_retention_report_v1"
        and report.get("status") == "COMPLETE_18_LOGICAL_CELLS_AND_REPLAY", "RETENTION_MATRIX_NOT_COMPLETE")
    expected = {(r, p) for r in REPRESENTATIONS for p in PARTITIONS}
    rows = exact_rows(report.get("rows"), ("representation", "partition"), expected, "RETENTION_EXACT_27_ROWS_REQUIRED")
    summaries = exact_rows(report.get("summaries"), ("representation",), {(r,) for r in REPRESENTATIONS}, "RETENTION_NINE_SUMMARIES_REQUIRED")
    points, means = [], []
    for representation in REPRESENTATIONS:
        metrics = []
        for part in PARTITIONS:
            row = rows[(representation, part)]
            historical = representation.startswith("LEGACY_")
            require(row.get("status") == ("REUSED_EXACT_PRIOR_ACTUAL_REPLAY" if historical else "COMPLETE_AND_REPLAYED"),
                "RETENTION_ROW_NOT_COMPLETE")
            expected_lambda = 10. if representation == "LEGACY_FULL_RNA20511_RIDGE" else 1.
            require(row.get("lambda_L2") == expected_lambda, "RETENTION_REGULARIZATION_ANNOTATION_CHANGED")
            metrics.append(components(row.get("metrics", {})))
        numerator = sum(m["Harrell_num"] for m in metrics); denominator = sum(m["Harrell_den"] for m in metrics)
        aggregate = numerator / denominator if denominator else None
        summary = summaries[(representation,)]
        require(summary.get("complete_partitions") == 3 and summary.get("status") == "COMPLETE", "RETENTION_SUMMARY_NOT_COMPLETE")
        equal_number(summary.get("Harrell"), aggregate, "RETENTION_COMPONENT_AGGREGATION_CHANGED")
        means.append(aggregate); points.append([m["Harrell"] for m in metrics])
    labels = ["Hallmark sum50", "Hallmark mean50", "ST gene union raw760", "ST signed program mean4", "Frozen S7 flat256",
        "Frozen S7 mean64", "RNA512 (historical; L2=1)", "Full RNA20511 (historical; L2=10)", "RNA512 PCA256 (historical; gene-L2=1)"]
    groups = [group("Partition " + str(p + 1), COLORS[p], [[row[p]] for row in points]) for p in PARTITIONS]
    return plan("retention_Harrell", "RNA representation retention", "Within-partition Harrell concordance", labels, groups, means,
        "Colored points: three partition scores. Black diamond: sum of within-partition numerators / sum of denominators. Historical full RNA uses L2=10; other readouts use L2=1. No cross-partition risk ranking.",
        (name,), "rows.metrics and summaries.Harrell", "sum within-partition concordance components; no risk concatenation",
        "one partition score; black diamond is component-pooled, not mean C-index")


def target_plans(reports):
    name = "R6_4_INNER_MATRIX.json"; report = reports.json(name)
    require(report.get("schema") == "bgb_en_r6_target_inner_report_v1"
        and report.get("counts") == {"PASS_FIT_AND_ACTUAL_REPLAY": 90, "PASS_ZERO_FIT_REFERENCE": 18}
        and report.get("actual_replay", {}).get("complete") is True
        and report.get("decision", {}).get("complete_source_and_target_evidence") is True,
        "TARGET_MATRIX_NOT_COMPLETE_AND_REPLAYED")
    require(report.get("source", {}).get("report_sha256") == reports.sources["R6_1_SOURCE_SUPERVISION.json"]["sha256"],
        "TARGET_SOURCE_REPORT_BINDING_CHANGED")
    rows = exact_rows(report.get("primary_DSS"), ("arm",), {(a,) for a in TARGET_ARMS}, "TARGET_TWELVE_ARMS_REQUIRED")
    per_arm = {}
    for arm in TARGET_ARMS:
        row = rows[(arm,)]
        require(row.get("status") == "COMPLETE_NINE_CELLS" and row.get("complete_cells") == 9
            and row.get("cross_partition_risk_ranking") is False, "TARGET_ARM_SUMMARY_INCOMPLETE")
        seeds = exact_rows(row.get("per_seed"), ("seed",), {(s,) for s in SEEDS}, "TARGET_THREE_SEED_COMPONENTS_REQUIRED")
        per_arm[arm] = {seed: components(seeds[(seed,)]) for seed in SEEDS}
        for metric in ("Harrell", "Uno60"):
            equal_number(row.get(metric), strict_mean([per_arm[arm][s][metric] for s in SEEDS]), "TARGET_SEED_MEAN_CHANGED")
    plans = []
    for metric in ("Harrell", "Uno60"):
        points = [[per_arm[arm][s][metric] - per_arm["B1"][s][metric]
            if per_arm[arm][s][metric] is not None and per_arm["B1"][s][metric] is not None else None for s in SEEDS] for arm in TARGET_ARMS]
        means = [strict_mean(row) for row in points]
        groups = [group("Seed " + str(s), COLORS[j], [[row[j]] for row in points]) for j, s in enumerate(SEEDS)]
        plans.append(plan("target_" + metric + "_delta_B1", "Target DSS ranking relative to B1", "Delta " + metric + " versus B1",
            list(TARGET_ARMS), groups, means,
            "Colored points: per-seed delta after summing three within-partition component counts. Black diamond: equal-three-seed mean. B0/B1 reuse fixed risks across seeds. These are inner-development results, not independent cohorts; no CI.",
            (name,), "primary_DSS.per_seed concordance components", "within-partition components summed per seed, then paired subtraction of B1, then equal seed mean",
            "one seed-level component-pooled comparison; repeated references are not refits", zero=True))
    return plans, report


def read_attention(reports):
    name = "R6_ATTENTION_DIAGNOSTICS.json"; before = set(reports.sources); index = reports.json(name)
    storage = {"storage_schema", "logical_object_sha256", "rows_shards", "rows_count"}
    require(index.get("storage_schema") == "bgb_en_r6_attention_diagnostics_sharded_index_v1"
        and "rows" not in index and index.get("rows_count") == 90, "ATTENTION_SHARDED_INDEX_REQUIRED")
    descriptors = exact_rows(index.get("rows_shards"), ("arm",), {(a,) for a in ARMS}, "ATTENTION_TEN_SHARDS_REQUIRED")
    rows = [None] * 90
    for (arm,), descriptor in descriptors.items():
        filename = f"R6_ATTENTION_DIAGNOSTICS_{arm}.json"
        require(descriptor.get("file") == filename and descriptor.get("row_count") == 9, "ATTENTION_SHARD_PATH_OR_COUNT_CHANGED")
        shard = reports.json(filename); record = reports.sources[filename]
        require(descriptor.get("sha256") == record["sha256"] and descriptor.get("bytes") == record["bytes"]
            and shard.get("schema") == "bgb_en_r6_attention_diagnostics_arm_shard_v1" and shard.get("arm") == arm
            and shard.get("contract_sha256") == index.get("contract_sha256")
            and shard.get("logical_object_sha256") == index.get("logical_object_sha256"), "ATTENTION_SHARD_HASH_CHANGED")
        exact_rows(shard.get("rows"), ("arm", "partition", "seed"), {(arm, p, s) for p in PARTITIONS for s in SEEDS}, "ATTENTION_SHARD_NINE_CELLS_REQUIRED")
        positions = descriptor.get("row_positions", [])
        require(len(positions) == len(set(positions)) == 9 and all(type(i) is int and 0 <= i < 90 for i in positions), "ATTENTION_ROW_POSITIONS_CHANGED")
        for position, row in zip(positions, shard["rows"]):
            require(rows[position] is None, "ATTENTION_DUPLICATE_POSITION")
            rows[position] = row
    require(all(r is not None for r in rows), "ATTENTION_MISSING_ROWS")
    value = {k: v for k, v in index.items() if k not in storage}; value["rows"] = rows
    require(canonical(value) == index.get("logical_object_sha256"), "ATTENTION_LOGICAL_SHA_CHANGED")
    return value, sorted(set(reports.sources) - before)


def read_risk(reports):
    name = "R6_RISK_DIAGNOSTICS.json"; before = set(reports.sources); index = reports.json(name)
    if index.get("schema") == "bgb_en_r6_risk_diagnostics_v1":
        return index, [name]
    require(index.get("schema") == "bgb_en_r6_risk_diagnostic_index_v1" and index.get("unit_count") == 90,
        "RISK_DIAGNOSTIC_INDEX_CHANGED")
    parts = exact_rows(index.get("parts"), ("arm",), {(a,) for a in ARMS}, "RISK_TEN_SHARDS_REQUIRED")
    units = []
    for (arm,), descriptor in parts.items():
        filename = f"R6_RISK_DIAGNOSTICS_{arm}.json"
        require(descriptor.get("file") == filename and descriptor.get("count") == 9, "RISK_SHARD_PATH_OR_COUNT_CHANGED")
        shard = reports.json(filename)
        require(descriptor.get("sha256") == reports.sources[filename]["sha256"]
            and shard.get("schema") == "bgb_en_r6_risk_diagnostic_arm_v1" and shard.get("arm") == arm
            and shard.get("contract_sha256") == index.get("logical_metadata", {}).get("contract_sha256"), "RISK_SHARD_HASH_CHANGED")
        exact_rows(shard.get("units"), ("arm", "partition", "seed"), {(arm, p, s) for p in PARTITIONS for s in SEEDS}, "RISK_SHARD_NINE_CELLS_REQUIRED")
        units.extend(shard["units"])
    units.sort(key=lambda r: (ARMS.index(r["arm"]), r["partition"], SEEDS.index(r["seed"])))
    value = dict(index["logical_metadata"], units=units)
    require(canonical(value) == index.get("logical_sha256"), "RISK_LOGICAL_SHA_CHANGED")
    return value, sorted(set(reports.sources) - before)


def diagnostic_plans(reports, target):
    risk, risk_sources = read_risk(reports)
    require(risk.get("status") == "COMPLETE_SAVED_METADATA_DIAGNOSTICS"
        and risk.get("contract_sha256") == target.get("contract_sha256")
        and risk.get("report_sha256") == reports.sources["R6_4_INNER_MATRIX.json"]["sha256"]
        and risk.get("model_or_array_or_label_files_read") is False, "RISK_DIAGNOSTICS_NOT_COMPLETE_OR_UNBOUND")
    expected = {(a, p, s) for a in ARMS for p in PARTITIONS for s in SEEDS}
    units = exact_rows(risk.get("units"), ("arm", "partition", "seed"), expected, "RISK_NINETY_CELLS_REQUIRED")
    main, cross = [], []
    for arm in ARMS:
        a, b = [], []
        for part in PARTITIONS:
            for seed in SEEDS:
                row = units[(arm, part, seed)]
                require(row.get("final_epoch") == 80 and row.get("cross_structurally_disabled") is (arm in ("B2", "B3")), "RISK_BRANCH_STRUCTURE_CHANGED")
                quantities = row.get("saved_scalar_distributions", {}).get("validation", {}).get("quantities", {})
                x = scalar(quantities.get("main_delta", {}).get("sd"), "MAIN_RISK_SD_UNAVAILABLE", low=0)
                y = scalar(quantities.get("cross_delta", {}).get("sd"), "CROSS_RISK_SD_UNAVAILABLE", low=0)
                if arm in ("B2", "B3"):
                    require(y == 0., "STRUCTURALLY_DISABLED_CROSS_SD_NONZERO")
                a.append(x); b.append(y)
        main.append(a); cross.append(b)
    groups = [group("Main effect (9 cell SDs + mean)", COLORS[0], main, offset=-.11, means=[statistics.mean(v) for v in main]),
        group("Interaction (9 cell SDs + mean)", COLORS[1], cross, marker="s", offset=.11, means=[statistics.mean(v) for v in cross])]
    risk_plan = plan("risk_branch_SD", "Validation risk-branch amplitude at epoch 80", "SD of log-risk correction (linear scale)",
        [a + (" [cross off]" if a in ("B2", "B3") else "") for a in ARMS], groups, None,
        "Small marks: saved SD within each validation cell; diamonds: equal-nine-cell mean. B2/B3 interaction is structurally zero. SD is not variance explained or a causal contribution; no log transform or artificial zero replacement.",
        risk_sources, "units.saved_scalar_distributions.validation.quantities.{main_delta,cross_delta}.sd",
        "saved per-cell validation SD and equal nine-cell mean, no patient risk reconstruction",
        "one trained partition-seed cell; two separate risk branches")
    attention, attention_sources = read_attention(reports)
    require(attention.get("schema") == "bgb_en_r6_attention_diagnostics_v1"
        and attention.get("status") == "COMPLETE_90_FIRST8_INFERENCE_DIAGNOSTICS"
        and attention.get("contract_sha256") == target.get("contract_sha256")
        and attention.get("replay_audit_sha256") == target.get("actual_replay", {}).get("sha256")
        and attention.get("unique_probed_patients") == 24 and attention.get("repeated_patient_evaluations") == 720
        and attention.get("independent_patient_count_is_not_720") is True
        and attention.get("fixed_first_valid_rows") == 8, "ATTENTION_SCOPE_OR_COMPLETENESS_CHANGED")
    units = exact_rows(attention.get("rows"), ("arm", "partition", "seed"), expected, "ATTENTION_NINETY_CELLS_REQUIRED")
    seed_groups, all_values = [], []
    for arm in ARMS:
        values = []
        for part in PARTITIONS:
            for seed in SEEDS:
                row = units[(arm, part, seed)]
                require(row.get("fixed_validation_rows") == 8, "ATTENTION_FIRST_EIGHT_SCOPE_CHANGED")
                summary = row.get("matched_wrong_patch_plus_null_TV", {})
                n = summary.get("n")
                require(type(n) is int and 0 <= n <= 8, "ATTENTION_ELIGIBLE_PATIENT_COUNT_INVALID")
                values.append(scalar(summary.get("mean"), "ATTENTION_TV_INVALID", low=0, high=1, nullable=n == 0))
                require(n != 0 or values[-1] is None, "EMPTY_ATTENTION_MEAN_MUST_BE_NA")
        all_values.append(values)
    for j, seed in enumerate(SEEDS):
        seed_groups.append(group("Seed " + str(seed) + " (3 cells)", COLORS[j], [[v[p * 3 + j] for p in PARTITIONS] for v in all_values]))
    attention_plan = plan("query_attention_TV", "Matched versus wrong-query attention sensitivity", "Patch + null total variation distance",
        list(ARMS), seed_groups, [strict_mean(v) for v in all_values],
        "Each point: cell-level mean over the fixed first 8 validation patients. The same 24 patients are reused across 90 models (720 evaluations), not 720 independent patients. Black diamond: equal-nine-cell mean; undefined cells stay NA.",
        attention_sources, "rows.matched_wrong_patch_plus_null_TV.mean", "patient-equal WSI averages from producer, then cell means; no pooling of patient arrays",
        "one partition-seed inference cell over fixed first-eight validation cases")
    return risk_plan, attention_plan


def build_plans(reports):
    plans = source_plans(reports)
    plans.append(retention_plan(reports))
    target, target_report = target_plans(reports); plans.extend(target)
    plans.extend(diagnostic_plans(reports, target_report))
    require(tuple(p["basename"] for p in plans) == BASENAMES, "FIXED_SEVEN_FIGURE_ORDER_CHANGED")
    reports.unchanged()
    return plans


def validate_svg(content):
    root = ET.fromstring(content)
    require(root.tag.endswith("}svg"), "OUTPUT_SVG_ROOT_REQUIRED")
    for node in root.iter():
        require(node.tag.rsplit("}", 1)[-1] != "image", "SVG_RASTER_IMAGE_FORBIDDEN")
        for key, value in node.attrib.items():
            if key.rsplit("}", 1)[-1] == "href":
                require(value.startswith("#"), "SVG_EXTERNAL_HREF_FORBIDDEN")
    require(b"http://www.w3.org/2000/svg" in content and b"<script" not in content,
        "SVG_STATIC_VECTOR_REQUIRED")


def draw(plan):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 15,
        "axes.labelsize": 11, "svg.hashsalt": SVG_HASHSALT, "svg.fonttype": "path",
        "figure.facecolor": "white", "axes.facecolor": "white"})
    figure = plt.figure(figsize=(11.5, 7.4))
    axes = figure.add_axes((.37, .27, .60, .61))
    n = len(plan["labels"])
    handles = []
    for gi, item in enumerate(plan["groups"]):
        require(len(item["values"]) == n, "PLOT_SERIES_DIMENSION_CHANGED")
        for y, values in enumerate(item["values"]):
            for j, value in enumerate(values):
                if value is None:
                    continue
                jitter = (j - (len(values) - 1) / 2) * .035 if len(values) > 1 else 0.
                # Keep equal-valued seed dots visible beside the aggregate.
                # These deterministic vertical offsets do not encode numbers.
                if plan["means"] is not None:
                    spacing = .05 if len(plan["groups"]) > 4 else .13
                    jitter += (gi - (len(plan["groups"]) - 1) / 2) * spacing
                axes.scatter([value], [y + item["offset"] + jitter], c=item["color"], marker=item["marker"],
                    s=24, alpha=.8, linewidths=.4, edgecolors="white", zorder=3)
            if item["means"] is not None and item["means"][y] is not None:
                axes.scatter([item["means"][y]], [y + item["offset"]], c=item["color"], marker="D", s=48,
                    edgecolors="black", linewidths=.5, zorder=5)
        handles.append(Line2D([], [], marker=item["marker"], color="none", markerfacecolor=item["color"],
            markeredgecolor=item["color"], markersize=5, label=item["name"]))
    if plan["means"] is not None:
        for y, value in enumerate(plan["means"]):
            if value is None:
                axes.text(.99, y, "NA", ha="right", va="center", transform=axes.get_yaxis_transform(), color="#777777", fontsize=8)
            else:
                axes.scatter([value], [y + .29], c="black", marker="D", s=42, zorder=6)
        handles.append(Line2D([], [], marker="D", color="none", markerfacecolor="black", markersize=5, label="Aggregate"))
    if plan["zero_line"]:
        axes.axvline(0, color="#777777", lw=.8, ls="--", zorder=1)
    axes.set_yticks(range(n), plan["labels"])
    axes.set_ylim(n - .5, -.5)
    axes.set_xlabel(plan["xlabel"])
    axes.set_title(plan["title"], loc="left", pad=14)
    axes.grid(axis="x", color="#DFE3E8", lw=.6)
    axes.set_axisbelow(True)
    axes.tick_params(axis="y", length=0, pad=8)
    axes.spines[["top", "right", "left"]].set_visible(False)
    axes.spines["bottom"].set_color("#9AA1A9")
    figure.legend(handles=handles, loc="lower left", bbox_to_anchor=(.365, .13), frameon=False,
        ncol=min(4, len(handles)), fontsize=9, handletextpad=.45, columnspacing=1.2)
    figure.text(.035, .038, "\n".join(textwrap.wrap(plan["note"], width=143)), fontsize=9,
        color="#47515D", va="bottom", linespacing=1.35)
    return figure


def _output_record(path, format):
    content = path.read_bytes()
    return dict(file=path.name, format=format, bytes=len(content), sha256=sha_bytes(content))


def render(report_dir, output):
    reports = Reports(report_dir)
    plans = build_plans(reports)  # No output is created until all evidence passes.
    output = Path(output).resolve()
    require(not any(output == p.resolve() or output.is_relative_to(p.resolve())
        for p in (ROOT / "reports/bgb_en", ROOT / "reports/bgb_en_r5")), "HISTORICAL_REPORT_NAMESPACES_PROTECTED")
    names = [basename + "." + extension for basename in BASENAMES for extension in ("svg", "png")] + [MANIFEST_NAME, README_NAME]
    require(not any((output / name).exists() for name in names), "FIGURE_OUTPUTS_EXIST_NO_OVERWRITE")
    output.mkdir(parents=True, exist_ok=True)
    figures = []
    generator = dict(file="scripts/bgb_en_r6_plot_diagnostics.py", sha256=sha_bytes(Path(__file__).read_bytes()))
    for item in plans:
        figure = draw(item)
        try:
            svg = output / (item["basename"] + ".svg"); png = output / (item["basename"] + ".png")
            with svg.open("xb") as stream:
                figure.savefig(stream, format="svg", metadata={"Date": None, "Creator": "BG-BEN R6 diagnostic plotter"})
            validate_svg(svg.read_bytes())
            with png.open("xb") as stream:
                figure.savefig(stream, format="png", dpi=160, metadata={"Software": "BG-BEN R6 diagnostic plotter"})
        finally:
            plt.close(figure)
        figures.append(dict(basename=item["basename"], title=item["title"], source_files=item["sources"],
            source_table=item["source_table"], aggregation=item["aggregation"], point_unit=item["point_unit"],
            axes=dict(x=item["xlabel"], x_unit=X_UNITS[item["basename"]], x_scale="linear",
                y="anonymous model / source-family / representation code", y_unit="categorical anonymous code"),
            uncertainty_intervals=False, pvalues=False, one_axes=True, outputs=[_output_record(svg, "SVG"), _output_record(png, "PNG")]))
    reports.unchanged()
    require(generator["sha256"] == sha_bytes(Path(__file__).read_bytes()), "GENERATOR_CHANGED_DURING_RENDER")
    readme = """# R6 普通诊断图

本目录只读取匿名报告生成普通统计诊断图，不执行训练、模型前向、患者级风险重算或模型选择。每个 SVG/PNG 配对来自同一个 matplotlib figure，均为单坐标轴；没有置信区间或 p 值。

| 文件 basename | 统计口径 |
|---|---|
| source_huber | 四个程序的全局 Huber 先在程序内等权；两 seed 先在供体内平均，再展示六供体点及供体等权总体值。 |
| source_within_tissue_rho | 沿用模型无关的组织/程序 eligibility；先组内 rho，再供体内两 seed 平均。12 次拟合不是12名独立供体。 |
| retention_Harrell | 九种表示的三个分区点；总体值为各分区 Harrell 分子和除以分母和。旧 fullRNA20511 固定 L2=10，其他为1。 |
| target_Harrell_delta_B1 | 每 seed 先合并三个分区的分子分母，再与同 seed B1 作差；展示三 seed 点及其等权均值。 |
| target_Uno60_delta_B1 | 与上图相同，使用既有 Uno60 加权分子/分母；不拼接风险后跨分区排序。 |
| risk_branch_SD | 第80轮 validation 中 main/cross log-risk 修正的已存标准差，展示九 cell 点与各自均值；线性坐标，B2/B3 交互结构零保持为零。 |
| query_attention_TV | 每 cell 使用固定前8验证患者的 patch+null TV 平均值；同一24名患者反复用于90模型，720次评价不等于720名独立患者。 |

图内小点只描述供体、分区、seed 或模型单元的离散情况，不构成独立外部验证。SD不是解释方差比例；注意力敏感性不是组织学真值或因果证据。缺失或无定义值显示NA，不用零替换。未完成矩阵拒绝生成整套图，已有同名产物拒绝覆盖。

`R6_FIGURE_BINDINGS.json` 绑定所有实际读取匿名文件的字节数/SHA、绘图器SHA、各图来源与轴单位，以及14个图文件的字节数/SHA。SVG只有向量图元和字体定义，不包含栅格 image 或外部 href。
"""
    with (output / README_NAME).open("x", encoding="utf-8") as stream:
        stream.write(readme)
    binding = dict(schema="bgb_en_r6_figure_bindings_v1", status="COMPLETE_7_ANONYMOUS_DIAGNOSTIC_FIGURES",
        figure_count=7, generator=generator, source_files=[reports.sources[name] for name in sorted(reports.sources)],
        figures=figures, readme=_output_record(output / README_NAME, "Markdown"),
        rendering=dict(backend="Agg", matplotlib_version=matplotlib.__version__, svg_hashsalt=SVG_HASHSALT,
            svg_fonttype="path", svg_metadata_Date=None, pair_saved_from_same_figure=True),
        runtime=dict(optimizer_fits=0, optimizer_updates=0, model_forwards=0, backward_calls=0,
            model_array_patient_or_label_files_read=False, new_risk_computation=False, network_used=False),
        limits=dict(no_CI_or_pvalues=True, no_model_or_epoch_selection=True, not_independent_external_validation=True))
    with (output / MANIFEST_NAME).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(binding, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    binding = render(args.report_dir, args.output)
    print(json.dumps(dict(status=binding["status"], figures=binding["figure_count"], optimizer_fits=0, model_forwards=0)))


if __name__ == "__main__":
    main()

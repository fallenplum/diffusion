"""Independent SPGEL aggregate arithmetic; no model, labels, or optimizer API."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(path):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(data["status"] == "COMPLETE_REGISTERED_SOURCE_PILOT_AND_REPLAY", "INCOMPLETE_SOURCE_REPORT")
    rows = data["donor_rows"]
    keys = {(r["arm"], r["heldout_donor_code"], r["seed"]): r for r in rows}
    expected = {(f"S_A{a}", d, s) for a in range(6) for d in range(6) for s in (31, 47)}
    require(len(rows) == 72 and set(keys) == expected, "EXACT_72_SOURCE_CELLS_REQUIRED")
    errors = []
    scalar_checks = 0

    def same(a, b):
        nonlocal scalar_checks
        scalar_checks += 1
        require((a is None) == (b is None), "UNDEFINED_METRIC_CHANGED")
        if a is not None:
            require(math.isfinite(a) and math.isfinite(b), "NONFINITE_METRIC")
            errors.append(abs(a-b))

    def mean(values):
        values = list(values)
        kept = [v for v in values if v is not None]
        return dict(mean=math.fsum(kept)/len(kept) if kept else None,
                    defined_count=len(kept), expected_count=len(values),
                    undefined_count=len(values)-len(kept), undefined_is_not_zero=True)

    def compare_mean(values, saved):
        actual = mean(values)
        for k, v in actual.items():
            if k == "mean":
                same(v, saved[k])
            else:
                require(v == saved[k], "METRIC_SUPPORT_CHANGED")

    for row in rows:
        a = row["task_A"]["native"]["per_output"]
        c = row["task_C"]["per_output"]
        require(len(a) == len(c) == 4, "FOUR_PROGRAMS_REQUIRED")
        values = [p["Pearson"] for p in a]
        strict = math.fsum(values)/4 if all(v is not None for v in values) else None
        same(strict, row["task_A"]["primary_donor_four_program_Pearson"])
        values = [p["AUPRC"] for p in c]
        strict = math.fsum(values)/4 if all(v is not None for v in values) else None
        same(strict, row["task_C"]["primary_four_program_AUPRC"])
        for mode in ("common", "native"):
            b = row["task_B_" + mode]
            if b["status"] != "COMPLETE":
                require(mode == "native" and row["arm"] in ("S_A0", "S_A1", "S_A2"),
                        "UNEXPECTED_MISSING_RETRIEVAL")
                continue
            require(len(b["per_section"]) == 2, "TWO_SLIDES_PER_DONOR_REQUIRED")
            for direction in ("image_to_gene", "gene_to_image"):
                for metric in ("Recall_at_1", "Recall_at_5", "median_rank", "pair_AUC"):
                    values = [s[direction][metric] for s in b["per_section"]]
                    same(math.fsum(values)/2 if all(v is not None for v in values) else None,
                         b[direction][metric])
    require(len(data["arm_seed_summary"]) == 12, "EXACT_ARM_SEED_SUMMARIES_REQUIRED")
    for row in data["arm_seed_summary"]:
        group = [keys[row["arm"], d, row["seed"]] for d in range(6)]
        require(sum(r["heldout_spots"] for r in group) == 13011, "SPOT_SUPPORT_CHANGED")
        compare_mean((r["task_A"]["primary_donor_four_program_Pearson"] for r in group),
                     row["task_A_primary_Pearson"])
        compare_mean((r["task_C"]["primary_four_program_AUPRC"] for r in group), row["task_C_AUPRC"])
        for mode in ("common", "native"):
            if row["task_B_"+mode]["status"] != "COMPLETE":
                continue
            for direction in ("image_to_gene", "gene_to_image"):
                for metric in ("Recall_at_1", "Recall_at_5", "median_rank", "pair_AUC"):
                    compare_mean((r["task_B_"+mode][direction][metric] for r in group),
                                 row["task_B_"+mode][direction][metric])
    require(len(data["paired_comparisons"]) == 20, "EXACT_PAIRED_COMPARISONS_REQUIRED")
    for pair in data["paired_comparisons"]:
        task, field, seed = pair["task"], pair["metric"], pair["seed"]
        actual = []
        for d in range(6):
            a = keys[pair["candidate"], d, seed][task][field]
            b = keys[pair["reference"], d, seed][task][field]
            delta = a-b if a is not None and b is not None else None
            saved = next(r["delta"] for r in pair["per_donor"] if r["heldout_donor_code"] == d)
            same(delta, saved)
            actual.append(delta)
        compare_mean(actual, pair["equal_donor_delta"])
        require(pair["positive_donors"] == sum(v is not None and v > 0 for v in actual)
                and pair["negative_donors"] == sum(v is not None and v < 0 for v in actual),
                "PAIRED_DIRECTION_COUNTS_CHANGED")
    require(max(errors) < 2e-10, "SOURCE_ARITHMETIC_REVIEW_FAILED")
    counters = data["counters"]
    require(counters["convex_optimizer_fits"] == 12 and counters["neural_optimizer_fits"] == 60
            and counters["optimizer_updates"] == 52000
            and counters["actual_saved_state_replays"] == counters["actual_saved_states"] == 192,
            "SOURCE_PILOT_BUDGET_CHANGED")
    require(data["s4_export"]["neural_optimizer_fits"] == 2
            and data["s4_export"]["actual_saved_state_replays"] == 6,
            "SOURCE_S4_BUDGET_CHANGED")
    return dict(schema="spgel_independent_anonymous_arithmetic_review_v1",
        status="PASS_INDEPENDENT_DONOR_PROGRAM_RETRIEVAL_MACROS_AND_PAIRED_DIFFERENCES",
        contract_sha256=data["contract_sha256"], source_summary_sha256=sha(path),
        reviewer_script_sha256=sha(__file__), actual_model_loads=0, new_optimizer_fits=0,
        this_review_is_not_an_additional_checkpoint_replay=True,
        donor_cells_checked=72, arm_seed_summaries_checked=12, paired_comparisons_checked=20,
        scalar_metric_checks=scalar_checks, maximum_absolute_arithmetic_error=max(errors),
        missing_metrics_preserve_null_and_support=True, independent_cohort_claimed=False,
        official_50HVG_completed=False)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = review(a.summary)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "scalar_metric_checks", "maximum_absolute_arithmetic_error")}))


if __name__ == "__main__":
    main()

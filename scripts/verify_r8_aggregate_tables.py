"""Independent arithmetic review using only anonymous R8 aggregate tables.

This is not a checkpoint replay and never imports a model or optimization code.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(report):
    result = read(report / "R8_RESULTS.json")
    lock = read(report / "R8_PROTOCOL_LOCK.json")
    assert result["contract_sha256"] == lock["contract_sha256"]
    assert result["status"] == "COMPLETE_81_CONVEX_FITS_AND_ACTUAL_SAVED_MODEL_REPLAYS"
    errors = []
    calculated = {}
    rows_checked = pairs_checked = 0
    for filename, key, paired_key in (
        ("R8_FOLD_METRICS.csv", "primary_DSS", "paired_comparisons"),
        ("R8_PFI_FOLD_METRICS.csv", "secondary_PFI_frozen_DSS_risk", "secondary_paired_comparisons"),
    ):
        with (report / filename).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 99
        indexed = {(r["arm"], int(r["partition"]), int(r["seed"])): r for r in rows}
        assert len(indexed) == 99
        assert set(r["arm"] for r in rows) == set(result[key])
        calc = {}
        for arm, saved in result[key].items():
            by_seed = []
            for seed in (31, 47, 83):
                support = [indexed[arm, part, seed] for part in (0, 1, 2)]
                per = {}
                for metric, num, den in (("Harrell", "Harrell_num", "Harrell_den"),
                                         ("Uno60", "Uno_num", "Uno_den")):
                    n = math.fsum(float(r[num]) for r in support)
                    d = math.fsum(float(r[den]) for r in support)
                    assert d > 0
                    per[metric] = n / d
                    original = next(r for r in saved["per_seed"] if r["seed"] == seed)
                    errors.extend((abs(per[metric] - original[metric]),
                                   abs(n - original[num]), abs(d - original[den])))
                    for r in support:
                        errors.append(abs(float(r[num]) / float(r[den]) - float(r[metric])))
                by_seed.append(per)
            calc[arm] = {m: math.fsum(r[m] for r in by_seed) / 3 for m in ("Harrell", "Uno60")}
            for metric, value in calc[arm].items():
                errors.append(abs(value - saved[metric]))
        for pair in result[paired_key]:
            c, b = pair["candidate"], pair["reference"]
            for metric in ("Harrell", "Uno60"):
                delta = calc[c][metric] - calc[b][metric]
                errors.append(abs(delta - pair["delta_" + metric]))
                repair, damage = (("repair", "damage") if metric == "Harrell"
                                  else ("Uno_repair", "Uno_damage"))
                errors.append(abs(delta - (pair[repair] - pair[damage])))
            pairs_checked += 1
        rows_checked += len(rows)
        calculated[key] = calc
    assert max(errors) < 2e-10
    primary_pairs = result["paired_comparisons"]
    gates = [p for p in primary_pairs if p["candidate"] == "R8-6"]
    assert len(gates) == 6 and not all(p["joint_directional_support"] for p in gates)
    assert result["budget"]["actual_convex_fits"] == 81
    assert result["budget"]["actual_saved_model_replays"] == 81
    assert result["budget"]["deterministic_unique_training_problems"] == 27
    assert result["outer_performance_evaluation"] is False
    return dict(
        schema="r8_independent_anonymous_arithmetic_review_v1",
        status="PASS_INDEPENDENT_WITHIN_FOLD_WEIGHTED_ARITHMETIC_AND_PAIR_CREDIT_IDENTITY",
        contract_sha256=lock["contract_sha256"],
        metadata_only=True, model_checkpoints_loaded=0, new_optimizer_fits=0,
        this_review_is_additional_model_replay=False,
        aggregate_rows_checked=rows_checked, paired_endpoint_comparisons_checked=pairs_checked,
        maximum_absolute_arithmetic_error=max(errors),
        within_partition_numerators_and_denominators_summed_before_seed_mean=True,
        cross_partition_patient_risk_ranking_used=False,
        deterministic_nominal_seeds_are_independent_evidence=False,
        primary_six_comparison_gate_passed=False,
        primary_candidate_minus_anchor={m: calculated["primary_DSS"]["R8-6"][m] -
                                            calculated["primary_DSS"]["R8-1"][m]
                                        for m in ("Harrell", "Uno60")},
        sources={n: sha(report / n) for n in ("R8_RESULTS.json", "R8_FOLD_METRICS.csv",
                 "R8_PFI_FOLD_METRICS.csv", "R8_PROTOCOL_LOCK.json")},
        reviewer_script_sha256=sha(Path(__file__)),
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = check(a.reports)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "aggregate_rows_checked",
          "paired_endpoint_comparisons_checked", "maximum_absolute_arithmetic_error")}))


if __name__ == "__main__":
    main()

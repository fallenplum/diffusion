"""Reuse the report producer's anonymous fixture; no private cohort or fit."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bgb_en_r6_independent_target_review.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reviewer = load("independent_r6_reviewer_test", SCRIPT)
fixture = load("existing_r6_report_fixture_for_independent_review", ROOT / "tests/test_bgb_en_r6_report_target.py")


def anonymous_package(tmp_path, complete=True):
    config, private, source_decision, source_report, _ = fixture.metadata_fixture(tmp_path, complete=complete)
    report, folds, pairs = fixture.r.build_report(config, private, source_decision, source_report)
    output = tmp_path / "anonymous"
    fixture.r.write_report(report, folds, pairs, output)
    return output, source_decision, config


def command(output, source_decision, config):
    return [sys.executable, str(SCRIPT), "--report-dir", str(output), "--source-decision", str(source_decision), "--config", str(config)]


def test_existing_anonymous_complete_fixture_passes_actual_independent_CLI(tmp_path):
    output, source_decision, config = anonymous_package(tmp_path)
    done = subprocess.run(command(output, source_decision, config), text=True, capture_output=True, cwd=ROOT, timeout=60)
    assert done.returncode == 0, done.stderr
    result = json.loads((output / "R6_TARGET_INDEPENDENT_REVIEW.json").read_text(encoding="utf-8"))
    assert result["status"] == "PASS_ANONYMOUS_ARITHMETIC_AND_CONTRACT_REVIEW"
    assert result["computation"]["formal_target_optimizer_updates"] == 7200
    assert result["computation"]["per_seed_components_independently_rebuilt"] == 72
    decision = result["independently_rebuilt_decision"]
    assert decision["decision"] == "INNER_GATE_PASSED_OPTIONAL_PENDING" and decision["outer_start_allowed"] is False
    assert decision["PFI_used_for_selection"] is False
    assert all(decision["core_flags"].values())
    # PFI is deliberately reversed in the existing fixture; it cannot select B7.
    pfi = next(row for row in result["arithmetic_summaries"] if row["arm"] == "B7" and row["endpoint"].startswith("PFI"))
    assert pfi["Harrell"] < .3
    comparison = result["comparisons"]["B7_minus_B1"]
    assert comparison["direct_repair"] - comparison["direct_damage"] == pytest.approx(comparison["Harrell_delta"])
    serialized = json.dumps(result)
    assert fixture.CANARY not in serialized and str(tmp_path) not in serialized
    assert not result["private_artifacts_labels_models_or_individual_risks_read"]
    assert result["underlying_model_replay_not_reexecuted"]
    assert (output / "R6_TARGET_INDEPENDENT_REVIEW.md").is_file()


@pytest.mark.parametrize("kind", ["fold_numerator", "direct_pair_credit"])
def test_anonymous_arithmetic_tampering_is_rejected_by_CLI(tmp_path, kind):
    output, source_decision, config = anonymous_package(tmp_path)
    name = "R6_SEED_FOLD.csv" if kind == "fold_numerator" else "R6_MECHANISM_ABLATIONS.csv"
    path = output / name
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream); fields = reader.fieldnames; rows = list(reader)
    if kind == "fold_numerator":
        # Also update the fold C itself: the independent summary check must catch it.
        row = next(row for row in rows if row["arm"] == "B7" and row["endpoint"] == "DSS")
        row["Harrell_num"] = str(float(row["Harrell_num"]) - .1)
        row["Harrell"] = str(float(row["Harrell_num"]) / float(row["Harrell_den"]))
    else:
        rows[0]["direct_repair"] = str(float(rows[0]["direct_repair"]) + .01)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)
    done = subprocess.run(command(output, source_decision, config), text=True, capture_output=True, cwd=ROOT, timeout=60)
    assert done.returncode != 0
    expected = "ARM_SUMMARY_ARITHMETIC_MISMATCH" if kind == "fold_numerator" else "PAIR_REPAIR_DAMAGE_OR_DELTA_IDENTITY_MISMATCH"
    assert expected in done.stderr
    assert not (output / "R6_TARGET_INDEPENDENT_REVIEW.json").exists()


def test_partial_producer_report_remains_inconclusive_without_overall_comparisons(tmp_path):
    output, source_decision, config = anonymous_package(tmp_path, complete=False)
    result = reviewer.review(output, source_decision, config)
    assert result["status"] == "INCONCLUSIVE_METADATA_INCOMPLETE"
    assert result["overall_winner"] is None and result["comparisons"] == []
    assert "arithmetic_summaries" not in result and "independently_rebuilt_decision" not in result

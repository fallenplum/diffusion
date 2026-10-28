"""Finite-grid, outcome-blind pairing and fail-closed R5 decision contracts."""
import copy
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pytest

from src.bgb_en.r5_protocol import (
    ALL_MAIN_ARMS, MAIN_ARMS, MECHANISM_CONTROLS, PARTITIONS, SEEDS,
    aggregate_metrics, decision_gate, hungarian_query_mismatch,
    make_protocol_lock, paired_comparison, rna_pattern_labels,
    select_finalist, source_hashes, verify_protocol_lock, write_protocol_lock,
)


def rows_for(scores, *, denominator=(10, 30, 60)):
    rows = []
    for arm, (h, u) in scores.items():
        for seed in SEEDS:
            for part, den in zip(PARTITIONS, denominator):
                rows.append(dict(arm=arm, seed=seed, inner_partition=part, status="COMPLETE",
                    active_parameters=100, metrics=dict(Harrell_num=h * den, Harrell_den=den,
                                                       Uno_num=u * den * 2, Uno_den=den * 2)))
    return rows


def main_fixture(selected=(.82, .81)):
    scores = {arm: (.79, .78) for arm in ALL_MAIN_ARMS}
    scores.update(A0=(.77, .76), A1=(.78, .77), A4=selected)
    return aggregate_metrics(rows_for(scores))


def mechanism_fixture(score=(.80, .79)):
    return aggregate_metrics(rows_for({arm: score for arm in MECHANISM_CONTROLS}), arms=MECHANISM_CONTROLS)


def test_hungarian_is_fixed_point_free_partition_bijection_and_lexicographic_optimum():
    indices = np.array([0, 2, 3, 5, 6])
    fields = [np.array(x) for x in (
        ["a", "x", "a", "b", "x", "b", "b"],
        ["C", "x", "R", "C", "x", "C", "R"],
        ["1", "x", "2", "1", "x", "2", "1"],
        ["p", "x", "p", "q", "x", "q", "p"],
    )]
    mapping, audit = hungarian_query_mismatch(indices, missing_pattern=fields[0], cancer=fields[1],
                                               stage=fields[2], platform=fields[3], seed=31)
    score = lambda targets: tuple(sum(values[a] != values[b] for a, b in zip(indices, targets)) for values in fields)
    best = min(score(targets) for targets in permutations(indices) if all(a != b for a, b in zip(indices, targets)))
    assert score(mapping[indices]) == best
    assert set(mapping[indices]) == set(indices)
    assert np.all(mapping[indices] != indices)
    assert np.array_equal(mapping[[1, 4]], [1, 4])
    assert audit["fixed_points"] == 0 and audit["outcomes_accessed"] is False
    again, audit_again = hungarian_query_mismatch(indices, missing_pattern=fields[0], cancer=fields[1],
                                                   stage=fields[2], platform=fields[3], seed=31)
    assert np.array_equal(mapping, again) and audit == audit_again


def test_pairing_never_trades_higher_priority_for_lower_priority_matches():
    # Exact RNA matching is possible and must dominate every platform reward.
    pattern = ["a", "a", "b", "b"]
    mapping, audit = hungarian_query_mismatch(range(4), missing_pattern=pattern,
        cancer=["C"] * 4, stage=["1"] * 4, platform=["p", "q", "p", "q"], seed=17)
    assert np.array_equal(mapping, [1, 0, 3, 2])
    assert audit["matched_counts"]["RNA_finite_pattern"] == 4


@pytest.mark.parametrize("indices", ([0], [0, 0], [-1, 0], [0, 4], [0.2, 1.5]))
def test_invalid_pairing_partition_rejected(indices):
    with pytest.raises(ValueError):
        hungarian_query_mismatch(indices, missing_pattern=["a"] * 4, cancer=["C"] * 4,
                                 stage=["1"] * 4, platform=["p"] * 4, seed=31)


def test_finite_mask_patterns_use_original_bits_not_platform_proxy():
    measured = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool)
    labels, audit = rna_pattern_labels(measured)
    assert labels[0] == labels[1] and len(set(labels)) == 3
    assert audit["complete_patients"] == 2 and audit["not_fully_measured_patients"] == 2
    with pytest.raises(ValueError):
        rna_pattern_labels(measured.astype(float))


def test_aggregation_weights_fold_denominators_then_averages_seeds():
    rows = rows_for({"A4": (.8, .7)})
    # One low-support fold changes one seed by .1 * .1 = .01, not .1 / 3.
    rows[0]["metrics"]["Harrell_num"] -= 1
    summary = aggregate_metrics(rows, arms=("A4",))
    assert summary["status"] == "COMPLETE"
    assert summary["arms"]["A4"]["per_seed"][0]["Harrell"] == pytest.approx(.79)
    assert summary["arms"]["A4"]["Harrell"] == pytest.approx((.79 + .8 + .8) / 3)
    assert summary["cross_partition_risk_comparisons"] is False


def test_missing_or_failed_cell_never_becomes_complete_mean():
    rows = rows_for({"A4": (.8, .7)})
    for candidate in (rows[:-1], [dict(row, status="FAILED") if at == 0 else row for at, row in enumerate(rows)]):
        result = aggregate_metrics(candidate, arms=("A4",))
        assert result["status"] == "INCOMPLETE"
        assert result["arms"]["A4"]["Harrell"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_metrics(rows + [rows[0]], arms=("A4",))


def test_primary_selection_requires_all_63_and_does_not_treat_point003_as_tie():
    summary = main_fixture()
    summary["arms"]["A5"]["Harrell"] = .8201
    summary["arms"]["A5"]["Uno60"] = .70
    assert select_finalist(summary)["selected_arm"] == "A5"
    summary["arms"]["A8"]["status"] = "INCOMPLETE"
    with pytest.raises(ValueError, match="63"):
        select_finalist(summary)


def test_paired_comparison_does_not_invent_pair_repair_from_C_difference():
    result = paired_comparison(main_fixture(), "A4", "A3")
    assert result["delta_Harrell"] == pytest.approx(.03)
    assert result["positive_Harrell_seeds"] == 3
    assert result["direct_pair_repair_damage_not_inferred"] is True
    assert "repair" not in result and "damage" not in result


def test_all45_and_replay_required_before_R6_eligibility():
    main = main_fixture()
    mechanism = mechanism_fixture()
    mechanism["arms"]["QUERY_MEAN"]["status"] = "INCOMPLETE"
    result = decision_gate(main, mechanism, "A4", qualification_passed=True, replay_passed=True)
    assert result["decision"] == "INCONCLUSIVE" and not result["R6_formal_start_eligible"]
    result = decision_gate(main, mechanism_fixture(), "A4", qualification_passed=True, replay_passed=False)
    assert result["decision"] == "INCONCLUSIVE" and not result["R6_formal_start_eligible"]
    falsely_complete = mechanism_fixture()
    falsely_complete["arms"]["QUERY_MEAN"]["completed_cells"] = 3
    result = decision_gate(main, falsely_complete, "A4", qualification_passed=True, replay_passed=True)
    assert result["decision"] == "INCONCLUSIVE" and not result["R6_formal_start_eligible"]


def test_GO_requires_mean_wrong_capacity_and_both_direction_controls():
    main, mechanism = main_fixture(), mechanism_fixture()
    result = decision_gate(main, mechanism, "A4", qualification_passed=True, replay_passed=True)
    assert result["decision"] == "R5_MECHANISM_DIRECTIONAL_GO"
    assert not result["R6_formal_start_eligible"]
    mean_wins = aggregate_metrics(rows_for({arm: ((.83, .82) if arm == "QUERY_MEAN" else (.80, .79))
                                           for arm in MECHANISM_CONTROLS}), arms=MECHANISM_CONTROLS)
    result = decision_gate(main, mean_wins, "A4", qualification_passed=True, replay_passed=True)
    assert result["decision"] == "R5_PERFORMANCE_ONLY_NOT_MECHANISM"
    assert result["R6_formal_start_eligible"]


def test_complete_negative_result_is_R6_eligible_without_broad_idea_rejection():
    result = decision_gate(main_fixture(selected=(.76, .75)), mechanism_fixture(), "A4",
                           qualification_passed=True, replay_passed=True)
    assert result["decision"] == "R5_READOUT_STOP_R6_ELIGIBLE"
    assert result["R6_formal_start_eligible"]
    assert result["independent_external_confirmation"] is False


def test_lock_resume_pins_config_source_and_prepared_identity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    paths = ("configs/bgb_en/r5_main_v1.json", "configs/bgb_en/r5_mechanism_v1.json")
    for name in paths:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / name).read_bytes())
    source = tmp_path / "src/bgb_en/fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text("fixed = 1\n")
    lock = make_protocol_lock(tmp_path, *paths, ["src/bgb_en/fixture.py"],
        data_fingerprints={"cohort": "a" * 64},
        prepared_receipt_sha256={str(i): str(i) * 64 for i in PARTITIONS})
    target = tmp_path / "reports/bgb_en_r5/PROTOCOL_LOCK.json"
    saved = write_protocol_lock(target, lock)
    assert write_protocol_lock(target, lock) == saved
    verify_protocol_lock(saved, tmp_path)
    mutated = copy.deepcopy(lock)
    mutated["planned_main_neural_fits"] = 62
    with pytest.raises(ValueError, match="overwrite"):
        write_protocol_lock(target, mutated)
    source.write_text("fixed = 2\n")
    with pytest.raises(ValueError, match="source bytes"):
        verify_protocol_lock(saved, tmp_path)
    with pytest.raises(ValueError):
        source_hashes(tmp_path, ["../outside.py"])

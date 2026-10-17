"""Anonymous assay reporting must preserve unavailable same-assay controls."""

import copy
import json

import numpy as np
import pytest

from scripts.run_phase22_sampling_audit import (
    METRICS, aggregate_arms, aggregate_by_assay, bootstrap_contrasts,
    measurement_eligibility_record, measurement_eligibility_summary,
)


def rows():
    output = []
    for assay, donors in (("Xenium", ("private-d1", "private-d2", "private-d3")),
                          ("Visium", ("private-d1",))):
        for donor in donors:
            for seed in (31, 47, 83):
                for fraction in (.25, .5, .75, 1.):
                    for index, arm in enumerate(("B0", "B1", "B2", "B3", "B4")):
                        absent = assay == "Visium" and arm == "B4"
                        output.append({
                            "assay": assay, "donor_id_private": donor,
                            "section_id_private": donor + "-" + assay,
                            "study_id_private": "private-study", "seed": seed,
                            "component_index": 0, "area_fraction": fraction,
                            "arm": arm,
                            "numerical_status": "UNAVAILABLE_DIFFERENT_DONOR" if absent else "CONVERGED",
                            "metrics": {} if absent else dict(zip(METRICS, (.1 + index / 100, .02 - index / 1000, .3))),
                        })
    return output


def test_assay_strata_keep_singleton_wrong_bulk_unavailable_without_ids():
    source = rows()
    frozen = copy.deepcopy(source)
    result = aggregate_by_assay(source)
    assert source == frozen
    assert set(result) == {"Visium", "Xenium"}
    assert result["Xenium"]["donors"] == 3
    assert result["Xenium"]["sections"] == 3
    assert result["Xenium"]["B4_support"]["evaluable_donors"] == 3
    assert result["Xenium"]["B4_support"]["unavailable_cells"] == 0
    visium = result["Visium"]
    assert visium["donors"] == 1 and visium["sections"] == 1
    assert visium["arms"]["B4"]["wrong_bulk_unavailable_cells"] == 12
    assert visium["B4_support"]["unavailable_donors"] == 1
    assert visium["B4_support"]["unavailable_cells"] == 12
    assert visium["B4_support"]["evaluable_donors"] == 0
    assert visium["B4_support"]["different_assay_bulk_substitution_allowed"] is False
    assert visium["primary_gate_recomputed_in_this_stratum"] is False
    for metric in METRICS:
        contrast = visium["paired_local_contrasts"]["B3_minus_B4"][metric]
        assert contrast == {"status": "UNAVAILABLE", "donors": 0}
    assert "private-d" not in json.dumps(result)
    assert "private-study" not in json.dumps(result)


def test_descriptive_stratum_uses_same_aggregation_and_does_not_change_pooled_output():
    source = rows()
    before = (aggregate_arms(source), bootstrap_contrasts(source))
    result = aggregate_by_assay(source)
    assert before == (aggregate_arms(source), bootstrap_contrasts(source))
    for assay, report in result.items():
        subset = [row for row in source if row["assay"] == assay]
        assert report["arms"] == aggregate_arms(subset)
        assert report["paired_local_contrasts"] == bootstrap_contrasts(subset)
        assert report["studies"] == 1
    for metric in METRICS:
        contrast = result["Xenium"]["paired_local_contrasts"]["B3_minus_B1"][metric]
        assert contrast["donors"] == 3
        expected = -.002 if metric == METRICS[1] else .02 if metric == METRICS[0] else 0.
        # The reported contrast direction is always B3 - B1.
        assert contrast["point"] == pytest.approx(expected)


def test_empty_assay_report_is_empty():
    assert aggregate_by_assay([]) == {}


def measurement_entry(assay="Visium", donor="private-d1", stage="test"):
    return {"assay": assay, "donor_id_private": donor, "section_id_private": donor + "-" + assay,
            "study_id": "private-study", "split": stage, "npz_sha256": "a" * 64,
            "features_sha256": "b" * 64, "feature_receipt_sha256": "c" * 64}


def test_partial_test_measurement_is_explicitly_ineligible_and_keeps_locked_dictionary_size():
    mask = np.ones(378, dtype=bool)
    mask[[1, 5, 8, 13, 21, 34, 55, 89]] = False
    before = mask.copy()
    record = measurement_eligibility_record(measurement_entry(), mask, stage="test", dictionary_gene_count=378)
    assert np.array_equal(mask, before)
    assert record["status"] == "INELIGIBLE_PARTIAL_MEASUREMENT"
    assert record["dictionary_gene_count"] == 378
    assert record["measured_dictionary_gene_count"] == 370
    assert record["unmeasured_dictionary_gene_count"] == 8
    assert record["missing_values_imputed"] is False
    assert record["dictionary_or_omega_refitted"] is False


def test_partial_validation_measurement_fails_instead_of_silently_excluding_or_imputing():
    with pytest.raises(ValueError, match="validation requires all fixed-dictionary genes"):
        measurement_eligibility_record(measurement_entry(stage="validation"),
            np.array([True, False, True]), stage="validation", dictionary_gene_count=3)
    with pytest.raises(ValueError, match="partition"):
        measurement_eligibility_record(measurement_entry(stage="test"),
            np.ones(3, dtype=bool), stage="validation", dictionary_gene_count=3)


def test_excluded_assay_remains_visible_in_public_summary_without_fake_metrics_or_identifiers():
    ledger = [measurement_eligibility_record(measurement_entry("Xenium", donor), np.ones(378, dtype=bool),
        stage="test", dictionary_gene_count=378) for donor in ("private-d1", "private-d2", "private-d3")]
    mask = np.ones(378, dtype=bool)
    mask[:8] = False
    ledger.append(measurement_eligibility_record(measurement_entry(), mask, stage="test", dictionary_gene_count=378))
    summary = measurement_eligibility_summary(ledger)
    assert summary["source_sections"] == 4
    assert summary["eligible_sections"] == 3
    assert summary["ineligible_sections"] == 1
    assert summary["eligible_assays"] == ["Xenium"]
    assert summary["fully_ineligible_assays"] == ["Visium"]
    assert summary["by_assay"]["Visium"]["unmeasured_dictionary_gene_count_by_ineligible_section"] == [8]
    assert summary["gate_generalization_to_ineligible_assays"] is False
    strata = aggregate_by_assay([row for row in rows() if row["assay"] == "Xenium"], ledger)
    assert strata["Xenium"]["donors"] == 3
    excluded = strata["Visium"]
    assert excluded["status"] == "INELIGIBLE_PARTIAL_MEASUREMENT"
    assert excluded["arms"] == excluded["paired_local_contrasts"] == {}
    assert excluded["B4_support"]["status"] == "NOT_EVALUATED_INELIGIBLE_MEASUREMENT"
    assert "private-d" not in json.dumps([summary, strata])
    assert "private-study" not in json.dumps([summary, strata])


def test_full_validation_eligibility_uses_same_cross_stage_policy_and_does_not_exclude():
    record = measurement_eligibility_record(measurement_entry(stage="validation"),
        np.ones(378, dtype=bool), stage="validation", dictionary_gene_count=378)
    validation = measurement_eligibility_summary([record])
    test_mask = np.ones(378, dtype=bool)
    test_mask[:8] = False
    test = measurement_eligibility_summary([measurement_eligibility_record(measurement_entry(), test_mask,
        stage="test", dictionary_gene_count=378)])
    assert validation["policy"] == test["policy"]
    assert validation["eligible_sections"] == 1 and validation["ineligible_sections"] == 0
    assert test["eligible_sections"] == 0 and test["ineligible_sections"] == 1

import copy

import pytest

from scripts import semantic_b0b7_p2_omics_public_report as report


def fixture():
    return {"schema": "semantic_b0b7_canonical_omics_source_provenance_v1",
            "status": "SOURCE_RAW_CANONICAL_CHAIN_VERIFIED",
            "source_sha256": report.EXECUTION_SOURCE_SHA,
            "inherited_arrays": {report.OLD_FIELD: {
                "shape": [512], "dtype": "<U20", "array_sha256": "a" * 64,
                "exact_parent_inheritance": True}},
            "example_aggregate": {"missing_count": 260}}


def test_projection_changes_only_hash_metadata_key_and_adds_provenance():
    value = fixture()
    before = copy.deepcopy(value)
    with pytest.raises(ValueError, match="row-level JSON field"):
        report.original.validate_public_json(value)
    public = report.project(value, "b" * 64, "c" * 64)
    report.original.validate_public_json(public)
    restored = copy.deepcopy(public)
    restored.pop("public_projection")
    restored["inherited_arrays"][report.OLD_FIELD] = restored["inherited_arrays"].pop(report.PUBLIC_FIELD)
    assert restored == before == value


def test_projection_cannot_allow_raw_fields_or_non_hash_identity_metadata():
    bad = fixture()
    bad["inherited_arrays"][report.OLD_FIELD]["values"] = [1, 2]
    with pytest.raises(ValueError, match="HASH_ONLY"):
        report.project(bad, "b" * 64, "c" * 64)
    bad = fixture()
    bad["predictions"] = [1, 2]
    with pytest.raises(ValueError, match="row-level JSON field"):
        report.project(bad, "b" * 64, "c" * 64)

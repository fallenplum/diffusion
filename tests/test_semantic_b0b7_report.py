"""Synthetic-only exact-publication tests; no patient or model files loaded."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("semantic_b0b7_report_tests",
                                            Path(__file__).resolve().parents[1] / "scripts/semantic_b0b7_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def synthetic_source():
    return {"schema": "SYNTHETIC_PUBLICATION_TEST_NOT_AN_EXPERIMENT",
            "metadata": {"中文": "严格原字节保留", "missing": None, "negative": -0.01},
            "cells": [{"arm": "B3", "seed": 31, "programs": [
                {"program_index": i, "pearson": None if i % 5 == 0 else (i - 25) / 100}
                for i in range(50)]},
                {"arm": "B4", "seed": 47, "nested": {"support": 0, "results": [False, True, 1.2e-5]}}]}


def test_all_nested_fields_and_original_utf8_bytes_roundtrip(tmp_path):
    source = synthetic_source()
    raw = REPORT.json_bytes(source)
    index, records = REPORT.lossless_package(raw, tmp_path)
    restored, restored_raw = REPORT.restore_public_report(tmp_path / "04_PRETEXT_LOSSLESS_INDEX.json")
    assert restored == source and restored_raw == raw
    assert index["canonical_json_sha256"] == REPORT.digest(REPORT.canonical(source))
    assert len(records) == 3 and len(restored["cells"][0]["programs"]) == 50
    assert all(r["bytes"] < 2 * 1024 * 1024 for r in records)


def test_source_nonstandard_whitespace_cannot_be_claimed_original_byte_exact(tmp_path):
    with pytest.raises(ValueError, match="PRELOCKED_CANONICAL_PRETTY"):
        REPORT.lossless_package(json.dumps(synthetic_source()).encode(), tmp_path)


def test_native_integer_key_sort_order_preserves_original_bytes_after_json_keys_become_strings(tmp_path):
    source = synthetic_source()
    source["cells"][0]["checkpoints"] = {50: {"loss": .3}, 100: {"loss": .2}, 200: {"loss": .1}, 400: {"loss": .4}}
    raw = REPORT.json_bytes(source)
    parsed = json.loads(raw)
    assert list(parsed["cells"][0]["checkpoints"]) == ["50", "100", "200", "400"]
    assert REPORT.json_bytes(parsed) != raw
    index, _ = REPORT.lossless_package(raw, tmp_path)
    assert index["source_key_order_overrides"]
    restored, restored_raw = REPORT.restore_public_report(tmp_path / "04_PRETEXT_LOSSLESS_INDEX.json")
    assert restored == parsed and restored_raw == raw


def test_modified_shard_rejected_before_reconstruction(tmp_path):
    index, _ = REPORT.lossless_package(REPORT.json_bytes(synthetic_source()), tmp_path)
    part = tmp_path / index["parts"][0]["file"]
    part.write_bytes(part.read_bytes() + b" ")
    with pytest.raises(ValueError, match="PART_FILE_HASH"):
        REPORT.restore_public_report(tmp_path / "04_PRETEXT_LOSSLESS_INDEX.json")


def test_changed_canonical_binding_rejected(tmp_path):
    index, _ = REPORT.lossless_package(REPORT.json_bytes(synthetic_source()), tmp_path)
    index["canonical_json_sha256"] = "0" * 64
    path = tmp_path / "04_PRETEXT_LOSSLESS_INDEX.json"
    path.write_bytes(REPORT.json_bytes(index))
    with pytest.raises(ValueError, match="CANONICAL_HASH"):
        REPORT.restore_public_report(path)


def test_part_path_escape_rejected(tmp_path):
    index, _ = REPORT.lossless_package(REPORT.json_bytes(synthetic_source()), tmp_path)
    index["parts"][0]["file"] = "../outside.json"
    path = tmp_path / "04_PRETEXT_LOSSLESS_INDEX.json"
    path.write_bytes(REPORT.json_bytes(index))
    with pytest.raises(ValueError, match="PATH_ESCAPE"):
        REPORT.restore_public_report(path)


def test_existing_reports_never_silently_overwritten(tmp_path):
    REPORT.write_once(tmp_path / "report.json", b"original")
    REPORT.write_once(tmp_path / "report.json", b"original")
    with pytest.raises(FileExistsError, match="BYTES_DIFFER"):
        REPORT.write_once(tmp_path / "report.json", b"replacement")
    assert (tmp_path / "report.json").read_bytes() == b"original"


def test_csv_splitting_preserves_every_row_under_public_byte_limit(tmp_path):
    records = REPORT.csv_parts(tmp_path, "SYNTHETIC_ONLY",
                               ({"row": i, "text": "合成" * 300} for i in range(2500)))
    assert len(records) > 1 and sum(x["data_rows"] for x in records) == 2500
    assert all(x["bytes"] < 2 * 1024 * 1024 for x in records)


def test_csv_split_obeys_unchanged_original_guard_20000_line_limit(tmp_path):
    records = REPORT.csv_parts(tmp_path, "SYNTHETIC_ROW_LIMIT", ({"small_row": i} for i in range(20_005)))
    assert len(records) == 2
    assert [x["data_rows"] for x in records] == [19_999, 6]
    assert all(len((tmp_path / x["file"]).read_text(encoding="utf-8").splitlines()) <= 20_000 for x in records)


def test_exact_chance_is_mean_reciprocals_not_reciprocal_mean_count():
    # Two anchors have 1 and 3 negatives: exact chance=.375, reciprocal
    # average candidate count=1/3, which is not the same statistic.
    chance = REPORT.chance_lookup({"rows": [{"scope": "global", "group": None,
        "mode": "hard", "direction": "image_to_gene", "anchors_covered": 2,
        "exact_chance_mean": .375}]})
    observed = {"anchors_covered": 2, "top1_tie_averaged": .45}
    combined = REPORT.attach_chance(observed, chance, "global", "ALL", "hard", "image_to_gene")
    assert combined["exact_chance_mean"] == .375
    assert abs(combined["top1_minus_exact_chance"] - .075) < 1e-12
    assert combined["exact_chance_mean"] != 1 / 3


def test_chance_support_mismatch_fails_and_unavailable_chance_is_not_invented():
    chance = REPORT.chance_lookup({"rows": [{"scope": "global", "group": None,
        "mode": "hard", "direction": "image_to_gene", "anchors_covered": 2,
        "exact_chance_mean": .375}]})
    with pytest.raises(ValueError, match="CHANCE_SUPPORT_DIFFERS"):
        REPORT.attach_chance({"anchors_covered": 3, "top1_tie_averaged": .5}, chance,
                             "global", "ALL", "hard", "image_to_gene")
    missing = REPORT.attach_chance({"anchors_covered": 2, "top1_tie_averaged": .5}, {},
                                   "global", "ALL", "hard", "image_to_gene")
    assert missing["exact_chance_mean"] is None and missing["top1_minus_exact_chance"] is None

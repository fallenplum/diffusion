"""Metadata-only privacy counterexamples; never load models or patient arrays."""
import hashlib
import json
from pathlib import Path
import re

import pytest

from scripts import semantic_b0b7_publication as publication


def record(root, relative, value):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else value.encode()
    path.write_bytes(raw)
    return {"file": relative, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


@pytest.mark.parametrize("path", ["../escape.json", "tmp/predictions.json", "reports/old/history.json",
    "reports/semantic_b0b7/weights.pt", publication.MANIFEST])
def test_unknown_escaping_and_generated_inputs_rejected(path):
    with pytest.raises(ValueError):
        publication.safe_name(path)


@pytest.mark.parametrize("payload", [
    json.dumps({"patient_ids": ["synthetic_person"]}),
    json.dumps({"prediction": [0.1, 0.2]}),
    json.dumps({"description": "hf_" + "a" * 24}),
    json.dumps({"description": "/root/" + "private_asset"}),
    "a" * (2 * 1024**2 + 1),
], ids=["patient_field", "raw_prediction_array", "credential", "private_path", "oversized"])
def test_original_scanner_privacy_and_size_limits_unchanged(tmp_path, payload):
    row = record(tmp_path, "reports/semantic_b0b7/SYNTHETIC.json", payload)
    with pytest.raises(ValueError):
        publication.scan(tmp_path, [row], re.compile(r"(?!x)x"))


def test_actual_identifier_pattern_is_still_applied(tmp_path):
    row = record(tmp_path, "reports/semantic_b0b7/SYNTHETIC.md", "synthetic_private_identifier")
    with pytest.raises(ValueError):
        publication.scan(tmp_path, [row], re.compile("synthetic_private_identifier"))


def test_explicit_inventory_rejects_changed_file_bytes(tmp_path):
    row = record(tmp_path, "reports/semantic_b0b7/SYNTHETIC.json", '{"status":"synthetic"}\n')
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps({"base_git_commit": publication.BASE_COMMIT, "files": [row]}))
    assert publication.inventory(tmp_path, path) == [row]
    (tmp_path / row["file"]).write_text('{"status":"changed"}\n')
    with pytest.raises(ValueError):
        publication.inventory(tmp_path, path)


def test_forged_completion_gate_is_rejected_before_claims(tmp_path):
    path = tmp_path / "gate.json"
    path.write_text(json.dumps({"status": "PASS", "frozen_sources": {str(i): "f" * 64 for i in range(13)}}))
    with pytest.raises(ValueError):
        publication.completion(tmp_path, tmp_path, path)


def test_duplicate_allowlist_entries_are_rejected(tmp_path):
    row = record(tmp_path, "reports/semantic_b0b7/SYNTHETIC.json", '{}\n')
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps({"base_git_commit": publication.BASE_COMMIT, "files": [row, row]}))
    with pytest.raises(ValueError):
        publication.inventory(tmp_path, path)


def test_reproduction_closure_cannot_omit_a_required_shard():
    needed = [publication.INDEX, "reports/semantic_b0b7/04_PRETEXT_LOSSLESS_CELL_B3_seed031.json"]
    with pytest.raises(ValueError):
        publication.require_closure([{"file": publication.INDEX}], needed)
    publication.require_closure([{"file": name} for name in needed], needed)

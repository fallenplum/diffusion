"""Anonymous metadata-only projection of one executed provenance audit.

Only a hash-metadata field name changes. The original receipt and its execution
source hash remain bound. This script never loads arrays or runs a model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_publication as original

SOURCE_REPORT_SHA = "be5f1c0bab6d6416a9e61953557ba47a0bc0a778113549f6baae76495ac79713"
EXECUTION_SOURCE_SHA = "9f8871b67e8dfe5696c976c135304bb7562d036ac072c4ae257910ce2e15e5ef"
OLD_FIELD = "patient_ids"
PUBLIC_FIELD = "cohort_row_identity_hash_only"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(value, source_report_sha256, writer_sha256):
    original.require(value["schema"] == "semantic_b0b7_canonical_omics_source_provenance_v1" and
                     value["status"] == "SOURCE_RAW_CANONICAL_CHAIN_VERIFIED" and
                     value["source_sha256"] == EXECUTION_SOURCE_SHA, "UNEXPECTED_EXECUTED_SOURCE_AUDIT")
    metadata = value["inherited_arrays"][OLD_FIELD]
    original.require(set(metadata) == {"shape", "dtype", "array_sha256", "exact_parent_inheritance"} and
                     metadata["shape"] == [512] and metadata["dtype"] == "<U20" and
                     metadata["exact_parent_inheritance"] is True and
                     len(metadata["array_sha256"]) == 64 and
                     PUBLIC_FIELD not in value["inherited_arrays"], "IDENTITY_METADATA_IS_NOT_HASH_ONLY")
    public = copy.deepcopy(value)
    public["inherited_arrays"][PUBLIC_FIELD] = public["inherited_arrays"].pop(OLD_FIELD)
    public["public_projection"] = {
        "source_report_sha256": source_report_sha256,
        "writer_source_sha256": writer_sha256,
        "changed_metadata_field": "inherited_arrays identity hash metadata key only",
        "numerical_values_changed": False, "array_values_loaded": False,
        "new_model_fits": 0, "new_model_forwards": 0,
    }
    original.validate_public_json(public)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original.require(sha(args.source) == SOURCE_REPORT_SHA, "EXECUTED_SOURCE_REPORT_BYTES_CHANGED")
    original.require(not args.output.exists(), "PUBLIC_PROJECTION_IS_APPEND_ONLY")
    value = json.loads(args.source.read_bytes().decode("utf-8"))
    public = project(value, SOURCE_REPORT_SHA, sha(__file__))
    raw = (json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    original.require(len(raw) <= 2 * 1024**2 and
                     not any(pattern.search(raw.decode("utf-8")) for pattern in original.PATTERNS.values()),
                     "ORIGINAL_PUBLIC_TEXT_RULES_REJECT_PROJECTION")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"status": "ANONYMOUS_METADATA_PROJECTION_ONLY_NOT_NEW_EXPERIMENT",
                      "source_report_sha256": SOURCE_REPORT_SHA,
                      "public_report_sha256": sha(args.output), "bytes": len(raw)}))


if __name__ == "__main__":
    main()

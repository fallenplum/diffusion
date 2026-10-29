"""Losslessly split source report diagnostics; verify exact producer-byte recovery.

No models, arrays, endpoints or optimizers are read. The frozen report producer
and its original output stay unchanged. Only the public serialization changes.
"""
import argparse
import hashlib
import json
from pathlib import Path


MAIN = "S3_PILOT_SUMMARY.json"
DIAGNOSTIC = "S3_CHECKPOINT_DIAGNOSTICS.json"
AUDIT = "S3_LOSSLESS_SERIALIZATION_AUDIT.json"
PUBLIC_SCHEMA = "spgel_source_sharded_metadata_report_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def compact(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n").encode()


def pretty(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()


def semantic(value):
    return sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"), allow_nan=False).encode())


def load_original_public_report(directory):
    directory = Path(directory)
    public = json.loads((directory/MAIN).read_bytes())
    require(public["schema"] == PUBLIC_SCHEMA, "EXPLICIT_SHARDED_REPORT_SCHEMA_REQUIRED")
    archive = public["_publication_archive"]
    reference = public["fixed_checkpoint_diagnostics"]
    require(reference["file"] == DIAGNOSTIC and reference["records"] == 192,
            "EXACT_FIXED_DIAGNOSTIC_REFERENCE_REQUIRED")
    raw = (directory/DIAGNOSTIC).read_bytes()
    require(sha_bytes(raw) == reference["sha256"], "DIAGNOSTIC_BYTES_CHANGED")
    diagnostic = json.loads(raw)
    require(diagnostic["schema"] == "spgel_public_checkpoint_diagnostics_v1"
            and diagnostic["contract_sha256"] == public["contract_sha256"]
            and len(diagnostic["records"]) == 192, "EXACT_192_DIAGNOSTIC_RECORDS_REQUIRED")
    identities = {(r["arm"], r["heldout_donor_code"], r["seed"], r["snapshot"])
                  for r in diagnostic["records"]}
    require(len(identities) == 192, "DIAGNOSTIC_IDENTITIES_DUPLICATED")
    restored = {k:v for k,v in public.items() if k != "_publication_archive"}
    restored["schema"] = archive["original_schema"]
    restored["fixed_checkpoint_diagnostics"] = diagnostic["records"]
    require(list(restored) == archive["original_top_level_keys"], "ORIGINAL_KEY_ORDER_CHANGED")
    raw = pretty(restored)
    require(len(raw) == archive["original_bytes"]
            and sha_bytes(raw) == archive["original_producer_sha256"]
            and semantic(restored) == archive["original_semantic_sha256"],
            "EXACT_ORIGINAL_PRODUCER_RECONSTRUCTION_FAILED")
    return restored, raw


def split(original, output):
    raw = Path(original).read_bytes()
    source = json.loads(raw)
    require(source["schema"] == "spgel_source_metadata_report_v1"
            and source["status"] == "COMPLETE_REGISTERED_SOURCE_PILOT_AND_REPLAY"
            and source["counters"]["actual_saved_state_replays"] == 192,
            "COMPLETE_ORIGINAL_REPORT_REQUIRED")
    require(raw == pretty(source), "ORIGINAL_PRODUCER_SERIALIZATION_CHANGED")
    require("_publication_archive" not in source, "ALREADY_TRANSFORMED_INPUT_BLOCKED")
    diagnostics = source["fixed_checkpoint_diagnostics"]
    require(len(diagnostics) == 192, "EXACT_DIAGNOSTIC_COUNT_REQUIRED")
    diagnostic_bytes = compact(dict(schema="spgel_public_checkpoint_diagnostics_v1",
        contract_sha256=source["contract_sha256"], records=diagnostics))
    public = dict(source)
    public["schema"] = PUBLIC_SCHEMA
    public["fixed_checkpoint_diagnostics"] = dict(status="LOSSLESS_EXTERNAL_DIAGNOSTIC_SHARD",
        file=DIAGNOSTIC, records=192, sha256=sha_bytes(diagnostic_bytes))
    public["_publication_archive"] = dict(
        schema="exact_source_report_archive_v1", original_schema=source["schema"],
        original_producer_sha256=sha_bytes(raw), original_bytes=len(raw),
        original_semantic_sha256=semantic(source), original_top_level_keys=list(source),
        serialization="UTF8_json_ensure_ascii_false_indent2_plus_final_LF",
        unchanged_original_retained_privately=True,
        original_and_public_metric_values_identical=True)
    main_bytes = compact(public)
    require(max(len(main_bytes), len(diagnostic_bytes)) <= 2*1024**2,
            "PUBLIC_SHARD_EXCEEDS_EXISTING_SIZE_CAP")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in (MAIN, DIAGNOSTIC, AUDIT):
        require(not (output/name).exists(), "EXISTING_PUBLIC_OUTPUT_PRESERVED")
    (output/MAIN).write_bytes(main_bytes)
    (output/DIAGNOSTIC).write_bytes(diagnostic_bytes)
    restored, recovered = load_original_public_report(output)
    require(restored == source and recovered == raw, "ORIGINAL_OBJECT_OR_BYTES_CHANGED")
    audit = dict(schema="spgel_lossless_public_report_serialization_v1",
        status="PASS_EXACT_ORIGINAL_PRODUCER_BYTES_AND_METRICS_RECONSTRUCTION",
        contract_sha256=source["contract_sha256"], original_producer_sha256=sha_bytes(raw),
        reconstructed_original_sha256=sha_bytes(recovered),
        original_semantic_sha256=semantic(source), original_bytes=len(raw),
        public_main_sha256=sha_bytes(main_bytes), public_main_bytes=len(main_bytes),
        diagnostic_shard_sha256=sha_bytes(diagnostic_bytes), diagnostic_shard_bytes=len(diagnostic_bytes),
        donor_rows_preserved=len(source["donor_rows"]), diagnostic_records_preserved=len(diagnostics),
        scalar_or_diagnostic_records_removed=0, new_optimizer_fits=0, model_loads=0,
        script_sha256=sha_bytes(Path(__file__).read_bytes()))
    (output/AUDIT).write_bytes(pretty(audit))
    return audit


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    a = parser.parse_args()
    if a.verify_only:
        _, raw = load_original_public_report(a.output)
        result = dict(status="PASS_EXACT_ORIGINAL_PUBLIC_REPORT_RECONSTRUCTION",
                      original_sha256=sha_bytes(raw), new_optimizer_fits=0)
    else:
        require(a.original is not None, "ORIGINAL_PRODUCER_REPORT_REQUIRED")
        result = split(a.original, a.output)
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()

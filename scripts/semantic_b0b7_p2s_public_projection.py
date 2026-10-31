"""Lossless public column shards for three locked 154-column aggregate CSVs.

The original reporter and its outputs remain unchanged. Each public shard has
at most 149 original columns plus a non-patient aggregate-row ordinal. Rebuilding
the original CSV with the declared original column order must match its bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCRIPT = "scripts/semantic_b0b7_p2s_public_projection.py"
TEST = "tests/test_semantic_b0b7_p2s_public_projection.py"
MAX_BYTES = 2 * 1024 ** 2
MAX_LINES = 20_000
MAX_COLUMNS = 150
ORDINAL = "aggregate_row_ordinal"


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes().decode("utf-8"))


def csv_bytes(header, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header); writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write(path, raw):
    require(len(raw) <= MAX_BYTES, "PUBLIC_PROJECTION_FILE_TOO_LARGE")
    path = Path(path)
    with path.open("xb") as stream: stream.write(raw)
    return {"file": path.name, "bytes": len(raw), "sha256": sha(path)}


def checked(root, name, expected):
    relative = Path(name); root = Path(root).resolve()
    require(not relative.is_absolute() and ".." not in relative.parts, "UNSAFE_PUBLIC_REFERENCE")
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root) and sha(path) == expected,
            "PUBLIC_SOURCE_OR_PART_BYTES_CHANGED")
    return path


def project_csv(source, output):
    raw = Path(source).read_bytes()
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8"))))
    require(rows and len(rows[0]) == 154 and len(set(rows[0])) == 154 and ORDINAL not in rows[0], "EXACT154_ORIGINAL_COLUMNS_REQUIRED")
    header, data = rows[0], rows[1:]
    require(len(rows) <= MAX_LINES and all(len(r) == 154 for r in data), "ORIGINAL_CSV_ROWS_OR_SHAPE_CHANGED")
    require(csv_bytes(header, data) == raw, "ORIGINAL_CSV_NOT_CANONICAL_LF_REFUSE_NORMALIZATION")
    parts = []
    for column_group, start in enumerate(range(0, len(header), MAX_COLUMNS - 1), 1):
        names = header[start:start + MAX_COLUMNS - 1]
        local_header = [ORDINAL, *names]
        encoded_header = csv_bytes(local_header, [])
        current, first, part_no, size = [], 0, 0, len(encoded_header)
        for number, row in enumerate(data):
            candidate = [str(number), *row[start:start + len(names)]]
            encoded = csv_bytes(local_header, [candidate])[len(encoded_header):]
            require(len(encoded_header) + len(encoded) <= MAX_BYTES, "SINGLE_PROJECTED_ROW_TOO_LARGE")
            if current and (len(current) + 2 > MAX_LINES or size + len(encoded) > MAX_BYTES):
                part_no += 1
                path = Path(output) / f"{Path(source).stem}_columns{column_group:02d}_rows{part_no:02d}.csv"
                info = write(path, encoded_header + b"".join(current))
                parts.append({**info, "columns": names, "row_start": first, "row_stop_exclusive": number})
                current, first, size = [], number, len(encoded_header)
            current.append(encoded); size += len(encoded)
        if current or not data:
            part_no += 1
            path = Path(output) / f"{Path(source).stem}_columns{column_group:02d}_rows{part_no:02d}.csv"
            info = write(path, encoded_header + b"".join(current))
            parts.append({**info, "columns": names, "row_start": first, "row_stop_exclusive": len(data)})
    return {"source_file": Path(source).name, "source_bytes": len(raw), "source_sha256": sha(source), "source_headers": header,
            "source_rows": len(data), "ordinal_field": ORDINAL, "ordinal_start": 0, "parts": parts}


def reconstruct(root, description):
    require(description["ordinal_field"] == ORDINAL and description["ordinal_start"] == 0, "PROJECTION_ORDINAL_PROTOCOL_CHANGED")
    header = description["source_headers"]; n = description["source_rows"]
    require(len(header) == len(set(header)) == 154 and 0 <= n < MAX_LINES, "PROJECTION_SOURCE_DIMENSIONS_CHANGED")
    values = [{} for _ in range(n)]
    for part in description["parts"]:
        path = checked(root, part["file"], part["sha256"])
        require(path.stat().st_size == part["bytes"] <= MAX_BYTES, "PROJECTION_PART_SIZE_CHANGED")
        rows = list(csv.reader(io.StringIO(path.read_bytes().decode("utf-8"))))
        columns = part["columns"]
        require(0 < len(columns) < MAX_COLUMNS and len(set(columns)) == len(columns) and set(columns) <= set(header) and
                rows[0] == [ORDINAL, *columns] and len(rows) <= MAX_LINES and
                len(rows) - 1 == part["row_stop_exclusive"] - part["row_start"], "PROJECTION_PART_HEADER_OR_RANGE_CHANGED")
        for ordinal, row in zip(range(part["row_start"], part["row_stop_exclusive"]), rows[1:]):
            require(0 <= ordinal < n and len(row) == 1 + len(columns) and row[0] == str(ordinal), "PROJECTION_ROW_ORDINAL_CHANGED")
            for name, value in zip(columns, row[1:]):
                require(name not in values[ordinal], "PROJECTION_DUPLICATE_CELL")
                values[ordinal][name] = value
    require(all(set(row) == set(header) for row in values), "PROJECTION_MISSING_CELL_OR_ROW")
    raw = csv_bytes(header, [[row[name] for name in header] for row in values])
    require(len(raw) == description["source_bytes"] and hashlib.sha256(raw).hexdigest() == description["source_sha256"],
            "PROJECTION_ORIGINAL_CSV_BYTES_NOT_EXACTLY_RESTORED")
    return raw


def project(source, expected_index_sha, output):
    source = Path(source).resolve(); original_index = checked(source, "INDEX.json", expected_index_sha)
    index = read(original_index)
    require(index["schema"] == "semantic_p2s_receipt_report_index_v1" and index["outer_results_loaded"] is False and
            index["new_fits_or_forwards"] == 0, "LOCKED_REPORT_INDEX_REQUIRED")
    names = [r["file"] for r in index["files"]]
    require(len(names) == len(set(names)) and "INDEX.json" not in names and "SOURCE_REPORT_INDEX.json" not in names, "ORIGINAL_REPORT_FILE_NAMES_INVALID")
    candidates = [r for r in index["files"] if r["file"].startswith("04_all_2376_readout_cells") and r["file"].endswith(".csv")]
    require(len(candidates) == 3, "EXACT_THREE_WIDE_ORIGINAL_CSV_REQUIRED")
    # Verify all source bytes before writing any public projection.
    for row in index["files"]:
        path = checked(source, row["file"], row["sha256"])
        require(path.stat().st_size == row["bytes"] <= MAX_BYTES, "ORIGINAL_REPORT_SIZE_CHANGED")
    target = Path(output); target.mkdir(parents=True, exist_ok=False)
    files, projections = [], []
    wide = {r["file"] for r in candidates}
    for row in index["files"]:
        path = source / row["file"]
        if row["file"] in wide:
            description = project_csv(path, target)
            require(reconstruct(target, description) == path.read_bytes(), "FULL_LOSSLESS_CSV_ROUNDTRIP_FAILED")
            projections.append(description)
            files.extend({k: p[k] for k in ("file", "bytes", "sha256")} for p in description["parts"])
        else:
            if path.suffix == ".csv":
                header = next(csv.reader(io.StringIO(path.read_text(encoding="utf-8")))); require(len(header) <= MAX_COLUMNS, "UNEXPECTED_OTHER_WIDE_CSV")
            files.append(write(target / row["file"], path.read_bytes()))
    files.append(write(target / "SOURCE_REPORT_INDEX.json", original_index.read_bytes()))
    publication = {**index, "schema": "semantic_p2s_lossless_publication_report_index_v1", "files": files,
        "source_report_index_file": "SOURCE_REPORT_INDEX.json", "source_report_index_sha256": expected_index_sha,
        "projection_source": {"file": SCRIPT, "sha256": sha(ROOT / SCRIPT)}, "projection_test": {"file": TEST, "sha256": sha(ROOT / TEST)},
        "column_projection": projections, "original_reporter_or_reports_modified": False,
        "source_wide_CSV_published_verbatim": False, "raw_patient_rows_in_projection": False,
        "aggregate_row_ordinal_is_not_patient_identifier": True, "new_fits_forwards_or_scoring": 0,
        "all_three_original_CSV_bytewise_reconstruction_verified": True}
    raw = (json.dumps(publication, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    from scripts import bgb_en_publication as guard
    guard.validate_public_json(publication)
    require(not any(p.search(raw.decode()) for p in guard.PATTERNS.values()), "PROJECTION_PUBLIC_METADATA_PATTERN_FAILED")
    info = write(target / "PUBLICATION_REPORT_INDEX.json", raw)
    return {"status": "LOSSLESS_THREE_CSV_PUBLIC_COLUMN_PROJECTION_COMPLETE", "physical_files": len(files) + 1,
            "projected_original_CSVs": 3, "publication_report_index": info, "new_fits_forwards_or_scoring": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-reports", "expected-source-report-index-sha256", "output-newdir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(project(args.source_reports, args.expected_source_report_index_sha256, args.output_newdir), sort_keys=True))


if __name__ == "__main__":
    main()

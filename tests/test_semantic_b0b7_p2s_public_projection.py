"""Lossless normal CSV column shards; synthetic aggregates, no private data."""
from copy import deepcopy
import csv
import io
import json

import pytest

from scripts import semantic_b0b7_p2s_public_projection as projection


def source_csv(path, rows=9):
    header = ["field_%03d" % i for i in range(154)]
    values = [[("comma,quoted\"value" if j == 151 else "" if j == 149 else str(i * 154 + j)) for j in range(154)] for i in range(rows)]
    path.write_bytes(projection.csv_bytes(header, values))
    return path, header, values


def test_154_columns_become_at_most150_with_exact_original_byte_roundtrip(tmp_path):
    source, header, values = source_csv(tmp_path / "wide.csv")
    output = tmp_path / "public"; output.mkdir()
    result = projection.project_csv(source, output)
    assert len(result["parts"]) == 2 and result["source_headers"] == header
    for part in result["parts"]:
        data = list(csv.reader(io.StringIO((output / part["file"]).read_text(encoding="utf-8"))))
        assert len(data[0]) <= 150 and data[0][0] == projection.ORDINAL
        assert [int(r[0]) for r in data[1:]] == list(range(9))
    assert projection.reconstruct(output, result) == source.read_bytes()


@pytest.mark.parametrize("change", ["missing_part", "duplicate_part", "sha", "ordinal", "row_range", "source_sha"])
def test_missing_or_changed_csv_cells_cannot_claim_lossless_reconstruction(tmp_path, change):
    source, _, _ = source_csv(tmp_path / "wide.csv")
    output = tmp_path / "public"; output.mkdir()
    result = projection.project_csv(source, output)
    if change == "missing_part": result["parts"].pop()
    elif change == "duplicate_part": result["parts"].append(deepcopy(result["parts"][0]))
    elif change == "sha": result["parts"][0]["sha256"] = "0" * 64
    elif change == "ordinal": result["ordinal_start"] = 1
    elif change == "row_range": result["parts"][0]["row_start"] = 1
    else: result["source_sha256"] = "0" * 64
    with pytest.raises(ValueError): projection.reconstruct(output, result)


def test_row_and_byte_splitting_preserve_all154_columns_without_changing_bound(tmp_path, monkeypatch):
    source, _, _ = source_csv(tmp_path / "wide.csv", rows=17)
    output = tmp_path / "public"; output.mkdir()
    # Synthetic short headers keep this below the test byte limit.
    monkeypatch.setattr(projection, "MAX_BYTES", 6000)
    monkeypatch.setattr(projection, "MAX_LINES", 20)
    result = projection.project_csv(source, output)
    assert len(result["parts"]) > 2
    assert all((output / p["file"]).stat().st_size <= 6000 for p in result["parts"])
    assert projection.reconstruct(output, result) == source.read_bytes()


def test_full_public_index_keeps_original_index_bytes_and_copies_all_normal_reports(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    files = []
    for i in range(1, 4):
        p, _, _ = source_csv(source / ("04_all_2376_readout_cells_%02d.csv" % i))
        files.append({"file": p.name, "bytes": p.stat().st_size, "sha256": projection.sha(p)})
    normal = source / "summary.json"; normal.write_bytes(b'{"complete":true}\n')
    files.append({"file": normal.name, "bytes": normal.stat().st_size, "sha256": projection.sha(normal)})
    index = {"schema": "semantic_p2s_receipt_report_index_v1", "outer_results_loaded": False, "new_fits_or_forwards": 0,
             "files": files, "source_sha256": "a" * 64}
    index_path = source / "INDEX.json"; index_path.write_text(json.dumps(index), encoding="utf-8")
    out = tmp_path / "public"
    result = projection.project(source, projection.sha(index_path), out)
    public = projection.read(out / "PUBLICATION_REPORT_INDEX.json")
    assert result["projected_original_CSVs"] == 3
    assert (out / "SOURCE_REPORT_INDEX.json").read_bytes() == index_path.read_bytes()
    assert (out / "summary.json").read_bytes() == normal.read_bytes()
    assert all(not (out / p["source_file"]).exists() for p in public["column_projection"])
    assert all(projection.reconstruct(out, p) == (source / p["source_file"]).read_bytes() for p in public["column_projection"])
    assert {r["file"] for r in public["files"]} == {p.name for p in out.iterdir() if p.name != "PUBLICATION_REPORT_INDEX.json"}
    with pytest.raises(FileExistsError): projection.project(source, projection.sha(index_path), out)

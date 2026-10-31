"""Read-only source-to-cache audit; decode covariates only, never outcome arrays.

The CLI requires explicit private input paths. Reports contain only aggregate
counts, source-file basenames, and hashes. Existing inputs are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_KEYS = (
    "patient_ids", "raw_rna", "raw_cna", "rna_genes", "cna_genes",
    "rna_gene_symbols", "cna_gene_symbols", "cancer_type",
)
PATIENT = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")
SOURCE_FILES = {"RNA": "data_mrna_seq_v2_rsem.txt", "CNA": "data_log2_cna.txt"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(json.dumps(list(a.shape)).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def same_array(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.number):
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def histogram(x):
    values, counts = np.unique(x, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def collapse_symbols(raw, symbols, log_rna):
    values = raw.copy()
    if log_rna:
        if not np.all(values[np.isfinite(values)] > -1):
            raise ValueError("RNA values violate the historical log2(x+1) domain")
        values = np.log2(values + np.float32(1))
    valid = sorted(set(s for s in symbols if s and s != "NAN"))
    locations = {}
    for index, symbol in enumerate(symbols):
        locations.setdefault(symbol, []).append(index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.stack(
            [np.nanmean(values[:, locations[s]], axis=1) for s in valid], axis=1
        ).astype(np.float64)
    return result, np.asarray(valid, dtype="U64"), locations


def reconstruct_raw(path, ids):
    table = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    source = table.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    wanted = set(ids.tolist())
    columns = {}
    for j, name in enumerate(table.columns[2:]):
        match = PATIENT.match(str(name).strip().upper())
        pid = match.group(1) if match else None
        if pid in wanted and str(name)[13:15] == "01":
            columns.setdefault(pid, []).append((str(name), j))
    if not all(pid in columns for pid in ids):
        raise ValueError("A cohort patient has no primary-tumor source column")
    selections = [sorted(columns[pid]) for pid in ids]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.stack(
            [np.nanmean(source[:, [index for _, index in selected]], axis=1)
             for selected in selections]
        )
    symbols = table.iloc[:, 0].fillna("").astype(str).str.strip().str.upper().to_numpy("U64")
    info = {
        "source_gene_rows": len(table), "source_sample_columns": source.shape[1],
        "patients_with_multiple_primary_columns": sum(len(s) > 1 for s in selections),
        "primary_columns_per_patient": histogram(np.asarray([len(s) for s in selections])),
        "selection": "Primary tumor sample type 01 only; sorted aliquots; float32 nanmean",
    }
    return result, symbols, info


def collapse_missing_audit(raw, canonical, symbols, locations, valid):
    blank = np.asarray([not s or s == "NAN" for s in symbols])
    duplicate_symbols = [s for s in valid if len(locations[s]) > 1]
    duplicated_columns = [j for s in duplicate_symbols for j in locations[s]]
    valid_to_column = {s: j for j, s in enumerate(valid)}
    raw_missing_recovered_by_duplicate = sum(
        int((~np.isfinite(raw[:, j]) & np.isfinite(canonical[:, valid_to_column[s]])).sum())
        for s in duplicate_symbols for j in locations[s]
    )
    return {
        "raw_nonfinite_cells": int((~np.isfinite(raw)).sum()),
        "canonical_nonfinite_cells": int((~np.isfinite(canonical)).sum()),
        "blank_symbol_source_rows_removed": int(blank.sum()),
        "duplicate_symbol_source_rows_removed": int(len(symbols) - blank.sum() - len(valid)),
        "duplicate_symbol_groups": len(duplicate_symbols),
        "blank_symbol_nonfinite_cells_removed": int((~np.isfinite(raw[:, blank])).sum()),
        "duplicate_symbol_raw_nonfinite_cells": int((~np.isfinite(raw[:, duplicated_columns])).sum()),
        "duplicate_symbol_missing_raw_cells_with_observed_same_symbol": raw_missing_recovered_by_duplicate,
        "canonical_missing_in_duplicate_symbol_columns": int(
            (~np.isfinite(canonical[:, [valid_to_column[s] for s in duplicate_symbols]])).sum()),
        "raw_nonfinite_per_patient": histogram((~np.isfinite(raw)).sum(1)),
        "canonical_nonfinite_per_patient": histogram((~np.isfinite(canonical)).sum(1)),
        "raw_nonfinite_per_gene_nonzero": histogram(
            (~np.isfinite(raw)).sum(0)[(~np.isfinite(raw)).any(0)]),
        "canonical_nonfinite_per_gene_nonzero": histogram(
            (~np.isfinite(canonical)).sum(0)[(~np.isfinite(canonical)).any(0)]),
        "no_missing_cell_filled_from_other_patients": True,
        "canonical_missing_rule": "Missing iff every primary-aliquot/same-symbol source contribution is missing",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Audit output is append-only; choose a new output file")
    canonical_manifest = json.loads((args.canonical.parent / "manifest.json").read_text())
    canonical_protocol = json.loads((args.canonical.parent / "protocol.json").read_text())
    parent_manifest = json.loads((args.parent.parent / "manifest.json").read_text())
    if Path(canonical_protocol["parent_run"]).resolve() != args.parent.parent.resolve():
        raise ValueError("Canonical protocol parent does not match the supplied parent")
    canonical_sha, parent_sha = sha256(args.canonical), sha256(args.parent)
    if canonical_sha != canonical_manifest["private_dataset_sha256"]:
        raise ValueError("Canonical container hash differs from its historical manifest")
    if parent_sha != parent_manifest["private_dataset_sha256"]:
        raise ValueError("Parent container hash differs from its historical manifest")
    parent_records = [record for record in canonical_manifest["sources"]
                      if Path(record["path"]).name == args.parent.name]
    if len(parent_records) != 1 or parent_records[0]["sha256"] != parent_sha:
        raise ValueError("Canonical manifest does not bind this exact parent container")
    inherited, result = {}, {}
    with np.load(args.canonical, allow_pickle=False) as cache, np.load(args.parent, allow_pickle=False) as parent:
        ids = cache["patient_ids"].astype(str)
        if len(ids) != len(set(ids.tolist())):
            raise ValueError("Patient IDs are not unique")
        cohort = cache["cancer_type"].astype(str)
        for key in ALLOWED_KEYS:
            if key == "cancer_type":
                continue
            if not same_array(cache[key], parent[key]):
                raise ValueError("An inherited covariate differs from the bound parent")
            inherited[key] = {
                "shape": list(cache[key].shape), "dtype": str(cache[key].dtype),
                "array_sha256": array_sha256(cache[key]), "exact_parent_inheritance": True,
            }
        for modality, filename in SOURCE_FILES.items():
            key = modality.lower()
            path = args.source_directory / filename
            source_sha = sha256(path)
            historical = [record for record in parent_manifest["inputs"]
                          if Path(record["path"]).name == filename]
            if len(historical) != 1 or historical[0]["sha256"] != source_sha:
                raise ValueError("Original source hash differs from the parent manifest")
            raw, symbols, selection = reconstruct_raw(path, ids)
            cached_raw = cache["raw_" + key]
            if not same_array(raw, cached_raw):
                raise ValueError("Primary-tumor original source does not exactly rebuild raw cache")
            rebuilt, valid, locations = collapse_symbols(raw, symbols, modality == "RNA")
            expected = cache[key + "_genes"]
            if not np.array_equal(valid, cache[key + "_gene_symbols"]):
                raise ValueError("Historical gene symbols/order are not reproduced")
            expected_missing = np.stack([
                (~np.isfinite(raw[:, locations[s]])).all(1) for s in valid], axis=1)
            if not np.array_equal(expected_missing, ~np.isfinite(expected)):
                raise ValueError("Canonical observed masks cannot be rebuilt from source measurements")
            exact = same_array(rebuilt, expected)
            precision = {"exact_canonical_values": exact}
            if not exact:
                if modality != "RNA":
                    raise ValueError("Canonical CNA differs from source symbol-aggregation")
                ref = np.log2((raw + np.float32(1)).astype(np.float64)).astype(np.float32)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ref = np.stack([np.nanmean(ref[:, locations[s]], axis=1) for s in valid], axis=1).astype(np.float64)
                absdiff = np.abs(rebuilt - expected)
                ulp = np.abs(np.spacing(expected.astype(np.float32))).astype(np.float64)
                relative = np.abs(ref - expected) / np.where(ulp > 0, ulp, 1.)
                max_ulp = float(np.nanmax(relative))
                if max_ulp > 1:
                    raise ValueError("RNA differs from rounded float64 reference by more than one float32 ULP")
                precision.update({
                    "kernel_rebuild_max_abs_difference": float(np.nanmax(absdiff)),
                    "float64_reference_max_float32_ULP": max_ulp,
                    "rounding_qualification": "Same raw values, symbols and masks; within one float32 ULP of float64 log reference; historical values retained",
                })
            missing = (~np.isfinite(expected)).sum(1)
            result[modality] = {
                "source_basename": filename, "source_sha256": source_sha,
                "source_bytes": path.stat().st_size, "source_hash_matches_historical_manifest": True,
                "raw_shape": list(raw.shape), "canonical_shape": list(expected.shape),
                "source_to_raw_exact": True, "canonical_symbols_and_order_exact": True,
                "canonical_observed_mask_exact": True, "precision": precision,
                "encoding": "log2(float32 raw + 1), symbol nanmean, float64 storage" if modality == "RNA"
                            else "Signed LOG2-VALUE CNA, no additional log, symbol nanmean, float64 storage",
                "global_imputation_in_canonical": False,
                "infinite_source_or_canonical_cells": int(np.isinf(raw).sum() + np.isinf(expected).sum()),
                "selection": selection,
                "missing_aggregation": collapse_missing_audit(raw, expected, symbols, locations, valid),
                "cohort_missingness": {
                    str(name): {"patients": int((cohort == name).sum()),
                                "canonical_nonfinite_cells": int(missing[cohort == name].sum()),
                                "patients_with_missing": int((missing[cohort == name] > 0).sum()),
                                "nonfinite_per_patient": histogram(missing[cohort == name])}
                    for name in np.unique(cohort)
                },
            }
    report = {
        "schema": "semantic_b0b7_canonical_omics_source_provenance_v1",
        "status": "SOURCE_RAW_CANONICAL_CHAIN_VERIFIED",
        "scope": "Covariate source reconstruction only; no survival or outer-test metrics",
        "source_sha256": sha256(Path(__file__)),
        "canonical_container_sha256": canonical_sha, "parent_container_sha256": parent_sha,
        "historical_manifest_container_hashes_match": True,
        "parent_dataset_bound_by_canonical_manifest": True,
        "patients": len(ids), "inherited_arrays": inherited, "modalities": result,
        "decoded_npz_keys": list(ALLOWED_KEYS),
        "survival_arrays_decoded": [], "outer_results_read": False,
        "model_forwards": 0, "model_fits": 0, "inputs_rewritten": False,
        "interpretation": [
            "raw_* retains source rows; *_genes retains sorted unique nonblank gene symbols.",
            "Missingness may decrease when blank symbol rows are removed or duplicate symbols have a measured counterpart.",
            "Same-symbol observed contribution aggregation is not cross-patient imputation.",
            "Gene-level missingness and cohort/platform confounding still require model and sensitivity evaluation.",
            "CNA signed dosage is not equivalent to molecular-program activation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report_sha256": sha256(args.output),
                      "RNA": result["RNA"], "CNA": result["CNA"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

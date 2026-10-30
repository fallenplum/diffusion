"""Create the two prelocked outcome-blind repetition partitions, preserving P1."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_b0b7_prepare import read_json, file_sha, legacy_data_module
from src.semantic_b0b7.data import pretext_split
from src.semantic_b0b7.p1r_data import molecular_memberships, build_partition_inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("assets", "legacy-root", "parent-input", "output-newdir"):
        p.add_argument("--" + field, type=Path, required=True)
    p.add_argument("--parent-input-sha256", required=True)
    a = p.parse_args()
    if file_sha(a.parent_input) != a.parent_input_sha256:
        raise ValueError("PARENT_INPUT_HASH_CHANGED")
    if a.output_newdir.exists():
        raise ValueError("REFUSE_OVERWRITE")
    torch.set_num_threads(2)
    assets = read_json(a.assets)
    old, _ = legacy_data_module(a.legacy_root)
    raw = old.load_raw(assets, include_endpoints=False)
    frozen = torch.load(a.parent_input, map_location="cpu", weights_only=False)
    frozen["frozen_input_sha256"] = a.parent_input_sha256
    members = molecular_memberships(raw, assets["gmt"])
    a.output_newdir.mkdir(parents=True)
    entries = [{"partition": 0, "file": str(a.parent_input), "sha256": a.parent_input_sha256,
                "identity": "UNMODIFIED_HISTORICAL_INPUT"}]
    for partition in (1, 2):
        fit, evaluation, sealed = pretext_split(raw["folds"], outer_fold=1, inner_partition=partition)
        data, prep = build_partition_inputs(raw, frozen, fit, evaluation, members,
                                            split_name="P1R_outer1_inner" + str(partition))
        path = a.output_newdir / ("partition" + str(partition) + ".private.pt")
        with path.open("xb") as handle:
            torch.save(data, handle)
        state = path.with_suffix(".preprocessing.pt")
        with state.open("xb") as handle:
            torch.save(prep, handle)
        entries.append({"partition": partition, "file": str(path), "sha256": file_sha(path),
                        "preprocessing_sha256": file_sha(state), "bytes": path.stat().st_size,
                        "fit_n": len(fit), "evaluation_n": len(evaluation), "sealed_n": len(sealed),
                        "identity": "NEW_PARTITION_ONLY_PREPROCESSING_NO_MODEL_FIT"})
        del data, prep
    manifest = {"schema": "semantic_b0b7_p1r_partition_manifest_v1", "partitions": entries,
                "source_sha256": file_sha(__file__),
                "builder_sha256": file_sha(ROOT / "src/semantic_b0b7/p1r_data.py"),
                "split_seed": 103001, "outcomes_read": False, "new_neural_fits": 0}
    with (a.output_newdir / "PARTITIONS.private.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

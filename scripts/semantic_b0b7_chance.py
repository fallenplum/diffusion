"""Exact retrieval chance under the existing validation support; zero fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.semantic_b0b7.pretext import PairMetadata, negative_support


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--expected-input-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    assert sha(args.inputs) == args.expected_input_sha256 and not args.output.exists()
    data = torch.load(args.inputs, map_location="cpu", weights_only=False)
    ix = np.asarray(data["valid_indices"], int)
    meta = {k: np.asarray(v)[ix] for k, v in data["metadata"].items()}
    gene_valid = np.asarray(data["omics"]["measured"])[ix].any(axis=(1, 2))
    scopes = [("global", None, np.ones(len(ix), bool))]
    for field in ("stage", "missing_pattern", "cohort"):
        scopes.extend((field, str(value), meta[field] == value) for value in np.unique(meta[field]))
    rows = []
    for scope, group, use in scopes:
        n = int(use.sum())
        if n < 2:
            continue
        metadata = PairMetadata(**{k: v[use] for k, v in meta.items()})
        score_valid = np.broadcast_to(gene_valid[use][None, :], (n, n)).copy()
        for mode in ("matched", "random"):
            negative, _ = negative_support(metadata, mode=mode)
            mask = negative & score_valid
            for direction, candidates, valid in (("image_to_gene", mask, score_valid),
                                                  ("gene_to_image", mask.T, score_valid.T)):
                count = candidates.sum(1)
                covered = valid.diagonal() & (count > 0)
                rows.append({"scope": scope, "group": group, "mode": "hard" if mode == "matched" else mode,
                             "direction": direction, "anchors": n, "anchors_covered": int(covered.sum()),
                             "exact_chance_mean": float(np.mean(1 / (count[covered] + 1))) if covered.any() else None,
                             "negative_count_min": int(count[covered].min()) if covered.any() else None,
                             "negative_count_max": int(count[covered].max()) if covered.any() else None,
                             "negative_count_mean": float(count[covered].mean()) if covered.any() else None})
    report = {"schema": "semantic_b0b7_retrieval_exact_chance_v1", "status": "COMPLETE",
              "input_sha256": args.expected_input_sha256, "source_sha256": sha(__file__), "rows": rows,
              "definition": "mean over the SAME eligible anchors of 1/(per-anchor valid negative count + 1)",
              "same_scope_support_as_existing_validation": True, "model_forwards": 0, "fits": 0,
              "DSS_read": False, "checkpoint_selection_changed": False,
              "interpretation": "descriptive chance reference; not a significance test or patient-matching proof"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "global": [x for x in rows if x["scope"] == "global"]}))


if __name__ == "__main__":
    main()

"""Evaluate fifteen locked G1 inner cells without any new model fitting."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase24_cdei_g1.evaluation import evaluate_stage, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("inner",), required=True)
    parser.add_argument("--source", required=True, help="locked training source SHA256")
    parser.add_argument("--private-root", type=Path, default=Path("/root/autodl-tmp/phase24_cdei_g1_private"))
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports/phase24_cdei_g1")
    args = parser.parse_args()
    from src.phase24_cdei_g1 import runtime
    if args.source != runtime.source():
        raise ValueError("expected source differs from current locked training bytes")
    holder = {}

    def endpoint_loader():
        from src.phase13_rna import data as inherited
        holder["data"] = inherited.load()
        return inherited.endpoints(holder["data"])

    def partition_loader(role, fold):
        from src.phase13_rna import data as inherited
        split = inherited.splits(holder["data"], "DSS", fold)
        return {"fit": split["inner_train"], "valid": split["inner_val"]}

    def anchor_loader(role, fold):
        import json
        import numpy as np
        import torch
        from src.phase19_attention import data as phase19
        from src.phase20_selective import data as phase20
        expected_source = phase20.cfg()["upstream_source"]
        features_path = phase19.P / "preprocessing" / f"fold{fold}_{role}.pt"
        feature_receipt = json.loads(features_path.with_suffix(".json").read_text())
        if (feature_receipt.get("sha256") != sha256_file(features_path)
                or feature_receipt.get("provenance", {}).get("source") != expected_source):
            raise ValueError("inherited feature cache has invalid source/hash")
        bank_path = phase19.P / "simple" / f"{role}_fold{fold}.pt"
        receipt = json.loads(bank_path.with_suffix(".json").read_text())
        if receipt.get("source") != expected_source or receipt.get("checkpoint_sha256") != sha256_file(bank_path):
            raise ValueError("inherited frozen anchor has invalid source/hash")
        features = torch.load(features_path, map_location="cpu", weights_only=False)
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        if bank.get("source") != expected_source:
            raise ValueError("serialized inherited anchor source mismatch")
        for key in ("fit", "valid"):
            if not np.array_equal(np.asarray(features[key]), np.asarray(bank[key])):
                raise ValueError("inherited feature/anchor partition mismatch")
        f, fit, valid = features["features"], np.asarray(features["fit"]), np.asarray(features["valid"])
        w = np.asarray(bank["models"]["J0"]["w"], np.float64)
        fit_anchor = (np.asarray(f["X"][fit], np.float64) @ w).astype(np.float32)
        for arm, key in (("J0", "X"), ("J_CS", "CS")):
            replay = np.asarray(f[key][valid] @ bank["models"][arm]["w"], np.float32)
            if not np.array_equal(replay, bank["models"][arm]["risk"]):
                raise ValueError("inherited frozen anchor cannot be reconstructed elementwise")
        return {"fit": fit, "valid": valid, "anchor_fit": fit_anchor,
                "anchor_valid": bank["models"]["J0"]["risk"], "J_CS_valid": bank["models"]["J_CS"]["risk"]}

    _, gate = evaluate_stage(args.private_root, args.report_dir, args.stage, args.source,
                             endpoint_loader, partition_loader=partition_loader, anchor_loader=anchor_loader)
    print("PHASE24_G1_EVALUATION", args.stage, gate["status"], flush=True)


if __name__ == "__main__":
    main()

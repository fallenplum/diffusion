"""Audit or execute the seven fixed outcome-free spatial source fits.

Weights and row-level predictions must be written to a private output directory.
The default audit stage performs zero model fits. Source paths are CLI inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bgb_en.external import (INPUT_NORMALIZATION, PROTOCOL, array_hash, atomic_json, fit_source_partition,
                                load_qualified_source, require, sha256, source_partitions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("audit", "train"), default="audit")
    parser.add_argument("--fit-index", type=int, choices=range(7))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=9033011)
    parser.add_argument("--epochs", type=int, choices=(20,), default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    require(not args.output_root.resolve().is_relative_to((ROOT/"reports").resolve()),
            "external weights and row predictions require a private directory outside public reports")
    torch.set_num_threads(2)
    data, audit = load_qualified_source(args.source_data, args.source_manifest)
    sources = {name: sha256(ROOT/name) for name in ("src/bgb_en/external.py", "scripts/bgb_en_pretrain_external.py", "src/bgb_en/pathology.py")}
    lineage = dict(protocol=PROTOCOL, sources=sources, source_data_sha256=audit["source_data_sha256"],
                   source_manifest_sha256=audit["source_manifest_sha256"], seed=args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root/"SOURCE_QUALIFICATION.json", dict(**audit, lineage=lineage))
    print(json.dumps(audit, sort_keys=True), flush=True)
    if args.stage == "audit" or args.dry_run:
        return
    for partition in source_partitions(data):
        index = partition["fit_index"]
        if args.fit_index is not None and index != args.fit_index:
            continue
        directory = args.output_root/f"fit_{index:02d}"
        receipt_path = directory/"COMPLETE.json"
        receipt_contract = dict(lineage=lineage, fit_index=index, role=partition["role"],
                                train_index_sha256=array_hash(partition["train"]),
                                valid_index_sha256=array_hash(partition["valid"]))
        if receipt_path.exists():
            require(args.resume, "completed source fit exists; use --resume for verified reuse")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            require(receipt.get("contract") == receipt_contract and receipt.get("status") == "COMPLETE_FIXED_SOURCE_FIT",
                    "existing source-fit receipt lineage differs")
            require(all(sha256(directory/name) == digest for name, digest in receipt["artifacts"].items()),
                    "existing source-fit artifact SHA differs")
            print(json.dumps(dict(fit_index=index, status="REUSED_VERIFIED_COMPLETE", actual_new_fits=0)), flush=True)
            continue
        require(not directory.exists(), "incomplete or unverified source fit exists; refuse automatic retraining")
        directory.mkdir()
        atomic_json(directory/"STARTED.json", receipt_contract)
        try:
            result = fit_source_partition(data, partition["train"], partition["valid"],
                                          device=args.device, seed=args.seed+index, epochs=args.epochs)
            torch.save(dict(schema="bgb_en_external_source_checkpoint_v2_unitnorm", input_normalization=INPUT_NORMALIZATION,
                            state_dict=result["state_dict"],
                            project_state_dict=result["project_state_dict"],
                            target_mean=torch.from_numpy(result["target_scaler"]["mean"]),
                            target_scale=torch.from_numpy(result["target_scaler"]["scale"]),
                            class_weights=torch.from_numpy(result["class_weights"]), contract=receipt_contract), directory/"checkpoint.pt")
            np.savez_compressed(directory/"PRIVATE_PARTITION.npz", train=partition["train"], valid=partition["valid"],
                                **({} if result["valid_prediction"] is None else result["valid_prediction"]))
            artifacts = {name: sha256(directory/name) for name in ("checkpoint.pt", "PRIVATE_PARTITION.npz")}
            if index == 6:
                torch.save(dict(schema="bgb_en_external_projection_transfer_v2_unitnorm", role="full_source_transfer",
                                input_normalization=INPUT_NORMALIZATION,
                                epochs=20, source_donors=6, project_state_dict=result["project_state_dict"],
                                source_data_sha256=audit["source_data_sha256"],
                                source_manifest_sha256=audit["source_manifest_sha256"],
                                TCGA_DSS_read=False, original_ST_test_read=False), directory/"UNI2_PROJECT_TRANSFER.pt")
                artifacts["UNI2_PROJECT_TRANSFER.pt"] = sha256(directory/"UNI2_PROJECT_TRANSFER.pt")
            scalar = {key: value for key, value in result.items()
                      if key not in ("state_dict", "project_state_dict", "target_scaler", "class_weights", "valid_prediction")}
            receipt = dict(status="COMPLETE_FIXED_SOURCE_FIT", contract=receipt_contract,
                           actual_new_supervised_fits=1, artifacts=artifacts, **scalar)
            atomic_json(receipt_path, receipt)
            print(json.dumps(dict(fit_index=index, status=receipt["status"], train=result["train_metrics"],
                                  heldout=result["valid_metrics"], epochs=20), sort_keys=True), flush=True)
        except BaseException as error:
            atomic_json(directory/"FAILED.json", dict(status="FAILED", exception_type=type(error).__name__))
            raise
    complete = []
    for index in range(7):
        path = args.output_root/f"fit_{index:02d}"/"COMPLETE.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            require(row.get("contract", {}).get("lineage") == lineage, "aggregate source fit lineage differs")
            complete.append(row)
    summary = dict(status="COMPLETE_SEVEN_SOURCE_FITS" if len(complete) == 7 else "PARTIAL_SOURCE_FITS",
                   completed_supervised_fits=len(complete), required_supervised_fits=7,
                   completed_heldout_fits=sum(row["contract"]["role"] == "leave_one_donor_out" for row in complete),
                   full_source_transfer_fit_complete=any(row["contract"]["role"] == "full_source_transfer" for row in complete),
                   total_optimizer_updates=sum(row["optimizer_updates"] for row in complete),
                   TCGA_DSS_read=False, original_ST_test_read=False, source_truth="tissue and local program proxies")
    atomic_json(args.output_root/"SUMMARY.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

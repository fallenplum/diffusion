"""Merge independent Phase10 fold directories without touching patient IDs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    for sub in ("oof", "cells", "private_fold_artifacts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    region_rows = []
    compute_rows = []
    for fold in range(1, 6):
        src = args.fold_root / f"fold{fold}"
        done = json.loads((src / "DONE.json").read_text(encoding="utf-8"))
        if done.get("status") != "PASS" or int(done.get("cells", 0)) != 6:
            raise RuntimeError(f"fold{fold} is not a complete six-arm cell set")
        for sub in ("oof", "cells"):
            for path in (src / sub).glob("*"):
                shutil.copy2(path, out / sub / path.name)
        for path in (src / "region_statistics.csv",):
            if path.exists(): region_rows.append(pd.read_csv(path))
        for path in (src / "compute_profile.csv",):
            if path.exists(): compute_rows.append(pd.read_csv(path))
    if len(list((out / "oof").glob("*.npz"))) != 30:
        raise RuntimeError("expected exactly 30 private OOF cells")
    if region_rows: pd.concat(region_rows, ignore_index=True).to_csv(out / "region_statistics.csv", index=False)
    if compute_rows: pd.concat(compute_rows, ignore_index=True).to_csv(out / "compute_profile.csv", index=False)
    for name in ("architecture_audit.json", "leakage_audit.json"):
        shutil.copy2(args.fold_root / "fold1" / name, out / name)
    (out / "MERGE_DONE.json").write_text(json.dumps({"status": "PASS", "folds": 5, "cells": 30}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

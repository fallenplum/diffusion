#!/usr/bin/env python
"""Standalone validator for completed Phase8-V0 artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.phase8_v0.runner import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root / "configs/phase8_v0_c3r_surv.json").read_text())
    print(json.dumps(validate(Path(args.output_dir), cfg), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Run Phase8-V0R append-only evaluation repair, diagnostics and controls."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase8_v0r.evaluation import reconstruct
from src.phase8_v0r.diagnostics import replay
from src.phase8_v0r.ridge import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["repair", "diagnostics", "baselines"], required=True)
    ap.add_argument("--formal-dir", default="/root/phase8_v0_c3r_surv")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    formal = Path(args.formal_dir); out = Path(args.output_dir)
    if args.mode == "repair": reconstruct(formal, out)
    elif args.mode == "diagnostics": replay(formal, out)
    else: run(formal, out)


if __name__ == "__main__":
    main()

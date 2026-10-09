#!/usr/bin/env python
"""Run Phase8-V0 C3R-Surv with explicit audit/smoke/full/validate/report modes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.phase8_v0.runner import audit_mode, full_mode, aggregate, report, validate
from src.phase8_v0.dataset import write_json, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["audit", "smoke", "full", "validate", "report"], required=True)
    ap.add_argument("--output-dir", default="/root/phase8_v0_c3r_surv")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; cfg = json.loads((root / "configs/phase8_v0_c3r_surv.json").read_text())
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if not (out / "protocol.json").exists():
        write_json(out / "protocol.json", cfg)
    if not (out / "manifest.json").exists():
        write_json(out / "manifest.json", {"status": "LOCKED", "protocol_sha256": digest(out / "protocol.json"), "repo_config": str(root / "configs/phase8_v0_c3r_surv.json")})
    if args.mode == "audit":
        audit_mode(out, cfg); return
    if args.mode == "smoke":
        full_mode(out, cfg, smoke=True); return
    if args.mode == "full":
        if (out / "DONE.json").exists() and not args.resume: raise RuntimeError("formal output exists; use --resume explicitly")
        full_mode(out, cfg, smoke=False); aggregate(out, cfg); validate(out, cfg); report(out)
        write_json(out / "DONE.json", {"status": "PASS", "protocol_sha256": digest(out / "protocol.json"), "cells": 105})
        return
    if args.mode == "validate":
        print(json.dumps(validate(out, cfg), ensure_ascii=False, indent=2)); return
    if args.mode == "report":
        report(out); return


if __name__ == "__main__":
    main()

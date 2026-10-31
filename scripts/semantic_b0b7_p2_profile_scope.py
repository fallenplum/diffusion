"""Separate DSS-stratified split planning; outputs row indices, not labels."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.semantic_b0b7_p2_matrix import raw_inputs
from scripts.semantic_b0b7_prepare import file_sha
from src.semantic_b0b7.p2_data import nested_split_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", required=True)
    p.add_argument("--legacy-root", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raw, _, _ = raw_inputs(a, endpoints=True)
    plan = nested_split_plan(raw["folds"], raw["e"], seed=27001)
    result = {role: {k: plan[name][k] for k in ("fit", "evaluation")}
              for role, name in (("inner", "outer1_inner0"), ("outer_final", "outer1_final"))}
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"scope_sha256": file_sha(a.output), "source_sha256": file_sha(__file__),
                      "scope_contains_outcome_values": False, "split_planner_reads_DSS": True,
                      "profile_process_will_read_DSS": False}))


if __name__ == "__main__":
    main()

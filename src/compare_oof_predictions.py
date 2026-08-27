from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_uni2_cox_baseline import concordance_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    root = args.root.resolve()

    prediction_path = (
        root
        / "outputs"
        / "baselines"
        / "multimodal_comparison"
        / "oof_predictions.csv"
    )
    frame = pd.read_csv(prediction_path)
    durations = frame["dss_months"].to_numpy(dtype=float)
    events = frame["dss_status"].to_numpy(dtype=int)
    risk_columns = [x for x in frame.columns if x.endswith("_oof_risk")]
    risks = {x.removesuffix("_oof_risk"): frame[x].to_numpy() for x in risk_columns}

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = [
        rng.integers(0, len(frame), len(frame)) for _ in range(args.bootstrap)
    ]
    rows = []
    for left, right in itertools.combinations(risks, 2):
        observed = concordance_index(durations, events, risks[left]) - concordance_index(
            durations, events, risks[right]
        )
        differences = []
        for index in bootstrap_indices:
            left_c = concordance_index(
                durations[index], events[index], risks[left][index]
            )
            right_c = concordance_index(
                durations[index], events[index], risks[right][index]
            )
            if np.isfinite(left_c) and np.isfinite(right_c):
                differences.append(left_c - right_c)
        differences_array = np.asarray(differences)
        rows.append(
            {
                "left_model": left,
                "right_model": right,
                "delta_c_left_minus_right": float(observed),
                "bootstrap_95_ci_low": float(
                    np.quantile(differences_array, 0.025)
                ),
                "bootstrap_95_ci_high": float(
                    np.quantile(differences_array, 0.975)
                ),
                "bootstrap_two_sided_p": float(
                    2
                    * min(
                        np.mean(differences_array <= 0),
                        np.mean(differences_array >= 0),
                    )
                ),
            }
        )

    output_path = prediction_path.parent / "paired_model_comparisons.csv"
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

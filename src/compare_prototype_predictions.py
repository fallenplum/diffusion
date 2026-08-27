from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from run_uni2_cox_baseline import concordance_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    root = args.root.resolve()

    base_dir = root / "outputs" / "baselines"
    conventional = pd.read_csv(
        base_dir / "multimodal_comparison" / "oof_predictions.csv"
    )
    prototypes = pd.read_csv(
        base_dir / "morphology_prototypes" / "oof_predictions.csv"
    )
    prototype_columns = [
        column
        for column in prototypes.columns
        if column == "patient_id" or column.endswith("_oof_risk")
    ]
    joined = conventional.merge(
        prototypes[prototype_columns],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(conventional):
        raise ValueError(
            f"Patient mismatch: conventional={len(conventional)}, joined={len(joined)}"
        )

    durations = joined["dss_months"].to_numpy(dtype=float)
    events = joined["dss_status"].to_numpy(dtype=int)
    risks = {
        "clinical": joined["clinical_oof_risk"].to_numpy(dtype=float),
        "uni2_mean": joined["uni2_oof_risk"].to_numpy(dtype=float),
        "prototype_only": joined["prototype_only_oof_risk"].to_numpy(dtype=float),
        "uni2_mean_pca32": joined["uni2_mean_pca32_oof_risk"].to_numpy(dtype=float),
        "uni2_mean_prototype": joined[
            "uni2_mean_prototype_oof_risk"
        ].to_numpy(dtype=float),
        "clinical_prototype": joined[
            "clinical_prototype_oof_risk"
        ].to_numpy(dtype=float),
        "clinical_uni2_mean": joined[
            "clinical_uni2_mean_oof_risk"
        ].to_numpy(dtype=float),
        "clinical_uni2_mean_prototype": joined[
            "clinical_uni2_mean_prototype_oof_risk"
        ].to_numpy(dtype=float),
    }

    rng = np.random.default_rng(args.seed)
    indices = [
        rng.integers(0, len(joined), len(joined))
        for _ in range(args.bootstrap)
    ]
    requested_pairs = [
        ("prototype_only", "uni2_mean_pca32"),
        ("uni2_mean_prototype", "uni2_mean_pca32"),
        ("clinical_prototype", "clinical"),
        ("clinical_uni2_mean", "clinical"),
        ("clinical_uni2_mean_prototype", "clinical_uni2_mean"),
        ("clinical_uni2_mean_prototype", "clinical"),
    ]
    rows = []
    for left, right in requested_pairs:
        observed = concordance_index(
            durations, events, risks[left]
        ) - concordance_index(durations, events, risks[right])
        differences = []
        for index in indices:
            left_c = concordance_index(
                durations[index], events[index], risks[left][index]
            )
            right_c = concordance_index(
                durations[index], events[index], risks[right][index]
            )
            if np.isfinite(left_c) and np.isfinite(right_c):
                differences.append(left_c - right_c)
        values = np.asarray(differences)
        rows.append(
            {
                "left_model": left,
                "right_model": right,
                "delta_c_left_minus_right": observed,
                "bootstrap_95_ci_low": np.quantile(values, 0.025),
                "bootstrap_95_ci_high": np.quantile(values, 0.975),
                "bootstrap_two_sided_p": 2
                * min(np.mean(values <= 0), np.mean(values >= 0)),
            }
        )
    result = pd.DataFrame(rows)
    output = (
        base_dir
        / "morphology_prototypes"
        / "paired_model_comparisons.csv"
    )
    result.to_csv(output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

"""Unified Phase8-V0 survival metrics."""
from __future__ import annotations

import numpy as np

from .cox import concordance, km_left


def fold_metrics(time, event, risk, train_time, train_event, fold):
    g = km_left(train_time, train_event, time)
    active = (event == 1) & (time <= 60)
    if np.any(g[active] <= 0): raise ValueError("IPCW support failure")
    weights = np.zeros(len(time)); weights[active] = 1.0 / g[active] ** 2
    return {"within_fold_harrell": concordance(time, event, risk),
            "within_fold_uno60": concordance(time, event, risk, weights), "uno_weight": weights,
            "censor_g": g, "pairs": int(((event[:, None] == 1) & ((time[:, None] < time[None, :]) | ((time[:, None] == time[None, :]) & (event[None, :] == 0)))).sum())}


def seed_summary(cells):
    out = {}
    for method, vals in cells.items():
        out[method] = {"within_fold_harrell": float(np.average([v["within_fold_harrell"] for v in vals], weights=[v["pairs"] for v in vals])),
                        "within_fold_uno60": float(np.average([v["within_fold_uno60"] for v in vals], weights=[v["pairs"] for v in vals])),
                        "fold_harrell_mean": float(np.mean([v["within_fold_harrell"] for v in vals])),
                        "fold_harrell_worst": float(np.min([v["within_fold_harrell"] for v in vals]))}
    return out

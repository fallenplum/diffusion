from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


NEURAL_SEEDS = (20260904, 20260905, 20260906)
BASELINE_SEED = 20260904


def set_seed(seed: int) -> None:
    """Set all local RNGs used by the benchmark."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def stable_int(value: str, modulo: int = 2**31 - 1) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) % modulo


def cox_loss(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
    """Negative Breslow partial log likelihood with T_j >= T_i risk sets."""
    if times.numel() == 0:
        return risks.sum() * 0.0
    order = torch.argsort(times, descending=True, stable=True)
    ordered_risk = risks[order]
    ordered_events = events[order].to(dtype=ordered_risk.dtype)
    log_risk = torch.logcumsumexp(ordered_risk, dim=0)
    n_events = torch.clamp(ordered_events.sum(), min=1.0)
    return ((log_risk - ordered_risk) * ordered_events).sum() / n_events


def cox_risk_gradient(times: np.ndarray, events: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Analytic derivative of :func:`cox_loss` with respect to risk scores."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    risks = np.asarray(risks, dtype=float)
    out = np.zeros_like(risks, dtype=float)
    event_indices = np.flatnonzero(events == 1)
    for i in event_indices:
        mask = times >= times[i]
        z = risks[mask]
        z = z - np.max(z)
        weights = np.exp(z)
        weights /= np.sum(weights)
        out[mask] += weights
        out[i] -= 1.0
    return out / max(1, len(event_indices))


def harrell_c(times: Iterable[float], events: Iterable[int], risks: Iterable[float]) -> float:
    t = np.asarray(list(times), dtype=float)
    e = np.asarray(list(events), dtype=int)
    r = np.asarray(list(risks), dtype=float)
    concordant = 0.0
    comparable = 0
    for i in np.flatnonzero(e == 1):
        mask = t > t[i]
        comparable += int(mask.sum())
        concordant += float((r[i] > r[mask]).sum())
        concordant += 0.5 * float((r[i] == r[mask]).sum())
    return float(concordant / comparable) if comparable else float("nan")


def censoring_survival(train_times: Iterable[float], train_events: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return event times and G(t)=P(C>=t) from an outer-train fold."""
    t = np.asarray(list(train_times), dtype=float)
    e = np.asarray(list(train_events), dtype=int)
    unique = np.sort(np.unique(t[np.isfinite(t)]))
    values: list[float] = []
    surv = 1.0
    for x in unique:
        at_risk = int(np.sum(t >= x))
        censor = int(np.sum((t == x) & (e == 0)))
        if at_risk:
            surv *= 1.0 - censor / at_risk
        values.append(surv)
    return unique, np.asarray(values, dtype=float)


def lookup_censor_survival(times: np.ndarray, km_times: np.ndarray, km_values: np.ndarray) -> np.ndarray:
    result = np.ones(len(times), dtype=float)
    for i, x in enumerate(times):
        where = np.flatnonzero(km_times <= x)
        result[i] = km_values[where[-1]] if len(where) else 1.0
    return result


def uno_c_index(
    test_times: Iterable[float],
    test_events: Iterable[int],
    risks: Iterable[float],
    train_times: Iterable[float],
    train_events: Iterable[int],
) -> tuple[float, float, float]:
    """Uno C numerator/denominator and ratio using outer-train censoring weights."""
    t = np.asarray(list(test_times), dtype=float)
    e = np.asarray(list(test_events), dtype=int)
    r = np.asarray(list(risks), dtype=float)
    km_t, km_g = censoring_survival(train_times, train_events)
    g = lookup_censor_survival(t, km_t, km_g)
    numerator = 0.0
    denominator = 0.0
    for i in np.flatnonzero(e == 1):
        if g[i] <= 1e-8:
            continue
        mask = t > t[i]
        weight = 1.0 / (g[i] ** 2)
        denominator += weight * float(mask.sum())
        numerator += weight * (float((r[i] > r[mask]).sum()) + 0.5 * float((r[i] == r[mask]).sum()))
    return float(numerator / denominator) if denominator else float("nan"), numerator, denominator


def aggregate_uno(rows: list[dict[str, Any]]) -> float:
    numerator = sum(float(row.get("uno_numerator", 0.0)) for row in rows)
    denominator = sum(float(row.get("uno_denominator", 0.0)) for row in rows)
    return float(numerator / denominator) if denominator else float("nan")


def run_functional_checks() -> dict[str, Any]:
    """Autograd, finite-difference, one-step descent and risk-direction gates."""
    times = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    events = np.asarray([1, 0, 1, 0, 1], dtype=int)
    values = np.asarray([0.4, -0.2, 0.3, 0.1, -0.1], dtype=float)
    risk = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    loss = cox_loss(torch.tensor(times), torch.tensor(events), risk)
    loss.backward()
    analytic = cox_risk_gradient(times, events, values)
    autograd = risk.grad.detach().cpu().numpy()
    epsilon = 1e-5
    finite = np.zeros_like(values)
    for j in range(len(values)):
        plus = values.copy(); plus[j] += epsilon
        minus = values.copy(); minus[j] -= epsilon
        finite[j] = (float(cox_loss(torch.tensor(times), torch.tensor(events), torch.tensor(plus))) - float(cox_loss(torch.tensor(times), torch.tensor(events), torch.tensor(minus)))) / (2 * epsilon)
    step = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    before = float(cox_loss(torch.tensor(times), torch.tensor(events), step))
    step.data -= 0.1 * step.grad if step.grad is not None else 0.0
    # The explicit gradient update above needs one backward pass.
    step = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    before = float(cox_loss(torch.tensor(times), torch.tensor(events), step))
    before_loss = cox_loss(torch.tensor(times), torch.tensor(events), step)
    before_loss.backward()
    with torch.no_grad():
        step -= 0.1 * step.grad
    after = float(cox_loss(torch.tensor(times), torch.tensor(events), step))
    # A high risk for an observed early event must reduce the loss relative to a reversed direction.
    risk_good = np.asarray([2.0, 0.0, 1.0, -0.5, -1.0])
    risk_bad = -risk_good
    direction = float(cox_loss(torch.tensor(times), torch.tensor(events), torch.tensor(risk_good)) < cox_loss(torch.tensor(times), torch.tensor(events), torch.tensor(risk_bad)))
    result = {
        "analytic_vs_autograd_max_abs": float(np.max(np.abs(analytic - autograd))),
        "analytic_vs_finite_difference_max_abs": float(np.max(np.abs(analytic - finite))),
        "autograd_match": bool(np.allclose(analytic, autograd, atol=1e-6, rtol=1e-5)),
        "finite_difference_match": bool(np.allclose(analytic, finite, atol=2e-5, rtol=2e-4)),
        "one_step_before": before,
        "one_step_after": after,
        "one_step_descent": bool(after < before),
        "risk_direction_sanity": bool(direction),
    }
    result["status"] = bool(result["autograd_match"] and result["finite_difference_match"] and result["one_step_descent"] and result["risk_direction_sanity"])
    return result


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**3))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def seconds(self) -> float:
        return float(time.perf_counter() - self.started)

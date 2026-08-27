from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def harrell_c(durations: np.ndarray, events: np.ndarray, risks: np.ndarray) -> float:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    risks = np.asarray(risks, dtype=float)
    concordant = 0.0
    comparable = 0
    for i in range(len(durations)):
        if events[i] != 1:
            continue
        mask = durations > durations[i]
        if not np.any(mask):
            continue
        other = risks[mask]
        comparable += int(mask.sum())
        concordant += float(np.sum(risks[i] > other))
        concordant += 0.5 * float(np.sum(risks[i] == other))
    return concordant / comparable if comparable else float("nan")


def censoring_km(train_times: np.ndarray, train_events: np.ndarray):
    times = np.asarray(train_times, dtype=float)
    censor = 1 - np.asarray(train_events, dtype=int)
    unique = np.sort(np.unique(times))
    survival = 1.0
    values = []
    for time in unique:
        at_risk = int(np.sum(times >= time))
        censored = int(np.sum((times == time) & (censor == 1)))
        if at_risk:
            survival *= 1.0 - censored / at_risk
        values.append(survival)
    unique = np.asarray(unique, dtype=float)
    values = np.asarray(values, dtype=float)

    def predict(query: np.ndarray | float, left_limit: bool = True) -> np.ndarray:
        q = np.asarray(query, dtype=float)
        side = "left" if left_limit else "right"
        index = np.searchsorted(unique, q, side=side) - 1
        result = np.ones_like(q, dtype=float)
        valid = index >= 0
        result[valid] = values[index[valid]]
        return result

    return predict


def uno_c(
    durations: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    censor_survival_at_event: np.ndarray,
) -> float:
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=int)
    risks = np.asarray(risks, dtype=float)
    g = np.asarray(censor_survival_at_event, dtype=float)
    numerator = 0.0
    denominator = 0.0
    for i in range(len(durations)):
        if events[i] != 1 or not np.isfinite(g[i]) or g[i] <= 1e-8:
            continue
        mask = durations > durations[i]
        n_pairs = int(mask.sum())
        if not n_pairs:
            continue
        weight = 1.0 / (g[i] ** 2)
        other = risks[mask]
        numerator += weight * (
            float(np.sum(risks[i] > other)) + 0.5 * float(np.sum(risks[i] == other))
        )
        denominator += weight * n_pairs
    return numerator / denominator if denominator > 0 else float("nan")


def fit_cox(times: np.ndarray, events: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    x = np.asarray(x, dtype=float)
    event_indices = np.flatnonzero(events == 1)
    if not len(event_indices):
        raise ValueError("No events in Cox training set")
    risk_masks = [times >= times[i] for i in event_indices]

    def objective(beta: np.ndarray):
        eta = x @ beta
        log_likelihood = 0.0
        gradient = np.zeros_like(beta)
        for event_index, risk_mask in zip(event_indices, risk_masks):
            risk_eta = eta[risk_mask]
            normalizer = logsumexp(risk_eta)
            weights = np.exp(risk_eta - normalizer)
            log_likelihood += eta[event_index] - normalizer
            gradient += x[event_index] - weights @ x[risk_mask]
        scale = float(len(event_indices))
        loss = -log_likelihood / scale + 0.5 * alpha * float(beta @ beta)
        grad = -gradient / scale + alpha * beta
        return loss, grad

    result = minimize(
        objective,
        np.zeros(x.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-9, "gtol": 1e-6},
    )
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Cox optimization failed: {result.message}")
    return np.asarray(result.x, dtype=float)


def scaler_pca(
    x_train: np.ndarray, x_eval: np.ndarray, n_components: int, seed: int
):
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    eval_scaled = scaler.transform(x_eval)
    n = min(n_components, len(x_train) - 1, x_train.shape[1])
    pca = PCA(n_components=n, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(train_scaled), pca.transform(eval_scaled), scaler, pca


def update_run_state(root: Path, completed_job: str, next_stage: str | None = None) -> None:
    path = root / "outputs" / "pathology_context_phase1" / "run_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    if completed_job not in state["completed_jobs"]:
        state["completed_jobs"].append(completed_job)
    state["pending_jobs"] = [x for x in state["pending_jobs"] if x != completed_job]
    if next_stage:
        state["stage"] = next_stage
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


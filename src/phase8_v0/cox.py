"""Numerically checked Efron Cox utilities used by Phase8-V0."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def efron_matrix(time, event):
    rows = []
    for t in np.unique(time[event == 1]):
        die = (time == t) & (event == 1)
        count = int(die.sum())
        risk = (time >= t).astype(float)
        for l in range(count):
            rows.append(risk - (l / count) * die)
    return np.stack(rows)


def efron_loss_grad(x, time, event, lam=0.0):
    a = efron_matrix(time, event)
    n_events = int(event.sum())
    event_x = x[event == 1].sum(0)

    def objective(v):
        risk = x @ v
        shift = risk.max()
        exp_r = np.exp(risk - shift)
        den = a @ exp_r
        loss = ((np.log(den) + shift).sum() - risk[event == 1].sum()) / n_events + .5 * lam * (v @ v)
        grad = (x.T @ (exp_r * (a.T @ (1.0 / den))) - event_x) / n_events + lam * v
        return float(loss), grad

    return objective


def fit_cox(x, time, event, lam):
    objective = efron_loss_grad(x=np.asarray(x, float), time=np.asarray(time, float), event=np.asarray(event, int), lam=lam)
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 5000, "ftol": 1e-13, "gtol": 1e-7, "maxls": 50})
    loss, gradient = objective(result.x)
    g = float(np.max(np.abs(gradient)))
    if not np.isfinite(loss) or g > 1e-5:
        raise RuntimeError(f"Cox fit failed: {result.message}; gradient={g}")
    return result.x, {"iterations": int(result.nit), "gradient_max": g, "objective": loss,
                      "optimizer_message": str(result.message)}


def km_left(train_time, train_event, query):
    unique = np.unique(train_time)
    surv = []
    value = 1.0
    for t in unique:
        n = (train_time >= t).sum() - ((train_time == t) & (train_event == 1)).sum()
        censor = ((train_time == t) & (train_event == 0)).sum()
        value *= 1 - censor / n if censor else 1.0
        surv.append(value)
    pos = np.searchsorted(unique, query, side="left") - 1
    return np.where(pos >= 0, np.asarray(surv)[np.maximum(pos, 0)], 1.0)


def concordance(time, event, risk, weights=None, folds=None):
    time = np.asarray(time); event = np.asarray(event); risk = np.asarray(risk)
    eligible = (event[:, None] == 1) & ((time[:, None] < time[None, :]) |
                 ((time[:, None] == time[None, :]) & (event[None, :] == 0)))
    if folds is not None:
        eligible &= np.asarray(folds)[:, None] == np.asarray(folds)[None, :]
    weight = np.ones(len(time)) if weights is None else np.asarray(weights)
    pair = eligible * weight[:, None]
    credit = (risk[:, None] > risk[None, :]).astype(float) + .5 * (risk[:, None] == risk[None, :])
    den = pair.sum()
    return float((pair * credit).sum() / den) if den else float("nan")


def finite_difference_check(seed=20261018):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(11, 7)); time = np.array([1, 1, 2, 2, 2, 3, 4, 4, 5, 6, 7.])
    event = np.array([1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    w = rng.normal(size=7) * .1
    fun = efron_loss_grad(x, time, event, .1)
    eps = 1e-6; numerical = []
    for j in range(7):
        wp = w.copy(); wm = w.copy(); wp[j] += eps; wm[j] -= eps
        numerical.append((fun(wp)[0] - fun(wm)[0]) / (2 * eps))
    return {"status": "PASS", "finite_difference_max_error": float(np.max(np.abs(fun(w)[1] - numerical)))}

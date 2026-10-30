"""One fold-local survival contract for the B0--B7 study.

Higher scores mean greater event risk. Equal-time event/censor pairs are
comparable; equal-time event/event pairs are not. Uno uses reverse Kaplan--
Meier estimated only on fit-train endpoints, evaluated *after* each observed
time (events leave before censoring jumps), and event times strictly < tau.
No training, model selection, patient input, or cross-fold pooling occurs here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch


def _endpoints(time, event):
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event)
    if (t.ndim != 1 or t.size == 0 or e.shape != t.shape
            or not np.isfinite(t).all() or (t < 0).any()
            or not np.isin(e, (0, 1)).all()):
        raise ValueError("ENDPOINTS_INVALID")
    return t, e.astype(bool)


def _risk(time, risk):
    r = np.asarray(risk, dtype=np.float64)
    if r.shape != time.shape or not np.isfinite(r).all():
        raise ValueError("RISK_INVALID")
    return r


@dataclass(frozen=True)
class PairComponents:
    numerator: float
    denominator: float
    comparable_pairs: int
    concordant_pairs: int
    discordant_pairs: int
    tied_risk_pairs: int
    event_censor_tied_time_pairs: int

    @property
    def score(self):
        return self.numerator / self.denominator if self.denominator > 0 else None


def pair_components(time, event, risk, anchor_weights=None, *, tied_tol=1e-8):
    """Count each directed comparable patient pair once, within one fold.

An IPCW weight belongs to the earlier-event anchor, not to both patients.
Zero-support inputs yield denominator=0 and score=None, never a made-up 0.5.
"""
    t, e = _endpoints(time, event)
    r = _risk(t, risk)
    if not np.isfinite(tied_tol) or tied_tol < 0:
        raise ValueError("TIED_RISK_TOLERANCE_INVALID")
    w = np.ones(len(t)) if anchor_weights is None else np.asarray(anchor_weights, dtype=np.float64)
    if w.shape != t.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("ANCHOR_WEIGHTS_INVALID")
    numerator = denominator = 0.0
    count = concordant = discordant = risk_ties = time_ties = 0
    for i in np.flatnonzero(e):
        other = np.flatnonzero((t > t[i]) | ((t == t[i]) & ~e))
        delta = r[i] - r[other]
        tied = np.abs(delta) <= tied_tol
        good = (delta > 0) & ~tied
        n = len(other)
        numerator += w[i] * (int(good.sum()) + 0.5 * int(tied.sum()))
        denominator += w[i] * n
        count += n
        concordant += int(good.sum())
        risk_ties += int(tied.sum())
        discordant += n - int(good.sum()) - int(tied.sum())
        time_ties += int(np.count_nonzero(t[other] == t[i]))
    return PairComponents(float(numerator), float(denominator), count, concordant,
                          discordant, risk_ties, time_ties)


def harrell_components(time, event, risk, *, tied_tol=1e-8):
    return pair_components(time, event, risk, tied_tol=tied_tol)


@dataclass(frozen=True)
class ReverseKaplanMeier:
    times: np.ndarray
    probability: np.ndarray

    def predict(self, query):
        q = np.asarray(query, dtype=np.float64)
        if q.ndim != 1 or not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("CENSOR_QUERY_INVALID")
        # The censoring tail is unknown if the final estimate is positive.
        if self.probability[-1] > 0 and (q > self.times[-1]).any():
            raise ValueError("CENSOR_QUERY_EXCEEDS_TRAIN_SUPPORT")
        ix = np.searchsorted(self.times, q, side="right") - 1
        return np.where(ix >= 0, self.probability[np.maximum(ix, 0)], 1.0)


def fit_censoring_km(train_time, train_event):
    """Reverse-KM, counting observed events before tied censoring events."""
    t, e = _endpoints(train_time, train_event)
    times = np.unique(t)
    prob = []
    survival = 1.0
    for at in times:
        censored = int(np.count_nonzero((t == at) & ~e))
        # This is intentionally not the ordinary event-KM denominator.
        risk_after_events = int(np.count_nonzero(t >= at)) - int(np.count_nonzero((t == at) & e))
        if censored:
            survival *= 1.0 - censored / risk_after_events
        prob.append(survival)
    times.setflags(write=False)
    probability = np.asarray(prob, dtype=np.float64)
    probability.setflags(write=False)
    return ReverseKaplanMeier(times, probability)


def uno_weights(train_time, train_event, time, event, *, tau=60.0):
    """Squared IPCW anchor weights with no epsilon clipping or test-KM fit."""
    t, e = _endpoints(time, event)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("UNO_TAU_INVALID")
    km = fit_censoring_km(train_time, train_event)
    eligible = e & (t < tau)
    probability = km.predict(t[eligible])
    if (probability <= 0).any():
        raise ValueError("IPCW_ZERO_CENSOR_SURVIVAL")
    weight = np.zeros(len(t), dtype=np.float64)
    weight[eligible] = 1.0 / probability**2
    return weight


def tau_support(train_time, train_event, *, tau=60.0):
    """Separate diagnostic for the documented positive-G(tau) requirement.

    The official IPCW scorer itself evaluates G only at eligible event times.
    Keeping this diagnostic explicit avoids silently changing its statistic.
    """
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("UNO_TAU_INVALID")
    try:
        probability = float(fit_censoring_km(train_time, train_event).predict(np.array([tau]))[0])
    except ValueError as exc:
        return {"supported": False, "probability": None, "reason": str(exc)}
    return {"supported": probability > 0, "probability": probability,
            "reason": None if probability > 0 else "IPCW_ZERO_CENSOR_SURVIVAL_AT_TAU"}


def uno60_components(train_time, train_event, time, event, risk, *, tau=60.0, tied_tol=1e-8):
    w = uno_weights(train_time, train_event, time, event, tau=tau)
    return pair_components(time, event, risk, w, tied_tol=tied_tol)


def evaluate_fold(train_time, train_event, time, event, risk, *, tau=60.0, tied_tol=1e-8):
    """One held-out fold; caller must enforce disjoint patient identity."""
    h = harrell_components(time, event, risk, tied_tol=tied_tol)
    u = uno60_components(train_time, train_event, time, event, risk, tau=tau, tied_tol=tied_tol)
    return {"Harrell": h.score, "Uno60": u.score,
            "Harrell_num": h.numerator, "Harrell_den": h.denominator,
            "Uno_num": u.numerator, "Uno_den": u.denominator,
            "comparable_pairs": h.comparable_pairs,
            "tau_support": tau_support(train_time, train_event, tau=tau)}


def aggregate_folds(rows: Iterable[dict]):
    """Sum within-fold numerators/denominators, without cross-fold risk pairs."""
    rows = list(rows)
    if not rows:
        raise ValueError("NO_FOLDS")
    keys = ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")
    out = {}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("FOLD_COMPONENT_INVALID")
        out[key] = float(values.sum())
    if out["Harrell_num"] > out["Harrell_den"] or out["Uno_num"] > out["Uno_den"]:
        raise ValueError("FOLD_CREDIT_EXCEEDS_DENOMINATOR")
    out["Harrell"] = out["Harrell_num"] / out["Harrell_den"] if out["Harrell_den"] else None
    out["Uno60"] = out["Uno_num"] / out["Uno_den"] if out["Uno_den"] else None
    out["fold_count"] = len(rows)
    return out


def efron_cox_loss(risk: torch.Tensor, time, event):
    """Per-event full-risk-set Efron negative log partial likelihood.

    The caller supplies every fit-train risk in one differentiable tensor.
    Event and censoring subjects at an event time both belong to the risk set.
    Log weights avoid exp-subtract cancellation for tied event groups.
    """
    if risk.ndim != 1 or not risk.is_floating_point() or not bool(torch.isfinite(risk).all()):
        raise ValueError("EFRON_RISK_INVALID")
    # Keep endpoint precision independent of float32 neural risk precision.
    t = torch.as_tensor(time, dtype=torch.float64, device=risk.device)
    e = torch.as_tensor(event, device=risk.device)
    if (t.shape != risk.shape or e.shape != risk.shape or risk.numel() == 0
            or not bool(torch.isfinite(t).all()) or bool((t < 0).any())
            or not bool(((e == 0) | (e == 1)).all()) or not bool((e == 1).any())):
        raise ValueError("EFRON_ENDPOINT_INVALID_OR_NO_EVENTS")
    observed = e == 1
    total = risk.new_zeros(())
    for at in torch.unique(t[observed], sorted=True):
        died = (t == at) & observed
        in_risk = t >= at
        count = int(died.sum().item())
        for ell in range(count):
            log_weight = torch.zeros_like(risk[in_risk])
            log_weight[died[in_risk]] = math.log1p(-ell / count)
            total = total + torch.logsumexp(risk[in_risk] + log_weight, dim=0)
        total = total - risk[died].sum()
    return total / observed.sum()


def efron_numpy(risk, time, event, *, ties="efron"):
    """Independent scalar patient-loop oracle, including Breslow comparison."""
    t, e = _endpoints(time, event)
    r = _risk(t, risk)
    if not e.any() or ties not in ("efron", "breslow"):
        raise ValueError("EFRON_ORACLE_NO_EVENTS_OR_TIES_INVALID")
    total = 0.0
    for at in sorted(set(t[e])):
        dead = [i for i in range(len(t)) if e[i] and t[i] == at]
        alive = [i for i in range(len(t)) if t[i] >= at]
        for ell in range(len(dead)):
            terms = []
            for j in alive:
                multiplier = 1.0 - ell / len(dead) if ties == "efron" and j in dead else 1.0
                terms.append(float(r[j]) + math.log(multiplier))
            maximum = max(terms)
            total += maximum + math.log(math.fsum(math.exp(x - maximum) for x in terms))
        total -= math.fsum(float(r[j]) for j in dead)
    return float(total / e.sum())

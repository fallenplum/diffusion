"""Synthetic functional gates for Phase 1.5R2.

No TCGA file is touched here.  The same functions are used by pytest and by
the experiment runner, which writes immutable JSON evidence under the R2
output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cox_gradients import (
    cox_loss_numpy,
    cox_risk_coefficients,
    cosine_similarity,
    finite_difference_gradient,
    legacy_wrong_cox_risk_coefficients,
)
from .rank_gradients import finite_difference_rank_gradient, rank_loss_numpy, rank_risk_coefficients


TIMES = np.array([5.0, 8.0, 10.0, 12.0, 15.0, 20.0], dtype=np.float64)
EVENTS = np.array([1, 0, 1, 1, 0, 0], dtype=np.int64)
RISKS = np.array([0.3, -0.2, 0.8, 0.1, -0.4, 0.5], dtype=np.float64)


def _write(path: Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def cox_loss_torch(times: np.ndarray, events: np.ndarray, risks: torch.Tensor) -> torch.Tensor:
    total = risks.new_zeros(())
    count = 0
    for index in range(len(times)):
        if int(events[index]) != 1:
            continue
        mask = torch.as_tensor(times >= times[index], dtype=torch.bool, device=risks.device)
        total = total + torch.logsumexp(risks[mask], dim=0) - risks[index]
        count += 1
    if count == 0:
        raise ValueError("at least one event is required")
    return total / count


def rank_loss_torch(times: np.ndarray, events: np.ndarray, risks: torch.Tensor) -> torch.Tensor:
    pair_i, pair_j = np.where((events[:, None] == 1) & (times[:, None] < times[None, :]))
    if not len(pair_i):
        return risks.new_zeros(())
    indices_i = torch.as_tensor(pair_i, dtype=torch.long, device=risks.device)
    indices_j = torch.as_tensor(pair_j, dtype=torch.long, device=risks.device)
    return F.softplus(-(risks[indices_i] - risks[indices_j])).mean()


def _comparison(left: np.ndarray, right: np.ndarray, rtol: float, atol: float) -> dict[str, Any]:
    difference = np.abs(left - right)
    return {
        "allclose": bool(np.allclose(left, right, rtol=rtol, atol=atol)),
        "max_abs_error": float(difference.max(initial=0.0)),
        "cosine_similarity": cosine_similarity(left, right),
    }


def run_cox_gradient_test(path: Path | None = None) -> dict[str, Any]:
    risks = torch.tensor(RISKS, dtype=torch.float64, requires_grad=True)
    loss = cox_loss_torch(TIMES, EVENTS, risks)
    loss.backward()
    autograd = risks.grad.detach().cpu().numpy()
    analytic = cox_risk_coefficients(TIMES, EVENTS, RISKS)
    finite_difference = finite_difference_gradient(TIMES, EVENTS, RISKS)
    legacy = legacy_wrong_cox_risk_coefficients(TIMES, EVENTS, RISKS)
    payload = {
        "test": "cox_analytic_vs_autograd_and_finite_difference",
        "times": TIMES.tolist(),
        "events": EVENTS.tolist(),
        "risks": RISKS.tolist(),
        "cox_loss": float(loss.detach()),
        "analytic_vs_autograd": _comparison(analytic, autograd, 1e-7, 1e-7),
        "analytic_vs_finite_difference": _comparison(analytic, finite_difference, 1e-4, 1e-5),
        "autograd_vs_finite_difference": _comparison(autograd, finite_difference, 1e-4, 1e-5),
        "legacy_wrong_vs_autograd": _comparison(legacy, autograd, 1e-7, 1e-7),
        "legacy_wrong_cosine_similarity": cosine_similarity(legacy, autograd),
        "corrected_cosine_similarity": cosine_similarity(analytic, autograd),
    }
    payload["status"] = bool(
        payload["analytic_vs_autograd"]["allclose"]
        and payload["analytic_vs_finite_difference"]["allclose"]
        and payload["autograd_vs_finite_difference"]["allclose"]
        and payload["corrected_cosine_similarity"] > 0.999999
    )
    return _write(path, payload)


def run_cox_descent_test(path: Path | None = None) -> dict[str, Any]:
    risk = nn.Parameter(torch.tensor(RISKS, dtype=torch.float64))
    optimizer = torch.optim.SGD([risk], lr=1e-3)
    loss_before = float(cox_loss_torch(TIMES, EVENTS, risk).detach())
    gradient = cox_risk_coefficients(TIMES, EVENTS, RISKS)
    optimizer.zero_grad(set_to_none=True)
    risk.grad = torch.as_tensor(gradient, dtype=risk.dtype)
    optimizer.step()
    loss_after = float(cox_loss_torch(TIMES, EVENTS, risk).detach())
    payload = {
        "test": "one_step_cox_descent",
        "learning_rate": 1e-3,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "loss_delta": loss_after - loss_before,
        "status": bool(loss_after < loss_before),
    }
    return _write(path, payload)


def run_rank_gradient_test(path: Path | None = None) -> dict[str, Any]:
    risks = torch.tensor(RISKS, dtype=torch.float64, requires_grad=True)
    loss = rank_loss_torch(TIMES, EVENTS, risks)
    loss.backward()
    autograd = risks.grad.detach().cpu().numpy()
    analytic = rank_risk_coefficients(TIMES, EVENTS, RISKS)
    finite_difference = finite_difference_rank_gradient(TIMES, EVENTS, RISKS)
    payload = {
        "test": "rank_analytic_vs_autograd_and_finite_difference",
        "n_comparable_pairs": int(np.sum((EVENTS[:, None] == 1) & (TIMES[:, None] < TIMES[None, :]))),
        "rank_loss": float(loss.detach()),
        "analytic_vs_autograd": _comparison(analytic, autograd, 1e-7, 1e-7),
        "analytic_vs_finite_difference": _comparison(analytic, finite_difference, 1e-4, 1e-5),
        "autograd_vs_finite_difference": _comparison(autograd, finite_difference, 1e-4, 1e-5),
    }
    payload["status"] = bool(
        payload["analytic_vs_autograd"]["allclose"]
        and payload["analytic_vs_finite_difference"]["allclose"]
        and payload["autograd_vs_finite_difference"]["allclose"]
    )
    return _write(path, payload)


def run_combined_gradient_test(path: Path | None = None) -> dict[str, Any]:
    rows = []
    for rank_lambda in (0.05, 0.2):
        risks = torch.tensor(RISKS, dtype=torch.float64, requires_grad=True)
        loss = cox_loss_torch(TIMES, EVENTS, risks) + rank_lambda * rank_loss_torch(TIMES, EVENTS, risks)
        loss.backward()
        autograd = risks.grad.detach().cpu().numpy()
        analytic = cox_risk_coefficients(TIMES, EVENTS, RISKS) + rank_lambda * rank_risk_coefficients(TIMES, EVENTS, RISKS)
        # The combined finite difference is evaluated directly to ensure the
        # two constituent formulas did not merely share the same bug.
        finite_difference = np.zeros_like(RISKS)
        epsilon = 1e-5
        for index in range(len(RISKS)):
            plus = RISKS.copy()
            minus = RISKS.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            finite_difference[index] = (
                cox_loss_numpy(TIMES, EVENTS, plus)
                + rank_lambda * rank_loss_numpy(TIMES, EVENTS, plus)
                - cox_loss_numpy(TIMES, EVENTS, minus)
                - rank_lambda * rank_loss_numpy(TIMES, EVENTS, minus)
            ) / (2.0 * epsilon)
        rows.append(
            {
                "rank_lambda": rank_lambda,
                "analytic_vs_autograd": _comparison(analytic, autograd, 1e-7, 1e-7),
                "analytic_vs_finite_difference": _comparison(analytic, finite_difference, 1e-4, 1e-5),
                "autograd_vs_finite_difference": _comparison(autograd, finite_difference, 1e-4, 1e-5),
            }
        )
    payload = {"test": "combined_cox_plus_rank_gradient", "lambdas": rows, "status": all(row["analytic_vs_autograd"]["allclose"] and row["analytic_vs_finite_difference"]["allclose"] and row["autograd_vs_finite_difference"]["allclose"] for row in rows)}
    return _write(path, payload)


def run_two_pass_equivalence_test(path: Path | None = None) -> dict[str, Any]:
    torch.manual_seed(20260830)
    features = torch.randn(10, 4, dtype=torch.float64)
    times = np.array([3.0, 5.0, 7.0, 8.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0], dtype=np.float64)
    events = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0], dtype=np.int64)
    template = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1)).double()
    direct = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1)).double()
    two_pass = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1)).double()
    direct.load_state_dict(template.state_dict())
    two_pass.load_state_dict(template.state_dict())
    direct_optimizer = torch.optim.SGD(direct.parameters(), lr=1e-3)
    two_pass_optimizer = torch.optim.SGD(two_pass.parameters(), lr=1e-3)

    direct_optimizer.zero_grad(set_to_none=True)
    direct_risk = direct(features).squeeze(-1)
    cox_loss_torch(times, events, direct_risk).backward()
    direct_optimizer.step()

    with torch.no_grad():
        first_pass_risk = two_pass(features).squeeze(-1).cpu().numpy()
    coefficients = cox_risk_coefficients(times, events, first_pass_risk)
    two_pass_optimizer.zero_grad(set_to_none=True)
    second_pass_risk = two_pass(features).squeeze(-1)
    pseudo_loss = torch.sum(torch.as_tensor(coefficients, dtype=second_pass_risk.dtype) * second_pass_risk)
    pseudo_loss.backward()
    two_pass_optimizer.step()

    differences = []
    for left, right in zip(direct.parameters(), two_pass.parameters()):
        differences.append(float(torch.max(torch.abs(left.detach() - right.detach())).cpu()))
    max_abs = max(differences, default=0.0)
    parameter_allclose = all(
        np.allclose(left.detach().cpu().numpy(), right.detach().cpu().numpy(), rtol=1e-5, atol=1e-6)
        for left, right in zip(direct.parameters(), two_pass.parameters())
    )
    payload = {
        "test": "two_pass_vs_direct_autograd_one_step",
        "n_patients": 10,
        "max_abs_parameter_error": max_abs,
        "allclose": bool(parameter_allclose),
        "status": bool(parameter_allclose and (max_abs < 1e-6 or max_abs <= 1e-5)),
    }
    return _write(path, payload)


def run_all(output_dir: Path | None = None) -> dict[str, Any]:
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    cox = run_cox_gradient_test(output_dir / "cox_gradient_test.json" if output_dir else None)
    descent = run_cox_descent_test(output_dir / "cox_descent_test.json" if output_dir else None)
    rank = run_rank_gradient_test(output_dir / "rank_gradient_test.json" if output_dir else None)
    combined = run_combined_gradient_test(output_dir / "combined_gradient_test.json" if output_dir else None)
    two_pass = run_two_pass_equivalence_test(output_dir / "two_pass_equivalence_test.json" if output_dir else None)
    summary = {
        "cox_autograd_match": bool(cox["analytic_vs_autograd"]["allclose"]),
        "cox_finite_difference_match": bool(cox["analytic_vs_finite_difference"]["allclose"] and cox["autograd_vs_finite_difference"]["allclose"]),
        "cox_one_step_descent": bool(descent["status"]),
        "rank_autograd_match": bool(rank["analytic_vs_autograd"]["allclose"]),
        "rank_finite_difference_match": bool(rank["analytic_vs_finite_difference"]["allclose"] and rank["autograd_vs_finite_difference"]["allclose"]),
        "combined_gradient_match": bool(combined["status"]),
        "two_pass_equivalence": bool(two_pass["status"]),
        "status": bool(cox["status"] and descent["status"] and rank["status"] and combined["status"] and two_pass["status"]),
    }
    return _write(output_dir / "functional_summary.json" if output_dir else None, summary)

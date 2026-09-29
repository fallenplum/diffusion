from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EarlyFusionModel(nn.Module):
    """B0: concatenated modality summaries followed by a Cox linear head."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = nn.Linear(input_dim, 1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features).squeeze(-1)


def fit_ridge_cox(
    features: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    alpha: float = 1e-2,
    steps: int = 220,
    device: torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """Fit the fixed B0 Cox head with the validated negative partial likelihood."""
    from .common import cox_loss

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(features, dtype=torch.float32, device=device)
    t = torch.tensor(times, dtype=torch.float32, device=device)
    e = torch.tensor(events, dtype=torch.float32, device=device)
    model = EarlyFusionModel(x.shape[1]).to(device)
    nn.init.zeros_(model.head.weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=0.0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        risk = model(x)
        loss = cox_loss(t, e, risk) + 0.5 * alpha * model.head.weight.square().sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    with torch.no_grad():
        final = cox_loss(t, e, model(x)) + 0.5 * alpha * model.head.weight.square().sum()
    return model.head.weight.detach().cpu().numpy().reshape(-1), float(final.detach().cpu())

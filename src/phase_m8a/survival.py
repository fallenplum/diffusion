from __future__ import annotations
import torch

def cox_loss_breslow_grouped(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
    """Grouped Breslow negative partial log-likelihood with tied event times."""
    times = times.reshape(-1)
    events = events.reshape(-1)
    risks = risks.reshape(-1)
    event_mask = events > 0
    n_events = event_mask.sum()
    if int(n_events.item()) == 0:
        return risks.sum() * 0.0
    unique_times = torch.unique(times[event_mask], sorted=True)
    terms = []
    for t in unique_times:
        deaths = event_mask & (times == t)
        risk_set = times >= t
        m = deaths.sum().to(dtype=risks.dtype)
        terms.append(m * torch.logsumexp(risks[risk_set], dim=0) - risks[deaths].sum())
    return torch.stack(terms).sum() / n_events.to(dtype=risks.dtype)

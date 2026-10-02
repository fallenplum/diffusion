from __future__ import annotations
import torch

def cox_loss_full(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
    """Historical full-risk-set position-wise Cox loss used by M7R."""
    order = torch.argsort(times, descending=True, stable=True)
    ordered_risk = risks[order]
    ordered_events = events[order]
    log_risk = torch.logcumsumexp(ordered_risk, dim=0)
    n = ordered_events.sum().clamp_min(1.0)
    return ((log_risk - ordered_risk) * ordered_events).sum() / n

def cox_loss_grouped(times: torch.Tensor, events: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
    event_mask=events>0; n=event_mask.sum()
    if int(n.item())==0:return risks.sum()*0
    terms=[]
    for t in torch.unique(times[event_mask],sorted=True):
        d=event_mask&(times==t); r=times>=t
        terms.append(d.sum().to(risks.dtype)*torch.logsumexp(risks[r],0)-risks[d].sum())
    return torch.stack(terms).sum()/n.to(risks.dtype)

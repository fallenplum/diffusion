import torch
from src.phase10_trimodal.math import efron_weights


def full_risk_efron(risk,time,event):
    # Each row of the coefficient matrix is a distinct Efron denominator.
    weights=torch.as_tensor(efron_weights(time,event),device=risk.device,dtype=risk.dtype)
    logw=torch.where(weights>0,weights.log(),torch.full_like(weights,-torch.inf))
    e=torch.as_tensor(event,device=risk.device,dtype=risk.dtype)
    return (torch.logsumexp(logw+risk[None,:],-1).sum()-(risk*e).sum())/e.sum()

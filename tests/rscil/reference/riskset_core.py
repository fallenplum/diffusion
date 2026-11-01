"""Risk-set geometry prototype for development only.

The score/information identities and orthogonal projection are CLASSICAL,
not a claim of original theory. Patient data are NEVER embedded here.
Efron tie handling matches the partial-likelihood geometry.
"""
import torch


def efron_nll(risk, time, event, *, average=True):
    """Negative Efron Cox log partial likelihood; risk higher means earlier event.

    Works with time ties; uses full risk set. No random mini-batch Cox.
    """
    r = risk.reshape(-1).to(torch.float64)
    t = torch.as_tensor(time, dtype=torch.float64, device=r.device).reshape(-1)
    e = torch.as_tensor(event, dtype=torch.bool, device=r.device).reshape(-1)
    if not (r.numel() == t.numel() == e.numel()):
        raise ValueError('shape mismatch')
    if not bool(torch.isfinite(r).all() and torch.isfinite(t).all()):
        raise ValueError('non-finite inputs')
    if int(e.sum()) == 0:
        raise ValueError('no events')
    loss = r.new_zeros(())
    for event_time in torch.unique(t[e]):
        dmask = e & (t == event_time)
        rmask = t >= event_time
        d = int(dmask.sum())
        rr = r[rmask]
        re = r[dmask]
        # Use common max for exact Efron denominator and stability.
        mx = rr.detach().max()
        total = torch.exp(rr - mx).sum()
        event_total = torch.exp(re - mx).sum()
        loss = loss - re.sum()
        for j in range(d):
            denominator = total - (j / d) * event_total
            loss = loss + mx + torch.log(denominator)
    return loss / int(e.sum()) if average else loss


def riskset_information(risk, time, event, F):
    """Observed information of a linear added covariate block under Efron.

    Risk is treated as the offset. Matrix = sum of risk-set weighted
    covariances at each Efron virtual failure. This is classical Cox math.
    """
    r = torch.as_tensor(risk, dtype=torch.float64).reshape(-1)
    t = torch.as_tensor(time, dtype=torch.float64, device=r.device).reshape(-1)
    e = torch.as_tensor(event, dtype=torch.bool, device=r.device).reshape(-1)
    x = torch.as_tensor(F, dtype=torch.float64, device=r.device)
    if x.ndim != 2 or x.size(0) != r.numel():
        raise ValueError('F must be [N,D]')
    out = torch.zeros((x.size(1), x.size(1)), dtype=x.dtype, device=x.device)
    for event_time in torch.unique(t[e]):
        dmask = e & (t == event_time)
        rmask = t >= event_time
        d = int(dmask.sum())
        xr = x[rmask]
        rr = r[rmask]
        is_event_in_risk = dmask[rmask]
        mx = rr.detach().max()
        w0 = torch.exp(rr - mx)
        for j in range(d):
            factor = torch.ones_like(w0)
            factor[is_event_in_risk] = 1.0 - j / d
            w = w0 * factor
            w = w / w.sum()
            avg = (w[:, None] * xr).sum(0)
            xc = xr - avg
            out = out + (xc * w[:, None]).T @ xc
    return (out + out.T) / 2


def orthogonalize_interaction(z, B, risk, time, event, ridge=1e-6):
    """Project z from linear nuisance span using Cox event-risk-set metric.

    Returns Z_perp and fit-only coefficient A, with Z_perp=Z-B@A.
    Projection alone DOES NOT add function-class expressivity when B remains
    in the downstream linear Cox head. Apply only within a training split.
    """
    z = torch.as_tensor(z, dtype=torch.float64)
    b = torch.as_tensor(B, dtype=torch.float64, device=z.device)
    if z.ndim == 1:
        z = z[:, None]
    if b.ndim != 2 or z.ndim != 2 or len(b) != len(z):
        raise ValueError('Z/B shape mismatch')
    cov = riskset_information(risk, time, event, torch.cat((b, z), 1))
    q = b.size(1)
    ibb = cov[:q, :q]
    ibz = cov[:q, q:]
    penalty = torch.eye(q, dtype=ibb.dtype, device=ibb.device) * float(ridge)
    coeff = torch.linalg.solve(ibb + penalty, ibz)
    return z - b @ coeff, coeff, cov

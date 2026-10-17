"""Budget-constrained gates and conservative attention redistribution.

These operators implement the locked Phase20 formulas, not a novelty claim.
The gate budget constrains sum(g), not the number of nonzero entries.
"""
import torch


def _arguments(scores, mask, fraction):
    if scores.ndim < 1 or scores.shape[-1] < 1 or not scores.is_floating_point():
        raise ValueError('scores must have a nonempty floating-point final axis')
    if mask.shape != scores.shape or mask.dtype != torch.bool or mask.device != scores.device:
        raise ValueError('mask must be Boolean and match scores')
    if not 0. <= fraction <= 1.:
        raise ValueError('budget fraction outside [0,1]')
    safe = torch.where(mask, scores, torch.zeros_like(scores)).double()
    n = mask.sum(-1, keepdim=True)
    lo = safe.masked_fill(~mask, torch.inf).amin(-1, keepdim=True)
    hi = safe.masked_fill(~mask, -torch.inf).amax(-1, keepdim=True)
    lo = torch.where(n > 0, lo, torch.zeros_like(lo))
    hi = torch.where(n > 0, hi, torch.zeros_like(hi))
    return safe, n, lo, hi


class _CappedSimplex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, mask, fraction):
        s, n, lo, hi = _arguments(scores, mask, fraction)
        target = n.double() * fraction
        lo = lo - 1.
        for _ in range(64):
            threshold = (lo + hi) * .5
            g = (s - threshold).clamp(0., 1.) * mask
            over = g.sum(-1, keepdim=True) > target
            lo, hi = torch.where(over, threshold, lo), torch.where(over, hi, threshold)
        g = (s - (lo+hi)*.5).clamp(0., 1.) * mask
        if fraction == 0.:
            g.zero_()
        elif fraction == 1.:
            g = mask.double()
        free = (g > 0.) & (g < 1.) & mask
        ctx.save_for_backward(free)
        return g.to(scores.dtype)

    @staticmethod
    def backward(ctx, grad):
        (free,) = ctx.saved_tensors
        partial = grad * free
        count = free.sum(-1, keepdim=True)
        center = partial.sum(-1, keepdim=True) / count.clamp_min(1)
        return (partial - center * free), None, None


class _DenseBudget(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, mask, fraction):
        s, n, lo, hi = _arguments(scores, mask, fraction)
        target = n.double() * fraction
        lo, hi = lo-40., hi+40.
        for _ in range(64):
            threshold = (lo+hi)*.5
            g = torch.sigmoid(s-threshold) * mask
            over = g.sum(-1, keepdim=True) > target
            lo, hi = torch.where(over, threshold, lo), torch.where(over, hi, threshold)
        g = torch.sigmoid(s-(lo+hi)*.5) * mask
        if fraction == 0.:
            g.zero_()
        elif fraction == 1.:
            g = mask.double()
        ctx.save_for_backward(g * (1.-g))
        return g.to(scores.dtype)

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        grad64 = grad.double()
        total = weight.sum(-1, keepdim=True)
        safe_total = torch.where(total > 0., total, torch.ones_like(total))
        centered = grad64 - (grad64*weight).sum(-1, keepdim=True)/safe_total
        return (weight*centered).to(grad.dtype), None, None


def capped_simplex(scores, mask, fraction=.25):
    """clip(score-tau,0,1), sum gate=fraction*nvalid, exact piecewise Jacobian."""
    return _CappedSimplex.apply(scores, mask, float(fraction))


def dense_budget_gate(scores, mask, fraction=.25):
    """sigmoid(score-tau), matching the same budget, with implicit Jacobian."""
    return _DenseBudget.apply(scores, mask, float(fraction))


def rank_one_energy_projection(summary):
    """sqrt(H) * (11^T/H) z, same H output coordinates; a rank-one map."""
    if summary.ndim < 1 or summary.shape[-1] != 16:
        raise ValueError('the locked summary axis has exactly sixteen coordinates')
    return (4. * summary.mean(-1, keepdim=True)).expand_as(summary)


def conservative_update(base, gate, delta, mask, rho=.5, diagnostic=False):
    """Redistribute selected probability mass, preserving unselected entries.

    w=a0*g; M=sum(w); Z=sum(w*exp(delta))/M;
    a1=a0+rho*w*(exp(delta)/Z-1). Empty selected mass is an identity.
    Accumulation in float64 avoids division artifacts for tiny selected mass;
    the returned probabilities retain the input dtype, without renormalizing.
    """
    if base.shape != gate.shape or base.shape != delta.shape or mask.shape != base.shape:
        raise ValueError('base/gate/delta/mask shapes differ')
    if mask.dtype != torch.bool or not 0. <= rho <= 1.:
        raise ValueError('invalid mask or rho')
    weight = torch.where(mask, base*gate, torch.zeros_like(base)).double()
    exp_delta = torch.where(mask, delta, torch.zeros_like(delta)).double().exp()
    mass = weight.sum(-1, keepdim=True)
    weighted_sum = (weight*exp_delta).sum(-1, keepdim=True)
    safe_mass = torch.where(mass > 0., mass, torch.ones_like(mass))
    normalizer = torch.where(mass > 0., weighted_sum/safe_mass, torch.ones_like(mass))
    change = rho * weight * (exp_delta/normalizer - 1.)
    updated = base + change.to(base.dtype)
    if not diagnostic:
        return updated
    tv = .5*(updated.double()-base.double()).abs().sum(-1)
    return updated, {'selected_probability_mass':mass.squeeze(-1),
                     'normalizer':normalizer.squeeze(-1), 'TV':tv,
                     'TV_bound':rho*mass.squeeze(-1),
                     'probability_sum_error':(updated.double().sum(-1)-base.double().sum(-1)).abs()}

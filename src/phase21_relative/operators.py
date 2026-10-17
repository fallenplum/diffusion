"""Direct conditional logit update with the same selector and correction cap."""
import torch


def direct_update(base, gate, delta, mask, rho=.5, diagnostic=False):
    """a1=(1-rho)*a0+rho*normalize(a0*exp(g*(delta-mu_w))).

    mu_w=sum(a0*g*delta)/sum(a0*g). All original base candidates remain
    in the global normalizer, including unselected entries. This intentionally
    differs from the old preservation of each unselected probability. No score
    variance normalization, learned scale, new selector or new parameters.
    For small delta its first derivative equals the conservative operator;
    changing normalization alone cannot recover an unidentifiable row offset.
    """
    if base.shape != gate.shape or base.shape != delta.shape or mask.shape != base.shape:
        raise ValueError('base/gate/delta/mask shapes differ')
    if mask.dtype != torch.bool or not 0 <= rho <= 1:
        raise ValueError('invalid mask or rho')
    b=base.double(); weight=torch.where(mask,b*gate.double(),torch.zeros_like(b))
    mass=weight.sum(-1,keepdim=True)
    safe=torch.where(mass>0,mass,torch.ones_like(mass))
    mu=(weight*torch.where(mask,delta.double(),torch.zeros_like(b))).sum(-1,keepdim=True)/safe
    relative=torch.where(mask,delta.double()-mu,torch.zeros_like(b))
    shift=gate.double()*relative
    reweighted=b*shift.exp()
    total=reweighted.sum(-1,keepdim=True)
    base_total=b.sum(-1,keepdim=True)
    safe_base=torch.where(base_total>0,base_total,torch.ones_like(base_total))
    normalizer=torch.where(base_total>0,total/safe_base,torch.ones_like(total))
    # Stable identity at delta=0, without suppressing its derivative.
    change=rho*b*(shift.exp()/normalizer-1.)
    change=torch.where(mass>0,change,torch.zeros_like(change))
    result=base+change.to(base.dtype)
    if not diagnostic: return result
    tv=.5*(result.double()-b).abs().sum(-1)
    return result,{'selected_probability_mass':mass.squeeze(-1),
                   'normalizer':total.squeeze(-1), 'TV':tv,
                   'TV_bound':rho*base_total.squeeze(-1),
                   'probability_sum_error':(result.double().sum(-1)-base_total.squeeze(-1)).abs()}

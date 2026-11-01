"""FP64 Efron likelihood and observed-information geometry.

Information is the SUM Hessian; loss is normalized by event count.
Projection is trained on complete fit risk sets, never evaluation outcomes.
"""
from __future__ import annotations
import torch


class EfronRiskSets:
    def __init__(self, time, event, *, device=None):
        self.time=torch.as_tensor(time,dtype=torch.float64,device=device)
        ev=torch.as_tensor(event,device=device)
        if self.time.ndim!=1 or ev.shape!=self.time.shape or len(ev)==0:raise ValueError('ENDPOINT_SHAPE')
        if not torch.isfinite(self.time).all() or (self.time<0).any():raise ValueError('INVALID_TIME')
        if not ((ev==0)|(ev==1)).all() or not ev.any():raise ValueError('INVALID_OR_ZERO_EVENTS')
        self.event=ev.bool();self.n_events=int(self.event.sum())
        self.order=torch.argsort(self.time,descending=True,stable=True)
        ts=self.time[self.order];es=self.event[self.order]
        _,inverse,counts=torch.unique_consecutive(ts,return_inverse=True,return_counts=True)
        ends=counts.cumsum(0)-1
        d=torch.zeros(len(counts),device=ts.device,dtype=torch.float64)
        d.scatter_add_(0,inverse,es.double())
        group=torch.repeat_interleave(torch.arange(len(counts),device=ts.device),d.long())
        starts=torch.cumsum(d.long(),0)-d.long()
        step=torch.arange(self.n_events,device=ts.device)-starts[group]
        self.ends=ends[group];self.fraction=step.double()/d[group]
        self.event_group=inverse[es];self.subgroup=group
        self.groups=len(counts);self.sorted_event=es
        event_times=ts[self.ends]
        available=self.time[None,:]>=event_times[:,None]
        tied=self.event[None,:]&(self.time[None,:]==event_times[:,None])
        factor=torch.where(tied,1-self.fraction[:,None],torch.ones_like(self.fraction[:,None]))
        self.log_weights=torch.where(available,torch.log(factor),-torch.inf)

    def validate(self,risk):
        if risk.dtype!=torch.float64 or risk.shape!=self.time.shape:raise ValueError('FP64_FULL_RISK_VECTOR_REQUIRED')
        if not torch.isfinite(risk).all():raise ValueError('NONFINITE_RISK')

    def moments(self,risk,features=None):
        self.validate(risk)
        logits=risk[None,:]+self.log_weights
        logden=torch.logsumexp(logits,1)
        if not torch.isfinite(logden).all():raise FloatingPointError('EMPTY_RISK_SET')
        if features is None:return logden
        if features.dtype!=torch.float64 or features.ndim!=2 or len(features)!=len(risk):raise ValueError('FP64_FEATURE_SHAPE')
        if not torch.isfinite(features).all():raise ValueError('NONFINITE_FEATURES')
        probability=torch.softmax(logits,1);mu=probability@features
        info=features.T@(probability.sum(0)[:,None]*features)-mu.T@mu
        return (info+info.T)/2

    def loss(self,risk,*,normalize=True):
        value=self.moments(risk).sum()-risk[self.event].sum()
        return value/self.n_events if normalize else value

    def information(self,risk,features):return self.moments(risk,features)


def fit_projection(b,z,*,geometry,risk=None,risksets=None,rho=1e-4):
    if rho<0 or b.ndim!=2 or z.ndim!=2 or len(b)!=len(z):raise ValueError('INVALID_PROJECTION_ARGUMENTS')
    with torch.no_grad():
        bb,zz=b.detach().double(),z.detach().double();xz=torch.cat([bb,zz],1)
        if geometry=='euclidean':info=xz.T@xz
        elif geometry=='riskset':
            if risksets is None or risk is None:raise ValueError('TRAIN_RISKSETS_REQUIRED')
            info=risksets.information(risk.detach().double(),xz)
        else:raise ValueError('UNKNOWN_GEOMETRY')
        d=bb.shape[1];ibb=info[:d,:d];ibz=info[:d,d:]
        mat=ibb+rho*torch.eye(d,dtype=bb.dtype,device=bb.device)
        a=torch.linalg.solve(mat,ibz)
        if not torch.isfinite(a).all():raise FloatingPointError('PROJECTION_SOLVE_FAILED')
        cross=ibz-ibb@a
        conditional=info[d:,d:]-ibz.T@a-a.T@ibz+a.T@ibb@a
        result={'ridge':rho,'condition_number':float(torch.linalg.cond(mat)),
                'I_BB_eigenvalues':torch.linalg.eigvalsh(ibb).cpu().tolist(),
                'I_conditional_eigenvalues':torch.linalg.eigvalsh(conditional).cpu().tolist(),
                'cross_information_norm':float(cross.norm()),
                'ridge_identity_error':float((cross-rho*a).abs().max())}
    return a.detach(),result

"""Full-risk-set training. Evaluation receives no outcome arguments."""
from __future__ import annotations
import hashlib
import time as clock
import torch
from .geometry import EfronRiskSets,fit_projection

SNAPSHOTS=(0,10,30,60,90)


def state_hash(model):
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):
        h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def refresh(model,batch,risksets,wrong=None):
    if model.arm not in ('S2','S3','S3_WRONG_TRAIN'):return None,{}
    bs,z,hg=model.raw(batch,wrong=wrong)
    a,diag=fit_projection(bs,z,geometry='euclidean' if model.arm=='S2' else 'riskset',
                          risk=bs.sum(1),risksets=risksets,rho=1e-4)
    with torch.no_grad():model.A.copy_(a);model.A_ready.fill_(True)
    return a,diag


def fit(model,batch,time,event,*,epochs=90,b=.01,n=.0005,wrong=None):
    if epochs not in SNAPSHOTS:raise ValueError('UNREGISTERED_EPOCH_BUDGET')
    named=list(model.named_parameters());device=next(model.parameters()).device
    rs=EfronRiskSets(time,event,device=device)
    opt=torch.optim.Adam([{'params':[p for name,p in named if model.group(name)==group],'lr':lr}
                          for group,lr in [('base',1e-5),('generator',1e-4),('output',3e-4)]])
    history=[];snapshots={};counts={g:0 for g in ('base','generator','output')}
    started=clock.monotonic();geometry_updates=0
    def save(epoch):
        nonlocal geometry_updates
        with torch.no_grad():
            _,diag=refresh(model,batch,rs,wrong)
            if diag:geometry_updates+=1
            out=model(batch)
        snapshots[epoch]={'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                          'hash':state_hash(model),'geometry':diag,'counts':dict(counts),
                          'train_Cox':float(rs.loss(out['risk']))}
    save(0)
    for epoch in range(1,epochs+1):
        model.stage(epoch);model.train();opt.zero_grad(set_to_none=True)
        before={name:p.detach().clone() for name,p in named if p.requires_grad}
        a,diag=refresh(model,batch,rs,wrong)
        if diag:geometry_updates+=1
        hg=None
        if wrong is not None:hg=model.main(batch)[0]['g'][wrong]
        out=model(batch,a=a,interaction_g_hidden=hg)
        lc=rs.loss(out['risk']);pb,pn=model.penalties();loss=lc+b*pb+n*pn
        if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_OBJECTIVE')
        # Cox-only gradient, distinct from the regularization gradient. Keep
        # Z's path alive; geometry and B are detached inside projection.
        gz=torch.autograd.grad(lc,out['Z'],retain_graph=True,allow_unused=True)[0] if out['Z'].requires_grad else None
        probes={}
        if epoch in (1,10,11,30,60,61,90):
            active=[(name,p) for name,p in named if p.requires_grad]
            grads=torch.autograd.grad(lc,[p for _,p in active],retain_graph=True,allow_unused=True)
            for group in counts:
                values=[g.square().sum() for (name,_),g in zip(active,grads) if g is not None and model.group(name)==group]
                probes['Cox_gradient_'+group]=float(torch.stack(values).sum().sqrt()) if values else 0.
        loss.backward();clip=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.));opt.step()
        for group in counts:
            counts[group]+=int(any(not torch.equal(p.detach(),before[name]) for name,p in named if name in before and model.group(name)==group))
        history.append({'epoch':epoch,'Cox':float(lc.detach()),'objective':float(loss.detach()),
                        'grad_Z_Cox_norm':float(gz.norm()) if gz is not None else 0.,
                        'main_SD':float(out['main'].detach().std(unbiased=False)),
                        'interaction_SD':float(out['interaction'].detach().std(unbiased=False)),
                        'clip_input_norm':clip,'geometry':diag,**probes})
        if epoch in SNAPSHOTS:save(epoch)
    model.eval()
    return {'history':history,'snapshots':snapshots,'updates':counts,'geometry_evaluations':geometry_updates,
            'elapsed_seconds':clock.monotonic()-started,'optimizer_updates':epochs}


@torch.no_grad()
def predict(model,batch):
    model.eval()
    return {k:v.detach().cpu() for k,v in model(batch).items() if k not in ('hidden_g',)}

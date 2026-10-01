from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from src.phase_m2.common import atomic_json, cox_loss
from src.phase_m2.data import load_raw_inputs
from .survival import cox_loss_breslow_grouped

def manual_grouped(times, events, risks):
    out = 0.0; ne = int(sum(events))
    for t in sorted({float(t) for t,e in zip(times,events) if e}):
        d=[i for i,(tt,e) in enumerate(zip(times,events)) if e and tt==t]
        r=[i for i,tt in enumerate(times) if tt>=t]
        m=len(d); x=max(risks[i] for i in r); out += m*(x+np.log(sum(np.exp(risks[i]-x) for i in r)))-sum(risks[i] for i in d)
    return out/ne

def grad_finite(times, events, vals, eps=1e-6):
    g=[]
    for j in range(len(vals)):
        a=vals.copy(); b=vals.copy(); a[j]+=eps; b[j]-=eps
        la=cox_loss_breslow_grouped(torch.tensor(times,dtype=torch.float64),torch.tensor(events,dtype=torch.float64),torch.tensor(a,dtype=torch.float64)).item()
        lb=cox_loss_breslow_grouped(torch.tensor(times,dtype=torch.float64),torch.tensor(events,dtype=torch.float64),torch.tensor(b,dtype=torch.float64)).item()
        g.append((la-lb)/(2*eps))
    return np.asarray(g)

def run(args):
    root=Path(args.data_root); repo=Path(args.repo_root); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    raw=load_raw_inputs(root, repo); times=np.asarray(raw.dss_time); events=np.asarray(raw.dss_event).astype(int)
    event_times=times[events==1]; unique, counts=np.unique(event_times, return_counts=True); tied=counts[counts>1]
    tie={"n_patients":int(len(times)),"n_events":int(events.sum()),"n_unique_event_times":int(len(unique)),"n_tied_event_time_groups":int(len(tied)),"n_events_inside_tied_groups":int(tied.sum()),"fraction_events_inside_tied_groups":float(tied.sum()/events.sum()),"max_event_tie_multiplicity":int(tied.max() if len(tied) else 1)}
    atomic_json(out/"cox_tie_audit.json",tie)
    # no-tie loss and gradient equivalence
    nt=np.arange(1,9,dtype=float); ne=np.array([1,0,1,1,0,1,0,1],dtype=float); rv=np.array([.2,-.1,.4,.3,.0,-.5,.1,.7])
    r1=torch.tensor(rv,dtype=torch.float64,requires_grad=True); l1=cox_loss(torch.tensor(nt),torch.tensor(ne),r1); l1.backward(); g1=r1.grad.detach().numpy()
    r2=torch.tensor(rv,dtype=torch.float64,requires_grad=True); l2=cox_loss_breslow_grouped(torch.tensor(nt),torch.tensor(ne),r2); l2.backward(); g2=r2.grad.detach().numpy()
    cases=[]
    for name,t,e,r in [("two_events",[1,1,2,3],[1,1,0,0],[.2,.4,-.1,.3]),("three_events",[2,2,2,4,5],[1,1,1,0,0],[.2,.4,-.1,.3,.6]),("event_censor_same",[1,1,2,3],[1,0,1,0],[.2,.4,-.1,.3])]:
        loss=cox_loss_breslow_grouped(torch.tensor(t,dtype=torch.float64),torch.tensor(e,dtype=torch.float64),torch.tensor(r,dtype=torch.float64)); manual=manual_grouped(t,e,r); cases.append({"name":name,"loss":float(loss),"manual":float(manual),"abs_diff":float(abs(float(loss)-manual))})
    tt=np.array([1.,1.,2.,3.,3.]); ee=np.array([1,1,0,1,0]); vv=np.array([.2,-.4,.3,.7,-.1]); rt=torch.tensor(vv,dtype=torch.float64,requires_grad=True); lt=cox_loss_breslow_grouped(torch.tensor(tt),torch.tensor(ee),rt); lt.backward(); analytic=rt.grad.detach().numpy(); finite=grad_finite(tt,ee,vv); fd_err=float(np.max(np.abs(analytic-finite)))
    rng=np.random.default_rng(8102026); real_risk=rng.normal(size=len(times)); ra=torch.tensor(real_risk,dtype=torch.float64,requires_grad=True); old=cox_loss(torch.tensor(times,dtype=torch.float64),torch.tensor(events,dtype=torch.float64),ra); old.backward(); oldg=ra.grad.detach().numpy(); rb=torch.tensor(real_risk,dtype=torch.float64,requires_grad=True); new=cox_loss_breslow_grouped(torch.tensor(times,dtype=torch.float64),torch.tensor(events,dtype=torch.float64),rb); new.backward(); newg=rb.grad.detach().numpy()
    audit={"status":"PASS" if abs(float(l1-l2))<1e-8 and np.max(np.abs(g1-g2))<1e-7 and all(x["abs_diff"]<1e-8 for x in cases) and fd_err<2e-5 else "FAIL","no_tie_loss_abs_diff":float(abs(float(l1-l2)),),"no_tie_gradient_max_abs_diff":float(np.max(np.abs(g1-g2))),"manual_tied_cases":cases,"finite_difference_max_abs_error":fd_err,"real_endpoint":{"seed":8102026,"historical_loss":float(old),"grouped_breslow_loss":float(new),"loss_difference":float(new-old),"historical_gradient_norm":float(np.max(np.abs(oldg))),"grouped_gradient_norm":float(np.max(np.abs(newg))),"gradient_max_abs_difference":float(np.max(np.abs(oldg-newg)))},"thresholds":{"no_tie_loss_abs_diff_lt":1e-8,"no_tie_gradient_max_abs_diff_lt":1e-7,"manual_tie_loss_abs_diff_lt":1e-8,"finite_difference_max_abs_error_lt":2e-5},"implementation":"src.phase_m8a.survival.cox_loss_breslow_grouped"}
    atomic_json(out/"cox_grouped_breslow_audit.json",audit)
    if audit["status"]!="PASS": raise SystemExit("Cox grouped audit failed")
    print(json.dumps({"status":"PASS","tie":tie,"audit":audit},ensure_ascii=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--repo-root",required=True); p.add_argument("--data-root",required=True); p.add_argument("--output",required=True); run(p.parse_args())

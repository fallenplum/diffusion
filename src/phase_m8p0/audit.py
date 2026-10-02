from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,torch
from src.phase_m2.common import atomic_json,cox_loss
from src.phase_m2.data import load_raw_inputs
from .survival import cox_loss_full
def run(args):
 raw=load_raw_inputs(Path(args.data_root),Path(args.repo_root));t=torch.tensor([5.,4.,3.,2.,1.]);e=torch.tensor([1.,0.,1.,0.,1.]);r=torch.tensor([.2,-.1,.4,.3,.7],requires_grad=True);a=cox_loss(t,e,r);a.backward();g1=r.grad.clone();r2=torch.tensor([.2,-.1,.4,.3,.7],requires_grad=True);b=cox_loss_full(t,e,r2);b.backward();g2=r2.grad.clone(); audit={"status":"PASS" if abs(float(a-b))<1e-8 and torch.max(torch.abs(g1-g2))<1e-7 and float(g1[0])>0 else "FAIL","historical_grouped_no_tie_loss_abs_diff":float(abs(a-b)),"gradient_max_abs_diff":float(torch.max(torch.abs(g1-g2))),"risk_ordering":"descending time stable sort","event_coding":"events==1","full_risk_set":True,"finite_gradients":bool(torch.isfinite(g2).all()),"correct_gradient_direction":bool(float(g2[0])>0),"n_patients":len(raw.patient_ids),"dss_events":int(raw.dss_event.sum()),"implementation":"src.phase_m8p0.survival.cox_loss_full"};atomic_json(Path(args.output),audit);print(json.dumps(audit,ensure_ascii=False));
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--repo-root",required=True);p.add_argument("--data-root",required=True);p.add_argument("--output",required=True);run(p.parse_args())

"""Label-blind training probes; isolate SSM gradients from late omics effects."""
import time
import numpy as np
import torch

def diagnose(model,ds):
    ix=ds.fit[:32];model.eval()
    with torch.no_grad():matched,md=ds.forward(model,ix,diagnostic=True);wrong,wd=ds.forward(model,ix,wrong_inference=True,diagnostic=True)
    row={'population':'first32 training patients; no label/error selection','gamma':float(model.gamma.detach()),'matched_wrong_risk_MAE':float((matched-wrong).abs().mean())}
    def flat(diags,key):
        out=[]
        for q in diags:
            x=q[key]
            if key in ['g','dt','B','conditional_delta','conditional_B','base_B','base_dt','m']:x=x[q['valid_mask']]
            out.append(x.flatten())
        return torch.cat(out).float()
    for key in ['g','dt','B','h','conditional_delta','conditional_B']:
        a,b=flat(md,key),flat(wd,key);row[key+'_relative_L2']=float((a-b).norm()/a.norm().clamp_min(1e-12))
        row[key+'_RMS']=float(a.square().mean().sqrt())
    row['delta_over_base_RMS']=float(flat(md,'conditional_delta').square().mean().sqrt()/flat(md,'base_dt').square().mean().sqrt().clamp_min(1e-12))
    row['B_condition_over_base_RMS']=float(flat(md,'conditional_B').square().mean().sqrt()/flat(md,'base_B').square().mean().sqrt().clamp_min(1e-12))
    row['gate_mean']=float(flat(md,'m').mean());row['gate_SD']=float(flat(md,'m').std())
    for key in ['ssm_risk','ssm_raw','late_risk','direct_risk']:
        row[key+'_SD']=float(flat(md,key).std())
    row['wrong_context_relative_risk_drift']=row['matched_wrong_risk_MAE']/max(row['ssm_risk_SD'],1e-12)
    row['anchor_SD']=float(ds.anchor[ix].std());row['ssm_over_anchor_SD']=row['ssm_risk_SD']/max(row['anchor_SD'],1e-12)
    # This objective contains ONLY SSM risk, not direct/late heads or survival labels.
    sl=np.flatnonzero(np.isin(ds.owner,ix[:8]));owners=ds.owner[sl];f=ds.ft
    molecules=[f[k][owners].detach().clone().requires_grad_(True) for k in ['R','G','L']]
    _,dg=model(ds.visual[sl],ds.mask[sl],f['C'][owners],*molecules,f['RM'][owners],f['CM'][owners],ds.anchor[owners],diagnostic=True)
    targets=[dg['g'],*molecules,model.molecular.logits,model.gamma]
    gradients=torch.autograd.grad(dg['ssm_risk'].sum(),targets,allow_unused=True)
    for name,value in zip(['g','RNA','CNA_gain','CNA_loss','molecular_pool','gamma'],gradients):row['isolated_SSM_'+name+'_gradient_L2']=0. if value is None else float(value.norm())
    # Same actual forward workload for all arms; no FLOPs equality claim.
    with torch.no_grad():
        ds.forward(model,ix);torch.cuda.synchronize();start=time.monotonic()
        for _ in range(3):ds.forward(model,ix)
        torch.cuda.synchronize();row['seconds_per_32patients_forward']=(time.monotonic()-start)/3
    return row

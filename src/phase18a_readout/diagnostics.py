import numpy as np
import torch
from src.phase17r_ssd.diagnostics import diagnose as parent_diagnose


def diagnose(model,ds):
    # Preserve the parent's label-blind SSM-isolated gradient and risk probes.
    row=parent_diagnose(model,ds);ix=ds.fit[:32]
    with torch.no_grad():
        _,d=ds.forward(model,ix,diagnostic=True);_,w=ds.forward(model,ix,diagnostic=True,wrong_inference=True)
    def y(q):return q['y'] if 'y' in q else (q['h']*q['Cpool'][:,None,:]).sum(-1)
    a=torch.cat([y(q) for q in d]);b=torch.cat([y(q) for q in w])
    row.update(z_mean=float(a.mean()),z_variance=float(a.var(unbiased=False)),z_mean_channel_variance=float(a.var(0,unbiased=False).mean()),z_matched_wrong_relative_L2=float((a-b).norm()/a.norm().clamp_min(1e-12)),training_probe_patients=len(ix),training_probe_WSIs=sum(len(q['y']) if 'y' in q else len(q['h']) for q in d))
    owners=ds.owner[np.flatnonzero(np.isin(ds.owner,ix))]
    ssm=torch.cat([q['ssm_risk'] for q in d]);pooled_y=torch.stack([a[torch.as_tensor(owners==pi)].mean(0) for pi in ix]);patient_ssm=torch.stack([ssm[torch.as_tensor(owners==pi)].mean() for pi in ix])
    patient_sd=float(patient_ssm.std(unbiased=False));anchor_sd=float(ds.anchor[ix].std(unbiased=False))
    row.update(patient_z_variance=float(pooled_y.var(unbiased=False)),patient_z_mean_channel_variance=float(pooled_y.var(0,unbiased=False).mean()),patient_SSM_risk_SD=patient_sd,patient_anchor_SD=anchor_sd,patient_SSM_over_anchor_SD=patient_sd/max(anchor_sd,1e-12),legacy_SSM_SD_scope='WSI outputs, inherited parent diagnostic; patient-level counterpart separately reported',z_variance_scope='first32 training patients, equal-WSI patient aggregate also reported')
    # Independently named targets: h_T squared norm is not survival reliance.
    pi=int(ds.fit[0]);sl=np.flatnonzero(ds.owner==pi);f=ds.ft;owners=ds.owner[sl]
    x=ds.visual[sl].detach().clone().requires_grad_(True)
    risk,dg=model(x,ds.mask[sl],f['C'][owners],*[f[k][owners] for k in ['R','G','L','RM','CM']],ds.anchor[owners],diagnostic=True)
    targets={'terminal_state_squared_norm':dg['h'].square().sum()/len(sl),'final_patient_risk':risk.mean()}
    rows=[]
    for name,target in targets.items():
        grad=torch.autograd.grad(target,x,retain_graph=True)[0].float().norm(dim=-1)
        mass=[];tail=[];head=[];ent=[]
        for si in range(len(sl)):
            n=int(ds.mask[sl[si]].sum());g=grad[si,:n];den=g.sum().clamp_min(1e-30);p=g/den
            mass.append(g.sum());tail.append(g[-max(1,int(np.ceil(n*.1))):].sum());head.append(g[:max(1,int(np.ceil(n*.1)))].sum());ent.append(float(-(p*p.clamp_min(1e-30).log()).sum()/np.log(max(n,2))))
        total=torch.stack(mass).sum().clamp_min(1e-30)
        rows.append({'target':name,'patients':1,'WSIs':len(sl),'valid_patch_positions':int(ds.mask[sl].sum()),'tail10_gradient_L2_mass':float(torch.stack(tail).sum()/total),'first10_gradient_L2_mass':float(torch.stack(head).sum()/total),'gradient_L2_mass_sum':float(total),'normalized_position_entropy_mean':float(np.mean(ent)),'target_value':float(target.detach()),'scope':'first training patient, equal WSI patient risk; geometry order; no label/error sampling'})
    row['position_gradient_probes']=rows
    return row

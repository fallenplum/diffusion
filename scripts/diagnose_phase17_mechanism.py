"""Label-blind training-patient interventions after protocol freeze; no refit."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from src.phase17_ssd.data import P,R,cfg,write
from src.phase17_ssd.train import Data,seed_all
from src.phase17_ssd.model import ConditionalS6

def main():
    histogram=[];rows=[];gradrows=[];speed=[]
    for path in sorted((P/'outer').glob('A8_*.pt')):
        z=torch.load(path,map_location='cpu',weights_only=False);seed_all(z['seed']+100*z['fold']);ds=Data(z['fold'],'outer',512,'A8',z['seed']);model=ConditionalS6(ds.f['C'].shape[1],ds.f['membership'],'A8',cfg()['training']['dropout']).cuda();model.load_state_dict(z['state']);model.eval()
        # Patient selection based on row order only, no event/time or model error.
        ix=ds.fit[:32]
        with torch.no_grad():matched,md=ds.forward(model,ix,diagnostic=True);wrong,wd=ds.forward(model,ix,diagnostic=True,wrong_inference=True)
        token=np.concatenate([q['m'][q['valid_mask']].numpy().reshape(-1) for q in md]);hh,bins=np.histogram(token,np.linspace(0,1,101))
        for k,n in enumerate(hh):histogram.append({'arm':'A8','seed':z['seed'],'fold':z['fold'],'bin_left':bins[k],'bin_right':bins[k+1],'tokens':int(n),'population':'first32 outer-training patients;label blind'})
        row={'arm':'A8','seed':z['seed'],'fold':z['fold'],'gate_valid_mean':float(token.mean()),'gate_valid_std':float(token.std()),'gate_below_001':float(np.mean(token<.01)),'gate_above_099':float(np.mean(token>.99))}
        for key in ['g','dt','B','h','conditional_delta','conditional_B']:
            aa=[];bb=[]
            for a,b in zip(md,wd):
                av,bv=a[key],b[key]
                if key!='h':av=av[a['valid_mask']];bv=bv[a['valid_mask']]
                aa.append(av.flatten());bb.append(bv.flatten())
            av=torch.cat(aa);bv=torch.cat(bb);row[key+'_matched_wrong_relative_L2']=float((av-bv).norm()/av.norm().clamp_min(1e-12))
            row[key+'_mean']=float(av.mean());row[key+'_std']=float(av.std())
        row['matched_wrong_risk_mean_abs']=float((matched-wrong).abs().mean());rows.append(row)
        # Gradient of actual final recurrent state (not risk main effects/C pool).
        slide=int(np.flatnonzero(ds.owner==ix[0])[0]);pi=int(ds.owner[slide]);n=int(ds.mask[slide].sum());x=ds.visual[slide:slide+1].detach().clone().requires_grad_();f=ds.ft
        _,dg=model(x,ds.mask[slide:slide+1],f['C'][pi:pi+1],*[f[k][pi:pi+1] for k in ['R','G','L','RM','CM']],ds.anchor[pi:pi+1],diagnostic=True)
        gg=torch.autograd.grad(dg['h'].square().sum(),x)[0][0,:n].norm(dim=-1).detach().cpu().numpy();total=gg.sum()
        for b in range(10):
            lo=int(n*b/10);hi=int(n*(b+1)/10);gradrows.append({'arm':'A8','seed':z['seed'],'fold':z['fold'],'position_decile':b+1,'patch_count':hi-lo,'gradient_norm_sum':float(gg[lo:hi].sum()),'gradient_fraction':float(gg[lo:hi].sum()/total) if total>0 else 0.,'target':'squared norm of actual final S6 state;one label-blind training WSI'})
        # A genuinely skipped recurrent branch: identical direct/late head only.
        owners=torch.as_tensor(ds.owner[np.flatnonzero(np.isin(ds.owner,ix))],device='cuda');sl=np.flatnonzero(np.isin(ds.owner,ix));f=ds.ft
        def head_only():
            r,g,l,rm,cm=[f[k][owners] for k in ['R','G','L','RM','CM']];mt,mm=model.molecular(r,g,l,rm,cm);ct=model.clinical(f['C'][owners]).reshape(-1,6,64)
            late=torch.cat([ct.mean(1),(mt*mm[:,:,None]).sum(1)/mm.sum(1,keepdim=True).clamp_min(1)],-1)
            return ds.anchor[owners]+model.direct(torch.cat([r,g,l],-1)).squeeze(-1)+model.late(late).squeeze(-1)
        def full():return model(ds.visual[sl],ds.mask[sl],f['C'][owners],*[f[k][owners] for k in ['R','G','L','RM','CM']],ds.anchor[owners])
        with torch.no_grad():
            # Verify exact output against the risk-only ablation before timing.
            off=model(ds.visual[sl],ds.mask[sl],f['C'][owners],*[f[k][owners] for k in ['R','G','L','RM','CM']],ds.anchor[owners],module_off=True);err=float((off-head_only()).abs().max());assert err<1e-6
            for label,fn in [('full',full),('SSM_actual_compute_removed',head_only)]:
                fn();torch.cuda.synchronize();start=time.monotonic()
                for _ in range(10):fn()
                torch.cuda.synchronize();seconds=(time.monotonic()-start)/10
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:fn()
                flop=sum(v.flops for v in prof.key_averages())
                speed.append({'arm':'A8','seed':z['seed'],'fold':z['fold'],'mode':label,'patient_count':len(ix),'WSI_count':len(sl),'seconds_per_forward':seconds,'patients_per_second':len(ix)/seconds,'profiled_matmul_conv_FLOPs_lower_bound':int(flop),'head_skip_equivalence_error':err,'FLOPs_scope':'torch profiler supported operations only;S6 elementwise scans not counted'})
        print('MECHANISM_CHECK',path.stem,flush=True)
    pd.DataFrame(histogram).to_csv(R/'GATE_HISTOGRAM.csv',index=False);pd.DataFrame(rows).to_csv(R/'CONTEXT_STATE_AUDIT.csv',index=False);pd.DataFrame(gradrows).to_csv(R/'LONG_RANGE_STATE_GRADIENT.csv',index=False);pd.DataFrame(speed).to_csv(R/'COMPUTE_ABLATION.csv',index=False)
    write(R/'DETAILED_MECHANISM_AUDIT.json',{'status':'PASS','checkpoints':len(rows),'context_swap':'training partition only;own direct and late evidence unchanged','no_outcome_selection':True,'no_histological_ground_truth_claim':True})

if __name__=='__main__':main()

"""Private checkpoint replay and interventions, separate from metric evaluator."""
import sys,json,argparse,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
from src.phase17_ssd.data import P,R,cfg,source,write,parent
from src.phase17_ssd.train import Data,seed_all
from src.phase17_ssd.model import ConditionalS6
from src.phase17_ssd.ssm import reference_scan

def replay(p,paper=False,mechanism=True):
    z=torch.load(p,map_location='cpu',weights_only=False);seed_all(z['seed']+100*z['fold']);arm=z['arm'];ds=Data(z['fold'],'outer',512,'A2' if paper else arm,z['seed'])
    if paper:
        from phase17_paper_models import PaperModel
        model=PaperModel(arm,ds.f['C'].shape[1],ds.f['membership']).cuda()
    else:model=ConditionalS6(ds.f['C'].shape[1],ds.f['membership'],arm,cfg()['training']['dropout']).cuda()
    model.load_state_dict(z['state'],strict=True);model.eval();torch.cuda.synchronize();start=time.monotonic()
    def evaluate(**kw):
        with torch.no_grad():
            if not paper:return ds.forward(model,ds.valid,**kw)
            batch=128 if arm=='MGCT' else 32
            return torch.cat([ds.forward(model,ds.valid[k:k+batch]) for k in range(0,len(ds.valid),batch)])
    result=evaluate();error=float(abs(result.cpu().numpy()-z['risk']).max())
    if error>2e-5:raise ValueError(('replay mismatch',p.name,error))
    record={'arm':arm,'seed':z['seed'],'fold':z['fold'],'max_risk_error':error,'seconds':time.monotonic()-start}
    orderrows=[];mods=[];states=[]
    if mechanism and not paper:
        with torch.no_grad():baseline,diag=ds.forward(model,ds.valid,diagnostic=True)
        risk0=baseline.cpu().numpy()
        for ordering in ['reverse','random']:
            value=evaluate(order=ordering).cpu().numpy();diff=value-risk0
            orderrows.append({'arm':arm,'seed':z['seed'],'fold':z['fold'],'ordering':ordering,'mean_absolute_risk_drift':float(abs(diff).mean()),'max_absolute_risk_drift':float(abs(diff).max()),'risk_Spearman':float(spearmanr(value,risk0).statistic),'risk_RMSE':float(np.sqrt(np.mean(diff**2)))})
        # Identical frozen model, change only omics state context; own direct/late
        # molecular main effects remain matched to the actual patient.
        if arm in ['A3','A5','A6','A7','A8','A9','A10','A11_NO_S32','A11_NO_CLIN']:
            saved_wrong=ds.wrong.copy()
            if arm=='A10':ds.wrong=np.arange(512)
            with torch.no_grad():wrong,wd=ds.forward(model,ds.valid,wrong_inference=arm!='A10',diagnostic=True)
            ds.wrong=saved_wrong
            hh=torch.cat([q['h'] for q in diag]);hw=torch.cat([q['h'] for q in wd]);gg=torch.cat([q['g'].reshape(-1,64) for q in diag]);gw=torch.cat([q['g'].reshape(-1,64) for q in wd])
            mods.append({'arm':arm,'seed':z['seed'],'fold':z['fold'],'intervention':'frozen_restore_matched_context' if arm=='A10' else 'frozen_wrong_context','risk_abs_mean':float((wrong-baseline).abs().mean()),'state_relative_L2':float((hh-hw).norm()/hh.norm().clamp_min(1e-12)),'g_relative_L2':float((gg-gw).norm()/gg.norm().clamp_min(1e-12))})
        for tag,kwargs,hook_name in [('SSM_off',{'module_off':True},None),('gate_m1',{'gate_override':1},None),('gate_constant_09',{'gate_override':.9},None),('delta_off',{},'delta_up'),('B_condition_off',{},'B_up'),('CrossAttn_context_off',{},'attn_out'),('Omics_direct_off',{},'direct')]:
            hook=None
            if hook_name:hook=getattr(model,hook_name).register_forward_hook(lambda module,args,out:torch.zeros_like(out))
            torch.cuda.synchronize();s=time.monotonic()
            try:value=evaluate(**kwargs)
            finally:
                if hook is not None:hook.remove()
            torch.cuda.synchronize();elapsed=time.monotonic()-s
            mods.append({'arm':arm,'seed':z['seed'],'fold':z['fold'],'intervention':tag,'risk_abs_mean':float((value-baseline).abs().mean()),'risk_abs_max':float((value-baseline).abs().max()),'seconds':elapsed})
        # Label-blind first8 training WSIs: actual recurrence state differences.
        if arm in ['A2','A3','A7','A8']:
            with torch.no_grad():_,training_diag=ds.forward(model,ds.fit[:2],diagnostic=True)
        else:training_diag=[]
        totals={}
        for q in training_diag:
            for b in range(len(q['u'])):
                n=int(q['valid_mask'][b].sum());u=q['u'][b:b+1,:n].cuda();dt=q['dt'][b:b+1,:n].cuda();B=q['B'][b:b+1,:n].cuda();A=q['A'].cuda()
                h=reference_scan(u,dt,B,A,states=True).cpu();previous=torch.cat([torch.zeros_like(h[:,:1]),h[:,:-1]],1);change=(h-previous).flatten(2).norm(dim=-1).squeeze(0).numpy()
                for k,v in enumerate(change):totals.setdefault(k,[]).append(float(v))
        states=[{'arm':arm,'seed':z['seed'],'fold':z['fold'],'coordinate_position':k+1,'WSI_count':len(v),'mean_state_change_norm':float(np.mean(v))} for k,v in totals.items()]
        # At the selected checkpoint, differentiate SSM-only risk. This removes
        # direct/late main effects from the evidence for contextual gradients.
        slides=np.flatnonzero(np.isin(ds.owner,ds.valid[:2]));owners=ds.owner[slides];ctx=ds.wrong[owners] if arm=='A10' else owners
        cc=ds.ft['C'][owners].detach().clone().requires_grad_();co=[ds.ft[k][ctx].detach().clone().requires_grad_() for k in ['R','G','L']]
        model.zero_grad(set_to_none=True)
        _,dg=model(ds.visual[slides],ds.mask[slides],cc,*[ds.ft[k][owners] for k in ['R','G','L','RM','CM']],ds.anchor[owners],context_omics=(*co,ds.ft['RM'][ctx],ds.ft['CM'][ctx]),diagnostic=True)
        grads=torch.autograd.grad(dg['ssm_risk'].sum(),[cc,*co],allow_unused=True)
        mods.append({'arm':arm,'seed':z['seed'],'fold':z['fold'],'intervention':'selected_checkpoint_SSM_only_input_gradient',**{name+'_gradient_norm':0. if g is None else float(g.norm()) for name,g in zip(['Clinical','RNA_context','CNA_gain_context','CNA_loss_context'],grads)}})
    return record,orderrows,mods,states

def main():
    p=argparse.ArgumentParser();p.add_argument('--mechanism',action='store_true');a=p.parse_args();rows=[];order=[];mods=[];states=[]
    for paper,base in [(False,P/'outer'),(True,P/'paper_outer')]:
        for path in sorted(base.glob('*.pt')):
            out=replay(path,paper,a.mechanism);rows.append(out[0]);order.extend(out[1]);mods.extend(out[2]);states.extend(out[3]);print('REPLAY',path.stem,out[0]['max_risk_error'],flush=True)
    pd.DataFrame(rows).to_csv(R/'CHECKPOINT_REPLAY.csv',index=False)
    if a.mechanism:
        pd.DataFrame(order).to_csv(R/'ORDER_SENSITIVITY.csv',index=False);pd.DataFrame(mods).to_csv(R/'MODULE_INTERVENTIONS.csv',index=False);pd.DataFrame(states).to_csv(R/'STATE_CHANGE_CURVES.csv',index=False)
    write(R/'CHECKPOINT_REPLAY.json',{'status':'PASS','replayed':len(rows),'maximum_risk_error':max(v['max_risk_error'] for v in rows),'scope':'actual private saved model state reconstructed on private inputs;not just public report arithmetic'})

if __name__=='__main__':main()

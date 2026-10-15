"""Same deployed MIL capacity;fixed-budget fit-only selector then frozen-score head."""
import sys,argparse,time,gc,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.phase16_semantic.core import base,SemanticBags,write,sha,parent,R
from src.phase10_trimodal.math import torch_cox,log_weights
from src.phase13_rna.math import inner_score
LP=Path('/root/phase16_mil_private');LC=ROOT/'configs/phase16_mil_two_stage.json'
def lsource():return hashlib.sha256((base.source()+sha(LC)+sha(Path(__file__))).encode()).hexdigest()

def stageB(cs,bags,t,e,fit,test,w,score,lam,epochs,seed,select):
    base.seed_all(seed);model=base.RoutingModel(w).cuda();model.score.data.copy_(score);model.score.requires_grad_(False);optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001);x=torch.tensor(cs,dtype=torch.float32,device='cuda');a=log_weights(t[fit],e[fit],'cuda');ee=torch.tensor(e[fit],device='cuda');v=log_weights(t[test],e[test],'cuda') if select else None;ve=torch.tensor(e[test],device='cuda') if select else None;history=[];best=float('inf');selected=epochs;state=None;ema=None
    for epoch in range(1,epochs+1):
        optimizer.zero_grad();risk=model(x,bags);data=torch_cox(risk[fit],a,ee);loss=data+lam/2*(model.base@model.base)+.001/2*((model.roi@model.roi)+(model.score@model.score));loss.backward();gradient={k:0. if p.grad is None else float(p.grad.norm()) for k,p in model.named_parameters()};optimizer.step();row={'epoch':epoch,'train_cox':float(data.detach()),'penalized_loss':float(loss.detach()),'gradient':gradient}
        if select:
            with torch.no_grad():rr=model(x,bags)[test];vl=float(torch_cox(rr,v,ve));ema=vl if ema is None else .3*vl+.7*ema;valc=inner_score(t[test],e[test],rr.cpu().numpy())
            row.update(val_cox=vl,val_ema=ema,val_Harrell=valc)
            if ema<best:best=ema;selected=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row)
    if select:model.load_state_dict(state)
    return model,history,selected

def run(budget,seed,fold,pilot=False):
    started=time.time();name=f'MIL_TWO_STAGE_{budget}_seed{seed}_fold{fold}';cell=LP/'cells'/f'{name}.json';identity={'budget':budget,'seed':seed,'fold':fold,'source':lsource()}
    if not pilot and cell.exists():
        c=json.loads(cell.read_text())
        if c['tag']!=identity or sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('stale MIL ablation')
        return
    d=parent.load();ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];states={};selected=80
    for role in ['inner'] if pilot else ['inner','outer']:
        a,ap=base.anchor(seed,fold,role);fit,test=a['fit'],a['test'];cs,ca=base.cs_features(d,fit);bagsA=base.Bags('MIL',budget,seed);selector,traceA,_=base.train(cs,bagsA,t,e,fit,test,a['w'],a['lambda'],2 if pilot else 80,seed+fold*100,False)
        with torch.no_grad():_,roi=bagsA.pool(selector.score,True);selector_risk=selector(torch.tensor(cs,dtype=torch.float32,device='cuda'),bagsA).cpu().numpy()
        selected_matrix=np.full((len(base.manifest()['slides']),max(len(v) for v in roi.values())),-1,np.int64)
        for i,ix in roi.items():selected_matrix[i,:len(ix)]=ix
        roi_path=LP/'rois'/f'{name}_{role}.npy';roi_path.parent.mkdir(parents=True,exist_ok=True);np.save(roi_path,selected_matrix);stateA={k:v.detach().cpu().clone() for k,v in selector.state_dict().items()};score=selector.score.detach().clone();del selector,bagsA;gc.collect();torch.cuda.empty_cache();bags=SemanticBags(roi_path,budget);model,trace,chosen=stageB(cs,bags,t,e,fit,test,a['w'],score,a['lambda'],2 if pilot else 80 if role=='inner' else selected,seed+fold*100,role=='inner')
        if role=='inner':selected=chosen
        states[role]={'fit':fit,'test':test,'CS_preprocessing':ca,'lambda_CS':a['lambda'],'StageA_state':stateA,'StageA_all_risk':selector_risk,'StageA_history':traceA,'StageB_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'StageB_history':trace,'ROI_path':str(roi_path),'ROI_sha256':sha(roi_path)}
        if pilot:
            if traceA[-1]['gradient']['score']<=0 or trace[-1]['gradient']['roi']<=0 or trace[-1]['gradient']['score']!=0:raise ValueError('two-stage gradient/freeze error')
            write(LP/'pilot'/f'{name}.json',{'status':'PASS','tag':identity,'outer_predictions':False,'StageA_attention_gradient':traceA[-1]['gradient']['score'],'StageB_attention_gradient':trace[-1]['gradient']['score'],'StageB_head_gradient':trace[-1]['gradient']['roi']});return
        if role=='outer':
            with torch.no_grad():risk=model(torch.tensor(cs,dtype=torch.float32,device='cuda'),bags)[test].cpu().numpy()
            parameters=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        del bags,model;gc.collect();torch.cuda.empty_cache()
    path=LP/'artifacts'/f'{name}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save({'tag':identity,'states':states,'risk':risk,'test_indices':test,'time':t[test],'event':e[test],'parameters':parameters,'StageB_trainable_parameters':trainable,'selected_epoch':selected,'extra_selector_epochs_per_role':80},path);write(cell,{'status':'COMPLETE','tag':identity,'artifact':str(path),'artifact_sha256':sha(path),'seconds':time.time()-started,'extra_StageA_training_fits':2,'outer_test_selection':False});print('TWO_STAGE_COMPLETE',name,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--budget',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();run(a.budget,a.seed,a.fold,a.pilot)

"""Matched Phase16 semantic ROI outer folds; unchanged80-epoch training protocol."""
import sys,argparse,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.phase16_semantic.core import *

def run(arm,seed,fold,pilot=False):
    started=time.time();name=label(arm,seed,fold);cell=P/'cells'/f'{name}.json'
    if not pilot and cell.exists():
        c=json.loads(cell.read_text())
        if c['tag']!=tag(arm,seed,fold) or sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('stale/corrupt semantic cell')
        return
    if arm not in arms():raise ValueError('unknown semantic arm')
    if json.loads((R/'SPACE_COMPATIBILITY.json').read_text())['status']!='PASS_OFFICIAL_INTERFACE':raise ValueError('space gate missing')
    if not pilot:
        g=json.loads((R/'HARD_GATES.json').read_text())
        if g['status']!='PASS' or g['source']!=source():raise ValueError('semantic hard/pilot gate missing')
    base.seed_all(seed);d=parent.load();ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];selector,mm=arm.rsplit('_',1);budget=int(mm);states={};history={};selected=80;retrieval={}
    for role in ['inner'] if pilot else ['inner','outer']:
        a,ap=base.anchor(seed,fold,role);fit,test=a['fit'],a['test'];path=directory(seed,fold,role);done=json.loads((path/'DONE.json').read_text());expected=done['metadata'][selector]['files'][str(budget)];indices=path/f'{selector}_{budget}.npy'
        if done['source']!=source() or sha(indices)!=expected or not np.array_equal(done['fit'],fit):raise ValueError('retrieval source/partition/hash mismatch')
        bags=SemanticBags(indices,budget);cs,ca=base.cs_features(d,fit);model,trace,best=base.train(cs,bags,t,e,fit,test,a['w'],a['lambda'],2 if pilot else 80 if role=='inner' else selected,seed+fold*100,role=='inner');history[role]=trace
        if role=='inner':selected=best
        states[role]={'fit':fit,'test':test,'preprocessing':ca,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'lambda_CS':a['lambda'],'anchor_sha256':sha(ap),'ROI_path':str(indices),'ROI_sha256':expected,'prompt_preprocessing_path':str(path/'preprocessing.pt'),'prompt_preprocessing_sha256':done['preprocessing_sha256']}
        if pilot:
            xx=torch.tensor(cs,dtype=torch.float32,device='cuda',requires_grad=True);risk=model(xx,bags);risk[fit].sum().backward();sg=float(xx.grad[fit,-32:].norm())
            if sg<=0 or trace[-1]['gradient']['score']<=0 or trace[-1]['gradient']['roi']<=0:raise ValueError('dead S32/ROI/attention path')
            write(P/'pilot'/f'{name}.json',{'status':'PASS','arm':arm,'seed':seed,'fold':fold,'S32_gradient':sg,'attention_gradient':trace[-1]['gradient']['score'],'ROI_gradient':trace[-1]['gradient']['roi'],'outer_test_prediction':False,'source':source(),'peak_GPU_MiB':torch.cuda.max_memory_allocated()/1024**2,'seconds':time.time()-started});return
        if role=='outer':
            with torch.no_grad():risk=model(torch.tensor(cs,dtype=torch.float32,device='cuda'),bags)[test].cpu().numpy();base_only=(torch.tensor(cs[test],dtype=torch.float32,device='cuda')@model.base).cpu().numpy();_,rows=bags.pool(model.score,True)
        del bags;gc.collect();torch.cuda.empty_cache()
    artifact=P/'artifacts'/f'{name}.pt';artifact.parent.mkdir(parents=True,exist_ok=True);z={'tag':tag(arm,seed,fold),'states':states,'history':history,'risk':risk,'base_only_risk':base_only,'test_indices':test,'time':t[test],'event':e[test],'selected_epoch':selected,'selected_rows':rows,'parameters':sum(p.numel() for p in model.parameters()),'peak_GPU_MiB':torch.cuda.max_memory_allocated()/1024**2};torch.save(z,artifact);write(cell,{'status':'COMPLETE','tag':z['tag'],'artifact':str(artifact),'artifact_sha256':sha(artifact),'seconds':time.time()-started,'selected_epoch':selected,'parameters':z['parameters'],'outer_test_selection':False,'candidate_pool_limited':True});print('COMPLETE',name,flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--fold',type=int,required=True);ap.add_argument('--pilot',action='store_true');a=ap.parse_args();run(a.arm,a.seed,a.fold,a.pilot)

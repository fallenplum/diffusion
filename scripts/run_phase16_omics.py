"""Fixed common-gene G5 inputs, linear molecular head, no further model search."""
import sys,time,argparse,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.phase16_semantic.omics import *
from src.phase13_rna.math import fit_ridge,inner_score

def feature_cache(d,fold,role,kind,random_mask):
    part=parent.splits(d,'DSS',fold);fit=part['inner_train' if role=='inner' else 'outer_train'];p=MP/'features'/f'fold{fold}_{role}_{kind}_{int(random_mask)}.pt'
    import fcntl,os
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if p.exists():
            z=torch.load(p,map_location='cpu',weights_only=False)
            if z['source']!=msource() or not np.array_equal(z['state']['fit'],fit):raise ValueError('stale molecular features')
        else:
            x,st=inputs(d,fit,fold,kind,random_mask);z={'source':msource(),'x':x,'state':st};temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
            with temp.open('wb') as handle:torch.save(z,handle)
            temp.replace(p)
    return z['x'],z['state'],p

def run(arm,seed,fold,pilot=False):
    started=time.time();name=label(arm,seed,fold);cell=MP/'cells'/f'{name}.json';spec=armdefs()[arm];identity={'arm':arm,'seed':seed,'fold':fold,'source':msource(),'semantic_source':source()}
    if not pilot and cell.exists():
        c=json.loads(cell.read_text())
        if c['tag']!=identity or sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('stale G5 cell')
        return
    accepted=json.loads((P/'EVALUATION_DONE.json').read_text())
    if accepted['status']!='PASS' or accepted['source']!=source():raise ValueError('G2 independent gate missing')
    if not pilot:
        g=json.loads((R/'G5_PILOT_GATE.json').read_text())
        if g['status']!='PASS' or g['source']!=msource():raise ValueError('G5 pilot missing')
    base.seed_all(seed);d=parent.load();ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];part=parent.splits(d,'DSS',fold);states={};histories={};selected=80;selection=[]
    for role in ['inner'] if pilot else ['inner','outer']:
        a,ap=base.anchor(seed,fold,role);fit,test=a['fit'],a['test'];x,preprocess,fp=feature_cache(d,fold,role,spec['kind'],spec['random']);states[role]={'fit':fit,'test':test,'feature_path':str(fp),'feature_sha256':sha(fp)}
        if spec['selector'] is None:
            grid=[.1,1.,10.,'OFF'] if role=='inner' else [chosen]
            for lam in grid:
                if lam=='OFF':w=np.zeros(x.shape[1]);info={'OFF':True}
                else:w,info=fit_ridge(x[fit],t[fit],e[fit],lam)
                if role=='inner':selection.append({'lambda':lam,'score':inner_score(t[test],e[test],x[test]@w),'w':w,'optimizer':info})
            if role=='inner':
                best=max(r['score'] for r in selection);tie=[r for r in selection if abs(r['score']-best)<1e-12];chosen=next((r['lambda'] for r in tie if r['lambda']=='OFF'),tie[0]['lambda'])
            states[role].update(w=w,lambda_value=chosen,optimizer=info);risk=x[test]@w;parameters=len(w)
            if pilot:
                write(MP/'pilot'/f'{name}.json',{'status':'PASS','tag':identity,'outer_predictions':False,'kind':'omics-only ridge','optimizer':info});return
        else:
            roi=directory(seed,fold,role)/f'{spec["selector"]}_{spec["budget"]}.npy';bags=SemanticBags(roi,spec['budget']);cs,ca=base.cs_features(d,fit);model,trace,best=train_model(cs,bags,x,t,e,fit,test,a['w'],a['lambda'],2 if pilot else 80 if role=='inner' else selected,seed+fold*100,role=='inner');histories[role]=trace
            if role=='inner':selected=best
            states[role].update(model={k:v.detach().cpu() for k,v in model.state_dict().items()},CS_preprocessing=ca,lambda_CS=a['lambda'],ROI_path=str(roi),ROI_sha256=sha(roi))
            if pilot:
                grad=trace[-1]['gradient']
                if any(grad[k]<=0 for k in ['base','score','roi','mol']):raise ValueError('G5 dead gradient')
                write(MP/'pilot'/f'{name}.json',{'status':'PASS','tag':identity,'outer_predictions':False,'gradients':grad,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/1024**2});return
            if role=='outer':
                with torch.no_grad():risk=model(torch.tensor(cs,dtype=torch.float32,device='cuda'),bags,torch.tensor(x,dtype=torch.float32,device='cuda'))[test].cpu().numpy()
                parameters=sum(p.numel() for p in model.parameters())
            del bags;gc.collect();torch.cuda.empty_cache()
    path=MP/'artifacts'/f'{name}.pt';path.parent.mkdir(parents=True,exist_ok=True);torch.save({'tag':identity,'states':states,'history':histories,'selection':selection,'risk':risk,'test_indices':test,'time':t[test],'event':e[test],'parameters':parameters,'selected_epoch':selected if spec['selector'] is not None else 0,'spec':spec},path);write(cell,{'status':'COMPLETE','tag':identity,'artifact':str(path),'artifact_sha256':sha(path),'outer_test_selection':False,'seconds':time.time()-started,'parameters':parameters});print('G5_COMPLETE',name,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();run(a.arm,a.seed,a.fold,a.pilot)

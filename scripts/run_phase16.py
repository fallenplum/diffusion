"""Locked Phase16 fits. Pilot never emits outer-test predictions."""
import argparse,sys,time,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from sklearn.decomposition import PCA
from src.phase16_routing.core import *
from src.phase13_rna.math import block_fit,inner_score
from src.phase10_trimodal.features import scale

def tag(arm,seed,fold):return {'arm':arm,'seed':seed,'fold':fold,'source':source(),'candidates':manifest()['source_sha256']}
def global_input(d,fit,kind):
    raw=np.load(P/'conch_global.npy') if kind=='CS_CONCH_GLOBAL' else np.asarray(d['titan'],float)
    v,state=scale(raw,fit);p=PCA(cfg()['global_pca'],svd_solver='full').fit(v[fit]);pc=p.transform(v);pc,pcstate=scale(pc,fit)
    return pc,{'fit':fit,'scale':state,'components':p.components_,'pca_mean':p.mean_,'pca_scale':pcstate}
def global_fit(d,seed,fold,kind):
    parts=parent.splits(d,'DSS',fold);ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];states={};selection=[];inner_models={}
    for role in ['inner','outer']:
        a,path=anchor(seed,fold,role);fit=a['fit'];test=a['test'];cs,ca=cs_features(d,fit);pc,pa=global_input(d,fit,kind);design=np.c_[cs,pc];candidates=cfg()['global_grid'] if role=='inner' else [winner]
        for lam in candidates:
            if lam=='OFF':w=np.r_[a['w'],np.zeros(pc.shape[1])];diagnostic={'off':True}
            else:w,diagnostic=block_fit(design[fit],t[fit],e[fit],np.r_[np.full(cs.shape[1],a['lambda']),np.full(pc.shape[1],lam)])
            if role=='inner':
                selection.append({'lambda':lam,'score':inner_score(t[test],e[test],design[test]@w)});inner_models[str(lam)]={'w':w,'risk':design[test]@w,'optimizer':diagnostic}
        if role=='inner':
            best=max(r['score'] for r in selection);ties=[r for r in selection if abs(r['score']-best)<1e-12];winner=next((r['lambda'] for r in ties if r['lambda']=='OFF'),ties[0]['lambda'])
        states[role]={'fit':fit,'test':test,'clinical_preprocess':ca,'global_preprocess':pa,'lambda_CS':a['lambda'],'anchor_path':str(path),'w':w,'optimizer':diagnostic}
    risk=design[test]@w
    return {'states':states,'selection':selection,'inner_models':inner_models,'chosen_lambda':winner,'risk':risk,'parameters':len(w),'selected_epoch':0}
def reference(d,arm,seed,fold):
    te=parent.splits(d,'DSS',fold)['outer_test']
    if arm=='CS_P15':a,path=anchor(seed,fold,'outer');risk=a['test_risk'];parameters=len(a['w'])
    else:
        if arm=='C4_LEGACY':path=Path('/root/phase12_c4_private/formal/oof')/f'R0_128_seed{seed}_fold{fold}.npz'
        else:path=Path('/root/phase13_rna_private/formal/oof')/f'DSS_{"C" if arm=="C_LEGACY" else "CS"}_seed{seed}_fold{fold}.npz'
        z=np.load(path)
        if not np.array_equal(z['test_indices'],te):raise ValueError('reference patient order mismatch')
        risk=z['risk'];parameters=None
    return {'risk':risk,'reference_path':str(path),'reference_sha256':sha(path),'actual_fit':False,'parameters':parameters,'selected_epoch':0,'scope':'C4 uses128 per slide,not equal patient-area;historical reference only' if arm=='C4_LEGACY' else 'identical endpoint/folds;frozen reference'}
def neural(d,arm,seed,fold,pilot=False):
    selector,mm=arm.rsplit('_',1);budget=int(mm);bags=Bags(selector,budget,seed);ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];states={};logs={};selected=cfg()['training']['epochs']
    for role in ['inner'] if pilot else ['inner','outer']:
        a,path=anchor(seed,fold,role);fit=a['fit'];valid=a['test'];cs,audit=cs_features(d,fit);epochs=2 if pilot else (cfg()['training']['epochs'] if role=='inner' else selected)
        model,history,best=train(cs,bags,t,e,fit,valid,a['w'],a['lambda'],epochs,seed+fold*100,role=='inner');logs[role]=history
        if role=='inner':selected=best
        states[role]={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'fit':fit,'test':valid,'preprocess':audit,'lambda_CS':a['lambda'],'anchor_path':str(path),'epochs':epochs,'selected_epoch':best}
    if pilot:
        # Verify S32 participates numerically in forward; no outer prediction.
        xx=torch.tensor(cs,dtype=torch.float32,device='cuda',requires_grad=True);r=model(xx,bags);r[fit].sum().backward();sgrad=float(xx.grad[fit,-32:].norm());sg=history[-1]['gradient']['score'];rg=history[-1]['gradient']['roi']
        if sgrad<=0 or sg<=0 or rg<=0:raise ValueError('dead S32/attention/ROI gradient')
        result={'status':'PASS','arm':arm,'seed':seed,'fold':fold,'S32_gradient':sgrad,'attention_gradient':sg,'ROI_gradient':rg,'outer_test_prediction_performed':False,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/1024**2}
    else:
        with torch.no_grad():risk=model(torch.tensor(cs,dtype=torch.float32,device='cuda'),bags)[valid].cpu().numpy();_,rows=bags.pool(model.score,True);base_only=(torch.tensor(cs[valid],dtype=torch.float32,device='cuda')@model.base).cpu().numpy()
        owner=np.load(P/'row_owners.npy',mmap_mode='r');m=manifest()
        for i,ids in rows.items():check_rows(ids,owner,m['slides'][i]['patient'])
        correction=cfg()['training']['residual_scale']*torch.tanh(bags.pool(model.score)@model.roi)
        result={'states':states,'training':logs,'risk':risk,'base_only_risk':base_only,'selected_epoch':selected,'parameters':sum(p.numel() for p in model.parameters()),'selected_rows':rows,'peak_GPU_MiB':torch.cuda.max_memory_allocated()/1024**2,'correction_abs_max':float(correction.detach().abs().max()),'extra_selector_training_fits':0,'MIL_two_stage_ablation':'NOT_PERFORMED_INTEGRATED_ONLY'}
    del bags,model;gc.collect();torch.cuda.empty_cache();return result
def run(arm,seed,fold,pilot):
    started=time.time();seed_all(seed);name=f'{arm}_seed{seed}_fold{fold}';target=P/'cells'/f'{name}.json'
    if not pilot and target.exists():
        c=json.loads(target.read_text());check_tag(c['tag'],arm,seed,fold)
        if sha(c['artifact'])!=c['artifact_sha256']:raise ValueError('artifact corruption')
        return
    d=parent.load();gate=json.loads((R/'G0_DECISION.json').read_text())
    if gate['nonsemantic_pool']!='PASS':raise ValueError('candidate G0 failed')
    if arm not in arm_names():raise ValueError('arm blocked or unknown')
    if not pilot:
        pg=json.loads((R/'PILOT_GATE.json').read_text())
        if pg['status']!='PASS' or pg['source_hash']!=source():raise ValueError('pilot gate missing/stale')
        hg=json.loads((R/'HARD_GATES.json').read_text())
        if hg['status']!='PASS' or hg['source_hash']!=source():raise ValueError('hard gate missing/stale')
    if arm.startswith(('RAND_','GRID_','DIVERSE_','MIL_')):z=neural(d,arm,seed,fold,pilot)
    elif arm in ['CS_CONCH_GLOBAL','CS_TITAN_NATIVE']:
        if pilot:
            # Inner-only path for global numeric pilot: never fit/evaluate outer.
            a,path=anchor(seed,fold,'inner');cs,_=cs_features(d,a['fit']);pc,_=global_input(d,a['fit'],arm);ev=parent.endpoints(d)['DSS'];w,info=block_fit(np.c_[cs,pc][a['fit']],ev['time'][a['fit']],ev['event'][a['fit']],np.r_[np.full(cs.shape[1],a['lambda']),np.ones(pc.shape[1])]);z={'status':'PASS','arm':arm,'seed':seed,'fold':fold,'outer_test_prediction_performed':False,'gradient_max':info['gradient_max']}
        else:z=global_fit(d,seed,fold,arm)
    else:z=reference(d,arm,seed,fold)
    if pilot:write(P/'pilot'/f'{name}.json',{**z,'seconds':time.time()-started,'source_hash':source()});print('PILOT_PASS',name,flush=True);return
    part=parent.splits(d,'DSS',fold);ev=parent.endpoints(d)['DSS'];te=part['outer_test'];z.update(tag=tag(arm,seed,fold),partitions=part,test_indices=te,test_time=ev['time'][te],test_event=ev['event'][te],actual_fit=arm not in ['C_LEGACY','CS_LEGACY','CS_P15','C4_LEGACY'])
    artifact=P/'artifacts'/f'{name}.pt';artifact.parent.mkdir(parents=True,exist_ok=True);torch.save(z,artifact)
    write(target,{'status':'COMPLETE','tag':z['tag'],'artifact':str(artifact),'artifact_sha256':sha(artifact),'actual_fit':z['actual_fit'],'parameters':z['parameters'],'selected_epoch':z['selected_epoch'],'seconds':time.time()-started,'outer_test_selection':False,'training_scope':'DSS only;PFI frozen transfer'})
    print('COMPLETE',name,'seconds',round(time.time()-started,2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',required=True);ap.add_argument('--seed',required=True,type=int);ap.add_argument('--fold',required=True,type=int);ap.add_argument('--pilot',action='store_true');a=ap.parse_args();run(a.arm,a.seed,a.fold,a.pilot)

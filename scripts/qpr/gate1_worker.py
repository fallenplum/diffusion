"""One locked seed: one train-only global KMeans plus A3/A4 neural fits."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.cluster import KMeans
from src.qpr.protocol import require_stage
from src.qpr.graph import native_edges
from src.qpr.kv import relation_features,project_edges
from src.qpr.query import MolecularQuery
from src.qpr.compressor import SparseCompressor,summary,answer,losses


def synthetic(seed,pca):
    rng=np.random.default_rng(seed);graphs=[];idx=1000
    for kind in ('A','B','C'):
        for split,count in [('train',16),('validation',4),('test',4)]:
            for j in range(count):
                side=8 if (j//2)%2==0 else 16;xy=np.array([(x,y) for x in range(side) for y in range(side)],dtype=np.int64)
                values=(np.arange(side*side)>=side*side//2).astype(float)
                h=(values[:,None]-.5)*rng.normal(0,.04,(1,32))+rng.normal(0,.002,(len(xy),32))
                if kind=='A':
                    pair=np.random.default_rng(seed*10000+{'train':0,'validation':100,'test':200}[split]+j//2)
                    base=(values[:,None]-.5)*pair.normal(0,.04,(1,32))+pair.normal(0,.002,(len(xy),32))
                    if j%2:
                        checker=xy.sum(1)%2;h=np.empty_like(base)
                        h[checker==0]=base[:len(base)//2];h[checker==1]=base[len(base)//2:]
                    else:h=base
                if kind=='B':h=rng.normal(0,.05,(len(xy),32))
                _,e,_=native_edges(xy,1);rel=relation_features(h,xy,e,1);k,v=project_edges(rel,pca)
                if kind=='C':k[:]=k[0];v[:]=v[0]
                graphs.append({'owner':idx,'slide_index':idx,'kind':kind,'split':split,'k':k,'v':v,'ends':e,'nodes':len(xy),'domain':'synthetic'});idx+=1
    return graphs


def metric(rows,domain,kind=None):
    chosen=[r for r in rows if r['domain']==domain and (kind is None or r['kind']==kind)]
    owners=sorted({r['owner'] for r in chosen});one=[]
    for o in owners:
        a=[r for r in chosen if r['owner']==o]
        one.append({k:float(np.mean([x[k] for x in a])) for k in ('true_mse','probe_mse','true_logZ','probe_logZ','relative_L2')})
    if not one:raise ValueError('EMPTY_DOMAIN_EVALUATION')
    return {'patients_or_synthetic_graphs':len(owners),'RMSE_true':math.sqrt(np.mean([x['true_mse'] for x in one])),
            'RMSE_probe':math.sqrt(np.mean([x['probe_mse'] for x in one])),
            'logZ_MAE_true':float(np.mean([x['true_logZ'] for x in one])),
            'logZ_MAE_probe':float(np.mean([x['probe_logZ'] for x in one])),
            'relative_L2_train_denominator':float(np.mean([x['relative_L2'] for x in one])),
            'patient_MSE_p50_p90_p95':np.quantile([.5*(x['true_mse']+x['probe_mse']) for x in one],[.5,.9,.95]).tolist()}


def main(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    lock=json.loads(Path(args.lock).read_text());require_stage('gate1',args.reports)
    if lock['status']!='LOCKED_BEFORE_FORMAL_FITS':raise ValueError('UNLOCKED_FORMAL_RUN')
    for name,h in lock['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('SOURCE_CHANGED:'+name)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);target=out/f'seed{args.seed}.json'
    if target.exists():raise ValueError('EXISTING_COMPLETE_SEED')
    started=time.monotonic()
    cache_receipt=json.loads(Path(args.cache_receipt).read_text())
    with Path(args.cache).open('rb') as fh:
        digest=hashlib.sha256()
        for block in iter(lambda:fh.read(8*1024**2),b''):digest.update(block)
    if digest.hexdigest()!=cache_receipt['cache_sha256']:raise ValueError('OUTCOME_FREE_CACHE_IDENTITY_CHANGED')
    if cache_receipt['DSS_file_opened'] or cache_receipt['endpoint_elements_decoded']!=0:raise ValueError('CACHE_ENDPOINT_CONTAMINATION')
    with Path(args.cache).open('rb') as f:data=pickle.load(f)
    query=MolecularQuery(data['clinical'].shape[1]).cuda()
    with torch.no_grad():
        qs=query(torch.tensor(data['program'],device='cuda'),torch.tensor(data['clinical'],device='cuda'),torch.tensor(data['s32'],device='cuda'))
    graphs=[dict(x,domain='real',kind='real') for x in data['graphs']]+synthetic(args.seed,data['pca_edge'])
    bank=[]
    for graph in graphs:
        idx=graph['slide_index'];owner=graph['owner'];query_index=idx-idx%2 if graph['kind']=='A' else idx
        rng=np.random.default_rng(args.seed*100000+query_index)
        probe=rng.normal(size=(206,32));probe=probe/np.linalg.norm(probe,axis=1,keepdims=True)*rng.uniform(size=(206,1))**(1/32)
        realq=qs[owner] if graph['domain']=='real' else torch.tensor(probe[:50],dtype=torch.float32,device='cuda')
        q=torch.cat([realq,torch.tensor(probe,dtype=torch.float32,device='cuda')],0).detach()
        qmask=np.r_[data['program_mask'][owner] if graph['domain']=='real' else np.ones(50,bool),np.ones(206,bool)]
        g={**graph,'k':torch.tensor(graph['k'],device='cuda'),'v':torch.tensor(graph['v'],device='cuda'),
           'ends':torch.tensor(graph['ends'],device='cuda'),'q':q,'qmask':torch.tensor(qmask,device='cuda')}
        with torch.no_grad():g['teacher']=tuple(x.detach() for x in answer(q,g['k'],g['v']))
        bank.append(g)
    fit=[g for g in bank if g['split']=='train'];valid=[g for g in bank if g['split']=='validation'];test=[g for g in bank if g['split']=='test']
    normalizer=float(torch.stack([g['teacher'][0].square().mean() for g in fit]).mean().sqrt().clamp_min(1e-8))
    pool=np.concatenate([np.c_[g['k'].cpu().numpy(),g['v'].cpu().numpy()][np.linspace(0,len(g['k'])-1,min(64,len(g['k']))).astype(int)] for g in fit])
    km=KMeans(32,random_state=args.seed,n_init=10,max_iter=300).fit(pool)
    centers=torch.tensor(km.cluster_centers_,device='cuda')
    def assignment(g,arm,model=None):
        if arm=='A1':
            rng=np.random.default_rng(args.seed+g['slide_index']*101);labels=torch.tensor(rng.integers(32,size=len(g['k'])),device='cuda')
            return torch.nn.functional.one_hot(labels,32).float()
        if arm=='A2':
            labels=torch.cdist(torch.cat([g['k'],g['v']],1),centers).argmin(1)
            return torch.nn.functional.one_hot(labels,32).float()
        return model(g['k'],g['v'],g['ends'],g['nodes'])
    def evaluate(arm,graphs,model=None):
        rows=[]
        with torch.no_grad():
            for g in graphs:
                if arm=='A0':pred,lz=g['teacher'];active=32;maximum=0.
                else:
                    s=assignment(g,'A4' if arm=='A5' else arm,model);pi,kb,vb=summary(g['k'],g['v'],s)
                    if arm=='A5':pred,lz=answer(g['q'],kb[pi>0],vb[pi>0])
                    else:pred,lz=answer(g['q'],kb,vb,pi)
                    active=int((pi>1e-5).sum());maximum=float(pi.max())
                true=g['teacher'][0];mask=g['qmask'];err=(pred-true).square().mean(-1);ze=(lz-g['teacher'][1]).abs()
                row={'owner':g['owner'],'domain':g['domain'],'kind':g['kind'],'active_slots':active,'max_slot_mass':maximum,
                     'true_mse':float(err[:50][mask[:50]].mean()) if mask[:50].any() else 0.,'probe_mse':float(err[50:].mean()),
                     'true_logZ':float(ze[:50][mask[:50]].mean()) if mask[:50].any() else 0.,'probe_logZ':float(ze[50:].mean()),
                     'relative_L2':float(err.mean().sqrt())/normalizer}
                rows.append(row)
        return rows
    results=[];curves=[];replays=[];topology_effects=[]
    for arm in ('A0','A1','A2'):
        rows=evaluate(arm,valid+test);results.append({'arm':arm,'real':metric(rows,'real'),'synthetic':{k:metric(rows,'synthetic',k) for k in ('A','B','C')},'neural_fits':0})
    train_owners=sorted({g['owner'] for g in fit});by_owner={o:[g for g in fit if g['owner']==o] for o in train_owners}
    for arm in ('A3','A4'):
        globals()['PROGRESS']={'seed':args.seed,'arm':arm,'completed_prior_neural_fits':len(curves),'epoch':0,'optimizer_updates_current_fit':0}
        torch.manual_seed(args.seed+90001);model=SparseCompressor().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best=math.inf;best_state=None;best_epoch=None;patience=0;updates=0;history=[];max_answer_gradient=0.
        t0=time.monotonic()
        for epoch in range(1,101):
            model.train();permutation=np.random.default_rng(args.seed+epoch).permutation(train_owners);epoch_loss=0.
            for first in range(0,len(permutation),16):
                optimizer.zero_grad(set_to_none=True);owners=permutation[first:first+16];total=0.
                for o in owners:
                    losses_one=[]
                    for g in by_owner[o]:
                        s=assignment(g,arm,model);la,lc=losses(g['k'],g['v'],s,g['q'],g['teacher'],g['qmask'])
                        losses_one.append((la if arm=='A4' else 0.)+.1*lc)
                    total=total+torch.stack(losses_one).mean()/len(owners)
                if not torch.isfinite(total):raise FloatingPointError('NONFINITE_QPR_TRAINING')
                total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();updates+=1;epoch_loss+=float(total.detach())
                globals()['PROGRESS'].update(epoch=epoch,optimizer_updates_current_fit=updates)
            model.eval();val=evaluate(arm,valid,model)
            vr=metric(val,'real');score=math.sqrt(.5*(vr['RMSE_true']**2+vr['RMSE_probe']**2))
            history.append({'epoch':epoch,'training_objective_batch_sum':epoch_loss,'validation_true_RMSE':vr['RMSE_true'],'validation_probe_RMSE':vr['RMSE_probe'],'selection_RMSE':score})
            if score<best:best=score;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;patience=0
            else:patience+=1
            if epoch%10==0:print(json.dumps({'seed':args.seed,'arm':arm,'epoch':epoch,'updates':updates}),flush=True)
            if patience>=10:break
        model.load_state_dict(best_state);model.eval();rows=evaluate(arm,valid+test,model)
        saved=out/f'.seed{args.seed}_{arm}_replay.pt';torch.save(model.state_dict(),saved)
        restored=SparseCompressor().cuda();restored.load_state_dict(torch.load(saved,weights_only=True));restored.eval()
        again=evaluate(arm,valid+test,restored)
        err=max(abs(a['true_mse']-b['true_mse']) for a,b in zip(rows,again))
        if err>1e-12:raise ValueError('ACTUAL_G1_CHECKPOINT_REPLAY_FAILED')
        filehash=hashlib.sha256(saved.read_bytes()).hexdigest();saved.unlink()
        results.append({'arm':arm,'real':metric(rows,'real'),'synthetic':{k:metric(rows,'synthetic',k) for k in ('A','B','C')},
                        'selected_epoch':best_epoch,'trained_epochs':epoch,'optimizer_updates':updates,'parameter_count':sum(p.numel() for p in model.parameters()),
                        'active_slots_min':min(x['active_slots'] for x in rows),'max_slot_mass':max(x['max_slot_mass'] for x in rows),'neural_fits':1,'elapsed_seconds':time.monotonic()-t0})
        curves.append({'arm':arm,'history':history});replays.append({'arm':arm,'checkpoint_sha256':filehash,'metric_replay_max_abs':err})
        (out/f'seed{args.seed}_PROGRESS.private.json').write_text(json.dumps({'progress':globals()['PROGRESS'],'completed_results':results,'completed_curves':curves,'replays':replays},indent=2)+'\n')
        if arm=='A4':
            wrong=evaluate('A5',valid+test,model);results.append({'arm':'A5','real':metric(wrong,'real'),'synthetic':{k:metric(wrong,'synthetic',k) for k in ('A','B','C')},'neural_fits':0})
            ga=[g for g in valid+test if g['domain']=='synthetic' and g['kind']=='A']
            with torch.no_grad():
                for first in range(0,len(ga),2):
                    g,h=ga[first:first+2]
                    pg,kg,vg=summary(g['k'],g['v'],assignment(g,'A4',model));ph,kh,vh=summary(h['k'],h['v'],assignment(h,'A4',model))
                    gg=answer(g['q'],kg,vg,pg)[0];hh=answer(h['q'],kh,vh,ph)[0]
                    topology_effects.append({'teacher_gap':float((g['teacher'][0]-h['teacher'][0]).norm()),'student_gap':float((gg-hh).norm())})
    payload={'seed':args.seed,'role':'OUTCOME_FREE_DEVELOPMENT_NOT_DSS','results':results,'curves':curves,'checkpoint_replays':replays,
             'train_real_patients':len(data['fit']),'validation_real_patients':len(data['validation']),'neural_fits':2,'global_training_KMeans_fits':1,
             'cache_receipt_sha256':hashlib.sha256(Path(args.cache_receipt).read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-started,
             'topology_effects':topology_effects,'DSS_labels_read':False,'torch_cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'formal_lock_sha256':hashlib.sha256(Path(args.lock).read_bytes()).hexdigest()}
    target.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');print(json.dumps({'seed':args.seed,'status':'COMPLETE'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--cache',required=True);p.add_argument('--cache-receipt',required=True);p.add_argument('--output',required=True);p.add_argument('--lock',required=True);p.add_argument('--reports',required=True);args=p.parse_args()
    try:main(args)
    except Exception as exc:
        Path(args.output).mkdir(parents=True,exist_ok=True)
        (Path(args.output)/f'seed{args.seed}_FAILURE.private.json').write_text(json.dumps({'status':'FAILED','error_type':type(exc).__name__,'message':str(exc),'progress':globals().get('PROGRESS')},indent=2)+'\n')
        raise

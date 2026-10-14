"""Immutable upstream; common budget diagnostic appendix, never overwrite core."""
import argparse,itertools,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_phase14 import *
from src.phase14_rna.models import Conditional,fit_model
APP=PRIVATE/'budget_appendix';ACONFIG=ROOT/'configs/phase14_budget_appendix.json'


def appendix_hash():return hashlib.sha256((source()+sha(ACONFIG)+sha(__file__)).encode()).hexdigest()


def job(ep,gene,seed,fold):
    name=unit(ep,gene,seed,fold);out=APP/'cells'/f'{name}.json'
    if out.exists():
        c=json.loads(out.read_text())
        if c['appendix_hash']!=appendix_hash() or sha(c['artifact'])!=c['sha256']:raise RuntimeError('stale appendix')
        return
    configuration={**cfg(),**json.loads(ACONFIG.read_text())};torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    cell=json.loads((PRIVATE/'cells'/f'{name}.json').read_text());original=torch.load(cell['artifact'],map_location='cpu',weights_only=False)
    if original['source_hash']!=source() or sha(cell['artifact'])!=cell['artifact_sha256']:raise RuntimeError('changed upstream')
    ev=parent.endpoints(parent.load())[ep];t,e=ev['time'],ev['event'];new={'states':{},'risks':{},'infos':{},'partitions':original['partitions'],'appendix_hash':appendix_hash()}
    for role in ['inner','outer']:
        f,m,a,fit,test=role_data(ep,gene,fold,role);cs=np.c_[f['C'],f['S']];producer=RNAOnly(m['clinical_mask'],m[gene],seed).cuda();producer.load_state_dict(original['states'][role+'_RANK'])
        _,a=extract_rna(producer,f[gene],fit);_,b=extract_rna(producer,f[gene],test);del producer
        # Recreate the exact sklearn fit/transform arithmetic of the core.
        # Stored-transform algebra has equivalent math but float32 rounding
        # can change Adam's very small gradients, so trajectory gate rejects it.
        r,rv,transform=reduce_rep(a['processed_full'],b['processed_full'],64)
        for key in ['input_mean','input_sd','pca_mean','components','output_mean','output_sd']:
            if not np.array_equal(transform[key],original['transforms'][role+'_RANK64'][key]):raise RuntimeError('upstream transform not exact '+key)
        base=json.loads((parent.PRIVATE/'formal/cells'/f'{ep}_CS_seed31_fold{fold}.json').read_text())['lambda']
        for arm,interaction,wrong in [('ADDITIVE',False,False),('INTERACT',True,False),('WRONG_ADDITIVE',False,True),('WRONG_INTERACT',True,True)]:
            rr=r[index_map(fit,original['permutations'][role+'_fit'])] if wrong else r;vv=rv[index_map(test,original['permutations'][role+'_test'])] if wrong else rv
            w=original['probes'][role+('_WRONG_COX' if wrong else '_COX')]['w'];q=Conditional(cs.shape[1],64,w,interaction,configuration,seed).cuda()
            info=fit_model(q,[cs[fit],rr],[cs[test],vv] if role=='inner' else [],t,e,fit,test if role=='inner' else np.array([],int),configuration,
                           epoch=None if role=='inner' else new['infos']['inner_'+arm]['selected_epoch'],penalty=lambda model:model.regularization(base,configuration))
            if role=='inner':
                old=original['downstream'][role+'_'+arm]['logs']
                err=max(abs(info['logs'][k]['validation_EMA']-old[k]['validation_EMA']) for k in range(80))
                if err>1e-10:raise RuntimeError('first80 trajectory changed '+str(err))
                info['first80_EMA_max_error']=err
            else:
                with torch.no_grad():risk=q(torch.tensor(cs[test],device='cuda'),torch.tensor(vv,device='cuda')).cpu().numpy()
                new['risks'][arm]=risk
            new['infos'][role+'_'+arm]=info;new['states'][role+'_'+arm]={k:v.cpu() for k,v in q.state_dict().items()};del q
    artifact=APP/'artifacts'/f'{name}.pt';artifact.parent.mkdir(parents=True,exist_ok=True);torch.save(new,artifact)
    # Logs kept once inprivateartifact;publiccell onlysummary/statistic notpatient-level.
    write(out,{'status':'COMPLETE','endpoint':ep,'gene':gene,'seed':seed,'fold':fold,'artifact':str(artifact),'sha256':sha(artifact),'appendix_hash':appendix_hash()});print('APPENDIX_COMPLETE',name,flush=True)


def schedule():
    configuration=cfg();jobs=list(itertools.product(configuration['endpoints'],configuration['genes'],configuration['seeds'],configuration['folds']));active=[];done=0;failed=[]
    (APP/'logs').mkdir(parents=True,exist_ok=True)
    while jobs or active:
        while jobs and len(active)<12:
            ep,gene,seed,fold=jobs.pop(0);name=unit(ep,gene,seed,fold);fp=(APP/'logs'/f'{name}.log').open('w');p=subprocess.Popen([sys.executable,'-u',__file__,'--endpoint',ep,'--gene',gene,'--seed',str(seed),'--fold',str(fold)],stdout=fp,stderr=subprocess.STDOUT,cwd=ROOT);active.append((p,name,fp))
        for p,name,fp in active[:]:
            if p.poll() is not None:
                fp.close();active.remove((p,name,fp));done+=1
                if p.returncode:failed.append(name);jobs.clear()
                print('EXIT',name,p.returncode,flush=True)
        write(APP/'STATE.json',{'done':done,'pending':len(jobs),'active':[p.pid for p,n,f in active],'failed':failed})
        if active:time.sleep(10)
    if failed:raise RuntimeError(str(failed))
    print('APPENDIX_ALL_COMPLETE',flush=True)


if __name__=='__main__':
    if '--schedule' in sys.argv:schedule()
    elif '--launch' in sys.argv:
        APP.mkdir(exist_ok=True);write(REPORT/'BUDGET_APPENDIX_LOCK.json',{'status':'LOCKED','appendix_hash':appendix_hash(),'scope':json.loads(ACONFIG.read_text()),'parent_source_hash':source()})
        env=os.environ.copy();env.update({'PYTHONPATH':'/root/phase10_trimodal_eval_deps','CUBLAS_WORKSPACE_CONFIG':':4096:8','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        with (APP/'run.log').open('w') as fp:p=subprocess.Popen([sys.executable,'-u',__file__,'--schedule'],stdout=fp,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
        print(json.dumps({'pid':p.pid,'appendix_hash':appendix_hash()}))
    else:
        ap=argparse.ArgumentParser();ap.add_argument('--endpoint',required=True);ap.add_argument('--gene',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--fold',type=int,required=True);a=ap.parse_args();job(a.endpoint,a.gene,a.seed,a.fold)

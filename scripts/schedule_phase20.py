"""All135 inner selections precede135 fresh outer fits; bounded parallelism."""
import json
import subprocess
import sys
import time
from src.phase20_selective.data import P,R,ROOT,cfg,source,write
from src.phase10_trimodal.data import digest
from scripts.run_phase20 import stem
from src.phase20_selective.provenance import require_record


def queue(stage):
    config=cfg();identity=source()
    jobs=[(a,s,f) for f in config['folds'] for s in config['seeds'] for a in config['arms']]
    if (P/(stage+'_DONE.json')).exists():
        done=require_record(P/(stage+'_DONE.json'))
        if done['status']!='PASS' or done['formal_fits']!=135: raise ValueError('invalid completed stage')
        for a,s,f in jobs:
            receipt=P/stage/(stem(a,s,f)+'.json');require_record(receipt,receipt.with_suffix('.pt'))
        print('VERIFIED_EXISTING_PHASE20_STAGE',stage,135,flush=True);return
    todo,live,completed,telemetry=list(jobs),[],[],[]
    gpu_path=P/(stage+'_GPU.json')
    if gpu_path.exists(): telemetry=json.loads(gpu_path.read_text())
    begin,last=time.monotonic(),0.
    (P/'logs').mkdir(parents=True,exist_ok=True)
    while todo or live:
        limits=json.loads((P/'WORKERS.json').read_text()) if (P/'WORKERS.json').exists() else {'workers':6}
        count=int(limits['workers'])
        if count<1 or count>12: raise ValueError('invalid resource limit')
        while todo and len(live)<count:
            arm,seed,fold=todo.pop(0);label=stem(arm,seed,fold);receipt=P/stage/(label+'.json')
            if receipt.exists():
                require_record(receipt,receipt.with_suffix('.pt'))
                completed.append(label);continue
            log=(P/'logs'/f'{stage}_{label}.log').open('w')
            child=subprocess.Popen([sys.executable,'scripts/run_phase20.py','--arm',arm,'--seed',str(seed),
                                    '--fold',str(fold),'--stage',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            live.append((child,log,label));print('START',stage,label,child.pid,flush=True)
        for row in list(live):
            child,log,label=row;code=child.poll()
            if code is None: continue
            log.close();live.remove(row)
            if code:
                for other,stream,_ in live: other.terminate();stream.close()
                write(P/'FAILED.json',{'source':identity,'role':stage,'cell':label,'exit':code})
                raise SystemExit('FAILED '+label)
            completed.append(label);print('DONE',stage,label,len(completed),flush=True)
        if time.monotonic()-last>=10:
            memory,util=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            telemetry.append({'unix_time':time.time(),'role':stage,'memory_MiB':memory,'GPU_percent':util,
                              'running':len(live),'workers_limit':count})
            write(P/(stage+'_GPU.json'),telemetry);last=time.monotonic()
        write(P/'PROGRESS.json',{'source':identity,'role':stage,'completed':len(completed),'total':len(jobs),
            'running':[q[2] for q in live],'pending':len(todo),'wall_seconds':time.monotonic()-begin})
        if live: time.sleep(2)
    origins={}
    for label in completed:
        origin=require_record(P/stage/(label+'.json'))['source'];origins[origin]=origins.get(origin,0)+1
    write(P/(stage+'_DONE.json'),{'status':'PASS','source':identity,'formal_fits':len(completed),'seconds':time.monotonic()-begin,
        'actual_fit_source_counts':origins})


def main():
    identity=source();config=cfg()
    require_record(P/'PREPARATION_DONE.json')
    queue('inner')
    rows=[json.loads(p.read_text()) for p in (P/'inner').glob('*.json')]
    if len(rows)!=135: raise ValueError('inner matrix incomplete')
    fractions={a:sum(q['summary']['hit_maximum'] for q in rows if q['arm']==a)/15 for a in config['arms']}
    ok=all(x<=config['engineering_gate']['maximum_cap_fraction_per_arm'] for x in fractions.values())
    if (R/'EPOCH_SELECTION_LOCK.json').exists():
        selection=require_record(R/'EPOCH_SELECTION_LOCK.json');budget=require_record(R/'BUDGET_AUDIT.json')
        expected={(q['arm'],q['seed'],q['fold']):(q['selected_epoch'],q['checkpoint_sha256']) for q in rows}
        actual={(q['arm'],q['seed'],q['fold']):(q['epoch'],q['checkpoint_sha256']) for q in selection['selections']}
        if expected!=actual or budget['inner_cap_fractions']!=fractions: raise ValueError('original inner lock changed')
    else:
        write(R/'BUDGET_AUDIT.json',{'status':'PASS' if ok else 'BUDGET_UNRESOLVED','source':identity,
            'inner_cap_fractions':fractions,'no_outer_results_used':True,'zero_cap_does_not_prove_global_convergence':True})
        write(R/'EPOCH_SELECTION_LOCK.json',{'status':'ALL135_INNER_SELECTIONS_LOCKED','source':identity,
            'selections':[{'arm':q['arm'],'seed':q['seed'],'fold':q['fold'],'epoch':q['selected_epoch'],
                           'checkpoint_sha256':q['checkpoint_sha256']} for q in rows],'no_outer_scores_used':True})
    queue('outer')
    origins={}
    for role in ['inner','outer']:
        for path in (P/role).glob('*.json'):
            origin=require_record(path,path.with_suffix('.pt'))['source'];origins[origin]=origins.get(origin,0)+1
    write(P/'TRAINING_DONE.json',{'status':'PASS','source':identity,'new_inner_fits':135,'new_outer_fits':135,
        'new_formal_fits':270,'new_unique_convex_fits':0,'qualified_inherited_unique_convex_models':20,
        'old_neural_checkpoints_used':0,'budget_status':'PASS' if ok else 'BUDGET_UNRESOLVED',
        'actual_fit_source_counts':origins,'prior_phase_neural_initializations_used':0,
        'smoke':json.loads((P/'SMOKE_COUNTS.json').read_text())})
    print('ALL_PHASE20_TRAINING_DONE',flush=True)


if __name__=='__main__': main()

"""Fifteen inner fits locked before fifteen fresh outer refits; GPU queue."""
import json
import subprocess
import sys
import time
from src.phase21_relative.data import ROOT,P,R,cfg,source,write
from scripts.run_phase21 import stem,record
from src.phase10_trimodal.data import digest

def queue(stage):
    todo=[(s,f) for f in cfg()['folds'] for s in cfg()['seeds']];live=[];done=[];telemetry=[]
    while todo or live:
        workers=int(json.loads((P/'WORKERS.json').read_text())['workers'])
        if not 1<=workers<=8: raise ValueError('invalid worker count')
        while todo and len(live)<workers:
            seed,fold=todo.pop(0);label=stem(seed,fold);receipt=P/stage/(label+'.json')
            if receipt.exists():
                row=record(receipt)
                if row['checkpoint_sha256']!=digest(receipt.with_suffix('.pt')): raise ValueError('checkpoint changed')
                done.append(label);continue
            log=(P/'logs'/f'{stage}_{label}.log').open('a')
            child=subprocess.Popen([sys.executable,'scripts/run_phase21.py','--seed',str(seed),'--fold',str(fold),'--stage',stage],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            live.append((child,log,label));print('START',stage,label,child.pid,flush=True)
        for row in list(live):
            child,log,label=row;code=child.poll()
            if code is None: continue
            log.close();live.remove(row)
            if code:
                for other,stream,_ in live: other.terminate();stream.close()
                write(P/'FAILED.json',{'source':source(),'role':stage,'cell':label,'exit':code});raise RuntimeError('failed '+label)
            done.append(label);print('DONE',stage,label,len(done),flush=True)
        memory,util=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        telemetry.append({'unix_time':time.time(),'GPU_MiB':memory,'GPU_percent':util,'running':len(live)})
        write(P/(stage+'_GPU.json'),telemetry)
        write(P/'PROGRESS.json',{'source':source(),'role':stage,'completed':len(done),'total':15,'running':[q[2] for q in live],'pending':len(todo)})
        if live: time.sleep(10)
    write(P/(stage+'_DONE.json'),{'status':'PASS','source':source(),'fits':15})

def main():
    queue('inner');rows=[record(P/'inner'/(stem(s,f)+'.json')) for f in cfg()['folds'] for s in cfg()['seeds']]
    write(R/'EPOCH_SELECTION_LOCK.json',{'status':'ALL15_INNER_EPOCHS_LOCKED','source':source(),
          'no_outer_results_used':True,'selections':[{k:q[k] for k in ['seed','fold','selected_epoch','checkpoint_sha256']} for q in rows],
          'hit_maximum_fraction':sum(q['summary']['hit_maximum'] for q in rows)/15})
    queue('outer')
    write(P/'TRAINING_DONE.json',{'status':'PASS','source':source(),'new_formal_fits':30,'reused_control_models':60,'no_neural_warm_start':True,
          'smoke':json.loads((P/'SMOKE_LEDGER.json').read_text()),'generator_steps':sum(record(p)['summary']['actual_epochs'] for role in ['inner','outer'] for p in (P/role).glob('*.json'))})
    print('PHASE21_TRAINING_DONE',flush=True)

if __name__=='__main__': main()

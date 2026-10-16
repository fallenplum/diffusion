import os,json,subprocess,sys,time
from src.phase18a_readout.data import P,R,ROOT,cfg,source,write,digest
from scripts.run_phase18a import stem

def queue(stage):
    c=cfg();jobs=[(arm,s,f) for f in c['folds'] for s in c['seeds'] for arm in c['new_arms']];todo=list(jobs);live=[];done=[];telemetry=[];begin=time.monotonic();last=0
    (P/'logs').mkdir(exist_ok=True)
    while todo or live:
        workers=4
        if (P/'WORKERS.json').exists():workers=int(json.loads((P/'WORKERS.json').read_text())['workers'])
        while todo and len(live)<workers:
            arm,s,f=todo.pop(0);label=stem(arm,s,f);receipt=P/stage/(label+'.json')
            if receipt.exists():
                q=json.loads(receipt.read_text());assert q['source']==source() and q['checkpoint_sha256']==digest(receipt.with_suffix('.pt'));done.append(label);continue
            h=(P/'logs'/f'{stage}_{label}.log').open('w');p=subprocess.Popen([sys.executable,'scripts/run_phase18a.py','--arm',arm,'--seed',str(s),'--fold',str(f),'--stage',stage],cwd=ROOT,stdout=h,stderr=subprocess.STDOUT);live.append((p,h,label));print('START',stage,label,p.pid,flush=True)
        for item in list(live):
            p,h,label=item;code=p.poll()
            if code is None:continue
            h.close();live.remove(item)
            if code:
                for pp,hh,_ in live:pp.terminate();hh.close()
                write(P/'FAILED.json',{'source':source(),'role':stage,'cell':label,'exit':code});raise SystemExit('FAILED '+label)
            done.append(label);print('DONE',stage,label,len(done),flush=True)
        if time.monotonic()-last>10:
            mem,util=map(float,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            telemetry.append({'unix_time':time.time(),'role':stage,'memory_MiB':mem,'GPU_percent':util,'running':len(live),'workers_limit':workers});last=time.monotonic();write(P/(stage+'_GPU.json'),telemetry)
        write(P/'PROGRESS.json',{'source':source(),'role':stage,'completed':len(done),'total':len(jobs),'running':[q[2] for q in live],'pending':len(todo),'minutes':(time.monotonic()-begin)/60})
        if live:time.sleep(2)
    write(P/(stage+'_DONE.json'),{'status':'PASS','source':source(),'cells':len(done),'seconds':time.monotonic()-begin})

def main():
    assert json.loads((R/'DATA_LEAKAGE_AUDIT.json').read_text())['status']=='PASS'
    queue('inner');write(R/'EPOCH_SELECTION_LOCK.json',{'source':source(),'status':'ALL30_INNER_SELECTIONS_COMPLETED_BEFORE_NEW_OUTER_FITS','architecture_and_schedule_prelocked':True,'outer_used_for_selection':False})
    queue('outer');smoke=json.loads((P/'SMOKE_DIAGNOSTIC_COUNTS.json').read_text());write(P/'TRAINING_DONE.json',{'status':'PASS','source':source(),'new_inner_fits':30,'new_outer_fits':30,'historical_primary_inner_references':30,'historical_primary_outer_references':30,'secondary_A3_outer_references':15,'diagnostic_short_fits':smoke['diagnostic_short_fits'],'phase18b_started':False});print('ALL_PHASE18A_TRAINING_DONE',flush=True)

if __name__=='__main__':main()

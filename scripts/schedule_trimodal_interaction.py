"""Detached, fail-closed feasibility queue; never evaluates outer-test patients."""
import json,os,subprocess,time,sys
from src.trimodal_interaction.data import P,R,ROOT,cfg,source,write,digest


def main():
    c=cfg();s=source();jobs=[]
    for fold in c['folds']:
        for arm in c['arms']:
            for seed in ([31] if arm=='A0' else c['seeds']):jobs.append((arm,seed,fold))
    pending=[]
    for arm,seed,fold in jobs:
        label=f'{arm}_seed{seed}_fold{fold}';p=P/'cells'/f'{label}.json'
        if p.exists():
            m=json.loads(p.read_text());assert m['source']==s and digest(P/'checkpoints'/f'{label}.pt')==m['checkpoint_sha256']
        else:pending.append((arm,seed,fold))
    active=[];P.joinpath('logs').mkdir(exist_ok=True);telemetry=[];last=0
    while pending or active:
        while pending and len(active)<3:
            arm,seed,fold=pending.pop(0);label=f'{arm}_seed{seed}_fold{fold}';handle=(P/'logs'/f'{label}.log').open('w')
            cmd=[sys.executable,'scripts/run_trimodal_interaction.py','--arm',arm,'--seed',str(seed),'--fold',str(fold)]
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT);active.append((child,handle,label))
        for item in list(active):
            child,handle,label=item;code=child.poll()
            if code is None:continue
            handle.close();active.remove(item)
            if code:
                for p,h,_ in active:p.terminate();h.close()
                write(P/'FAILED.json',{'cell':label,'exit':code});raise SystemExit('FAILED '+label)
            print('QUEUE_DONE',label,len(pending),'pending',flush=True)
        if time.monotonic()-last>10:
            telemetry.append({'time':time.time(),'running':len(active),'pending':len(pending),'GPU':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()});last=time.monotonic();write(P/'GPU_TELEMETRY.json',telemetry)
        time.sleep(1)
    write(P/'DONE.json',{'status':'ALL_LOCKED_INNER_CELLS_FINISHED','source':s,'real_neural_fits':81,'real_convex_baseline_fits':3,'evaluation_baseline_references':9,'outer_fits':0,'patient_OOF_files':0,'cells':len(jobs)})
    print('ALL_DONE',len(jobs),flush=True)

if __name__=='__main__':main()

"""Sequential R1 -> conditional R2 -> gated R3; outer phase only after decisions."""
import sys,time,json,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17r_ssd.data import P,R,cfg,source,write
from src.phase10_trimodal.data import digest
from scripts.run_phase17r import stem

def queue(jobs,label):
    todo=list(jobs);live=[];done=[];fail=[];monitor=[];last=0;start=time.time();(P/'logs').mkdir(parents=True,exist_ok=True)
    while todo or live:
        while todo and len(live)<4:
            arm,s,f,scale,stage,direction=todo.pop(0);suffix='' if direction=='forward' else '_reverse';name=stem(arm,s,f,scale);cell=P/(stage+suffix)/(name+'.json')
            if cell.exists():
                q=json.loads(cell.read_text());assert q['source']==source() and q['checkpoint_sha256']==digest(cell.with_suffix('.pt'));done.append(name);continue
            log=P/'logs'/f'{stage}{suffix}_{name}.log';h=log.open('w');args=[sys.executable,'scripts/run_phase17r.py','--arm',arm,'--seed',str(s),'--fold',str(f),'--scale',str(scale),'--stage',stage,'--direction',direction]
            pr=subprocess.Popen(args,cwd=ROOT,stdout=h,stderr=subprocess.STDOUT,env=os.environ.copy());live.append({'name':name,'proc':pr,'handle':h,'log':str(log)});print('START',label,name,pr.pid,flush=True)
        for q in live[:]:
            code=q['proc'].poll()
            if code is None:continue
            q['handle'].close();live.remove(q)
            if code:fail.append({'name':q['name'],'exit':code,'log':q['log']});todo=[]
            else:done.append(q['name'])
            print('FINISH',label,q['name'],code,len(done),flush=True)
        if time.time()-last>15:
            p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if p.returncode==0:
                mem,util=[float(x.strip()) for x in p.stdout.split(',')];monitor.append({'unix_time':time.time(),'stage':label,'memory_MB':mem,'GPU_percent':util})
            last=time.time()
        write(P/'PROGRESS.json',{'source':source(),'stage':label,'completed':len(done),'total':len(jobs),'running':[q['name'] for q in live],'remaining':len(todo),'failures':fail,'wall_minutes':(time.time()-start)/60})
        if live:time.sleep(3)
    import pandas as pd
    pd.DataFrame(monitor).to_csv(R/(label+'_GPU.csv'),index=False)
    write(P/(label+'_DONE.json'),{'status':'FAIL' if fail else 'PASS','source':source(),'completed':len(done),'expected':len(jobs),'failures':fail,'wall_seconds':time.time()-start})
    if fail:raise RuntimeError(fail)

def main():
    cs=cfg();pairs=[(s,f) for f in cs['folds'] for s in cs['seeds']]
    decision=R/'STAGE_DECISION.json'
    # Resume outer runs without recomputing any architecture decision.
    if not decision.exists() or not json.loads(decision.read_text())['architecture_decisions_locked_before_outer']:
        queue([(arm,s,f,1.,'inner','forward') for s,f in pairs for arm in cs['R1']],'R1_INNER')
        subprocess.run([sys.executable,'scripts/decide_phase17r.py','--stage','R1'],cwd=ROOT,check=True)
        if json.loads(decision.read_text())['R2']['run']:
            queue([(arm,s,f,scale,'inner','forward') for s,f in pairs for arm in cs['R2']['models'] for scale in cs['R2']['scales']],'R2_INNER')
            subprocess.run([sys.executable,'scripts/decide_phase17r.py','--stage','R2'],cwd=ROOT,check=True)
    dec=json.loads(decision.read_text());assert dec['source']==source() and dec['architecture_decisions_locked_before_outer']
    outer=[(arm,s,f,1.,'outer','forward') for s,f in pairs for arm in cs['R1']]
    if dec['R2']['run']:outer += [(q['arm'],q['seed'],q['fold'],q['scale'],'outer','forward') for q in dec['R2']['selected']]
    if dec['R3']['run']:
        arm=dec['R3_base_arm'];rows=[q for q in dec['R2']['selected'] if q['arm']==arm] if arm.startswith('R2') else [{'arm':arm,'seed':s,'fold':f,'scale':1.} for s,f in pairs]
        queue([(q['arm'],q['seed'],q['fold'],q['scale'],'inner','reverse') for q in rows],'R3_INNER')
        outer += [(q['arm'],q['seed'],q['fold'],q['scale'],'outer','reverse') for q in rows]
    queue(outer,'OUTER')
    write(P/'TRAINING_DONE.json',{'status':'PASS','source':source(),'inner_fits':len(list((P/'inner').glob('*.pt')))+len(list((P/'inner_reverse').glob('*.pt'))),'outer_fits':len(outer),'R3_executed':dec['R3']['run'],'outer_used_for_stage_decisions':False})
    print('ALL_TRAINING_DONE',flush=True)

if __name__=='__main__':main()

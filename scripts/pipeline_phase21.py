"""Finish the frozen S16 audit before occupying the GPU with fresh fits."""
import json
import subprocess
import sys
import time
from src.phase21_relative.data import P,ROOT,write

def main():
    def execute(args): subprocess.run([sys.executable,'scripts/audit_phase21_saved.py',*args],cwd=ROOT,check=True)
    execute(['--snapshot'])
    todo=[(role,fold) for role in ['inner','outer'] for fold in [1,2,3,4,5]];live=[];done=[]
    while todo or live:
        while todo and len(live)<8:
            role,fold=todo.pop(0);label=f'saved_{role}_fold{fold}'
            log=(P/'logs'/(label+'.log')).open('a')
            child=subprocess.Popen([sys.executable,'scripts/audit_phase21_saved.py','--role',role,'--fold',str(fold)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            live.append((child,log,label))
        for row in list(live):
            child,log,label=row;code=child.poll()
            if code is None: continue
            log.close();live.remove(row)
            if code:
                for other,stream,_ in live: other.terminate();stream.close()
                write(P/'SAVED_AUDIT_FAILED.json',{'cell':label,'exit':code});raise RuntimeError(label)
            done.append(label);print('SAVED_AUDIT_JOB_DONE',label,len(done),flush=True)
        write(P/'SAVED_AUDIT_PROGRESS.json',{'completed':len(done),'jobs':10,'running':[r[2] for r in live],'pending':len(todo)})
        if live: time.sleep(10)
    execute(['--collect'])
    subprocess.run([sys.executable,'scripts/start_phase21.py'],cwd=ROOT,check=True)
    print('FROZEN_AUDIT_COMPLETE_NEW_TRAINING_STARTED',flush=True)

if __name__=='__main__': main()

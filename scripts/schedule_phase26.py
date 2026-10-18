"""Launch exactly six independent GPU cells after qualification; never evaluate."""
from pathlib import Path
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase26_multiquery import runtime


if __name__=='__main__':
    identity=runtime.source();runtime.authorization(identity)
    jobs=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for fold in (1,2,3):
        for seed in (31,47):
            directory=runtime.cell_dir(seed,fold);directory.mkdir(parents=True,exist_ok=True)
            if (directory/'DONE_PRIVATE.json').exists():runtime.verify_done(directory,identity,seed,fold);continue
            if (directory/'STARTED_PRIVATE.json').exists():raise ValueError('existing attempt; no silent restart')
            stream=(directory/'formal.log').open('w')
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_phase26.py'),'--seed',str(seed),'--fold',str(fold)],
                                     cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            jobs.append((seed,fold,process,stream));time.sleep(.5)
    failed=[]
    for seed,fold,process,stream in jobs:
        code=process.wait();stream.close()
        if code:failed.append({'seed':seed,'fold':fold,'exit':code})
        else:runtime.verify_done(runtime.cell_dir(seed,fold),identity,seed,fold)
    runtime.atomic_json(runtime.PRIVATE/'SCHEDULER_PRIVATE.json',{'source':identity,'status':'FAILED' if failed else 'COMPLETE',
        'new_cells_launched':len(jobs),'failures':failed,'evaluation_started':False})
    if failed:raise SystemExit(1)
    print('PHASE26_ALL_SIX_CELLS_COMPLETE_NO_EVALUATION',flush=True)

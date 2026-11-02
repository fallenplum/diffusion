"""Exactly 36 source/data-bound G2 fits; never substitutes missing cells."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.qpr.protocol import require_stage
ARMS=('B1','B2','B3','B4','B5','B3_QUERY_OFF')


def launch(a):
    require_stage('gate2',a.reports);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    locksha=hashlib.sha256(Path(a.lock).read_bytes()).hexdigest()
    jobs=[(inner,arm,seed) for inner in (1,2,3) for seed in (31,47) for arm in ARMS]
    def one(job):
        inner,arm,seed=job;name=f'inner{inner}_{arm}_seed{seed}';done=out/name/'COMPLETE.private.json'
        if done.exists():
            data=json.loads(done.read_text())
            if data['formal_lock_sha256']!=locksha:raise ValueError('RESUME_FORMAL_SOURCE_CHANGED')
            return {'cell':name,'status':'REUSED_COMPLETE'}
        cache=str(Path(a.ram_root)/f'qpr_G2_inner{inner}_20261102.private.pkl')
        receipt=str(Path(a.prepared)/f'inner{inner}_READY.private.json')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (out/(name+'.log')).open('w') as stream:
            cmd=[sys.executable,str(ROOT/'scripts/qpr/gate2_worker.py'),'--cache',cache,'--cache-receipt',receipt,'--output',a.output,'--reports',a.reports,'--lock',a.lock,'--inner',str(inner),'--arm',arm,'--seed',str(seed)]
            code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return {'cell':name,'status':'COMPLETE' if code==0 else 'FAILED','exit_code':code}
    statuses=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(one,jobs):
            statuses.append(result);print(json.dumps(result),flush=True)
            (out/'RUN_STATE.json').write_text(json.dumps({'planned':36,'cells':statuses},indent=2)+'\n')
    if any(x['status']=='FAILED' for x in statuses):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for x in ('output','prepared','ram-root','reports','lock'):p.add_argument('--'+x,required=True)
    p.add_argument('--workers',type=int,default=4);launch(p.parse_args())

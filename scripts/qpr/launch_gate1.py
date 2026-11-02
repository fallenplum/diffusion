"""Ten locked seeds, at most twenty neural fits. Reuse complete source-bound seeds."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.qpr.protocol import require_stage


def launch(a):
    require_stage('gate1',a.reports)
    locked=json.loads(Path(a.lock).read_text())
    if locked['status']!='LOCKED_BEFORE_FORMAL_FITS':raise ValueError('NOT_LOCKED')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    def one(seed):
        target=out/f'seed{seed}.json'
        if target.exists():
            d=json.loads(target.read_text())
            if d['formal_lock_sha256']!=hashlib.sha256(Path(a.lock).read_bytes()).hexdigest():raise ValueError('RESUME_LOCK_CHANGED')
            return {'seed':seed,'status':'REUSED_COMPLETE'}
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (out/f'seed{seed}.log').open('w') as log:
            cmd=[sys.executable,str(ROOT/'scripts/qpr/gate1_worker.py'),'--seed',str(seed),'--cache',a.cache,'--cache-receipt',a.cache_receipt,'--output',a.output,'--lock',a.lock,'--reports',a.reports]
            code=subprocess.call(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        return {'seed':seed,'status':'COMPLETE' if code==0 else 'FAILED','exit_code':code}
    statuses=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as p:
        for result in p.map(one,range(9101,9111)):
            statuses.append(result);print(json.dumps(result),flush=True)
            (out/'RUN_STATE.json').write_text(json.dumps({'role':'OUTCOME_FREE_GATE1','cells':statuses},indent=2))
    if any(x['status']=='FAILED' for x in statuses):raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--cache-receipt',required=True);p.add_argument('--output',required=True);p.add_argument('--lock',required=True);p.add_argument('--reports',required=True);p.add_argument('--workers',type=int,default=6);launch(p.parse_args())

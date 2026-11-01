"""Bounded resumable scheduler: all locked cells, no architecture search."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--device',default='cuda');args=p.parse_args()
    lock=json.loads((ROOT/'configs/rscil/synthetic_lock.json').read_text(encoding='utf8'));out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    jobs=[(n,sc,s) for n in lock['sample_sizes'] for sc in lock['scenarios'] for s in lock['seeds']]
    def one(job):
        n,sc,s=job;name=f'{n}_{sc}_{s}'
        if (out/(name+'.json')).exists():return {'cell':name,'status':'REUSED_COMPLETE'}
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        with (out/(name+'.log')).open('w',encoding='utf8') as stream:
            code=subprocess.call([sys.executable,str(ROOT/'scripts/rscil/run_synthetic.py'),'--n',str(n),'--scenario',sc,'--seed',str(s),'--output',str(out),'--device',args.device],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return {'cell':name,'status':'COMPLETE' if code==0 else 'FAILED','returncode':code}
    completed=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();completed.append(row);print(json.dumps(row),flush=True)
            (out/'scheduler_status.json').write_text(json.dumps({'completed':completed,'planned':len(jobs)},indent=2)+'\n',encoding='utf8')
    if any(r['status']=='FAILED' for r in completed):raise SystemExit(1)


if __name__=='__main__':main()

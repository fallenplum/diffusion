"""One command for fresh matrix + independent OOF audit; use a fresh checkout."""
import argparse,subprocess,sys,os,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--output',required=True);ap.add_argument('--plan',action='store_true');a=ap.parse_args();root=Path(__file__).resolve().parents[1];out=Path(a.output)
 commands=[[sys.executable,'scripts/run_phase10_trimodal.py','gates'],[sys.executable,'scripts/schedule_phase10_trimodal.py','--stage','baselines','--workers',str(a.workers),'--output',str(out)],[sys.executable,'scripts/schedule_phase10_trimodal.py','--stage','mechanisms','--workers',str(a.workers),'--output',str(out)]]
 if a.plan:print(json.dumps({'commands':commands,'private_dataset':'/root/phase7a_foundation_headroom_v1/private_dataset.npz','existing_cache_required':'/root/phase10_trimodal_private/cache','scope':'all 420 formal records and independent OOF/paired evaluation; supplemental probes are separate commands'},indent=2));return 0
 if out.exists() and any(out.iterdir()):raise RuntimeError('fresh output required; current archived experiment must remain intact')
 out.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env.update({'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':'/root/phase10_trimodal_eval_deps','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'});exits=[]
 for command in commands:
  r=subprocess.call(command,cwd=root,env=env);exits.append(r)
  if r:break
 (out/'REPRODUCE_EXIT.json').write_text(json.dumps({'command_exits':exits,'complete':len(exits)==3 and all(x==0 for x in exits)}));return next((x for x in exits if x),0)
if __name__=='__main__':raise SystemExit(main())

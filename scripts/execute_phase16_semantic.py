"""Run legal-space gates, pilots, then overlap full retrieval and formal fitting."""
import sys,subprocess,os,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase16_semantic.core import P,MODEL,cfg,write,source
env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
def command(script,args=[]):return [sys.executable,str(ROOT/'scripts'/script)]+args
def run(script,args=[]):
    print('START',script,args,flush=True);subprocess.run(command(script,args),cwd=ROOT,env=env,check=True);print('FINISHED',script,flush=True)
def main():
    P.mkdir(exist_ok=True)
    while True:
        try:
            meta=json.loads((MODEL/'DOWNLOAD_COMPLETE.json').read_text());ready=all((MODEL/r['file']).exists() and (MODEL/r['file']).stat().st_size==r['bytes'] for r in meta['files'])
        except (FileNotFoundError,json.JSONDecodeError):ready=False
        if ready:break
        time.sleep(2)
    run('prepare_phase16_semantic.py',['--pilot']);run('launch_phase16_semantic.py',['--pilot']);run('gate_phase16_semantic.py')
    log=open(P/'full_prepare.log','w');prepare=subprocess.Popen(command('prepare_phase16_semantic.py'),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);training=subprocess.Popen(command('launch_phase16_semantic.py'),cwd=ROOT,env=env)
    while prepare.poll() is None or training.poll() is None:
        if prepare.poll() not in [None,0]:training.terminate();raise RuntimeError('full retrieval failed;training stopped')
        if training.poll() not in [None,0]:prepare.terminate();raise RuntimeError('formal training failed;retrieval stopped')
        time.sleep(2)
    log.close()
    if prepare.returncode or training.returncode:raise RuntimeError('semantic pipeline failed')
    write(P/'FIT_STAGE_DONE.json',{'status':'PASS','source':source(),'new_formal_fits':270,'prior_fold_evaluations':210,'G4':'BLOCKED_RAW_WSI_QA','evaluation_pending':True});print('SEMANTIC_FIT_STAGE_DONE',flush=True)
if __name__=='__main__':main()

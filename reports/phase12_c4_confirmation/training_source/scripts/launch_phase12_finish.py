import subprocess,sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=Path('/root/phase12_c4_private');env=os.environ.copy();env['PYTHONPATH']='/root/phase10_trimodal_eval_deps'
with (P/'final_audit.log').open('a') as log:
    process=subprocess.Popen([sys.executable,'scripts/finish_phase12.py'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(P/'final_audit.pid').write_text(str(process.pid));print(json.dumps({'pid':process.pid,'log':str(P/'final_audit.log')}))

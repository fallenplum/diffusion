"""Reproduce the fixed clinical baseline and compact RNA protocol."""
import os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for name in ['run_phase16r_light_rna.py','validate_phase16r_light_rna.py','report_phase16r_light_rna.py']:
        subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()

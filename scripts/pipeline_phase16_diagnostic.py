"""Two independent mechanism diagnostics and RNA-free visual baselines."""
import os,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    env=os.environ.copy();env.update(PYTHONPATH='/root/phase10_trimodal_eval_deps',CUBLAS_WORKSPACE_CONFIG=':4096:8',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    # These entrypoints share the provenance lock and volatile cache; a replay
    # serializes their setup to avoid racing shared metadata writes.
    for name in ['run_phase16_rna_diagnostic.py','run_phase16_visual_diagnostic.py','audit_phase16_query_sensitivity.py']:subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,env=env,check=True)
    for name in ['evaluate_phase16_diagnostic.py','audit_phase16_diagnostic_sources.py','compare_phase16_diagnostic_supplement.py','report_phase16_diagnostic.py']:subprocess.run([sys.executable,str(ROOT/'scripts'/name)],cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()

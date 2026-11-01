"""Real-cohort entry is intentionally not executable before the synthetic gate."""
import runpy
from pathlib import Path
if __name__=='__main__':
    runpy.run_path(str(Path(__file__).with_name('preflight.py')),run_name='__main__')
    raise RuntimeError('TCGA_INNER_IMPLEMENTATION_REQUIRES_PASSING_SYNTHETIC_GATE_AND_SOURCE_REVIEW')

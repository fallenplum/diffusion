"""Run actual numerical contracts, retaining an auditable pytest transcript."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,'-m','pytest','tests/rscil','-q'],cwd=ROOT))

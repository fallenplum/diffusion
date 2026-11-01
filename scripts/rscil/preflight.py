"""Fail closed before touching any real-cohort endpoint loader."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.rscil.protocol import require_synthetic_go
if __name__=='__main__':
    require_synthetic_go(json.loads((ROOT/'reports/rscil/decision.json').read_text(encoding='utf-8')))

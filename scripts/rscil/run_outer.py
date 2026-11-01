"""No accidental access to repeatedly observed outer endpoints."""
import json
from pathlib import Path
if __name__=='__main__':
    d=json.loads((Path(__file__).resolve().parents[2]/'reports/rscil/decision.json').read_text(encoding='utf-8'))
    if d.get('inner_pass') is not True:raise ValueError('OUTER_DENIED_BEFORE_INNER_GO')
    raise RuntimeError('OUTER_IMPLEMENTATION_REQUIRES_FROZEN_ALL_OUTER_SELECTIONS')

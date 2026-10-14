"""Independent raw source audits, strictly NEW report directory."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for name in ['audit_phase10_raw_sources.py','audit_phase10_omics_source.py']:
    path=ROOT/'scripts'/name
    code=path.read_text().replace('reports/phase10_trimodal_mechanism_audit','reports/phase12_c4_confirmation')
    exec(compile(code,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})

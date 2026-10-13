"""Reuse independently source-based raw-data audits in a NEW output directory."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for name in ['audit_phase10_raw_sources.py','audit_phase10_omics_source.py']:
    source=ROOT/'scripts'/name
    text=source.read_text().replace('reports/phase10_trimodal_mechanism_audit','reports/phase11_semantic_anchored_trimodal')
    exec(compile(text,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})

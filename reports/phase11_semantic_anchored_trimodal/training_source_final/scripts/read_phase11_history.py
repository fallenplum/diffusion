"""Print aggregate historical conclusions, avoiding identifiable records."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
registry=json.loads((ROOT/'reports/phase10_trimodal_mechanism_audit/SOURCE_REGISTRY.json').read_text())
for source in registry['historical_sources']:
    path=ROOT/source['path']
    if not path.exists():continue
    text=path.read_text();lines=[line for line in text.splitlines() if re.search('结论|机制|NO_GO|INCONCLUSIVE|裁决|decision|Conclusion|主要|限制',line,re.I)]
    print(source['path'])
    for line in lines[:4]+lines[-3:]:print(re.sub(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}','[private case]',line)[:600])

import json,csv,torch
from pathlib import Path
from src.po_phase0.runner import cox_loss_grouped
r=Path('reports/po_phase0_baseline_rerun')
rows=list(csv.DictReader((r/'fold_metrics.csv').open())); seeds=list(csv.DictReader((r/'seed_metrics.csv').open())); audit=json.loads((r/'clinical_removal_audit.json').read_text()); m=json.loads((r/'run_manifest.json').read_text())
assert len(rows)==135 and len(seeds)==27 and m['formal_cells_completed']==135 and m['decision']=='C. PO_FUSION_NOT_SUPPORTED'
assert all(x['status'] in ('PASS','PASS_SOURCE_AUDIT') for x in audit['models'].values())
assert len(list(csv.DictReader((r/'failure_ledger.csv').open())))==0
v=cox_loss_grouped(torch.tensor([1.,2.,2.]),torch.tensor([1.,1.,0.]),torch.tensor([0.2,0.1,0.3]))
assert bool(torch.isfinite(v))
print('PASS',len(rows),len(seeds),float(v))


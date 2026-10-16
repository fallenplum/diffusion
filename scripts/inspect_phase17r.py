"""Compact monitoring; never reads outer prediction values."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase17r_ssd.data import P,R
def main():
    progress=json.loads((P/'PROGRESS.json').read_text())
    print(json.dumps({k:progress[k] for k in ['stage','completed','total','running','failures','wall_minutes']}),flush=True)
    if (R/'STAGE_DECISION.json').exists():
        q=json.loads((R/'STAGE_DECISION.json').read_text());print('DECISION',json.dumps({'R2_run':q['R2']['run'],'R3_run':q['R3']['run'],'locked':q['architecture_decisions_locked_before_outer']}),flush=True)
    if (P/'AUDITS_DONE.json').exists():print('AUDITS_DONE',flush=True)
    if progress['stage']=='R1_INNER':
        import torch,numpy as np
        for arm in ['R1_A8_full','R1_A8_noDirect']:
            rows=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted((P/'inner').glob(arm+'_*.pt'))]
            if not rows:continue
            keys=['g_relative_L2','delta_over_base_RMS','B_condition_over_base_RMS','h_relative_L2','isolated_SSM_g_gradient_L2','gamma','ssm_over_anchor_SD']
            print(arm,json.dumps({'completed':len(rows),'selected_epoch_median':float(np.median([z['selected_epoch'] for z in rows])),**{k:float(np.median([z['mechanism'][k] for z in rows])) for k in keys}}),flush=True)
    if progress['stage']=='R2_INNER':
        import torch,numpy as np
        for arm in ['R2_A8','R2_FiLM','R2_Input']:
            for scale in [.25,1.]:
                rows=[torch.load(p,map_location='cpu',weights_only=False) for p in sorted((P/'inner').glob(f'{arm}_s{scale:g}_*.pt'))]
                if rows:print(arm,scale,json.dumps({'candidate_cells':len(rows),'selected_epoch_median':float(np.median([z['selected_epoch'] for z in rows])),'state_relative_L2_median':float(np.median([z['mechanism']['h_relative_L2'] for z in rows]))}),flush=True)
if __name__=='__main__':main()

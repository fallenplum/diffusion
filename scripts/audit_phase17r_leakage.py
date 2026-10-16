"""Actually alter outer labels then repeat short inner fits; retain no checkpoint."""
import sys,copy,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.phase17r_ssd.data import P,R,cfg,source,write,old
from src.phase17r_ssd.train import train_cell

def main():
    original=old.parent.endpoints;d=old.parent.load();base=original(d);te=old.parent.splits(d,'DSS',1)['outer_test'];modified=copy.deepcopy(base);modified['DSS']['time'][te]=modified['DSS']['time'][te]*9+77;modified['DSS']['event'][te]=1-modified['DSS']['event'][te]
    rows=[];arms=['R1_A8_noDirect']
    if json.loads((R/'STAGE_DECISION.json').read_text())['R2']['run']:arms.append('R2_A8')
    for arm in arms:
        a=train_cell(arm,31,1,'inner',1.,epochs=2)
        try:
            old.parent.endpoints=lambda _:modified
            b=train_cell(arm,31,1,'inner',1.,epochs=2)
        finally:old.parent.endpoints=original
        error=max(float((a['state'][k]-b['state'][k]).abs().max()) for k in a['state'] if a['state'][k].is_floating_point());assert error==0.
        assert a['history']==b['history'] and a['selected_epoch']==b['selected_epoch'] and np.array_equal(a['anchor_risk'],b['anchor_risk']) and np.array_equal(a['risk'],b['risk'])
        rows.append({'arm':arm,'actual_training_runs':2,'epochs_each':2,'outer_event_and_time_actually_mutated':True,'maximum_parameter_error':error,'selection_and_train_curves_identical':True,'predictions_identical':True})
        del a,b;torch.cuda.empty_cache()
    write(R/'DATA_LEAKAGE_AUDIT.json',{'status':'PASS','source':source(),'short_fits':len(rows)*2,'checkpoint_saved':False,'rows':rows,'preprocessing':'parent artifact hash locks plus inner/outer fit-index assertions at each real training call'})
    print('LABEL_MUTATION_AUDIT_PASS',len(rows)*2,flush=True)

if __name__=='__main__':main()

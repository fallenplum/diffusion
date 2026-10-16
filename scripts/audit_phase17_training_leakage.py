import sys,copy,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from src.phase17_ssd.data import P,R,parent,write
from src.phase17_ssd.train import train_cell

def main():
    original=parent.endpoints;d=parent.load();base=original(d);te=parent.splits(d,'DSS',1)['outer_test'];modified=copy.deepcopy(base)
    modified['DSS']['time'][te]=modified['DSS']['time'][te]*9+77
    modified['DSS']['event'][te]=1-modified['DSS']['event'][te]
    a=train_cell('A8',31,1,'inner',16,epochs=2)
    try:
        parent.endpoints=lambda ignored:modified
        b=train_cell('A8',31,1,'inner',16,epochs=2)
    finally:parent.endpoints=original
    error=max(float((a['state'][k]-b['state'][k]).abs().max()) for k in a['state'] if a['state'][k].is_floating_point())
    assert error==0 and a['selected_epoch']==b['selected_epoch'] and a['history']==b['history']
    import json
    q=json.loads((R/'DATA_LEAKAGE_AUDIT.json').read_text());q.update(status='PASS',complete_training_checkpoint_mutation_test={'status':'PASS','epochs_each':2,'outer_labels_actually_replaced':True,'max_parameter_difference':error,'identical_selection_and_curves':True})
    write(R/'DATA_LEAKAGE_AUDIT.json',q)
    # No smoke/diagnostic checkpoints are retained.
    print('FULL_TRAINING_LABEL_MUTATION_PASS',error,flush=True)

if __name__=='__main__':main()

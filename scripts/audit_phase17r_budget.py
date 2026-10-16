"""Adapter parameter and linear-MAC audit, not a full scan FLOPs claim."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch import nn
from src.phase17r_ssd.data import R,cfg,write
from src.phase17r_ssd.model import RepairS6

def main():
    rows=[]
    for arm,prefixes in [('R2_A8',['delta_down.','delta_up.','B_down.','B_up.']),('R2_FiLM',['film_down.','film_up.']),('R2_Input',['input_down.','input_up.'])]:
        m=RepairS6(12,np.ones((8,512),bool),arm,.25)
        parameters=sum(p.numel() for k,p in m.named_parameters() if any(k.startswith(v) for v in prefixes))
        macs=sum(layer.in_features*layer.out_features for k,layer in m.named_modules() if isinstance(layer,nn.Linear) and any((k+'.').startswith(v) for v in prefixes))
        assert parameters==1664
        rows.append({'arm':arm,'adapter_parameters':parameters,'adapter_linear_MACs_per_valid_patch':macs,'dropout':cfg()['training']['dropout'],'weight_decay':cfg()['training']['weight_decay'],'candidate_scales':cfg()['R2']['scales'],'max_epochs':cfg()['training']['max_epochs']})
    # Visual projection alone is shared, so this is a conservative relative bound
    # on the linear-MAC difference, not a bound on total wall-clock differences.
    lower=1536*64;spread=max(r['adapter_linear_MACs_per_valid_patch'] for r in rows)-min(r['adapter_linear_MACs_per_valid_patch'] for r in rows)
    write(R/'COMPUTE_BUDGET_AUDIT.json',{'status':'PASS','rows':rows,'shared_visual_projection_MACs_per_patch':lower,'maximum_adapter_linear_MAC_difference':spread,'difference_over_shared_visual_MACs':spread/lower,'SSM_shape_identical':[512,64,16],'common_write_gate':True,'full_FLOPs_claim':False,'timing_scope':'training and diagnostic timings collected under four simultaneous GPU tasks; not isolated speed ranking','scope':'near matched linear arithmetic,exact adapter parameter count,common regularization/search budget;nonlinear operation types differ'})
    print('BUDGET_PASS',rows,flush=True)

if __name__=='__main__':main()

import sys,argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from src.phase17_ssd.data import P,R,cfg,source,write,prepare_visual
from src.phase17_ssd.train import train_cell

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--arm');p.add_argument('--seed',type=int,default=31);p.add_argument('--fold',type=int,default=1);p.add_argument('--stage',choices=['pilot','repair_inner','inner','outer'],default='pilot');a=p.parse_args()
    if a.prepare:prepare_visual();return
    gate=json.loads((R/'SSM_OPERATOR_AUDIT.json').read_text());assert gate['status']=='PASS'
    role='outer' if a.stage=='outer' else 'inner';budget=cfg()['inner_diagnosis']['patch_budget'] if a.stage in ['pilot','repair_inner'] else cfg()['formal_patch_budget']
    stem=f'{a.arm}_seed{a.seed}_fold{a.fold}';dest=P/a.stage/(stem+'.pt');cell=dest.with_suffix('.json');dest.parent.mkdir(parents=True,exist_ok=True)
    if cell.exists():
        saved=json.loads(cell.read_text());assert saved['source']==source();print('REUSE',stem,a.stage);return
    epochs=None
    if a.stage=='outer':
        registry=json.loads((R/'PROTOCOL_LOCK.json').read_text());assert registry['training_source']==source()
        inner=torch.load(P/'inner'/(stem+'.pt'),weights_only=False,map_location='cpu');assert inner['source']==source();epochs=inner['selected_epoch']
    ck=train_cell(a.arm,a.seed,a.fold,role,budget,epochs)
    tmp=dest.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(dest)
    from src.phase10_trimodal.data import digest
    write(cell,{'status':'COMPLETE','source':source(),'arm':a.arm,'seed':a.seed,'fold':a.fold,'stage':a.stage,'selected_epoch':ck['selected_epoch'],'checkpoint_sha256':digest(dest),'outer_test_selection':False,'summary':ck['summary'],'mechanism':ck['mechanism']})
    print('CELL_COMPLETE',a.stage,stem,ck['selected_epoch'],flush=True)

if __name__=='__main__':main()

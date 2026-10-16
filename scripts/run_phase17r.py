import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.phase17r_ssd.data import P,R,cfg,source,write
from src.phase17r_ssd.train import train_cell
from src.phase10_trimodal.data import digest

def stem(arm,seed,fold,scale=1.):return f'{arm}_s{scale:g}_seed{seed}_fold{fold}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--scale',type=float,default=1.);p.add_argument('--stage',choices=['inner','outer'],required=True);p.add_argument('--direction',choices=['forward','reverse'],default='forward');a=p.parse_args()
    assert json.loads((R/'OPERATOR_AUDIT.json').read_text())['status']=='PASS'
    lock=json.loads((R/'PROTOCOL_LOCK.json').read_text());assert lock['source']==source()
    assert a.arm in cfg()['R1']+cfg()['R2']['models']
    assert a.scale in cfg()['R2']['scales'] if a.arm.startswith('R2') else a.scale==1.
    suffix='' if a.direction=='forward' else '_reverse';base=P/(a.stage+suffix);base.mkdir(parents=True,exist_ok=True)
    name=stem(a.arm,a.seed,a.fold,a.scale);path=base/(name+'.pt');cell=path.with_suffix('.json')
    if cell.exists():
        z=json.loads(cell.read_text());assert z['source']==source() and z['checkpoint_sha256']==digest(path);print('REUSE',name,flush=True);return
    epochs=None
    if a.direction=='reverse':assert json.loads((R/'STAGE_DECISION.json').read_text())['R3']['run']
    if a.stage=='outer':
        final=json.loads((R/'STAGE_DECISION.json').read_text());assert final['source']==source() and final['architecture_decisions_locked_before_outer']
        inner=torch.load(P/('inner'+suffix)/(name+'.pt'),map_location='cpu',weights_only=False);assert inner['source']==source();epochs=inner['selected_epoch']
        if a.arm.startswith('R2'):assert any(z['arm']==a.arm and z['seed']==a.seed and z['fold']==a.fold and z['scale']==a.scale for z in final['R2']['selected'])
    ck=train_cell(a.arm,a.seed,a.fold,a.stage,a.scale,epochs,a.direction)
    tmp=path.with_suffix('.tmp');torch.save(ck,tmp);tmp.replace(path)
    write(cell,{'status':'COMPLETE','source':source(),'arm':a.arm,'seed':a.seed,'fold':a.fold,'scale':a.scale,'role':a.stage,'direction':a.direction,'selected_epoch':ck['selected_epoch'],'checkpoint_sha256':digest(path),'summary':ck['summary'],'outer_test_used_for_selection':False})
    print('CELL_COMPLETE',a.stage,name,ck['selected_epoch'],flush=True)

if __name__=='__main__':main()

"""Two joint dropout/WD presets; selection only on the two pilot inner folds."""
import sys,copy,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.phase17_ssd import train as trainer
from src.phase17_ssd.data import P,R,cfg,write,source

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--fold',type=int,required=True);a=p.parse_args()
    preset=copy.deepcopy(cfg());preset['training']['dropout']=.25;preset['training']['weight_decay']=.01
    trainer.cfg=lambda:preset
    target=P/'tuning'/f'T1_{a.arm}_fold{a.fold}.pt';target.parent.mkdir(exist_ok=True)
    z=trainer.train_cell(a.arm,31,a.fold,'inner',128);z['tuning_preset']={'dropout':.25,'weight_decay':.01};torch.save(z,target)
    write(target.with_suffix('.json'),{'status':'COMPLETE','source':source(),'preset':z['tuning_preset'],'best_validation_EMA':min(r['validation_loss_EMA'] for r in z['history'])})
    print('TUNING_COMPLETE',a.arm,a.fold,flush=True)

if __name__=='__main__':main()

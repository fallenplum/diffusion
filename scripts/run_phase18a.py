import argparse,json
import torch
from src.phase18a_readout.data import P,R,cfg,source,write,digest
from src.phase18a_readout.train import train_cell

def stem(arm,seed,fold):return f'{arm}_seed{seed}_fold{fold}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['L0','L1'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--stage',choices=['inner','outer'],required=True);a=p.parse_args()
    c=cfg();assert a.seed in c['seeds'] and a.fold in c['folds'];assert json.loads((R/'PROTOCOL_LOCK.json').read_text())['source']==source()
    assert json.loads((R/'OPERATOR_PARITY.json').read_text())['status']=='PASS'
    assert json.loads((R/'HISTORICAL_REUSE_AUDIT.json').read_text())['status']=='PASS'
    base=P/a.stage;base.mkdir(parents=True,exist_ok=True);path=base/(stem(a.arm,a.seed,a.fold)+'.pt');receipt=path.with_suffix('.json')
    if receipt.exists():
        q=json.loads(receipt.read_text());assert q['source']==source() and q['checkpoint_sha256']==digest(path);print('REUSE_VERIFIED',path.stem,flush=True);return
    epochs=None
    if a.stage=='outer':
        ip=P/'inner'/path.name;iq=json.loads(ip.with_suffix('.json').read_text());assert iq['source']==source() and iq['checkpoint_sha256']==digest(ip)
        inner=torch.load(ip,map_location='cpu',weights_only=False);epochs=inner['selected_epoch']
    result=train_cell(a.arm,a.seed,a.fold,a.stage,epochs);tmp=path.with_suffix('.tmp');torch.save(result,tmp);tmp.replace(path)
    write(receipt,{'status':'COMPLETE','source':source(),'checkpoint_sha256':digest(path),'arm':a.arm,'role':a.stage,'seed':a.seed,'fold':a.fold,'selected_epoch':result['selected_epoch'],'outer_test_used_for_selection':False,'summary':result['summary']})
    print('CELL_COMPLETE',a.stage,path.stem,result['selected_epoch'],flush=True)

if __name__=='__main__':main()

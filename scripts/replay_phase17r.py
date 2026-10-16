import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np,pandas as pd
from src.phase17r_ssd.data import P,R,cfg,source,write,Data,seed_all
from src.phase17r_ssd.model import RepairS6
from src.phase10_trimodal.data import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args();paths=sorted((P/'outer').glob('*.pt'))+sorted((P/'outer_reverse').glob('*.pt'));rows=[]
    assert json.loads((P/'TRAINING_DONE.json').read_text())['status']=='PASS'
    for path in paths[a.shard::a.shards]:
        z=torch.load(path,map_location='cpu',weights_only=False);cell=json.loads(path.with_suffix('.json').read_text());assert cell['checkpoint_sha256']==digest(path) and z['source']==source()
        seed_all(z['seed']+100*z['fold']);ds=Data(z['fold'],'outer',z['seed'],z['direction']);model=RepairS6(ds.f['C'].shape[1],ds.f['membership'],z['arm'],z['scale'],cfg()['training']['dropout']).cuda();model.load_state_dict(z['state']);model.eval()
        if z['arm']!='R1_A8_full':assert model.direct is None and not any(k.startswith('direct.') for k in z['state'])
        assert np.array_equal(ds.fit,z['fit']) and np.array_equal(ds.valid,z['validation_or_test']) and np.array_equal(ds.wrong,z['wrong_map'])
        errors=[]
        with torch.no_grad():
            for key,kwargs in [('risk',{}),('wrong_risk',{'wrong_inference':True}),('SSM_off_risk',{'module_off':True})]:
                risk=ds.forward(model,ds.valid,**kwargs).cpu().numpy();errors.append(float(abs(risk-z[key]).max()))
            model.gamma.fill_(1.);risk=ds.forward(model,ds.valid).cpu().numpy();errors.append(float(abs(risk-z['gamma_one_risk']).max()))
        error=max(errors);assert error<1e-6,(path.stem,error)
        normalization={}
        if 'A8' in z['arm'] and z['direction']=='forward':
            with torch.no_grad():
                model.gamma.copy_(z['state']['gamma'].to(model.gamma.device));ix=ds.fit[:32]
                _,md=ds.forward(model,ix,diagnostic=True);_,wd=ds.forward(model,ix,diagnostic=True,wrong_inference=True)
            for key in ['g','normalized_g']:
                aa=torch.cat([q[key][q['valid_mask']].flatten() for q in md]);bb=torch.cat([q[key][q['valid_mask']].flatten() for q in wd]);normalization[key+'_relative_L2_probe']=float((aa-bb).norm()/aa.norm().clamp_min(1e-12))
            normalization['normalized_over_raw_context_difference']=normalization['normalized_g_relative_L2_probe']/max(normalization['g_relative_L2_probe'],1e-12)
        rows.append({'arm':z['arm'],'seed':z['seed'],'fold':z['fold'],'scale':z['scale'],'direction':z['direction'],'max_risk_replay_error':error,'noDirect_verified':z['arm']!='R1_A8_full','checkpoint_sha256':cell['checkpoint_sha256'],**normalization})
        del model,ds;torch.cuda.empty_cache();print('REPLAY',path.stem,error,flush=True)
    pd.DataFrame(rows).to_csv(R/f'CHECKPOINT_REPLAY_{a.shard}.csv',index=False)
    write(P/f'REPLAY_SHARD_{a.shard}.json',{'status':'PASS','source':source(),'replayed':len(rows),'maximum_error':max((r['max_risk_replay_error'] for r in rows),default=0.)})

if __name__=='__main__':main()

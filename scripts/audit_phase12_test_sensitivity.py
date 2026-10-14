"""Fixed C4-g0 models: test-only zero /20 locked wrong-patient RNA contexts."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from src.phase11_semantic.models import SemanticModel
from src.phase11_semantic.data import PatchCache
from src.phase11_semantic.train import forward_partition
from src.phase12_c4.train import derange
from validate_phase11_independent import components,ipcw
R=ROOT/'reports/phase12_c4_confirmation';P=Path('/root/phase12_c4_private');OLD=Path('/root/phase11_semantic_anchor_private');cfg=json.loads((ROOT/'configs/phase12_c4_confirmation.json').read_text());d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));torch.set_num_threads(2)
t=d['DSS_time'];e=d['DSS_event'];folds=d['folds'];rows=[];cache=PatchCache(OLD/'cache',128);private=P/'fixed_model_sensitivity';private.mkdir(exist_ok=True)
for seed in cfg['seeds']:
    allrisks={name:np.empty(512) for name in ['matched','zero']+['permutation_'+str(j) for j in range(20)]};sums={name:np.zeros(4) for name in allrisks}
    for fold in cfg['folds']:
        tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold);z=dict(np.load(P/'preprocessing'/f'fold{fold}_outer.npz'));f={k.split('__',2)[2]:v for k,v in z.items() if k.startswith('f__top__')}
        ck=torch.load(OLD/'formal/checkpoints'/f'C4_seed{seed}_fold{fold}.pt',map_location='cpu',weights_only=False);model=SemanticModel('C4',ck['clinical_mask'],ck['rna_membership'],64).cuda().eval();model.load_state_dict(ck['state']);before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};weights=ipcw(t[tr],e[tr],t[te],e[te]);saved={}
        for name in allrisks:
            data={**f,'R':f['R'].copy()}
            if name=='zero':data['R'][te]=0 # standardized zero = TRAIN mean, not raw no-expression
            elif name.startswith('permutation_'):
                index=int(name.split('_')[-1]);source,meta=derange(d,te,cfg['test_permutation_seeds'][index]+fold);data['R'][te]=f['R'][source]
            with torch.no_grad():risk=forward_partition(model,d,data,cache,te,'C4',7301+303,64)[0].cpu().numpy()
            if name=='matched':
                expected=np.load(OLD/'formal/oof'/f'C4_seed{seed}_fold{fold}.npz')['risk']
                if np.max(abs(expected-risk))>2e-6:raise RuntimeError('fixed original model replay mismatch')
            hn,hd=components(t[te],e[te],risk);un,ud=components(t[te],e[te],risk,weights);sums[name]+=np.array([hn,hd,un,ud]);allrisks[name][te]=risk;saved[name]=risk
        if any(not torch.equal(value.detach().cpu(),before[key]) for key,value in model.state_dict().items()):raise RuntimeError('test sensitivity modified checkpoint weights')
        np.savez_compressed(private/f'seed{seed}_fold{fold}.npz',test_indices=te,**saved);del model;torch.cuda.empty_cache()
    for name,s in sums.items():rows.append({'seed':seed,'condition':name,'Harrell':s[0]/s[1],'Uno60':s[2]/s[3],'scope':'FIXED_MODEL_TEST_ONLY,no refit,20 contexts not independent patients'})
frame=pd.DataFrame(rows);frame.to_csv(R/'FIXED_MODEL_RNA_SENSITIVITY.csv',index=False)
summary=[]
for seed in cfg['seeds']:
    matched=frame[(frame.seed==seed)&(frame.condition=='matched')].iloc[0];permuted=frame[(frame.seed==seed)&frame.condition.str.startswith('permutation_')]
    for metric in ['Harrell','Uno60']:summary.append({'seed':seed,'metric':metric,'matched':matched[metric],'zero':frame[(frame.seed==seed)&(frame.condition=='zero')][metric].iloc[0],'permuted_mean':permuted[metric].mean(),'permuted_context_SD':permuted[metric].std(),'matched_minus_permuted_mean':matched[metric]-permuted[metric].mean()})
pd.DataFrame(summary).to_csv(R/'FIXED_MODEL_SENSITIVITY_SUMMARY.csv',index=False);print('FIXED_MODEL_SENSITIVITY_COMPLETE_NO_TRAINING')

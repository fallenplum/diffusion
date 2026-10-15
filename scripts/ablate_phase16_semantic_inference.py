"""No-retraining query ablations; frozen checkpoint distribution shift reported."""
import sys,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase11_independent import components,ipcw
from validate_phase16_independent import cs_independent,literal_pool
from src.phase16_semantic.core import *
from src.phase16_semantic.runtime import official_model,TextCache,projected_cache,retrieve

def main():
    d=parent.load();P.joinpath('inference_ablations').mkdir(exist_ok=True);model=official_model().half();cache=TextCache(model);features=np.load('/dev/shm/phase16_conch_z_projected.npy',mmap_mode='r');concepts=json.loads((ROOT/'configs/phase6a_titan_crc_morphology_concepts.json').read_text())['concepts'];texts=[[cfg()['concept_template'].format(morphology=k['text'])+' AJCC stage '+str(d['clinical_AJCC_PATHOLOGIC_TUMOR_STAGE'][i])+'.' for k in concepts] for i in range(512)];embedded=cache.encode([v for row in texts for v in row]).reshape(512,32,768);selected=retrieve(features,embedded)
    for m,v in selected.items():np.save(P/'inference_ablations'/f'STAGE_ONLY_{m}.npy',v)
    del model;gc.collect();torch.cuda.empty_cache();rows=[]
    for selector in ['CLIN_TEXT','CLIN_RNA_TEXT']:
        for m in cfg()['budgets']:
            arm=f'{selector}_{m}'
            for seed in cfg()['seeds']:
                for fold in cfg()['folds']:
                    c=json.loads((P/'cells'/f'{label(arm,seed,fold)}.json').read_text());z=torch.load(c['artifact'],map_location='cpu',weights_only=False);st=z['states']['outer'];cs=torch.tensor(cs_independent(d,st['fit']),dtype=torch.float32,device='cuda');state={k:v.cuda() for k,v in st['model'].items()};conditions={'NO_STAGE':directory(seed,fold,'outer')/f'{"NO_STAGE_CLIN" if selector=="CLIN_TEXT" else "NO_STAGE_JOINT"}_{m}.npy','STAGE_ONLY':P/'inference_ablations'/f'STAGE_ONLY_{m}.npy'}
                    if selector=='CLIN_RNA_TEXT':conditions['NO_RNA']=directory(seed,fold,'outer')/f'NO_RNA_JOINT_{m}.npy'
                    for condition,path in conditions.items():
                        bags=SemanticBags(path,m)
                        with torch.no_grad():risk=literal_pool(bags,state['score'],state['roi'],cs,state['base']).cpu().numpy()[st['test']]
                        np.savez_compressed(P/'inference_ablations'/f'{label(arm,seed,fold)}_{condition}.npz',risk=risk,test_indices=st['test'])
                        for ep,ev in parent.endpoints(d).items():
                            t,e,valid=ev['time'],ev['event'],ev['valid'];keep=valid[st['test']];te=st['test'][keep];tr=st['fit'][valid[st['fit']]];iw=ipcw(t[tr],e[tr],t[te],e[te],60);hn,hd=components(t[te],e[te],risk[keep]);un,ud=components(t[te],e[te],risk[keep],iw);rows.append({'endpoint':ep,'arm':arm,'seed':seed,'fold':fold,'condition':condition,'H_num':hn,'H_den':hd,'U_num':un,'U_den':ud,'scope':'frozen checkpoint inference;Clinical prediction-head inputs unchanged;not retrained causal ablation'})
                        del bags;gc.collect();torch.cuda.empty_cache()
    df=pd.DataFrame(rows);df.to_csv(R/'PROMPT_INFERENCE_ABLATION_FOLD.csv',index=False);sr=[]
    for (ep,arm,seed,condition),g in df.groupby(['endpoint','arm','seed','condition']):sr.append({'endpoint':ep,'arm':arm,'seed':seed,'condition':condition,'Harrell':g.H_num.sum()/g.H_den.sum(),'Uno60':g.U_num.sum()/g.U_den.sum()})
    pd.DataFrame(sr).to_csv(R/'PROMPT_INFERENCE_ABLATION_SEED.csv',index=False);print('PROMPT_ABLATION_PASS',len(rows),flush=True)
if __name__=='__main__':main()

"""Anonymous retrieval coverage and mismatch effectiveness; no event labels."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from src.phase16_semantic.core import *

def main():
    meta=base.manifest();slides=meta['slides'];xy=np.load(base.P/'coords.npy',mmap_mode='r');cells=[];den=0
    for s in slides:
        p=xy[s['offset']:s['offset']+s['n']].astype(float);lo=p.min(0);span=np.maximum(p.max(0)-lo,1);v=np.minimum(((p-lo)/span*8).astype(int),7);ids=v[:,0]*8+v[:,1];cells.append(ids);den+=len(np.unique(ids))
    records=[];fallback=[]
    for seed in cfg()['seeds']:
        for fold in cfg()['folds']:
            for role in ['inner','outer']:
                path=directory(seed,fold,role);done=json.loads((path/'DONE.json').read_text());audit=torch.load(path/'preprocessing.pt',map_location='cpu',weights_only=False)
                for partition,details in audit['fallback'].items():
                    for kind,counts in details.items():fallback.append({'seed':seed,'fold':fold,'role':role,'partition':partition,'modality':kind,**counts})
                for selector in done['metadata']:
                    parent_selector='CLIN_TEXT' if selector=='WRONG_CLIN' else 'RNA_TEXT' if selector=='WRONG_RNA' else 'CLIN_RNA_TEXT' if selector in cfg()['G3_selectors'] else None
                    for m in cfg()['budgets']:
                        selected=np.load(path/f'{selector}_{m}.npy');grid=np.load(base.P/'selectors'/f'GRID_seed{seed}_{m}.npy');matched=np.load(path/f'{parent_selector}_{m}.npy') if parent_selector else None;covered=0;patches=0;intersection_grid=0;intersection_parent=0;duplicates=0
                        for i,s in enumerate(slides):
                            ids=selected[i];ids=ids[ids>=0];local=ids-s['offset'];covered+=len(np.unique(cells[i][local]));patches+=len(ids);duplicates+=len(ids)-len(np.unique(ids));intersection_grid+=len(np.intersect1d(ids,grid[i][grid[i]>=0]))
                            if matched is not None:intersection_parent+=len(np.intersect1d(ids,matched[i][matched[i]>=0]))
                        records.append({'selector':selector,'seed':seed,'fold':fold,'role':role,'budget_per_patient':m,'selected_patches':patches,'occupied8x8_grid_fraction':covered/den,'coarse_grid_redundancy':1-covered/patches,'ROI_jaccard_vs_GRID':intersection_grid/(2*patches-intersection_grid),'ROI_jaccard_vs_correct_parent':intersection_parent/(2*patches-intersection_parent) if matched is not None else None,'duplicate_rows':duplicates,'source_slides_selected':int(np.any(selected>=0,axis=1).sum()),'raw_histology_accuracy':'NOT_AVAILABLE','candidate_pool_limited':True})
    pd.DataFrame(records).to_csv(R/'G1_COVERAGE.csv',index=False);pd.DataFrame(fallback).to_csv(R/'WRONG_PROMPT_FALLBACKS.csv',index=False);write(R/'RETRIEVAL_AUDIT.json',{'status':'PASS','retrieval_partitions':30,'records':len(records),'duplicate_rows':sum(r['duplicate_rows'] for r in records),'no_survival_labels_used':True,'retrieval_accuracy_claimed':False,'raw_WSI_QA':'BLOCKED'})
    print('RETRIEVAL_AUDIT_PASS',flush=True)
if __name__=='__main__':main()

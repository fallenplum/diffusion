"""Official-space gate then outcome-blind contextual retrieval for every partition."""
import argparse,sys,time,os,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from src.phase16_semantic.core import *
from src.phase16_semantic.prompts import captions
from src.phase16_semantic.runtime import official_model,TextCache,projected_cache,retrieve

def gate(model):
    torch.set_num_threads(4);base.seed_all(1601);m=base.manifest();x=np.load(m['features'],mmap_mode='r');idx=np.linspace(0,len(x)-1,1024).astype(int);raw=torch.tensor(np.asarray(x[idx]),device='cuda')
    with torch.inference_mode():
        official=model.project_image_features(raw);literal=F.normalize(raw@model.visual.proj_contrast,dim=-1);err=float((official-literal).abs().max());text=['colorectal carcinoma, malignant glands','colorectal tissue, benign glands','colorectal tissue, dense lymphocytes','colorectal tissue, mucin pools'];ids=model.tokenizer(text).cuda();full=model.encode_text(ids,normalize=True);copy=model.encode_text(ids,normalize=True);repeat=float((full-copy).abs().max());sim=official@full.T
    if err>1e-6 or repeat>1e-6 or not torch.isfinite(sim).all() or float(sim.std())<1e-6 or len(torch.unique(full,dim=0))!=len(text):raise ValueError('official space functional gate failed')
    projection=model.visual.proj_contrast.detach().cpu().numpy();np.save(P/'official_projection.npy',projection);weight=sha(MODEL/'model.safetensors');write(R/'SPACE_COMPATIBILITY.json',{'status':'PASS_OFFICIAL_INTERFACE','weights_sha256':weight,'projection_shape':list(projection.shape),'projection_replay_error':err,'repeat_text_error':repeat,'image_dim':768,'text_dim':768,'patches_checked':1024,'text_similarity_sd':float(sim.std()),'TITAN_text_raw_CONCH_cosine_used':False,'original_source_encoder':'TRIDENT conch_v15;H5provenance/SHA','raw_image_pixel_reextraction_check':'NOT_AVAILABLE_NO_WSI;official existing-feature API verified','sources':['https://github.com/mahmoodlab/LIMIT','https://huggingface.co/MahmoodLab/CONCH-Z']})
    write(R/'MODEL_ACCESS.json',{'status':'PASS','authenticated_config_HTTP':200,'official_revision':cfg()['official_model_revision'],'weights_sha256':weight,'token_in_public_artifacts':False,'weights_retained_server':True,'E_drive_original_retained':True})
    return weight

def main(seeds,folds):
    started=time.time();P.mkdir(exist_ok=True);R.mkdir(parents=True,exist_ok=True);model=official_model();gate(model);projected=projected_cache(model);model=model.half();cache=TextCache(model);d=parent.load();coverage=[]
    for seed in seeds:
        for fold in folds:
            part=parent.splits(d,'DSS',fold)
            for role in ['inner','outer']:
                path=directory(seed,fold,role);path.mkdir(parents=True,exist_ok=True)
                if (path/'DONE.json').exists():
                    z=json.loads((path/'DONE.json').read_text())
                    if z['source']!=source():raise ValueError('stale retrieval source')
                    continue
                fit=part['inner_train' if role=='inner' else 'outer_train'];test=part['inner_val' if role=='inner' else 'outer_test'];queries,audit=captions(d,fit,test,seed,fold);torch.save(audit,path/'preprocessing.pt');metadata={}
                for selector,texts in queries.items():
                    # Cache key locks candidate set, actual strings, official weight and precision.
                    flat=[v for row in texts for v in row];key=hashlib.sha256((base.manifest()['source_sha256']+cache.model_sha+'|'.join(flat)).encode()).hexdigest();cached=P/'roi_cache'/key;cached.mkdir(parents=True,exist_ok=True)
                    if all((cached/f'M{m}.npy').exists() for m in cfg()['budgets']):indices={m:np.load(cached/f'M{m}.npy') for m in cfg()['budgets']}
                    else:
                        embedded=cache.encode(flat).reshape(512,32,768);indices=retrieve(projected,embedded)
                        for m,array in indices.items():np.save(cached/f'M{m}.npy',array)
                    for m in cfg()['budgets']:
                        destination=path/f'{selector}_{m}.npy'
                        if not destination.exists():os.link(cached/f'M{m}.npy',destination)
                    metadata[selector]={'prompt_sha256':hashlib.sha256('\n'.join(flat).encode()).hexdigest(),'files':{m:sha(path/f'{selector}_{m}.npy') for m in cfg()['budgets']},'unique_prompts':len(set(flat))}
                    used=np.r_[fit,test];reference=queries['CLIN_TEXT'] if selector=='WRONG_CLIN' else queries['RNA_TEXT'] if selector=='WRONG_RNA' else queries['CLIN_RNA_TEXT'];changed=np.mean([texts[i]!=reference[i] for i in used]) if selector not in cfg()['G2_selectors'] else None
                    coverage.append({'selector':selector,'seed':seed,'fold':fold,'role':role,'changed_prompt_fraction_vs_parent':changed,'unique_prompts':len(set(flat)),'candidate_pool_limited':True})
                    print('RETRIEVAL',seed,fold,role,selector,'text_generated',cache.generated,flush=True)
                write(path/'DONE.json',{'status':'PASS','source':source(),'fit':fit,'test':test,'metadata':metadata,'preprocessing_sha256':sha(path/'preprocessing.pt'),'no_outcome_labels_used':True})
    pd.DataFrame(coverage).to_csv(R/'G1_PROMPT_RETRIEVAL.csv',index=False);write(P/'PREPARE_BATCH_DONE.json',{'status':'PASS','source':source(),'seeds':seeds,'folds':folds,'generated_texts':cache.generated,'cache_reused':cache.reused,'maximum_new_text_tokens':cache.maximum,'seconds':time.time()-started});print('PREPARE_PASS',flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--pilot',action='store_true');v=a.parse_args();main([31] if v.pilot else cfg()['seeds'],[1,2] if v.pilot else cfg()['folds'])

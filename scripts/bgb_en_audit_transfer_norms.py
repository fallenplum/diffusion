"""Read-only scale qualification for external UNI2 to TCGA transfer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import h5py
import numpy as np
from src.bgb_en.data import load_raw, read_patch_cache, require, sha256


def describe(values):
    values=np.asarray(values,np.float64)
    require(values.ndim==1 and len(values) and np.isfinite(values).all(),'invalid embedding norms')
    return {'n':len(values),'mean':float(values.mean()),'SD':float(values.std()),
            'quantiles':{str(q):float(np.quantile(values,q)) for q in (0,.01,.05,.25,.5,.75,.95,.99,1)}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets-json',required=True)
    parser.add_argument('--stage-b-prepared',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.time()
    assets=json.loads(Path(args.assets_json).read_text())
    raw=load_raw(assets,include_endpoints=False)
    cache=read_patch_cache(assets['patch_uni2'],verify_hash=False)
    with np.load(args.stage_b_prepared,allow_pickle=False) as source:
        source_x=np.asarray(source['X'],np.float32)
        source_neighbor=np.asarray(source['neighbor_X'],np.float32)
    source_norm=np.linalg.norm(source_x,axis=1)
    source_neighbor_norm=np.linalg.norm(source_neighbor,axis=1)
    cached_norm=np.linalg.norm(np.asarray(cache['features'])[cache['mask']],axis=1)
    lookup={pid:index for index,pid in enumerate(raw['patient_ids'])}
    paths=[path for path in Path(assets['uni2_root']).glob('*/*.h5') if path.stem[:12] in lookup]
    paths.sort(key=lambda path:(lookup[path.stem[:12]],path.name))
    require(len(paths)==len(cache['owner']),'native UNI2 and selected cache slide counts differ')
    all_raw,selected_raw,replay_errors=[],[],[]
    for index,path in enumerate(paths):
        with h5py.File(path,'r') as handle:
            value=np.asarray(handle['features'],np.float32).reshape(-1,1536)
        require(np.isfinite(value).all(),'nonfinite native UNI2')
        norm=np.linalg.norm(value,axis=1)
        all_raw.append(norm)
        rows=cache['rowids'][index,cache['mask'][index]]
        selected_raw.append(norm[rows])
        normalized=value[rows]/norm[rows,None]
        actual=np.asarray(cache['features'][index,cache['mask'][index]])
        replay_errors.append(float(np.max(np.abs(normalized-actual))))
    all_raw,selected_raw=np.concatenate(all_raw),np.concatenate(selected_raw)
    require(max(replay_errors)<=1e-7,'TCGA cache is not the matching native-H5 L2 normalization')
    result={'schema':'bgb_en_external_transfer_norm_qualification_v1','status':'FAIL_RAW_SOURCE_TO_L2_TARGET_SCALE_MATCH',
            'outcome_reads':0,'model_fits':0,'source_feature_space':'original Stage B UNI2 X; no norm transform',
            'target_feature_space':'matching original TCGA UNI2 selected rows divided by per-row L2 norm',
            'stage_b_X_shape':list(source_x.shape),'stage_b_X_L2':describe(source_norm),
            'stage_b_neighbor_X_L2':describe(source_neighbor_norm),
            'TCGA_selected_cache_L2':describe(cached_norm),'TCGA_matching_original_selected_H5_L2':describe(selected_raw),
            'TCGA_all_original_H5_L2':describe(all_raw),
            'target_L2_normalization_replay_max_abs_error':max(replay_errors),
            'source_raw_median_to_target_cache_median_ratio':float(np.median(source_norm)/np.median(cached_norm)),
            'stage_b_prepared_sha256':sha256(args.stage_b_prepared),
            'target_cache_READY_sha256':sha256(Path(assets['patch_uni2'])/'READY.json'),
            'recommendation':'Before fresh external pretraining, apply the same explicit row-L2 transform to external X; retain all historical Stage B artifacts unchanged. This fixes a measurement-scale mismatch, not biological domain shift.',
            'domain_alignment_proven':False,'wall_seconds':time.time()-started}
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    require('TCGA-' not in text and '/root/' not in text,'private identifier/path escaped transfer audit')
    Path(args.output).write_text(text,encoding='utf-8')
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()

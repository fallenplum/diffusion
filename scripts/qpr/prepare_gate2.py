"""Fresh, fully cross-fitted R4 anchor and real-edge caches for three splits."""
import argparse,hashlib,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.qpr.protocol import require_stage
from src.qpr.data import make_spatial_partition
from src.bgb_en.data import load_raw,native_conch_catalog
from src.bgb_en.protocol import fixed_inner_splits
from src.semantic_b0b7.data import read_membership
from src.semantic_b0b7.p2_data import attach_legacy_union
from src.semantic_b0b7.p2_anchor import fit_r4_anchor,array_sha
from src.semantic_m123r.io import partition_endpoints,sha


def prepare(a):
    require_stage('gate2',a.reports);lock=json.loads(Path(a.lock).read_text())
    if lock['status']!='LOCKED_BEFORE_FORMAL_FITS':raise ValueError('G2_NOT_LOCKED')
    for name,h in lock['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('G2_SOURCE_CHANGED:'+name)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    assets=json.loads(Path(a.assets).read_text());raw=load_raw(assets,include_endpoints=False)
    dev=np.flatnonzero(raw['folds']!=1);t,e=partition_endpoints(assets['endpoints'],raw['patient_ids'],dev,dev)
    split_event=np.zeros(512,int);split_event[dev]=e.astype(int)
    parts=fixed_inner_splits(raw['folds'],split_event,outer=1,seed=62001)
    catalog=native_conch_catalog(assets['conch_inventory'],root_override=a.conch_root)
    gmt=Path(assets['gmt']).read_text();names,member=read_membership(gmt,raw['rna_symbols'])
    anchor_raw,union_audit=attach_legacy_union(raw,assets['legacy_groups'])
    anchor_raw['t']=np.full(512,np.nan);anchor_raw['e']=np.full(512,np.nan)
    anchor_raw['t'][dev]=t;anchor_raw['e'][dev]=e
    mean_path=Path(assets['patch_grid'])/'full_mean.npy'
    if not mean_path.is_file():raise ValueError('EXACT_R4_FULL_MEAN_SOURCE_MISSING')
    anchor_raw['conch_mean']=np.asarray(np.load(mean_path,mmap_mode='r'),np.float64)
    anchor_raw['audit'].update(union_audit);anchor_raw['audit']['R4_full_CONCH_mean_sha256']=sha(mean_path)
    split_rows=[]
    for inner,(fit,valid) in enumerate(parts,1):
        started=time.monotonic();cache=Path(a.ram_root)/f'qpr_G2_inner{inner}_20261102.private.pkl'
        receipt=out/f'inner{inner}_READY.private.json'
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if sha(cache)!=old['cache_sha256'] or old['lock_sha256']!=sha(a.lock):raise ValueError('G2_RESUME_CHANGED')
            split_rows.append(old);continue
        data=make_spatial_partition(raw,catalog,fit,valid,gmt)
        anchor=fit_r4_anchor(anchor_raw,fit,valid,member,expected_union_sha256=array_sha(anchor_raw['_direct_gene_universe']),
                             legacy_root=a.legacy_root,expected_legacy_source_hashes=lock['legacy_anchor_sources'],
                             teacher_seed=62011+inner,pipeline_crossfit_seed=62111+inner)
        with (out/f'inner{inner}_ANCHOR.private.pkl').open('wb') as f:pickle.dump(anchor,f,protocol=5)
        data.update(fit=fit,evaluation=valid,r0_fit=anchor.training_offset,r0_evaluation=anchor.evaluation_offset,
                    train_time=anchor_raw['t'][fit],train_event=anchor_raw['e'][fit],evaluation_time=anchor_raw['t'][valid],evaluation_event=anchor_raw['e'][valid],
                    cancer=raw['cancer'],stage=raw['clinical_raw']['AJCC_PATHOLOGIC_TUMOR_STAGE'],anchor_receipt=anchor.receipt,
                    covariance_patient_rows=dev,role='G2_DSS_TRAIN_SIDE_ONLY_OUTER1_SEALED')
        with cache.open('wb') as f:pickle.dump(data,f,protocol=5)
        binding={'inner':inner,'fit_patients':len(fit),'validation_patients':len(valid),'fit_sha256':array_sha(fit),'validation_sha256':array_sha(valid),
                 'fit_events':int(anchor_raw['e'][fit].sum()),'validation_events':int(anchor_raw['e'][valid].sum()),
                 'cache_sha256':sha(cache),'cache_bytes':cache.stat().st_size,'lock_sha256':sha(a.lock),
                 'Cox_anchor_fits':anchor.receipt['Cox_optimizer_fits'],'scalar_anchor_fits':anchor.receipt['scalar_pair_optimizer_fits'],
                 'full_pipeline_honest_training_offsets':anchor.receipt['beta_supervision_excluded_for_each_downstream_training_patient'],
                 'decoded_endpoint_rows':len(dev),'outer1_test_endpoint_elements_decoded':0,'elapsed_seconds':time.monotonic()-started}
        receipt.write_text(json.dumps(binding,indent=2)+'\n');split_rows.append(binding);print(json.dumps(binding),flush=True)
    (out/'SPLIT_MANIFEST.private.json').write_text(json.dumps({'outer1_development_patients':len(dev),'development_events':int(e.sum()),'outer_test_rows_decoded':0,'splits':split_rows},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('assets','conch-root','legacy-root','output','ram-root','reports','lock'):p.add_argument('--'+name,required=True)
    prepare(p.parse_args())

"""Gate0 native source audit. No endpoint or RNA/clinical array is decoded."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import h5py
from src.qpr.graph import metadata_hop,native_edges,spatial_sample,diagnostics


def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):d.update(b)
    return d.hexdigest()


def run(config,out,conch_root):
    start=time.monotonic();a=json.loads(Path(config).read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    endpoint=Path(a['endpoints']).resolve()
    def guard(name,args):
        if name=='open' and isinstance(args[0],(str,bytes)):
            if Path(args[0]).resolve()==endpoint:raise RuntimeError('DSS_FILE_ACCESS_FORBIDDEN')
    sys.addaudithook(guard)
    with np.load(a['cohort'],allow_pickle=True) as z:
        # Explicit allowlist: do not load an entire archive or endpoint field.
        ids=z['patient_ids'].astype(str);folds=z['folds']
    digest=hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    if len(ids)!=512 or len(set(ids))!=512 or digest!='76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f':raise ValueError('CANONICAL_PATIENT_IDENTITY')
    inv=json.loads(Path(a['conch_inventory']).read_text());source=inv['chosen_conch']
    if len(source)!=519 or inv.get('errors') or inv.get('missing_conch'):raise ValueError('CANONICAL_SLIDE_INVENTORY')
    rows=[];private=[];coverage=np.zeros(512,int);permutations=[];tested_n=set();total_nodes=0;total_edges=0
    for i,r in enumerate(sorted(source,key=lambda v:(int(v['patient_index']),str(v['stem'])))):
        old=Path(r['path']);p=Path(conch_root)/old.parent.name/old.name
        patient=int(r['patient_index']);error='';record={'slide_index':i,'N':0,'E':0}
        try:
            if not p.is_file() or p.is_symlink() or p.stat().st_size!=int(r['bytes']):raise ValueError('NATIVE_SOURCE_MISSING_OR_BYTE_MISMATCH')
            h=sha(p)
            if h!=r['sha256']:raise ValueError('NATIVE_SOURCE_HASH_MISMATCH')
            with h5py.File(p,'r') as f:
                xy=f['coords'][:];x=f['features'][:]
                attrs={k:v.item() if isinstance(v,np.generic) else v for k,v in f['coords'].attrs.items()}
            if x.ndim!=2 or x.shape!=(len(xy),768) or not np.isfinite(x).all() or np.any(np.linalg.norm(x,axis=1)<=1e-12):raise ValueError('INVALID_NATIVE_FEATURES')
            if not old.stem.startswith(ids[patient]):raise ValueError('PATIENT_SLIDE_MAPPING_MISMATCH')
            hop=metadata_hop(attrs);canon,edges,order=native_edges(xy,hop)
            chosen,area=spatial_sample(canon,edges)
            record.update(N=len(xy),E=len(edges),selected_edges=len(chosen),sampling_ratio=len(chosen)/len(edges) if len(edges) else 0.,
                          spatial_bin_coverage=area,hop_level0_px=hop,patch_size_level0=int(attrs['patch_size_level0']),
                          target_mag=float(attrs['target_magnification']),level0_mag=float(attrs['level0_magnification']),
                          mpp_provided='mpp' in attrs,**diagnostics(len(xy),edges))
            if len(edges):coverage[patient]+=1
            if len(permutations)<10 and len(xy) not in tested_n:
                tested_n.add(len(xy));perm=np.random.default_rng(80001+i).permutation(len(xy))
                c2,e2,o2=native_edges(xy[perm],hop);s2,_=spatial_sample(c2,e2)
                ok=np.array_equal(canon,c2) and np.array_equal(edges,e2) and np.array_equal(chosen,s2) and np.array_equal(x[order],x[perm][o2])
                if not ok:raise ValueError('REAL_ROW_PERMUTATION_FAILED')
                permutations.append({'slide_index':i,'N':len(xy),'E':len(edges),'pass':True})
            total_nodes+=len(xy);total_edges+=len(edges)
            private.append({'patient':patient,'slide':i,'path':str(p),'source_sha256':h})
        except (ValueError,OSError,KeyError) as exc:error=str(exc)
        record['error']=error;rows.append(record)
        if i%50==0:print(json.dumps({'audited':i+1,'planned':len(source)}),flush=True)
    fields=sorted({k for row in rows for k in row})
    with (out/'SPATIAL_COVERAGE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    identity={'cohort_sha256':sha(a['cohort']),'patient_order_sha256':digest,'folds_sha256':hashlib.sha256(folds.tobytes()).hexdigest(),
              'inventory_sha256':sha(a['conch_inventory']),'slides':len(rows),'patients':512,
              'patients_with_eligible_edges':int((coverage>0).sum()),'zero_edge_patients':int((coverage==0).sum()),
              'invalid_slides':sum(bool(v['error']) for v in rows),'total_native_nodes':total_nodes,'total_true_edges':total_edges,
              'native_H5_hashes_verified':len(private),'real_permutation_tests':permutations,'DSS_file_opened':False,'DSS_columns_decoded':False,
              'cohort_arrays_decoded':['patient_ids','folds'],'elapsed_seconds':time.monotonic()-start}
    (out/'DATA_AUDIT.json').write_text(json.dumps(identity,indent=2)+'\n')
    (out/'PRIVATE_SOURCE_REGISTRY.private.json').write_text(json.dumps(private))
    print(json.dumps(identity),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True);p.add_argument('--conch-root',required=True);a=p.parse_args();run(a.assets,a.output,a.conch_root)

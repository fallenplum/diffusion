"""No-DSS gate1 cache. All estimators fit only designated development patients."""
import argparse,hashlib,json,pickle,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.decomposition import PCA
from src.bgb_en.data import load_raw,read_h5_slide,native_conch_catalog
from src.bgb_en.protocol import clinical_fit,scale_fit,scale_apply
from src.semantic_b0b7.data import ObservedScale,read_membership
from src.qpr.graph import metadata_hop,native_edges,spatial_sample
from src.qpr.kv import relation_features,project_edges


def prepare(args):
    from src.qpr.protocol import require_stage
    require_stage('gate1',args.reports)
    lock=json.loads(Path(args.lock).read_text())
    if lock['status']!='LOCKED_BEFORE_FORMAL_FITS':raise ValueError('FORMAL_SOURCE_LOCK_REQUIRED')
    for name,h in lock['sources'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('PREPROCESSING_SOURCE_CHANGED')
    started=time.monotonic();assets=json.loads(Path(args.assets).read_text());endpoint=Path(assets['endpoints']).resolve()
    def no_endpoints(name,values):
        if name=='open' and isinstance(values[0],(str,bytes)) and Path(values[0]).resolve()==endpoint:raise RuntimeError('NO_DSS_ENDPOINT_GUARD')
    sys.addaudithook(no_endpoints)
    raw=load_raw(assets,include_endpoints=False)
    dev=np.flatnonzero(raw['folds']!=1);rng=np.random.default_rng(62001);order=rng.permutation(dev)
    nv=int(.2*len(dev));valid=np.sort(order[:nv]);fit=np.sort(order[nv:])
    catalog=native_conch_catalog(assets['conch_inventory'],root_override=args.conch_root)
    def read(row):
        d=read_h5_slide(row['path'],feature_dim=768);hop=metadata_hop(d['coordinate_attrs']);xy,ed,ix=native_edges(d['coords'],hop);selected,_=spatial_sample(xy,ed)
        return d['features'][ix],xy,selected,hop
    node_samples=[]
    for i,r in enumerate(catalog):
        if r['patient'] in set(fit):
            x,xy,e,h=read(r);node_samples.append(x[np.linspace(0,len(x)-1,min(64,len(x))).astype(int)])
    pca_node=PCA(32,svd_solver='full').fit(np.concatenate(node_samples))
    examples=[];pool=[]
    for i,r in enumerate(catalog):
        if r['patient'] not in set(dev):continue
        x,xy,e,hop=read(r);h=pca_node.transform(x).astype(np.float32);rel=relation_features(h,xy,e,hop)
        if r['patient'] in set(fit):pool.append(rel[np.linspace(0,len(rel)-1,min(64,len(rel))).astype(int)])
        examples.append({'owner':r['patient'],'slide_index':i,'evidence':rel,'ends':e,'nodes':len(x),'split':'train' if r['patient'] in set(fit) else 'validation'})
    pca_edge=PCA(64,svd_solver='full').fit(np.concatenate(pool))
    for x in examples:
        x['k'],x['v']=project_edges(x.pop('evidence'),pca_edge)
    names,member=read_membership(Path(assets['gmt']).read_text(),raw['rna_symbols'])
    rna,measured=ObservedScale.fit(raw['rna'],fit).transform(raw['rna']);cnt=measured.astype(np.float32)@member.T
    coverage=cnt/np.maximum(member.sum(1),1)[None];mask=(coverage>=.5)&(cnt>=5)
    values=np.where(mask,rna@member.T/np.maximum(cnt,1),0)
    program=np.stack([values,mask.astype(float),coverage],-1).astype(np.float32)
    clinical,cstate=clinical_fit(raw,fit);sstate=scale_fit(raw['S32'],fit);s32=scale_apply(raw['S32'],sstate)
    obj={'graphs':examples,'fit':fit,'validation':valid,'clinical':clinical.astype(np.float32),'s32':s32.astype(np.float32),'program':program,'program_mask':mask,
         'pca_node':pca_node,'pca_edge':pca_edge,'pathways':names,'DSS_opened':False,'endpoint_arrays_decoded':0}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as f:pickle.dump(obj,f,protocol=5)
    receipt={'status':'OUTCOME_FREE_CACHE_READY','fit_patients':len(fit),'validation_patients':len(valid),'graphs':len(examples),'bytes':output.stat().st_size,
             'cache_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'fit_sha256':hashlib.sha256(fit.tobytes()).hexdigest(),'valid_sha256':hashlib.sha256(valid.tobytes()).hexdigest(),
             'DSS_file_opened':False,'endpoint_elements_decoded':0,'PCA_node_training_rows':sum(len(x) for x in node_samples),'PCA_edge_training_rows':sum(len(x) for x in pool),
             'elapsed_seconds':time.monotonic()-started}
    Path(args.receipt).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--conch-root',required=True);p.add_argument('--output',required=True);p.add_argument('--receipt',required=True);p.add_argument('--reports',required=True);p.add_argument('--lock',required=True);a=p.parse_args();prepare(a)

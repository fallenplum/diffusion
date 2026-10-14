import hashlib,json
from pathlib import Path
import h5py,numpy as np,pandas as pd

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(p,v):
    p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8');tmp.replace(p)

def load_private(path):
    d=dict(np.load(path,allow_pickle=True));ids=d['patient_ids'].astype(str)
    if len(ids)!=512 or len(set(ids))!=512 or not np.all(d['DSS_valid']) or d['DSS_event'].sum()!=65:raise ValueError('cohort gate failed')
    if hashlib.sha256('\n'.join(ids).encode()).hexdigest()!='76a144d544b7461d669487625855dc7de65f115a6a2f65d7a20002b2a8e5008f':raise ValueError('patient fingerprint mismatch')
    for k,n in [('static',32),('titan',768),('uni2',1536)]:
        if d[k].shape!=(512,n) or not np.isfinite(d[k]).all():raise ValueError(k+' dimension/finite gate')
    if sorted(np.unique(d['folds']))!=[1,2,3,4,5]:raise ValueError('invalid folds')
    if np.isinf(d['rna_genes']).any() or np.isinf(d['cna_genes']).any():raise ValueError('infinite omics')
    for key in ['rna_gene_symbols','cna_gene_symbols']:
        if len(set(d[key].astype(str)))!=len(d[key]) or not all(d[key].astype(str)):raise ValueError('symbol registry failure')
    return d

def build_patch_cache(d,manifest,root,output,cap=128):
    out=Path(output);out.mkdir(exist_ok=True,parents=True);tab=pd.read_csv(manifest).query("status=='EXACT_SLIDE_MATCH'")
    ids=d['patient_ids'].astype(str);loc={p:i for i,p in enumerate(ids)}
    if len(tab)!=519 or set(tab.patient_id)!=set(ids) or tab.filename.duplicated().any():raise ValueError('slide/patient manifest mismatch')
    maxslides=int(tab.groupby('patient_id').size().max());shape=(len(ids),maxslides,cap,768)
    features=np.lib.format.open_memmap(out/'patch.npy',mode='w+',dtype='float32',shape=shape);features[:]=0
    mask=np.zeros(shape[:3],bool);coords=np.zeros(shape[:3]+(2,),np.int64);nslide={};rows=[];total=0
    for row in tab.sort_values(['patient_id','filename']).itertuples(index=False):
        p=Path(root)/row.delivery_relative_path
        if digest(p)!=row.conch_sha256:raise ValueError('H5 SHA failure')
        with h5py.File(p,'r') as f:
            x=np.asarray(f['features'],np.float32);xy=np.asarray(f['coords'])
            if x.ndim!=2 or x.shape[1]!=768 or xy.shape!=(len(x),2) or not np.isfinite(x).all():raise ValueError('H5 shape/finite failure')
            if f['features'].attrs.get('encoder')!='conch_v15' or int(f['coords'].attrs['patch_size'])!=512 or int(f['coords'].attrs['target_magnification'])!=20:raise ValueError('encoder/scale failure')
            if not np.issubdtype(xy.dtype,np.integer) or (xy<0).any() or len(np.unique(xy,axis=0))!=len(xy):raise ValueError('coordinate failure')
        order=np.lexsort((xy[:,1],xy[:,0]));sel=order[np.linspace(0,len(x)-1,min(cap,len(x)),dtype=int)]
        i=loc[row.patient_id];s=nslide.get(i,0);nslide[i]=s+1;features[i,s,:len(sel)]=x[sel];mask[i,s,:len(sel)]=True;coords[i,s,:len(sel)]=xy[sel];total+=len(x)
        rows.append({'patient_id':row.patient_id,'filename':row.filename,'fold':int(d['folds'][i]),'sha256':row.conch_sha256,'original_patches':len(x),'selected_patches':len(sel)})
    features.flush();np.save(out/'mask.npy',mask);np.save(out/'coords.npy',coords)
    write_json(out/'private_slide_manifest.json',rows)
    audit={'status':'PASS','patients':len(ids),'slides':len(tab),'original_patches':total,'selected_patches':int(mask.sum()),'cache_shape':list(shape),'cache_sha256':digest(out/'patch.npy'),'manifest_sha256':digest(manifest),'dataset_sha256':digest('/root/phase7a_foundation_headroom_v1/private_dataset.npz'),'fold_hash':hashlib.sha256(d['folds'].tobytes()).hexdigest(),'patient_hash':hashlib.sha256('\n'.join(ids).encode()).hexdigest(),'source_encoder':'CONCHv1.5, 512@20X; no UNI2 row pairing','multi_slide':'equal slide weight','rna_shapes':{k:list(d[k].shape) for k in ['raw_rna','raw_cna','rna_genes','cna_genes']},'rna_missing_block':{'patients':int((np.isnan(d['raw_rna']).sum(1)==3024).sum()),'columns':3024,'platform_identity':'missing-pattern proxy only; direct GAII aliquot mapping not verified'},'clinical_fields':['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']}
    write_json(out/'data_manifest.json',audit);return audit

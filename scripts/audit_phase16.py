"""Outcome-blind G0 inventory; never infer spatial/semantic compatibility by dim."""
import hashlib,json,os,re,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import h5py,numpy as np,pandas as pd,torch
from src.phase13_rna import data as parent
P=Path('/root/phase16_routing_private');R=ROOT/'reports/phase16_semantic_routing'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def plain(x):
    if isinstance(x,bytes):return x.decode(errors='replace')
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=plain,allow_nan=False))


def discover():
    h5=[];wsi=[];models=[];skip={'miniconda3','.git','.cache','.vscode-server','conda-pkgs','node_modules','__pycache__','site-packages'}
    for directory,dirs,files in os.walk('/root',followlinks=False):
        dirs[:]=[d for d in dirs if d not in skip and not Path(directory,d).is_symlink()]
        for name in files:
            p=Path(directory,name);suffix=p.suffix.lower()
            if suffix in ['.h5','.hdf5']:h5.append(p)
            elif suffix in ['.svs','.ndpi','.mrxs']:wsi.append(p)
            elif suffix in ['.safetensors','.bin'] and ('model' in str(p).lower()):models.append(p)
    return h5,wsi,models


def main():
    P.mkdir(exist_ok=True);R.mkdir(parents=True,exist_ok=True);started=time.time();d=parent.load();ids=d['patient_ids'].astype(str);lookup={x:i for i,x in enumerate(ids)};h5,wsi,models=discover();inventory=[];errors=[]
    for path in h5:
        match=re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}',path.name)
        if not match or match[0] not in lookup:continue
        try:
            with h5py.File(path,'r') as f:
                if 'features' not in f:continue
                feature=f['features'];dim=int(feature.shape[-1]);n=int(np.prod(feature.shape[:-1]));attrs={k:plain(v) for k,v in feature.attrs.items()};coords=f['coords'][:].reshape(-1,2) if 'coords' in f else None
                coordinateattrs={k:plain(v) for k,v in f['coords'].attrs.items()} if coords is not None else {}
                row={'path':str(path),'stem':path.stem,'patient_index':lookup[match[0]],'bytes':path.stat().st_size,'sha256':sha(path),'keys':list(f.keys()),'dim':dim,'feature_shape':list(feature.shape),'patches':n,'feature_attrs':attrs,'coordinate_attrs':coordinateattrs,'file_attrs':{k:plain(v) for k,v in f.attrs.items()},'coords':coords is not None,'coordinates_shape':list(coords.shape) if coords is not None else None}
                if coords is not None:row.update({'coordinate_min':coords.min(0).tolist(),'coordinate_max':coords.max(0).tolist(),'coordinate_duplicates':n-len(np.unique(coords,axis=0)),'coordinates_integer':bool(np.issubdtype(coords.dtype,np.integer))})
                if dim==768 and attrs.get('encoder')=='conch_v15':
                    row['kind']='CONCH_V15';finite=True
                    for start in range(0,n,4096):finite=finite and bool(np.isfinite(feature[start:start+4096]).all())
                    row['finite']=finite
                elif dim==1536:row['kind']='UNI2H_CANDIDATE'
                else:row['kind']='OTHER'
                inventory.append(row)
        except Exception as exc:errors.append({'path':str(path),'error_type':type(exc).__name__,'error':str(exc)[:200]})
    # Authoritative historical exact slide association; hashes determine which
    # discovered files reproduce the original H5 candidate pool.
    previous=json.loads(Path('/root/phase11_semantic_anchor_private/cache/private_slide_manifest.json').read_text());conch=[r for r in inventory if r['kind']=='CONCH_V15'];chosen=[];missing=[]
    for old in previous:
        candidates=[r for r in conch if r['stem']==Path(old['filename']).stem and r['sha256']==old['h5_sha256'] and r['patient_index']==old['patient_index']]
        if not candidates:missing.append({'patient_index':old['patient_index'],'stem':Path(old['filename']).stem})
        else:
            row=sorted(candidates,key=lambda r:len(r['path']))[0]
            if row['patches']!=old['original'] or not row['finite'] or not row['coords'] or row['coordinate_duplicates']:raise RuntimeError('original candidate pool not reproducible')
            chosen.append(row)
    uni=[r for r in inventory if r['kind']=='UNI2H_CANDIDATE'];pairs=[]
    for con in chosen:
        candidates=[r for r in uni if r['stem']==con['stem'] and r['patient_index']==con['patient_index']]
        candidate=sorted(candidates,key=lambda r:len(r['path']))[0] if candidates else None
        pairs.append({'slide_number':len(pairs)+1,'conch_coordinates_present':con['coords'],'uni2_present':candidate is not None,'uni2_coordinates_present':bool(candidate and candidate['coords']),'conch_level0_size':con['coordinate_attrs'].get('patch_size_level0'),'uni2_level0_size':candidate['coordinate_attrs'].get('patch_size_level0') if candidate else None,'status':'BLOCKED_RAW_WSI_OR_COORD_VALIDATION' if not wsi or candidate is None or not candidate['coords'] else 'NEEDS_30_ROI_VISUAL_QA','area_overlap_fraction':None,'unmatched_fraction':None})
    if not (R/'COORDINATE_AUDIT.json').exists():pd.DataFrame(pairs).to_csv(R/'COORD_ALIGNMENT.csv',index=False)
    weightinfo=[]
    for path in models:
        info={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
        if path.suffix=='.safetensors':
            from safetensors import safe_open
            with safe_open(path,framework='pt') as f:info['keys']=list(f.keys());info['key_count']=len(info['keys'])
        elif 'CONCH' in str(path):
            z=torch.load(path,map_location='cpu',weights_only=True)
            if isinstance(z,dict) and 'state_dict' in z:z=z['state_dict']
            info['projection_keys']={k:list(v.shape) for k,v in z.items() if hasattr(v,'shape') and ('proj_contrast' in k or 'text_projection' in k)};del z
        weightinfo.append(info)
    rawRNA=np.asarray(d['raw_rna']);rawCNA=np.asarray(d['raw_cna']);rnasyms=d['rna_gene_symbols'].astype(str);cnasyms=d['cna_gene_symbols'].astype(str);missingmask=np.isnan(rawRNA);patterns=Counter(missingmask.sum(1).tolist())
    summary={'patients':len(ids),'expected_slides':len(previous),'CONCH_slides_recovered':len(chosen),'CONCH_patients':len({r['patient_index'] for r in chosen}),'CONCH_original_patches':sum(r['patches'] for r in chosen),'historical_128_patches':sum(min(128,r['original']) for r in previous),'historical_1024_patches':sum(r['selected'] for r in previous),'UNI2_candidate_slides':len(uni),'UNI2_with_coords':sum(r['coords'] for r in uni),'raw_WSI_files_discovered':len(wsi),'raw_RNA_shape':list(rawRNA.shape),'raw_CNA_shape':list(rawCNA.shape),'RNA_CNA_common_symbols':len(set(rnasyms)&set(cnasyms)),'RNA_missing_gene_counts_distribution':dict(patterns),'platform_identity':'not confirmed;missing-pattern proxy only','CNA_semantics':'gene-level log2 copy-number dosage;gain/loss separate;not RNA expression or mutation','candidate_pool_limited':len(chosen)!=len(previous),'tissue_mask':'source H5 candidate set only;no independent raw-image tissue mask verification','dataset_sha256':sha(parent.DATA),'inventory_errors':len(errors),'inventory_seconds':time.time()-started}
    scopefile=ROOT/'configs/phase16_scope_repair.json'
    if scopefile.exists():
        scope=json.loads(scopefile.read_text());summary.update(candidate_pool_limited=bool(missing or scope['candidate_pool_limited']),original_cohort_slides=scope['original_UNI2_cohort_slides'],missing_original_CONCH_slides=scope['missing_original_CONCH_slides'])
    summary['UNI2_patches_all_candidate_files']=sum(r['patches'] for r in uni)
    summary['UNI2_coordinate_row_counts_match_features']=all(r['coordinates_shape']==[r['patches'],2] for r in uni)
    write(P/'ASSET_INVENTORY.json',{'h5':inventory,'chosen_conch':chosen,'missing_conch':missing,'raw_wsi':[str(x) for x in wsi],'models':weightinfo,'errors':errors});write(P/'ASSET_SUMMARY.json',summary)
    # Exact image/text interface verification is conservative until the gated
    # CONCH-Z projection+text weights can be loaded and functionally replayed.
    space={'status':'SEMANTIC_ROUTE_BLOCKED_PENDING_CONCH_Z','TITAN_text_to_raw_CONCH_v15_cosine':'PROHIBITED_NOT_SAME_SPACE','TITAN_scope':'verified slide text alignment;retain S32/native baseline','original_CONCH_v15_release':'patch visual weights;no compatible contrastive projection+text checkpoint in audited assets','CONCH_Z_official_revision':'60211fced9c25a014494e248eedb6a6be80109f7','CONCH_Z_required_operations':['project_image_features(raw768)','encode_text(tokenizer(prompts),normalize=True)'],'CONCH_Z_access':'new user token authenticated config GET403:request awaiting review from repo authors;no bypass','sources':['https://github.com/mahmoodlab/LIMIT','https://github.com/mahmoodlab/TITAN/issues/17','https://github.com/mahmoodlab/CONCH']}
    write(R/'SPACE_COMPATIBILITY.json',space)
    decision={'status':'PARTIAL_G0','nonsemantic_pool':'PASS' if len(chosen)==len(previous) else 'CANDIDATE_POOL_LIMITED','semantic_route':'BLOCKED_CONCH_Z_ACCESS','dual_visual':'BLOCKED_RAW_WSI_OR_COORD_VALIDATION' if not wsi or not all(p['uni2_coordinates_present'] for p in pairs) else 'PENDING_VISUAL_QA','G5_semantic_omics':'BLOCKED_DEPENDS_ON_SEMANTIC_G2','continue':['G1_RAND_GRID_DIVERSE','G2_nonsemantic_and_anchors','MIL_train_only_pilot'],'candidate_pool_limited':summary['candidate_pool_limited']}
    write(R/'G0_DECISION.json',decision)
    text=['# G0 数据与接口审计','',json.dumps(summary,indent=2,ensure_ascii=False),'','## 语义空间','',json.dumps(space,indent=2,ensure_ascii=False),'','## 坐标与授权','',f'检查{len(pairs)}张CONCH候选切片；UNI2有坐标切片数{summary["UNI2_with_coords"]}；原始WSI文件数{len(wsi)}。未完成30组原图/空间验证前不运行双视觉分支，不按行号匹配。',
          '现有TITAN权重沿用此前用户下载/服务器常驻授权；官方模型限非商业研究。CONCH-Z需独立人工审批；不尝试绕过门禁。CONCH1.0模型若存在，不能直接替代CONCH1.5预计算特征的语义空间。','',
          '模型权重与每张H5的SHA和完整患者切片关联留在私有ASSET_INVENTORY.json；公开只含匿名覆盖与错误计数。原数据只读；RNA/CNA预处理及后续文本阈值均在对应训练分区拟合。']
    (R/'DATA_INTERFACE_AUDIT.md').write_text('\n'.join(text),encoding='utf8');print(json.dumps(summary));print(json.dumps(decision))


if __name__=='__main__':main()

# Frozen-artifact public projection

This is a lossless, alias-labelled publication projection, not the original executable artifact. Restore and verify the original bytes with `scripts/materialize_hest8_public_projections.py` before use. Scientific computation used the original SHA listed in the projection manifest.

```python
"""Frozen official HEST/TRIDENT UNI2-h BF16 extraction in a new namespace.

Pinned primary model-build and RGB Dataset ASTs execute their unchanged local
weight branches. No RNA, gene list, clinical or survival input is accepted.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,math,os,random,time,traceback
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parents[1]
WEIGHT_SHA='6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e'
PRIMARY_SHA={'benchmark.py':'02db96fe0cbe08202b22577cda97fd55820dfa97d359e7f9cb53cf88dc7b77d3',
    'st_dataset.py':'cf0886ba5435c83e7bd4a36d6819e6be824c8c0de1393ba2be2c8b53422133f2',
    'trident_load.py':'74189dfd5f1ec6ec46c87e830a2e1699c0f3f4717b0994a960a6915521a274fd'}

def require(ok,code):
    if not bool(ok):raise ValueError(code)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def runtime():
    import timm,torchvision
    return dict(torch=torch.__version__,timm=timm.__version__,torchvision=torchvision.__version__,
        CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(),
        matmul_TF32=torch.backends.cuda.matmul.allow_tf32,cudnn_TF32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,deterministic=torch.are_deterministic_algorithms_enabled())

def primary_functions(folder):
    folder=Path(folder)
    require(all(sha(folder/name)==digest for name,digest in PRIMARY_SHA.items()),'PINNED_PRIMARY_CODE_BYTES_CHANGED')
    trident=ast.parse((folder/'trident_load.py').read_text())
    helper=next(n for n in trident.body if isinstance(n,ast.FunctionDef) and n.name=='_resolve_target_img_size')
    cls=next(n for n in trident.body if isinstance(n,ast.ClassDef) and n.name=='UNIv2InferenceEncoder')
    build=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_build'))
    build.name='official_uni2_build'
    source=ast.parse((folder/'st_dataset.py').read_text())
    dataset=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='H5PatchDataset')
    module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')],level=0),helper,build,dataset],type_ignores=[]))
    ns=dict(torch=torch,traceback=traceback,h5py=h5py,np=np,Image=Image,Dataset=Dataset)
    exec(compile(module,'<pinned_official_UNIv2_and_H5PatchDataset>','exec'),ns)
    return ns['official_uni2_build'],ns['H5PatchDataset']

class LocalWeights:
    def __init__(self,path):self.path=str(path)
    def _get_weights_path(self):return self.path
    def ensure_has_internet(self,*_):raise ValueError('NETWORK_MODEL_LOADING_FORBIDDEN')

def setup(args):
    config=read(args.config)
    require(config['schema']=='hest_official8_uni2_bf16_feature_recipe_v1'
        and config['seed']==1 and config['batch_size']==128 and config['num_workers']==1
        and config['forward_precision']=='bfloat16' and config['parameter_dtype']=='float32'
        and config['model_weight_sha256']==WEIGHT_SHA and config['row_L2_normalized'] is False
        and config['dataset_repo']=='MahmoodLab/hest-bench'
        and config['dataset_revision']=='b039b5965ed7092f280d4f336da7972ef9a10e1a'
        and config['extractor_sha256']==sha(__file__) and len(config['patches'])==8
        and {v['section'] for v in config['patches']}=={'PUBLIC_COAD_SECTION_1','PUBLIC_COAD_SECTION_2','PUBLIC_COAD_SECTION_3','PUBLIC_COAD_SECTION_4','PUBLIC_READ_SECTION_1','PUBLIC_READ_SECTION_2','PUBLIC_READ_SECTION_3','PUBLIC_READ_SECTION_4'},
        'FROZEN_OFFICIAL_FEATURE_RECIPE_CHANGED')
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA_BF16_REQUIRED_NO_CPU_SUBSTITUTION')
    require(sha(args.weights)==WEIGHT_SHA,'OFFICIAL_UNI2_WEIGHT_SHA_CHANGED')
    random.seed(1);np.random.seed(1);torch.manual_seed(1)
    # Keep official default TF32/determinism settings; do not import the S4
    # training runtime, convert the model to BF16, or inherit its stricter flags.
    build,data_class=primary_functions(args.primary_code)
    model,transform,precision=build(LocalWeights(args.weights))
    require(precision==torch.bfloat16 and all(p.dtype==torch.float32 for p in model.parameters()),
        'OFFICIAL_FP32_PARAMETERS_PLUS_BF16_AUTOCAST_CHANGED')
    model.eval().requires_grad_(False).to('cuda')
    return config,model,transform,data_class

def decode(v):return [x.decode('utf-8') if isinstance(x,bytes) else str(x) for x in np.asarray(v).reshape(-1)]

def inspect_patch(path):
    with h5py.File(path,'r') as h:
        key='img' if 'img' in h else ('imgs' if 'imgs' in h else 'images')
        barcode='barcodes' if 'barcodes' in h else 'barcode'
        n=len(h[key]);require(h[key].shape==(n,224,224,3) and h[key].dtype==np.uint8,'OFFICIAL_RGB_224_UINT8_REQUIRED')
        names=decode(h[barcode][:]);coords=np.asarray(h['coords'][:])
        require(n>0 and len(names)==len(set(names))==n and coords.shape==(n,2)
            and np.isfinite(coords).all(),'OFFICIAL_BARCODE_OR_COORDINATE_ROWS_INVALID')
        numeric={str(k):float(v) for k,v in h[key].attrs.items() if np.asarray(v).ndim==0
                 and np.issubdtype(np.asarray(v).dtype,np.number)}
    return dict(rows=n,barcodes=names,coords=coords,numeric_image_attributes=numeric)

def prepare_first(path,transform,count):
    with h5py.File(path,'r') as h:
        key='img' if 'img' in h else ('imgs' if 'imgs' in h else 'images')
        raw=h[key][:count]
    require(raw.shape[1:]==(224,224,3) and raw.dtype==np.uint8,'PROFILE_TRUE_RGB_SHAPE')
    x=torch.stack([transform(Image.fromarray(v)) for v in raw])
    # Exact image-level no-op witness for Resize(224)+CenterCrop(224).
    from torchvision import transforms
    identity=max(float(np.max(np.abs(np.asarray(transforms.CenterCrop(224)(transforms.Resize(224)(Image.fromarray(v)))).astype(int)-v.astype(int)))) for v in raw)
    require(identity==0.,'RGB_RESIZE_CROP_NOT_IDENTITY')
    return x,identity

def qualify(args):
    require(not args.output.exists(),'EXISTING_FORMAL_QUALIFICATION_PRESERVED')
    config,model,transform,_=setup(args)
    sample=inspect_patch(args.patch)
    with h5py.File(args.patch,'r') as probe:
        origin=str(probe.attrs['official_sample_id'])
        entry=next(v for v in config['patches'] if v['section']==origin)
        require(str(probe.attrs['origin_source_patch_sha256'])==entry['sha256']
            and str(probe.attrs['origin_dataset_revision'])==config['dataset_revision']
            and str(probe.attrs['scope'])=='device_operator_only_first128_correct_official_bench',
            'DEVICE_PROFILE_MUST_USE_AUTHORIZED_OFFICIAL8_SOURCE_SUBSET')
    require(sample['rows']>=128,'FIXED_128_TRUE_RGB_DEVICE_PROFILE_REQUIRED')
    x,identity=prepare_first(args.patch,transform,128)
    with torch.inference_mode():
        q=x[:8].to('cuda').float()
        fp32=model(q)
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):
            bf16=model(q);again=model(q)
        require(bf16.shape==(8,1536) and torch.isfinite(bf16).all()
            and (bf16.float().norm(dim=1)>0).all(),'TRUE_BF16_FORWARD_GATE_FAILED')
        repeat=float((bf16.float()-again.float()).abs().max())
        require(repeat<=1e-5,'BF16_REPEAT_FORWARD_UNSTABLE')
        cosine=torch.nn.functional.cosine_similarity(fp32.float(),bf16.float(),dim=1)
        relative=(fp32.float()-bf16.float()).norm(dim=1)/fp32.float().norm(dim=1)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        full=x.to('cuda').float()
        with torch.amp.autocast('cuda',dtype=torch.bfloat16):profile=model(full)
        torch.cuda.synchronize();seconds=time.monotonic()-started
        require(profile.shape==(128,1536) and torch.isfinite(profile).all()
            and (profile.float().norm(dim=1)>0).all(),'FIXED_BATCH128_PROFILE_FAILED')
    receipt=dict(schema='hest_official_uni2_BF16_device_qualification_v1',status='PASS_REAL_BF16_FIXED_BATCH128_NO_OPTIMIZER',
        scope='DEVICE_OPERATOR_QUALIFICATION_NOT_BENCHMARK_FEATURE',config_sha256=sha(args.config),extractor_sha256=sha(__file__),
        primary_source_SHA256=PRIMARY_SHA,model_weight_sha256=WEIGHT_SHA,profile_RGB_payload_sha256=sha(args.patch),
        profile_official_sample_id=origin,profile_origin_patch_sha256=entry['sha256'],
        input_pixels=[224,224,3],input_RGB_uint8=True,resize_center_crop_max_pixel_error=identity,
        model_parameter_dtype='float32',forward_autocast_dtype='bfloat16',actual_output_dtype=str(bf16.dtype),
        stored_dtype='float32_container_conversion_only_if_needed',raw_feature_dimension=1536,row_L2_normalized=False,
        fixed_seed=1,batch_size=128,num_workers=1,actual_repeat_max_error=repeat,
        FP32_comparison_is_diagnostic_not_an_official_BF16_acceptance_threshold=True,
        current_FP32_BF16_cosine_min=float(cosine.min()),current_FP32_BF16_cosine_mean=float(cosine.mean()),
        current_FP32_BF16_relative_L2_max=float(relative.max()),profile_seconds=seconds,
        profile_patches_per_second=128/seconds,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        runtime=runtime(),new_optimizer_fits=0,new_optimizer_updates=0,benchmark_features_saved=0,
        clinical_RNA_or_survival_files_read=False)
    write(args.output,receipt);return receipt

def extract(args):
    require(not args.output.exists() and not args.output.with_suffix('.json').exists(),'EXISTING_NEW_FEATURE_PRESERVED')
    config,model,transform,data_class=setup(args)
    q=read(args.qualification)
    require(q['status']=='PASS_REAL_BF16_FIXED_BATCH128_NO_OPTIMIZER'
        and q['config_sha256']==sha(args.config) and q['extractor_sha256']==sha(__file__)
        and q['runtime']==runtime(),'FORMAL_FEATURES_REQUIRE_SAME_CODE_CONFIG_AND_QUALIFIED_RUNTIME')
    entries={row['section']:row for row in config['patches']}
    require(args.section in entries,'SECTION_OUTSIDE_FIXED_EIGHT')
    source=entries[args.section]
    require(args.patch.stat().st_size==source['bytes'] and sha(args.patch)==source['sha256'],
        'OFFICIAL_BENCH_PATCH_FILE_BYTES_OR_SHA_CHANGED')
    sample=inspect_patch(args.patch);n=sample['rows']
    data=data_class(str(args.patch),img_transform=transform)
    loader=DataLoader(data,batch_size=128,shuffle=False,num_workers=1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();cursor=0;output_dtypes=set();norm_min=float('inf');norm_max=0.
    try:
        with h5py.File(args.output,'x') as h:
            embeddings=h.create_dataset('embeddings',shape=(n,1536),dtype='float32',chunks=(min(n,128),1536))
            barcodes=h.create_dataset('barcodes',shape=(n,1),dtype=h5py.string_dtype('utf-8'))
            coords=h.create_dataset('coords',shape=(n,2),dtype=sample['coords'].dtype)
            h.attrs.update(official_sample_id=args.section,source_patch_sha256=source['sha256'],
                forward_precision='bfloat16',parameter_dtype='float32',stored_dtype='float32',
                row_L2_normalized=False,feature_recipe_sha256=sha(args.config),model_weight_sha256=WEIGHT_SHA,
                HEST_commit=config['HEST_commit'],TRIDENT_commit=config['TRIDENT_commit'],
                dataset_revision=config['dataset_revision'],coordinates='native_patch_upper_left_pixels',
                RGB_pixels=224,nominal_image_support_um=112.)
            for batch in loader:
                imgs=batch['imgs'].to('cuda').float()
                with torch.inference_mode(),torch.amp.autocast('cuda',dtype=torch.bfloat16):out=model(imgs)
                require(out.shape==(len(imgs),1536) and torch.isfinite(out).all(),'OFFICIAL_FEATURE_FORWARD_INVALID')
                output_dtypes.add(str(out.dtype))
                values=out.float().cpu().numpy() # storage conversion, never FP32 model arithmetic substitution
                names=list(batch['barcodes']);xy=batch['coords'].cpu().numpy()
                end=cursor+len(imgs)
                require(names==sample['barcodes'][cursor:end] and np.array_equal(xy,sample['coords'][cursor:end]),
                    'OFFICIAL_PATCH_ORDER_OR_COORDINATES_CHANGED')
                norms=np.linalg.norm(values,axis=1);require((norms>0).all(),'ZERO_RAW_FEATURE')
                norm_min=min(norm_min,float(norms.min()));norm_max=max(norm_max,float(norms.max()))
                embeddings[cursor:end]=values;barcodes[cursor:end,0]=names;coords[cursor:end]=xy;cursor=end
            require(cursor==n,'OFFICIAL_FEATURE_ROWS_INCOMPLETE')
        torch.cuda.synchronize()
        with h5py.File(args.output,'r') as h:
            require(decode(h['barcodes'][:])==sample['barcodes'] and np.array_equal(h['coords'][:],sample['coords'])
                and h['embeddings'].shape==(n,1536),'SAVED_FEATURE_INTERFACE_OR_ORDER_FAILED')
        receipt=dict(schema='hest_official8_uni2_bf16_feature_v1',status='COMPLETE_OFFICIAL_BF16_RAW_EMBEDDINGS',
            official_sample_id=args.section,task=source['task'],source_dataset_repo=config['dataset_repo'],
            source_dataset_revision=config['dataset_revision'],source_patch_sha256=source['sha256'],source_patch_bytes=source['bytes'],
            model_weight_sha256=WEIGHT_SHA,model_revision=config['model_revision'],config_sha256=sha(args.config),
            extractor_sha256=sha(__file__),qualification_sha256=sha(args.qualification),primary_source_SHA256=PRIMARY_SHA,
            HEST_commit=config['HEST_commit'],TRIDENT_commit=config['TRIDENT_commit'],
            forward_precision='bfloat16',parameter_dtype='float32',stored_dtype='float32',actual_output_dtypes=sorted(output_dtypes),
            row_L2_normalized=False,embeddings_key='embeddings',barcodes_key='barcodes',coords_key='coords',
            rows=n,dimension=1536,barcodes_unique=True,barcodes_and_coordinates_in_original_patch_order=True,
            nominal_RGB_pixels=[224,224,3],nominal_image_support_um=112.,numeric_image_attributes=sample['numeric_image_attributes'],
            coordinates_unit='native_patch_upper_left_pixels_not_micrometers',raw_feature_norm_min=norm_min,raw_feature_norm_max=norm_max,
            batch_size=128,num_workers=1,seed=1,elapsed_seconds=time.monotonic()-started,
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),runtime=runtime(),feature_file_sha256=sha(args.output),
            feature_file_bytes=args.output.stat().st_size,new_optimizer_fits=0,new_optimizer_updates=0,
            clinical_RNA_or_survival_files_read=False,old_FP32_features_overwritten=False)
        write(args.output.with_suffix('.json'),receipt);return receipt
    except BaseException as error:
        write(args.output.with_suffix('.FAILED.json'),dict(status='FAILED_FEATURE_EXTRACTION_PRESERVED_NO_AUTORETRY',
            error_type=type(error).__name__,completed_rows=cursor,new_optimizer_fits=0,new_optimizer_updates=0))
        raise

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=('qualify','extract'),required=True)
    for name in ('config','primary-code','weights','patch','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--qualification',type=Path);p.add_argument('--section')
    a=p.parse_args();v=qualify(a) if a.mode=='qualify' else extract(a)
    print(json.dumps({k:v[k] for k in ('status','rows','dimension','profile_patches_per_second','peak_GPU_allocated_bytes','feature_file_sha256','new_optimizer_fits') if k in v}))

if __name__=='__main__':main()

```

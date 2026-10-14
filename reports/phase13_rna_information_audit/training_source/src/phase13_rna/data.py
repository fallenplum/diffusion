"""Immutable parent inputs; explicit DSS/PFI provenance and split isolation."""
import json,hashlib,warnings,zipfile,xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
import numpy as np
from src.phase10_trimodal.data import load_private,digest,write_json
from src.phase10_trimodal.features import ahash,clinical,scale,CLINICAL

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=Path('/root/phase13_rna_private')
PARENT=Path('/root/phase12_c4_private')
OLD=Path('/root/phase11_semantic_anchor_private')
REPORT=ROOT/'reports/phase13_rna_information_audit'
DATA=Path('/root/phase7a_foundation_headroom_v1/private_dataset.npz')
CONFIG=ROOT/'configs/phase13_rna_information_audit.json'


def config():return json.loads(CONFIG.read_text(encoding='utf8'))


@lru_cache(maxsize=1)
def source_hash():
    paths=sorted((ROOT/'src/phase13_rna').glob('*.py'))
    paths+=[ROOT/'scripts'/n for n in ['run_phase13.py','prepare_phase13.py','gate_phase13.py','schedule_phase13.py']]
    paths+=[CONFIG,ROOT/'tests/test_phase13_rna.py']
    paths+=sorted((ROOT/'src/phase11_semantic').glob('*.py'))+sorted((ROOT/'src/phase12_c4').glob('*.py'))
    paths+=[ROOT/'src/phase10_trimodal'/n for n in ['data.py','features.py','math.py']]
    paths+=[ROOT/'scripts/run_phase11.py',ROOT/'scripts/run_phase12.py',ROOT/'configs/phase12_c4_confirmation.json',ROOT/'configs/phase11_semantic_anchored.json',ROOT/'configs/phase10_molecular_groups.json']
    return hashlib.sha256(''.join(str(p.relative_to(ROOT)).replace('\\','/')+digest(p) for p in paths).encode()).hexdigest()


def cdr_table(path):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        strings=[''.join(el.itertext()) for el in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si',ns)]
        root=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
        records=[]
        for row in root.find('s:sheetData',ns):
            record={}
            for cell in row:
                ref=cell.attrib['r'];column=0
                for character in ref:
                    if character.isalpha():column=column*26+ord(character.upper())-64
                    else:break
                value=cell.find('s:v',ns)
                if value is None:continue
                record[column-1]=strings[int(value.text)] if cell.attrib.get('t')=='s' else value.text
            records.append(record)
    header=records[0]
    return [{header[k]:v for k,v in record.items() if k in header} for record in records[1:]],list(header.values())


def endpoints(d):
    registry=json.loads((REPORT/'SOURCE_REGISTRY.json').read_text())
    if digest(PRIVATE/'endpoints.npz')!=registry['endpoints_npz_sha256']:raise ValueError('endpoint artifact corrupted')
    z=dict(np.load(PRIVATE/'endpoints.npz'))
    if str(z['patient_hash'])!=ahash(d['patient_ids']):raise ValueError('endpoint patient order changed')
    return {ep:{key:z[ep+'_'+key] for key in ['time','event','valid']} for ep in config()['endpoints']}


@lru_cache(maxsize=1)
def load():return load_private(DATA)


def splits(d,endpoint,fold):
    m=json.loads((PARENT/'formal/manifests'/f'R0_128_seed31_fold{fold}.json').read_text())
    ev=endpoints(d)[endpoint];answer={key:np.asarray(m[key],int) for key in ['outer_train','outer_test','inner_train','inner_val']}
    for key in answer:answer[key]=answer[key][ev['valid'][answer[key]]]
    if set(answer['inner_train'])&set(answer['inner_val']) or set(answer['inner_train'])|set(answer['inner_val'])!=set(answer['outer_train']):raise ValueError('inner partition not isolated')
    if set(answer['outer_train'])&set(answer['outer_test']):raise ValueError('outer fold leakage')
    return answer


def parent_features(fold,role):
    stem=PARENT/'preprocessing'/f'fold{fold}_{role}'
    meta=json.loads(stem.with_suffix('.json').read_text());saved=dict(np.load(stem.with_suffix('.npz')))
    if digest(stem.with_suffix('.npz'))!=meta['npz_sha256']:raise ValueError('parent feature hash changed')
    return meta,saved


def features(d,endpoint,fold,role):
    meta,z=parent_features(fold,role);cfg=config();partition=splits(d,endpoint,fold);fit=partition['inner_train' if role=='inner' else 'outer_train']
    arrays={};audit={'endpoint':endpoint,'fold':fold,'role':role,'fit_indices':fit.tolist(),'fit_hash':ahash(fit),'source_fields':CLINICAL,'parent_preprocessing_sha256':digest(PARENT/'preprocessing'/f'fold{fold}_{role}.npz'),'gene_selection_source':'frozen Phase12 TRAIN indices;NO new endpoint outcomes'}
    if endpoint=='DSS':
        if meta['fit_indices']!=fit.tolist():raise ValueError('DSS feature fit drift')
        for name,spec in cfg['genes'].items():arrays[name]=z['f__'+spec['variant']+'__R']
        for key in ['C','S']:arrays[key]=z['f__top__'+key]
    else:
        raw,coding=clinical(d,fit);arrays['C'],cs=scale(raw,fit);arrays['S'],ss=scale(np.asarray(d['static'],float),fit)
        audit['clinical_coding']=coding;audit['CS_state_hash']=ahash(np.r_[cs['mean'],cs['sd'],ss['mean'],ss['sd']])
        for name,spec in cfg['genes'].items():
            indices=np.array(meta['top_indices'] if name=='TOP' else meta['phase12'][spec['variant']+'_indices'],int)
            raw=np.asarray(d['rna_genes'][:,indices],float)
            with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);median=np.nanmedian(raw[fit],axis=0)
            median=np.where(np.isfinite(median),median,0);values=np.where(np.isfinite(raw),raw,median);mean=values[fit].mean(0);sd=values[fit].std(0)
            arrays[name]=(values-mean)/np.where(sd>1e-8,sd,1)
            audit[name+'_RNA_state_hash']=ahash(np.r_[median,mean,sd])
    arrays={k:v.astype(np.float32) for k,v in arrays.items()}
    masks={name:z['gm_'+spec['group']] for name,spec in cfg['genes'].items()};masks['clinical_mask']=z['clinical_mask']
    audit['feature_hashes']={k:ahash(v[fit]) for k,v in arrays.items()};audit['membership_hashes']={k:ahash(v) for k,v in masks.items()}
    return arrays,masks,audit


def standardize(train,test):
    mean=train.mean(0);sd=train.std(0);sd=np.where(sd>1e-8,sd,1)
    return (train-mean)/sd,(test-mean)/sd,mean,sd


def cached_features(d,endpoint,fold,role):
    stem=PRIVATE/'preprocessing'/f'{endpoint}_fold{fold}_{role}'
    audit=json.loads(stem.with_suffix('.json').read_text());z=dict(np.load(stem.with_suffix('.npz')))
    fit=splits(d,endpoint,fold)['inner_train' if role=='inner' else 'outer_train']
    if audit['fit_hash']!=ahash(fit):raise ValueError('cached preprocess fit changed')
    if audit.get('npz_sha256')!=digest(stem.with_suffix('.npz')):raise ValueError('cached preprocess corrupted')
    values={k:v for k,v in z.items() if not k.startswith('mask_')};masks={k[5:]:v for k,v in z.items() if k.startswith('mask_')}
    for k,v in values.items():
        if ahash(v[fit])!=audit['feature_hashes'][k]:raise ValueError('cached training matrix changed')
    return values,masks,audit


def original_cell(gene,seed,fold):
    arm=config()['genes'][gene]['fusion_arm'];stem=f'{arm}_seed{seed}_fold{fold}'
    cell=json.loads((PARENT/'formal/cells'/f'{stem}.json').read_text())
    ck=Path(cell['parent_checkpoint_path']) if cell.get('reuse_parent') else PARENT/'formal/checkpoints'/f'{stem}.pt'
    if digest(ck)!=cell['checkpoint_sha256']:raise ValueError('historical checkpoint changed')
    original=json.loads((OLD/'formal/cells'/f'C4_seed{seed}_fold{fold}.json').read_text()) if gene=='TOP' else cell
    return cell,original,ck,PARENT/'formal/representations'/f'{stem}.npz'

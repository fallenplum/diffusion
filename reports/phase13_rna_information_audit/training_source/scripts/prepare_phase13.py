"""Original TCGA-CDR PFI verification and parent provenance, no model scoring."""
import sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from src.phase13_rna.data import *


def main():
    d=load();cfg=config();REPORT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    if subprocess.run(['git','merge-base','--is-ancestor',cfg['parent_commit'],'HEAD'],cwd=ROOT).returncode:raise RuntimeError('wrong parent ancestry')
    from run_phase12 import sources_hash
    if sources_hash()!=cfg['parent_training_source_hash']:raise RuntimeError('Phase12 training source changed')
    published=json.loads((ROOT/'reports/phase12_c4_confirmation/INDEPENDENT_VALIDATION.json').read_text())
    if published['status']!='PASS' or published['checkpoint_and_provenance']['checkpoint_records_replayed']!=465:raise RuntimeError('parent incomplete')
    file=PRIVATE/'sources/TCGA-CDR-SupplementalTableS1.xlsx';rows,columns=cdr_table(file);mapping={v['bcr_patient_barcode']:v for v in rows if 'bcr_patient_barcode' in v}
    if len(mapping)!=len([v for v in rows if 'bcr_patient_barcode' in v]):raise RuntimeError('duplicate original CDR patient')
    ids=d['patient_ids'].astype(str)
    def number(row,name):
        try:return float(row[name])
        except (KeyError,ValueError):return np.nan
    arrays={};audit={}
    for ep,alias in [('DSS','DSS'),('PFI','PFS')]:
        event=np.array([number(mapping[i],ep) for i in ids]);days=np.array([number(mapping[i],ep+'.time') for i in ids]);time=days/30.417
        valid=np.isfinite(time)&(time>0)&np.isin(event,[0,1]);cached_t=d[alias+'_time'];cached_e=d[alias+'_event']
        finite=np.isfinite(time)&np.isfinite(cached_t);error=float(abs(time[finite]-cached_t[finite]).max())
        if error>1e-5 or not np.array_equal(event[np.isfinite(event)],cached_e[np.isfinite(event)]):raise RuntimeError('original CDR disagreement '+ep)
        if not np.array_equal(valid,d[alias+'_valid']):raise RuntimeError('endpoint validity disagreement '+ep)
        # Cached months are the historical rounding; preserve them exactly.
        arrays.update({ep+'_time':cached_t,ep+'_event':cached_e.astype(int),ep+'_valid':valid})
        audit[ep]={'original_CDR_column':ep,'original_time_column':ep+'.time','cached_alias':alias,'original_units':'days','days_per_month':30.417,'max_month_rounding_error':error,'event_exact':True,'valid_exact':True,'n':int(valid.sum()),'events':int(cached_e[valid].sum()),'excluded_nonpositive_or_missing':int((~valid).sum()),'folds':{str(f):{'n':int((valid&(d['folds']==f)).sum()),'events':int(cached_e[valid&(d['folds']==f)].sum())} for f in cfg['folds']}}
    np.savez_compressed(PRIVATE/'endpoints.npz',patient_hash=ahash(d['patient_ids']),**arrays)
    registry={'status':'PASS','parent_git_sha':cfg['parent_commit'],'parent_training_source_hash':sources_hash(),'dataset_sha256':digest(DATA),'endpoints_npz_sha256':digest(PRIVATE/'endpoints.npz'),'patient_hash':ahash(d['patient_ids']),'fold_hash':ahash(d['folds']),
              'original_CDR_sha256':digest(file),'original_CDR_url':cfg['PFI_original_source'],'PFI_definition_source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC6066282/','endpoint_audit':audit,'RNA_sets':'same Phase12 TRAIN-onlyindices;PFI refitscaler only;frozen groupmasks','missing_proxy':{'complete':309,'missing3024':203,'verified_platform_identity':False}}
    write_json(REPORT/'SOURCE_REGISTRY.json',registry);write_json(REPORT/'ENDPOINT_AUDIT.json',{'status':'PASS',**audit})
    for ep in cfg['endpoints']:
        for fold in cfg['folds']:
            for role in ['inner','outer']:
                values,masks,meta=features(d,ep,fold,role);directory=PRIVATE/'preprocessing';directory.mkdir(exist_ok=True)
                cached=directory/f'{ep}_fold{fold}_{role}.npz';np.savez_compressed(cached,**values,**{'mask_'+k:v for k,v in masks.items()})
                meta['npz_sha256']=digest(cached)
                write_json(directory/f'{ep}_fold{fold}_{role}.json',meta)
    print(json.dumps({'status':'PFI_ORIGINAL_CDR_VERIFIED','endpoints':audit}))


if __name__=='__main__':main()

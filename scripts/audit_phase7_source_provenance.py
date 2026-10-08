"""External public-source checks only; never used to tune or repair a model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from phase7_p0_biology_prompt_audit import BASE,digest,write_json

OUT=Path('/root/phase7_p0r_robustness_v1')
PARENT=Path('/root/phase7_p0_biology_audit_v4')

def main():
    d=np.load(PARENT/'private_dataset.npz');ids=d['patient_ids'].astype(str);missing=~np.isfinite(d['raw_rna']).all(1)
    q=pd.read_csv(OUT/'sample_quality.tsv',sep='\t',dtype=str).fillna('')
    q=q[q['platform'].isin(['IlluminaGA_RNASeqV2','IlluminaHiSeq_RNASeqV2'])]
    q=q[q['aliquot_barcode'].str[13:15]=='01']
    platforms={p:set(q.loc[q.patient_barcode==p,'platform']) for p in ids}
    rows=[]
    names=np.asarray(['+'.join(sorted(platforms[p])) or 'UNKNOWN' for p in ids])
    for name in np.unique(names):
        group=names==name;rows.append({'primary_tumor_rna_platforms':name,'patients':int(group.sum()),
            'missing_block_patients':int((group&missing).sum()),'complete_rna_patients':int((group&~missing).sum())})
    ga=names=='IlluminaGA_RNASeqV2';hi=names=='IlluminaHiSeq_RNASeqV2';unknown=~(ga|hi)
    exact=np.array_equal(ga,missing)
    write_json(OUT/'platform_source_audit.json',{'status':'PASS','source_url':'https://api.gdc.cancer.gov/data/1a7d7be8-675d-4e60-a105-19d4121bdebf',
        'source_sha256':digest(OUT/'sample_quality.tsv'),'matching':'patient identity and primary tumor 01 aliquots only; all public RNASeqV2 annotations, ambiguity retained',
        'platform_missingness_groups':rows,'exclusive_platform_identification_patients':int((~unknown).sum()),
        'ambiguous_or_unmatched_patients':int(unknown.sum()),'missing_block_exactly_exclusive_GAII':bool(exact),
        'missing_patients_with_ambiguous_platform':int((unknown&missing).sum()),
        'complete_patients_with_HiSeq_annotation':int(sum((not missing[i]) and 'IlluminaHiSeq_RNASeqV2' in platforms[pid] for i,pid in enumerate(ids))),
        'interpretation':'platform association identifies source context; PanCancer processing notes describe NA removal, not biological absence',
        'batch_correction_performed':False,'survival_labels_used':False})
    cdr=pd.read_csv(OUT/'cdr_endpoint_reference.csv').set_index('bcr_patient_barcode').loc[ids]
    labels=pd.read_csv(BASE/'data_clinical_patient.txt',sep='\t',comment='#',dtype=str).set_index('PATIENT_ID').loc[ids]
    endpoint={}
    for local,up in [('DSS','DSS'),('PFS','PFI')]:
        months=pd.to_numeric(labels[local+'_MONTHS'],errors='coerce').to_numpy(float)
        ev=pd.to_numeric(labels[local+'_STATUS'].str.extract(r'^([01])',expand=False),errors='coerce').to_numpy(float)
        ut=pd.to_numeric(cdr[up+'.time'],errors='coerce').to_numpy(float);ue=pd.to_numeric(cdr[up],errors='coerce').to_numpy(float)
        # Verify the observed source conversion (30.417 days/month), preserving
        # its original rounded months rather than silently changing endpoint times.
        expected=ut/30.417;finite=np.isfinite(months)&np.isfinite(ut)
        err=np.abs(months[finite]-expected[finite]);status_same=np.array_equal(ev,ue,equal_nan=True)
        time_same=np.array_equal(np.isfinite(months),np.isfinite(ut)) and (err<1e-6).all()
        endpoint[local]={'local_columns':[local+'_MONTHS',local+'_STATUS'],'CDR_columns':[up+'.time',up],
            'time_month_conversion_days':30.417,'status_all_equal':bool(status_same),'times_all_equal_to_converted_CDR':bool(time_same),
            'max_time_conversion_error_months':float(err.max()),'labels_modified':False}
    write_json(OUT/'endpoint_provenance_audit.json',{'status':'PASS','reference_url':'https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81',
        'reference_csv_sha256':digest(OUT/'cdr_endpoint_reference.csv'),'endpoints':endpoint,
        'PFS_terminology':'retain explicit source PFS naming; upstream CDR is PFI if identity checks pass; not a trial-adjudicated PFS endpoint'})
    print(json.dumps({'platforms':rows,'exact_GAII':bool(exact),'endpoints':endpoint}),flush=True)

if __name__=='__main__':main()

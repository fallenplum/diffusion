"""Computational spatial overlaps; explicitly NOT raw histology visual QA."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,h5py
from scipy.spatial import cKDTree
from src.phase16_routing.core import *
def main():
    inv=json.loads((P/'ASSET_INVENTORY.json').read_text());uni=[r for r in inv['h5'] if r['kind']=='UNI2H_CANDIDATE'];rows=[];examples=[];directory=P/'coordinate_maps';directory.mkdir(exist_ok=True)
    for i,con in enumerate(sorted(inv['chosen_conch'],key=lambda r:(r['patient_index'],r['stem']))):
        candidates=[r for r in uni if r['stem']==con['stem'] and r['patient_index']==con['patient_index']]
        if not candidates:rows.append({'slide_number':i+1,'status':'BLOCKED_NO_UNI2'});continue
        u=sorted(candidates,key=lambda r:len(r['path']))[0]
        with h5py.File(con['path']) as h:c=h['coords'][:].reshape(-1,2).astype(float)
        with h5py.File(u['path']) as h:v=h['coords'][:].reshape(-1,2).astype(float)
        ca=con['coordinate_attrs'];ua=u['coordinate_attrs'];cs=ca['patch_size_level0'];us=ua['patch_size_level0'];width=ca['level0_width'];height=ca['level0_height']
        if ca['target_magnification']!=ua['target_magnification'] or ca['level0_magnification']!=ua['level0_magnification']:raise ValueError('magnification mismatch')
        hits=cKDTree(v+us/2).query_ball_point(c+cs/2,(cs+us)/2,p=np.inf);ii=np.repeat(np.arange(len(c)),[len(x) for x in hits]);jj=np.concatenate([np.asarray(x,int) for x in hits]);lo=np.maximum(c[ii],v[jj]);hi=np.minimum(np.minimum(c[ii]+cs,v[jj]+us),[width,height]);area=np.maximum(hi-lo,0).prod(1);keep=area>0;ii,jj,area=ii[keep],jj[keep],area[keep];counts=np.bincount(ii,minlength=len(c));inter=np.bincount(ii,weights=area,minlength=len(c));size=np.maximum(np.minimum(c+cs,[width,height])-c,0).prod(1);coverage=inter/np.maximum(size,1)
        if np.max(coverage)>1+1e-6:raise ValueError('overlapping UNI2 tiles require union area correction')
        np.savez_compressed(directory/f'slide_{i+1:04d}.npz',conch_row=ii,uni2_row=jj,intersection_area=area,conch_h5_sha256=con['sha256'],uni2_h5_sha256=u['sha256'])
        rows.append({'slide_number':i+1,'conch_coordinates_present':True,'uni2_present':True,'uni2_coordinates_present':True,'conch_level0_size':cs,'uni2_level0_size':us,'conch_patches':len(c),'uni2_patches':len(v),'area_overlap_fraction':float(coverage.mean()),'unmatched_fraction':float((counts==0).mean()),'uni2_subpatches_median':float(np.median(counts)),'status':'COMPUTATIONAL_OVERLAP_ONLY_BLOCKED_RAW_WSI_QA'})
        if len(examples)<30:
            j=int(np.random.default_rng(160+i).integers(len(c)));sel=jj[ii==j];examples.append({'example':len(examples)+1,'slide_number':i+1,'CONCH_xy':c[j].tolist(),'CONCH_size':cs,'UNI2_xy':v[sel].tolist(),'UNI2_size':us})
    pd.DataFrame(rows).to_csv(R/'COORD_ALIGNMENT.csv',index=False);write(P/'coordinate_examples.json',examples)
    summary={'status':'BLOCKED_RAW_WSI_VISUAL_QA','computationally_mapped_slides':len(rows),'conch_patches':sum(r.get('conch_patches',0) for r in rows),'uni2_patches_in_paired_slides':sum(r.get('uni2_patches',0) for r in rows),'unmatched_mean_across_slides':float(np.mean([r['unmatched_fraction'] for r in rows])),'area_mean_across_slides':float(np.mean([r['area_overlap_fraction'] for r in rows])),'sampled_coordinate_examples':len(examples),'raw_histology_examples_verified':0,'same_row_matching_used':False,'extra_UNI2_files_not_added_to_cohort':len(uni)-len(rows)}
    write(R/'COORDINATE_AUDIT.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

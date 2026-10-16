"""Independently recompute expression selection and CNA scaling from raw arrays."""
import sys,json,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd
from src.phase17_ssd.data import P,R,molecular_features,parent,write
from validate_phase17_independent import independent_cs

def scaling(x,fit):
    with warnings.catch_warnings():warnings.simplefilter('ignore');median=np.nanmedian(x[fit],axis=0)
    median=np.where(np.isfinite(median),median,0);values=np.where(np.isfinite(x),x,median);mean=values[fit].mean(0);sd=values[fit].std(0)
    return (values-mean)/np.where(sd>1e-8,sd,1),sd

def main():
    d=parent.load();raw=np.asarray(d['rna_genes'],float);symbols=d['rna_gene_symbols'].astype(str);cna_symbols=d['cna_gene_symbols'].astype(str);cloc={v:k for k,v in enumerate(cna_symbols)};sets=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());union=set().union(*map(set,sets.values()));annotated=np.array([s in union for s in symbols]);rows=[]
    for fold in range(1,6):
        parts=parent.splits(d,'DSS',fold)
        for role in ['inner','outer']:
            fit=parts['inner_train' if role=='inner' else 'outer_train'];f,a=molecular_features(d,fold,role);meta,z=parent.parent_features(fold,role)
            values,sd=scaling(raw,fit);eligible=np.flatnonzero((np.isfinite(raw[fit]).mean(0)>=.9)&(sd>1e-8));member=eligible[annotated[eligible]];chosen=member[np.argsort(-sd[member],kind='stable')[:512]]
            if len(chosen)<512:
                other=eligible[~annotated[eligible]];chosen=np.r_[chosen,other[np.argsort(-sd[other],kind='stable')[:512-len(chosen)]]]
            assert np.array_equal(chosen,meta['phase12']['enriched_indices'])
            real=np.array([[s in group for s in symbols[chosen]] for group in sets.values()],float);unsupported=int(np.sum((real==0)&(f['membership']!=0)))
            if unsupported:raise ValueError(('fabricated pathway membership inherited',fold,role,unsupported))
            raw_cna=np.full((512,512),np.nan)
            for k,s in enumerate(symbols[chosen]):
                if s in cloc:raw_cna[:,k]=d['cna_genes'][:,cloc[s]]
            gain,_=scaling(np.maximum(raw_cna,0),fit);loss,_=scaling(np.maximum(-raw_cna,0),fit);c,s=independent_cs(d,fit)
            errors={k:float(abs(a-f[k]).max()) for k,a in [('R',values[:,chosen].astype(np.float32)),('G',gain.astype(np.float32)),('L',loss.astype(np.float32)),('C',c),('S',s)]}
            assert max(errors.values())<1e-6 and np.array_equal(f['RM'].astype(bool),np.isfinite(raw[:,chosen])) and np.array_equal(f['CM'].astype(bool),np.isfinite(raw_cna))
            rows.append({'fold':fold,'role':role,'fit_patients':len(fit),'train_missing_proxy_patients':int((np.isnan(d['raw_rna'][fit]).sum(1)==3024).sum()),'pathway_covered_genes':int((real.sum(0)>0).sum()),'unsupported_memberships':unsupported,'CNA_measured_fraction':float(np.isfinite(raw_cna).mean()),**{k+'_max_error':v for k,v in errors.items()}})
    pd.DataFrame(rows).to_csv(R/'INDEPENDENT_PREPROCESSING.csv',index=False);write(R/'PREPROCESSING_AUDIT.json',{'status':'PASS','fit_local_banks':len(rows),'no_fabricated_membership':True,'all_512_genes_true_members':bool(all(v['pathway_covered_genes']==512 for v in rows)),'sources':'raw RNA/CNA independently recomputed;group JSON;literal trainfit clinical coding;no outcome gene selection'})
    print('INDEPENDENT_PREPROCESSING_PASS',rows,flush=True)

if __name__=='__main__':main()

"""Independent preprocessing/coefficient audit and actual checkpoint replay."""
import sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from validate_phase13_independent import digest,ahash,raw_endpoints,expected_models
from src.phase13_rna.models import RNAOnly
from src.phase13_rna.taps import fusion_taps
from src.phase12_c4.models import C4Model,convert_parent_state
from src.phase12_c4.cache import CachedPatches
P=Path('/root/phase13_rna_private');R=ROOT/'reports/phase13_rna_information_audit';F=P/'formal';PARENT=Path('/root/phase12_c4_private');OLD=Path('/root/phase11_semantic_anchor_private')


def rebuild(d,fit,fold,role,cfg):
    source=json.loads((PARENT/'preprocessing'/f'fold{fold}_{role}.json').read_text());parts=[]
    for field in ['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE']:
        strings=d['clinical_'+field].astype(str)
        if field=='AGE':
            x=np.array([float(v) if v.replace('.','',1).isdigit() else np.nan for v in strings]);x=np.where(np.isfinite(x),x,np.nanmedian(x[fit]));parts.append(x[:,None])
        else:parts.append(np.column_stack([strings==value for value in np.unique(strings[fit])]))
    c=np.concatenate(parts,1).astype(float);cs=c[fit].std(0);cs[cs<1e-8]=1.;s=d['static'].astype(float);ss=s[fit].std(0);ss[ss<1e-8]=1.
    out={'C':((c-c[fit].mean(0))/cs).astype(np.float32),'S':((s-s[fit].mean(0))/ss).astype(np.float32)}
    for gene,spec in cfg['genes'].items():
        selected=np.asarray(source['top_indices'] if gene=='TOP' else source['phase12'][spec['variant']+'_indices'],int);x=d['rna_genes'][:,selected].astype(float)
        median=np.nanmedian(x[fit],axis=0);median=np.where(np.isfinite(median),median,0);x=np.where(np.isfinite(x),x,median);mean=x[fit].mean(0);sd=x[fit].std(0)
        out[gene]=((x-mean)/np.where(sd>1e-8,sd,1)).astype(np.float32)
    return out


def coefficient_gradient(x,t,e,w,penalty):
    risk=x@w;gradient=np.zeros(len(t))
    for value in np.unique(t[e==1]):
        deaths=np.flatnonzero((t==value)&(e==1));at_risk=np.flatnonzero(t>=value);shift=risk[at_risk].max();rr=np.exp(risk[at_risk]-shift);dd=np.exp(risk[deaths]-shift);number=len(deaths)
        for step in range(number):
            denominator=rr.sum()-step/number*dd.sum();gradient[at_risk]+=rr/denominator;gradient[deaths]-=step/number*dd/denominator
        gradient[deaths]-=1
    return x.T@gradient/e.sum()+penalty*w


def base_matrix(f,design,ep,fold,role):
    parts=[];base=design.get('base')
    if base:parts.append(f[base].astype(float) if base in ['C','S'] else np.c_[f['C'],f['S']].astype(float))
    if design.get('rna') and not design.get('off'):
        x=f[design['gene']].astype(float);kind=design['rna']
        if kind=='raw_fixed':n=design['normalization'];x=(x-np.array(n['mean']))/np.array(n['sd'])
        elif kind in ['pca16','residual']:
            state=np.load(P/'transforms'/f'{ep}_{design["gene"]}_fold{fold}_{role}.npz')
            if kind=='pca16':x=(x-state['PCA_mean'])@state['PCA_components'].T
            else:x=(x-np.c_[f['C'],f['S']]@state['CS_RNA_map']-state['residual_mean'])/state['residual_sd']
        parts.append(x)
    return np.concatenate(parts,1)


def producer_taps(d,cfg,gene,seed,fold,kind,ep,preprocessing,endpoints):
    f=preprocessing[ep,fold,'outer'];valid=endpoints[ep]['valid'];tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold))
    if kind=='RNA_ONLY':
        ck=P/'producers'/f'{ep}_{gene}_RNA_ONLY_seed{seed}_fold{fold}_outer.pt';archive=torch.load(ck,map_location='cpu',weights_only=False);model=RNAOnly(archive['clinical_mask'],archive['membership'],seed).cuda().eval();result={};native=None
        for name,state in [('before',archive['initial_state']),('after',archive['state'])]:
            model.load_state_dict(state)
            for role,ix in [('train',tr),('test',te)]:
                with torch.no_grad():risk,taps=model(torch.tensor(f[gene][ix],device='cuda'))
                for key,value in taps.items():result.setdefault(name+'_'+key,{})[role]=value.cpu().numpy()
                if role=='test' and name=='after':native=risk.cpu().numpy()
        del model;torch.cuda.empty_cache();return result,native
    spec=cfg['genes'][gene];cell=json.loads((PARENT/'formal/cells'/f'{spec["fusion_arm"]}_seed{seed}_fold{fold}.json').read_text());ck=Path(cell['parent_checkpoint_path']) if cell.get('reuse_parent') else PARENT/'formal/checkpoints'/f'{spec["fusion_arm"]}_seed{seed}_fold{fold}.pt';archive=torch.load(ck,map_location='cpu',weights_only=False)
    saved=np.load(P/'preprocessing'/f'DSS_fold{fold}_outer.npz');mask=saved['mask_'+gene];cm=saved['mask_clinical_mask'];model=C4Model(cm,mask).cuda().eval();cache=CachedPatches(OLD/'cache',128,True);frozen=preprocessing['DSS',fold,'outer'];frozen={**frozen,'R':frozen[gene]};result={};native=None
    for name,state in [('before',archive['initial_state']),('after',archive['state'])]:
        if gene=='TOP':state=convert_parent_state(state)
        model.load_state_dict(state)
        for role,ix in [('train',np.flatnonzero(d['folds']!=fold)),('test',np.flatnonzero(d['folds']==fold))]:
            risk,taps=fusion_taps(model,d,frozen,cache,ix);keep=valid[ix]
            for key,value in taps.items():result.setdefault(name+'_'+key,{})[role]=value[keep]
            if role=='test' and name=='after':native=risk[keep]
    del model,cache;torch.cuda.empty_cache();return result,native


def main():
    torch.set_num_threads(2);cfg=json.loads((ROOT/'configs/phase13_rna_information_audit.json').read_text());d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));endpoints=raw_endpoints(d,cfg);pre={};pre_rows=[];transform_rows=[]
    for ep in cfg['endpoints']:
        for fold in cfg['folds']:
            for role in ['inner','outer']:
                stem=P/'preprocessing'/f'{ep}_fold{fold}_{role}';meta=json.loads(stem.with_suffix('.json').read_text());fit=np.array(meta['fit_indices'],int);actual=rebuild(d,fit,fold,role,cfg);saved=np.load(stem.with_suffix('.npz'));pre[ep,fold,role]=actual
                for key,value in actual.items():
                    error=float(abs(value-saved[key]).max());pre_rows.append({'endpoint':ep,'fold':fold,'role':role,'feature':key,'max_error':error})
                    if error>2e-6:raise RuntimeError('independent preprocessing mismatch')
                for gene in cfg['genes']:
                    state=np.load(P/'transforms'/f'{ep}_{gene}_fold{fold}_{role}.npz');r=actual[gene][fit].astype(float);a=np.c_[actual['C'],actual['S']][fit].astype(float);mean=r.mean(0)
                    _,_,vh=np.linalg.svd(r-mean,full_matrices=False);components=state['PCA_components'];subspace=float(abs(components.T@components-vh[:16].T@vh[:16]).max());mapping=np.linalg.solve(a.T@a/len(a)+cfg['residualizer_lambda']*np.eye(a.shape[1]),a.T@r/len(a));map_error=float(abs(mapping-state['CS_RNA_map']).max())
                    if abs(mean-state['PCA_mean']).max()>1e-8 or subspace>1e-6 or map_error>1e-6:raise RuntimeError('PCA/residual training fit mismatch')
                    transform_rows.append({'endpoint':ep,'gene':gene,'fold':fold,'role':role,'PCA_subspace_error':subspace,'CS_RNA_map_error':map_error})
    rows=[];faults=[];cache={};failure=[]
    for ep in cfg['endpoints']:
        ev=endpoints[ep];t,e,valid=[ev[k] for k in ['time','event','valid']]
        for fold in cfg['folds']:
            tr=np.flatnonzero(valid&(d['folds']!=fold));te=np.flatnonzero(valid&(d['folds']==fold));f=pre[ep,fold,'outer']
            for modelname in expected_models(cfg):
                for seed in cfg['seeds']:
                    label=f'{ep}_{modelname}_seed{seed}_fold{fold}'
                    try:
                        cell=json.loads((F/'cells'/f'{label}.json').read_text());design=cell['design'];z=np.load(cell['prediction_path']);target=z['risk'];gradient=None
                        if design['family']=='BASE':
                            x=base_matrix(f,design,ep,fold,'outer');w=np.load(cell['checkpoint_path'])['w'];risk=x[te]@w
                            if cell.get('RNA_lambda') not in [None,'OFF']:base_dim=(f[design['base']].shape[1] if design['base'] in ['C','S'] else f['C'].shape[1]+32);penalty=np.r_[np.full(base_dim,cell['lambda']),np.full(len(w)-base_dim,cell['RNA_lambda'])]
                            else:penalty=np.full(len(w),cell['lambda'])
                            gradient=float(abs(coefficient_gradient(x[tr],t[tr],e[tr],w,penalty)).max())
                        else:
                            gene=design['gene'];kind=design['producer_kind'];key=ep,gene,seed,fold,kind
                            if key not in cache:cache[key]=producer_taps(d,cfg,gene,seed,fold,kind,ep,pre,endpoints)
                            taps,native=cache[key]
                            if design['family']=='NEURAL_NATIVE':risk=native
                            else:
                                a=taps[design['tap']]['train'].astype(float);b=taps[design['tap']]['test'].astype(float);n=design['normalization'];mean=a.mean(0);sd=a.std(0);sd=np.where(sd>1e-8,sd,1)
                                if abs(mean-np.array(n['mean'])).max()>2e-6 or abs(sd-np.array(n['sd'])).max()>2e-6:raise RuntimeError('probe normalization not train-local')
                                a=(a-np.array(n['mean']))/np.array(n['sd']);b=(b-np.array(n['mean']))/np.array(n['sd'])
                                if design.get('base'):a=np.c_[f['C'][tr],f['S'][tr],a];b=np.c_[f['C'][te],f['S'][te],b]
                                w=np.load(cell['checkpoint_path'])['w'];risk=b@w;penalty=np.array(n['penalty']);gradient=float(abs(coefficient_gradient(a,t[tr],e[tr],w,penalty)).max())
                        error=float(abs(risk-target).max())
                        if error>2e-6:raise RuntimeError('checkpoint/OOF mismatch')
                        if gradient is not None and gradient>2e-5:raise RuntimeError('independent Cox stationarity')
                        rows.append({'endpoint':ep,'model':modelname,'seed':seed,'fold':fold,'max_error':error,'independent_gradient_max':gradient})
                        if ep=='DSS' and fold==1 and seed==31 and modelname=='HALL_R_fixed1':
                            for name,altered in [('wrong_risk_direction',-risk),('corrupted_coefficients',x[te]@(w+1)),('wrong_patient_OOF',np.roll(risk,1))]:
                                try:
                                    if abs(altered-target).max()>2e-6:raise ValueError('injected mismatch rejected')
                                except ValueError:faults.append({'fault':name,'actually_rejected':True})
                                else:raise RuntimeError('replay fault admitted')
                    except Exception as error:failure.append({'cell':label,'error':str(error)})
            cache.clear()
    pd.DataFrame(pre_rows).to_csv(R/'PREPROCESS_RECONSTRUCTION.csv',index=False);pd.DataFrame(transform_rows).to_csv(R/'PCA_RESIDUAL_FIT_AUDIT.csv',index=False);pd.DataFrame(rows).to_csv(R/'CHECKPOINT_REPLAY.csv',index=False)
    result={'status':'FAIL' if failure else 'PASS','checkpoint_records':len(rows),'expected_records':3330,'independent_preprocessing_matrices':len(pre_rows),'independent_PCA_residual_transforms':len(transform_rows),'faults':faults,'failures':failure}
    (R/'REPLAY_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'replayed':len(rows),'first_failures':failure[:5]}));return 2 if failure else 0


if __name__=='__main__':raise SystemExit(main())

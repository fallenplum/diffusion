"""Separate checkpoint/provenance audit; evaluator does not import this module."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from sklearn.decomposition import PCA
from src.phase11_semantic.models import SemanticModel
from src.phase11_semantic.data import PatchCache
from src.phase11_semantic.train import forward_partition
from validate_phase11_independent import digest,check_manifest

PRIVATE=Path('/root/phase11_semantic_anchor_private');REPORT=ROOT/'reports/phase11_semantic_anchored_trimodal'


def ahash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def independent_features(d,fit):
    fields=['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE'];out={};coding={}
    for ck,names in [('C',fields),('C_min',fields[:3])]:
        parts=[];coding[ck]={}
        for name in names:
            values=d['clinical_'+name].astype(str)
            if name=='AGE':
                v=pd.to_numeric(pd.Series(values),errors='coerce').to_numpy(float);med=np.nanmedian(v[fit]);v=np.where(np.isfinite(v),v,med);parts.append(v[:,None]);coding[ck][name]={'median':float(med)}
            else:
                levels=np.unique(values[fit]);parts.append(np.stack([values==lev for lev in levels],1));coding[ck][name]={'levels':levels.tolist()}
        mat=np.concatenate(parts,1).astype(float);mean=mat[fit].mean(0);sd=mat[fit].std(0);sd[sd<1e-8]=1;out[ck]=((mat-mean)/sd).astype(np.float32)
    s=d['static'].astype(float);mean=s[fit].mean(0);sd=s[fit].std(0);sd[sd<1e-8]=1;out['S']=((s-mean)/sd).astype(np.float32)
    r=d['rna_genes'].astype(float)
    import warnings
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(r[fit],0)
    med[~np.isfinite(med)]=0;r=np.where(np.isfinite(r),r,med);mean=r[fit].mean(0);sd=r[fit].std(0);usable=(np.isfinite(d['rna_genes'][fit]).mean(0)>=.9)&(sd>1e-8)
    eligible=np.flatnonzero(usable);top=eligible[np.argsort(-sd[eligible],kind='stable')[:512]]
    out['R']=((r-mean)/np.where(sd>1e-8,sd,1))[:,top].astype(np.float32)
    t=d['titan'].astype(float);tm=t[fit].mean(0);ts=t[fit].std(0);ts[ts<1e-8]=1;t=(t-tm)/ts;p=PCA(32,svd_solver='full').fit(t[fit]);out['T']=p.transform(t).astype(np.float32)
    return out,coding,top,ahash(np.r_[med,mean,sd])


def preprocess_audit(d):
    rows=[];failure=[];groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text())
    for fold in range(1,6):
        for role in ['inner','outer']:
            stem=PRIVATE/'preprocessing'/f'fold{fold}_{role}';m=json.loads(stem.with_suffix('.json').read_text());z=dict(np.load(stem.with_suffix('.npz')))
            fit=np.array(m['fit_indices'],int);ref,coding,top,rhash=independent_features(d,fit)
            if rhash!=m['rna_fit_hash'] or top.tolist()!=m['top_indices'] or coding['C']!=m['C']['coding'] or coding['C_min']!=m['C_min']['coding']:failure.append({'fold':fold,'role':role,'error':'fit state/source identity'})
            for name,v in ref.items():
                saved=z['f_'+name];err=float(np.max(abs(v.astype(float)-saved.astype(float))))
                rows.append({'fold':fold,'role':role,'feature':name,'max_abs_error':err,'fit_n':len(fit)})
                if err>2e-6:failure.append({'fold':fold,'role':role,'feature':name,'error':err})
            true=z['m_true'];random=z['m_random'];symbols=d['rna_gene_symbols'][top].astype(str)
            gm=np.array([[g in values for g in symbols] for values in groups.values()],np.float32)
            biologic=int((gm.sum(0)>0).sum())
            for j in np.flatnonzero(gm.sum(0)==0):gm[j%8,j]=1
            perm=np.array(m['groups']['permutation']);expected=np.zeros_like(gm);expected[:,perm]=gm
            if not np.array_equal(gm,true) or not np.array_equal(expected,random) or biologic!=m['groups']['biologic_covered'] or not np.array_equal(true@true.T,random@random.T):failure.append({'fold':fold,'role':role,'error':'grouping provenance/overlap'})
    pd.DataFrame(rows).to_csv(REPORT/'PREPROCESS_RECONSTRUCTION.csv',index=False)
    return {'status':'FAIL' if failure else 'PASS','transforms_checked':len(rows),'failures':failure,'method':'independent raw train-only reconstruction; no training feature imports'}


def ridge_gradient(x,t,e,w,lam):
    r=x@w;grad=-x[e==1].sum(0);loss=-r[e==1].sum()
    for time in np.unique(t[e==1]):
        deaths=(t==time)&(e==1);count=int(deaths.sum());high=np.max(r[t>=time]);base=np.exp(r-high)
        for j in range(count):
            weight=(t>=time).astype(float)-j/count*deaths;terms=base*weight;den=terms.sum();loss+=high+np.log(den);grad+=(terms@x)/den
    return grad/e.sum()+lam*w


def replay(d):
    cfg=json.loads((ROOT/'configs/phase11_semantic_anchored.json').read_text(encoding='utf-8-sig'));out=PRIVATE/'formal';rows=[];fail=[]
    caches={b:PatchCache(PRIVATE/'cache',b) for b in [128,512,1024]}
    for cp in sorted((out/'cells').glob('*.json')):
        cell=json.loads(cp.read_text());arm=cell['arm'];seed=cell['seed'];fold=cell['fold'];stem=f'{arm}_seed{seed}_fold{fold}'
        if 'reuse_of' in cell:continue
        try:
            m=json.loads((out/'manifests'/f'{stem}.json').read_text());tr,te,_,_=check_manifest(m,d['folds'],fold)
            z=dict(np.load(PRIVATE/'preprocessing'/f'fold{fold}_outer.npz'));f={k[2:]:v for k,v in z.items() if k.startswith('f_')}
            expected=np.load(out/'oof'/f'{stem}.npz')['risk']
            if arm in ['A0','A1','A2','A3','A4']:
                ck=out/'checkpoints'/f'{stem}.npz'
                if digest(ck)!=cell['checkpoint_sha256']:raise ValueError('Ridge checkpoint hash')
                w=np.load(ck)['w'];names={'A0':['C'],'A1':['C','S'],'A2':['C','T'],'A3':['R'],'A4':['C','S','R']}[arm]
                x=np.concatenate([f[k] for k in names],1).astype(float);risk=x[te]@w;g=ridge_gradient(x[tr],d['DSS_time'][tr],d['DSS_event'][tr],w,cell['lambda'])
                if abs(g).max()>2e-5:raise ValueError('Ridge independent optimality residual')
                gradmax=float(abs(g).max())
            else:
                ck=out/'checkpoints'/f'{stem}.pt'
                if digest(ck)!=cell['checkpoint_sha256']:raise ValueError('neural checkpoint hash')
                archive=torch.load(ck,map_location='cpu',weights_only=False);base=cell['base_arm']
                if archive['arm']!=base or archive['seed']!=seed or archive['epochs']!=cell['selected_epoch'] or archive['fit_hash']!=ahash(tr):raise ValueError('checkpoint container metadata')
                model=SemanticModel(base,archive['clinical_mask'],archive['rna_membership'],cfg['hidden_dim']).cuda().eval();model.load_state_dict(archive['state'])
                cache=caches[cell['patch_budget']]
                with torch.no_grad():risk=forward_partition(model,d,f,cache,te,base,cfg['split_seed']+303,cfg['patient_chunk'])[0].cpu().numpy()
                # These are input contribution properties, NOT causal influence.
                subset=te[:8];p,mask=cache.batch(subset,'cuda');ckey='C_min' if base=='D2' else 'C'
                ri=np.array(m['outer_test_modality_indices']['RNA'])[:8];si=np.array(m['outer_test_modality_indices']['S32'])[:8]
                values=[torch.tensor(f[ckey][subset],device='cuda'),torch.tensor(f['S'][si],device='cuda'),torch.tensor(f['R'][ri],device='cuda'),p,mask]
                with torch.no_grad():
                    original=model(*values);changed=list(values);changed[1]=values[1]*0;sr=float((original-model(*changed)).abs().max())
                    changed=list(values);changed[2]=values[2]*0;rr=float((original-model(*changed)).abs().max())
                    changed=list(values);changed[3]=p.clone();changed[3][~mask]=1e15;pad=float((original-model(*changed)).abs().max())
                if sr==0 or pad!=0:raise ValueError('S32/padding contribution contract')
                if base in ['A5','A6','B6','D3'] and rr!=0:raise ValueError('unexpected RNA prediction path')
                if base not in ['A5','A6','B6','D3'] and rr==0:raise ValueError('disconnected RNA prediction')
                gradmax=None;del model;torch.cuda.empty_cache()
            err=float(np.max(abs(expected-risk)))
            if err>cfg['prediction_replay_atol']:raise ValueError('OOF/checkpoint mismatch '+str(err))
            rows.append({'arm':arm,'seed':seed,'fold':fold,'max_abs_error':err,'independent_Ridge_gradient_max':gradmax,
                         'S32_response':sr if gradmax is None else None,'RNA_response':rr if gradmax is None else None,'padding_response':pad if gradmax is None else None})
        except Exception as ex:fail.append({'arm':arm,'seed':seed,'fold':fold,'error':str(ex)})
    pd.DataFrame(rows).to_csv(REPORT/'CHECKPOINT_REPLAY.csv',index=False)
    return {'status':'FAIL' if fail else 'PASS','actual_fit_checkpoints':len(rows),'failures':fail,'atol':cfg['prediction_replay_atol']}


if __name__=='__main__':
    torch.set_num_threads(2);d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True))
    a=preprocess_audit(d);b=replay(d);result={'preprocessing':a,'checkpoints':b,'status':'PASS' if a['status']==b['status']=='PASS' else 'FAIL'}
    (REPORT/'REPLAY_AND_PROVENANCE_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(0 if result['status']=='PASS' else 2)

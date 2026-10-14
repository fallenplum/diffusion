"""Separate checkpoint, graph-control and train-only preprocessing verifier."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch
from src.phase12_c4.models import C4Model
from src.phase11_semantic.models import SemanticModel
from src.phase11_semantic.data import PatchCache
from src.phase11_semantic.train import forward_partition
from replay_phase11 import independent_features,ridge_gradient
from validate_phase12_independent import check_manifest,digest,ahash
PRIVATE=Path('/root/phase12_c4_private');REPORT=ROOT/'reports/phase12_c4_confirmation'


def assert_risk_matches(risk,expected,tol=2e-6):
    if np.asarray(risk).shape!=np.asarray(expected).shape or not np.isfinite(risk).all() or np.max(abs(np.asarray(risk)-expected))>tol:raise ValueError('checkpoint/OOF risk mismatch or reversed direction')


def graph_contract(model,x,kind,no_anchor=False,backward=False):
    model.eval();model.zero_grad(set_to_none=True);risk=model(*x);details={}
    for name,index in [('S32',1),('RNA',2)]:
        altered=list(x);altered[index]=x[index]*0+33
        diff=float((risk-model(*altered)).abs().max().detach());details[name+'_response']=diff
        if name=='RNA' and kind=='null' and diff!=0:raise ValueError('Null uses real patient RNA')
        if not (name=='RNA' and kind=='null') and diff==0:raise ValueError('lost '+name+' input')
    # Add a masked position even when this real subset has no native padding.
    padded=list(x);padded[3]=torch.cat([x[3],torch.zeros_like(x[3][:,:,:1])],dim=2);padded[4]=torch.cat([x[4],torch.zeros_like(x[4][:,:,:1])],dim=2)
    baseline=model(*padded)
    if float((risk-baseline).abs().max().detach())>2e-6:raise ValueError('extra masked patch changes risk beyond floating tolerance')
    altered=list(padded);altered[3]=padded[3].clone();pattern=torch.linspace(-1e4,1e4,x[3].shape[-1],device=x[3].device);altered[3][~padded[4]]=pattern
    padding=float((baseline-model(*altered)).abs().max().detach());details['padding_response']=padding
    if padding!=0:raise ValueError('padding enters attention')
    if no_anchor and len(list(model.anchor.parameters())):raise ValueError('NoAnchor still has trainable linear risk')
    if backward:
        risk.square().mean().backward()
        encoder=model.rna.encoder
        if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.concept.scalar.parameters()):raise ValueError('S32 module detached')
        parameter=model.rna.constant if kind=='null' else encoder.weight
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().sum()==0:raise ValueError('RNA module detached')
    return details


def independent_rna(d,fit):
    import warnings
    raw=d['rna_genes'].astype(float)
    with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);med=np.nanmedian(raw[fit],axis=0)
    med[~np.isfinite(med)]=0;values=np.where(np.isfinite(raw),raw,med);mean=values[fit].mean(0);sd=values[fit].std(0);missing=np.isnan(raw[fit]).mean(0)
    eligible=np.flatnonzero((missing<=.1)&(sd>1e-8));standard=(values-mean)/np.where(sd>1e-8,sd,1)
    return standard,mean,sd,missing,eligible


def independent_group(gm,mean,sd,missing,seed):
    n=len(mean);mr=np.argsort(np.argsort(mean,kind='stable'),kind='stable')*5//n;vr=np.argsort(np.argsort(sd,kind='stable'),kind='stable')*5//n;levels=np.unique(missing);b=(mr*5+vr)*len(levels)+np.searchsorted(levels,missing)
    rng=np.random.default_rng(seed);perm=np.arange(n)
    for k in np.unique(b):
        order=rng.permutation(np.flatnonzero(b==k))
        if len(order)>1:perm[order]=np.roll(order,1)
    out=np.zeros_like(gm);out[:,perm]=gm;return out,perm


def preprocessing(d,cfg):
    groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());union=set().union(*map(set,groups.values()));symbols=d['rna_gene_symbols'].astype(str);annotated=np.array([s in union for s in symbols]);rows=[]
    for fold in cfg['folds']:
        for role in ['inner','outer']:
            stem=PRIVATE/'preprocessing'/f'fold{fold}_{role}';a=json.loads(stem.with_suffix('.json').read_text());z=dict(np.load(stem.with_suffix('.npz')));fit=np.array(a['fit_indices'],int)
            base,coding,top,state=independent_features(d,fit);standard,mean,sd,missing,eligible=independent_rna(d,fit)
            if state!=a['rna_fit_hash'] or top.tolist()!=a['top_indices']:raise RuntimeError('base fit changed')
            if coding['C']!=a['C']['coding'] or coding['C_min']!=a['C_min']['coding']:raise RuntimeError('clinical coding differs from independent train fit')
            fields=['AGE','SEX','AJCC_PATHOLOGIC_TUMOR_STAGE','PATH_T_STAGE','PATH_N_STAGE','PATH_M_STAGE'];cm=np.zeros((6,base['C'].shape[1]),np.float32);start=0
            for j,name in enumerate(fields):
                size=1 if name=='AGE' else len(coding['C'][name]['levels']);cm[j,start:start+size]=1;start+=size
            if not np.array_equal(cm,z['clinical_mask']):raise RuntimeError('clinical field token mask is not field-local')
            path=eligible[annotated[eligible]];path=path[np.argsort(-sd[path],kind='stable')];enriched=path[:512]
            if len(enriched)<512:
                other=eligible[~annotated[eligible]];other=other[np.argsort(-sd[other],kind='stable')];enriched=np.r_[enriched,other[:512-len(enriched)]]
            if enriched.tolist()!=a['phase12']['enriched_indices']:raise RuntimeError('enriched selection not train-local variance/annotation')
            matched=np.array(a['phase12']['matched_indices']);gm=np.array([[s in values for s in symbols[top]] for values in groups.values()],np.float32)
            # Independently repeat the locked matching algorithm with TRAIN
            # statistics; verifying only the reported gaps would miss test-
            # informed gene choice disguised behind train-local scaling.
            count=len(eligible);mrank=np.argsort(np.argsort(mean[eligible],kind='stable'),kind='stable');vrank=np.argsort(np.argsort(sd[eligible],kind='stable'),kind='stable');mr=mrank/count;vr=vrank/count
            levels=np.unique(missing[eligible]);bb=(mrank*5//count*5+vrank*5//count)*len(levels)+np.searchsorted(levels,missing[eligible])
            loc={int(g):j for j,g in enumerate(eligible)};selected_set=set(enriched);pool=set(int(g) for g in eligible if not annotated[g] and g not in selected_set);rng=np.random.default_rng(cfg['matched_gene_set_seed']);answer=np.empty(512,int)
            for pos in rng.permutation(512):
                target=int(enriched[pos]);j=loc[target];exact=[h for h in sorted(pool) if bb[loc[h]]==bb[j] and missing[h]==missing[target]]
                if exact:h=int(rng.choice(exact))
                else:
                    candidates=[h for h in sorted(pool) if missing[h]==missing[target]]
                    distance=np.array([abs(mr[loc[h]]-mr[j])+abs(vr[loc[h]]-vr[j]) for h in candidates]);best=np.flatnonzero(distance==distance.min());h=int(candidates[int(rng.choice(best))])
                answer[pos]=h;pool.remove(h)
            if not np.array_equal(answer,matched):raise RuntimeError('matched gene selection not independently reproducible from TRAIN statistics')
            for j in np.flatnonzero(gm.sum(0)==0):gm[j%8,j]=1
            for group,seed in enumerate(cfg['group_seeds']):
                ref,perm=independent_group(gm,mean[top],sd[top],missing[top],seed)
                if not np.array_equal(ref,z['gm_g'+str(group)]) or perm.tolist()!=a['phase12']['grouping']['g'+str(group)]['permutation']:raise RuntimeError('group mapping/seed provenance')
            enriched_gm=np.array([[s in values for s in symbols[enriched]] for values in groups.values()],np.float32)
            for j in np.flatnonzero(enriched_gm.sum(0)==0):enriched_gm[j%8,j]=1
            ref,perm=independent_group(enriched_gm,mean[enriched],sd[enriched],missing[enriched],cfg['group_seeds'][0])
            if not np.array_equal(enriched_gm,z['gm_enriched_true']) or not np.array_equal(ref,z['gm_enriched_random']) or not np.array_equal(ref,z['gm_matched_random']):raise RuntimeError('enriched/random structure not matched')
            if len(set(matched))!=512 or annotated[matched].any() or set(matched)&set(enriched) or not np.array_equal(missing[matched],missing[enriched]):raise RuntimeError('matched gene-set support/observation mismatch')
            # Validate every declared strict/fallback match against independent
            # training statistics, rather than trust a coverage label.
            mr=np.argsort(np.argsort(mean[eligible],kind='stable'),kind='stable')/len(eligible);vr=np.argsort(np.argsort(sd[eligible],kind='stable'),kind='stable')/len(eligible);loc={int(g):j for j,g in enumerate(eligible)}
            for record in a['phase12']['matching']:
                j=record['selected_position'];g=int(enriched[j]);h=int(matched[j]);gapm=abs(mr[loc[g]]-mr[loc[h]]);gaps=abs(vr[loc[g]]-vr[loc[h]])
                if abs(gapm-record['mean_rank_gap'])>1e-12 or abs(gaps-record['SD_rank_gap'])>1e-12:raise RuntimeError('matching summary fabricated')
            for variant,indices in [('top',top),('enriched',enriched),('matched',matched)]:
                for key,value in base.items():
                    ref=(standard[:,indices].astype(np.float32) if key=='R' else value);saved=z['f__'+variant+'__'+key];err=float(abs(saved-ref).max());rows.append({'fold':fold,'partition':role,'variant':variant,'feature':key,'max_abs_error':err})
                    if err>2e-6:raise RuntimeError('train-only independent reconstruction mismatch')
    pd.DataFrame(rows).to_csv(REPORT/'PREPROCESS_RECONSTRUCTION.csv',index=False);return len(rows)


def replay(d,cfg):
    rows=[];failure=[];out=PRIVATE/'formal';caches={b:PatchCache('/root/phase11_semantic_anchor_private/cache',b) for b in [128,512,1024]}
    for path in sorted((out/'cells').glob('*.json')):
        cell=json.loads(path.read_text());arm=cell['arm'];seed=cell['seed'];fold=cell['fold'];spec=cfg['arms'][arm];stem=f'{arm}_seed{seed}_fold{fold}'
        try:
            m=json.loads((out/'manifests'/f'{stem}.json').read_text());tr,te,it,iv=check_manifest(m,d,fold,spec);z=dict(np.load(PRIVATE/'preprocessing'/f'fold{fold}_outer.npz'));f={k.split('__',2)[2]:v for k,v in z.items() if k.startswith('f__'+spec['genes']+'__')};expected=np.load(out/'oof'/f'{stem}.npz')['risk']
            ck=Path(cell['parent_checkpoint_path']) if spec.get('reuse') else (out/'checkpoints'/f'{stem}.npz' if spec['kind']=='ridgeRNA' else out/'checkpoints'/f'{stem}.pt')
            if digest(ck)!=cell['checkpoint_sha256']:raise RuntimeError('checkpoint hash changed')
            if spec['kind'] in ['ridge','ridgeRNA']:
                w=np.load(ck)['w'];x=np.c_[f['C'],f['S']].astype(float) if arm=='A1_ref' else f['R'].astype(float);risk=x[te]@w
                if spec.get('reuse'):
                    pc=json.loads((Path('/root/phase11_semantic_anchor_private/formal/cells')/f'{spec["reuse"]}_seed{seed}_fold{fold}.json').read_text());lam=pc['lambda']
                else:lam=cell['lambda']
                grad=ridge_gradient(x[tr],d['DSS_time'][tr],d['DSS_event'][tr],w,lam)
                if abs(grad).max()>2e-5:raise RuntimeError('independent Ridge gradient')
                properties={}
            else:
                archive=torch.load(ck,map_location='cpu',weights_only=False);original=spec.get('reuse')
                if not np.array_equal(archive['clinical_mask'],z['clinical_mask']):raise RuntimeError('checkpoint clinical token mask differs from verified preprocessing')
                if spec['kind']!='legacy_noRNA' and not np.array_equal(archive['rna_membership'],z['gm_'+spec['group']]):raise RuntimeError('checkpoint RNA grouping differs from locked preprocessing')
                if original:
                    model=SemanticModel(original,archive['clinical_mask'],archive['rna_membership'],64).cuda().eval();base=original
                else:
                    if archive['phase12_arm']!=arm or archive['control_kind']!=spec['kind'] or archive['no_anchor']!=spec.get('no_anchor',False) or archive['source_hash']!=cell['source_hash'] or archive['fit_hash']!=ahash(tr):raise RuntimeError('checkpoint control metadata')
                    model=C4Model(archive['clinical_mask'],archive['rna_membership'],64,spec['kind'],spec.get('no_anchor',False)).cuda().eval();base='C4'
                    # Reconstruct wrong inputs solely from saved split-local map.
                    maps=m['effective_modality_mapping'];source=np.array(maps['outer_test']['RNA']);f={**f,'R':f['R'].copy()};f['R'][te]=z['f__'+spec['genes']+'__R'][source]
                model.load_state_dict(archive['state']);cache=caches[spec['budget']]
                with torch.no_grad():risk=forward_partition(model,d,f,cache,te,base,7301+303,64)[0].cpu().numpy()
                properties={}
                if not original:
                    ix=te[:8];p,mask=cache.batch(ix,'cuda');x=[torch.tensor(f['C'][ix],device='cuda'),torch.tensor(f['S'][ix],device='cuda'),torch.tensor(f['R'][ix],device='cuda'),p,mask]
                    properties=graph_contract(model,x,spec['kind'],spec.get('no_anchor',False),True)
                del model;torch.cuda.empty_cache()
            assert_risk_matches(risk,expected);rows.append({'arm':arm,'seed':seed,'fold':fold,'actual_fit':cell['actual_fit'],'max_abs_error':float(abs(risk-expected).max()),**properties})
        except Exception as ex:failure.append({'arm':arm,'seed':seed,'fold':fold,'error':str(ex)})
    pd.DataFrame(rows).to_csv(REPORT/'CHECKPOINT_REPLAY.csv',index=False);return rows,failure


if __name__=='__main__':
    torch.set_num_threads(2);cfg=json.loads((ROOT/'configs/phase12_c4_confirmation.json').read_text());d=dict(np.load('/root/phase7a_foundation_headroom_v1/private_dataset.npz',allow_pickle=True));n=preprocessing(d,cfg);rows,failure=replay(d,cfg)
    result={'status':'FAIL' if failure else 'PASS','preprocessing_arrays_independently_rebuilt':n,'checkpoint_records_replayed':len(rows),'failures':failure};(REPORT/'REPLAY_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(2 if failure else 0)

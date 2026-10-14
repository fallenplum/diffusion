"""Immutable Phase12 training cells and exact parent reuse with private hashes."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from src.phase10_trimodal.data import load_private,digest,write_json
from src.phase10_trimodal.features import ahash
from src.phase10_trimodal.math import fit_ridge
from src.phase11_semantic.train import choose_ridge
from run_phase11 import inner_split,selected_reps,probes
from src.phase12_c4.features import prepare
from src.phase12_c4.train import train,prepare_inputs,forward_partition
from src.phase12_c4.cache import CachedPatches
PRIVATE=Path('/root/phase12_c4_private');PARENT=Path('/root/phase11_semantic_anchor_private');REPORT=ROOT/'reports/phase12_c4_confirmation';CONFIG=ROOT/'configs/phase12_c4_confirmation.json'
DATA=Path('/root/phase7a_foundation_headroom_v1/private_dataset.npz')


def config():
    parent=json.loads((ROOT/'configs/phase11_semantic_anchored.json').read_text(encoding='utf-8-sig'));new=json.loads(CONFIG.read_text())
    for name in ['inner_epochs','minimum_epoch','ema_alpha','learning_rate','weight_decay','gradient_clip','hidden_dim','patient_chunk','ridge_grid','tau']:
        if new[name]!=parent[name]:raise RuntimeError('unregistered schedule difference '+name)
    return {**parent,**new}


def sources_hash():
    paths=sorted((ROOT/'src/phase12_c4').glob('*.py'))+sorted((ROOT/'src/phase11_semantic').glob('*.py'))
    paths+=[ROOT/'src/phase10_trimodal'/n for n in ['data.py','features.py','math.py']]
    paths+=[Path(__file__),ROOT/'scripts/run_phase11.py',CONFIG,ROOT/'configs/phase11_semantic_anchored.json',ROOT/'configs/phase10_molecular_groups.json',ROOT/'tests/test_phase12_c4.py']
    return hashlib.sha256(''.join(str(p.relative_to(ROOT)).replace('\\','/')+digest(p) for p in paths).encode()).hexdigest()


def features_for(d,fold):
    cfg=config();tr=np.flatnonzero(d['folds']!=fold);te=np.flatnonzero(d['folds']==fold);it,iv=inner_split(d,tr,fold);groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());result=[]
    for role,fit in [('inner',it),('outer',tr)]:
        stem=PRIVATE/'preprocessing'/f'fold{fold}_{role}';meta=stem.with_suffix('.json');npz=stem.with_suffix('.npz')
        if meta.exists():
            a=json.loads(meta.read_text())
            if a['source_hash']!=sources_hash() or a['fit_hash']!=ahash(fit) or digest(npz)!=a['npz_sha256']:raise RuntimeError('stale features')
            z=dict(np.load(npz))
            f={variant:{k.split('__',2)[2]:value for k,value in z.items() if k.startswith('f__'+variant+'__')} for variant in ['top','enriched','matched']}
            cm=z['clinical_mask'];gm={k[3:]:v for k,v in z.items() if k.startswith('gm_')}
        else:
            f,cm,gm,a=prepare(d,fit,groups,cfg['group_seeds'],cfg['matched_gene_set_seed']);stem.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(npz,clinical_mask=cm,**{'f__'+variant+'__'+k:v for variant,arrays in f.items() for k,v in arrays.items()},**{'gm_'+k:v for k,v in gm.items()})
            a.update(source_hash=sources_hash(),npz_sha256=digest(npz));write_json(meta,a)
        result.append((f,cm,gm,a))
    return tr,te,it,iv,*result


def run(arm,seed,fold):
    cfg=config();spec=cfg['arms'][arm];gate=json.loads((REPORT/'HARD_GATES.json').read_text())
    if gate['status']!='PASS' or gate['source_hash']!=sources_hash():raise RuntimeError('gate absent/stale')
    out=PRIVATE/'formal'
    for name in ['cells','oof','manifests','checkpoints','representations','probes']:(out/name).mkdir(parents=True,exist_ok=True)
    stem=f'{arm}_seed{seed}_fold{fold}';cellpath=out/'cells'/f'{stem}.json'
    if cellpath.exists():
        old=json.loads(cellpath.read_text())
        if old['status']!='COMPLETE' or old['source_hash']!=sources_hash():raise RuntimeError('stale completed cell')
        print(json.dumps({'skip':stem}),flush=True);return
    d=load_private(DATA);tr,te,it,iv,inner,outer=features_for(d,fold);vi,cmi,gmi,ai=inner;vo,cmo,gmo,ao=outer
    fi=vi[spec['genes']];fo=vo[spec['genes']];mi=gmi[spec['group']];mo=gmo[spec['group']];start=time.monotonic()
    manifest={'outer_train':tr.tolist(),'outer_test':te.tolist(),'inner_train':it.tolist(),'inner_val':iv.tolist(),'patient_hash':ahash(d['patient_ids']),'fold_hash':ahash(d['folds']),
        'preprocess_inner':ai,'preprocess_outer':ao,'spec':spec,'inner_membership_hash':ahash(mi),'outer_membership_hash':ahash(mo)}
    if spec.get('reuse'):
        original=spec['reuse'];parentstem=f'{original}_seed{seed}_fold{fold}';pc=PARENT/'formal/cells'/f'{parentstem}.json';pz=PARENT/'formal/oof'/f'{parentstem}.npz'
        old=json.loads(pc.read_text())
        if digest(pz)!=old['prediction_sha256']:raise RuntimeError('parent risk hash changed')
        z=dict(np.load(pz));risk=z['risk'].copy()
        if not np.array_equal(z['test_indices'],te):raise RuntimeError('parent test identity')
        info={'actual_fit':False,'reuse_parent':original,'parent_record_sha256':digest(pc),'parent_prediction_sha256':digest(pz),'parent_source_hash':old['source_hash'],
            'parent_checkpoint_path':str(PARENT/'formal/checkpoints'/f'{parentstem}.npz' if original in ['A1','A3'] else PARENT/'formal/checkpoints'/f'{parentstem}.pt'),
            'checkpoint_sha256':old['checkpoint_sha256'],'parameters':old['parameters'],'selected_epoch':old['selected_epoch'],'parent_seconds':old['seconds']}
        for folder in ['probes','representations']:
            source=PARENT/'formal'/folder/f'{parentstem}.npz'
            if source.exists():(out/folder/f'{stem}.npz').write_bytes(source.read_bytes())
        if 'probes' in json.loads((PARENT/'formal/manifests'/f'{parentstem}.json').read_text()):manifest['probes']=json.loads((PARENT/'formal/manifests'/f'{parentstem}.json').read_text())['probes']
    elif spec['kind']=='ridgeRNA':
        lam,sel=choose_ridge(fi['R'][it].astype(float),fi['R'][iv].astype(float),d['DSS_time'][it],d['DSS_event'][it],d['DSS_time'][iv],d['DSS_event'][iv],cfg['ridge_grid'])
        w,diag=fit_ridge(fo['R'][tr].astype(float),d['DSS_time'][tr],d['DSS_event'][tr],lam);risk=fo['R'][te].astype(float)@w
        ck=out/'checkpoints'/f'{stem}.npz';np.savez_compressed(ck,w=w)
        info={'actual_fit':True,'parameters':len(w),'lambda':lam,'inner_selection':sel,'optimizer':diag,'selected_epoch':0,'checkpoint_sha256':digest(ck)}
    else:
        ii_features,imaps=prepare_inputs(d,fi,[('inner_train',it,101),('inner_val',iv,202)],spec['kind'],cfg['split_seed'])
        oo_features,omaps=prepare_inputs(d,fo,[('outer_train',tr,101),('outer_test',te,303)],spec['kind'],cfg['split_seed'])
        manifest['effective_modality_mapping']={**imaps,**omaps}
        xa=np.c_[fi['C'],fi['S']].astype(float)
        lam,sel=choose_ridge(xa[it],xa[iv],d['DSS_time'][it],d['DSS_event'][it],d['DSS_time'][iv],d['DSS_event'][iv],cfg['ridge_grid'])
        cache=CachedPatches(PARENT/'cache',spec['budget'],cfg.get('gpu_preload',True))
        model,ii,_=train(d,ii_features,cmi,mi,cache,it,iv,seed,cfg,lam,spec['kind'],spec.get('no_anchor',False))
        ir=selected_reps(model,ii,d,ii_features,cache,it,iv,'C4',cfg) if spec['kind']!='null' else None
        ii.pop('initial_state');del model;torch.cuda.empty_cache();epoch=ii['selected_epoch'];ck=out/'checkpoints'/f'{stem}.pt'
        model,oi,bef=train(d,oo_features,cmo,mo,cache,tr,np.array([],int),seed,cfg,lam,spec['kind'],spec.get('no_anchor',False),epoch,ck)
        model.eval()
        with torch.no_grad():risk,reps,*_=forward_partition(model,d,oo_features,cache,te,'C4',cfg['split_seed']+303,cfg['patient_chunk'])
        risk=risk.cpu().numpy();test_after=reps.cpu().numpy()
        if ir is not None:
            current=copy.deepcopy(model.state_dict());model.load_state_dict(oi['initial_state']);model.eval()
            with torch.no_grad():test_before=forward_partition(model,d,oo_features,cache,te,'C4',cfg['split_seed']+303,cfg['patient_chunk'])[1].cpu().numpy()
            model.load_state_dict(current)
            pr,pmeta=probes(d,ir,dict(zip(['before','after'],bef)),{'before':test_before,'after':test_after},it,iv,tr,te,cfg)
            np.savez_compressed(out/'probes'/f'{stem}.npz',test_indices=te,**pr);manifest['probes']=pmeta
            np.savez_compressed(out/'representations'/f'{stem}.npz',train_indices=tr,test_indices=te,before_train=bef[0],after_train=bef[1],before_test=test_before,after_test=test_after,**{'inner_'+k:v for k,v in ir.items()})
        oi.pop('initial_state');saved=torch.load(ck,map_location='cpu',weights_only=False)
        saved.update(phase12_arm=arm,control_kind=spec['kind'],no_anchor=spec.get('no_anchor',False),gene_mode=spec['genes'],group=spec['group'],source_hash=sources_hash())
        torch.save(saved,ck)
        info={'actual_fit':True,'parameters':oi['parameters'],'selected_epoch':epoch,'inner':ii,'outer':oi,'anchor_lambda':lam,'anchor_selection':sel,
              'anchor_effective':not spec.get('no_anchor',False),'patch_budget':spec['budget'],'checkpoint_sha256':digest(ck),'patch_loader_calls':cache.calls,'patch_loader_seconds_host_submission':cache.load_seconds,'gpu_cache_preload_seconds':getattr(cache,'preload_seconds',0)}
        del model,cache;torch.cuda.empty_cache()
    if len(risk)!=len(te) or not np.isfinite(risk).all():raise RuntimeError('invalid OOF')
    pred=out/'oof'/f'{stem}.npz';np.savez_compressed(pred,test_indices=te,risk=risk,time=d['DSS_time'][te],event=d['DSS_event'][te],arm=arm,seed=seed,fold=fold)
    mp=out/'manifests'/f'{stem}.json';write_json(mp,manifest)
    info.update(status='COMPLETE',arm=arm,seed=seed,fold=fold,spec=spec,source_hash=sources_hash(),protocol_sha256=digest(CONFIG),prediction_sha256=digest(pred),manifest_sha256=digest(mp),seconds=time.monotonic()-start,outer_test_selection=False)
    write_json(cellpath,info);print(json.dumps({'completed':stem,'actual_fit':info['actual_fit'],'selected_epoch':info['selected_epoch'],'seconds':round(info['seconds'],1)}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arms',required=True);ap.add_argument('--seed',type=int,default=31);ap.add_argument('--fold',type=int,default=1);a=ap.parse_args()
    torch.set_num_threads(2)
    for arm in a.arms.split(','):run(arm,a.seed,a.fold)

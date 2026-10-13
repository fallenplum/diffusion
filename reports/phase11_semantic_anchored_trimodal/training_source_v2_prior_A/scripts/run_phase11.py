"""Immutable, resumable Phase11 cells; no outer metric is used for selection."""
import argparse, hashlib, json, os, sys, time, copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np, torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from src.phase10_trimodal.data import load_private,digest,write_json
from src.phase10_trimodal.math import fit_ridge
from src.phase10_trimodal.features import ahash,scale
from src.phase11_semantic.features import prepare
from src.phase11_semantic.data import PatchCache
from src.phase11_semantic.train import train,choose_ridge,anchor_inputs,forward_partition,modality_indices

PRIVATE=Path('/root/phase11_semantic_anchor_private')
REPORT=ROOT/'reports/phase11_semantic_anchored_trimodal'
CONFIG=ROOT/'configs/phase11_semantic_anchored.json'


def sources_hash():
    paths=sorted((ROOT/'src/phase11_semantic').glob('*.py'))+[ROOT/'src/phase10_trimodal'/n for n in ['math.py','features.py','data.py']]
    paths += [Path(__file__),CONFIG,ROOT/'configs/phase10_molecular_groups.json',ROOT/'tests/test_phase11_semantic.py']
    return hashlib.sha256(''.join(str(p.relative_to(ROOT)).replace('\\','/')+digest(p) for p in paths).encode()).hexdigest()


def inner_split(d,tr,fold):
    strata=d['DSS_event'][tr].astype(str)+d['cancer_type'][tr].astype(str)
    a,b=next(StratifiedShuffleSplit(1,test_size=.2,random_state=7301+fold).split(tr,strata))
    return tr[a],tr[b]


def features_for(d,fold):
    tr=np.flatnonzero(d['folds']!=fold);te=np.flatnonzero(d['folds']==fold);it,iv=inner_split(d,tr,fold)
    groups=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text())
    result=[]
    for role,fit in [('inner',it),('outer',tr)]:
        base=PRIVATE/'preprocessing'/f'fold{fold}_{role}'
        meta=base.with_suffix('.json');data=base.with_suffix('.npz')
        if meta.exists():
            a=json.loads(meta.read_text())
            if a['source_hash']!=sources_hash() or a['fit_hash']!=ahash(fit) or digest(data)!=a['npz_sha256']:raise RuntimeError('stale preprocessing')
            z=dict(np.load(data));f={k[2:]:v for k,v in z.items() if k.startswith('f_')};m={k[2:]:v for k,v in z.items() if k.startswith('m_')}
        else:
            f,m,a=prepare(d,fit,groups);base.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(data,**{'f_'+k:v for k,v in f.items()},**{'m_'+k:v for k,v in m.items()})
            a.update({'source_hash':sources_hash(),'npz_sha256':digest(data)})
            write_json(meta,a)
        result.append((f,m,a))
    return tr,te,it,iv,*result


def ridge_x(f,arm):
    keys={'A0':['C'],'A1':['C','S'],'A2':['C','T'],'A3':['R'],'A4':['C','S','R']}[arm]
    return np.concatenate([f[k] for k in keys],1).astype(float)


def selected_reps(model,info,d,f,cache,it,iv,arm,cfg):
    current=copy.deepcopy(model.state_dict());out={}
    for name,state in [('before',info['initial_state']),('after',current)]:
        model.load_state_dict(state);model.eval()
        with torch.no_grad():
            out[name+'_train']=forward_partition(model,d,f,cache,it,arm,cfg['split_seed']+101,cfg['patient_chunk'])[1].cpu().numpy()
            out[name+'_val']=forward_partition(model,d,f,cache,iv,arm,cfg['split_seed']+202,cfg['patient_chunk'])[1].cpu().numpy()
    model.load_state_dict(current);return out


def probes(d,inner,outer_train,outer_test,it,iv,tr,te,cfg):
    out={};meta=[]
    for state in ['before','after']:
        for tap,sl in [('raw',slice(0,64)),('processed',slice(64,128))]:
            for dim in [64,16]:
                xt=inner[state+'_train'][:,sl].astype(float); xv=inner[state+'_val'][:,sl].astype(float)
                xf=outer_train[state][:,sl].astype(float);xe=outer_test[state][:,sl].astype(float)
                mean=xt.mean(0);sd=xt.std(0);sd[sd<1e-8]=1;xt=(xt-mean)/sd;xv=(xv-mean)/sd
                fm=xf.mean(0);fs=xf.std(0);fs[fs<1e-8]=1;xf=(xf-fm)/fs;xe=(xe-fm)/fs
                if dim==16:
                    p=PCA(16,svd_solver='full').fit(xt);xt=p.transform(xt);xv=p.transform(xv)
                    q=PCA(16,svd_solver='full').fit(xf);xf=q.transform(xf);xe=q.transform(xe)
                lam,sel=choose_ridge(xt,xv,d['DSS_time'][it],d['DSS_event'][it],d['DSS_time'][iv],d['DSS_event'][iv],cfg['ridge_grid'])
                w,diag=fit_ridge(xf,d['DSS_time'][tr],d['DSS_event'][tr],lam)
                name=f'{state}_{tap}_{dim}';out[name]=xe@w;meta.append({'probe':name,'lambda':lam,'inner_selection':sel,'optimizer':diag})
    return out,meta


def run(arm,seed,fold):
    cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig'));gate=json.loads((REPORT/'HARD_GATES.json').read_text())
    if gate['status']!='PASS' or gate['source_hash']!=sources_hash():raise RuntimeError('D/M/S gates absent or changed source')
    out=PRIVATE/'formal'
    for name in ['cells','oof','manifests','checkpoints','representations','probes']:(out/name).mkdir(parents=True,exist_ok=True)
    stem=f'{arm}_seed{seed}_fold{fold}';cp=out/'cells'/f'{stem}.json'
    if cp.exists():
        old=json.loads(cp.read_text())
        if old['status']!='COMPLETE' or old['source_hash']!=sources_hash():raise RuntimeError('stale cell')
        print(json.dumps({'reused_completed':stem}),flush=True);return
    d=load_private('/root/phase7a_foundation_headroom_v1/private_dataset.npz')
    tr,te,it,iv,inner,outer=features_for(d,fold);fi,mi,ai=inner;fo,mo,ao=outer
    manifest={'outer_train':tr.tolist(),'outer_test':te.tolist(),'inner_train':it.tolist(),'inner_val':iv.tolist(),
              'preprocess_inner':ai,'preprocess_outer':ao,'patient_hash':ahash(d['patient_ids']),'fold_hash':ahash(d['folds'])}
    start=time.monotonic();aliases=cfg['exact_aliases']
    if arm in aliases:
        original=aliases[arm];base=out/'cells'/f'{original}_seed{seed}_fold{fold}.json'
        if not base.exists():raise RuntimeError('alias before source completed')
        orig=json.loads(base.read_text());risk=np.load(out/'oof'/f'{original}_seed{seed}_fold{fold}.npz')['risk'].copy()
        if orig['source_hash']!=sources_hash():raise RuntimeError('alias stale source')
        info={k:v for k,v in orig.items() if k not in ['arm','seed','fold','status','prediction_sha256','manifest_sha256']}
        info['reuse_of']=original;info['actual_fit']=False;info['reuse_scope']='identical model/input/seed; OOF exact copy'
        if arm=='C0':
            for directory in ['probes','representations']:
                source=out/directory/f'{original}_seed{seed}_fold{fold}.npz'
                if source.exists(): (out/directory/f'{stem}.npz').write_bytes(source.read_bytes())
    elif arm in ['A0','A1','A2','A3','A4']:
        xi=ridge_x(fi,arm);xo=ridge_x(fo,arm)
        lam,sel=choose_ridge(xi[it],xi[iv],d['DSS_time'][it],d['DSS_event'][it],d['DSS_time'][iv],d['DSS_event'][iv],cfg['ridge_grid'])
        w,diag=fit_ridge(xo[tr],d['DSS_time'][tr],d['DSS_event'][tr],lam);risk=xo[te]@w
        ck=out/'checkpoints'/f'{stem}.npz';np.savez_compressed(ck,w=w)
        info={'parameters':len(w),'lambda':lam,'selection':sel,'optimizer':diag,'selected_epoch':0,'actual_fit':True,'checkpoint_sha256':digest(ck)}
    else:
        base_arm,budget=cfg['patch_arms'].get(arm,[arm,128]);cache=PatchCache(PRIVATE/'cache',budget)
        xt=anchor_inputs(fi,d,it,base_arm,cfg['split_seed']+101);xv=anchor_inputs(fi,d,iv,base_arm,cfg['split_seed']+202)
        lam,sel=choose_ridge(xt,xv,d['DSS_time'][it],d['DSS_event'][it],d['DSS_time'][iv],d['DSS_event'][iv],cfg['ridge_grid'])
        model,ii,_=train(d,fi,mi,cache,it,iv,base_arm,seed,cfg,lam)
        ir=selected_reps(model,ii,d,fi,cache,it,iv,base_arm,cfg) if base_arm in ['B0','C1','C2','C3','C4','C5'] and arm not in cfg['patch_arms'] else None
        ii.pop('initial_state');del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
        ep=ii['selected_epoch'];ck=out/'checkpoints'/f'{stem}.pt'
        model,oi,bef= train(d,fo,mo,cache,tr,np.array([],np.int64),base_arm,seed,cfg,lam,ep,ck)
        model.eval()
        with torch.no_grad():risk,reps,*_=forward_partition(model,d,fo,cache,te,base_arm,cfg['split_seed']+303,cfg['patient_chunk'])
        risk=risk.cpu().numpy();test_after=reps.cpu().numpy()
        if ir is not None:
            current=copy.deepcopy(model.state_dict());model.load_state_dict(oi['initial_state']);model.eval()
            with torch.no_grad():test_before=forward_partition(model,d,fo,cache,te,base_arm,cfg['split_seed']+303,cfg['patient_chunk'])[1].cpu().numpy()
            model.load_state_dict(current)
            pr,pmeta=probes(d,ir,dict(zip(['before','after'],bef)),{'before':test_before,'after':test_after},it,iv,tr,te,cfg)
            np.savez_compressed(out/'probes'/f'{stem}.npz',test_indices=te,**pr);manifest['probes']=pmeta
            np.savez_compressed(out/'representations'/f'{stem}.npz',train_indices=tr,test_indices=te,before_train=bef[0],after_train=bef[1],before_test=test_before,after_test=test_after,**ir)
        oi.pop('initial_state');info={'parameters':oi['parameters'],'selected_epoch':ep,'actual_fit':True,'inner':ii,'outer':oi,
                                     'base_arm':base_arm,'patch_budget':budget,'anchor_selection':sel,'anchor_lambda':lam,'checkpoint_sha256':digest(ck)}
        for role,ix,ss in [('inner_train',it,101),('inner_val',iv,202),('outer_train',tr,101),('outer_test',te,303)]:
            ri,si=modality_indices(d,ix,base_arm,cfg['split_seed']+ss);manifest[role+'_modality_indices']={'RNA':ri.tolist(),'S32':si.tolist()}
        del model
    if len(risk)!=len(te) or not np.isfinite(risk).all():raise RuntimeError('invalid OOF')
    pred=out/'oof'/f'{stem}.npz';np.savez_compressed(pred,test_indices=te,risk=risk,time=d['DSS_time'][te],event=d['DSS_event'][te],arm=arm,seed=seed,fold=fold)
    mp=out/'manifests'/f'{stem}.json';write_json(mp,manifest)
    info.update({'status':'COMPLETE','arm':arm,'seed':seed,'fold':fold,'source_hash':sources_hash(),'protocol_sha256':digest(CONFIG),
                 'prediction_sha256':digest(pred),'manifest_sha256':digest(mp),'seconds':time.monotonic()-start,'outer_test_selection':False})
    write_json(cp,info);print(json.dumps({'completed':stem,'epoch':info['selected_epoch'],'seconds':round(info['seconds'],1),'actual_fit':info['actual_fit']}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arms',required=True);ap.add_argument('--seed',type=int,default=31);ap.add_argument('--fold',type=int,default=1);a=ap.parse_args()
    torch.set_num_threads(int(os.environ.get('PHASE11_THREADS','2')))
    for arm in a.arms.split(','):run(arm,a.seed,a.fold)

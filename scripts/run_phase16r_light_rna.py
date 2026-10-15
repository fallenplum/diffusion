"""Uniform convex fits; every inner-validation upstream transform is fit-local."""
import sys,json,time,concurrent.futures,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from sklearn.model_selection import StratifiedKFold
from src.phase16r_light.core import *
from src.phase10_trimodal.math import fit_ridge
from src.phase13_rna.math import block_fit

def pair_parts(t,e,r):
    i,j=np.where((e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0))))
    v=r[i]-r[j];return float(np.where(abs(v)<=1e-8,.5,v>0).sum()),len(i)

def fit(x,t,e,lam):
    if lam=='OFF':return np.zeros(x.shape[1]),{'OFF':True,'gradient_max':0.,'actual_optimizer_fit':False}
    w,info=fit_ridge(x,t,e,float(lam));return w,{**info,'actual_optimizer_fit':True}

def winner(rows):
    best=max(r['score'] for r in rows);tied=[r for r in rows if abs(r['score']-best)<1e-12]
    return next((r for r in tied if r['lambda']=='OFF'),max([r for r in tied if r['lambda']!='OFF'],key=lambda r:r['lambda'],default=tied[0]))

def refs(d,seed,fold,test,outer):
    result={}
    oldp=PREVIOUS/'visual_folds'/f'fold{fold}.pt';old=torch.load(oldp,map_location='cpu',weights_only=False);oldm=old['models']['CS_PLUS_VISUAL__CONCH32__SELECTED'];design=outer['x']['LOCAL']
    if abs(design-old['features']['outer']['CS_PLUS_VISUAL__CONCH32']).max()>1e-10:raise ValueError('previous LOCAL32 feature construction not reproduced')
    if abs(design[test]@oldm['w']-oldm['risk']).max()>1e-10:raise ValueError('previous 0.82592 comparator replay failed')
    result['LOCAL_SINGLE_INNER_REF']={'risk':oldm['risk'],'reference_path':str(oldp),'reference_sha256':sha(oldp),'actual_optimizer_fit':False,'reference_kind':'historical single inner split;not3 independent seeds'}
    path=Path('/root/phase15_complement_private/anchors')/f'DSS_seed{seed}_fold{fold}_outer.pt';anchor=torch.load(path,map_location='cpu',weights_only=False)
    if not np.array_equal(anchor['test'],test) or abs(outer['x']['CS'][test]@anchor['w']-anchor['test_risk']).max()>1e-10:raise ValueError('P15 Clinical/S32 reference replay')
    result['CS_P15_REF']={'risk':anchor['test_risk'],'reference_path':str(path),'reference_sha256':sha(path),'actual_optimizer_fit':False,'reference_kind':'historical nested CS;differentlambda grid'}
    path=parent.PARENT/'formal/oof'/f'R0_128_seed{seed}_fold{fold}.npz';z=np.load(path)
    if not np.array_equal(z['test_indices'],test):raise ValueError('C4 patient order')
    result['C4_REF']={'risk':z['risk'],'reference_path':str(path),'reference_sha256':sha(path),'actual_optimizer_fit':False,'reference_kind':'original C4 includesRNA;architecture-specific training retained'}
    path=PREVIOUS/'folds'/f'fold{fold}.pt';z=torch.load(path,map_location='cpu',weights_only=False);m=z['models']['HALL_PCA8__FIXED1']
    if abs(outer['x']['HALL512']-z['features']['outer']['HALL512']).max()>1e-10:raise ValueError('Hallmark512 feature reproduction')
    result['PCA8_WHITE_REF']={'risk':m['risk'],'reference_path':str(path),'reference_sha256':sha(path),'actual_optimizer_fit':False,'reference_kind':'historical whitenedPCA8 fixed1'}
    return result

def run(seed,fold,visual):
    started=time.time();name=label(seed,fold);cell=P/'cells'/f'{name}.json';artifact=P/'artifacts'/f'{name}.pt'
    if cell.exists():
        old=json.loads(cell.read_text())
        if old['source']!=source() or sha(artifact)!=old['sha256']:raise ValueError('stale baseline/compact cell')
        return
    d=parent.load();ev=parent.endpoints(d)['DSS'];t,e=ev['time'],ev['event'];part=parent.splits(d,'DSS',fold);train,test=part['outer_train'],part['outer_test'];keys=cfg()['clinical_inputs']+cfg()['RNA_inputs'];cv={key:[] for key in keys};inner=[]
    for k,(ii,jj) in enumerate(StratifiedKFold(cfg()['inner_folds'],shuffle=True,random_state=seed+fold*100+cfg()['split_seed_offset']).split(train,e[train])):
        fitidx,valid=train[ii],train[jj];z,path=bank(d,visual,fitidx);wrong,wa=wrong_design(d,z,fitidx,valid,seed+fold*1000+k*11);designs={**z['x'],'WRONG_LOCAL':wrong};trial={}
        for model in keys:
            records=[]
            for lam in cfg()['lambda_grid']:
                w,diag=fit(designs[model][fitidx],t[fitidx],e[fitidx],lam);r=designs[model][valid]@w;n,den=pair_parts(t[valid],e[valid],r);records.append({'lambda':lam,'w':w,'risk':r,'H_num':n,'H_den':den,'optimizer':diag})
            trial[model]=records
        inner.append({'fit':fitidx,'test':valid,'feature_path':str(path),'feature_sha256':sha(path),'wrong':wa,'models':trial});print('INNER_DONE',name,k,flush=True)
    for model in keys:
        for lam in cfg()['lambda_grid']:
            selected=[next(r for r in z['models'][model] if r['lambda']==lam) for z in inner];cv[model].append({'lambda':lam,'score':sum(r['H_num'] for r in selected)/sum(r['H_den'] for r in selected),'H_num':sum(r['H_num'] for r in selected),'H_den':sum(r['H_den'] for r in selected)})
    chosen={model:winner(cv[model])['lambda'] for model in keys};outer,path=bank(d,visual,train);wrong,wa=wrong_design(d,outer,train,test,seed+fold*1000+97);designs={**outer['x'],'WRONG_LOCAL':wrong};models={}
    # Fixed1 is deterministic except the wrong-patient control. Reuse its
    # seed31 record explicitly instead of pretending identical fits are seeds.
    shared=None
    if seed!=31:
        previous=P/'artifacts'/f'{label(31,fold)}.pt';shared=torch.load(previous,map_location='cpu',weights_only=False)
    for model in keys:
        for schedule,lam in [('FIXED1',1.),('CV3',chosen[model])]:
            name_model=model+'__'+schedule
            if schedule=='FIXED1' and model!='WRONG_LOCAL' and shared is not None:
                old=shared['models'][name_model];models[name_model]={**old,'actual_optimizer_fit':False,'reuse_cell':label(31,fold),'reuse_model':name_model};continue
            w,diag=fit(designs[model][train],t[train],e[train],lam);models[name_model]={'w':w,'risk':designs[model][test]@w,'lambda':lam,'optimizer':diag,'actual_optimizer_fit':diag['actual_optimizer_fit'],'design':model,'parameters':len(w)}
    # Select compact family,dimension,lambda exclusively from the inner folds.
    options=[]
    for family in ['PCA','BPCA']:
        for dim in [16,32,64,128]:
            name_rep=family+str(dim)
            for row in cv[name_rep]:options.append({**row,'representation':name_rep,'dimension':dim,'family':family})
    best=max(v['score'] for v in options);tied=[v for v in options if abs(v['score']-best)<1e-12];tied.sort(key=lambda v:(v['lambda']!='OFF',v['dimension'],v['family']!='PCA',-v['lambda'] if v['lambda']!='OFF' else 0));adaptive=tied[0];name_model=adaptive['representation']+'__CV3'
    if models[name_model]['lambda']!=adaptive['lambda']:raise ValueError('adaptive selector lambda inconsistency')
    models['COMPACT_ADAPTIVE']={**models[name_model],'actual_optimizer_fit':False,'reuse_model':name_model,'selection':adaptive}
    for name_model,design,lam in [('CS_AT_LOCAL_LAMBDA','CS',chosen['LOCAL']),('LOCAL_AT_CS_LAMBDA','LOCAL',chosen['CS'])]:
        w,diag=fit(designs[design][train],t[train],e[train],lam);models[name_model]={'w':w,'risk':designs[design][test]@w,'lambda':lam,'optimizer':diag,'actual_optimizer_fit':diag['actual_optimizer_fit'],'design':design,'parameters':len(w)}
    ncs=designs['CS'].shape[1]
    for name_model,design in [('LOCAL_BLOCK','LOCAL'),('WRONG_LOCAL_BLOCK','WRONG_LOCAL')]:
        if chosen['CS']=='OFF':
            wpath,diag=fit(designs[design][train,ncs:],t[train],e[train],1.);w=np.r_[np.zeros(ncs),wpath];penalty=None
        else:penalty=np.r_[np.full(ncs,chosen['CS']),np.ones(32)];w,diag=block_fit(designs[design][train],t[train],e[train],penalty)
        models[name_model]={'w':w,'risk':designs[design][test]@w,'base_lambda':chosen['CS'],'path_lambda':1.,'penalty':penalty,'optimizer':diag,'actual_optimizer_fit':True,'design':design,'parameters':len(w)}
    models.update(refs(d,seed,fold,test,outer));artifact.parent.mkdir(exist_ok=True)
    torch.save({'source':source(),'seed':seed,'fold':fold,'part':part,'inner':inner,'CV':cv,'chosen':chosen,'outer_feature_path':str(path),'outer_feature_sha256':sha(path),'outer_wrong':wa,'models':models,'adaptive':adaptive,'test_time':t[test],'test_event':e[test]},artifact)
    write(cell,{'status':'COMPLETE','source':source(),'sha256':sha(artifact),'models':len(models),'actual_outer_optimizer_fits':sum(m.get('actual_optimizer_fit',False) for m in models.values()),'outer_test_selection':False,'seconds':time.time()-started});print('FORMAL_DONE',label(seed,fold),'seconds',round(time.time()-started,1),flush=True)

def main():
    torch.set_num_threads(2);lock();visual=visual_sources()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(lambda f:run(31,f,visual),cfg()['folds']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:list(pool.map(lambda sf:run(sf[0],sf[1],visual),[(s,f) for s in [47,83] for f in cfg()['folds']]))
    cells=[json.loads(p.read_text()) for p in (P/'cells').glob('*.json')]
    if len(cells)!=15:raise ValueError('incomplete formal matrix')
    result={'status':'PASS','cells':15,'model_slots':sum(c['models'] for c in cells),'actual_outer_optimizer_fits':sum(c['actual_outer_optimizer_fits'] for c in cells),'source':source()};write(P/'TRAINING_DONE.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

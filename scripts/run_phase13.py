"""Complete RNA information diagnosis; immutable cells, strict local fits."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from sklearn.decomposition import PCA
from src.phase13_rna.data import *
from src.phase13_rna.math import fit_ridge,block_fit,choose,choose_block,linear_map
from src.phase13_rna.models import train_rna,extract_rna,RNAOnly
from src.phase13_rna.taps import fusion_taps
from src.phase13_rna.records import save,absent,provenance,stem
from src.phase13_rna.probes import fixed_probe,reference_risk_map
from src.phase12_c4.train import train as fusion_train
from src.phase12_c4.cache import CachedPatches
from src.phase12_c4.models import C4Model,convert_parent_state


def base_inputs(f,name):return f[name] if name in ['C','S'] else np.c_[f['C'],f['S']]


def transforms(f,gene,fit):
    r=f[gene].astype(float);a=base_inputs(f,'CS').astype(float);pca=PCA(16,svd_solver='full').fit(r[fit]);w=linear_map(a[fit],r[fit],config()['residualizer_lambda'])
    residual=r-a@w;mean=residual[fit].mean(0);sd=residual[fit].std(0);sd=np.where(sd>1e-8,sd,1)
    return {'raw':r,'pca16':pca.transform(r),'residual':(residual-mean)/sd},{'PCA_components':pca.components_,'PCA_mean':pca.mean_,'CS_RNA_map':w,'residual_mean':mean,'residual_sd':sd}


def save_transform(endpoint,gene,fold,role,state):
    directory=PRIVATE/'transforms';directory.mkdir(exist_ok=True);p=directory/f'{endpoint}_{gene}_fold{fold}_{role}.npz'
    np.savez_compressed(p,**state);return str(p)


def parent_ridge(gene,seed,fold):
    arm={'TOP':'RNA512_ref','HALL':'RNA_H1','MATCH':'RNA_MR'}[gene];c=json.loads((PARENT/'formal/cells'/f'{arm}_seed{seed}_fold{fold}.json').read_text())
    path=Path(c['parent_checkpoint_path']) if c.get('reuse_parent') else PARENT/'formal/checkpoints'/f'{arm}_seed{seed}_fold{fold}.npz'
    if digest(path)!=c['checkpoint_sha256']:raise RuntimeError('parent RNA-only checkpoint changed')
    previous=json.loads((OLD/'formal/cells'/f'A3_seed{seed}_fold{fold}.json').read_text()) if gene=='TOP' else c
    return np.load(path)['w'],previous['lambda'],str(path)


def baseline_lambda(endpoint,fold,name='CS'):
    c=json.loads((PRIVATE/'formal/cells'/f'{stem(endpoint,name,31,fold)}.json').read_text());return c['lambda']


def base_job(endpoint,fold):
    d=load();cfg=config();part=splits(d,endpoint,fold);it,iv,tr,te=[part[k] for k in ['inner_train','inner_val','outer_train','outer_test']];ev=endpoints(d)[endpoint];t,e=ev['time'],ev['event']
    fi,_,ai=cached_features(d,endpoint,fold,'inner');fo,_,ao=cached_features(d,endpoint,fold,'outer');manifest=provenance(part,ai,ao,d)
    def record(model,w,design,details):
        if not absent(endpoint,model,31,fold):return
        x=design_matrix(fo,design,endpoint,fold,'outer');risk=x[te]@w
        first=None
        for seed in cfg['seeds']:
            first=save(endpoint,model,seed,fold,te,t[te],e[te],risk,manifest,w=w if seed==31 else None,design=design,
                       extra={**details,'actual_fit':details.get('actual_fit',True) if seed==31 else False,'reuse_of':None if seed==31 else stem(endpoint,model,31,fold),'parameters':len(w),'seconds':time.monotonic()-started},reuse_checkpoint=None if seed==31 else first['checkpoint_path'])
    started=time.monotonic()
    for name in ['C','S','CS']:
        if not absent(endpoint,name,31,fold):continue
        xa=base_inputs(fi,name).astype(float);xo=base_inputs(fo,name).astype(float)
        if endpoint=='DSS' and name=='CS':
            cp=OLD/'formal/checkpoints'/f'A1_seed31_fold{fold}.npz';parent=json.loads((OLD/'formal/cells'/f'A1_seed31_fold{fold}.json').read_text());w=np.load(cp)['w'];lam=parent['lambda'];selection=parent.get('selection');diag={'parent_exact':True};actual=False
        else:lam,selection=choose(xa[it],xa[iv],t[it],e[it],t[iv],e[iv],cfg['ridge_grid']);w,diag=fit_ridge(xo[tr],t[tr],e[tr],lam);actual=True
        record(name,w,{'family':'BASE','base':name},{'lambda':lam,'selection':selection,'optimizer':diag,'actual_fit':actual})
    for gene in cfg['genes']:
        ti,si=transforms(fi,gene,it);to,so=transforms(fo,gene,tr)
        for role,state in [('inner',si),('outer',so)]:save_transform(endpoint,gene,fold,role,state)
        for suffix,kind in [('R_tuned','raw'),('PCA16_tuned','pca16'),('RESID_tuned','residual')]:
            name=gene+'_'+suffix
            if not absent(endpoint,name,31,fold):continue
            if endpoint=='DSS' and suffix=='R_tuned':w,lam,cp=parent_ridge(gene,31,fold);selection=[];diag={'parent_exact':True};actual=False
            else:lam,selection=choose(ti[kind][it],ti[kind][iv],t[it],e[it],t[iv],e[iv],cfg['ridge_grid']);w,diag=fit_ridge(to[kind][tr],t[tr],e[tr],lam);actual=True
            record(name,w,{'family':'BASE','gene':gene,'rna':kind},{'lambda':lam,'selection':selection,'optimizer':diag,'actual_fit':actual})
        name=gene+'_R_fixed1'
        if absent(endpoint,name,31,fold):
            risk,w,normalization=fixed_probe(fo[gene][tr],fo[gene][te],t[tr],e[tr],cfg['probe_lambda'])
            record(name,w,{'family':'BASE','gene':gene,'rna':'raw_fixed','normalization':normalization},{'lambda':cfg['probe_lambda'],'optimizer':normalization['optimizer']})
        for baseline,kind,suffix in [('C','raw','C_RAW'),('S','raw','S_RAW'),('CS','raw','CS_RAW'),('CS','pca16','CS_PCA16')]:
            name=gene+'_'+suffix
            if not absent(endpoint,name,31,fold):continue
            ba=base_inputs(fi,baseline).astype(float);bo=base_inputs(fo,baseline).astype(float);base_lambda=baseline_lambda(endpoint,fold,baseline)
            rlam,selection=choose_block(ba[it],ti[kind][it],ba[iv],ti[kind][iv],t[it],e[it],t[iv],e[iv],base_lambda,cfg['RNA_block_grid'])
            if rlam=='OFF':w,diag=fit_ridge(bo[tr],t[tr],e[tr],base_lambda)
            else:w,diag=block_fit(np.c_[bo[tr],to[kind][tr]],t[tr],e[tr],np.r_[np.full(bo.shape[1],base_lambda),np.full(to[kind].shape[1],rlam)])
            record(name,w,{'family':'BASE','base':baseline,'gene':gene,'rna':kind,'off':rlam=='OFF'},
                   {'lambda':base_lambda,'RNA_lambda':rlam,'selection':selection,'optimizer':diag})
        w,_,_=parent_ridge(gene,31,fold) if endpoint=='DSS' else (np.load(json.loads((PRIVATE/'formal/cells'/f'{stem(endpoint,gene+"_R_tuned",31,fold)}.json').read_text())['checkpoint_path'])['w'],None,None)
        reference=to['raw'][te]@w;predicted=(base_inputs(fo,'CS')[te]@so['CS_RNA_map'])@w
        r2=1-np.sum((reference-predicted)**2)/max(np.sum((reference-reference.mean())**2),1e-20)
        directory=PRIVATE/'diagnostics';directory.mkdir(exist_ok=True)
        write_json(directory/f'{endpoint}_{gene}_CS_predictability_fold{fold}.json',{'endpoint':endpoint,'gene':gene,'fold':fold,'n':len(te),'events':int(e[te].sum()),'heldout_reference_RNA_risk_R2':float(r2),'heldout_reference_RNA_risk_correlation':float(np.corrcoef(reference,predicted)[0,1]),'notice':'unsupervised CS->RNA featuremap andTRAIN-onlyRNA Cox;not conditional independence'})
    print('BASE_COMPLETE',endpoint,fold,flush=True)


def design_matrix(f,design,endpoint,fold,role):
    pieces=[]
    if design.get('base'):pieces.append(base_inputs(f,design['base']).astype(float))
    if design.get('rna') and not design.get('off'):
        r=f[design['gene']].astype(float);kind=design['rna']
        if kind=='raw_fixed':
            n=design['normalization'];r=(r-np.array(n['mean']))/np.array(n['sd'])
        elif kind in ['pca16','residual']:
            state=np.load(PRIVATE/'transforms'/f'{endpoint}_{design["gene"]}_fold{fold}_{role}.npz')
            if kind=='pca16':r=(r-state['PCA_mean'])@state['PCA_components'].T
            else:r=(r-base_inputs(f,'CS')@state['CS_RNA_map']-state['residual_mean'])/state['residual_sd']
        pieces.append(r)
    return np.concatenate(pieces,1)


def probes_for(endpoint,gene,seed,fold,producer_kind,part,fi,fo,ai,ao,taps,checkpoint,info,native_risk):
    d=load();cfg=config();ev=endpoints(d)[endpoint];t,e=ev['time'],ev['event'];tr,te=part['outer_train'],part['outer_test'];prefix='RO' if producer_kind=='RNA_ONLY' else 'F'
    producer={'kind':producer_kind,'checkpoint_path':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'selection':info,'producer_coordinate_system':'single inner orouter model,never crossfold stacked'}
    manifest=provenance(part,ai,ao,d,producer)
    raw_cell=json.loads((PRIVATE/'formal/cells'/f'{stem(endpoint,gene+"_R_tuned",31,fold)}.json').read_text());rw=np.load(raw_cell['checkpoint_path'])['w'];rtrain=fo[gene][tr].astype(float)@rw;rtest=fo[gene][te].astype(float)@rw
    name=prefix+'_'+gene+('_HEAD' if prefix=='RO' else '_ORIGINAL')
    if absent(endpoint,name,seed,fold):save(endpoint,name,seed,fold,te,t[te],e[te],native_risk,manifest,reuse_checkpoint=checkpoint,design={'family':'NEURAL_NATIVE','producer_kind':producer_kind,'gene':gene},extra={'actual_fit':prefix=='RO','parameters':info['parameters'],'selected_epoch':info['selected_epoch']})
    diagnostics=[];maps={};base_lambda=baseline_lambda(endpoint,fold)
    for name,representation in taps.items():
        for conditional in [False,True] if name in (cfg['conditional_probe_taps'] if prefix=='RO' else cfg['fusion_conditional_probe_taps']) else [False]:
            model=('CS_' if conditional else '')+prefix+'_'+gene+'_'+name
            if not absent(endpoint,model,seed,fold):continue
            risk,w,normalization=fixed_probe(representation['train'],representation['test'],t[tr],e[tr],cfg['probe_lambda'],
                                             base_inputs(fo,'CS')[tr].astype(float) if conditional else None,base_inputs(fo,'CS')[te].astype(float) if conditional else None,base_lambda)
            design={'family':'PROBE','gene':gene,'producer_kind':producer_kind,'tap':name,'base':'CS' if conditional else None,'normalization':normalization}
            save(endpoint,model,seed,fold,te,t[te],e[te],risk,manifest,w=w,design=design,extra={'actual_fit':True,'parameters':len(w),'lambda':cfg['probe_lambda'],'base_lambda':base_lambda,'selected_epoch':info['selected_epoch'],'optimizer':normalization['optimizer']})
        decoded,diag,ww,mean,sd,ym,ys=reference_risk_map(representation['train'],representation['test'],rtrain,rtest,cfg['probe_lambda'])
        diagnostics.append({'endpoint':endpoint,'gene':gene,'seed':seed,'fold':fold,'producer_kind':producer_kind,'tap':name,'n':len(te),'events':int(e[te].sum()),**diag})
        maps[name+'__w']=ww;maps[name+'__mean']=mean;maps[name+'__sd']=sd;maps[name+'__target_mean']=ym;maps[name+'__target_sd']=ys
    directory=PRIVATE/'diagnostics';directory.mkdir(exist_ok=True)
    write_json(directory/f'{endpoint}_{gene}_{prefix}_seed{seed}_fold{fold}.json',diagnostics);np.savez_compressed(directory/f'{endpoint}_{gene}_{prefix}_reference_maps_seed{seed}_fold{fold}.npz',**maps)


def rna_job(endpoint,gene,seed,fold):
    d=load();cfg=config();part=splits(d,endpoint,fold);it,iv,tr,te=[part[k] for k in ['inner_train','inner_val','outer_train','outer_test']];ev=endpoints(d)[endpoint];t,e=ev['time'],ev['event']
    fi,mi,ai=cached_features(d,endpoint,fold,'inner');fo,mo,ao=cached_features(d,endpoint,fold,'outer');directory=PRIVATE/'producers';directory.mkdir(exist_ok=True)
    label=f'{endpoint}_{gene}_RNA_ONLY_seed{seed}_fold{fold}';meta=directory/f'{label}.json'
    states={};tables={};infos={}
    for role,features,masks,fit,val in [('inner',fi,mi,it,iv),('outer',fo,mo,tr,np.array([],int))]:
        ck=directory/f'{label}_{role}.pt'
        if ck.exists():
            archived=torch.load(ck,map_location='cpu',weights_only=False)
            if archived['source_hash']!=source_hash() or archived['fit_hash']!=ahash(np.asarray(fit,np.int64)):raise RuntimeError('stale RNA producer')
            model=RNAOnly(masks['clinical_mask'],masks[gene],seed).cuda();model.load_state_dict(archived['state']);info=archived['info'];initial=archived['initial_state']
        else:
            model,info=train_rna(features[gene],masks['clinical_mask'],masks[gene],fit,val,t,e,seed,cfg['training'],None if role=='inner' else infos['inner']['selected_epoch']);initial=info.pop('initial_state')
            torch.save({'state':model.state_dict(),'initial_state':initial,'info':info,'clinical_mask':masks['clinical_mask'],'membership':masks[gene],'source_hash':source_hash(),'fit_hash':ahash(np.asarray(fit,np.int64)),'endpoint':endpoint,'gene':gene},ck)
        infos[role]=info;current=copy.deepcopy(model.state_dict());test=iv if role=='inner' else te
        for state,weights in [('before',initial),('after',current)]:
            model.load_state_dict(weights);_,a=extract_rna(model,features[gene],fit);risk,b=extract_rna(model,features[gene],test)
            for name in a:tables[role+'_'+state+'_'+name]={'train':a[name],'test':b[name]}
            if role=='outer' and state=='after':native=risk
        del model;torch.cuda.empty_cache()
    write_json(meta,{'status':'COMPLETE','endpoint':endpoint,'gene':gene,'seed':seed,'fold':fold,'inner':infos['inner'],'outer':infos['outer'],'source_hash':source_hash()})
    taps={name[6:]:value for name,value in tables.items() if name.startswith('outer_')}
    # Cache one useful canonical full representation. Other taps replay from
    # checkpoints; avoid storing all duplicate high-dimensional arrays.
    np.savez_compressed(directory/f'{label}_processed_full_cache.npz',train_indices=tr,test_indices=te,inner_train_indices=it,inner_val_indices=iv,
                        train=tables['outer_after_processed_full']['train'],test=tables['outer_after_processed_full']['test'],inner_train=tables['inner_after_processed_full']['train'],inner_val=tables['inner_after_processed_full']['test'])
    probes_for(endpoint,gene,seed,fold,'RNA_ONLY',part,fi,fo,ai,ao,taps,directory/f'{label}_outer.pt',infos['outer'],native)
    print('RNA_ONLY_COMPLETE',endpoint,gene,seed,fold,flush=True)


def fusion_job(gene,seed,fold,gate_only=False):
    d=load();cfg=config();part=splits(d,'DSS',fold);it,iv,tr,te=[part[k] for k in ['inner_train','inner_val','outer_train','outer_test']]
    fi,mi,ai=cached_features(d,'DSS',fold,'inner');fo,mo,ao=cached_features(d,'DSS',fold,'outer');cell,original,ck,reprfile=original_cell(gene,seed,fold)
    fi={**fi,'R':fi[gene]};fo={**fo,'R':fo[gene]};cache=CachedPatches(OLD/'cache',128,True);directory=PRIVATE/'producers';directory.mkdir(exist_ok=True)
    label=f'DSS_{gene}_FUSION_seed{seed}_fold{fold}';inner=directory/f'{label}_inner.pt';base=json.loads((ROOT/'configs/phase11_semantic_anchored.json').read_text());old=np.load(reprfile)
    if inner.exists():
        archive=torch.load(inner,map_location='cpu',weights_only=False)
        if archive['source_hash']!=source_hash():raise RuntimeError('stale reconstructed inner state')
        ii=archive['info'];model=C4Model(mi['clinical_mask'],mi[gene],64).cuda();model.load_state_dict(archive['state']);initial=archive['initial_state']
    else:
        model,ii,_=fusion_train(d,fi,mi['clinical_mask'],mi[gene],cache,it,iv,seed,base,original['anchor_lambda'])
        initial=ii.pop('initial_state');torch.save({'state':model.state_dict(),'initial_state':initial,'clinical_mask':mi['clinical_mask'],'membership':mi[gene],'info':ii,'source_hash':source_hash()},inner)
    if ii['selected_epoch']!=cell['selected_epoch']:raise RuntimeError('inner epoch reconstruction mismatch')
    selected=copy.deepcopy(model.state_dict());tables={};errors=[]
    for state,weights in [('before',initial),('after',selected)]:
        model.load_state_dict(weights)
        for name,indices in [('train',it),('test',iv)]:
            _,taps=fusion_taps(model,d,fi,cache,indices)
            for key,value in taps.items():tables['inner_'+state+'_'+key+'_'+name]=value
            expected=old['inner_'+state+('_train' if name=='train' else '_val')];actual=np.c_[taps['raw_mean'],taps['processed_mean']];err=float(abs(actual-expected).max());errors.append(err)
            if err>2e-6:raise RuntimeError('reconstructed inner representation differs from old saved values')
    del model;torch.cuda.empty_cache()
    info={'status':'PASS','gene':gene,'seed':seed,'fold':fold,'selected_epoch':ii['selected_epoch'],'reconstruction_max_error':max(errors),'full_inner_epochs':ii['actual_epochs'],'source_hash':source_hash(),'outer_parent_checkpoint':str(ck),'outer_parent_checkpoint_sha256':digest(ck)}
    write_json(directory/f'{label}_reconstruction.json',info)
    if gate_only:del cache;return info
    archive=torch.load(ck,map_location='cpu',weights_only=False);model=C4Model(mo['clinical_mask'],mo[gene],64).cuda();states=[('before',archive['initial_state']),('after',archive['state'])]
    native=None
    for state,weights in states:
        if gene=='TOP':weights=convert_parent_state(weights)
        model.load_state_dict(weights)
        for name,indices in [('train',tr),('test',te)]:
            risk,taps=fusion_taps(model,d,fo,cache,indices)
            if state=='after' and name=='test':native=risk
            for key,value in taps.items():tables['outer_'+state+'_'+key+'_'+name]=value
            expected=old[state+('_train' if name=='train' else '_test')];err=float(abs(np.c_[taps['raw_mean'],taps['processed_mean']]-expected).max())
            if err>2e-6:raise RuntimeError('final representation replay mismatch')
    expected=np.load(PARENT/'formal/oof'/f'{cfg["genes"][gene]["fusion_arm"]}_seed{seed}_fold{fold}.npz')['risk']
    if np.max(abs(expected-native))>2e-6:raise RuntimeError('final native risk replay mismatch')
    del model,cache;torch.cuda.empty_cache()
    np.savez_compressed(directory/f'{label}_processed_full_cache.npz',train_indices=tr,test_indices=te,inner_train_indices=it,inner_val_indices=iv,
                        train=tables['outer_after_processed_full_train'],test=tables['outer_after_processed_full_test'],inner_train=tables['inner_after_processed_full_train'],inner_val=tables['inner_after_processed_full_test'])
    for endpoint in cfg['endpoints']:
        local=splits(d,endpoint,fold);newfi,_,newai=cached_features(d,endpoint,fold,'inner');newfo,_,newao=cached_features(d,endpoint,fold,'outer')
        ti=np.searchsorted(np.sort(tr),local['outer_train']);tj=np.searchsorted(np.sort(te),local['outer_test'])
        # Parent partitions are saved in sorted outer order; do not align by
        # array row when an endpoint excludes patients.
        if not np.array_equal(tr,np.sort(tr)) or not np.array_equal(te,np.sort(te)):raise RuntimeError('unexpected unsorted outer IDs')
        taps={state+'_'+key:{'train':tables['outer_'+state+'_'+key+'_train'][ti],'test':tables['outer_'+state+'_'+key+'_test'][tj]} for state in ['before','after'] for key in cfg['fusion_taps']}
        probes_for(endpoint,gene,seed,fold,'FUSION_DSS_FROZEN',local,newfi,newfo,newai,newao,taps,ck,{'parameters':original['parameters'],'selected_epoch':cell['selected_epoch'],'source_endpoint':'DSS','reconstruction':info},native[tj])
    print('FUSION_TAPS_COMPLETE',gene,seed,fold,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['BASE','RNA','FUSION'],required=True);ap.add_argument('--endpoint',default='DSS');ap.add_argument('--gene',default='HALL');ap.add_argument('--seed',type=int,default=31);ap.add_argument('--fold',type=int,default=1);a=ap.parse_args()
    torch.set_num_threads(2);gate=json.loads((REPORT/'HARD_GATES.json').read_text())
    if gate['status']!='PASS' or gate['source_hash']!=source_hash():raise RuntimeError('gate absent/stale')
    if a.family=='BASE':base_job(a.endpoint,a.fold)
    elif a.family=='RNA':rna_job(a.endpoint,a.gene,a.seed,a.fold)
    else:fusion_job(a.gene,a.seed,a.fold)


if __name__=='__main__':main()

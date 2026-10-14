"""Read-only historical reproduction and data audit before any Phase12 fit."""
import json,sys,subprocess,platform,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,pandas as pd,torch,h5py
from scipy.optimize import minimize
from run_phase12 import PRIVATE,PARENT,REPORT,DATA,config,features_for,sources_hash
from src.phase10_trimodal.data import load_private,digest,write_json
from src.phase10_trimodal.features import ahash
from src.phase11_semantic.data import PatchCache,nested_order
from src.phase11_semantic.models import SemanticModel
from src.phase11_semantic.train import forward_partition
from src.phase12_c4.models import C4Model,convert_parent_state
from replay_phase11 import independent_features,ridge_gradient,ridge_objective
from validate_phase11_independent import check_record


def historical_audit(d,cfg):
    rows=[];independent=[];refits=[];fresh={};cache=PatchCache(PARENT/'cache',128)
    for fold in cfg['folds']:
        tr,te,it,iv,inner,outer=features_for(d,fold);vi,cmi,gmi,ai=inner;vo,cmo,gmo,ao=outer
        for role,arrays,a,fit in [('inner',vi,ai,it),('outer',vo,ao,tr)]:
            expected,coding,top,state=independent_features(d,fit)
            for key,value in expected.items():
                err=float(abs(value-arrays['top'][key]).max());independent.append({'fold':fold,'partition':role,'feature':key,'max_abs_error':err})
                if err>2e-6:raise RuntimeError('historical independent feature mismatch')
            if top.tolist()!=a['top_indices'] or state!=a['rna_fit_hash']:raise RuntimeError('historical train state mismatch')
            old=dict(np.load(PARENT/'preprocessing'/f'fold{fold}_{role}.npz'))
            if not np.array_equal(old['m_random'],(gmi if role=='inner' else gmo)['g0']):raise RuntimeError('g0 membership mismatch')
        f=vo['top']
        for arm in ['A1','A6','A3','C3','C4']:
            for seed in cfg['seeds']:
                stem=f'{arm}_seed{seed}_fold{fold}';cell=json.loads((PARENT/'formal/cells'/f'{stem}.json').read_text());z=dict(np.load(PARENT/'formal/oof'/f'{stem}.npz'));m=json.loads((PARENT/'formal/manifests'/f'{stem}.json').read_text())
                expected_source=json.loads((ROOT/'reports/phase11_semantic_anchored_trimodal/P2_SOURCE_AMENDMENT.json').read_text())['prior_source_hash'] if arm.startswith('A') else json.loads((ROOT/'reports/phase11_semantic_anchored_trimodal/HARD_GATES.json').read_text())['source_hash']
                check_record(cell,z,m,d,arm,seed,fold,expected_source,digest(ROOT/'configs/phase11_semantic_anchored.json'))
                if digest(PARENT/'formal/oof'/f'{stem}.npz')!=cell['prediction_sha256']:raise RuntimeError('parent OOF hash')
                if arm in ['A1','A3']:
                    ck=PARENT/'formal/checkpoints'/f'{stem}.npz';w=np.load(ck)['w'];x=np.c_[f['C'],f['S']].astype(float) if arm=='A1' else f['R'].astype(float)
                    risk=x[te]@w;gradient=ridge_gradient(x[tr],d['DSS_time'][tr],d['DSS_event'][tr],w,cell['lambda'])
                    if abs(gradient).max()>2e-5:raise RuntimeError('parent ridge optimality')
                    if arm=='A1':
                        if fold not in fresh:
                            fit=minimize(lambda v:ridge_objective(x[tr],d['DSS_time'][tr],d['DSS_event'][tr],v,cell['lambda']),np.zeros(x.shape[1]),jac=True,method='BFGS',options={'maxiter':3000,'gtol':1e-10});fresh[fold]=fit.x
                        gg=ridge_gradient(x[tr],d['DSS_time'][tr],d['DSS_event'][tr],fresh[fold],cell['lambda']);bound=(np.linalg.norm(gg)+np.linalg.norm(gradient))/cell['lambda']
                        if abs(gg).max()>2e-7 or np.linalg.norm(fresh[fold]-w)>bound+1e-7:raise RuntimeError('independent A1 refit')
                        refits.append({'seed':seed,'fold':fold,'risk_error':float(abs(x[te]@fresh[fold]-z['risk']).max()),'coefficient_L2_error':float(np.linalg.norm(fresh[fold]-w)),'strong_convexity_bound':float(bound)})
                    adapter_error=None
                else:
                    ck=PARENT/'formal/checkpoints'/f'{stem}.pt';state=torch.load(ck,map_location='cpu',weights_only=False)
                    model=SemanticModel(arm,state['clinical_mask'],state['rna_membership'],cfg['hidden_dim']).cuda().eval();model.load_state_dict(state['state'])
                    with torch.no_grad():risk=forward_partition(model,d,f,cache,te,arm,cfg['split_seed']+303,cfg['patient_chunk'])[0].cpu().numpy()
                    adapter_error=None
                    if arm in ['C3','C4']:
                        adapter=C4Model(state['clinical_mask'],state['rna_membership'],cfg['hidden_dim']).cuda().eval();adapter.load_state_dict(convert_parent_state(state['state']))
                        with torch.no_grad():adapted=forward_partition(adapter,d,f,cache,te,'C4',cfg['split_seed']+303,cfg['patient_chunk'])[0].cpu().numpy()
                        adapter_error=float(abs(adapted-risk).max());del adapter
                        if adapter_error>2e-6:raise RuntimeError('new adapter changes old forward')
                    del model;torch.cuda.empty_cache()
                if digest(ck)!=cell['checkpoint_sha256']:raise RuntimeError('parent checkpoint mutated')
                err=float(abs(risk-z['risk']).max())
                if err>2e-6:raise RuntimeError('STOP: Phase11 checkpoint cannot reproduce OOF')
                rows.append({'arm':arm,'seed':seed,'fold':fold,'OOF_file_sha256':cell['prediction_sha256'],'checkpoint_sha256':cell['checkpoint_sha256'],'max_risk_error':err,'new_adapter_error':adapter_error})
    pd.DataFrame(rows).to_csv(REPORT/'PHASE11_REPLAY.csv',index=False);pd.DataFrame(independent).to_csv(REPORT/'PARENT_PREPROCESS_RECONSTRUCTION.csv',index=False);pd.DataFrame(refits).to_csv(REPORT/'A1_INDEPENDENT_REFIT.csv',index=False)
    return {'status':'PASS','checkpoint_records':len(rows),'independent_feature_matrices':len(independent),'A1_fresh_unique_refits':len(fresh),'tolerance':2e-6}


def data_audit(d):
    manifest=json.loads((PARENT/'cache/data_manifest.json').read_text());paths={'dataset':DATA,'patch':PARENT/'cache/patch1024.npy','coords':PARENT/'cache/coords1024.npy','slides':PARENT/'cache/private_slide_manifest.json'}
    expected={'dataset':'dataset_sha256','patch':'patch_sha256','coords':'coordinates_sha256','slides':'private_slide_manifest_sha256'}
    for name,path in paths.items():
        if digest(path)!=manifest[expected[name]]:raise RuntimeError('historical data/cache mutated '+name)
    source=pd.read_csv('/root/phase10_trimodal_private/cache/private_cohort_manifest.csv').query("status=='EXACT_SLIDE_MATCH'");delivery={r.filename:r.delivery_relative_path for r in source.itertuples(index=False)}
    rows=json.loads(paths['slides'].read_text());patch=np.load(paths['patch'],mmap_mode='r');coords=np.load(paths['coords'],mmap_mode='r');counts={str(b):0 for b in [128,512,1024]}
    for row in rows:
        file=Path('/root/autodl-tmp/phase10_conch')/delivery[row['filename']]
        if digest(file)!=row['h5_sha256']:raise RuntimeError('source H5 changed')
        with h5py.File(file,'r') as h:
            xy=h['coords'][:];x=h['features'][:]
            if not np.isfinite(x).all() or x.shape!=(len(xy),768) or xy.shape!=(len(x),2) or not np.issubdtype(xy.dtype,np.integer) or (xy<0).any() or len(np.unique(xy,axis=0))!=len(xy):raise RuntimeError('H5 data gate')
            if h['features'].attrs['encoder']!='conch_v15' or h['coords'].attrs['patch_size']!=512 or h['coords'].attrs['target_magnification']!=20:raise RuntimeError('H5 encoder/grid')
            sel=nested_order(xy)
            if not np.array_equal(sel,np.array(row['selected_original_indices'])):raise RuntimeError('nested sampler provenance')
            j=row['offset'];n=row['selected']
            if not np.array_equal(x[sel],patch[j:j+n]) or not np.array_equal(xy[sel],coords[j:j+n]):raise RuntimeError('cached features/coords differ')
            for b in counts:counts[b]+=min(int(b),len(sel))
        if row['fold']!=d['folds'][row['patient_index']]:raise RuntimeError('slide crosses patient fold')
    fixed=pd.read_csv('/root/phase10_trimodal_private/cache/private_fixed_patient_folds.csv');idcol=next(c for c in fixed if 'patient' in c.lower());fc=next(c for c in fixed if 'fold' in c.lower());mapping=dict(zip(fixed[idcol].astype(str),fixed[fc].astype(int)))
    if not np.array_equal([mapping[p] for p in d['patient_ids'].astype(str)],d['folds']):raise RuntimeError('original patient folds')
    return {'status':'PASS','patients':512,'DSS_events':65,'WSIs':len(rows),'source_H5_and_cache_elementwise_checked':len(rows),'nested_budgets':counts,'data_hashes':{n:digest(p) for n,p in paths.items()},'patient_hash':ahash(d['patient_ids']),'fold_hash':ahash(d['folds']),'shared_cache':'read-only,no duplication','platform_identity':'missing3024 proxy only,not confirmed sequencing platform'}


def main():
    torch.set_num_threads(2);cfg=config();REPORT.mkdir(parents=True,exist_ok=True);PRIVATE.mkdir(parents=True,exist_ok=True)
    commit=cfg['parent_commit']
    # The first run started at exactly this commit. A published Phase12 checkout
    # is its descendant; provenance must remain verifiable there too.
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=commit and subprocess.run(['git','merge-base','--is-ancestor',commit,head],cwd=ROOT).returncode:
        raise RuntimeError('Phase11 parent is not an ancestor of this checkout')
    references=[]
    names=['RESULTS.md','DECISION.md','BUG_LOG.csv','PROTOCOL_LOCK.json','MODEL_MATRIX.csv','INDEPENDENT_AUDIT.md','KNOWN_LIMITATIONS.md','SOURCE_REGISTRY.json']
    refs=[ROOT/'reports/phase11_semantic_anchored_trimodal'/name for name in names]+list((ROOT/'src/phase11_semantic').glob('*.py'))
    for p in refs:
        blob=subprocess.check_output(['git','show',cfg['parent_commit']+':'+str(p.relative_to(ROOT))],cwd=ROOT)
        if hashlib.sha256(blob).hexdigest()!=digest(p):raise RuntimeError('parent source/report changed '+str(p.relative_to(ROOT)))
        references.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p),'ref':commit,'read_bytes':len(blob)})
    d=load_private(DATA);print('START_PARENT_DATA_AUDIT',flush=True);data=data_audit(d);write_json(REPORT/'DATA_AUDIT.json',data)
    print('START_PARENT_CHECKPOINT_REPLAY',flush=True);parent=historical_audit(d,cfg);write_json(REPORT/'PARENT_HARD_GATE.json',{'status':'PASS','data':data,'reproduction':parent,'source_hash':sources_hash()})
    # Current source dictionary must agree with historical MMP column registry.
    dictionary=json.loads((ROOT/'configs/phase10_molecular_groups.json').read_text());csv=ROOT/'src/phase_m2/metadata/mmp_hallmarks_signatures.csv';table=pd.read_csv(csv)
    agreement={k:set(table[k].dropna().astype(str))==set(v) for k,v in dictionary.items()}
    if not all(agreement.values()):raise RuntimeError('frozen Hallmark registry mismatch')
    import importlib.metadata as md
    registry={'parent_git_sha':commit,'references':references,'data':data,'source_hash':sources_hash(),'python':sys.version,'platform':platform.platform(),'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(),'dependencies':{k:md.version(k) for k in ['torch','numpy','scipy','pandas','scikit-learn','h5py','scikit-survival']},
        'Hallmark':{'dictionary_sha256':digest(ROOT/'configs/phase10_molecular_groups.json'),'source_MMP_CSV_sha256':digest(csv),'column_sets_exact':agreement,'reference':'https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp','scope':'same8 historical frozen sets,not a current MSigDB update'},'timestamp_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write_json(REPORT/'SOURCE_REGISTRY.json',registry)
    print(json.dumps({'status':'PASS','data':data['WSIs'],'replayed_parent_records':parent['checkpoint_records']}),flush=True)


if __name__=='__main__':main()

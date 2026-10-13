"""Phase10 entrypoint: audit -> gates -> baselines -> mechanism matrix."""
import argparse,sys,json,hashlib,time,subprocess,os,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.phase10_trimodal.data import load_private,build_patch_cache,write_json,digest
from src.phase10_trimodal.features import prepare,ahash
from src.phase10_trimodal.train import inner_split,ridge_x,choose_ridge,train_neural,predict_neural,teacher_crossfit
from src.phase10_trimodal.math import fit_ridge

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/phase10_trimodal_mechanism_audit'

def sources_hash():
    paths=sorted((ROOT/'src/phase10_trimodal').glob('*.py'))+[Path(__file__),ROOT/'configs/phase10_trimodal_mechanism_audit.json',ROOT/'configs/phase10_molecular_groups.json',ROOT/'tests/test_phase10_trimodal_audit.py']
    return hashlib.sha256(''.join(str(p.relative_to(ROOT))+digest(p) for p in paths).encode()).hexdigest()

def groups():
    names=['HALLMARK_G2M_CHECKPOINT','HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION','HALLMARK_INTERFERON_GAMMA_RESPONSE','HALLMARK_HYPOXIA','HALLMARK_TGF_BETA_SIGNALING','HALLMARK_DNA_REPAIR','HALLMARK_WNT_BETA_CATENIN_SIGNALING','HALLMARK_ANGIOGENESIS']
    p=ROOT/'configs/phase10_molecular_groups.json'
    return json.loads(p.read_text()) if p.exists() else {n:pd.read_csv(ROOT/'src/phase_m2/metadata/mmp_hallmarks_signatures.csv')[n].dropna().astype(str).tolist() for n in names}

def cell_path(out,arm,seed,fold):return out/'cells'/f'{arm}_seed{seed}_fold{fold}.json'

def save_cell(out,arm,seed,fold,te,risk,info,private_manifest):
    for name in ['cells','oof','manifests','checkpoints','representations']:(out/name).mkdir(parents=True,exist_ok=True)
    pred=out/'oof'/f'{arm}_seed{seed}_fold{fold}.npz'
    np.savez_compressed(pred,test_indices=te,risk=risk,arm=arm,seed=seed,fold=fold)
    manifest=out/'manifests'/f'{arm}_seed{seed}_fold{fold}.json'
    write_json(manifest,private_manifest)
    record={'status':'COMPLETE','arm':arm,'seed':seed,'fold':fold,'test_n':len(te),'prediction_sha256':digest(pred),'private_manifest_sha256':digest(manifest),'source_hash':sources_hash(),'protocol_sha256':digest(ROOT/'configs/phase10_trimodal_mechanism_audit.json'),'outer_test_used_for_selection':False,**info}
    write_json(cell_path(out,arm,seed,fold),record)

def gates(args):
    result=subprocess.run([sys.executable,'-m','unittest','tests.test_phase10_trimodal_audit','-v'],cwd=ROOT,capture_output=True,text=True)
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'unit_test_output.txt').write_text(result.stdout+result.stderr,encoding='utf8')
    report={'status':'PASS' if result.returncode==0 else 'FAIL','returncode':result.returncode,'source_hash':sources_hash(),'test_output_sha256':digest(REPORT/'unit_test_output.txt'),'scope':'synthetic properties only; private data/baseline gates separate'}
    write_json(REPORT/'synthetic_audit.json',report);print(json.dumps(report));return result.returncode

def run_job(args):
    cfg=json.loads((ROOT/'configs/phase10_trimodal_mechanism_audit.json').read_text());out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    gate=json.loads((REPORT/'synthetic_audit.json').read_text())
    if gate['status']!='PASS' or gate['source_hash']!=sources_hash():raise RuntimeError('synthetic gate absent or source changed')
    cache=Path(args.cache);data_gate=json.loads((cache/'data_manifest.json').read_text())
    if data_gate['status']!='PASS':raise RuntimeError('data gate')
    d=load_private(args.data);patch=np.load(cache/'patch.npy',mmap_mode='r');mask=np.load(cache/'mask.npy')
    fold=args.fold;seed=args.seed;tr=np.flatnonzero(d['folds']!=fold);te=np.flatnonzero(d['folds']==fold);it,iv=inner_split(d,tr,cfg['split_seed']+fold)
    if set(tr)&set(te) or set(it)&set(iv) or set(it)|set(iv)!=set(tr):raise RuntimeError('split leak')
    selected_arms=[a for a in cfg['arms'] if (a.startswith('A') if args.stage=='baselines' else not a.startswith('A'))]
    if args.stage=='mechanisms':
        ag=json.loads((out/'baseline_gate.json').read_text())
        if ag['status']!='PASS' or ag['source_hash']!=sources_hash():raise RuntimeError('Hard Gate A baseline check missing')
    if args.arms:selected_arms=args.arms.split(',')
    if args.smoke:selected_arms=['A5','B0','C0','D3','E1'];cfg=copy.deepcopy(cfg);cfg['epochs']=10;cfg['minimum_selection_epoch']=5
    fi,ai=prepare(d,it,groups());fo,ao=prepare(d,tr,groups())
    clambda,clsel=choose_ridge(fi['C'],d['DSS_time'],d['DSS_event'],it,iv,cfg['ridge_grid'])
    teacher_i=teacher_o=None;tia=toa=None
    if any(a in selected_arms for a in ['D2','D3']):
        teacher_i,tia=teacher_crossfit(d,it,groups(),cfg);teacher_o,toa=teacher_crossfit(d,tr,groups(),cfg)
    for arm in selected_arms:
        cp=cell_path(out,arm,seed,fold)
        if cp.exists() and not args.smoke:
            old=json.loads(cp.read_text())
            if old['status']!='COMPLETE' or old['source_hash']!=sources_hash():raise RuntimeError('stale cell resume prohibited')
            continue
        start=time.monotonic();manifest={'outer_train':tr.tolist(),'outer_test':te.tolist(),'inner_train':it.tolist(),'inner_val':iv.tolist(),'preprocess_inner':ai,'preprocess_outer':ao,'clinical_init_selection':clsel,'train_patient_hash':ahash(d['patient_ids'][tr]),'test_patient_hash':ahash(d['patient_ids'][te]),'fold_hash':ahash(d['folds'])}
        if arm=='D0':
            src=cell_path(out,'B0',seed,fold)
            if not src.exists():raise ValueError('D0 B0 reuse absent')
            z=dict(np.load(out/'oof'/f'B0_seed{seed}_fold{fold}.npz'));info=json.loads(src.read_text());risk=z['risk'];info['reuse_of']='B0';info.pop('arm',None);info.pop('status',None)
        elif arm in ['A0','A0_min','A1','A3','A4','A7','A8']:
            xi=ridge_x(fi,arm);xo=ridge_x(fo,arm);lam,sel=choose_ridge(xi,d['DSS_time'],d['DSS_event'],it,iv,cfg['ridge_grid']);w,ci=fit_ridge(xo[tr],d['DSS_time'][tr],d['DSS_event'][tr],lam);risk=xo[te]@w
            info={'lambda':lam,'inner_selection':sel,'selected_inner_Harrell':next(r['inner_Harrell'] for r in sel if r['lambda']==lam),'total_parameters':len(w),'selected_epoch':0,'actual_epochs':0,'optimizer':ci};np.savez_compressed(out/f'private_{arm}_seed{seed}_fold{fold}_coefficients.npz',coefficients=w)
        else:
            m,inner_info,_=train_neural(d,fi,patch,mask,it,iv,arm,seed,cfg,clambda,teacher_i)
            del m
            if torch.cuda.is_available():torch.cuda.empty_cache()
            ep=inner_info['selected_epoch'];(out/'checkpoints').mkdir(exist_ok=True);ck=out/'checkpoints'/f'{arm}_seed{seed}_fold{fold}.pt'
            model,outer_info,reps=train_neural(d,fo,patch,mask,tr,np.array([],int),arm,seed,cfg,clambda,teacher_o,ep,ck)
            risk,test_reps,aux=predict_neural(model,d,fo,patch,mask,te,arm,cfg)
            (out/'representations').mkdir(exist_ok=True)
            np.savez_compressed(out/'representations'/f'{arm}_seed{seed}_fold{fold}.npz',train_indices=tr,test_indices=te,**{'before_'+k:v for k,v in reps[0].items()},**{'after_'+k:v for k,v in reps[1].items()},**{'test_'+k:v for k,v in test_reps.items()},**{'aux_test_'+k:v for k,v in aux.items()})
            info={'total_parameters':outer_info['total_parameters'],'selected_epoch':ep,'actual_epochs':ep,'inner_epochs_run':cfg['epochs'],'selected_inner_Harrell':inner_info['selected_inner_Harrell'],'inner_training':inner_info,'outer_training':outer_info,'checkpoint_sha256':digest(ck),'peak_cuda_MB':max(inner_info['peak_cuda_MB'],outer_info['peak_cuda_MB'])}
            manifest['teacher_inner']=tia;manifest['teacher_outer']=toa
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
        if not np.isfinite(risk).all() or len(risk)!=len(te):raise RuntimeError('prediction gate')
        info['seconds']=time.monotonic()-start
        save_cell(out,arm,seed,fold,te,risk,info,manifest)
        print(json.dumps({'completed':arm,'seed':seed,'fold':fold,'seconds':round(info['seconds'],2),'epoch':info['selected_epoch'],'peak_cuda_MB':info.get('peak_cuda_MB')}),flush=True)
    return 0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['data','gates','run']);ap.add_argument('--data',default='/root/phase7a_foundation_headroom_v1/private_dataset.npz');ap.add_argument('--manifest',default=str(ROOT/'reports/phase6_conch_v15_download_audit/cohort_manifest.csv'));ap.add_argument('--features',default='/root/autodl-tmp/phase10_conch');ap.add_argument('--cache',default='/root/phase10_trimodal_private/cache');ap.add_argument('--output',default='/root/phase10_trimodal_private/formal');ap.add_argument('--fold',type=int,default=1);ap.add_argument('--seed',type=int,default=31);ap.add_argument('--stage',choices=['baselines','mechanisms'],default='baselines');ap.add_argument('--arms');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    torch.set_num_threads(int(os.environ.get('PHASE10_THREADS','2')))
    if args.command=='gates':return gates(args)
    if args.command=='data':print(json.dumps(build_patch_cache(load_private(args.data),args.manifest,args.features,args.cache)));return 0
    return run_job(args)

if __name__=='__main__':raise SystemExit(main())

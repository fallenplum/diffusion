"""Separate immutable neural extraction from fit-only low-rank convex heads."""
from __future__ import annotations
import contextlib
import importlib.util
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch
from src.phase26_multiquery import data as old_data, runtime as old_runtime
from src.phase26_multiquery.model import MultiQuery
from src.phase25_local.train import seed_all
from src.phase25_scale_diagnostic import residual_summary
from . import runtime
from .features import extract_cell
from .math import fit_projection, transform_projection, fit_offset_ridge, efron_nll, pair_diagnostic


def require(value,message):
    if not bool(value):raise ValueError(message)


def _replay_module():
    p=runtime.ROOT/'scripts/replay_phase26.py'
    spec=importlib.util.spec_from_file_location('_phase27_parent_replay',p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def input_contract(seed,fold):
    old_path=old_runtime.cell_dir(seed,fold)
    old_runtime.verify_done(old_path,runtime.LEGACY_SOURCE,seed,fold)
    old_manifest=old_runtime.REPORT/'ARTIFACT_MANIFEST.json'
    manifest=json.loads(old_manifest.read_text())
    require(manifest['training_source']==runtime.LEGACY_SOURCE,'Phase26 manifest source changed')
    for row in manifest['public_files']:
        require(runtime.sha(runtime.ROOT/row['path'])==row['sha256'],'Phase26 public artifact changed')
    replay=_replay_module();main=replay.verify_main(runtime.LEGACY_SOURCE)
    old_neural=json.loads((old_runtime.REPORT/'REPLAY_AUDIT.json').read_text())
    require(old_neural['status']=='PASS' and old_neural['source']==runtime.LEGACY_SOURCE and
            old_neural['maximum_absolute_risk_error']==0 and old_neural['replay_script_sha256']==
            runtime.sha(runtime.ROOT/'scripts/replay_phase26.py'),'Phase26 neural replay changed')
    return old_path,replay,main


def _blocked(*args,**kwargs):raise RuntimeError('feature extraction forbids labels/fitting/optimizer')


def extract(seed,fold,device='cuda'):
    identity=runtime.source();path=runtime.feature_path(seed,fold);receipt=path.with_suffix('.json')
    require(not path.exists() and not receipt.exists(),'feature extraction already exists; refusing silent rerun')
    began=time.monotonic();old_path,replay,main=input_contract(seed,fold)
    old_sha=runtime.sha(old_path/'checkpoint.pt');prepared_sha=main['prepared_sha256_by_fold'][fold]
    seed_all(seed,fold)
    with contextlib.ExitStack() as stack:
        for obj,name in ((old_data.Data,'fit_labels'),(old_data.old.Data,'fit_labels'),
                         (old_data.old.parent,'endpoints'),(old_data.old.parent,'splits'),
                         (old_data,'prepare'),(old_data,'build_prepared'),(old_data,'fit_ridge'),
                         (old_data.old,'fit_preprocessor'),(old_data.old.PCA,'fit'),
                         (torch.optim.Optimizer,'__init__')):
            stack.enter_context(patch.object(obj,name,_blocked))
        ds,audit,_=replay.safe_features(fold,seed,device,prepared_sha,main['legacy_prepared_sha256_by_fold'][fold])
        record=torch.load(old_path/'checkpoint.pt',map_location='cpu',weights_only=False)
        np.testing.assert_array_equal(record['fit'],ds.fit);np.testing.assert_array_equal(record['valid'],ds.valid)
        result=extract_cell(ds,record,seed,fold,device)
    require(runtime.source()==identity and runtime.sha(old_path/'checkpoint.pt')==old_sha,
            'feature extraction changed source or original checkpoint')
    result.update(source=identity,legacy_source=runtime.LEGACY_SOURCE,stage='prototype',seed=int(seed),fold=int(fold),
        fit=np.array(ds.fit),valid=np.array(ds.valid),base_fit=ds.base[ds.fit].cpu().numpy(),
        M0_valid=np.array(ds.R0_valid),CP_valid=np.array(ds.CP_valid),RNA_valid=np.array(ds.RNA_valid),
        old_risks={a:np.array(record['arms'][a]['risk']) for a in runtime.ARMS},
        old_fit_risks={a:np.array(record['arms'][a]['fit_risk']) for a in runtime.ARMS},
        legacy_checkpoint_sha256=old_sha,prepared_sha256=prepared_sha,
        private_fit_label_bytes_deserialized_but_not_used=True)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(result,path)
    runtime.atomic_json(receipt,{'status':'COMPLETE','source':identity,'stage':'prototype','seed':seed,'fold':fold,
        'sha256':runtime.sha(path),'legacy_checkpoint_sha256':old_sha,'prepared_sha256':prepared_sha,
        'wall_seconds':time.monotonic()-began,'new_neural_steps':0,'new_predictor_fits':0,
        'audit':result['audit']})
    return result


def read_features(seed,fold):
    path=runtime.feature_path(seed,fold);r=json.loads(path.with_suffix('.json').read_text());identity=runtime.source()
    require(r.get('status')=='COMPLETE' and r.get('source')==identity and r.get('sha256')==runtime.sha(path),
            'feature receipt/hash/source differs')
    old_path,_,main=input_contract(seed,fold)
    require(r['legacy_checkpoint_sha256']==runtime.sha(old_path/'checkpoint.pt') and
            r['prepared_sha256']==main['prepared_sha256_by_fold'][fold], 'old feature-source receipt differs')
    x=torch.load(path,map_location='cpu',weights_only=False)
    require(x['source']==identity and x['seed']==seed and x['fold']==fold,'feature payload cell differs')
    return x,r


def fit_cell(seed,fold):
    identity=runtime.source();folder=runtime.cell_dir(seed,fold)
    if (folder/'DONE_PRIVATE.json').exists():return runtime.verify_done(folder,identity,seed,fold)
    require(not (folder/'STARTED_PRIVATE.json').exists(),'prior head fit attempt needs explicit audit')
    began=time.monotonic();features,fr=read_features(seed,fold);saved,sr=old_data.read_prepared(fold)
    require(fr['prepared_sha256']==sr['sha256'],'feature/main prepared mismatch')
    for key in ('fit','valid'):np.testing.assert_array_equal(features[key],saved[key])
    np.testing.assert_array_equal(features['base_fit'],saved['base'][saved['fit']])
    folder.mkdir(parents=True,exist_ok=True)
    runtime.atomic_json(folder/'STARTED_PRIVATE.json',{'source':identity,'seed':seed,'fold':fold})
    t,e=np.array(saved['fit_time']),np.array(saved['fit_event']);b=np.array(features['base_fit'])
    output={key:features[key] for key in ('source','stage','seed','fold','fit','valid','M0_valid','CP_valid','RNA_valid',
              'old_risks','legacy_checkpoint_sha256','prepared_sha256')}
    output['feature_receipt_sha256']=runtime.sha(runtime.feature_path(seed,fold).with_suffix('.json'))
    output['heads']={}
    main_pair=pair_diagnostic(t,e,b);main_cox=efron_nll(t,e,b)
    for arm in runtime.ARMS:
        projection=fit_projection(features['fit_features'][arm])
        x=transform_projection(features['fit_features'][arm],projection)
        coef,optimizer=fit_offset_ridge(x,t,e,b,lam=10.)
        fit_risk=b+x@coef
        valid={name:np.array(features['M0_valid'])+
            transform_projection(features['valid_features'][arm][name],projection)@coef for name in runtime.CONTROLS}
        fit_delta=fit_risk-b;ev=np.asarray(e)==1;correlation=float(np.corrcoef(fit_delta,e)[0,1]) if fit_delta.std()>0 else None
        diagnostic={'main_Efron_NLL':main_cox,'old_Efron_NLL':efron_nll(t,e,features['old_fit_risks'][arm]),
            'new_Efron_NLL':efron_nll(t,e,fit_risk),'main_pair':main_pair,
            'old_pair':pair_diagnostic(t,e,features['old_fit_risks'][arm]),'new_pair':pair_diagnostic(t,e,fit_risk),
            'fit_correction':residual_summary(b,fit_risk,float(b.std())),
            'valid_correction':residual_summary(features['M0_valid'],valid['MATCHED'],float(b.std())),
            'coefficient_norm':float(np.linalg.norm(coef)),'PCA_explained_variance_ratio':projection['explained_variance_ratio_sum'],
            'input_fit_rank':projection['input_fit_rank'],'fit_Pearson_correction_DSS_indicator':correlation,
            'fit_DSS_event_mean_correction':float(fit_delta[ev].mean()),
            'fit_DSS_censored_mean_correction':float(fit_delta[~ev].mean()),
            'fit_correction_mean_zero_error':float(abs(fit_delta.mean())),
            'fit_standardized_PC_covariance_max_error':float(abs(x.T@x/len(x)-np.eye(16)).max()),
            'fit_only_pair_loss_not_Efron_additive_decomposition':True}
        output['heads'][arm]={'coef':coef,'projection':projection,'fit_risk':fit_risk,'valid_risks':valid,
            'optimizer':optimizer,'fit_diagnostics':diagnostic,'parameter_count':16}
    require(runtime.source()==identity,'source changed during readout fitting')
    torch.save(output,folder/'head.pt')
    runtime.atomic_json(folder/'DONE_PRIVATE.json',{'status':'COMPLETE','source':identity,'stage':'prototype',
        'seed':seed,'fold':fold,'heads':3,'new_convex_fits':3,'new_neural_steps':0,
        'checkpoint_sha256':runtime.sha(folder/'head.pt'),'feature_receipt_sha256':output['feature_receipt_sha256'],
        'wall_seconds':time.monotonic()-began})
    return output

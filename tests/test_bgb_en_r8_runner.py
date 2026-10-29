"""Synthetic coefficient serialization/replay, no protected experiment assets."""
import argparse
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import torch

from scripts import bgb_en_r8_run as runner
from src.bgb_en_r8.pipeline import FrozenPatientEvidence,FrozenPartitionTransform


FIT=hashlib.sha256(b"synthetic-r8-runner-train").hexdigest()


def setup(tmp_path,monkeypatch):
    rng=np.random.default_rng(31);n=60;h=rng.normal(size=(n,4,64));g=rng.normal(size=h.shape)
    d=FrozenPatientEvidence(rng.normal(size=(n,4)),h,h+.01*g,g,np.ones((n,4),bool),"a"*64,"R6_S7_locked")
    tr=np.arange(40);va=np.arange(40,n);trans=FrozenPartitionTransform.fit(d,tr,fit_identity_sha256=FIT)
    base=rng.normal(size=n);rna=rng.normal(size=n);beta=.07;cp=base-beta*rna
    t=np.round(np.exp(-.2*base+.2*rng.normal(size=n)),2);e=(rng.random(n)<.7).astype(int)
    mapping=np.arange(n);mapping[tr]=np.roll(tr,1);mapping[va]=np.roll(va,1)
    scope=dict(train=tr,valid=va,t=t,e=e,pfi_t=t*1.2,pfi_e=e,pfi_valid=np.ones(n,bool),
        base=base,CP=cp,RNA=rna,beta=beta,original_train_identity_sha256=FIT)
    payload=dict(data=d,transform=trans,scoped=scope,query_mapping=mapping)
    origin=dict(source_state_sha256="a"*64,source_origin=dict(synthetic=True),source_version="R6_S7_locked",
        source_program_definition_sha256="b"*64)
    monkeypatch.setattr(runner,"load_features",lambda *args:(payload,origin))
    conf=tmp_path/"config.json";conf.write_text("{}")
    args=argparse.Namespace(private=tmp_path/"private",config=conf,arm="R8-6",partition=0,seed=31,resume=False)
    ctx=dict(lock=dict(contract_sha256="c"*64,feature_receipt_sha256={"p0_MATCHED":"d"*64}))
    return args,ctx,payload


def test_actual_synthetic_convex_fit_serialization_and_frozen_risk_replay(tmp_path,monkeypatch):
    args,ctx,payload=setup(tmp_path,monkeypatch)
    receipt=runner.run_fit(args,ctx)
    assert receipt["actual_convex_fits"]==1 and receipt["actual_neural_fits"]==receipt["neural_updates"]==0
    assert receipt["registered_coefficients"]==32
    assert receipt["risk_diagnostics"]["train"]["SD_ratio_to_fixed_offset"]<=.1+1e-8
    replay,risk=runner.replay_one(args,ctx,"R8-6",0,31)
    assert replay["maximum_risk_error"]==0 and replay["changed_pair_credits"]==0
    assert replay["convex_KKT"]["new_optimizer_fits"]==0 and risk.shape==(60,)
    assert set(receipt["frozen_controls"])=={"residual_off","interaction_off","matched_query","wrong_query","train_mean_query"}
    with pytest.raises(ValueError,match="COMPLETED_FIT_CHANGED_NO_RETRY"):
        runner.run_fit(args,ctx)
    args.resume=True
    assert runner.run_fit(args,ctx)==receipt


def test_nonrandom_convex_seed_repeats_are_reported_and_numerically_identical(tmp_path,monkeypatch):
    args,ctx,_=setup(tmp_path,monkeypatch)
    a=runner.run_fit(args,ctx);ra=runner.replay_one(args,ctx,args.arm,0,31)[1]
    args.seed=47;b=runner.run_fit(args,ctx);rb=runner.replay_one(args,ctx,args.arm,0,47)[1]
    assert a["deterministic_solver_no_seed_randomization"] is True
    assert b["nominal_seeds_are_not_independent_random_cohorts"] is True
    assert np.array_equal(ra,rb)


def test_tampered_coefficients_cannot_pass_by_updating_only_artifact_sha(tmp_path,monkeypatch):
    args,ctx,_=setup(tmp_path,monkeypatch);runner.run_fit(args,ctx)
    folder=args.private/"cells"/runner.cell_name(args.arm,0,31)
    path=folder/"model.private.pt";saved=torch.load(path,map_location="cpu",weights_only=False)
    saved["model"].coefficient[0]+=.3
    torch.save(saved,path)
    done=json.loads((folder/"COMPLETE.json").read_text());done["artifact_sha256"]=runner.sha(path)
    (folder/"COMPLETE.json").write_text(json.dumps(done))
    with pytest.raises(ValueError,match="KKT_OR_TRAIN_SD_CERTIFICATE_FAILED"):
        runner.replay_one(args,ctx,args.arm,0,31)


def test_error_cell_is_preserved_and_not_automatically_restarted(tmp_path,monkeypatch):
    args,ctx,_=setup(tmp_path,monkeypatch)
    def fail(*args,**kwargs):raise RuntimeError("SYNTHETIC_SOLVER_FAILURE")
    monkeypatch.setattr(runner,"fit_protected_offset_cox",fail)
    with pytest.raises(RuntimeError,match="SYNTHETIC_SOLVER_FAILURE"):
        runner.run_fit(args,ctx)
    folder=args.private/"cells"/runner.cell_name(args.arm,0,31)
    assert (folder/"ERROR_RECEIPT.json").exists() and not (folder/"COMPLETE.json").exists()
    args.resume=True
    with pytest.raises(ValueError,match="FAILED_OR_PARTIAL_CELL"):
        runner.run_fit(args,ctx)


def test_r8_5_wrong_context_is_frozen_diagnostic_not_bilinear_retraining(tmp_path,monkeypatch):
    args,ctx,payload=setup(tmp_path,monkeypatch)
    d=payload["data"]
    payload["data"]=replace(d,wrong_conditioned_image_tokens=np.roll(d.conditioned_image_tokens,1,axis=0))
    args.arm="R8-5"
    done=runner.run_fit(args,ctx)
    assert set(done["frozen_controls"])=={"residual_off","wrong_query"}
    assert done["frozen_controls"]["wrong_query"]["frozen_not_retrained"] is True
    assert done["registered_coefficients"]==8 and done["actual_convex_fits"]==1
    replay,_=runner.replay_one(args,ctx,args.arm,0,31)
    assert replay["maximum_risk_error"]==0 and replay["changed_pair_credits"]==0

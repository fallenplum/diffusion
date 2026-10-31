"""Synthetic worker scopes, rank accounting and saved-head arithmetic only."""
import copy
import gzip
import itertools
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from scripts import semantic_b0b7_p2s_risk_worker as worker
from src.semantic_b0b7.p2_anchor import AnchorBundle
from src.semantic_b0b7 import p2s_survival as risk


def binding(path):return {"path":str(Path(path).resolve()),"sha256":worker.sha(path)}


def meta():
    return {"split":"outer1_inner0","family":"MAIN","seed":31,"lambda_pair":.1,"epoch":400,
            "input_sha256":"a"*64,"source_checkpoint_sha256":"b"*64}


@pytest.mark.parametrize("field,value",[("split","outer1_final"),("split","outer2_inner0"),
    ("family","ATTN_NATIVE"),("seed",999),("lambda_pair",0.),("epoch",600)])
def test_worker_rejects_scope_expansion_without_opening_data(field,value):
    values=meta();values[field]=value
    with pytest.raises(ValueError,match="NOT_IN_LOCKED_INNER"):
        worker.metadata(values)


def test_checkpoint_aliases_are_equal_and_the_original_producer_name_is_retained():
    parsed=worker.metadata(meta())
    assert parsed["source_checkpoint_sha256"]==parsed["checkpoint_sha256"]=="b"*64
    bad={**meta(),"checkpoint_sha256":"c"*64}
    with pytest.raises(ValueError,match="ALIAS_CONFLICT"):worker.metadata(bad)


def test_private_file_sha_and_old_outer_path_are_fail_closed(tmp_path):
    path=tmp_path/"input.npz";path.write_bytes(b"synthetic")
    actual=binding(path);assert worker.checked_file(actual)==path
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="BYTES_CHANGED"):worker.checked_file(actual)
    forbidden=tmp_path/"outer1_final.private.pkl";forbidden.write_bytes(b"owned")
    with pytest.raises(ValueError,match="OUTER_TEST_OR_HISTORICAL"):
        worker.checked_file(binding(forbidden))


def synthetic_cache(tmp_path):
    n=409;fit=np.arange(273,dtype=np.int64);held=np.arange(273,n,dtype=np.int64)
    values={"P":np.zeros((n,8,64),np.float32),"G":np.zeros((n,50,64),np.float32),
        "C":np.zeros((n,64),np.float32),"mask":np.ones((n,50),bool),"M2":np.zeros((n,2),np.float32),
        "u":np.ones((2,64),np.float32),"v":np.ones((2,64),np.float32),"prior":np.zeros((8,50),np.float32),
        "fit_local":fit,"eval_local":held,"global_rows":np.arange(n,dtype=np.int64),
        "fit_global":fit,"eval_global":held}
    values.update({f"clinical_{i}":np.full(n,"synthetic",dtype="U12") for i in range(6)})
    endpoint={"global_rows":values["global_rows"].copy(),"t":np.arange(1,n+1,dtype=float),"e":(np.arange(n)%7==0).astype(int)}
    cache=tmp_path/"cache.npz";scope=tmp_path/"scope.npz"
    np.savez_compressed(cache,**values);np.savez_compressed(scope,**endpoint)
    return values,endpoint,{"feature_cache":binding(cache),"endpoint_scope":binding(scope)}


@pytest.mark.parametrize("corruption",["endpoint_order","extra_secondary","fit_global","M2dtype"])
def test_exact409_cache_scope_and_original_fp32_mechanism_are_required(tmp_path,corruption):
    cache,endpoint,job=synthetic_cache(tmp_path)
    actual,labels,tr,ev=worker.load_scoped_inputs(job)
    assert len(tr)+len(ev)==409 and actual["M2"].dtype==np.float32
    if corruption=="endpoint_order":endpoint["global_rows"]=endpoint["global_rows"][::-1]
    elif corruption=="extra_secondary":endpoint["PFI_time"]=np.asarray([{"never_decoded":True}],object)
    elif corruption=="fit_global":cache["fit_global"]=cache["fit_global"][::-1]
    else:cache["M2"]=cache["M2"].astype(np.float64)
    a=Path(job["feature_cache"]["path"]);b=Path(job["endpoint_scope"]["path"])
    np.savez_compressed(a,**cache);np.savez_compressed(b,**endpoint)
    job={"feature_cache":binding(a),"endpoint_scope":binding(b)}
    with pytest.raises(ValueError):worker.load_scoped_inputs(job)


def test_existing_anchor_exact_rows_offsets_and_no_beta_export(tmp_path):
    cache,_,job=synthetic_cache(tmp_path)
    tr,ev=cache["fit_global"],cache["eval_global"]
    rng=np.random.default_rng(6);a=rng.normal(size=len(tr));b=rng.normal(size=len(ev))
    receipt={"split":"outer1_inner0","p2_lock_sha256":worker.PARENT_P2_SHA,"fit_hash":risk.array_sha(tr),
        "evaluation_hash":risk.array_sha(ev),"teacher_seed":27111,"pipeline_crossfit_seed":27111,
        "full_pipeline_crossfit_for_training_offset":True,"beta_supervision_excluded_for_each_downstream_training_patient":True,
        "Cox_optimizer_fits":52,"scalar_pair_optimizer_fits":4,
        "train_offset_sha256":risk.array_sha(a),"evaluation_offset_sha256":risk.array_sha(b),"full_beta":.1234567}
    anchor=AnchorBundle(tr,ev,a,b,b.copy(),a.copy(),a.copy(),{},receipt.copy())
    path=tmp_path/"outer1_inner0.private.pkl";path.write_bytes(pickle.dumps(anchor,protocol=5))
    receipt["artifact_sha256"]=worker.sha(path)
    note=tmp_path/"outer1_inner0.private.json";note.write_text(json.dumps(receipt))
    job["anchor"]={**binding(path),"receipt_path":str(note.resolve()),"receipt_sha256":worker.sha(note)}
    _,public=worker.load_anchor(job,worker.metadata(meta()),cache)
    assert "full_beta" not in public and ".1234567" not in json.dumps(public)
    assert public["new_teacher_or_scalar_fits"]==0 and public["reused_original_teacher_Cox_fits"]==52
    bad=copy.deepcopy(cache);bad["fit_global"]=bad["fit_global"][::-1]
    with pytest.raises(ValueError,match="CANONICAL_ROWS_CHANGED"):
        worker.load_anchor(job,worker.metadata(meta()),bad)


def test_completed72_registry_session_and_state_meta_are_closed(tmp_path):
    cells=[];states=[]
    for split,family,seed,pair in itertools.product(("outer1_inner0","outer1_inner1","outer1_inner2"),
            ("MAIN","ATTN_MATCHED"),(31,47,83),(.1,.03)):
        cells.append({"split":split,"family":family,"seed":seed,"lambda_pair":pair,
                      "input_sha256":"a"*64,"parent_state400_sha256":"b"*64})
        for epoch in (400,800):states.append({"split":split,"family":family,"seed":seed,"lambda_pair":pair,
            "epoch":epoch,"input_sha256":"a"*64,"source_checkpoint_sha256":"b"*64})
    study={"cells":cells};study_sha="c"*64;readout_sha="d"*64
    registry={"schema":"semantic_p2s_completed_states_v1","study_lock_sha256":study_sha,
              "readout_lock_sha256":readout_sha,"states":states}
    r=tmp_path/"registry.json";r.write_text(json.dumps(registry))
    session={"schema":"semantic_p2s_readout_session_v1","study_lock_sha256":study_sha,"readout_lock_sha256":readout_sha,
             "state_registry_sha256":worker.sha(r),"endpoint_scope_sha256":{"outer1_inner0":"e"*64}}
    s=tmp_path/"session.json";s.write_text(json.dumps(session))
    job={"clinical_keys":list(worker.CLINICAL_FIELDS),"session_binding":binding(s),"state_registry":binding(r),
         "endpoint_scope":{"sha256":"e"*64}}
    assert worker.validate_job_chain(job,worker.metadata(meta()),study,study_sha,readout_sha)["completed_source_states_verified"]==72
    changed=copy.deepcopy(job);changed["endpoint_scope"]["sha256"]="f"*64
    with pytest.raises(ValueError,match="ENDPOINT_SCOPE_CHANGED"):
        worker.validate_job_chain(changed,worker.metadata(meta()),study,study_sha,readout_sha)
    registry["states"][-1]=copy.deepcopy(registry["states"][0]);r.write_text(json.dumps(registry))
    job["state_registry"]=binding(r);session["state_registry_sha256"]=worker.sha(r);s.write_text(json.dumps(session));job["session_binding"]=binding(s)
    with pytest.raises(ValueError,match="COMPLETED72"):
        worker.validate_job_chain(job,worker.metadata(meta()),study,study_sha,readout_sha)


def test_child_lock_anchor_binding_and_anonymous_extraction_compute_are_exact(tmp_path):
    cache,endpoint,job=synthetic_cache(tmp_path)
    job["anchor"]={"sha256":"a"*64,"receipt_sha256":"b"*64}
    child={"anchors":{"outer1_inner0":{"sha256":"a"*64,"receipt_sha256":"b"*64}}}
    worker.validate_child_job_bindings(job,worker.metadata(meta()),child,"c"*64,"d"*64)
    changed=copy.deepcopy(job);changed["anchor"]["sha256"]="e"*64
    with pytest.raises(ValueError,match="ANCHOR_NOT_IN_CHILD_LOCK"):
        worker.validate_child_job_bindings(changed,worker.metadata(meta()),child,"c"*64,"d"*64)
    job["feature_extraction_compute"]={"frozen_encoder_feature_extractions":1,"branch_batches_per_modality":13,
        "elapsed_seconds":.1,"new_neural_optimizer_steps":0,"stored_feature_bytes":Path(job["feature_cache"]["path"]).stat().st_size}
    assert worker.extraction_compute(job)==job["feature_extraction_compute"]
    job["feature_extraction_compute"]["branch_batches_per_modality"]=14
    with pytest.raises(ValueError,match="COMPUTE_CHANGED"):worker.extraction_compute(job)


def features(n=40,seed=17):
    rng=np.random.default_rng(seed)
    return {"pathology":rng.normal(size=(n,24)),"omics":rng.normal(size=(n,28)),"clinical":rng.normal(size=(n,20))}


def test_counted_main_map_matches_the_immutable_operator_without_extra_fits():
    f=features();cost={"actual_PCA_fits":0}
    actual=worker.fit_main_counted(f,8,cost);oracle=risk.MainProjection.fit(f,width=8)
    assert cost["actual_PCA_fits"]==3 and actual.receipt==oracle.receipt
    np.testing.assert_array_equal(actual.transform(f),oracle.transform(f))


def test_rank_failures_only_mark_affected_designs_and_count_completed_pca():
    rng=np.random.default_rng(19);f=features()
    f["pathology"]=rng.normal(size=(40,10))@rng.normal(size=(10,24))
    m2=rng.normal(size=(40,2));mechanisms={}
    for arm,mode in itertools.product(("B4","B5"),("INTER","B7")):
        for rank in (2,4):mechanisms[f"{arm}_R{rank}_{mode}"]={"train":m2 if rank==2 else np.column_stack((m2,m2))}
    source={"train":f,"mechanisms":mechanisms,"unsupported":{}}
    cost={"actual_PCA_fits":0}
    maps,designs,unsupported=worker.prepare_designs(source,risk.readout_specs(),cost)
    assert set(maps)=={8} and len(designs)==5 and len(unsupported)==13 and cost["actual_PCA_fits"]==4
    assert "D8_SHARED_R0_NO_INTER" in designs
    assert all("D16" in name or "R4" in name for name in unsupported)
    row=worker.row_for(risk.ReadoutSpec(8,4,"B4","INTER"),1.,na_reason=next(iter(unsupported.values())))
    assert row["status"]=="NA_NO_SUPPORTED_RANK" and row["DSS_coefficients"] is None and row["planned_DSS_coefficients"]==28


def test_nonrank_software_error_is_not_converted_to_scientific_na(monkeypatch):
    monkeypatch.setattr(worker,"fit_main_counted",lambda *a:(_ for _ in ()).throw(ValueError("SOFTWARE_BUG")))
    with pytest.raises(ValueError,match="SOFTWARE_BUG"):
        worker.prepare_designs({"train":features()},risk.readout_specs(),{"actual_PCA_fits":0})


def manual_payload():
    train=features(n=40,seed=29);ev=features(n=8,seed=30)
    main=risk.MainProjection.fit(train,width=8);spec=risk.ReadoutSpec(8,0,"SHARED","NO_INTER")
    design=risk.ReadoutDesign.fit(spec,main,train);w=np.linspace(-.02,.02,24)
    model=risk.DiagnosticCox(w,1.,{})
    t=np.arange(1,41,dtype=float)*4;e=(np.arange(40)%4==0).astype(int)
    et=np.arange(1,9,dtype=float)*10;ee=(np.arange(8)%3==0).astype(int)
    a=np.zeros(40);b=np.zeros(8)
    r=model.predict(design.transform(train),a);v=model.predict(design.transform(ev),b)
    receipt={"training_metrics":risk.evaluate_safe(t,e,t,e,r),"evaluation_metrics":risk.evaluate_safe(t,e,et,ee,v)}
    cell=risk.DiagnosticCell(design,model,r,v,r.copy(),v.copy(),receipt)
    return {"cells":{worker.key(spec,1.):cell},"designs":{spec.name:design},"main_maps":{8:main},
        "design_matrices":{spec.name:{"train":design.transform(train),"evaluation":design.transform(ev)}},
        "source":{"train":train,"evaluation":ev,"mechanisms":{}},"training_offset":a,"evaluation_offset":b,
        "train_time":t,"train_event":e,"evaluation_time":et,"evaluation_event":ee}


def test_actual_gzip_reload_replays_bytes_components_and_shared_design_identity(tmp_path):
    payload=manual_payload();path=tmp_path/"owned.private.pkl.gz"
    main={k:payload["source"].pop(k) for k in ("train","evaluation")}
    payload["source"]["main_train_sha256"]={k:risk.array_sha(v) for k,v in main["train"].items()}
    payload["source"]["main_evaluation_sha256"]={k:risk.array_sha(v) for k,v in main["evaluation"].items()}
    with gzip.open(path,"xb") as stream:pickle.dump(payload,stream,protocol=5)
    with gzip.open(path,"rb") as stream:loaded=pickle.load(stream)
    records=worker.replay_cells(loaded,main)
    assert len(records)==1 and all(r["actual_saved_head_replay"] and r["scorer_components_exact"] for r in records.values())
    cell=next(iter(loaded["cells"].values()))
    assert cell.design is loaded["designs"][cell.design.spec.name] and cell.design.main is loaded["main_maps"][8]
    changed=copy.deepcopy(loaded);changed["design_matrices"][cell.design.spec.name]["train"][0,0]+=.01
    with pytest.raises(ValueError,match="DESIGN_MATRIX_RECOMPUTATION_DIFFERS"):worker.replay_cells(changed,main)
    cell.model.coefficient[0]+=.001
    with pytest.raises(ValueError,match="REPLAY_BYTES_DIFFER"):worker.replay_cells(loaded,main)


def test_compact_source_does_not_duplicate_raw_main_or_large_background_state(tmp_path):
    from sklearn.linear_model import Ridge
    main=features(n=40);model=Ridge().fit(np.arange(40.)[:,None],np.arange(40.))
    bg={"mean":np.zeros((40,2000)),"full_model":model,"audit":{"background_models":4,"DSS_read":False}}
    source={"train":main,"evaluation":features(n=8),"mechanisms":{"small":{"train":np.ones((40,2))}},
            "unsupported":{},"background":bg,"probe_receipt":{},"mechanism_source":"synthetic"}
    compact=worker.compact_source(source,{"sha256":"a"*64})
    assert not {"train","evaluation","background"}.intersection(compact)
    assert compact["background_numeric_state_sha256"]==worker.numeric_state_sha(bg)
    assert compact["background_audit"]==bg["audit"] and not compact["background_state_or_raw_main_arrays_stored"]
    modified=copy.deepcopy(bg);modified["mean"][0,0]=1
    assert worker.numeric_state_sha(modified)!=compact["background_numeric_state_sha256"]


def test_attention_source_uses_supplied_original_m2_and_zero_background_or_semantic_probe(monkeypatch):
    def forbidden(*a,**k):raise AssertionError("Semantic factor/background operator was called")
    monkeypatch.setattr(risk,"frozen_source_views",forbidden)
    n=409;cache={"P":np.ones((n,8,64),np.float32),"G":np.ones((n,50,64),np.float32),
        "C":np.ones((n,64),np.float32),"mask":np.ones((n,50),bool),"M2":np.arange(n*2,dtype=np.float32).reshape(n,2)}
    cost={"actual_bgRidge":0};tr=np.arange(273);ev=np.arange(273,n)
    source=worker.prepare_source(cache,tr,ev,"ATTN_MATCHED",cost)
    assert cost["actual_bgRidge"]==0 and source["probe_receipt"] is None
    np.testing.assert_array_equal(source["mechanisms"]["ATTN_MATCHED_R2_INTER"]["train"],cache["M2"][tr])


def test_public_receipt_cannot_export_ndarray_or_private_paths(tmp_path):
    with pytest.raises(ValueError,match="PUBLIC_ARRAY"):
        worker.public_write(tmp_path/"forbidden.json",{"data":np.arange(4)})
    worker.public_write(tmp_path/"good.json",{"status":"synthetic","fit_count":0,"source_hash":"a"*64})
    assert (tmp_path/"good.json").is_file()

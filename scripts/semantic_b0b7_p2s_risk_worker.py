"""One locked, CPU-only inner P2S frozen-readout job; no neural forward.

Only a 409-row outer-training scope and its existing inner anchor are accepted.
All learned arrays and models remain in one private gzip pickle. The separate
receipt contains hashes, scalar diagnostics and complete optimizer traces only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import re
import struct
import sys
import time

STARTUP_THREADS={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.semantic_b0b7 import p2s_survival as risk
from src.semantic_b0b7.data import CLINICAL_FIELDS

SCRIPT="scripts/semantic_b0b7_p2s_risk_worker.py"
TEST="tests/test_semantic_b0b7_p2s_risk_worker.py"
PURE="src/semantic_b0b7/p2s_survival.py"
PURE_TEST="tests/test_semantic_b0b7_p2s_survival.py"
PARENT_P2_SHA="b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
RANK_NA=frozenset(("INSUFFICIENT_REAL_FEATURE_DIMENSIONS_NO_PADDING",
    "COMPACT_BLOCK_RELATIVE_RANK_INSUFFICIENT_NO_NOISE_WHITENING",
    "COMPACT_BLOCK_HAS_INACTIVE_DIMENSION_NO_FAKE_CAPACITY",
    "P2S_MECHANISM_HAS_INACTIVE_COLUMNS_NO_PADDING",
    "P2S_MECHANISM_EFFECTIVE_RANK_INSUFFICIENT_NO_RETRY"))
PROBE_RANK_NA=frozenset(("P2S_ORIGINAL_FACTOR_ROWS_DEPENDENT_NO_RETRY","P2S_FIXED_COMPLEMENT_DEGENERATE_NO_RETRY"))
META_KEYS=("split","family","seed","lambda_pair","epoch","input_sha256","checkpoint_sha256")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    require(not isinstance(value,np.ndarray),"PUBLIC_ARRAY_FORBIDDEN")
    return value


def canonical(value):return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def numeric_state_sha(value):
    """Hash nuisance state without retaining its large arrays or coefficients."""
    h=hashlib.sha256()
    def emit(tag,raw):h.update(tag);h.update(struct.pack("!Q",len(raw)));h.update(raw)
    def visit(x):
        if isinstance(x,np.ndarray):
            require(not x.dtype.hasobject,"READOUT_BACKGROUND_OBJECT_ARRAY")
            emit(b"array",str(x.dtype).encode()+str(x.shape).encode()+np.ascontiguousarray(x).tobytes())
        elif isinstance(x,np.generic):emit(b"numpy_scalar",str(x.dtype).encode()+x.tobytes())
        elif isinstance(x,dict):
            require(all(isinstance(k,str) for k in x),"READOUT_BACKGROUND_NONSTRING_KEYS")
            emit(b"dict",str(len(x)).encode())
            for k in sorted(x):emit(b"key",k.encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            emit(type(x).__name__.encode(),str(len(x)).encode())
            for item in x:visit(item)
        elif type(x).__module__.startswith("sklearn.") and hasattr(x,"__dict__"):
            emit(b"estimator",(type(x).__module__+"."+type(x).__name__).encode());visit(vars(x))
        elif x is None:emit(b"none",b"")
        elif isinstance(x,bool):emit(b"bool",bytes((x,)))
        elif isinstance(x,int):emit(b"int",str(x).encode())
        elif isinstance(x,float):emit(b"float",struct.pack("!d",x))
        elif isinstance(x,str):emit(b"str",x.encode())
        else:raise ValueError("READOUT_UNSUPPORTED_BACKGROUND_STATE_TYPE")
    visit(value);return h.hexdigest()


def digest(value):return isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}",value) is not None


def checked_file(binding,*,suffix=None):
    require(isinstance(binding,dict) and set(("path","sha256")).issubset(binding) and digest(binding["sha256"]),
            "PRIVATE_INPUT_BINDING_INVALID")
    path=Path(binding["path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),"PRIVATE_INPUT_MISSING_OR_SYMLINK")
    require(re.search(r"(?:^|[/\\])outer[1-5]_final(?:[/.\\]|$)|(?:^|[/\\])outer[/\\]fold[1-5](?:[/.\\]|$)",str(path)) is None,
            "OUTER_TEST_OR_HISTORICAL_OUTER_PATH_FORBIDDEN")
    if suffix is not None:require(path.name.endswith(suffix),"PRIVATE_INPUT_SUFFIX_CHANGED")
    require(sha(path)==binding["sha256"],"PRIVATE_INPUT_BYTES_CHANGED")
    return path


def metadata(job):
    values=dict(job.get("meta",job))
    checkpoint=values.get("checkpoint_sha256",values.get("source_checkpoint_sha256"))
    if "checkpoint_sha256" in values and "source_checkpoint_sha256" in values:
        require(values["checkpoint_sha256"]==values["source_checkpoint_sha256"],"READOUT_CHECKPOINT_ALIAS_CONFLICT")
    out={k:values[k] for k in META_KEYS if k!="checkpoint_sha256"};out["checkpoint_sha256"]=checkpoint
    require(re.fullmatch(r"outer1_inner[0-2]",out["split"]) is not None and out["family"] in ("MAIN","ATTN_MATCHED") and
        out["seed"] in (31,47,83) and out["lambda_pair"] in (.1,.03) and out["epoch"] in (400,800) and
        digest(out["input_sha256"]) and digest(checkpoint),"READOUT_JOB_NOT_IN_LOCKED_INNER_STUDY")
    out["source_checkpoint_sha256"]=checkpoint
    return out


def meta_identity(meta):return tuple(meta[k] for k in META_KEYS)


def validate_locks(readout_path,readout_sha,study_path,study_sha):
    require(digest(readout_sha) and sha(readout_path)==readout_sha and sha(study_path)==study_sha,"READOUT_OR_STUDY_LOCK_BYTES_CHANGED")
    child,study=read(readout_path),read(study_path)
    require(child["schema"]=="semantic_b0b7_p2s_readout_lock_v1" and child["status"]=="LOCKED_BEFORE_P2S_DSS_FITS" and
        child["parent_study_lock_sha256"]==study_sha,"READOUT_CHILD_LOCK_SCOPE_CHANGED")
    require(study["schema"]=="semantic_b0b7_p2s_lock_v1" and study["status"]=="LOCKED_BEFORE_P2S_OPTIMIZER" and
        study["parent_master_sha256"]==PARENT_P2_SHA,"READOUT_PARENT_STUDY_SCOPE_CHANGED")
    require({SCRIPT,TEST,PURE,PURE_TEST}.issubset(child["readout_sources"]),"READOUT_SOURCE_CLOSURE_INCOMPLETE")
    for source_set in (study["frozen_sources"],child["readout_sources"]):
        for name,expected in source_set.items():
            path=(ROOT/name).resolve()
            require(path.is_relative_to(ROOT.resolve()) and path.is_file() and sha(path)==expected,"READOUT_FROZEN_SOURCE_CHANGED")
    require(all(child["readout_sources"].get(name,expected)==expected for name,expected in study["frozen_sources"].items()),
            "READOUT_MUST_NOT_REBIND_STUDY_SOURCES")
    if "primary_lambda" in child:require(child["primary_lambda"]==risk.PRIMARY_LAMBDA==1.,"READOUT_PRIMARY_LAMBDA_CHANGED")
    if "ridge_lambdas" in child:require(tuple(child["ridge_lambdas"])==risk.RIDGE_LAMBDAS,"READOUT_LAMBDA_GRID_CHANGED")
    return child,study


def validate_job_chain(job,meta,study,study_sha,readout_sha):
    require(tuple(job["clinical_keys"])==CLINICAL_FIELDS,"READOUT_CLINICAL_FIELD_ORDER_CHANGED")
    matches=[c for c in study["cells"] if all(c[k]==meta[k] for k in ("split","family","seed","lambda_pair"))]
    require(len(matches)==1 and matches[0]["input_sha256"]==meta["input_sha256"],"READOUT_STATE_INPUT_NOT_IN_STUDY")
    if meta["lambda_pair"]==.1 and meta["epoch"]==400:
        require(meta["checkpoint_sha256"]==matches[0]["parent_state400_sha256"],"READOUT_ORIGINAL400_CHECKPOINT_CHANGED")
    session_path=checked_file(job["session_binding"],suffix=".json")
    registry_path=checked_file(job["state_registry"],suffix=".json")
    session,registry=read(session_path),read(registry_path)
    require(session["schema"]=="semantic_p2s_readout_session_v1" and session["readout_lock_sha256"]==readout_sha and
        session["study_lock_sha256"]==study_sha and session["state_registry_sha256"]==job["state_registry"]["sha256"],
        "READOUT_SESSION_BINDING_CHANGED")
    endpoint=session["endpoint_scope_sha256"]
    expected_endpoint=endpoint[meta["split"]] if isinstance(endpoint,dict) else endpoint
    require(expected_endpoint==job["endpoint_scope"]["sha256"],"READOUT_SESSION_ENDPOINT_SCOPE_CHANGED")
    require(registry["schema"]=="semantic_p2s_completed_states_v1" and registry["study_lock_sha256"]==study_sha and
        registry["readout_lock_sha256"]==readout_sha,"READOUT_COMPLETED_STATE_REGISTRY_CHANGED")
    entries=[metadata(r.get("meta",r)) for r in registry["states"]]
    require(len(entries)==len({meta_identity(e) for e in entries})==72,"READOUT_COMPLETED72_STATE_REGISTRY_REQUIRED")
    expected=set(itertools.product(("outer1_inner0","outer1_inner1","outer1_inner2"),("MAIN","ATTN_MATCHED"),
                                  (31,47,83),(.1,.03),(400,800)))
    require({tuple(e[k] for k in ("split","family","seed","lambda_pair","epoch")) for e in entries}==expected,
            "READOUT_COMPLETED_STATE_SCOPE_CHANGED")
    require(sum(meta_identity(e)==meta_identity(meta) for e in entries)==1,"READOUT_CHECKPOINT_NOT_UNIQUE_IN_COMPLETED_REGISTRY")
    return {"session_sha256":job["session_binding"]["sha256"],"state_registry_sha256":job["state_registry"]["sha256"],
            "completed_source_states_verified":72}


def validate_child_job_bindings(job,meta,child,readout_sha,study_sha):
    anchored=child["anchors"][meta["split"]]
    require(job["anchor"]["sha256"]==anchored["sha256"] and
            job["anchor"]["receipt_sha256"]==anchored["receipt_sha256"],"READOUT_ANCHOR_NOT_IN_CHILD_LOCK")
    for key,expected in (("readout_lock_sha256",readout_sha),("study_lock_sha256",study_sha)):
        if key in job:require(job[key]==expected,"READOUT_JOB_LOCK_ALIAS_CHANGED")


def extraction_compute(job):
    value=job["feature_extraction_compute"]
    required={"frozen_encoder_feature_extractions","branch_batches_per_modality","elapsed_seconds",
              "new_neural_optimizer_steps","stored_feature_bytes"}
    require(set(value)==required and value["frozen_encoder_feature_extractions"]==1 and
        value["branch_batches_per_modality"]==13 and value["new_neural_optimizer_steps"]==0 and
        type(value["elapsed_seconds"]) in (int,float) and math.isfinite(value["elapsed_seconds"]) and value["elapsed_seconds"]>=0 and
        value["stored_feature_bytes"]==Path(job["feature_cache"]["path"]).stat().st_size,"READOUT_FEATURE_EXTRACTION_COMPUTE_CHANGED")
    return dict(value)


def integer_rows(value,name):
    x=np.asarray(value)
    require(x.ndim==1 and x.dtype.kind in "iu" and len(np.unique(x))==len(x),"READOUT_UNIQUE_INTEGER_ROWS_REQUIRED:"+name)
    return x.astype(np.int64)


def load_scoped_inputs(job):
    cache_path=checked_file(job["feature_cache"],suffix=".npz")
    endpoint_path=checked_file(job["endpoint_scope"],suffix=".npz")
    with np.load(cache_path,allow_pickle=False) as archive:
        required={"P","G","C","mask","M2","u","v","prior","fit_local","eval_local","global_rows",
                  "fit_global","eval_global"}|{f"clinical_{i}" for i in range(6)}
        require(set(archive.files)==required,"READOUT_FROZEN_CACHE_FIELDS_CHANGED")
        cache={k:np.array(archive[k],copy=True) for k in required}
    with np.load(endpoint_path,allow_pickle=False) as archive:
        require(set(archive.files)=={"global_rows","t","e"},"READOUT_DSS_SCOPE_MUST_CONTAIN_ONLY_THREE_TRAIN409_FIELDS")
        endpoints={k:np.array(archive[k],copy=True) for k in archive.files}
    rows=integer_rows(cache["global_rows"],"global_rows")
    require(rows.shape==(409,) and (rows>=0).all() and (rows<512).all() and
            np.array_equal(endpoints["global_rows"],cache["global_rows"]),"READOUT_ENDPOINT_SCOPE_ORDER_OR409_CHANGED")
    tr=integer_rows(cache["fit_local"],"fit_local");ev=integer_rows(cache["eval_local"],"eval_local")
    require(len(tr)>16 and len(ev)>1 and np.array_equal(np.sort(np.r_[tr,ev]),np.arange(409)),
            "READOUT_LOCAL_PARTITIONS_MUST_EXACTLY_COVER_TRAIN409")
    require(np.array_equal(cache["fit_global"],rows[tr]) and np.array_equal(cache["eval_global"],rows[ev]),
            "READOUT_LOCAL_GLOBAL_ROW_BINDING_CHANGED")
    shapes={"P":(409,8,64),"G":(409,50,64),"C":(409,64),"mask":(409,50),"M2":(409,2),
            "u":(2,64),"v":(2,64),"prior":(8,50)}
    for name,shape in shapes.items():
        require(cache[name].shape==shape and np.isfinite(cache[name]).all(),"READOUT_CACHE_SHAPE_OR_FINITE_CHANGED:"+name)
    require(cache["mask"].dtype==bool and cache["M2"].dtype==np.float32 and
            all(cache[k].dtype.kind=="f" for k in ("P","G","C","u","v","prior")) and (cache["prior"]>=0).all(),
            "READOUT_CACHE_DTYPES_OR_ORIGINAL_FP32_MECHANISM_CHANGED")
    for i in range(6):require(cache[f"clinical_{i}"].shape==(409,) and cache[f"clinical_{i}"].dtype.kind=="U", "READOUT_RAW_CLINICAL_STRINGS_REQUIRED")
    require(endpoints["t"].shape==endpoints["e"].shape==(409,) and np.isfinite(endpoints["t"]).all() and
            (endpoints["t"]>=0).all() and np.isin(endpoints["e"],(0,1)).all() and endpoints["e"][tr].sum()>0,
            "READOUT_CURRENT_FIT_DSS_SCOPE_INVALID")
    return cache,endpoints,tr,ev


def load_anchor(job,meta,cache):
    binding=job["anchor"]
    path=checked_file(binding,suffix=".pkl")
    receipt_path=checked_file({"path":binding["receipt_path"],"sha256":binding["receipt_sha256"]},suffix=".json")
    receipt=read(receipt_path)
    require(receipt["artifact_sha256"]==binding["sha256"] and receipt["split"]==meta["split"] and
        receipt["p2_lock_sha256"]==PARENT_P2_SHA,"READOUT_SHARED_ANCHOR_SOURCE_CHANGED")
    # This own-study artifact is byte-qualified before normal class decoding.
    from src.semantic_b0b7.p2_anchor import AnchorBundle
    with path.open("rb") as stream:anchor=pickle.load(stream)
    require(isinstance(anchor,AnchorBundle) and np.array_equal(anchor.train_rows,cache["fit_global"]) and
        np.array_equal(anchor.evaluation_rows,cache["eval_global"]),"READOUT_ANCHOR_CANONICAL_ROWS_CHANGED")
    fit_hash=risk.array_sha(np.asarray(cache["fit_global"],np.int64));eval_hash=risk.array_sha(np.asarray(cache["eval_global"],np.int64))
    inner=int(meta["split"][-1]);teacher=27111+inner
    require(receipt["fit_hash"]==anchor.receipt["fit_hash"]==fit_hash and
        receipt["evaluation_hash"]==anchor.receipt["evaluation_hash"]==eval_hash and
        receipt["teacher_seed"]==receipt["pipeline_crossfit_seed"]==teacher and
        receipt["full_pipeline_crossfit_for_training_offset"] and receipt["beta_supervision_excluded_for_each_downstream_training_patient"] and
        receipt["Cox_optimizer_fits"]==52 and receipt["scalar_pair_optimizer_fits"]==4,
        "READOUT_ANCHOR_COMPLETE_PIPELINE_EXCLUSION_OR_HASH_CHANGED")
    require(anchor.training_offset.shape==(len(cache["fit_global"]),) and anchor.evaluation_offset.shape==(len(cache["eval_global"]),) and
        np.isfinite(anchor.training_offset).all() and np.isfinite(anchor.evaluation_offset).all() and
        np.array_equal(anchor.evaluation_offset,anchor.b0_exact_evaluation) and
        risk.array_sha(anchor.training_offset)==receipt["train_offset_sha256"] and
        risk.array_sha(anchor.evaluation_offset)==receipt["evaluation_offset_sha256"],"READOUT_SHARED_R0_OFFSETS_CHANGED")
    public={"artifact_sha256":binding["sha256"],"receipt_sha256":binding["receipt_sha256"],
        "fit_rows_sha256":fit_hash,"evaluation_rows_sha256":eval_hash,
        "training_offset_sha256":receipt["train_offset_sha256"],"evaluation_offset_sha256":receipt["evaluation_offset_sha256"],
        "reused_original_teacher_Cox_fits":52,"reused_original_scalar_fits":4,"new_teacher_or_scalar_fits":0,
        "full_pipeline_crossfit_for_training_offset":True,"own_labels_excluded_from_teacher_and_beta":True}
    return anchor,public


def base_main(cache):
    g=np.where(cache["mask"][...,None],cache["G"],0.)
    return {"pathology":cache["P"].reshape(409,-1),"omics":g.reshape(409,-1),"clinical":np.asarray(cache["C"],np.float64)}


def compact_source(source,cache_binding):
    bg=source["background"]
    return {"mechanisms":source["mechanisms"],"unsupported":source["unsupported"],
        "probe_receipt":source["probe_receipt"],"mechanism_source":source["mechanism_source"],
        "main_train_sha256":{k:risk.array_sha(v) for k,v in source["train"].items()},
        "main_evaluation_sha256":{k:risk.array_sha(v) for k,v in source["evaluation"].items()},
        "background_audit":None if bg is None else bg["audit"],
        "background_numeric_state_sha256":None if bg is None else numeric_state_sha(bg),
        "background_state_or_raw_main_arrays_stored":False,
        "reconstruction_source_cache_sha256":cache_binding["sha256"],
        "reconstruction":"validated_raw_feature_cache_plus_fixed_source_algorithm_no_outcome_inputs"}


def reload_main_for_replay(cache_binding,fit_local,eval_local):
    path=checked_file(cache_binding,suffix=".npz")
    with np.load(path,allow_pickle=False) as archive:
        cache={key:np.array(archive[key],copy=True) for key in ("P","G","C","mask")}
    main=base_main(cache)
    result={"train":{k:v[fit_local] for k,v in main.items()},"evaluation":{k:v[eval_local] for k,v in main.items()}}
    require(sha(path)==cache_binding["sha256"],"READOUT_CACHE_CHANGED_DURING_REPLAY_RELOAD")
    return result


def prepare_source(cache,tr,ev,family,cost):
    if family=="ATTN_MATCHED":
        main=base_main(cache)
        return {"train":{k:v[tr] for k,v in main.items()},"evaluation":{k:v[ev] for k,v in main.items()},
            "mechanisms":{"ATTN_MATCHED_R2_INTER":{"train":cache["M2"][tr],"evaluation":cache["M2"][ev]}},
            "unsupported":{},"background":None,"probe_receipt":None,
            "mechanism_source":"original_frozen_FP32_attention_operator_supplied_by_locked_feature_producer"}
    outputs={"pre_residual_visual_tokens":cache["P"],"pre_residual_gene_tokens":cache["G"],
             "gene_mask":cache["mask"],"additive_clinical":cache["C"]}
    clinical={field:cache[f"clinical_{i}"] for i,field in enumerate(CLINICAL_FIELDS)}
    try:
        source=risk.frozen_source_views(outputs,clinical,tr,ev,u_weight=cache["u"],v_weight=cache["v"],relation_prior=cache["prior"])
        cost["actual_bgRidge"]=source.receipt["background_models"]
        return {"train":source.train,"evaluation":source.evaluation,"mechanisms":source.mechanisms,
            "unsupported":{},"background":source.background,"probe_receipt":source.probe.receipt,
            "mechanism_source":"same_frozen_MAIN_two_pretrained_plus_prelocked_complement_probes"}
    except ValueError as exc:
        if str(exc) not in PROBE_RANK_NA:raise
        reason=str(exc)
    # A missing rank-four complement must not disable otherwise supported
    # rank-two heads or the same main-effect-only comparator.
    main=base_main(cache)
    flat=np.column_stack((main["pathology"],main["omics"]))
    bg=risk.background_crossfit(main["clinical"],flat,tr,ev,clinical_raw=clinical,seed=103017,alpha=1.)
    cost["actual_bgRidge"]=bg["audit"]["background_models"]
    p,g=cache["P"],np.where(cache["mask"][...,None],cache["G"],0.)
    mp=bg["mean"][:,:512].reshape(p.shape);mg=bg["mean"][:,512:].reshape(g.shape)
    probe=risk.FrozenFactorProbe(np.asarray(cache["u"],np.float64),np.asarray(cache["v"],np.float64),
        {"factor4_unsupported_rank_reason":reason,"resampled_candidates":0})
    mechanisms={};unsupported={}
    for arm in ("B4","B5"):
        for mode in ("INTER","B7"):
            value=risk.frozen_factor_mechanism(p,g,cache["mask"],probe,cache["prior"],factor_rank=2,mode=mode,base_arm=arm,
                mu_visual=mp if arm=="B5" else None,mu_gene=mg if arm=="B5" else None)
            mechanisms[f"{arm}_R2_{mode}"]={"train":value[tr],"evaluation":value[ev]}
            unsupported[f"{arm}_R4_{mode}"]=reason
    return {"train":{k:v[tr] for k,v in main.items()},"evaluation":{k:v[ev] for k,v in main.items()},
        "mechanisms":mechanisms,"unsupported":unsupported,"background":bg,"probe_receipt":probe.receipt,
        "mechanism_source":"original_rank2_supported_rank4_unavailable_no_resampling"}


def fit_main_counted(features,width,cost):
    """Same immutable BlockProjection path; count completed PCA even if gate fails."""
    arrays={k:risk.matrix(v,k) for k,v in features.items()};blocks={}
    for key in ("pathology","omics","clinical"):
        size=8 if key=="clinical" else width
        require(len(arrays[key])>size and arrays[key].shape[1]>=size,"INSUFFICIENT_REAL_FEATURE_DIMENSIONS_NO_PADDING")
        try:
            blocks[key]=risk.BlockProjection.fit(arrays[key],size,minimum_sd=1e-10,
                relative_singular_tolerance=risk.RANK_RELATIVE_TOLERANCE)
        except ValueError as exc:
            if str(exc) in RANK_NA:cost["actual_PCA_fits"]+=1
            raise
        cost["actual_PCA_fits"]+=1
    return risk.MainProjection(width,blocks,{"main_pca":width,"clinical_pca":8,"actual_PCA_fits":3,
        "main_coefficients":2*width+8,"train_only":True,"outcomes_used":False,
        "input_train_sha256":{k:risk.array_sha(v) for k,v in arrays.items()},
        "blocks":{k:v.receipt for k,v in blocks.items()},"reusable_across_all_modes_factors_lambdas":True})


def mkey(spec):return f"{spec.base_arm}_R{spec.factor_rank}_{spec.mode}"


def prepare_designs(source,specs,cost):
    maps={};unsupported={};designs={}
    for width in (8,16):
        try:maps[width]=fit_main_counted(source["train"],width,cost)
        except ValueError as exc:
            if str(exc) not in RANK_NA:raise
            for spec in specs:
                if spec.main_pca==width:unsupported[spec.name]=str(exc)
    for spec in specs:
        if spec.name in unsupported:continue
        if spec.mode!="NO_INTER" and mkey(spec) in source["unsupported"]:
            unsupported[spec.name]=source["unsupported"][mkey(spec)];continue
        mechanism=None if spec.mode=="NO_INTER" else source["mechanisms"][mkey(spec)]["train"]
        try:designs[spec.name]=risk.ReadoutDesign.fit(spec,maps[spec.main_pca],source["train"],mechanism)
        except ValueError as exc:
            if str(exc) not in RANK_NA:raise
            unsupported[spec.name]=str(exc)
    return maps,designs,unsupported


def key(spec,lam):return spec.name+"/lambda"+str(float(lam))


def replay_cells(payload,main):
    results={};matrices={}
    require({k:risk.array_sha(v) for k,v in main["train"].items()}==payload["source"]["main_train_sha256"] and
        {k:risk.array_sha(v) for k,v in main["evaluation"].items()}==payload["source"]["main_evaluation_sha256"],
        "P2S_RELOADED_RAW_CACHE_MAIN_VALUES_DIFFER")
    for spec_name,design in payload["designs"].items():
        spec=design.spec;m=None if spec.mode=="NO_INTER" else payload["source"]["mechanisms"][mkey(spec)]
        train=design.transform(main["train"],None if m is None else m["train"])
        evaluation=design.transform(main["evaluation"],None if m is None else m["evaluation"])
        saved=payload["design_matrices"][spec_name]
        require(risk.array_sha(train)==risk.array_sha(saved["train"]) and risk.array_sha(evaluation)==risk.array_sha(saved["evaluation"]),
                "P2S_ACTUAL_SAVED_DESIGN_MATRIX_RECOMPUTATION_DIFFERS")
        matrices[spec_name]={"train":train,"evaluation":evaluation}
    for name,cell in payload["cells"].items():
        spec=cell.design.spec
        a=cell.model.predict(matrices[spec.name]["train"],payload["training_offset"])
        b=cell.model.predict(matrices[spec.name]["evaluation"],payload["evaluation_offset"])
        require(risk.array_sha(a)==risk.array_sha(cell.training_risk) and risk.array_sha(b)==risk.array_sha(cell.evaluation_risk),
                "P2S_ACTUAL_SAVED_HEAD_REPLAY_BYTES_DIFFER")
        training=risk.evaluate_safe(payload["train_time"],payload["train_event"],payload["train_time"],payload["train_event"],a)
        evaluation=risk.evaluate_safe(payload["train_time"],payload["train_event"],payload["evaluation_time"],payload["evaluation_event"],b)
        require(training==cell.receipt["training_metrics"] and evaluation==cell.receipt["evaluation_metrics"],
                "P2S_ACTUAL_SAVED_REPLAY_SCORER_COMPONENTS_DIFFER")
        results[name]={"actual_saved_head_replay":True,"training_risk_byte_hash_equal":True,"evaluation_risk_byte_hash_equal":True,
            "scorer_components_exact":True,"max_abs_risk_error":0.,"neural_forward":False,"new_optimizer_fits":0}
    return results


def row_for(spec,lam,cell=None,na_reason=None):
    row={"spec_name":spec.name,"spec":asdict(spec),"ridge_lambda":float(lam),"primary_lambda":1.,
        "is_primary_lambda":lam==1.,"status":"NA_NO_SUPPORTED_RANK" if na_reason else "FITTED_AND_REPLAYED",
        "NA_reason":na_reason,"planned_DSS_coefficients":spec.coefficients,"DSS_coefficients":None if na_reason else spec.coefficients,
        "train_metrics":None,"evaluation_metrics":None,"anchor_evaluation_metrics":None,"risk_optimizer":None,"design":None,
        "repair_vs_r0":None,"repair_vs_nointer":None,"repair_vs_B7":None,"decomposition_scalar_sd":None,
        "main_coefficient_drift":None,"frozen_extra_off_evaluation_metrics":None}
    if cell is not None:
        row.update(train_metrics=cell.receipt["training_metrics"],evaluation_metrics=cell.receipt["evaluation_metrics"],
            anchor_evaluation_metrics=cell.receipt["anchor_evaluation_metrics"],risk_optimizer=cell.model.receipt,design=cell.design.receipt,
            repair_vs_r0=cell.receipt["repair_damage_vs_shared_r0"],
            frozen_extra_off_evaluation_metrics=cell.receipt["frozen_mechanism_zero_evaluation_metrics"],
            training_metrics_scope=cell.receipt["training_metrics_scope"],study_scope=cell.receipt["study_scope"])
    return row


def add_refit_diagnostics(rows,payload):
    private={};cells=payload["cells"]
    def repair(cell,reference):
        return risk.risk_repair_damage(payload["evaluation_time"],payload["evaluation_event"],cell.evaluation_risk,
            reference.evaluation_risk,payload["train_time"],payload["train_event"])
    for row in rows:
        if row["status"]!="FITTED_AND_REPLAYED":continue
        spec=risk.ReadoutSpec(**row["spec"]);lam=row["ridge_lambda"];cell=cells[key(spec,lam)]
        if spec.mode=="NO_INTER":continue
        baseline=cells.get(key(risk.ReadoutSpec(spec.main_pca,0,"SHARED","NO_INTER"),lam))
        if baseline is not None:
            m=payload["source"]["mechanisms"][mkey(spec)]["evaluation"]
            d=risk.decompose_joint_vs_refit(cell,baseline,payload["source"]["evaluation"],m,payload["evaluation_offset"])
            private[key(spec,lam)]=d
            row["repair_vs_nointer"]=repair(cell,baseline);row["main_coefficient_drift"]=d["main_coefficient_drift"]
            row["decomposition_scalar_sd"]={"total_delta_vs_refit_sd":float(np.std(d["joint_risk"]-d["real_refit_no_interaction_risk"])),
                "main_coef_delta_sd":float(np.std(d["Z_times_beta_full_minus_beta_refit"])),
                "mechanism_delta_sd":float(np.std(d["M_times_gamma"])),
                "frozen_extra_off_delta_vs_refit_sd":float(np.std(d["frozen_extra_features_off_risk"]-d["real_refit_no_interaction_risk"])),
                "identity_max_error":d["max_abs_decomposition_error"]}
        if spec.mode=="INTER" and spec.base_arm in ("B4","B5"):
            reference=cells.get(key(risk.ReadoutSpec(spec.main_pca,spec.factor_rank,spec.base_arm,"B7"),lam))
            if reference is not None:row["repair_vs_B7"]=repair(cell,reference)
    return private


def public_write(path,value):
    from scripts import bgb_en_publication as original
    value=clean(value);original.validate_public_json(value)
    raw=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()
    require(len(raw)<=2*1024**2 and not any(p.search(raw.decode()) for p in original.PATTERNS.values()),
            "READOUT_PUBLIC_SCALAR_RECEIPT_RULE_FAILED")
    with Path(path).open("xb") as stream:stream.write(raw)


def run(readout_lock,readout_sha,study_lock,study_sha,job_path,job_sha,output):
    require(digest(job_sha) and sha(job_path)==job_sha,"READOUT_JOB_BYTES_CHANGED")
    child,study=validate_locks(readout_lock,readout_sha,study_lock,study_sha)
    job=read(job_path);meta=metadata(job);chain=validate_job_chain(job,meta,study,study_sha,readout_sha)
    validate_child_job_bindings(job,meta,child,readout_sha,study_sha)
    require(not Path(output).exists(),"READOUT_OUTPUT_MUST_BE_NEW")
    cache,endpoints,tr,ev=load_scoped_inputs(job);anchor,anchor_public=load_anchor(job,meta,cache)
    compute=extraction_compute(job)
    for name,values in (("fit_global_sha256",cache["fit_global"]),("eval_global_sha256",cache["eval_global"])):
        if name in job:require(job[name]==risk.array_sha(values),"READOUT_JOB_GLOBAL_ROW_HASH_CHANGED")
    require(STARTUP_THREADS=={k:"2" for k in STARTUP_THREADS},"READOUT_FORMAL_THREAD_ENV_REQUIRED_BEFORE_IMPORT")
    from threadpoolctl import threadpool_info
    require(all(p["num_threads"]==2 for p in threadpool_info() if p["user_api"]=="blas"),"READOUT_BLAS_THREADS_MUST_BE2")
    specs=risk.readout_specs() if meta["family"]=="MAIN" else risk.attention_readout_specs()
    cost={"actual_PCA_fits":0,"actual_bgRidge":0,"actual_Cox_fits":0,"actual_Cox_fit_attempts":0}
    started=time.monotonic();source=prepare_source(cache,tr,ev,meta["family"],cost)
    maps,designs,unsupported=prepare_designs(source,specs,cost)
    design_matrices={}
    for name,design in designs.items():
        m=None if design.spec.mode=="NO_INTER" else source["mechanisms"][mkey(design.spec)]
        design_matrices[name]={"train":design.transform(source["train"],None if m is None else m["train"]),
            "evaluation":design.transform(source["evaluation"],None if m is None else m["evaluation"])}
    target=Path(output);target.mkdir(parents=True)
    cells={};rows=[]
    ledger=target/"COMPUTE_ATTEMPTS.private.json"
    def save_ledger():
        # Deliberately private, and updated before each optimizer attempt so a
        # software failure cannot erase already consumed optimization budget.
        temp=ledger.with_suffix(".tmp")
        temp.write_text(json.dumps({"schema":"semantic_p2s_readout_attempt_ledger_v1","meta":meta,
            **cost,"completed_cell_keys":list(cells),"expected_fits":len(specs)*3},indent=2)+"\n")
        temp.replace(ledger)
    save_ledger()
    for spec in specs:
        for lam in risk.RIDGE_LAMBDAS:
            if spec.name in unsupported:
                rows.append(row_for(spec,lam,na_reason=unsupported[spec.name]));continue
            m=None if spec.mode=="NO_INTER" else source["mechanisms"][mkey(spec)]
            cost["actual_Cox_fit_attempts"]+=1
            save_ledger()
            cell=risk.fit_diagnostic_cell(designs[spec.name],source["train"],source["evaluation"],
                endpoints["t"][tr],endpoints["e"][tr],endpoints["t"][ev],endpoints["e"][ev],
                anchor.training_offset,anchor.evaluation_offset,ridge_lambda=lam,
                train_mechanism=None if m is None else m["train"],evaluation_mechanism=None if m is None else m["evaluation"],
                bindings={**meta,"outer_test_accessed":False,"job_sha256":job_sha,"readout_lock_sha256":readout_sha})
            cells[key(spec,lam)]=cell;rows.append(row_for(spec,lam,cell));cost["actual_Cox_fits"]+=1
            save_ledger()
    for row in rows:
        width=row["spec"]["main_pca"]
        row["mainmap_sha256"]=canonical(maps[width].receipt) if width in maps else None
    payload={"schema":"semantic_p2s_private_shared_design_cells_v1","meta":meta,"cells":cells,"main_maps":maps,
        "designs":designs,"design_matrices":design_matrices,"source":compact_source(source,job["feature_cache"]),
        "feature_cache":dict(job["feature_cache"]),"fit_local":tr,"eval_local":ev,
        "training_offset":anchor.training_offset,"evaluation_offset":anchor.evaluation_offset,
        "offsets":{"train":anchor.training_offset,"evaluation":anchor.evaluation_offset},
        "train_time":endpoints["t"][tr],"train_event":endpoints["e"][tr],
        "evaluation_time":endpoints["t"][ev],"evaluation_event":endpoints["e"][ev]}
    payload["cells_by_spec"]={name:{str(float(lam)):cells[key(design.spec,lam)] for lam in risk.RIDGE_LAMBDAS}
        for name,design in designs.items()}
    # Compute local dependency diagnostics with the in-memory validated main
    # values, but retain only their small vectors, not duplicate raw P/G/C.
    dependency_payload={**payload,"source":source}
    payload["decompositions"]=add_refit_diagnostics(rows,dependency_payload)
    artifact=target/"READOUT_MODELS.private.pkl.gz"
    with gzip.open(artifact,"xb",compresslevel=3) as stream:pickle.dump(payload,stream,protocol=5)
    artifact_sha=sha(artifact)
    with gzip.open(artifact,"rb") as stream:reloaded=pickle.load(stream)
    replay_main=reload_main_for_replay(reloaded["feature_cache"],reloaded["fit_local"],reloaded["eval_local"])
    replays=replay_cells(reloaded,replay_main)
    require(sha(artifact)==artifact_sha,"READOUT_PERSISTED_ARTIFACT_CHANGED_DURING_REPLAY")
    for name in ("feature_cache","endpoint_scope","state_registry","session_binding","anchor"):
        checked_file(job[name])
    checked_file({"path":job["anchor"]["receipt_path"],"sha256":job["anchor"]["receipt_sha256"]})
    for row in rows:
        if row["status"]=="FITTED_AND_REPLAYED":
            spec=risk.ReadoutSpec(**row["spec"]);row["saved_head_replay"]=replays[key(spec,row["ridge_lambda"])]
    primary_unsupported=any(s.main_pca==8 and s.factor_rank in (0,2) and s.name in unsupported for s in specs)
    na_count=sum(r["status"]=="NA_NO_SUPPORTED_RANK" for r in rows)
    status="PARTIAL_PRIMARY_UNSUPPORTED_RANK" if primary_unsupported else "COMPLETE_WITH_NONPRIMARY_RANK_NA" if na_count else "COMPLETE"
    receipt={"schema":"semantic_b0b7_p2s_readout_state_v1","status":status,"meta":meta,"rows":rows,
        "readout_lock_sha256":readout_sha,"study_lock_sha256":study_sha,"job_sha256":job_sha,**chain,
        "feature_cache_sha256":job["feature_cache"]["sha256"],"endpoint_scope_sha256":job["endpoint_scope"]["sha256"],
        "feature_extraction_compute":compute,
        "raw_main_or_large_background_state_duplicated_in_gzip":False,
        "anchor":anchor_public,"artifact_sha256":artifact_sha,"artifact_bytes":artifact.stat().st_size,
        **cost,"NA_count":na_count,"expected_fits":len(specs)*3,"actual_saved_head_replays":len(replays),
        "actual_new_teacher_or_scalar_fits":0,"main_map_receipts":{str(k):v.receipt for k,v in maps.items()},
        "probe_receipt":source["probe_receipt"],"mechanism_source":source["mechanism_source"],
        "primary_lambda":1.,"all_lambdas_reported_without_selection":True,"no_selection":True,
        "scope":"post_P2_development_inner_diagnostic_not_independent_confirmation","outer_test_accessed":False,
        "new_neural_fits":0,"new_model_forwards":0,"elapsed_seconds":time.monotonic()-started,
        "frozen_ablation_is_dependency_only_true_nointer_is_independently_refit":True,
        "learned_coefficients_or_risks_exported":False,"CI_or_p_values_computed":False}
    public_write(target/"READOUT.private.json",receipt)
    return {"status":status,"actual_Cox_fits":cost["actual_Cox_fits"],"expected_fits":len(specs)*3,
        "NA_count":na_count,"actual_saved_head_replays":len(replays),"receipt_sha256":sha(target/"READOUT.private.json")}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("readout-lock","expected-readout-lock-sha256","study-lock","expected-study-lock-sha256",
                 "job","expected-job-sha256","output-newdir"):parser.add_argument("--"+name,required=True)
    a=parser.parse_args()
    print(json.dumps(run(a.readout_lock,a.expected_readout_lock_sha256,a.study_lock,a.expected_study_lock_sha256,
        a.job,a.expected_job_sha256,a.output_newdir),sort_keys=True))

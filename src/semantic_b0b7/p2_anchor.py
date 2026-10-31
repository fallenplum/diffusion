"""Exact historical R4 recipe plus fully cross-fitted CP_SCALAR offsets.

B0's fit->evaluation recipe is unchanged. Downstream training offsets add a
second crossfit of the ENTIRE recipe, including its supervised scalar beta.
Thus a held patient's event/time cannot select that patient's own offset.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np


LEGACY_VISUAL_SOURCE_SHA256="4898df75eb03539e85ef394772fe2bda991bfbe325154cc3241426e168c39194"
LEGACY_DEPENDENCIES=("src/bgb_en/baselines.py","src/bgb_en/protocol.py","src/bgb_en/survival.py",
                     "src/phase10_trimodal/math.py","src/phase10_trimodal/reference.py")
ROW_FIELDS=frozenset(("patient_ids","folds","S32","titan","uni2_mean","rna","cna","platform","cancer",
                      "rna_measured","cna_measured","t","e","valid","conch_local","conch_mean",
                      "conch_global_matched","conch_global_all"))


def array_sha(value):
    x=np.ascontiguousarray(value)
    header=json.dumps({"dtype":str(x.dtype),"shape":list(x.shape)},sort_keys=True).encode()
    return hashlib.sha256(header+x.tobytes()).hexdigest()


def _rows(indices,n):
    x=np.asarray(indices)
    if (x.ndim!=1 or x.dtype.kind not in "iu" or len(x)<2 or len(np.unique(x))!=len(x)
            or (x<0).any() or (x>=n).any()):
        raise ValueError("ANCHOR_PARTITION_ROWS_INVALID")
    return x.astype(np.int64)


def legacy_source_hashes(root):
    root=Path(root)
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in LEGACY_DEPENDENCIES}


def load_legacy_backend(root,*,expected_source_hashes=None):
    """Load the pinned original recipe under an isolated relative namespace.

    The legacy survival module imports phase10 math via the src namespace;
    append its source directory without shadowing the new semantic package.
    """
    root=Path(root).resolve()
    sources=legacy_source_hashes(root)
    if expected_source_hashes is not None and sources!=expected_source_hashes:
        raise ValueError("R4_LEGACY_SOURCE_HASH_CHANGED")
    import src
    legacy_src=str(root/"src")
    if legacy_src not in src.__path__:
        src.__path__.append(legacy_src)
    name="_semantic_r4_legacy_"+hashlib.sha256(str(root).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package=types.ModuleType(name)
        package.__path__=[str(root/"src/bgb_en")]
        package.__package__=name
        sys.modules[name]=package
    module=importlib.import_module(name+".baselines")
    for label,relative in (("src.phase10_trimodal.math","src/phase10_trimodal/math.py"),
                           ("src.phase10_trimodal.reference","src/phase10_trimodal/reference.py")):
        loaded=importlib.import_module(label)
        actual=hashlib.sha256(Path(loaded.__file__).read_bytes()).hexdigest()
        if actual!=sources[relative]:
            raise ValueError("MIXED_LEGACY_MATH_SOURCE_IMPORT")
    return module.prepare_anchors,sources


def _validate_recipe(raw,expected_union_sha256,expected_visual_sha256):
    required={"t","e","clinical_raw","S32","rna","cna","rna_symbols","cna_symbols",
              "conch_local","conch_mean","_direct_gene_universe","audit"}
    if not required.issubset(raw):
        raise ValueError("R4_EXACT_RAW_ASSETS_REQUIRED")
    n=len(raw["t"]); universe=np.asarray(raw["_direct_gene_universe"])
    if (universe.dtype!=bool or universe.shape!=(np.asarray(raw["rna"]).shape[1],)
            or universe.sum()<1 or array_sha(universe)!=expected_union_sha256):
        raise ValueError("R4_REQUIRES_PINNED_LEGACY_EIGHT_PROGRAM_UNION")
    if (raw["audit"].get("visual_sources_sha256")!=expected_visual_sha256
            or expected_visual_sha256!=LEGACY_VISUAL_SOURCE_SHA256):
        raise ValueError("R4_REQUIRES_ORIGINAL_AUDITED_GRID_LOCAL_VISUAL_SOURCE")
    if any(np.asarray(raw[k]).shape!=(n,768) for k in ("conch_local","conch_mean")):
        raise ValueError("R4_GRID_LOCAL_AND_GLOBAL_768_REQUIRED")
    if np.asarray(raw["S32"]).shape!=(n,32):
        raise ValueError("R4_FIXED_S32_REQUIRED")
    return n


def _partition_raw(raw,fit,evaluation):
    """Do not even transform other outer-test rows in an inner teacher call."""
    n=len(raw["t"])
    selected=np.sort(np.r_[fit,evaluation])
    lookup={int(v):i for i,v in enumerate(selected)}
    local_fit=np.asarray([lookup[int(v)] for v in fit],np.int64)
    local_evaluation=np.asarray([lookup[int(v)] for v in evaluation],np.int64)
    result={}
    for key,value in raw.items():
        if key.startswith("pfi_"):
            continue
        if key in ROW_FIELDS:
            array=np.asarray(value)
            if len(array)!=n:
                raise ValueError("R4_RAW_ROW_ALIGNMENT_CHANGED")
            result[key]=array[selected].copy()
        elif key=="clinical_raw":
            result[key]={field:np.asarray(values)[selected].copy() for field,values in value.items()}
        else:
            result[key]=value
    # The original recipe receives real labels ONLY on its current fit rows.
    permitted=np.zeros(len(selected),bool);permitted[local_fit]=True
    for key in ("t","e"):
        endpoint=np.asarray(result[key],np.float64)
        endpoint[~permitted]=np.nan
        result[key]=endpoint
    if "valid" in result:
        result["valid"]=permitted
    return result,local_fit,local_evaluation,selected


def _compact_state(prepared):
    return {"preprocessing":prepared["features"]["state"],"teachers":prepared["teachers"],
            "full_models":prepared["full_models"],"beta":prepared["beta"],
            "scalar_audit":prepared["scalar_audit"],"risk_distribution":prepared["risk_distribution"],
            "pair_teacher_rule":prepared["pair_teacher_rule"]}


@dataclass
class AnchorBundle:
    train_rows:np.ndarray
    evaluation_rows:np.ndarray
    training_offset:np.ndarray
    evaluation_offset:np.ndarray
    b0_exact_evaluation:np.ndarray
    full_fit_train_risk_diagnostic:np.ndarray
    full_fit_train_in_sample_risk_diagnostic:np.ndarray
    state:dict
    receipt:dict


def fit_r4_anchor(raw,fit_rows,evaluation_rows,membership,*,expected_union_sha256,
                  legacy_root=None,expected_legacy_source_hashes=None,backend=None,
                  teacher_seed=27011,pipeline_crossfit_seed=27011,
                  expected_visual_sha256=LEGACY_VISUAL_SOURCE_SHA256,downstream_crossfit=True):
    """52 Cox + 4 scalar fits for a fully legal downstream train/eval offset.

    With downstream_crossfit=False this is exactly one original R4 pipeline
    (13 Cox + 1 scalar), suitable for the separately labelled B0_EXACT recipe.
    Never use that recipe's own supervised-beta training risk as an OOF offset.
    """
    n=_validate_recipe(raw,expected_union_sha256,expected_visual_sha256)
    fit=_rows(fit_rows,n);evaluation=_rows(evaluation_rows,n)
    if np.intersect1d(fit,evaluation).size:
        raise ValueError("R4_FIT_EVALUATION_OVERLAP")
    if backend is None:
        if legacy_root is None:
            raise ValueError("PINNED_LEGACY_R4_BACKEND_ROOT_REQUIRED")
        backend,sources=load_legacy_backend(legacy_root,expected_source_hashes=expected_legacy_source_hashes)
    else:
        sources={"backend":"explicitly_injected_for_synthetic_qualification_only"}
    calls=[]
    def call(train,held):
        local,a,b,selection=_partition_raw(raw,train,held)
        prepared=backend(local,a,b,membership,seed=teacher_seed)
        if prepared["upstream_optimizer_fits"]!=13 or prepared["upstream_preprocessing_fits"]!=4:
            raise ValueError("ORIGINAL_R4_TEACHER_BUDGET_CHANGED")
        risk=np.asarray(prepared["base"],np.float64)
        if risk.shape!=(len(selection),) or not np.isfinite(risk).all():
            raise ValueError("R4_RETURNED_RISK_INVALID")
        calls.append({"fit_hash":array_sha(train),"held_hash":array_sha(held),
                      "fit_patients":len(train),"evaluation_patients":len(held),
                      "Cox_optimizer_fits":13,"scalar_pair_optimizer_fits":1,
                      "preprocessing_fits":4})
        return prepared,a,b,risk
    full,full_fit,full_eval,full_risk=call(fit,evaluation)
    train_offset=full_risk[full_fit].copy()
    eval_offset=full_risk[full_eval].copy()
    crossfit_states=[]
    if downstream_crossfit:
        order=np.random.default_rng(pipeline_crossfit_seed).permutation(fit)
        blocks=np.array_split(order,3)
        index={int(v):i for i,v in enumerate(fit)}
        train_offset=np.full(len(fit),np.nan)
        for group,held in enumerate(blocks):
            train=np.setdiff1d(fit,held)
            prepared,a,b,risk=call(train,held)
            positions=np.asarray([index[int(v)] for v in held],int)
            train_offset[positions]=risk[b]
            crossfit_states.append({"group":group,"fit_rows":train,"held_rows":held,
                                    "state":_compact_state(prepared)})
    if not np.isfinite(train_offset).all():
        raise ValueError("R4_PIPELINE_CROSSFIT_OFFSET_INCOMPLETE")
    full_in_sample=np.asarray(full["full_risks"]["LOCAL"],float)+full["beta"]*np.asarray(full["full_risks"]["R"],float)
    diagnostic=full_in_sample[full_fit]
    expected_calls=4 if downstream_crossfit else 1
    if len(calls)!=expected_calls:
        raise ValueError("R4_PIPELINE_FIT_LEDGER_CHANGED")
    receipt={"schema":"semantic_b0b7_p2_r4_anchor_v1","recipe":"ORIGINAL_GRID_S32_CLINICAL_PLUS_LEGACY_RNA512_SAME_TEACHER_SCALAR",
             "fit_hash":array_sha(fit),"evaluation_hash":array_sha(evaluation),
             "fit_patients":len(fit),"evaluation_patients":len(evaluation),
             "legacy_source_sha256":sources,"visual_sources_sha256":expected_visual_sha256,
             "legacy_eight_program_union_sha256":expected_union_sha256,
             "full_CONCH_mean_sha256":raw["audit"].get("R4_full_CONCH_mean_sha256"),
             "legacy_groups_file_sha256":raw["audit"].get("legacy_groups_sha256"),
             "teacher_seed":teacher_seed,"pipeline_crossfit_seed":pipeline_crossfit_seed,
             "full_pipeline_crossfit_for_training_offset":downstream_crossfit,
             "beta_supervision_excluded_for_each_downstream_training_patient":downstream_crossfit,
             "evaluation_is_original_full_fit_R4_recipe":True,"shared_full_pipeline_is_B0_exact_source":True,
             "legacy_pipeline_calls":len(calls),"Cox_optimizer_fits":13*len(calls),
             "scalar_pair_optimizer_fits":len(calls),"preprocessing_fits":4*len(calls),
             "calls":calls,"evaluation_labels_available_to_teachers":False,
             "train_offset_sha256":array_sha(train_offset),"evaluation_offset_sha256":array_sha(eval_offset),
             "full_beta":float(full["beta"]),"train_offset_mean":float(train_offset.mean()),
             "train_offset_sd":float(train_offset.std()),"full_fit_in_sample_mean":float(diagnostic.mean()),
             "full_fit_in_sample_sd":float(diagnostic.std()),
             "crossfit_in_sample_RMS_gap":float(np.sqrt(np.mean((train_offset-diagnostic)**2))),
             "training_offset_is_R4_old_recipe_diagnostic_only":not downstream_crossfit}
    state={"full":_compact_state(full),"crossfit":crossfit_states}
    return AnchorBundle(fit.copy(),evaluation.copy(),train_offset,eval_offset,eval_offset.copy(),
                        full_risk[full_fit].copy(),diagnostic.copy(),state,receipt)

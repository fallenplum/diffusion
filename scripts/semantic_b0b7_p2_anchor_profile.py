"""One prelocked, CPU-only E0 anchor profile on outer1/inner0.

No neural optimizer or outer-test metric is executed. Exact R4 full-fit source
and its fully cross-fitted downstream offsets are saved only privately. A
standalone E0 lock is not authorization for the formal P2 outer experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.semantic_b0b7_p2_matrix import raw_inputs
from src.semantic_b0b7.p2_anchor import fit_r4_anchor,legacy_source_hashes,array_sha
from src.semantic_b0b7.p2_data import nested_split_plan


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(2**20),b""):
            h.update(chunk)
    return h.hexdigest()


def write(path,value):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")


def main(a):
    import torch
    torch.set_num_threads(2)
    output=Path(a.output)
    if output.exists():
        raise FileExistsError("E0_PROFILE_DIRECTORY_ALREADY_EXISTS_NO_OVERWRITE")
    output.mkdir(parents=True)
    sources=["scripts/semantic_b0b7_p2_anchor_profile.py","scripts/semantic_b0b7_p2_matrix.py",
             "scripts/semantic_b0b7_prepare.py","src/semantic_b0b7/p2_anchor.py",
             "src/semantic_b0b7/p2_data.py","src/semantic_b0b7/p1r_data.py",
             "src/semantic_b0b7/data.py"]
    source_hash={p:sha(ROOT/p) for p in sources}
    assets=json.loads(Path(a.assets).read_text())
    asset_roles={role:sha(assets[role]) for role in ("cohort","endpoints","visual_sources","gmt","legacy_groups")}
    mean=Path(assets["patch_grid"])/"full_mean.npy"
    ready=Path(assets["patch_grid"])/"READY.json"
    receipt=json.loads(ready.read_text())
    if sha(mean)!=receipt["array_sha256"]["full_mean"]:
        raise ValueError("E0_ORIGINAL_R4_FULL_MEAN_CACHE_CHANGED")
    asset_roles["R4_full_mean"]=sha(mean);asset_roles["R4_grid_ready"]=sha(ready)
    legacy=legacy_source_hashes(a.legacy_root)
    lock={"schema":"semantic_b0b7_e0_anchor_only_lock_v1","status":"LOCKED_BEFORE_E0_ANCHOR_FITS",
          "scope":"outer1_inner0_only_no_outer_test_evaluation","nested_split_seed":27001,
          "outer_stratified_rng_seed":27002,"teacher_seed":27111,"pipeline_crossfit_seed":27111,
          "threads":2,"Cox_fits_maximum":52,"scalar_pair_fits_maximum":4,"neural_fits_authorized":0,
          "outer_metrics_authorized":False,"frozen_sources":source_hash,"asset_sha256":asset_roles,
          "legacy_source_sha256":legacy,"standalone_profile_is_not_formal_P2_lock":True}
    write(output/"E0_PROFILE_LOCK.private.json",lock)
    started=time.perf_counter()
    raw,membership,_=raw_inputs(a,endpoints=True)
    plan=nested_split_plan(raw["folds"],raw["e"],seed=27001)
    split=plan["outer1_inner0"]
    write(output/"E0_PARTITION.private.json",split)
    prepared_at=time.perf_counter()
    bundle=fit_r4_anchor(raw,split["fit"],split["evaluation"],membership[1],
                         expected_union_sha256=array_sha(raw["_direct_gene_universe"]),
                         legacy_root=a.legacy_root,expected_legacy_source_hashes=legacy,
                         teacher_seed=27111,pipeline_crossfit_seed=27111,downstream_crossfit=True)
    fitted_at=time.perf_counter()
    with (output/"anchor.private.pkl").open("xb") as stream:
        pickle.dump(bundle,stream,protocol=5)
    all_states=[bundle.state["full"]]+[x["state"] for x in bundle.state["crossfit"]]
    cox=[];scalar=[]
    for state in all_states:
        for teacher in state["teachers"]:
            cox.extend(m["optimizer"] for m in teacher["models"].values())
        cox.extend(m["optimizer"] for m in state["full_models"].values())
        scalar.append(state["scalar_audit"])
    if len(cox)!=52 or len(scalar)!=4:
        raise ValueError("E0_ACTUAL_TEACHER_OPTIMIZER_LEDGER_NOT_52_PLUS_4")
    if any(not npfinite(x["loss"]) or x["gradient_max"]>2e-5 for x in cox):
        raise ValueError("E0_ACTUAL_R4_COX_CONVERGENCE_FAILED")
    if any(not npfinite(x["loss"]) or x["gradient_max"]>1e-6 for x in scalar):
        raise ValueError("E0_ACTUAL_R4_SCALAR_CONVERGENCE_FAILED")
    sources_unchanged=all(sha(ROOT/p)==value for p,value in source_hash.items())
    result={"schema":"semantic_b0b7_e0_real_anchor_profile_v1","status":"REAL_E0_ANCHOR_FIT_COMPLETED",
            "scope":"outer1_inner0_no_outer_test_metrics","standalone_lock_sha256":sha(output/"E0_PROFILE_LOCK.private.json"),
            "asset_sha256":asset_roles,"frozen_sources":source_hash,"legacy_source_sha256":legacy,
            "fit_rows_sha256":split["fit_sha256"],"evaluation_rows_sha256":split["evaluation_sha256"],
            "fit_patients":len(split["fit"]),"evaluation_patients":len(split["evaluation"]),
            "teacher_seed":27111,"pipeline_crossfit_seed":27111,"threads":2,
            "actual_Cox_fits":len(cox),"actual_scalar_pair_fits":len(scalar),
            "Cox_gradient_max":max(x["gradient_max"] for x in cox),
            "scalar_gradient_max":max(x["gradient_max"] for x in scalar),
            "actual_total_Cox_LBFGS_iterations":sum(x["iterations"] for x in cox),
            "raw_preparation_seconds":prepared_at-started,"anchor_fit_seconds":fitted_at-prepared_at,
            "total_seconds_before_export":fitted_at-started,"peak_process_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "anchor_artifact_sha256":sha(output/"anchor.private.pkl"),"anchor_artifact_bytes":(output/"anchor.private.pkl").stat().st_size,
            "anchor_receipt":bundle.receipt,"sources_unchanged_during_profile":sources_unchanged,
            "neural_models_fitted":0,"outer_test_metric_evaluations":0,
            "P2_matrix_started":False,"profile_training_cost_must_not_be_omitted":True,
            "future_reuse_requires_exact_formal_lock_compatibility":True,"confidence_intervals_computed":False}
    write(output/"E0_PROFILE_ANCHOR_AUDIT.json",result)
    print(json.dumps({k:result[k] for k in ("status","actual_Cox_fits","actual_scalar_pair_fits","Cox_gradient_max","anchor_fit_seconds","peak_process_RSS_bytes","sources_unchanged_during_profile")},sort_keys=True))


def npfinite(value):
    return isinstance(value,(int,float)) and abs(value)<float("inf")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--assets",required=True)
    cli.add_argument("--legacy-root",required=True)
    cli.add_argument("--output",required=True)
    main(cli.parse_args())

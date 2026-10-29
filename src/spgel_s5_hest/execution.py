"""20 development fits, 20 final fits, and actual state replay.

Heldout protection is defined per outer fold. In reverse CV, the same source
can be another fold's legal training data. The role-based official-test loader
is not called until every final state is frozen. Single-slide inner spatial
blocks have no exclusion buffer and do not remove spatial autocorrelation.
"""
from __future__ import annotations
from pathlib import Path
import time
import traceback
import joblib
import numpy as np
import psutil
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from ..spgel_hest.assets import validate_assets,load_partition,read_json,sha256,canonical_hash
from ..spgel_hest.official import require,upstream_raw_metrics,official_round4,json_safe_nonfinite,aggregate_upstream_round4
from ..spgel_hest_neural.graph import build_graph
from .data import coordinates,inner_indices,fit_inputs,save_cache,load_cache,graph_payload
from .models import build_model,parameter_counts
from .protocol import ROOT,ARMS,CHECKPOINTS,write_json,recipe,runtime,private_root,source_hashes,array_sha,name

def _seal(value):
    value["internal_contract_sha256"]=canonical_hash(value)
    return value

def _json_guard(path):
    r=read_json(path)
    require(r["internal_contract_sha256"]==canonical_hash({k:v for k,v in r.items() if k!="internal_contract_sha256"}),"S5_METADATA_SEAL_CHANGED")
    return r

def _check(assets_path,config_path,private):
    cfg=recipe(config_path);lock=_json_guard(Path(private)/"S5B_DEV_RUN_LOCK.json")
    require(lock["status"]=="LOCKED_BEFORE_TWENTY_DEVELOPMENT_FITS"
        and lock["new_scaler_fits"]==lock["new_PCA_fits"]==4 and lock["new_model_fits"]==0
        and lock["official_test_expression_loaded"] is False
        and {(r["dataset"],r["fold"]) for r in lock["partitions"]}=={(d,f) for d in ("COAD","READ") for f in (0,1)}
        and len(lock["partitions"])==4 and lock["source_files"]==source_hashes() and lock["runtime"]==runtime()
        and lock["config_sha256"]==sha256(config_path) and lock["assets_sha256"]==sha256(assets_path),"S5_FROZEN_SOURCES_RUNTIME_OR_ASSETS_CHANGED")
    require(sha256(Path(private)/"ZERO_OPTIMIZER_PROFILE.private.json")==lock["zero_optimizer_profile_sha256"],"S5_QUALIFICATION_CHANGED")
    return cfg,lock,validate_assets(assets_path)

def _context(cache,prefix):
    xy,slide=cache[prefix+"_xy"],cache[prefix+"_slide"]
    graph,indices,valid=graph_payload(xy,slide)
    eye=build_graph(xy,slide,identity=True)
    edge=graph.edge_index[:,graph.edge_index[0]!=graph.edge_index[1]]
    degree=np.bincount(edge[0],minlength=len(xy))
    values=np.asarray(1/degree[edge[0]],np.float32)
    neighbours=torch.sparse_coo_tensor(torch.from_numpy(edge),torch.from_numpy(values),(len(xy),len(xy))).coalesce()
    audit=dict(graph.audit,graphsage_messages=dict(self_edges=0,edges=int(edge.shape[1]),
        maximum_real_neighbors=int(degree.max(initial=0)),isolated_rows=int((degree==0).sum()),
        message_edge_sha256=array_sha(edge),mean_uses_actual_nonself_neighbor_count=True,
        identity_control="SELF_ONLY_MESSAGE_WITH_MATCHED_PARAMETERS_AND_INITIALIZATION"))
    return dict(adjacency=graph.torch_adjacency(),identity_adjacency=eye.torch_adjacency(),
        neighbor_adjacency=neighbours,indices=torch.from_numpy(indices),valid=torch.from_numpy(valid)),audit

def _predict(model,x,context):
    model.eval()
    with torch.inference_mode():
        out=model(torch.from_numpy(x),context,None)["prediction"].numpy()
    require(out.shape==(len(x),50) and np.isfinite(out).all(),"S5_IMAGE_ONLY_PREDICTION_INVALID")
    return out

def _gradient(model):
    out={}
    for k,p in model.named_parameters():
        require(p.grad is not None and torch.isfinite(p.grad).all(),"S5_MISSING_OR_NONFINITE_GRADIENT")
        out[k]=float(p.grad.norm())
    return out

def _profile(cache):
    rows=[]
    torch.set_num_threads(2)
    ctx,_=_context(cache,"train");x=torch.from_numpy(cache["train_x"]);y=torch.from_numpy(cache["train_y"].astype(np.float32))
    for arm in ARMS:
        model=build_model(arm);before={k:array_sha(v.detach().numpy()) for k,v in model.state_dict().items()}
        start=time.monotonic();base_rss=psutil.Process().memory_info().rss
        with threadpool_limits(limits=2):
            r=model(x,ctx,y);loss=F.mse_loss(r["prediction"],y)+0.1*r["auxiliary_loss"]
            require(torch.isfinite(loss),"S5_ZERO_OPTIMIZER_NONFINITE_LOSS");loss.backward();gradient=_gradient(model)
        require(before=={k:array_sha(v.detach().numpy()) for k,v in model.state_dict().items()},"S5_ZERO_OPTIMIZER_STATE_CHANGED")
        rows.append(dict(arm=arm,elapsed_seconds=time.monotonic()-start,RSS_before=base_rss,
            RSS_after_backward=psutil.Process().memory_info().rss,parameters=parameter_counts(model),gradient_norms=gradient,
            optimizer_steps=0,model_state_unchanged=True,test_expression_used=False))
        del model,r,loss
    available=psutil.virtual_memory().available
    need=max(r["RSS_after_backward"] for r in rows)*2+2*2**30
    require(available>=need,"S5_TWO_CPU_WORKER_RAM_QUALIFICATION_FAILED")
    return dict(status="PASS_FIVE_REAL_INNER_TRAIN_FORWARD_BACKWARD_ZERO_OPTIMIZER",units=rows,
        available_RAM_bytes=available,estimated_two_worker_RAM_requirement_bytes=need,
        new_fits=0,optimizer_steps=0)

def prepare_development(assets_path,config_path,private):
    private=private_root(private);cfg=recipe(config_path)
    require(not (private/"S5B_PREPARATION_STARTED.private.json").exists(),"S5_NO_AUTOMATIC_PREPARATION_RETRY")
    initial=dict(source_files=source_hashes(),runtime=runtime(),config_sha256=sha256(config_path),assets_sha256=sha256(assets_path))
    write_json(private/"S5B_PREPARATION_STARTED.private.json",initial)
    assets=validate_assets(assets_path);rows=[];count=0
    try:
        for d in cfg["tasks"]:
            for f in cfg["folds"]:
                part=load_partition(assets,d,f,"train")
                xy=coordinates(assets,d,part);left,right,split=inner_indices(xy,part.specimen_index)
                pipe,a,b=fit_inputs(part.features[left],part.features[right]);count+=1
                folder=private/"dev_inputs"/name(d,f)
                meta=save_cache(folder,dict(train_x=a,validation_x=b,train_y=part.targets[left],validation_y=part.targets[right],
                    train_xy=xy[left],validation_xy=xy[right],train_slide=part.specimen_index[left],validation_slide=part.specimen_index[right]),pipe,
                    dict(dataset=d,fold=f,genes=part.genes,outer_train_ordering_sha256=part.ordering_sha256,inner_split=split,
                    preprocessing_fit_scope="INNER_TRAIN_ONLY",official_test_expression_loaded=False,new_scaler_fits=1,new_PCA_fits=1))
                rows.append(dict(dataset=d,fold=f,cache_metadata_sha256=sha256(folder/"CACHE.private.json"),train_rows=len(a),validation_rows=len(b),split=split))
        largest=max(rows,key=lambda x:x["train_rows"])
        cache,_=load_cache(private/"dev_inputs"/name(largest["dataset"],largest["fold"]),largest)
        profile=_profile(cache);write_json(private/"ZERO_OPTIMIZER_PROFILE.private.json",profile)
        require(initial==dict(source_files=source_hashes(),runtime=runtime(),config_sha256=sha256(config_path),assets_sha256=sha256(assets_path)),"S5_SCIENCE_CHANGED_DURING_QUALIFICATION")
        lock=_seal(dict(schema="s5_hest_development_lock_v1",status="LOCKED_BEFORE_TWENTY_DEVELOPMENT_FITS",**initial,
            partitions=rows,new_scaler_fits=count,new_PCA_fits=count,new_model_fits=0,official_test_expression_loaded=False,
            zero_optimizer_profile_sha256=sha256(private/"ZERO_OPTIMIZER_PROFILE.private.json"),development_fits=20,final_fits=20,
            test_expression_role_definition="own_fold_heldout_not_used_in_training_or_selection_reverse_CV_training_reads_allowed",
            single_slide_inner_block_has_spatial_exclusion_buffer=False))
        write_json(private/"S5B_DEV_RUN_LOCK.json",lock)
        return lock
    except BaseException as e:
        write_json(private/"PREPARATION_ERROR.private.json",dict(status="FAILED_NO_AUTOMATIC_RETRY",type=type(e).__name__,attempted_PCA_fits=count,new_model_fits=0))
        (private/"preparation_error.private.log").write_text(traceback.format_exc(),encoding="utf-8");raise

def _partition(lock,d,f):
    return next(r for r in lock["partitions"] if (r["dataset"],r["fold"])==(d,f))

def _train(cache,arm,epochs,folder,contract,development,binding):
    torch.set_num_threads(2);model=build_model(arm)
    x=torch.from_numpy(cache["train_x"]);y=torch.from_numpy(cache["train_y"].astype(np.float32))
    ctx,graph=_context(cache,"train")
    valctx,valgraph=_context(cache,"validation") if development else (None,None)
    initial={k:array_sha(v.detach().numpy()) for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    curve=[];snapshots=[];start=time.monotonic()
    with threadpool_limits(limits=2):
        for epoch in range(1,epochs+1):
            model.train();opt.zero_grad(set_to_none=True)
            out=model(x,ctx,y,epoch=epoch);main=F.mse_loss(out["prediction"],y);loss=main+0.1*out["auxiliary_loss"]
            require(torch.isfinite(loss),"S5_NONFINITE_TRAIN_LOSS");loss.backward();grads=_gradient(model);opt.step()
            require(all(torch.isfinite(p).all() for p in model.parameters()),"S5_NONFINITE_MODEL_PARAMETER")
            curve.append(dict(epoch=epoch,pre_update_main_MSE=float(main.detach()),pre_update_auxiliary_loss=float(out["auxiliary_loss"].detach()),
                pre_update_total_loss=float(loss.detach()),gradient_norms=grads,optimizer_updates=epoch,
                paired_ITC_audit=out.get("pairing_audit")))
            if (development and epoch not in CHECKPOINTS) or (not development and epoch!=epochs):continue
            train_pred=_predict(model,cache["train_x"],ctx)
            val_pred=_predict(model,cache["validation_x"],valctx) if development else None
            state=folder/f"epoch{epoch:03d}.private.pt";prediction=folder/f"epoch{epoch:03d}.private.npz"
            torch.save(dict(schema="s5_hest_state_v1",arm=arm,epoch=epoch,phase="DEV" if development else "FINAL",**binding,
                state_dict=model.state_dict(),optimizer_state=opt.state_dict(),contract=contract),state)
            with prediction.open("xb") as stream:
                payload=dict(train_prediction=train_pred)
                if development:payload["validation_prediction"]=val_pred
                np.savez_compressed(stream,**payload)
            snapshots.append(dict(epoch=epoch,state_sha256=sha256(state),predictions_sha256=sha256(prediction),
                post_update_train_MSE=float(np.mean((train_pred-cache["train_y"])**2)),
                validation_MSE=float(np.mean((val_pred-cache["validation_y"])**2)) if development else None))
    return dict(status="COMPLETE_ONE_DEVELOPMENT_FIT" if development else "COMPLETE_ONE_FINAL_FIT",
        arm=arm,phase="DEV" if development else "FINAL",**binding,epochs=epochs,optimizer_updates=epochs,new_model_fits=1,parameters=parameter_counts(model),
        initial_state_sha256=initial,learning_curve=curve,snapshots=snapshots,elapsed_seconds=time.monotonic()-start,
        train_graph=graph,validation_graph=valgraph,official_test_expression_loaded=False,
        test_expression_role_statement="own_fold_test_not_used_for_training_or_selection_other_CV_fold_train_reads_are_permitted",
        inference_reads_RNA=False,auxiliary_supervision_only_train_50genes=True,internal_contract_sha256=contract)

def fit_development(assets_path,config_path,private,dataset,fold,arm):
    cfg,lock,_=_check(assets_path,config_path,private);private=Path(private)
    expected=_partition(lock,dataset,fold);folder=private/"dev_cells"/name(dataset,fold,arm)
    require(not folder.exists(),"S5_DEVELOPMENT_CELL_ALREADY_EXISTS_NO_RETRY");folder.mkdir(parents=True)
    try:
        cache,_=load_cache(private/"dev_inputs"/name(dataset,fold),expected)
        result=_train(cache,arm,1280,folder,lock["internal_contract_sha256"],True,
            dict(dataset=dataset,fold=fold,input_cache_metadata_sha256=expected["cache_metadata_sha256"]))
        result.update(dataset=dataset,fold=fold,seed=1,input_cache_metadata_sha256=expected["cache_metadata_sha256"])
        write_json(folder/"COMPLETE.private.json",result);return result
    except BaseException as e:
        write_json(folder/"ERROR.private.json",dict(status="FAILED_NO_AUTOMATIC_RETRY",type=type(e).__name__))
        (folder/"error.private.log").write_text(traceback.format_exc(),encoding="utf-8");raise

def _check_cell(folder,row,contract,expected):
    phase=expected["phase"];epochs=expected["epochs"]
    require(phase in ("DEV","FINAL") and row["phase"]==phase
        and row["status"]==("COMPLETE_ONE_DEVELOPMENT_FIT" if phase=="DEV" else "COMPLETE_ONE_FINAL_FIT")
        and all(row[k]==expected[k] for k in ("dataset","fold","arm","input_cache_metadata_sha256"))
        and row["epochs"]==epochs and (epochs==1280 if phase=="DEV" else epochs in CHECKPOINTS)
        and [s["epoch"] for s in row["snapshots"]]==(list(CHECKPOINTS) if phase=="DEV" else [epochs]),
        "S5_CELL_PHASE_IDENTITY_CACHE_OR_EXACT_EPOCH_SET_CHANGED")
    require(row["internal_contract_sha256"]==contract and row["new_model_fits"]==1
        and row["optimizer_updates"]==row["epochs"]==len(row["learning_curve"])
        and [r["epoch"] for r in row["learning_curve"]]==list(range(1,epochs+1))
        and row["official_test_expression_loaded"] is False,"S5_CELL_CONTRACT_CHANGED")
    private=Path(folder).parents[1]
    cache_path=private/("dev_inputs" if phase=="DEV" else "final_inputs")/name(expected["dataset"],expected["fold"])/"CACHE.private.json"
    require(sha256(cache_path)==expected["input_cache_metadata_sha256"],"S5_CURRENT_CELL_CACHE_METADATA_CHANGED")
    for s in row["snapshots"]:
        e=s["epoch"]
        require(sha256(folder/f"epoch{e:03d}.private.pt")==s["state_sha256"]
            and sha256(folder/f"epoch{e:03d}.private.npz")==s["predictions_sha256"],"S5_STATE_OR_PREDICTION_CHANGED")

def _validate_state(state,row,epoch):
    require(state["schema"]=="s5_hest_state_v1" and state["epoch"]==epoch
        and state["contract"]==row["internal_contract_sha256"]
        and all(state[k]==row[k] for k in ("dataset","fold","arm","phase","input_cache_metadata_sha256")),
        "S5_CHECKPOINT_PHASE_DATASET_FOLD_OR_INPUT_CACHE_CHANGED")

def select_schedule(assets_path,config_path,private):
    cfg,lock,_=_check(assets_path,config_path,private);private=Path(private);selections=[]
    for d in cfg["tasks"]:
        for f in cfg["folds"]:
            rows=[]
            for arm in ARMS:
                folder=private/"dev_cells"/name(d,f,arm);r=read_json(folder/"COMPLETE.private.json")
                _check_cell(folder,r,lock["internal_contract_sha256"],dict(dataset=d,fold=f,arm=arm,phase="DEV",epochs=1280,
                    input_cache_metadata_sha256=_partition(lock,d,f)["cache_metadata_sha256"]))
                require(r["epochs"]==1280 and [x["epoch"] for x in r["snapshots"]]==list(CHECKPOINTS),"S5_FIXED_DEVELOPMENT_BUDGET_CHANGED")
                rows.append(r)
            scores={str(e):float(np.mean([next(s["validation_MSE"] for s in r["snapshots"] if s["epoch"]==e) for r in rows])) for e in CHECKPOINTS}
            require(all(np.isfinite(v) for v in scores.values()),"S5_INNER_SELECTION_MSE_NONFINITE")
            selected=min(CHECKPOINTS,key=lambda e:(scores[str(e)],e))
            selections.append(dict(dataset=d,fold=f,selected_epoch=selected,inner_mean_MSE=scores,
                uses_only_this_outer_fold_training_partition=True,
                budget_assessment="BUDGET_EDGE_NOT_CONVERGENCE_CONFIRMED" if selected==1280 else "INNER_SELECTED_NOT_CONVERGENCE_PROOF",
                development_completion_sha256={a:sha256(private/"dev_cells"/name(d,f,a)/"COMPLETE.private.json") for a in ARMS}))
    result=_seal(dict(schema="s5_hest_final_schedule_lock_v1",status="LOCKED_FROM_20_DEVELOPMENT_FITS_NO_OFFICIAL_TEST",selections=selections,
        source_files=source_hashes(),development_contract=lock["internal_contract_sha256"],official_test_expression_loaded=False,new_model_fits=0))
    write_json(private/"S5B_FINAL_SCHEDULE_LOCK.json",result);return result

def _schedule(private,lock):
    selected=_json_guard(Path(private)/"S5B_FINAL_SCHEDULE_LOCK.json")
    require(selected["status"]=="LOCKED_FROM_20_DEVELOPMENT_FITS_NO_OFFICIAL_TEST"
        and selected["development_contract"]==lock["internal_contract_sha256"] and selected["source_files"]==source_hashes()
        and len(selected["selections"])==4 and {(r["dataset"],r["fold"]) for r in selected["selections"]}=={(d,f) for d in ("COAD","READ") for f in (0,1)}
        and selected["official_test_expression_loaded"] is False,"S5_FINAL_SELECTION_CHANGED")
    for r in selected["selections"]:
        values=[]
        for arm,digest in r["development_completion_sha256"].items():
            path=Path(private)/"dev_cells"/name(r["dataset"],r["fold"],arm)/"COMPLETE.private.json"
            require(sha256(path)==digest,"S5_DEV_CHANGED_AFTER_SELECTION")
            values.append(read_json(path))
        require(set(r["development_completion_sha256"])==set(ARMS),"S5_SELECTION_REQUIRES_ALL_FIVE_ARMS")
        scores={str(e):float(np.mean([next(s["validation_MSE"] for s in a["snapshots"] if s["epoch"]==e) for a in values])) for e in CHECKPOINTS}
        require(scores==r["inner_mean_MSE"] and r["selected_epoch"]==min(CHECKPOINTS,key=lambda e:(scores[str(e)],e)),"S5_SELECTION_NOT_EXACT_OWN_FOLD_INNER_RULE")
    return selected

def prepare_final(assets_path,config_path,private):
    cfg,lock,assets=_check(assets_path,config_path,private);private=Path(private);selected=_schedule(private,lock)
    require(not (private/"FINAL_PREPARATION_STARTED.private.json").exists(),"S5_FINAL_PREPARATION_NO_RETRY")
    write_json(private/"FINAL_PREPARATION_STARTED.private.json",dict(schedule_sha256=sha256(private/"S5B_FINAL_SCHEDULE_LOCK.json")))
    rows=[]
    for d in cfg["tasks"]:
        for f in cfg["folds"]:
            part=load_partition(assets,d,f,"train");xy=coordinates(assets,d,part)
            pipe,x,_=fit_inputs(part.features,None);folder=private/"final_inputs"/name(d,f)
            meta=save_cache(folder,dict(train_x=x,train_y=part.targets,train_xy=xy,train_slide=part.specimen_index),pipe,
                dict(dataset=d,fold=f,genes=part.genes,outer_train_ordering_sha256=part.ordering_sha256,
                    preprocessing_fit_scope="FULL_OFFICIAL_OUTER_TRAIN_ONLY",official_test_expression_loaded=False,new_scaler_fits=1,new_PCA_fits=1))
            rows.append(dict(dataset=d,fold=f,cache_metadata_sha256=sha256(folder/"CACHE.private.json"),train_rows=len(x)))
    final=_seal(dict(schema="s5_hest_final_input_lock_v1",status="LOCKED_FULL_TRAIN_INPUTS_BEFORE_TWENTY_FINAL_FITS",partitions=rows,
        development_contract=lock["internal_contract_sha256"],schedule_sha256=sha256(private/"S5B_FINAL_SCHEDULE_LOCK.json"),
        source_files=source_hashes(),new_scaler_fits=4,new_PCA_fits=4,new_model_fits=0,official_test_expression_loaded=False))
    write_json(private/"S5B_FINAL_INPUT_LOCK.json",final);return final

def _final_lock(private,lock):
    _schedule(private,lock);r=_json_guard(Path(private)/"S5B_FINAL_INPUT_LOCK.json")
    require(r["status"]=="LOCKED_FULL_TRAIN_INPUTS_BEFORE_TWENTY_FINAL_FITS" and r["new_scaler_fits"]==r["new_PCA_fits"]==4
        and r["new_model_fits"]==0 and r["official_test_expression_loaded"] is False
        and len(r["partitions"])==4 and {(a["dataset"],a["fold"]) for a in r["partitions"]}=={(d,f) for d in ("COAD","READ") for f in (0,1)}
        and r["source_files"]==source_hashes() and r["development_contract"]==lock["internal_contract_sha256"]
        and r["schedule_sha256"]==sha256(Path(private)/"S5B_FINAL_SCHEDULE_LOCK.json"),"S5_FINAL_INPUT_LOCK_CHANGED")
    return r

def fit_final(assets_path,config_path,private,dataset,fold,arm):
    cfg,lock,_=_check(assets_path,config_path,private);private=Path(private);final=_final_lock(private,lock)
    expected=_partition(final,dataset,fold);selected=next(r for r in _schedule(private,lock)["selections"] if (r["dataset"],r["fold"])==(dataset,fold))
    folder=private/"final_cells"/name(dataset,fold,arm)
    require(not folder.exists(),"S5_FINAL_CELL_EXISTS_NO_RETRY");folder.mkdir(parents=True)
    try:
        cache,_=load_cache(private/"final_inputs"/name(dataset,fold),expected)
        result=_train(cache,arm,selected["selected_epoch"],folder,final["internal_contract_sha256"],False,
            dict(dataset=dataset,fold=fold,input_cache_metadata_sha256=expected["cache_metadata_sha256"]))
        result.update(dataset=dataset,fold=fold,seed=1,input_cache_metadata_sha256=expected["cache_metadata_sha256"])
        write_json(folder/"COMPLETE.private.json",result);return result
    except BaseException as e:
        write_json(folder/"ERROR.private.json",dict(status="FAILED_NO_AUTOMATIC_RETRY",type=type(e).__name__))
        (folder/"error.private.log").write_text(traceback.format_exc(),encoding="utf-8");raise

def evaluate_final(assets_path,config_path,private):
    cfg,lock,assets=_check(assets_path,config_path,private);private=Path(private);final=_final_lock(private,lock)
    binding={}
    for d in cfg["tasks"]:
        for f in cfg["folds"]:
            for arm in ARMS:
                folder=private/"final_cells"/name(d,f,arm);r=read_json(folder/"COMPLETE.private.json")
                selected=next(s for s in _schedule(private,lock)["selections"] if (s["dataset"],s["fold"])==(d,f))
                _check_cell(folder,r,final["internal_contract_sha256"],dict(dataset=d,fold=f,arm=arm,phase="FINAL",epochs=selected["selected_epoch"],
                    input_cache_metadata_sha256=_partition(final,d,f)["cache_metadata_sha256"]))
                require((r["dataset"],r["fold"],r["arm"])==(d,f,arm) and r["epochs"]==selected["selected_epoch"],"S5_FINAL_COMPLETION_OR_SELECTED_EPOCH_CHANGED")
                binding[name(d,f,arm)]=sha256(folder/"COMPLETE.private.json")
    require(len(binding)==20 and not (private/"OFFICIAL_TEST_OPEN_GATE.private.json").exists(),"S5_TEST_REQUIRES_ALL_TWENTY_FINAL_FITS_AND_NO_PREVIOUS_OPEN")
    write_json(private/"OFFICIAL_TEST_OPEN_GATE.private.json",dict(status="OPENED_AFTER_ALL_20_FINAL_STATES_FROZEN",final_completion_sha256=binding,new_model_fits=0))
    rows=[];torch.set_num_threads(2)
    for d in cfg["tasks"]:
        for f in cfg["folds"]:
            part=load_partition(assets,d,f,"test")
            trainfolder=private/"final_inputs"/name(d,f);expected=_partition(final,d,f);_,meta=load_cache(trainfolder,expected)
            pipeline=joblib.load(trainfolder/"pipeline.private.joblib")
            with threadpool_limits(limits=2):x=np.asarray(pipeline.transform(part.features),np.float32)
            xy=coordinates(assets,d,part);testfolder=private/"test_inputs"/name(d,f)
            testmeta=save_cache(testfolder,dict(test_x=x,test_y=part.targets,test_xy=xy,test_slide=part.specimen_index),pipeline,
                dict(dataset=d,fold=f,genes=part.genes,preprocessing_fit_scope="NO_TEST_FITS_TRANSFORM_ONLY",test_ordering_sha256=part.ordering_sha256,
                    new_scaler_fits=0,new_PCA_fits=0,official_test_open_gate_sha256=sha256(private/"OFFICIAL_TEST_OPEN_GATE.private.json")))
            cache,_=load_cache(testfolder,dict(cache_metadata_sha256=sha256(testfolder/"CACHE.private.json")));ctx,graph=_context(cache,"test")
            for arm in ARMS:
                folder=private/"final_cells"/name(d,f,arm);done=read_json(folder/"COMPLETE.private.json");s=done["snapshots"][0]
                state=torch.load(folder/f"epoch{s['epoch']:03d}.private.pt",map_location="cpu",weights_only=True)
                _validate_state(state,done,s["epoch"])
                model=build_model(arm);model.load_state_dict(state["state_dict"],strict=True)
                with threadpool_limits(limits=2):pred=_predict(model,x,ctx)
                path=folder/"official_test_prediction.private.npz"
                with path.open("xb") as stream:np.savez_compressed(stream,prediction=pred)
                raw=upstream_raw_metrics(part.targets,pred,part.genes)
                r=dict(dataset=d,fold=f,arm=arm,epoch=s["epoch"],state_sha256=s["state_sha256"],COMPLETE_sha256=binding[name(d,f,arm)],
                    test_cache_metadata_sha256=sha256(testfolder/"CACHE.private.json"),test_prediction_sha256=sha256(path),
                    full_precision_metrics=json_safe_nonfinite(raw),official_round4=json_safe_nonfinite(official_round4(raw)),
                    test_rows=len(x),test_graph=graph,first_test_evaluation_after_all_twenty_final_fits=True,new_model_fits=0)
                write_json(folder/"EVALUATION.private.json",r);rows.append(r)
    result=dict(schema="s5_hest_official_evaluation_v1",status="COMPLETE_TWENTY_FROZEN_FINAL_TEST_EVALUATIONS",units=rows,new_model_fits=0,
        final_input_contract=final["internal_contract_sha256"],test_gate_sha256=sha256(private/"OFFICIAL_TEST_OPEN_GATE.private.json"))
    write_json(private/"S5B_OFFICIAL_EVALUATION.private.json",result);return result

def _validated_evaluation(private,final):
    private=Path(private);evaluation=read_json(private/"S5B_OFFICIAL_EVALUATION.private.json")
    rows=evaluation["units"]
    require(evaluation["status"]=="COMPLETE_TWENTY_FROZEN_FINAL_TEST_EVALUATIONS"
        and evaluation["final_input_contract"]==final["internal_contract_sha256"]
        and evaluation["test_gate_sha256"]==sha256(private/"OFFICIAL_TEST_OPEN_GATE.private.json")
        and len(rows)==20 and {(r["dataset"],r["fold"],r["arm"]) for r in rows}=={(d,f,a) for d in ("COAD","READ") for f in (0,1) for a in ARMS},"S5_OFFICIAL_EVALUATION_SCOPE_CHANGED")
    gate=read_json(private/"OFFICIAL_TEST_OPEN_GATE.private.json")
    require(gate["status"]=="OPENED_AFTER_ALL_20_FINAL_STATES_FROZEN" and len(gate["final_completion_sha256"])==20,"S5_TEST_GATE_CHANGED")
    for r in rows:
        cell=name(r["dataset"],r["fold"],r["arm"]);folder=private/"final_cells"/cell
        require(read_json(folder/"EVALUATION.private.json")==r
            and r["COMPLETE_sha256"]==gate["final_completion_sha256"][cell]==sha256(folder/"COMPLETE.private.json")
            and r["test_prediction_sha256"]==sha256(folder/"official_test_prediction.private.npz")
            and r["test_cache_metadata_sha256"]==sha256(private/"test_inputs"/name(r["dataset"],r["fold"])/"CACHE.private.json")
            and r["first_test_evaluation_after_all_twenty_final_fits"] is True and r["new_model_fits"]==0,"S5_EVALUATION_CHANGED_FROM_ACTUAL_CELL_ARTIFACTS")
        testfolder=private/"test_inputs"/name(r["dataset"],r["fold"]);meta=read_json(testfolder/"CACHE.private.json")
        require(sha256(testfolder/"inputs.private.npz")==meta["input_file_sha256"]
            and sha256(testfolder/"pipeline.private.joblib")==meta["pipeline_sha256"],"S5_CURRENT_OFFICIAL_TEST_CACHE_BYTES_CHANGED")
    return evaluation

def replay_all(assets_path,config_path,private):
    cfg,lock,_=_check(assets_path,config_path,private);private=Path(private);final=_final_lock(private,lock)
    evaluation=_validated_evaluation(private,final)
    rows=[];torch.set_num_threads(2)
    for stage,input_dir,cells_dir,contract in (("DEV","dev_inputs","dev_cells",lock["internal_contract_sha256"]),("FINAL","final_inputs","final_cells",final["internal_contract_sha256"])):
        for d in cfg["tasks"]:
            for f in cfg["folds"]:
                cache,_=load_cache(private/input_dir/name(d,f),_partition(lock if stage=="DEV" else final,d,f));ctx,_=_context(cache,"train")
                if stage=="DEV":otherctx,_=_context(cache,"validation");otherx=cache["validation_x"]
                else:
                    testfolder=private/"test_inputs"/name(d,f);tmeta=read_json(testfolder/"CACHE.private.json")
                    testcache,_=load_cache(testfolder,dict(cache_metadata_sha256=sha256(testfolder/"CACHE.private.json")))
                    otherctx,_=_context(testcache,"test");otherx=testcache["test_x"]
                for arm in ARMS:
                    folder=private/cells_dir/name(d,f,arm);done=read_json(folder/"COMPLETE.private.json")
                    epoch=1280 if stage=="DEV" else next(r["selected_epoch"] for r in _schedule(private,lock)["selections"] if (r["dataset"],r["fold"])==(d,f))
                    _check_cell(folder,done,contract,dict(dataset=d,fold=f,arm=arm,phase=stage,epochs=epoch,
                        input_cache_metadata_sha256=_partition(lock if stage=="DEV" else final,d,f)["cache_metadata_sha256"]))
                    for snap in done["snapshots"]:
                        e=snap["epoch"];state=torch.load(folder/f"epoch{e:03d}.private.pt",map_location="cpu",weights_only=True)
                        _validate_state(state,done,e)
                        model=build_model(arm);model.load_state_dict(state["state_dict"],strict=True)
                        with threadpool_limits(limits=2):a=_predict(model,cache["train_x"],ctx);b=_predict(model,otherx,otherctx)
                        with np.load(folder/f"epoch{e:03d}.private.npz",allow_pickle=False) as stored:
                            err=float(np.max(np.abs(a-stored["train_prediction"]),initial=0))
                            if stage=="DEV":err=max(err,float(np.max(np.abs(b-stored["validation_prediction"]),initial=0)))
                        if stage=="FINAL":
                            ev=read_json(folder/"EVALUATION.private.json")
                            require(ev["test_cache_metadata_sha256"]==sha256(testfolder/"CACHE.private.json")
                                and ev["test_prediction_sha256"]==sha256(folder/"official_test_prediction.private.npz"),"S5_TEST_ARTIFACT_CHANGED")
                            with np.load(folder/"official_test_prediction.private.npz",allow_pickle=False) as stored:err=max(err,float(np.max(np.abs(b-stored["prediction"]),initial=0)))
                            require(json_safe_nonfinite(upstream_raw_metrics(testcache["test_y"],b,tmeta["genes"]))==ev["full_precision_metrics"],"S5_REPLAY_OFFICIAL_METRICS_CHANGED")
                        else:require(float(np.mean((b-cache["validation_y"])**2))==snap["validation_MSE"],"S5_REPLAY_INNER_MSE_CHANGED")
                        require(err<=1e-6,"S5_REPLAY_PREDICTION_ERROR")
                        rows.append(dict(stage=stage,dataset=d,fold=f,arm=arm,epoch=e,status="PASS",actual_state_loaded=True,
                            actual_forward_replayed=True,maximum_prediction_error=err,new_model_fits=0,optimizer_steps=0,
                            COMPLETE_sha256=sha256(folder/"COMPLETE.private.json"),state_sha256=snap["state_sha256"]))
    require(len(rows)==120,"S5_EXACT_120_SAVED_STATE_REPLAYS_REQUIRED")
    r=dict(schema="s5_hest_replay_v2",status="PASS_120_ACTUAL_SAVED_STATE_REPLAYS",units=rows,passed_states=120,new_model_fits=0,optimizer_steps=0,
        official_evaluation_sha256=sha256(private/"S5B_OFFICIAL_EVALUATION.private.json"))
    write_json(private/"S5B_REPLAY.private.json",r);return r

def report(assets_path,config_path,private,output):
    cfg,lock,_=_check(assets_path,config_path,private);private=Path(private);final=_final_lock(private,lock);selected=_schedule(private,lock)
    replay=read_json(private/"S5B_REPLAY.private.json");evaluation=_validated_evaluation(private,final)
    require(replay["passed_states"]==len(replay["units"])==120 and len(evaluation["units"])==20
        and replay["official_evaluation_sha256"]==sha256(private/"S5B_OFFICIAL_EVALUATION.private.json")
        and all(r["actual_state_loaded"] and r["actual_forward_replayed"] and r["new_model_fits"]==r["optimizer_steps"]==0 for r in replay["units"]),"S5_ACTUAL_REPLAY_REQUIRED")
    tasks={d:{a:aggregate_upstream_round4([r["official_round4"] for r in sorted(evaluation["units"],key=lambda x:x["fold"]) if r["dataset"]==d and r["arm"]==a]) for a in ARMS} for d in cfg["tasks"]}
    cells=[];total_steps=0;expected_replays=set()
    for stage,sub,contract in (("DEV","dev_cells",lock["internal_contract_sha256"]),("FINAL","final_cells",final["internal_contract_sha256"])):
        for d in cfg["tasks"]:
            for f in cfg["folds"]:
                for a in ARMS:
                    folder=private/sub/name(d,f,a);r=read_json(folder/"COMPLETE.private.json")
                    epoch=1280 if stage=="DEV" else next(s["selected_epoch"] for s in selected["selections"] if (s["dataset"],s["fold"])==(d,f))
                    _check_cell(folder,r,contract,dict(dataset=d,fold=f,arm=a,phase=stage,epochs=epoch,
                        input_cache_metadata_sha256=_partition(lock if stage=="DEV" else final,d,f)["cache_metadata_sha256"]))
                    total_steps+=r["optimizer_updates"]
                    matching=[x for x in replay["units"] if (x["stage"],x["dataset"],x["fold"],x["arm"])==(stage,d,f,a)]
                    require(len(matching)==len(r["snapshots"]) and all(x["COMPLETE_sha256"]==sha256(folder/"COMPLETE.private.json") for x in matching),"S5_COMPLETION_CHANGED_AFTER_REPLAY")
                    expected_replays.update((stage,d,f,a,s["epoch"],s["state_sha256"],sha256(folder/"COMPLETE.private.json")) for s in r["snapshots"])
                    curve=[x for x in r["learning_curve"] if x["epoch"]==1 or x["epoch"]%20==0 or x["epoch"]==r["epochs"]]
                    cells.append(dict(stage=stage,dataset=d,fold=f,arm=a,epochs=r["epochs"],optimizer_updates=r["optimizer_updates"],parameters=r["parameters"],
                        diagnostic_curve_stride20=curve,snapshots=r["snapshots"],elapsed_seconds=r["elapsed_seconds"],COMPLETE_sha256=sha256(folder/"COMPLETE.private.json")))
    _validate_replay_set(replay,expected_replays)
    r=dict(schema="s5_hest_shared_feature_results_v2",status="COMPLETE_TWENTY_DEVELOPMENT_AND_TWENTY_FINAL_FITS_AND_120_REPLAYS",tasks=tasks,cells=cells,
        selections=selected["selections"],actual_development_fits=20,actual_final_fits=20,actual_optimizer_updates=total_steps,actual_saved_state_replays=120,
        new_scaler_fits=8,new_PCA_fits=8,config_sha256=sha256(config_path),source_files=source_hashes(),
        development_lock_sha256=sha256(private/"S5B_DEV_RUN_LOCK.json"),final_schedule_sha256=sha256(private/"S5B_FINAL_SCHEDULE_LOCK.json"),
        final_inputs_sha256=sha256(private/"S5B_FINAL_INPUT_LOCK.json"),evaluation_sha256=sha256(private/"S5B_OFFICIAL_EVALUATION.private.json"),
        replay_sha256=sha256(private/"S5B_REPLAY.private.json"),maximum_replay_error=max(x["maximum_prediction_error"] for x in replay["units"]),
        boundaries=dict(official_test_not_used_for_selection=True,tissue_truth_or_teacher_used=False,extra_gene_membership_used=False,
            queries_are_anonymous_not_medical_programs=True,SPGEL50_is_not_original_A5=True,BLEEP_or_TRIPLEX_reproduced=False,
            optimizer_convergence_proved=False,new_independent_cohort_claimed=False,DSS_PFI_or_R8_read=False,
            own_fold_test_excluded_from_training_and_selection=True,reverse_CV_train_access_to_same_source_is_permitted=True,
            single_slide_inner_block_has_spatial_exclusion_buffer=False))
    write_json(output,r);return r

def _validate_replay_set(replay,expected):
    rows=replay["units"]
    keys=[(r["stage"],r["dataset"],r["fold"],r["arm"],r["epoch"],r["state_sha256"],r["COMPLETE_sha256"]) for r in rows]
    require(len(expected)==120 and len(keys)==len(set(keys))==120 and set(keys)==expected,
        "S5_REPLAY_EPOCH_STATE_SET_NOT_EXACTLY_120_UNIQUE_STATES")

"""New S5 development-donor mechanism contract; never mutates historical runs."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from src.spgel.contracts import SourceStatistics, array_hash, require
from src.spgel import training as inherited
from src.spgel_s5.models import ARMS, S5ProgramEncoder
from src.spgel_s5.losses import objective
from src.spgel_s5.sampling import make_wrong_pairs, make_batch_candidates, permute_program_membership

ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "src/spgel_s5/__init__.py", "src/spgel_s5/models.py", "src/spgel_s5/sampling.py",
    "src/spgel_s5/losses.py", "src/spgel_s5/execution.py", "src/spgel_s5/diagnostics.py",
    "configs/spgel_s5/mechanism_v1.json", "scripts/spgel_s5.py", "scripts/spgel_s5_schedule.py",
    "tests/test_spgel_s5_models.py", "tests/test_spgel_s5_sampling.py",
    "tests/test_spgel_s5_execution.py", "tests/test_spgel_s5_diagnostics.py")
read_json, write_json, sha256, canonical_hash = inherited.read_json, inherited.write_json, inherited.sha256, inherited.canonical_hash


def source_hashes():
    return {name: sha256(ROOT/name) for name in FILES}


def recipe(path):
    value = read_json(path)
    require(value["schema"] == "spgel_s5_mechanism_recipe_v1" and tuple(value["arms"]) == ARMS
        and value["heldout_donors"] == list(range(6)) and value["seeds"] == [31, 47], "S5_MATRIX_CHANGED")
    require(value["training"] == dict(epochs=40, batch_size=512, optimizer="Adam", learning_rate=.001,
        weight_decay=.0001, gradient_clip_norm=5., save_epochs=[10,20,40], primary_epoch=40,
        drop_last=False, heldout_model_or_epoch_selection=False), "S5_TRAINING_CONTRACT_CHANGED")
    require(value["budget"] == dict(new_neural_fits=84, optimizer_updates=72800, saved_state_replays=252,
        automatic_retry_fits=0, M7_M8_fits_in_this_lock=0), "S5_BUDGET_CHANGED")
    require(value["loss"] == dict(program_Huber=1.,tissue_CE=.1,bidirectional_ITC=.1,temperature=.1,
        both_anchor_directions_in_all_arms=True), "S5_LOSS_PACKAGE_CHANGED")
    require(value["scope"]["DSS_PFI_TCGA_or_R8_loaded"] is False, "SURVIVAL_INFORMATION_NOT_ALLOWED")
    return value


def load_source(assets_path, config):
    reference_path = ROOT/config["reference_source_recipe"]
    old_lock = read_json(ROOT/"reports/spgel/SOURCE_RUN_LOCK.json")
    require(sha256(reference_path) == old_lock["config_sha256"] and inherited.source_hashes() == old_lock["sources"],
        "HISTORICAL_SOURCE_CONTRACT_CHANGED")
    return inherited.load_assets(assets_path, read_json(reference_path))


def unit_name(arm, donor, seed):
    require(arm in ARMS and donor in range(6) and seed in (31,47), "S5_UNREGISTERED_CELL")
    return f"{arm}_d{donor}_seed{seed}"


def _mapping_path(private, donor, seed):
    return Path(private)/"qualification"/f"donor{donor}_seed{seed}.private.npz"


def mapping(private, lock, donor, seed):
    entry = next(x for x in lock["partitions"] if x["donor"] == donor and x["seed"] == seed)
    path = _mapping_path(private, donor, seed)
    require(sha256(path) == entry["mapping_sha256"], "S5_MAPPING_CHANGED")
    with np.load(path, allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    return arrays, entry


def _model(source, stats, arm, seed, maps, device):
    up, down = (maps["up"], maps["down"]) if arm == "M6" else (source.up, source.down)
    return S5ProgramEncoder(arm, up, down, source.gene_symbols, seed=seed,
                           tissue_program_mean=stats.tissue_program_mean).to(device)


def _batch(source, stats, rows, maps, seed, device):
    partners = maps["partner_global_rows"][rows]
    require((partners >= 0).all(), "S5_NONTRAIN_PARTNER")
    ni, ng, active, audit = make_batch_candidates(source, rows, partners,
        maps["active"][rows], stats, seed)
    genes = stats.genes_z(source, np.concatenate((rows, partners)), device)
    return (genes, torch.as_tensor(ni, device=device), torch.as_tensor(ng, device=device),
            torch.as_tensor(active, dtype=torch.bool, device=device), audit)


def _gradients(model, *, allow_missing=False):
    result = {}
    for name in ("gene_to_image", "image_to_gene"):
        module = getattr(model, name)
        if module is None:
            result[name] = None
        else:
            gradient = module.weight.grad
            if gradient is None and allow_missing:
                result[name] = 0.
                continue
            require(gradient is not None and torch.isfinite(gradient).all(), "S5_ACTIVE_QUERY_GRADIENT_MISSING")
            result[name] = float(gradient.norm().detach())
    return result


def make_lock(source, config_path, private, *, device="cpu"):
    inherited.configure_runtime();config = recipe(config_path);private = inherited.private_directory(private)
    require(not (private/"PREPARATION_STARTED.private.json").exists(), "S5_PREPARATION_NO_AUTOMATIC_RETRY")
    sources, old_sources = source_hashes(), inherited.source_hashes()
    write_json(private/"PREPARATION_STARTED.private.json", dict(config_sha256=sha256(config_path),sources=sources,
        inherited_sources=old_sources,new_optimizer_fits=0,new_optimizer_updates=0))
    (private/"qualification").mkdir(exist_ok=False)
    partitions=[]
    for donor in range(6):
        train, held=source.split(donor);stats=SourceStatistics.fit(source,train,heldout_donors=[donor])
        for seed in (31,47):
            pairs=make_wrong_pairs(source,train,stats,seed);null=permute_program_membership(source,train,stats,seed)
            path=_mapping_path(private,donor,seed)
            with path.open("xb") as stream:
                np.savez_compressed(stream,partner_global_rows=pairs["partner_global_rows"],active=pairs["active"],
                    up=null["up"],down=null["down"],gene_permutation=null["gene_permutation"])
            require(not pairs["active"][held].any() and (pairs["partner_global_rows"][held] == -1).all(),
                "S5_HELDOUT_IN_TRAIN_PAIR_MAPPING")
            partitions.append(dict(donor=donor,seed=seed,train_rows=len(train),heldout_rows=len(held),
                train_rows_sha256=array_hash(train),heldout_rows_sha256=array_hash(held),mapping_sha256=sha256(path),
                pairing_audit=pairs["audit"],membership_audit=null["audit"],
                updates_per_fit=40*math.ceil(len(train)/512)))
    train, held=source.split(0);stats=SourceStatistics.fit(source,train,heldout_donors=[0])
    with np.load(_mapping_path(private,0,31),allow_pickle=False) as archive:maps={k:archive[k] for k in archive.files}
    count=min(512,len(train));rows=train[np.floor((np.arange(count)+.5)*len(train)/count).astype(int)]
    image=source.image_batch(rows,allowed_rows=train,device=device)
    genes,ni,ng,active,negative_audit=_batch(source,stats,rows,maps,31,device)
    require(active.any() and (ni>=0).any() and (ng>=0).any(), "S5_NO_PAIRABLE_PROBE_SUPPORT")
    target=torch.as_tensor(stats.targets_z(source,rows),dtype=torch.float32,device=device)
    mask=torch.as_tensor(source.target_mask[rows],device=device)
    labels=torch.as_tensor(source.tissue[rows],dtype=torch.long,device=device)
    weights=torch.as_tensor(stats.class_weights,dtype=torch.float32,device=device)
    profiles=[]
    for arm in ARMS:
        model=_model(source,stats,arm,31,maps,device);model.eval()
        before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        start=time.monotonic()
        outputs=model.image_only_predict(image);prediction=outputs["program_predictions"]
        repeated=model.image_only_predict(image)["program_predictions"]
        require(torch.equal(prediction,repeated), "S5_NONDETERMINISTIC_IMAGE_FORWARD")
        prediction.square().mean().backward()
        require(all(p.grad is None for p in model.gene_encoder.parameters()), "S5_RNA_IN_IMAGE_PREDICTION")
        model.zero_grad(set_to_none=True);outputs=model.image_only_predict(image)
        loss,loss_audit=objective(model,outputs,target,mask,labels,weights,genes,ni,ng,active)
        loss.backward();grads=_gradients(model)
        for name,value in grads.items():require(value is None or value>0, "S5_ZERO_ACTIVE_QUERY_GRADIENT")
        require(all(torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()),
            "S5_ZERO_OPTIMIZER_PROFILE_MUTATED_STATE")
        profiles.append(dict(arm=arm,parameters=model.descriptor()["parameters"],query_gradient_norms=grads,
            active_image_anchors=loss_audit["contrastive"]["active_image_anchors"],
            active_gene_anchors=loss_audit["contrastive"]["active_gene_anchors"],
            train_loss=float(loss.detach()),elapsed_seconds=time.monotonic()-start,
            optimizer_instances=0,optimizer_updates=0,state_unchanged=True))
        del model
    require(source_hashes()==sources and inherited.source_hashes()==old_sources, "S5_SOURCES_CHANGED_DURING_QUALIFICATION")
    update_budget=sum(x["updates_per_fit"] for x in partitions)*7
    require(update_budget==72800, "S5_ACTUAL_UPDATE_BUDGET_DIFFERS")
    profile=dict(schema="spgel_s5_real_qualification_v1",status="PASS_REAL_DATA_IMAGE_ONLY_AND_ALL_EFFECTIVE_PAIRED_PATHS",
        profiles=profiles,probe_rows_sha256=array_hash(rows),probe_count=len(rows),negative_audit=negative_audit,
        new_optimizer_fits=0,optimizer_instances=0,optimizer_updates=0)
    write_json(private/"QUALIFICATION.private.json",profile)
    lock=dict(schema="spgel_s5_run_lock_v1",status="LOCKED_BEFORE_84_NEW_FITS",config_sha256=sha256(config_path),
        sources=sources,inherited_sources=old_sources,partitions=partitions,
        reference_source_binding=read_json(ROOT/config["reference_source_recipe"])["source_binding"],
        qualification_sha256=sha256(private/"QUALIFICATION.private.json"),budget=config["budget"],
        no_survival_or_R8_inputs=True,source_development_donors_not_new_external_confirmation=True)
    lock["contract_sha256"]=canonical_hash(lock);write_json(private/"S5_RUN_LOCK.json",lock)
    return lock


def validate_lock(config_path,private):
    config=recipe(config_path);lock=read_json(Path(private)/"S5_RUN_LOCK.json")
    require(lock["status"]=="LOCKED_BEFORE_84_NEW_FITS" and lock["config_sha256"]==sha256(config_path)
        and lock["sources"]==source_hashes() and lock["inherited_sources"]==inherited.source_hashes()
        and lock["contract_sha256"]==canonical_hash({k:v for k,v in lock.items() if k!="contract_sha256"})
        and lock["qualification_sha256"]==sha256(Path(private)/"QUALIFICATION.private.json"),
        "S5_LOCK_OR_SOURCES_CHANGED")
    return lock,config


def fit_cell(source,config_path,private,*,arm,donor,seed,device="cpu"):
    inherited.configure_runtime();lock,config=validate_lock(config_path,private)
    train,held=source.split(donor);name=unit_name(arm,donor,seed)
    folder=Path(private)/"cells"/name
    require(not folder.exists(), "S5_EXISTING_CELL_NO_AUTOMATIC_RETRY")
    folder.mkdir(parents=True);started=time.monotonic();updates=0
    try:
        stats=SourceStatistics.fit(source,train,heldout_donors=[donor]);maps,part=mapping(private,lock,donor,seed)
        require(array_hash(train)==part["train_rows_sha256"] and array_hash(held)==part["heldout_rows_sha256"],
            "S5_PARTITION_CHANGED")
        model=_model(source,stats,arm,seed,maps,device)
        initial={k:array_hash(v.detach().cpu().numpy()) for k,v in model.state_dict().items()}
        optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
        weights=torch.as_tensor(stats.class_weights,dtype=torch.float32,device=device)
        snapshots=[];curves=[];saved_arrays={}
        if str(device).startswith("cuda"):torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(1,41):
            model.train()
            # This ordering also matches the old source permutation; pairing
            # support, not image sample order, is the new common S5 protocol.
            order=np.random.default_rng(np.random.SeedSequence([seed,epoch,donor+1,71001])).permutation(train)
            means=dict(total=0.,prediction_Huber=0.,tissue_CE=0.,ITC=0.)
            counts=dict(active_image_anchors=0,active_gene_anchors=0,batch_rows=0)
            grads=dict(gene_to_image=0.,image_to_gene=0.);negative_hash=hashlib.sha256()
            for batch_id,start in enumerate(range(0,len(order),512)):
                rows=order[start:start+512]
                image=source.image_batch(rows,allowed_rows=train,device=device)
                target=torch.as_tensor(stats.targets_z(source,rows),dtype=torch.float32,device=device)
                mask=torch.as_tensor(source.target_mask[rows],device=device)
                tissue=torch.as_tensor(source.tissue[rows],dtype=torch.long,device=device)
                batch_seed=int(np.random.SeedSequence([seed,epoch,batch_id,donor+1]).generate_state(1)[0])
                genes,ni,ng,active,negative_audit=_batch(source,stats,rows,maps,batch_seed,device)
                negative_hash.update(np.ascontiguousarray(rows,np.int64).tobytes())
                negative_hash.update(ni.detach().cpu().numpy().tobytes());negative_hash.update(ng.detach().cpu().numpy().tobytes())
                optimizer.zero_grad(set_to_none=True);outputs=model.image_only_predict(image)
                loss,audit=objective(model,outputs,target,mask,tissue,weights,genes,ni,ng,active)
                require(torch.isfinite(loss), "S5_NONFINITE_LOSS")
                loss.backward()
                qgrad=_gradients(model,allow_missing=not (audit["contrastive"]["active_image_anchors"]
                    or audit["contrastive"]["active_gene_anchors"]))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                require(torch.isfinite(norm), "S5_NONFINITE_GRADIENT")
                optimizer.step();updates+=1
                proportion=len(rows)/len(train);means["total"]+=float(loss.detach())*proportion
                for key in ("prediction_Huber","tissue_CE","ITC"):means[key]+=float(audit[key].detach())*proportion
                for key in grads:grads[key]+=(qgrad[key] or 0.)*proportion
                for key in ("active_image_anchors","active_gene_anchors"):counts[key]+=audit["contrastive"][key]
                counts["batch_rows"]+=len(rows)
            require(updates==epoch*math.ceil(len(train)/512), "S5_UPDATE_COUNTER_CHANGED")
            curve=dict(epoch=epoch,optimizer_updates=updates,losses=means,query_gradient_norms_mean=grads,
                paired_support=counts,negative_candidate_sha256=negative_hash.hexdigest())
            curves.append(curve)
            write_json(folder/"PROGRESS.json",dict(status="RUNNING",arm=arm,donor=donor,seed=seed,
                epoch=epoch,optimizer_updates=updates,elapsed_seconds=time.monotonic()-started),progress=True)
            if epoch not in (10,20,40):continue
            label=f"epoch{epoch}";path=folder/f"checkpoint_{label}.pt"
            inherited.torch_write(path,dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                optimizer_state=optimizer.state_dict(),stats=asdict(stats),arm=arm,donor=donor,seed=seed,epoch=epoch,
                optimizer_updates=updates,contract_sha256=lock["contract_sha256"],
                mapping_sha256=part["mapping_sha256"],config_sha256=sha256(config_path),
                train_rows_sha256=array_hash(train),heldout_rows_sha256=array_hash(held)))
            snapshot,arrays=inherited._snapshot(model,source,train,held,stats,None,device=device,
                epoch=epoch,stage="S5",checkpoint_label=label,epoch_losses=means,include_retrieval=epoch==40)
            snapshot["checkpoint_sha256"]=sha256(path)
            if epoch==40:
                from .diagnostics import paired_diagnostics
                snapshot["paired_path_diagnostics"]=paired_diagnostics(model,source,held,stats,device=device)
            snapshots.append(snapshot);saved_arrays[label]=arrays
        # Prediction arrays are separate from weights and stay private.
        prediction_path=folder/"predictions.private.npz"
        with prediction_path.open("xb") as stream:
            np.savez_compressed(stream,**{label+"::"+key:value for label,arrays in saved_arrays.items() for key,value in arrays.items()})
        done=dict(schema="spgel_s5_completed_cell_v1",status="COMPLETE_ONE_LOCKED_S5_FIT",arm=arm,donor=donor,seed=seed,
            contract_sha256=lock["contract_sha256"],config_sha256=sha256(config_path),
            mapping_sha256=part["mapping_sha256"],train_rows=len(train),heldout_rows=len(held),
            train_rows_sha256=array_hash(train),heldout_rows_sha256=array_hash(held),descriptor=model.descriptor(),
            initial_state_tensor_sha256=initial,snapshots=snapshots,learning_curve=curves,
            predictions_sha256=sha256(prediction_path),actual_neural_fits=1,actual_optimizer_updates=updates,
            actual_saved_states=3,elapsed_seconds=time.monotonic()-started,
            peak_torch_GPU_bytes=int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") else 0,
            wrong_pairing_changes_main_targets=False,image_only_inference_uses_RNA=False,
            new_external_confirmation=False,automatic_retry_fits=0)
        write_json(folder/"COMPLETE.json",done);return done
    except BaseException as error:
        write_json(folder/"ERROR_RECEIPT.private.json",dict(status="FAILED_NO_AUTOMATIC_RETRY",failure_type=type(error).__name__,
            actual_optimizer_updates=updates,automatic_retry_fits=0))
        (folder/"error.private.log").write_text(traceback.format_exc(),encoding="utf-8");raise


def replay_cell(source,config_path,private,*,arm,donor,seed,device="cpu"):
    inherited.configure_runtime();lock,_=validate_lock(config_path,private)
    folder=Path(private)/"cells"/unit_name(arm,donor,seed);done=read_json(folder/"COMPLETE.json")
    train,held=source.split(donor);stats=SourceStatistics.fit(source,train,heldout_donors=[donor])
    maps,part=mapping(private,lock,donor,seed)
    require(done["status"]=="COMPLETE_ONE_LOCKED_S5_FIT" and done["arm"]==arm and done["donor"]==donor and done["seed"]==seed
        and done["config_sha256"]==sha256(config_path) and done["mapping_sha256"]==part["mapping_sha256"]
        and done["train_rows_sha256"]==array_hash(train) and done["heldout_rows_sha256"]==array_hash(held)
        and done["contract_sha256"]==lock["contract_sha256"] and done["predictions_sha256"]==sha256(folder/"predictions.private.npz")
        and done["actual_optimizer_updates"]==part["updates_per_fit"] and len(done["snapshots"])==3,
        "S5_COMPLETE_OR_PREDICTIONS_CHANGED")
    require([(x["label"],x["epoch"]) for x in done["snapshots"]]==[("epoch10",10),("epoch20",20),("epoch40",40)],
        "S5_EXACT_THREE_SNAPSHOT_IDENTITIES_REQUIRED")
    model=_model(source,stats,arm,seed,maps,device);records=[]
    with np.load(folder/"predictions.private.npz",allow_pickle=False) as reference:
        for stored in done["snapshots"]:
            label=stored["label"];path=folder/f"checkpoint_{label}.pt"
            require(sha256(path)==stored["checkpoint_sha256"], "S5_CHECKPOINT_BYTES_CHANGED")
            state=torch.load(path,map_location="cpu",weights_only=False)
            require(state["arm"]==arm and state["donor"]==donor and state["seed"]==seed
                and state["config_sha256"]==sha256(config_path) and state["epoch"]==stored["epoch"]
                and state["contract_sha256"]==lock["contract_sha256"] and state["mapping_sha256"]==part["mapping_sha256"]
                and state["train_rows_sha256"]==array_hash(train) and state["heldout_rows_sha256"]==array_hash(held)
                and state["optimizer_updates"]==state["epoch"]*math.ceil(len(train)/512), "S5_CHECKPOINT_LINEAGE_CHANGED")
            for key,value in asdict(stats).items():
                actual=state["stats"][key]
                require(np.array_equal(actual,value) if isinstance(value,np.ndarray) else actual==value,
                    "S5_REPLAY_TRAIN_STATISTICS_CHANGED")
            model.load_state_dict(state["state_dict"],strict=True)
            current,arrays=inherited._snapshot(model,source,train,held,stats,None,device=device,epoch=state["epoch"],stage="S5",
                checkpoint_label=label,epoch_losses=stored["losses"],include_retrieval=state["epoch"]==40)
            error=max(float(np.max(np.abs(value-reference[label+"::"+key]),initial=0.)) for key,value in arrays.items())
            metric_error=max(inherited._metric_difference(current[split],stored[split]) for split in ("train","heldout"))
            require(error<=2e-6 and metric_error<=1e-8, "S5_ACTUAL_STATE_REPLAY_FAILED")
            if state["epoch"]==40:
                from .diagnostics import paired_diagnostics
                diagnostic=paired_diagnostics(model,source,held,stats,device=device)
                require(inherited._metric_difference(diagnostic,stored["paired_path_diagnostics"])<=1e-8,
                    "S5_PAIRED_DIAGNOSTIC_REPLAY_FAILED")
            records.append(dict(epoch=state["epoch"],checkpoint_sha256=sha256(path),actual_saved_state_loaded=True,
                actual_image_and_native_pair_forward_replayed=True,maximum_prediction_error=error,
                maximum_metric_error=metric_error,new_optimizer_fits=0,new_optimizer_updates=0))
    result=dict(schema="spgel_s5_cell_replay_v1",status="PASS",arm=arm,donor=donor,seed=seed,
        contract_sha256=lock["contract_sha256"],COMPLETE_sha256=sha256(folder/"COMPLETE.json"),
        predictions_sha256=done["predictions_sha256"],states=records,new_optimizer_fits=0,new_optimizer_updates=0)
    destination=Path(private)/"replay";destination.mkdir(exist_ok=True)
    write_json(destination/(unit_name(arm,donor,seed)+".json"),result);return result

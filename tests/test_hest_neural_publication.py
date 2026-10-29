"""Administrative boundary tests with anonymous JSON; no model fits or arrays."""
import copy
import csv
import json
from pathlib import Path

import pytest

from scripts import hest_neural_publication as p


def write(root, name, value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8")
    return path


@pytest.fixture
def complete(tmp_path,monkeypatch):
    root=tmp_path;private=root/"private";private.mkdir()
    names=["configs/spgel_hest_neural/pilot_v1.json"]+[f"src/spgel_hest_neural/f{i}.py" for i in range(10)]
    for name in names:
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text("{}")
    sources={n:p.sha(root/n) for n in names};monkeypatch.setattr(p,"SOURCE_SHA",p.canonical(sources))
    old={}
    for i in range(11):
        name=f"old/f{i}.py";dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text("original")
        old[name]=p.sha(dest)
    monkeypatch.setattr(p.prior,"RIDGE_SOURCE_SHA",p.canonical(old))
    freeze=write(root,"freeze.json",dict(sources=sources,sources_canonical_sha256=p.SOURCE_SHA))
    parts=[dict(dataset=d,fold=f,input_cache_sha256=p.canonical([d,f])) for d in ("COAD","READ") for f in (0,1)]
    lock=dict(schema="spgel_hest_neural_run_lock_v1",status="LOCKED_BEFORE_TWELVE_NEURAL_FITS",
        source_files=sources,config_sha256=p.sha(root/names[0]),assets_sha256="a"*64,
        actual_preparation_new_PCA_fits=4,actual_preparation_new_scaler_fits=0,
        actual_preparation_new_Ridge_fits=0,actual_preparation_new_neural_fits=0,
        inherited_Ridge_source_files=old,partitions=parts)
    lock["internal_contract_sha256"]=p.canonical(lock)
    write(private,"HEST_NEURAL_RUN_LOCK.json",lock);contract=lock["internal_contract_sha256"]
    public=[];units=[]
    for d,f,a in sorted(p.EXPECTED):
        cache=next(x["input_cache_sha256"] for x in parts if (x["dataset"],x["fold"])==(d,f))
        done=dict(schema="spgel_hest_neural_cell_v1",status="COMPLETE_ONE_FIXED40_NEURAL_FIT",
            dataset=d,fold=f,arm=a,seed=1,internal_contract_sha256=contract,
            actual_epochs=40,actual_optimizer_updates=40,new_neural_fits=1,new_Ridge_fits=0,new_scaler_fits=0,
            new_PCA_fits=0,input_cache_sha256=cache,train_loss_only=True,test_RNA_in_predict=False,
            test_labels_select_epoch=False,source_pretrained_weights_loaded=False,
            learning_curve=[dict(epoch=i) for i in range(1,41)],
            snapshots=[dict(epoch=e,state_sha256=p.canonical([d,f,a,e,"state"]),
                predictions_sha256=p.canonical([d,f,a,e,"prediction"])) for e in (10,20,40)])
        cp=write(private,f"cells/{d}_fold{f}_seed1_{a}/COMPLETE.json",done)
        public.append({k:done[k] for k in ("dataset","fold","arm","seed","learning_curve","snapshots")})
        for s in done["snapshots"]:
            units.append(dict(dataset=d,fold=f,arm=a,seed=1,epoch=s["epoch"],status="PASS",
                actual_saved_state_loaded=True,actual_train_and_test_forward_replayed=True,
                new_neural_fits=0,new_Ridge_fits=0,new_PCA_fits=0,COMPLETE_sha256=p.sha(cp),
                state_sha256=s["state_sha256"],predictions_sha256=s["predictions_sha256"],
                input_cache_sha256=cache,maximum_prediction_error=0.))
    replay=dict(schema="spgel_hest_neural_replay_v1",status="PASS_36_ACTUAL_SAVED_STATE_REPLAYS",
        internal_contract_sha256=contract,passed_states=36,passed_fits=12,units=units,
        new_neural_fits=0,new_Ridge_fits=0,new_PCA_fits=0,optimizer_updates=0,model_or_epoch_selection_performed=False)
    rp=write(private,"NEURAL_REPLAY_AUDIT.json",replay)
    report=dict(schema="spgel_hest_neural_fixed12_report_v1",status="COMPLETE_12_FIXED40_FITS_AND_36_ACTUAL_REPLAYS",
        code_sha256=sources,config_sha256=lock["config_sha256"],assets_sha256=lock["assets_sha256"],
        internal_contract_sha256=contract,replay_audit_sha256=p.sha(rp),per_cell=public,
        maximum_actual_replay_error=0.,ledger=dict(new_neural_fits=12,actual_optimizer_updates=480,
            actual_saved_state_replays=36,new_unsupervised_PCA_reconstructions=4,new_scaler_fits=0,
            new_Ridge_fits=0,old_source_weight_loads=0,automatic_retry_fits=0),
        scope=dict(primary_epoch=40,snapshot10_20_40_diagnostic_only=True,fixed40_budget_proves_convergence=False,
            test_labels_in_predict_or_epoch_selection=False,new_independent_cohort_claimed=False,
            READ_donors_previously_used_in_source_development=True,COAD_preview_donor_identity="UNCONFIRMED",
            graph_versus_MLP_difference_exclusively_attributed_to_topology=False))
    write(root,p.REPORT,report)
    return root,private,freeze


def change_replay(fixture,change):
    root,private,_=fixture;rp=private/"NEURAL_REPLAY_AUDIT.json"
    replay=p.read(rp);change(replay);write(private,rp.name,replay)
    report=p.read(root/p.REPORT);report["replay_audit_sha256"]=p.sha(rp);write(root,p.REPORT,report)


def test_complete_metadata_receipts_do_not_require_or_read_state_files(complete):
    root,private,freeze=complete
    result=p.metadata_evidence(root,private,freeze)
    assert result["actual_neural_fits"]==12 and result["actual_state_replays"]==36
    assert result["arrays_or_models_read_by_publication"] is False
    assert not list(private.rglob("*.pt")) and not list(private.rglob("*.npz"))


@pytest.mark.parametrize("key,value",[("actual_saved_state_loaded",False),
    ("new_neural_fits",1),("state_sha256","0"*64),("maximum_prediction_error",-1.),
    ("maximum_prediction_error",2e-6)])
def test_forged_replay_cannot_be_accepted_by_updating_only_report_hash(complete,key,value):
    change_replay(complete,lambda r:r["units"][0].__setitem__(key,value))
    with pytest.raises(ValueError,match="ACTUAL_NN_REPLAY_FLAGS_OR_ARTIFACT_SHA_CHANGED"):
        p.metadata_evidence(*complete)


def test_35_replays_cannot_claim_36_by_retaining_counters(complete):
    change_replay(complete,lambda r:r["units"].pop())
    with pytest.raises(ValueError,match="36_ACTUAL_REPLAYS_REQUIRED"):
        p.metadata_evidence(*complete)


def test_public_learning_curve_must_equal_actual_completion(complete):
    root,_,_=complete;report=p.read(root/p.REPORT)
    report["per_cell"][0]["learning_curve"][0]["epoch"]=99;write(root,p.REPORT,report)
    with pytest.raises(ValueError,match="COMPLETE_OR_PUBLIC_REPORT_DRIFT"):
        p.metadata_evidence(*complete)


def test_fixed_budget_cannot_be_renamed_convergence_or_external_validation(complete):
    root,_,_=complete;report=p.read(root/p.REPORT)
    report["scope"]["new_independent_cohort_claimed"]=True;write(root,p.REPORT,report)
    with pytest.raises(ValueError,match="UNSUPPORTED_SELECTION_CONVERGENCE_OR_INDEPENDENCE_CLAIM"):
        p.metadata_evidence(*complete)


def test_source_change_after_completion_still_fails(complete):
    root,_,_=complete;(root/"src/spgel_hest_neural/f0.py").write_text("changed")
    with pytest.raises(ValueError,match="FROZEN_NN_SOURCE_CHANGED"):
        p.metadata_evidence(*complete)


def test_local_only_scan_keeps_old_array_rule(tmp_path):
    name="reports/spgel/HEST_NEURAL_FIXED12_ANON_20261029.json"
    path=write(tmp_path,name,dict(predictions=[1.,2.]))
    row=dict(file=name,sha256=p.sha(path),bytes=path.stat().st_size)
    with pytest.raises(ValueError,match="private row-level JSON field"):
        p.local_scan(tmp_path,[row])


@pytest.mark.parametrize("name",["../private.json","reports/spgel/S3_RESULTS.md",
    "src/spgel_hest/execution.py","src/spgel_hest_neural/model.pt"])
def test_new_namespace_never_authorizes_old_science_or_binary(name):
    with pytest.raises(ValueError):p.safe_name(name)


@pytest.fixture
def proof_ready(complete):
    root,private,freeze=complete
    evidence=p.metadata_evidence(*complete)
    dest=root/p.REVIEW_SCRIPT;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text("# independent synthetic review")
    outputs={}
    for name,count in ((p.FOLD_CSV,36),(p.GENE_CSV,1800)):
        with (root/name).open("w",newline="") as f:
            writer=csv.writer(f);writer.writerow(["aggregate_unit"])
            writer.writerows([[i] for i in range(count)])
        outputs[Path(name).name]=dict(sha256=p.sha(root/name),rows=count)
    write(root,p.REVIEW,dict(schema="hest_neural_fixed12_metadata_arithmetic_audit_v1",
        status="PASS_12_FITS_480_UPDATES_36_ACTUAL_REPLAYS_AND_METRIC_ARITHMETIC",
        independent_review_script_sha256=p.sha(root/p.REVIEW_SCRIPT),
        producer_report_sha256=evidence["public_report_sha256"],neural_lock_file_sha256=evidence["lock_sha256"],
        actual_replay_file_sha256=evidence["actual_replay_sha256"],
        source_freeze_manifest_sha256=evidence["source_freeze_manifest_sha256"],
        source_canonical_sha256=p.SOURCE_SHA,internal_contract_sha256=evidence["internal_contract_sha256"],
        actual_counts=dict(neural_fits=12,optimizer_updates=480,saved_state_replays=36,
            new_unsupervised_PCA_reconstructions=4,new_Ridge_fits=0),
        reviewer_operations=dict(model_fits=0,model_forwards=0,PCA_fits=0,
            model_or_array_artifact_deserializations=0,current_artifact_file_hashing=True,
            actual_replays_run_by_producer_not_reviewer=True),outputs=outputs))
    cells=[dict(dataset=d,fold=f,arm=a,seed=1,actual_neural_fits=1,actual_epochs=40,
        optimizer_updates=40,saved_epoch_states=[10,20,40],
        COMPLETE_sha256=evidence["complete_file_sha256"][f"{d}_fold{f}_seed1_{a}"])
        for d,f,a in sorted(p.EXPECTED)]
    write(root,p.LEDGER,dict(schema="hest_neural_actual_execution_ledger_v1",
        status="PASS_12_ACTUAL_FIXED40_FITS_480_UPDATES_36_ACTUAL_REPLAYS_4_PCA",
        actual_neural_fits=12,actual_optimizer_updates=480,actual_saved_state_replays=36,
        new_unsupervised_PCA_reconstructions=4,new_scaler_fits=0,new_Ridge_fits=0,automatic_retry_fits=0,
        source_canonical_sha256=p.SOURCE_SHA,lock_file_sha256=evidence["lock_sha256"],
        replay_file_sha256=evidence["actual_replay_sha256"],public_report_sha256=evidence["public_report_sha256"],
        internal_contract_sha256=evidence["internal_contract_sha256"],expression_or_clinical_transferred=False,
        common_in_out_initialization_actual_equal=True,graph_identity_initialization_actual_equal=True,
        three_arms_share_one_cache_per_fold=True,cells=cells))
    return complete


def test_complete_requires_independent_csv_and_ledger_bindings(proof_ready):
    result=p.completion_bundle(*proof_ready)
    assert result["status"]=="PASS_ACTUAL_NN_COMPLETION_AND_INDEPENDENT_REVIEW_BINDINGS"
    assert len(result["public_proof_sha256"])==6


def test_post_review_csv_replacement_with_same_row_count_is_rejected(proof_ready):
    root,_,_=proof_ready;path=root/p.GENE_CSV
    text=path.read_text();path.write_text(text.replace("aggregate_unit","aggregate_fake"))
    with pytest.raises(ValueError,match="NN_CSV_BYTES_OR_COUNTS_CHANGED"):
        p.completion_bundle(*proof_ready)


def test_ledger_cannot_credit_a_different_complete_certificate(proof_ready):
    root,_,_=proof_ready;ledger=p.read(root/p.LEDGER)
    ledger["cells"][0]["COMPLETE_sha256"]="0"*64;write(root,p.LEDGER,ledger)
    with pytest.raises(ValueError,match="LEDGER_COMPLETE_OR_UPDATE_COUNTS_CHANGED"):
        p.completion_bundle(*proof_ready)

"""Pure synthetic receipts: no real data, model load, forward or optimization."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import spgel_s5_report as report
from tests import test_spgel_report as legacy


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


def token(value, probe=128):
    return dict(RMS=value,valid_spot_programs=4*probe,
        per_program=[dict(program_index=k,valid_spots=probe,RMS=value) for k in range(4)])


def attention(value, probe=128):
    return dict(mean_row_L1=value,maximum_absolute_difference=value/4,
        valid_spot_programs=4*probe,
        per_program=[dict(program_index=k,valid_spots=probe,mean_row_L1=value) for k in range(4)])


def diagnostic(arm, held_sha):
    gi=arm not in ("M0","M1","M4");ig=arm not in ("M0","M1","M3")
    return dict(status="COMPLETE_THREE_FROZEN_TASK_B_RUNS",probe_count=128,paired_score_calls=3,
        optimizer_instances=0,optimizer_updates=0,backward_calls=0,
        target_tissue_library_or_survival_reads=False,model_state_unchanged=True,
        existing_gradients_unchanged=True,training_modes_restored=True,
        model_state_sha256="f"*64,partition_rows_sha256=held_sha,
        gene_to_image=dict(base_to_conditioned=token(.1 if arm not in ("M0","M4") else 0.),
            changed_gene_context=token(.05 if gi else 0.),base_attention_change=attention(.1),
            changed_gene_attention=attention(.01 if gi else 0.)),
        image_to_gene=dict(base_to_conditioned=token(.1 if arm not in ("M0","M3") else 0.),
            changed_image_context=token(.05 if ig else 0.),base_up_attention_change=attention(.1),
            base_down_attention_change=attention(.1),changed_image_up_attention=attention(.01 if ig else 0.),
            changed_image_down_attention=attention(.01 if ig else 0.)),
        paired_score_change={key:dict(valid_pairs=128,mean_difference=.01,mean_absolute_difference=.02,RMS=.02)
            for key in ("rolled_genes","rolled_images")})


@pytest.fixture
def matrix(tmp_path,monkeypatch):
    cfg_path=report.ROOT/"configs/spgel_s5/mechanism_v1.json"
    config=report.read_json(cfg_path);cfg_sha=report.sha(cfg_path)
    old=report.read_json(report.ROOT/"reports/spgel/SOURCE_RUN_LOCK.json")
    reference=report.read_json(report.ROOT/config["reference_source_recipe"])
    private=tmp_path/"synthetic_private"
    (private/"cells").mkdir(parents=True);(private/"replay").mkdir()
    profiles=[]
    for arm in report.ARMS:
        gradients={d:(None if arm=="M0" or (arm=="M3" and d=="image_to_gene")
            or (arm=="M4" and d=="gene_to_image") else .1) for d in ("gene_to_image","image_to_gene")}
        profiles.append(dict(arm=arm,parameters=report.PARAMETERS[arm],state_unchanged=True,
            optimizer_instances=0,optimizer_updates=0,query_gradient_norms=gradients))
    q=dict(status="PASS_REAL_DATA_IMAGE_ONLY_AND_ALL_EFFECTIVE_PAIRED_PATHS",profiles=profiles,
        new_optimizer_fits=0,optimizer_instances=0,optimizer_updates=0)
    dump(private/"QUALIFICATION.private.json",q)
    partitions=[]
    for donor in range(6):
        for seed in report.SEEDS:
            count=13011-report.SIZES[donor]
            path=private/"qualification"/f"donor{donor}_seed{seed}.private.npz"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"SYNTHETIC_MAPPING_HASH_ONLY_{donor}_{seed}".encode())
            train_sha=report.canonical(["fixture_train",donor]);held_sha=report.canonical(["fixture_held",donor])
            partitions.append(dict(donor=donor,seed=seed,train_rows=count,heldout_rows=report.SIZES[donor],
                updates_per_fit=40*math.ceil(count/512),mapping_sha256=report.sha(path),
                train_rows_sha256=train_sha,heldout_rows_sha256=held_sha,
                pairing_audit=dict(train_rows_sha256=train_sha,training_rows=count,seed=seed,
                    all_arms_share_supported_anchors=True,regression_and_tissue_truth_unchanged=True,
                    same_tissue_and_train_depth_quartile=True,heldout_metadata_used=False,
                    candidate_gene_values_or_program_targets_used=False,ITC_active_rows=count,
                    groups=[dict(training_rows=count,ITC_active_rows=count)]),
                membership_audit=dict(train_rows_sha256=train_sha,seed=seed,heldout_measurements_used=False,
                    source_arrays_mutated=False,values_gene_IDs_outputs_and_medical_truth_columns_permuted=False,
                    signed_row_sizes_exactly_preserved=True,all_signed_pairwise_overlaps_exactly_preserved=True,
                    permutation_keeps_coverage_mean_sd_bins=True,changed_signed_membership_cells=2)))
    lock=dict(schema="spgel_s5_run_lock_v1",status="LOCKED_BEFORE_84_NEW_FITS",config_sha256=cfg_sha,
        budget=config["budget"],no_survival_or_R8_inputs=True,
        source_development_donors_not_new_external_confirmation=True,
        sources={name:report.sha(report.ROOT/name) for name in report.SOURCE_FILES},
        inherited_sources=old["sources"],reference_source_binding=reference["source_binding"],
        qualification_sha256=report.sha(private/"QUALIFICATION.private.json"),partitions=partitions)
    lock["contract_sha256"]=report.canonical(lock);dump(private/"S5_RUN_LOCK.json",lock)
    by_part={(p["donor"],p["seed"]):p for p in partitions}
    offsets=dict(M0=0.,M1=.01,M2=.03,M3=.015,M4=.02,M5=-.02,M6=.005)
    for arm in report.ARMS:
        for donor in range(6):
            for seed in report.SEEDS:
                part=by_part[donor,seed];key=arm,donor,seed;folder=private/"cells"/report.name(key)
                folder.mkdir()
                pcc=(donor+1)/10+offsets[arm]+(.02 if seed==47 else 0.)
                snapshots=[]
                for epoch in (10,20,40):
                    label=f"epoch{epoch}";p=folder/f"checkpoint_{label}.pt"
                    p.write_bytes(f"SYNTHETIC_STATE_HASH_ONLY_{arm}_{donor}_{seed}_{epoch}".encode())
                    h=legacy.heldout(part["heldout_rows"],donor,"S_A5",final=epoch==40)
                    h["task_A"]=legacy.task_a(part["heldout_rows"],pcc)
                    h["task_C"]=legacy.task_c(part["heldout_rows"],pcc)
                    state=dict(label=label,epoch=epoch,checkpoint_sha256=report.sha(p),
                        train=dict(task_A=legacy.task_a(part["train_rows"],.7)),heldout=h,
                        losses=dict(total=.5,prediction_Huber=.3,tissue_CE=1.,ITC=1.))
                    if epoch==40:state["paired_path_diagnostics"]=diagnostic(arm,part["heldout_rows_sha256"])
                    snapshots.append(state)
                pred=folder/"predictions.private.npz";pred.write_bytes(b"SYNTHETIC_PREDICTION_HASH_ONLY")
                keys=("backbone.weight","gene_encoder.up","gene_encoder.down","gene_to_image.weight","image_to_gene.weight")
                initial={name:report.canonical([donor,seed,name]) for name in keys
                    if not (arm in ("M0","M4") and name=="gene_to_image.weight")
                    and not (arm in ("M0","M3") and name=="image_to_gene.weight")}
                if arm=="M6":initial["gene_encoder.up"]=report.canonical(["null",donor,seed])
                curves=[dict(epoch=e,optimizer_updates=e*math.ceil(part["train_rows"]/512),
                    losses=dict(total=.5,prediction_Huber=.3,tissue_CE=1.,ITC=1.),
                    query_gradient_norms_mean=dict(gene_to_image=0. if arm in ("M0","M4") else .1,
                        image_to_gene=0. if arm in ("M0","M3") else .1),
                    paired_support=dict(batch_rows=part["train_rows"],active_image_anchors=part["train_rows"],
                        active_gene_anchors=part["train_rows"]),
                    negative_candidate_sha256=report.canonical([donor,seed,e])) for e in range(1,41)]
                cell=dict(schema="spgel_s5_completed_cell_v1",status="COMPLETE_ONE_LOCKED_S5_FIT",
                    arm=arm,donor=donor,seed=seed,contract_sha256=lock["contract_sha256"],config_sha256=cfg_sha,
                    mapping_sha256=part["mapping_sha256"],train_rows_sha256=part["train_rows_sha256"],
                    heldout_rows_sha256=part["heldout_rows_sha256"],train_rows=part["train_rows"],heldout_rows=part["heldout_rows"],
                    actual_neural_fits=1,actual_optimizer_updates=part["updates_per_fit"],actual_saved_states=3,
                    automatic_retry_fits=0,wrong_pairing_changes_main_targets=False,image_only_inference_uses_RNA=False,
                    new_external_confirmation=False,descriptor=dict(arm=arm,parameters=report.PARAMETERS[arm],
                        registered_parameters=report.PARAMETERS[arm],image_only_prediction_uses_conditional_reread=False,
                        two_anchor_contrastive_losses_preserved=True),
                    predictions_sha256=report.sha(pred),snapshots=snapshots,learning_curve=curves,
                    initial_state_tensor_sha256=initial,elapsed_seconds=1.,peak_torch_GPU_bytes=1024)
                dump(folder/"COMPLETE.json",cell)
                replay=dict(schema="spgel_s5_cell_replay_v1",status="PASS",arm=arm,donor=donor,seed=seed,
                    contract_sha256=lock["contract_sha256"],COMPLETE_sha256=report.sha(folder/"COMPLETE.json"),
                    predictions_sha256=cell["predictions_sha256"],new_optimizer_fits=0,new_optimizer_updates=0,
                    states=[dict(epoch=x["epoch"],checkpoint_sha256=x["checkpoint_sha256"],actual_saved_state_loaded=True,
                        actual_image_and_native_pair_forward_replayed=True,new_optimizer_fits=0,new_optimizer_updates=0,
                        maximum_prediction_error=1e-8,maximum_metric_error=1e-12) for x in snapshots])
                dump(private/"replay"/(report.name(key)+".json"),replay)
    # Only independent source-byte qualification is mocked for this receipt
    # fixture; source_assets() itself has a separate byte-hashing test below.
    monkeypatch.setattr(report,"source_assets",lambda *a:dict(synthetic_fixture_only=True,source_arrays_deserialized=False))
    return private,cfg_path,lock


def loaded(matrix):
    private,config,_=matrix
    return report.load_matrix(private,config,private/"source_pointer_not_read_in_fixture.json")


def test_complete_84_cells_252_actual_receipts_equal_donor_macro_and_safe_exports(matrix,tmp_path,monkeypatch):
    monkeypatch.setattr(np,"load",lambda *a,**k:(_ for _ in ()).throw(AssertionError("ARRAY_DESERIALIZATION")))
    config,lock,cells,ledger=loaded(matrix)
    result=report.build_report(config,lock,cells,ledger)
    assert ledger["actual_neural_fits"]==84 and ledger["actual_optimizer_updates"]==72800
    assert ledger["actual_saved_state_replays"]==252 and ledger["checkpoint_or_array_deserializations_by_reporter"]==0
    m2=next(x for x in result["arm_summary"] if x["arm"]=="M2")
    assert m2["per_seed"][0]["task_A_Pearson"]["mean"]==pytest.approx(.38)
    assert m2["per_seed"][1]["task_A_Pearson"]["mean"]==pytest.approx(.40)
    assert m2["task_A_two_seed_mean"]["mean"]==pytest.approx(.39)
    weighted=np.average([.1*(d+1)+.03 for d in range(6)],weights=report.SIZES)
    assert abs(weighted-.38)>1e-3
    assert len(result["donor_rows"])==84 and len(result["program_rows"])==336
    assert len(result["checkpoint_diagnostics"])==252
    assert {x["reference"] for x in result["primary_evidence"]}=={"M1","M5"}
    out=tmp_path/"public";out.mkdir();(out/"S5A_IMPLEMENTATION_FIDELITY.md").write_text("preserve")
    receipt=report.write_report(out,result)
    assert receipt["per_donor_rows"]==84 and receipt["per_program_rows"]==336 and receipt["paired_rows"]==576
    assert (out/"S5A_IMPLEMENTATION_FIDELITY.md").read_text()=="preserve"
    assert len(list(out.iterdir()))==9
    assert report.restore_report(out/"S5A_RESULTS.json")==result
    assert all(path.stat().st_size<=2*1024**2 for path in out.glob("*.json"))
    public="\n".join(p.read_text(encoding="utf-8") for p in out.iterdir())
    assert str(matrix[0]) not in public and "SYNTHETIC_STATE_HASH_ONLY" not in public
    with pytest.raises(report.old.ReportContractError,match="EXISTING_RESULTS"):
        report.write_report(out,result)


@pytest.mark.parametrize("kind",("missing_replay","duplicate_epoch","no_actual_forward","refit","state_bytes"))
def test_partial_forged_or_modified_artifacts_do_not_allow_complete_report(matrix,kind):
    private,_,_=matrix;rp=private/"replay/M2_d0_seed31.json";r=report.read_json(rp)
    if kind=="missing_replay":rp.unlink()
    elif kind=="state_bytes":(private/"cells/M2_d0_seed31/checkpoint_epoch20.pt").write_bytes(b"changed")
    else:
        if kind=="duplicate_epoch":r["states"][1]["epoch"]=10
        if kind=="no_actual_forward":r["states"][0]["actual_image_and_native_pair_forward_replayed"]=False
        if kind=="refit":r["states"][0]["new_optimizer_fits"]=1
        dump(rp,r)
    with pytest.raises(report.old.ReportContractError):loaded(matrix)


@pytest.mark.parametrize("kind",("support","initial_state","parameters","M6_rename"))
def test_capacity_common_support_and_membership_null_are_actual_metadata_guards(matrix,kind):
    private,_,_=matrix;arm="M6" if kind=="M6_rename" else "M1"
    cp=private/"cells"/f"{arm}_d0_seed31/COMPLETE.json";r=report.read_json(cp)
    if kind=="support":r["learning_curve"][0]["paired_support"]["active_gene_anchors"]-=1
    if kind=="initial_state":r["initial_state_tensor_sha256"]["gene_to_image.weight"]="e"*64
    if kind=="parameters":r["descriptor"]["parameters"]+=1
    if kind=="M6_rename":r["initial_state_tensor_sha256"]["gene_encoder.up"]=report.read_json(
        private/"cells/M2_d0_seed31/COMPLETE.json")["initial_state_tensor_sha256"]["gene_encoder.up"]
    dump(cp,r);rp=private/"replay"/f"{arm}_d0_seed31.json";replay=report.read_json(rp)
    replay["COMPLETE_sha256"]=report.sha(cp);dump(rp,replay)
    with pytest.raises(report.old.ReportContractError):loaded(matrix)


def test_null_program_or_AUPRC_support_invalidates_strict_aggregate_without_donor_dropping(matrix):
    config,lock,cells,ledger=loaded(matrix)
    state=cells["M2",0,31]["snapshots"][-1]
    a=state["heldout"]["task_A"]
    for scale in ("native","train_Z"):
        a[scale]["per_output"][0]["Pearson"]=None;a[scale]["defined_Pearson_count"]=3
    a["defined_program_count"]=3;a["primary_donor_four_program_Pearson"]=None
    c=state["heldout"]["task_C"];c["per_output"][0].update(AUPRC=None,high_activity_count=0,prevalence=0.)
    c.update(primary_four_program_AUPRC=None,defined_AUPRC_count=3)
    result=report.build_report(config,lock,cells,ledger)
    summary=next(x for x in result["arm_summary"] if x["arm"]=="M2")
    assert summary["per_seed"][0]["task_A_Pearson"]["mean"] is None
    assert summary["per_seed"][0]["task_A_Pearson"]["defined_count"]==5
    assert summary["task_A_two_seed_mean"]["mean"] is None
    assert summary["task_C_two_seed_mean"]["mean"] is None
    assert result["primary_evidence"][0]["donor_bootstrap_CI95"] is None


def test_bootstrap_samples_six_donors_keeping_seed_pair_together_and_is_fixed():
    values=[[.01*(d+1) for d in range(6)],[-.02*(d+1) for d in range(6)]]
    result=report.paired_bootstrap(values)
    donors=np.asarray(values).mean(0)
    indices=np.random.default_rng(510531).integers(0,6,size=(2000,6))
    expected=np.quantile(donors[indices].mean(1),[.025,.975])
    assert result["bootstrap_replicates"]==2000 and result["bootstrap_seed"]==510531
    assert result["CI95"]==pytest.approx(expected)
    assert result["two_seed_mean"]["mean"]==pytest.approx(donors.mean())
    assert result==report.paired_bootstrap(values)
    assert report.paired_bootstrap([[None]*6,[.1]*6])["CI95"] is None


def test_report_sanitizes_diagnostic_metadata_and_rejects_impossible_M1_cross_path():
    raw=diagnostic("M1","a"*64)
    raw["patient_id"]="PRIVATE_IDENTIFIER_MUST_NOT_ESCAPE"
    raw["gene_to_image"]["changed_gene_context"]["private_spot_values"]=[10.,20.]
    raw["paired_score_change"]["rolled_genes"]["private_tag"]="PRIVATE_IDENTIFIER_MUST_NOT_ESCAPE"
    result=report.diagnostic_summary(raw,"M1")
    assert "PRIVATE_IDENTIFIER" not in json.dumps(result) and "private_spot_values" not in json.dumps(result)
    raw["gene_to_image"]["changed_gene_context"]=token(.1)
    with pytest.raises(report.old.ReportContractError,match="CLOSED_CROSS_CONTEXT"):
        report.diagnostic_summary(raw,"M1")


def test_source_qualification_hashes_bytes_without_array_decoding(tmp_path,monkeypatch):
    root=tmp_path/"cache";root.mkdir()
    filenames=dict(manifest="manifest.private.json",arrays="source_arrays.private.npz",counts="gene_counts.private.npz")
    for key,name in filenames.items():(root/name).write_bytes(("SYNTHETIC_"+key).encode())
    receipt=tmp_path/"fresh.json"
    fresh=dict(status="PASS_FRESH_RAW_FULL_LIBRARY_AND_EXISTING_CACHE_REPLAY",
        source_cache_manifest_sha256=report.sha(root/filenames["manifest"]),DSS_or_PFI_read=False,
        original_ST_test_read=False,source_files_mutated=False)
    dump(receipt,fresh)
    binding=dict(cache_manifest_sha256=report.sha(root/filenames["manifest"]),
        source_arrays_sha256=report.sha(root/filenames["arrays"]),gene_counts_sha256=report.sha(root/filenames["counts"]),
        fresh_requalification_sha256=report.sha(receipt))
    pointer=tmp_path/"assets.json";dump(pointer,dict(cache_root=str(root),fresh_source_receipt=str(receipt)))
    monkeypatch.setattr(np,"load",lambda *a,**k:(_ for _ in ()).throw(AssertionError("MUST_NOT_DESERIALIZE")))
    result=report.source_assets(pointer,binding)
    assert result["source_arrays_deserialized"] is False
    (root/filenames["arrays"]).write_bytes(b"changed")
    with pytest.raises(report.old.ReportContractError,match="SOURCE_BYTES_CHANGED"):
        report.source_assets(pointer,binding)


def test_nonfinite_public_result_fails_before_any_output_file_creation(matrix,tmp_path):
    config,lock,cells,ledger=loaded(matrix)
    result=report.build_report(config,lock,cells,ledger);result["bad_extra_scalar"]=float("nan")
    path=tmp_path/"nonfinite_public"
    with pytest.raises(ValueError):report.write_report(path,result)
    assert not path.exists()


def test_lossless_training_component_rejects_tamper_without_weakening_source_matrix(matrix,tmp_path):
    config,lock,cells,ledger=loaded(matrix)
    result=report.build_report(config,lock,cells,ledger)
    out=tmp_path/"sharded";report.write_report(out,result)
    assert report.encoded_json(report.restore_report(out/"S5A_RESULTS.json"))==report.encoded_json(result)
    side=out/"S5A_TRAINING_DIAGNOSTICS.json";raw=side.read_bytes();side.write_bytes(raw+b" ")
    with pytest.raises(report.old.ReportContractError,match="COMPONENT_BYTES_CHANGED"):
        report.restore_report(out/"S5A_RESULTS.json")
    side.write_bytes(raw)
    main=out/"S5A_RESULTS.json";value=report.read_json(main)
    value["training_diagnostics"]["file"]="../outside.json";dump(main,value)
    with pytest.raises(report.old.ReportContractError,match="NAME_OR_SUPPORT_CHANGED"):
        report.restore_report(main)


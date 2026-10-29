"""Anonymous metadata-only R8 delivery tests; no labels, tensors or model calls."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import bgb_en_r8_report as report


LEVELS={"R8-0":.50,"R8-1":.60,"R8-2":.61,"R8-3":.62,"R8-4":.64,
        "R8-5":.63,"R8-6":.67,"R8-8a_RANDOM_SOURCE":.66,"R8-8b_WRONG_PROGRAM":.65,
        "R8-9_WRONG_PATIENT":.64,"R8-10_NO_INTERACTION":.63}


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def concordance(arm,partition,reference="R8-1",*,undefined_uno=False):
    # Distinct denominator weights expose an incorrect unweighted fold mean.
    denominator=(10.,30.,60.)[partition]
    h=LEVELS[arm]+.03*partition;u=LEVELS[arm]-.04+.01*partition
    dh=LEVELS[arm]-LEVELS[reference];du=dh
    return dict(Harrell=h,Harrell_num=h*denominator,Harrell_den=denominator,
        Uno60=None if undefined_uno else u,
        Uno_num=0. if undefined_uno else u*denominator*2,
        Uno_den=0. if undefined_uno else denominator*2,
        repair=.01+max(dh,0),damage=.01+max(-dh,0),
        Uno_repair=None if undefined_uno else .01+max(du,0),
        Uno_damage=None if undefined_uno else .01+max(-du,0))


def family(arm,partition):
    return f"p{partition}_"+("RANDOM_NO_ST" if arm=="R8-8a_RANDOM_SOURCE" else "MATCHED")


def fixture_tree(tmp_path,monkeypatch,*,undefined_uno=False):
    root=tmp_path/"anonymous_checkout";root.mkdir()
    monkeypatch.setattr(report,"ROOT",root)
    operator=root/"src"/"frozen_operator.py";operator.parent.mkdir()
    operator.write_text("# synthetic scientific source binding\n",encoding="utf-8")
    source_config=root/"configs"/"source.json";put(source_config,dict(synthetic=True))
    config_path=root/"configs"/"r8.json"
    config=dict(source_version="R6_S7_locked",gamma=.1,ridge_lambda=1.,source_prior_scale=.1,
        cosine_temperature=1.,partitions=[0,1,2],seeds=[31,47,83],primary_candidate="R8-6",
        trainable_arms=list(report.ARMS),wrong_program_permutation=[1,2,3,0],
        global_pca_components=8,program_pca_components=2)
    put(config_path,config)
    private=tmp_path/"anonymous_private";private.mkdir()
    upstream=private/"upstream_lock.json";put(upstream,dict(status="SYNTHETIC_FROZEN_UPSTREAM"))
    source_states=dict(MATCHED="c"*64,RANDOM_NO_ST="d"*64)
    sources={"src/frozen_operator.py":report.sha256(operator),"configs/source.json":report.sha256(source_config)}
    ready_hashes={};qualified=[]
    for p in report.PARTITIONS:
        for kind in ("MATCHED","RANDOM_NO_ST"):
            name=f"p{p}_{kind}";folder=private/"features"/name;folder.mkdir(parents=True)
            # Deliberately not a pickle; report may hash bytes but never load them.
            artifact=folder/"features.private.pt";artifact.write_bytes(("FAKE_FEATURE_BYTES_"+name).encode())
            ready=dict(status="COMPLETE_R8_SCOPED_FROZEN_FEATURES",partition=p,source_kind=kind,
                source_version="R6_S7_RANDOM_NO_ST_locked" if kind=="RANDOM_NO_ST" else "R6_S7_locked",
                source_state_sha256=source_states[kind],config_sha256=report.sha256(config_path),
                source_config_sha256=report.sha256(source_config),
                source_patch_cache_receipt_sha256="e"*64,
                prepared_origin=dict(prepared_receipt_sha256="f"*64,prepared_artifact_sha256="a"*64),
                source_origin=dict(complete_source_state_sha256=source_states[kind]),
                source_program_order=["P0","P1","P2","P3"],source_program_definition_sha256="b"*64,
                scoped_rows=409,outer_test_rows_consumed=0,old_container_rows=512,old_container_physically_loaded=True,
                source_visual_forward_uses_RNA=False,source_spot_scaler_applied_to_bulk=False,
                source_fit_count=0,teacher_fit_count=0,target_supervised_fit_count=0,target_neural_updates=0,
                unchanged_raw_RNA512_teacher_and_CP_offset=True,
                locked_G_bulk_representation="frozen_source_gene64_then_train_program_PCA2",
                feature_generation_source_sha256=sources,artifact_sha256=report.sha256(artifact),
                PCA=dict(unsupervised_PCA_fits=10),query_mapping=dict(train_validation_crossings=0,self_pairs=0))
            put(folder/"READY.json",ready);ready_hashes[name]=report.sha256(folder/"READY.json")
            qualified.append(dict(name=name,receipt_sha256=ready_hashes[name],**ready))
    qualification=dict(status="PASS_R8_FROZEN_SOURCE_SCOPED_409_FEATURE_QUALIFICATION",
        config_sha256=report.sha256(config_path),source_config_sha256=report.sha256(source_config),
        R6_target_lock_sha256=report.sha256(upstream),source_version="R6_S7_locked",
        feature_families=6,source_neural_fits=0,target_neural_fits=0,target_convex_fits=0,
        unsupervised_PCA_fits=60,source_family_selected_by_DSS=False,rows=qualified,
        feature_generation_source_sha256=sources,endpoint_or_validation_scores_used_for_feature_selection=False,
        outer_test_evaluation=False)
    put(private/"R8_FEATURE_QUALIFICATION.json",qualification)
    lock=dict(schema="bgb_en_r8_frozen_evidence_protocol_lock_v1",status="LOCKED_BEFORE_R8_TARGET_FITS",
        config_sha256=report.sha256(config_path),qualification_sha256=report.sha256(private/"R8_FEATURE_QUALIFICATION.json"),
        sources=sources,feature_receipt_sha256=ready_hashes,source_full_state_sha256=source_states,
        source_config_sha256=report.sha256(source_config),source_R6_target_lock_sha256=report.sha256(upstream),
        planned_convex_fits=81,planned_neural_fits=0,planned_neural_updates=0,source_or_teacher_refits=0,
        primary_candidate="R8-6",gamma=.1,ridge_lambda=1.)
    lock["contract_sha256"]=report.canonical_hash(lock)
    put(private/"R8_PROTOCOL_LOCK.json",lock)
    units=[]
    for arm in report.ARMS:
        for p in report.PARTITIONS:
            for seed in report.SEEDS:
                folder=private/"cells"/f"{arm}_p{p}_seed{seed}";folder.mkdir(parents=True)
                artifact=folder/"model.private.pt";artifact.write_bytes(f"FAKE_MODEL_BYTES_{arm}_{p}_{seed}".encode())
                feature=family(arm,p);kind="RANDOM_NO_ST" if arm=="R8-8a_RANDOM_SOURCE" else "MATCHED"
                fit=dict(status="CONVERGED_KKT_AND_TRAIN_SD_PROTECTION",registered_coefficients=report.PARAMETERS[arm],
                    learned_intercepts=0,gamma=.1,ridge_lambda=1.,training_riskset_complete=True,
                    validation_or_test_labels_used=False,hyperparameters_selected_from_metrics=False,
                    iterations=4,function_evaluations=6,train_design_rank=report.PARAMETERS[arm],
                    kkt_stationarity_max=1e-8,kkt_complementarity=1e-10,constraint_slack=.1,
                    residual_SD_ratio=.05)
                contract=dict(contract_sha256=lock["contract_sha256"],config_sha256=report.sha256(config_path),
                    source_state_sha256=source_states[kind],feature_receipt_sha256=ready_hashes[feature],
                    arm=arm,partition=p,seed=seed,registered_coefficients=report.PARAMETERS[arm],gamma=.1,ridge_lambda=1.)
                diagnostics=dict(train=dict(SD=.05,SD_ratio_to_fixed_offset=.05),validation=dict(SD=.2,SD_ratio_to_fixed_offset=.2))
                controls=dict(residual_off=dict(frozen_not_retrained=True,
                    recipient_gene_values_masks_RNA512_offset_and_images_unchanged=True,
                    metrics_vs_anchor=concordance("R8-1",p,undefined_uno=undefined_uno),
                    full_relative_to_control=concordance(arm,p,undefined_uno=undefined_uno),
                    valid_risk_difference=dict(rows=100,mean=0.,SD=.05,SD_ratio_to_fixed_offset=.05)))
                cell=dict(status="COMPLETE_R8_CONVEX_FIT",contract=contract,arm=arm,partition=p,seed=seed,
                    actual_convex_fits=1,actual_neural_fits=0,neural_updates=0,source_refits=0,teacher_refits=0,
                    registered_coefficients=report.PARAMETERS[arm],actual_fit=fit,
                    deterministic_solver_no_seed_randomization=True,nominal_seeds_are_not_independent_random_cohorts=True,
                    metrics=concordance(arm,p,undefined_uno=undefined_uno),
                    train_metrics=concordance(arm,p,undefined_uno=undefined_uno),
                    PFI_frozen_DSS_risk=concordance(arm,p,undefined_uno=undefined_uno),
                    risk_diagnostics=diagnostics,frozen_controls=controls,
                    source_version="R6_S7_RANDOM_NO_ST_locked" if kind=="RANDOM_NO_ST" else "R6_S7_locked",
                    source_program_definition_sha256="b"*64,source_origin=dict(synthetic=True),
                    feature_receipt_sha256=ready_hashes[feature],artifact_sha256=report.sha256(artifact),elapsed_seconds=.02)
                put(folder/"COMPLETE.json",cell)
                cert=dict(status="PASS_ACTUAL_SAVED_COEFFICIENT_KKT_AND_SD_REPLAY",actual_saved_coefficient_forward_replayed=True,
                    new_optimizer_fits=0,gamma=.1,ridge_lambda=1.,stationarity_max=1e-8,complementarity=1e-10,
                    constraint_slack=.1,train_residual_SD_ratio=.05)
                units.append(dict(status="PASS_ACTUAL_R8_SAVED_COEFFICIENT_AND_FEATURE_REPLAY",arm=arm,partition=p,seed=seed,
                    actual_saved_risk_forward_replayed=True,maximum_risk_error=0.,maximum_metric_or_control_error=0.,
                    changed_pair_credits=0,new_optimizer_fits=0,new_neural_updates=0,convex_KKT=cert,
                    completion_sha256=report.sha256(folder/"COMPLETE.json"),artifact_sha256=cell["artifact_sha256"],
                    feature_receipt_sha256=ready_hashes[feature]))
    refs=[]
    for arm in report.REFS:
        for p in report.PARTITIONS:
            for seed in report.SEEDS:
                value=dict(status="COMPLETE_R8_ZERO_FIT_REFERENCE",arm=arm,partition=p,seed=seed,
                    contract_sha256=lock["contract_sha256"],actual_convex_fits=0,actual_neural_fits=0,actual_new_teacher_fits=0,
                    metrics=concordance(arm,p,undefined_uno=undefined_uno),
                    PFI_frozen_DSS_risk=concordance(arm,p,undefined_uno=undefined_uno))
                path=private/"references"/f"{arm}_p{p}_seed{seed}.json";put(path,value)
                refs.append(dict(arm=arm,partition=p,seed=seed,actual_baseline_values_replayed=True,
                    reference_sha256=report.sha256(path),maximum_metric_error=0.,new_optimizer_fits=0))
    direct=[dict(candidate=a,reference=b,partition=p,seed=s,
        candidate_vs_reference=concordance(a,p,b,undefined_uno=undefined_uno),
        PFI_frozen_DSS_risk=concordance(a,p,b,undefined_uno=undefined_uno),
        direct_actual_saved_risk_pair_credits=True,bootstrap_CI_or_pvalue_computed=False)
        for a,b in report.COMPARISONS for p in report.PARTITIONS for s in report.SEEDS]
    checks=[dict(arm=a,partition=p,bitwise_identical_predictions=True,maximum_absolute_difference=0.,
        identical_deterministic_repetitions_are_not_independent_random_evidence=True)
        for a in report.ARMS for p in report.PARTITIONS]
    replay=dict(status="PASS_COMPLETE_R8_81_SAVED_RISK_REPLAYS",contract_sha256=lock["contract_sha256"],
        actual_saved_models=81,passed_units=81,failed_units=0,new_optimizer_fits=0,source_or_teacher_refits=0,
        units=units,zero_fit_reference_replays=refs,paired_retrained_controls=direct,deterministic_seed_checks=checks)
    put(private/"REPLAY_AUDIT.json",replay)
    return dict(root=root,private=private,config=config_path,replay=replay,lock=lock,operator=operator,
                output=tmp_path/"public")


@pytest.fixture
def tree(tmp_path,monkeypatch):
    return fixture_tree(tmp_path,monkeypatch)


def build(tree):
    return report.build(tree["private"],tree["config"],tree["output"])


def save_replay(tree):
    put(tree["private"]/"REPLAY_AUDIT.json",tree["replay"])


def test_full_anonymous_delivery_uses_weighted_components_and_fixed_candidate(tree):
    result=build(tree)
    assert result["budget"]["actual_convex_fits"]==result["budget"]["actual_saved_model_replays"]==81
    assert result["budget"]["deterministic_unique_training_problems"]==27
    assert len(result["diagnostic_index"])==9 and len(list(tree["output"].iterdir()))==17
    assert abs(result["primary_DSS"]["R8-6"]["Harrell"]-(.67+.03*(30+2*60)/100))<1e-12
    decision=result["decision"]
    assert decision["primary_candidate"]=="R8-6"
    assert decision["independent_random_seed_evidence"] is False
    assert decision["nominal_seed_repetitions"]==3
    assert decision["all_nominal_seed_predictions_bitwise_identical"] is True
    assert decision["PFI_used_for_selection"] is False
    assert decision["R8_9_bilinear_control_confirms_R8_5_patch_routing"] is False
    with (tree["output"]/"R8_FOLD_METRICS.csv").open(encoding="utf-8-sig",newline="") as stream:
        assert len(list(csv.DictReader(stream)))==99
    public="\n".join(p.read_text(encoding="utf-8-sig") for p in tree["output"].iterdir())
    assert str(tree["private"]) not in public and "FAKE_MODEL_BYTES" not in public
    with pytest.raises(ValueError,match="NO_OVERWRITE"):
        build(tree)


def test_eighty_completed_fits_never_emit_a_complete_report(tree):
    next((tree["private"]/"cells").glob("*/COMPLETE.json")).unlink()
    with pytest.raises(ValueError,match="COMPLETE_FIT_MATRIX"):
        build(tree)
    assert not (tree["output"]/"R8_RESULTS.json").exists()


@pytest.mark.parametrize("change",["missing_unit","metric_error","changed_credit"])
def test_actual_replay_cannot_be_replaced_by_counts_or_near_risks(tree,change):
    if change=="missing_unit":tree["replay"]["units"].pop()
    if change=="metric_error":tree["replay"]["units"][0]["maximum_metric_or_control_error"]=1e-5
    if change=="changed_credit":tree["replay"]["units"][0]["changed_pair_credits"]=1
    save_replay(tree)
    with pytest.raises(ValueError):
        build(tree)
    assert not (tree["output"]/"R8_DECISION.json").exists()


@pytest.mark.parametrize("field,value",[("stationarity_max",5e-6+1e-9),("complementarity",5e-6+1e-9),
    ("constraint_slack",-1e-8-1e-10),("train_residual_SD_ratio",.1+1e-8+1e-10),
    ("complementarity",-1.),("train_residual_SD_ratio",-.1)])
def test_kkt_and_training_sd_certificates_require_mathematical_bounds(tree,field,value):
    tree["replay"]["units"][0]["convex_KKT"][field]=value;save_replay(tree)
    with pytest.raises(ValueError,match="KKT_OR_TRAIN_SD_BOUND"):
        build(tree)


@pytest.mark.parametrize("changed",["source_code","prepared_receipt","feature_bytes","model_bytes"])
def test_frozen_scientific_and_data_bindings_are_real_byte_bindings(tree,changed):
    if changed=="source_code":tree["operator"].write_text("# modified scientific code\n")
    if changed=="prepared_receipt":
        ready=tree["private"]/"features"/"p0_MATCHED"/"READY.json"
        value=read(ready);value["prepared_origin"]["prepared_receipt_sha256"]="0"*64;put(ready,value)
    if changed=="feature_bytes":
        (tree["private"]/"features"/"p0_MATCHED"/"features.private.pt").write_bytes(b"CHANGED_FAKE_FEATURE")
    if changed=="model_bytes":
        (tree["private"]/"cells"/"R8-2_p0_seed31"/"model.private.pt").write_bytes(b"CHANGED_FAKE_MODEL")
    with pytest.raises(ValueError):
        build(tree)
    assert not (tree["output"]/"R8_RESULTS.json").exists()


def test_repair_damage_must_equal_the_actual_paired_increment(tree):
    for row in tree["replay"]["paired_retrained_controls"]:
        if row["candidate"]=="R8-6" and row["reference"]=="R8-4":
            row["candidate_vs_reference"]["repair"]+=.01
    save_replay(tree)
    with pytest.raises(ValueError,match="PAIR_CREDIT_AND_METRIC_DISAGREE|PARTITION_REPAIR_DAMAGE_CHANGED"):
        build(tree)
    assert not tree["output"].exists() or not any(tree["output"].iterdir())


def test_primary_candidate_cannot_follow_a_favorable_result(tree):
    value=read(tree["config"]);value["primary_candidate"]="R8-2";put(tree["config"],value)
    with pytest.raises(ValueError,match="FIXED_CONFIG_CHANGED"):
        build(tree)


def test_undefined_uno_stays_null_and_prevents_a_positive_joint_decision(tmp_path,monkeypatch):
    tree=fixture_tree(tmp_path,monkeypatch,undefined_uno=True)
    result=build(tree)
    assert result["primary_DSS"]["R8-6"]["Uno60"] is None
    assert result["decision"]["status"]=="INCONCLUSIVE_UNDEFINED_METRIC"
    assert all(r["delta_Uno60"] is None and r["Uno_repair"] is None for r in result["paired_comparisons"])


def test_opposite_partition_credit_errors_cannot_hide_in_a_correct_aggregate(tree):
    selected=[row for row in tree["replay"]["paired_retrained_controls"]
              if row["candidate"]=="R8-6" and row["reference"]=="R8-4" and row["seed"]==31]
    assert len(selected)==3
    for field in ("candidate_vs_reference","PFI_frozen_DSS_risk"):
        originals={row["partition"]:row[field]["repair"] for row in selected}
        for row in selected:
            if row["partition"]==0:row[field]["repair"]+=.03
            if row["partition"]==1:row[field]["repair"]-=.01
        weighted=sum((row[field]["repair"]-row[field]["damage"])*row[field]["Harrell_den"] for row in selected)
        denominator=sum(row[field]["Harrell_den"] for row in selected)
        # The old seed-level identity still holds exactly; each changed partition is wrong.
        assert abs(weighted/denominator-(LEVELS["R8-6"]-LEVELS["R8-4"]))<1e-12
        save_replay(tree)
        with pytest.raises(ValueError,match="DIRECT_PARTITION_REPAIR_DAMAGE_CHANGED"):
            build(tree)
        assert not tree["output"].exists() or not any(tree["output"].iterdir())
        for row in selected:row[field]["repair"]=originals[row["partition"]]


def test_direct_candidate_components_must_match_the_same_saved_cell(tree):
    row=tree["replay"]["paired_retrained_controls"][0]
    for field in ("candidate_vs_reference","PFI_frozen_DSS_risk"):
        metrics=row[field];original=(metrics["Harrell"],metrics["Harrell_num"])
        metrics["Harrell"]+=.02
        metrics["Harrell_num"]=metrics["Harrell"]*metrics["Harrell_den"]
        # Keep the row's own ratio valid so only cross-record binding can reject it.
        save_replay(tree)
        with pytest.raises(ValueError,match="DIRECT_CANDIDATE_PARTITION_METRIC_CHANGED"):
            build(tree)
        assert not tree["output"].exists() or not any(tree["output"].iterdir())
        metrics["Harrell"],metrics["Harrell_num"]=original

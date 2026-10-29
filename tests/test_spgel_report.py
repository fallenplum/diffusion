"""Synthetic-only qualification tests for source report completeness/privacy."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from scripts import spgel_source_report as report


def config():
    return dict(schema="spgel_source_pilot_recipe_v1",
        trainable_arms=["S_A"+str(k) for k in range(6)], heldout_donor_codes=list(range(6)),
        seeds=[31, 47], primary_candidate="S_A5", program_order=["P0", "P1", "P2", "P3"],
        training=dict(save_epochs=[10, 20, 40], epochs=40, primary_epoch=40),
        budget=dict(pilot_logical_cells=72, pilot_convex_optimizer_fits=12,
            pilot_neural_optimizer_fits=60, neural_optimizer_updates=52000,
            neural_updates_per_fit_by_heldout_donor=[880,720,960,880,960,800], neural_saved_checkpoint_states=180,
            convex_solver_final_states=12, pilot_actual_replay_states=192,
            unplanned_or_automatic_retry_fits=0),
        evaluation=dict(official_HVG_task="NOT_RUN_SEPARATE_OFFICIAL_HVG_LOG1P_ONLY_CONTRACT"),
        scope=dict(DSS_or_PFI_read=False, TCGA_loaded=False, R8_results_influence_source=False,
            source_test_for_model_selection=False, independent_source_confirmation_claimed=False,
            histological_or_cell_source_truth_claimed=False))


def test_null_correlations_keep_the_support_and_do_not_turn_into_zero():
    result=report.mean_defined([.4, None, .6, None])
    assert result==dict(mean=.5, defined_count=2, expected_count=4, undefined_count=2,
                        undefined_is_not_zero=True)
    assert report.mean_defined([None, None])["mean"] is None
    for bad in (float("nan"), float("inf"), True, "0.2"):
        with pytest.raises(report.ReportContractError, match="INVALID_SCALAR"):
            report.mean_defined([bad])


def test_json_only_reader_refuses_model_files_and_ambiguous_metadata(tmp_path):
    tensor=tmp_path/"checkpoint.pt"
    tensor.write_bytes(b"this is not opened by the reporter")
    with pytest.raises(report.ReportContractError, match="JSON_METADATA_ONLY"):
        report.read_metadata(tensor)
    path=tmp_path/"receipt.json"
    path.write_text('{"status":"COMPLETE","status":"FAILED"}')
    with pytest.raises(report.ReportContractError, match="DUPLICATE_JSON_KEY"):
        report.read_metadata(path)
    path.write_text('{"metric": NaN}')
    with pytest.raises(report.ReportContractError, match="NONFINITE_JSON_CONSTANT"):
        report.read_metadata(path)
    path.write_text(json.dumps({"safe": None}))
    assert report.read_metadata(path)=={"safe":None}


def test_scope_budget_and_fixed_candidate_cannot_be_selected_from_results():
    original=config()
    assert report.validate_config(original) is original
    for key, value, error in (("primary_candidate", "S_A1", "SOURCE_MATRIX"),
                              ("seeds", [31], "SOURCE_MATRIX"),
                              ("program_order", ["P0"]*4, "FOUR_FIXED")):
        altered=copy.deepcopy(original);altered[key]=value
        with pytest.raises(report.ReportContractError, match=error):
            report.validate_config(altered)
    altered=copy.deepcopy(original);altered["budget"]["pilot_logical_cells"]=71
    with pytest.raises(report.ReportContractError, match="SOURCE_BUDGET"):
        report.validate_config(altered)
    altered=copy.deepcopy(original);altered["scope"]["DSS_or_PFI_read"]=True
    with pytest.raises(report.ReportContractError, match="SOURCE_SCOPE"):
        report.validate_config(altered)


def regression(rows, pearson):
    per=[dict(column=k,observed_count=rows,Pearson=pearson,Spearman=pearson,
              MAE=.5,MSE=.4,Huber_delta1=.2) for k in range(4)]
    return dict(per_output=per,Mean_Pearson=pearson,defined_Pearson_count=4,
                Mean_MAE=.5,Mean_MSE=.4,Mean_Huber_delta1=.2)


def task_a(rows, pearson):
    return dict(task="IMAGE_ONLY_ST_PREDICTION",native=regression(rows,pearson),
        train_Z=regression(rows,pearson),primary_donor_four_program_Pearson=pearson,
        program_count=4,defined_program_count=4,scale="raw_signed_log1pCP10K_program_and_current_train_Z",
        primary_requires_all_four_defined=True,true_RNA_or_tissue_not_prediction_input=True)


def task_c(rows, pearson):
    result=regression(rows,pearson)
    for row in result["per_output"]:
        row.update(high_activity_count=rows//2,AUPRC=.6,prevalence=(rows//2)/rows,
                   high_activity_threshold_from_train=1.)
    result.update(task="PROGRAM_TO_REGION_LOCALIZATION",threshold_fit_scope="source_training_rows_only",
        lesion_segmentation_truth_claimed=False,primary_four_program_AUPRC=.6,
        defined_AUPRC_count=4,primary_requires_all_four_defined=True)
    return result


def task_b(rows=2000,donor=0):
    direction=dict(Recall_at_1=.2,Recall_at_5=.5,median_rank=3.,pair_AUC=.7,queries_count=256)
    sections=[]
    for number,count in enumerate((rows//2,rows-rows//2)):
        stats=dict(direction,queries_count=min(count,128),candidates_count=count,candidate_negative_count=count-1)
        sections.append(dict(section_code=donor*2+number,anchors_count=min(count,128),candidates_count=count,
            image_to_gene=stats.copy(),gene_to_image=stats.copy()))
    return dict(status="COMPLETE",candidate_scope="entire_same_heldout_slide_no_tissue_depth_filter",
        query_anchor_cap_per_slide=128,image_to_gene=direction.copy(),gene_to_image=direction.copy(),per_section=sections)


def heldout(rows, donor, arm, *, final):
    pearson=(donor+1)/10
    left=rows//2;right=rows-left
    sections=[dict(section_code=donor*2,spots_count=left,task_A=task_a(left,pearson)),
              dict(section_code=donor*2+1,spots_count=right,task_A=task_a(right,pearson))]
    within=dict(task="within_slide_tissue_stratified_prediction",
        tissue_truth_used_for_evaluation_groups_only=True,labels_used_as_model_inputs=False,
        groups=[dict(section_code=s["section_code"],tissue_code=0,spots_count=s["spots_count"],
                     metrics=regression(s["spots_count"],pearson)) for s in sections])
    diagnostic=dict(status="NOT_EVALUATED_FOR_TASKB")
    native=task_b(rows,donor) if arm in ("S_A3","S_A4","S_A5") else dict(status="NOT_APPLICABLE_NO_TRAINED_GENE_ENCODER")
    return dict(task_A=task_a(rows,pearson),task_C=task_c(rows,pearson),
        section_metrics=dict(task="within_slide_program_prediction",sections=sections),
        tissue_metrics=dict(image_predicted_tissue=dict(status="NOT_APPLICABLE_NO_TISSUE_HEAD"),within_slide_tissue=within),
        task_B_common=task_b(rows,donor) if final else diagnostic.copy(),task_B_native=native if final else diagnostic.copy())


def synthetic_cells(cfg, cfg_sha, contract, *, stage="pilot"):
    sizes=[1984,4119,1091,1827,932,3058]
    result={}
    for key in report.expected_units(cfg,stage=stage):
        arm,donor,seed=key;convex=arm=="S_A0"
        test=sizes[donor] if stage=="pilot" else 0;train=13011-test
        snapshots=[]
        for label,epoch in report._state_labels(arm).items():
            primary=label in ("solver_final","epoch40")
            snapshots.append(dict(label=label,epoch=epoch,
                checkpoint_sha256=report.canonical_sha256([arm,donor,seed,label]),
                train=dict(task_A=task_a(train,.7)),
                heldout=heldout(test,donor,arm,final=primary) if stage=="pilot" else dict(status="NOT_APPLICABLE"),
                losses=dict(solver_iterations=12,solver_gradient_infinity_norm=1e-8) if convex else
                    {"epoch_minibatch_preupdate_mean_"+name:.5 for name in ("total","prediction_Huber","tissue_CE","ITC")}))
        result[key]=dict(schema="spgel_source_cell_v1",status="COMPLETE_SPGEL_SOURCE_FIT",
            arm=arm,heldout_donor=donor,seed=seed,stage=stage,fit_kind="convex" if convex else "neural",
            epochs=None if convex else 40,optimizer_updates=0 if convex else 40*math.ceil(train/512),
            convex_optimizer_fits=int(convex),neural_optimizer_fits=int(not convex),train_rows=train,heldout_rows=test,
            parameters=dict(optimized=20,registered=32,analytic_program_table_coefficients=12 if convex else 0),
            elapsed_seconds=2.,peak_GPU_bytes=1024,config_sha256=cfg_sha,contract_sha256=contract,
            artifact_sha256=report.canonical_sha256([arm,donor,seed]),snapshots=snapshots)
    return result


def replay(cells, hashes, cfg_sha, contract, *, stage="pilot"):
    units=[]
    for key,cell in cells.items():
        arm,donor,seed=key
        units.append(dict(arm=arm,heldout_donor=donor,seed=seed,status="PASS",new_optimizer_fits=0,
            new_optimizer_updates=0,actual_saved_state_loaded=True,completion_sha256=hashes[key],
            artifact_sha256=cell["artifact_sha256"],snapshots=[dict(label=s["label"],
                checkpoint_sha256=s["checkpoint_sha256"],actual_state_forward_replayed=True,
                maximum_prediction_error=1e-8,maximum_metric_error=1e-12) for s in cell["snapshots"]]))
    states=192 if stage=="pilot" else 6
    return dict(schema="spgel_source_replay_v1",
        status="PASS_COMPLETE_SPGEL_PILOT_72_FITS_192_REPLAYS" if stage=="pilot" else "PASS_COMPLETE_SPGEL_S4_2_FITS_6_REPLAYS",
        expected_units=len(cells),passed_units=len(cells),failed_units=0,expected_states=states,passed_states=states,
        units=units,config_sha256=cfg_sha,contract_sha256=contract)


@pytest.fixture
def matrix(tmp_path):
    cfg=config();cfg_path=tmp_path/"config.json";cfg_path.write_text(json.dumps(cfg))
    cfg_sha=report.file_sha256(cfg_path)
    lock=dict(schema="spgel_source_run_lock_v1",status="LOCKED_BEFORE_SPGEL_FITS",
        config_sha256=cfg_sha,sources={"src/example.py":"b"*64},
        pilot=dict(logical_cells=72,neural_fits=60,convex_fits=12,actual_saved_states=192,optimizer_updates=52000),
        s4=dict(neural_fits=2,actual_saved_states=6,optimizer_updates=2080))
    contract=report.canonical_sha256(lock);lock["contract_sha256"]=contract
    cells=synthetic_cells(cfg,cfg_sha,contract)
    hashes={}
    for key,cell in cells.items():
        folder=tmp_path/"pilot"/"cells"/report.cell_name(key);folder.mkdir(parents=True)
        path=folder/"COMPLETE.json";path.write_text(json.dumps(cell));hashes[key]=report.file_sha256(path)
    verified=replay(cells,hashes,cfg_sha,contract)
    (tmp_path/"pilot"/"REPLAY_AUDIT.json").write_text(json.dumps(verified))
    return tmp_path,cfg,cfg_sha,contract,cells,hashes,verified,lock


def test_complete_source_matrix_replay_and_equal_donor_macro(matrix):
    private,cfg,sha,contract,_,_,_,lock=matrix
    cells,audit=report.load_validated_matrix(private,cfg,sha,lock)
    result=report.build_report(cfg,cells,audit,config_sha256=sha,contract_sha256=contract)
    assert result["counters"]["logical_cells"]==72
    assert result["counters"]["actual_saved_states"]==result["replay"]["passed_states"]==192
    assert result["counters"]["convex_optimizer_fits"]==12
    assert result["counters"]["neural_optimizer_fits"]==60
    assert result["counters"]["optimizer_updates"]==52000
    assert len(result["donor_rows"])==72
    assert len(result["fixed_checkpoint_diagnostics"])==192
    assert all(abs(r["task_A_primary_Pearson"]["mean"]-.35)<1e-12 for r in result["arm_seed_summary"])
    assert result["official_50HVG"].startswith("NOT_RUN")
    assert result["full_requested_research_program_complete"] is False
    output=private/"public"
    decision=report.write_report(output,result)
    assert decision["pilot_optimizer_fits"]==72
    assert len(list(output.glob("*")))==5
    with pytest.raises(report.ReportContractError,match="REFUSE_OVERWRITE"):
        report.write_report(output,result)
    public="\n".join(p.read_text(encoding="utf-8") for p in output.iterdir())
    assert str(private) not in public and "checkpoint.pt" not in public


def test_missing_or_extra_completed_cell_cannot_produce_partial_means(matrix):
    private,cfg,sha,_,_,_,_,lock=matrix
    path=next((private/"pilot"/"cells").glob("*/COMPLETE.json"))
    path.unlink()
    with pytest.raises(report.ReportContractError,match="NO_PARTIAL_MEANS"):
        report.load_validated_matrix(private,cfg,sha,lock)


@pytest.mark.parametrize("mutation",["missing_state","forged_forward","refit","changed_hash","large_error"])
def test_saved_state_replay_requires_real_coverage_and_no_new_fitting(matrix,mutation):
    _,_,sha,contract,cells,hashes,verified,_=matrix
    altered=copy.deepcopy(verified);unit=altered["units"][0]
    if mutation=="missing_state":unit["snapshots"].pop()
    if mutation=="forged_forward":unit["snapshots"][0]["actual_state_forward_replayed"]=False
    if mutation=="refit":unit["new_optimizer_fits"]=1
    if mutation=="changed_hash":unit["completion_sha256"]="f"*64
    if mutation=="large_error":unit["snapshots"][0]["maximum_prediction_error"]=1e-3
    with pytest.raises(report.ReportContractError):
        report.validate_replay(altered,cells,hashes,config_sha256=sha,contract_sha256=contract,stage="pilot")


def test_one_undefined_program_invalidates_primary_four_program_macro():
    task=task_a(20,.5)
    task["native"]["per_output"][0]["Pearson"]=None
    task["native"]["defined_Pearson_count"]=task["defined_program_count"]=3
    task["primary_donor_four_program_Pearson"]=None
    result=report.task_a_summary(task,rows=20)
    assert result["primary_donor_four_program_Pearson"] is None
    assert result["native"]["defined_programs"]==3
    task["primary_donor_four_program_Pearson"]=.5
    with pytest.raises(report.ReportContractError,match="PRIMARY_DONOR"):
        report.task_a_summary(task,rows=20)


def test_s4_training_evidence_is_never_heldout_performance():
    cfg=config();sha="a"*64;contract="b"*64
    cells=synthetic_cells(cfg,sha,contract,stage="s4")
    result=report.summarize_s4(cfg,cells,dict(passed_units=2,passed_states=6),
        config_sha256=sha,contract_sha256=contract)
    assert result["performance_is_training_only"] is True
    assert result["source_checkpoint_selected_from_pilot_or_survival_scores"] is False
    assert all(r["heldout_rows"]==0 and r["independent_performance_evidence"] is False for r in result["models"])


def test_current_source_metric_producer_matches_reporter_without_real_data():
    import numpy as np
    from src.spgel.metrics import image_prediction_metrics, localization_metrics
    rng=np.random.default_rng(781)
    native=rng.normal(size=(32,4));mask=np.ones_like(native,dtype=bool)
    mean=np.array([1.,2.,3.,4.]);sd=np.array([.5,1.,1.5,2.])
    pz=(native-mean)/sd+.1*rng.normal(size=native.shape)
    image=image_prediction_metrics(pz,native,mask,mean,sd)
    parsed=report.task_a_summary(image,rows=32)
    assert abs(parsed["primary_donor_four_program_Pearson"]-image["primary_donor_four_program_Pearson"])<1e-12
    localization=localization_metrics(pz*sd+mean,native,mask,np.zeros(4))
    parsed=report.task_c_summary(localization,rows=32)
    assert parsed["primary_four_program_AUPRC"]==localization["primary_four_program_AUPRC"]


def test_report_accepts_the_actual_locked_source_config_not_only_a_fixture():
    path=Path(__file__).resolve().parents[1]/"configs/spgel/pilot_v1.json"
    current=report.read_metadata(path)
    assert report.validate_config(current) is current
    assert current["budget"]["neural_optimizer_updates"]==52000
    assert current["budget"]["neural_updates_per_fit_by_heldout_donor"]==[880,720,960,880,960,800]


@pytest.mark.parametrize("mutation",["total_updates","donor_updates","extra_budget_key"])
def test_actual_budget_fields_cannot_be_changed_or_silently_ignored(mutation):
    path=Path(__file__).resolve().parents[1]/"configs/spgel/pilot_v1.json"
    current=report.read_metadata(path)
    if mutation=="total_updates":current["budget"]["neural_optimizer_updates"]+=1
    if mutation=="donor_updates":current["budget"]["neural_updates_per_fit_by_heldout_donor"][0]+=1
    if mutation=="extra_budget_key":current["budget"]["unregistered_extra_optimizer_fits"]=1
    with pytest.raises(report.ReportContractError,match="SOURCE_BUDGET_CHANGED"):
        report.validate_config(current)

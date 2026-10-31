"""Independent verifier synthetic qualification; never read cohort results."""
import copy
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import verify_semantic_b0b7_p2 as audit


def test_explicit_postcompletion_gate_precedes_every_input_read(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):
        raise AssertionError("input must stay unread")
    monkeypatch.setattr(audit,"sha",forbidden)
    with pytest.raises(ValueError,match="AUTHORIZATION_REQUIRED_BEFORE_ANY_INPUT_READ"):
        audit.verify_run("unread","unread","unread",tmp_path/"unused",cohort_path="unread",endpoint_path="unread",report_dir="unread")


def test_harrell_same_time_event_censor_but_not_event_event_and_risk_ties():
    t=np.array([1.,1.,1.,3.]);e=np.array([1,1,0,0]);r=np.array([3.,2.,1.,0.])
    value=audit.pair_statistic(t,e,r)
    assert value["pairs"]==4 and value["event_censor_tied_time_pairs"]==2
    assert value["score"]==1 and audit.pair_statistic(t,e,-r)["score"]==0
    assert audit.pair_statistic(t,e,np.zeros(4))["score"]==.5
    assert audit.pair_statistic([1,2],[1,0],[1e-9,0])["score"]==.5


def test_reverse_km_tied_death_leaves_before_censor_and_right_continuous():
    t=[1.,1.,2.,80.];e=[1,0,1,0]
    assert audit.censor_km_value(t,e,.99)==1.
    assert audit.censor_km_value(t,e,1.)==pytest.approx(2/3)
    assert audit.censor_km_value(t,e,60.)==pytest.approx(2/3)
    assert audit.censor_km_value(t,e,90.)==0.
    with pytest.raises(ValueError,match="EXCEEDS_TRAIN_SUPPORT"):
        audit.censor_km_value([1,2],[0,1],3.)


def test_uno_strict_tau_and_prelocked_zero_support_are_preserved():
    value=audit.independent_metrics([1,1,2,80],[1,0,1,0],[1,60,70],[1,1,0],[2,1,0])
    assert value["Uno60_valid"] is True
    assert value["Uno_den"]==pytest.approx(2*(1.5**2))  # event exactly 60 is excluded
    assert value["Uno60"]==1.
    no=audit.independent_metrics([1,20],[1,0],[1,10],[1,0],[1,0])
    assert no["Harrell"]==1 and no["Uno60"] is None and not no["Uno60_valid"]
    assert no["Uno60_NA_reason"]=="IPCW_ZERO_CENSOR_SURVIVAL_AT_TAU"
    tail=audit.independent_metrics([1,20],[0,1],[1,10],[1,0],[1,0])
    assert tail["Uno60_NA_reason"]=="CENSOR_QUERY_EXCEEDS_TRAIN_SUPPORT"


def test_independent_efron_gradient_finite_difference_ties_and_shift():
    t=np.array([1,1,1,3,5.]);e=np.array([1,1,0,1,0]);r=np.array([.2,-.4,.8,.5,-.2])
    value,g=audit.efron_gradient(r,t,e)
    numeric=[]
    for i in range(len(r)):
        step=np.zeros(len(r));step[i]=1e-6
        numeric.append((audit.efron_gradient(r+step,t,e)[0]-audit.efron_gradient(r-step,t,e)[0])/2e-6)
    assert np.max(np.abs(g-numeric))<1e-8 and abs(g.sum())<1e-14
    shifted,sg=audit.efron_gradient(r+1000,t,e)
    assert shifted==pytest.approx(value,abs=1e-12) and np.allclose(sg,g,atol=1e-13)
    x=np.arange(15).reshape(5,3)/10;w=np.array([.1,-.2,.3]);base=r.copy()
    objective,gradient=audit.cox_objective(x,w,base,t,e,1.)
    for i in range(3):
        step=np.zeros(3);step[i]=1e-6
        delta=(audit.cox_objective(x,w+step,base,t,e,1.)[0]-audit.cox_objective(x,w-step,base,t,e,1.)[0])/2e-6
        assert gradient[i]==pytest.approx(delta,abs=1e-8)


def test_clean_independent_math_matches_locked_scorer_only_as_test_oracle():
    from src.semantic_b0b7.survival import evaluate_fold,efron_numpy
    rng=np.random.default_rng(442)
    for case in range(12):
        trt=np.r_[rng.integers(1,80,35),120.];tre=np.r_[rng.integers(0,2,35),0]
        t=rng.integers(1,90,13).astype(float);e=rng.integers(0,2,13);e[0]=1
        r=np.round(rng.normal(size=13),1)
        expected=evaluate_fold(trt,tre,t,e,r)
        actual=audit.independent_metrics(trt,tre,t,e,r)
        for key in ("Harrell","Harrell_num","Harrell_den","Uno60","Uno_num","Uno_den"):
            assert actual[key]==pytest.approx(expected[key],rel=1e-12,abs=1e-10)
        assert audit.efron_gradient(r,t,e)[0]==pytest.approx(efron_numpy(r,t,e),abs=1e-12)


def test_passive_decode_no_model_import_or_constructor_and_rejects_arbitrary_globals(tmp_path):
    from src.semantic_b0b7.p2_survival import OffsetCox
    model=OffsetCox(np.array([1.,2.]),1.,{"synthetic":True})
    path=tmp_path/"owned.pkl";path.write_bytes(pickle.dumps(model,protocol=5))
    loaded=audit.passive_pickle(path)
    assert isinstance(loaded,audit.PassiveRecord) and np.array_equal(loaded.coefficient,model.coefficient)
    class Forbidden:
        def __reduce__(self):
            return eval,("1+1",)
    path.write_bytes(pickle.dumps(Forbidden()))
    with pytest.raises(ValueError,match="UNAPPROVED_PRIVATE_PICKLE_GLOBAL"):
        audit.passive_pickle(path)


def synthetic_head(root):
    from src.semantic_b0b7.p2_survival import fit_risk_cell,replay_risk_cell
    rng=np.random.default_rng(606);n=43;m=10
    features=lambda nn:{k:rng.normal(size=(nn,18 if k!="mechanism" else 2)) for k in ("pathology","omics","clinical","mechanism")}
    tr,ev=features(n),features(m)
    tt=np.linspace(1,130,n);te=(np.arange(n)%3==0).astype(float)
    et=np.linspace(1,95,m);ee=(np.arange(m)%3==0).astype(float)
    base=rng.normal(size=n);offset=rng.normal(size=m)
    phase={"selected_state_sha256":"a"*64,"input_sha256":"b"*64}
    expected={"lock_sha256":"c"*64,"split":"outer1_final","arm":"B5","seed":31,"phase_a_source_family":"MAIN","ridge_lambda":1.}
    binding={k:v for k,v in expected.items() if k!="ridge_lambda"}
    binding.update(phase_a_selected_state_sha256="a"*64,input_training_sha256="b"*64,anchor_training_sha256=audit.array_sha(base))
    cell=fit_risk_cell(tr,ev,tt,te,et,ee,base,offset,ridge_lambda=1.,role="outer_final",bindings=binding,
        selection_record={"uses_outer_test":False,"ridge_lambda":1.})
    compact={"train_time":tt,"train_event":te,"evaluation_time":et,"evaluation_event":ee,
             "training_offset":base,"evaluation_offset":offset,"train_design":cell.design.transform(tr),"evaluation_design":cell.design.transform(ev)}
    root.mkdir(parents=True)
    artifact=root/"lambda1.0.private.pkl";artifact.write_bytes(pickle.dumps(cell,protocol=5))
    row={**cell.receipt,"artifact_sha256":audit.sha(artifact),"saved_linear_head_replay":replay_risk_cell(cell,ev,offset,tt,te,et,ee)}
    (root/"lambda1.0.private.json").write_text(json.dumps(row))
    (root/"lambda1.0.compact.private.pkl").write_bytes(pickle.dumps(compact,protocol=5))
    split={"fit":list(range(n)),"evaluation":list(range(n,n+m))}
    anchor=SimpleNamespace(training_offset=base,evaluation_offset=offset)
    return expected,phase,split,np.r_[tt,et],np.r_[te,ee],anchor,cell,compact,row


def test_real_small_synthetic_saved_head_has_independent_math_and_no_forward(tmp_path):
    values=synthetic_head(tmp_path/"risk")
    result,case=audit.verify_head(tmp_path/"risk",*values[:6])
    assert result["max_abs_evaluation_risk_error"]==0.
    assert result["independent_ridge_gradient_max"]<=2e-5
    assert result["metric_component_max_abs_error"]<1e-9
    assert case["risk"].shape==(10,)


@pytest.mark.parametrize("tamper",["risk_bytes","canonical_row_order","design_bytes","anchor_offset","coefficient"])
def test_saved_head_risk_label_design_anchor_coefficient_tampering_is_rejected(tmp_path,tamper):
    folder=tmp_path/"risk";values=list(synthetic_head(folder))
    expected,phase,split,t,e,anchor,cell,compact,row=values
    if tamper=="risk_bytes":
        with (folder/"lambda1.0.private.pkl").open("ab") as f:f.write(b" ")
    elif tamper=="canonical_row_order":
        compact["evaluation_time"]=compact["evaluation_time"][::-1].copy()
    elif tamper=="design_bytes":
        compact["train_design"][0,0]+=.01
    elif tamper=="anchor_offset":
        anchor.evaluation_offset=anchor.evaluation_offset+.01
    else:
        cell.model.coefficient[0]+=.01
        (folder/"lambda1.0.private.pkl").write_bytes(pickle.dumps(cell,protocol=5))
        row["artifact_sha256"]=audit.sha(folder/"lambda1.0.private.pkl")
        (folder/"lambda1.0.private.json").write_text(json.dumps(row))
    (folder/"lambda1.0.compact.private.pkl").write_bytes(pickle.dumps(compact,protocol=5))
    with pytest.raises(ValueError):
        audit.verify_head(folder,expected,phase,split,t,e,anchor)


def test_outer_phase_hashes_global_integer_rows_and_only_saved_fixed_epoch_states(tmp_path):
    private=tmp_path;split={"role":"outer_final","fit":[0,2,4],"evaluation":[1,3]}
    definition={"name":"SYNTHETIC_OUTER","source_arm":"B4"}
    lock={"phase_a_lock":{"frozen_sources":{"synthetic.py":"a"*64}},"phase_a_families":{"MAIN":definition}}
    folder=private/"phase_a/outer1_final/MAIN_seed31";folder.mkdir(parents=True)
    state=folder/"state_epoch050.private.pt";state.write_bytes(b"owned_synthetic_checkpoint_not_a_model")
    prep=private/"preprocessing/prep.private.pkl";prep.parent.mkdir();prep.write_bytes(b"owned_synthetic_preprocessing")
    row={"status":"COMPLETE","identity":"NEW_FIT","seed":31,"frozen_sources":lock["phase_a_lock"]["frozen_sources"],
        "master_lock_sha256":audit.canonical(lock["phase_a_lock"]),"DSS_OS_PFI_accessed":False,"P2_main_pretraining_eligible":True,
        "variant":"SYNTHETIC_OUTER","spec":{"source_arm":"B4"},"input_sha256":"d"*64,
        "train_rows_sha256":audit.metadata_tensor_sha(np.array(split["fit"],np.int64)),"valid_rows_sha256":None,
        "optimizer_steps":50,"history":[{"epoch":i,"optimizer_step":i,"gradient_norms":{"weight":1.},
            "train_program_loss":.1,"train_pair_loss":.2,"train_auxiliary_loss":.12} for i in range(1,51)],
        "evaluate_valid":False,"train_patients":3,"valid_patients":0,"fixed_epoch":50,"selected_epoch":50,
        "states":[{"epoch":50,"file":state.name,"sha256":audit.sha(state),"actual_replay":True,"replay_max_error":0.}],
        "source_state_file":state.name,"selected_checkpoint_sha256":audit.sha(state)}
    path=folder/"cell.private.json";path.write_text(json.dumps(row))
    binding={"p2_lock_sha256":"c"*64,"split":"outer1_final","family":"MAIN","seed":31,"cell_receipt_sha256":audit.sha(path),
        "training_only_input_sha256":"d"*64,"shared_preprocessing":{"file":prep.name,"sha256":audit.sha(prep)}}
    (folder/"MATRIX_BINDING.private.json").write_text(json.dumps(binding))
    actual=audit.verify_phase(private,lock,"c"*64,"outer1_final","MAIN",31,np.array(["synthetic"]*5),split)
    assert actual["optimizer_steps"]==50 and len(actual["checkpoint_hashes"])==1
    state.write_bytes(state.read_bytes()+b"x")
    with pytest.raises(ValueError,match="ARTIFACT_SHA_CHANGED"):
        audit.verify_phase(private,lock,"c"*64,"outer1_final","MAIN",31,np.array(["synthetic"]*5),split)


def test_seed_aggregation_uses_five_fold_micro_then_equal_seed_means_not_raw_cells():
    rows=[]
    for seed,ratio in zip(audit.SEEDS,(.3,.5,.7)):
        for fold in audit.FOLDS:
            den=1 if fold==1 else 100;num=1 if fold==1 else den*ratio
            rows.append({"arm":"B5","seed":seed,"fold":fold,"metrics":{
                "Harrell_num":num,"Harrell_den":den,"Uno_num":num*2,"Uno_den":den*3}})
    actual=audit.aggregate(rows)["B5"]
    assert actual["Harrell"]==pytest.approx(np.mean([(1+400*r)/401 for r in (.3,.5,.7)]))
    assert actual["Uno60"]==pytest.approx(actual["Harrell"]*2/3)
    with pytest.raises(ValueError,match="INCOMPLETE"):
        audit.aggregate(rows[:-1])


def test_public_output_disallows_vectors_ids_and_self_private_paths(tmp_path):
    for value in ({"bad":np.arange(3)},{"risk_vector":[1.,2.]},{"path":"/root/"+"private_file"}):
        with pytest.raises(ValueError):audit.public_write(tmp_path/"not_written.json",value)
    audit.public_write(tmp_path/"good.json",{"status":"synthetic","count":5,"new_neural_forwards":0,
        "official_definition_sources":audit.OFFICIAL_SOURCES})
    assert (tmp_path/"good.json").is_file()


def test_official_child_receives_private_arrays_only_in_stdin_and_requires_package_version(monkeypatch):
    def fake_run(command,**kwargs):
        assert command[1]=="-I" and kwargs["env"]["CUDA_VISIBLE_DEVICES"]==""
        payload=json.loads(kwargs["input"])
        assert payload["cases"][0]["risk"]==[1.,0.]
        return SimpleNamespace(returncode=0,stdout=json.dumps({"status":"PASS_REAL_OFFICIAL_PACKAGE_METRICS_API","cases":1,"package_version":"0.23.1"}),stderr="")
    monkeypatch.setattr(audit.subprocess,"run",fake_run)
    value=audit.official_oracle([{"risk":np.array([1.,0.])}],"synthetic_isolated_target")
    assert value["cases"]==1


def test_canonical_reader_only_loads_dss_fields_and_preserves_patient_dtype(tmp_path,monkeypatch):
    patient=np.array([f"synthetic_{i:04d}" for i in range(512)],dtype="U20")
    folds=np.resize(np.arange(1,6),512)
    cohort=tmp_path/"synthetic_cohort.npz"
    # An unrelated object field would fail allow_pickle=False if accessed.
    np.savez(cohort,patient_ids=patient,folds=folds,unused_other=np.array([{"unread":1}],object))
    t=np.linspace(1,150,512);e=np.zeros(512);e[:65]=1
    endpoint=tmp_path/"synthetic_dss.npz"
    def write_endpoint(order_hash):
        np.savez(endpoint,patient_hash=order_hash,DSS_time=t,DSS_event=e,DSS_valid=np.ones(512,bool),
                 PFI_time=np.array([{"unread_secondary":True}],object))
    write_endpoint(hashlib.sha256(patient.tobytes()).hexdigest())
    monkeypatch.setattr(audit,"COHORT_SHA",audit.sha(cohort));monkeypatch.setattr(audit,"ENDPOINT_SHA",audit.sha(endpoint))
    ids,actual_folds,time,event=audit.canonical_labels(cohort,endpoint)
    assert ids.dtype==patient.dtype and np.array_equal(time,t) and event.sum()==65
    write_endpoint(hashlib.sha256(patient.astype("U14").tobytes()).hexdigest())
    monkeypatch.setattr(audit,"ENDPOINT_SHA",audit.sha(endpoint))
    with pytest.raises(ValueError,match="ENDPOINT_ORDER_CHANGED"):
        audit.canonical_labels(cohort,endpoint)


def test_independent_split_validation_rejects_row_reordering_even_with_rehashed_plan():
    from src.semantic_b0b7.p2_data import nested_split_plan
    rng=np.random.default_rng(100);folds=np.resize(np.arange(1,6),512)
    e=np.zeros(512);e[rng.choice(512,65,replace=False)]=1
    plan={"partitions":nested_split_plan(folds,e,seed=27001)}
    audit.verify_splits(plan,{"inner_split_seed":27001},np.arange(512),folds,np.arange(512),e)
    changed=copy.deepcopy(plan)
    row=changed["partitions"]["outer1_inner0"]
    row["fit"][0],row["fit"][1]=row["fit"][1],row["fit"][0]
    row["fit_sha256"]=audit.array_sha(np.asarray(row["fit"],np.int64))
    with pytest.raises(ValueError,match="NESTED_ROW_ORDER_CHANGED"):
        audit.verify_splits(changed,{"inner_split_seed":27001},np.arange(512),folds,np.arange(512),e)


def test_independent_report_comparison_catches_aggregation_change_without_reporter_import(tmp_path):
    rows=[]
    for arm in ("B0","B5"):
        for seed in ((31,) if arm=="B0" else audit.SEEDS):
            for fold in audit.FOLDS:
                num=(5 if arm=="B0" else 6)*fold;den=10*fold
                rows.append({"arm":arm,"fold":fold,"seed":seed,"metrics":{
                    "Harrell":num/den,"Harrell_num":num,"Harrell_den":den,
                    "Uno60":num/den,"Uno_num":num*2,"Uno_den":den*2}})
    summaries=audit.aggregate(rows)
    (tmp_path/"05_fold_seed_metrics.json").write_text(json.dumps([{k:v for k,v in r.items() if k!="metrics"}|r["metrics"] for r in rows]))
    aggregate_path=tmp_path/"05_seed_aggregated_metrics.json"
    aggregate_path.write_text(json.dumps(list(summaries.values())))
    (tmp_path/"06_paired_comparisons.json").write_text(json.dumps([{"left":"B5","reference":"B0",
        "Harrell_delta":.1,"Uno60_delta":.1,"confidence_intervals_computed":False,
        "bootstrap_replicates":0,"p_value_computed":False}]))
    result=audit.compare_report(tmp_path,rows,summaries)
    assert result["reporter_imported_or_called"] is False and result["fold_rows"]==20
    changed=copy.deepcopy(list(summaries.values()));next(r for r in changed if r["arm"]=="B5")["Harrell"]+=.001
    aggregate_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match="AGGREGATION_CHANGED"):
        audit.compare_report(tmp_path,rows,summaries)


def test_forged_solver_objective_is_rejected_by_fresh_efron_ridge_math(tmp_path):
    folder=tmp_path/"risk";expected,phase,split,t,e,anchor,cell,c,row=synthetic_head(folder)
    row["risk_optimizer"]["objective"]+=.01
    (folder/"lambda1.0.private.json").write_text(json.dumps(row))
    with pytest.raises(ValueError,match="EFRON_RIDGE_OBJECTIVE_OR_GRADIENT_CHANGED"):
        audit.verify_head(folder,expected,phase,split,t,e,anchor)


@pytest.mark.parametrize("corruption",["one_missing","duplicate","unexpected","all_files_missing"])
def test_independent_report_requires_all_twelve_comparisons_exactly_once(tmp_path,corruption):
    arms={arm for pair in audit.CORE_COMPARISONS for arm in pair}
    rows=[]
    for arm in sorted(arms):
        for seed in ((31,) if arm=="B0" else audit.SEEDS):
            for fold in audit.FOLDS:
                rows.append({"arm":arm,"fold":fold,"seed":seed,"metrics":{
                    "Harrell":.5,"Harrell_num":5*fold,"Harrell_den":10*fold,
                    "Uno60":.5,"Uno_num":10*fold,"Uno_den":20*fold}})
    summaries=audit.aggregate(rows)
    (tmp_path/"05_fold_seed_metrics.json").write_text(json.dumps([
        {k:v for k,v in r.items() if k!="metrics"}|r["metrics"] for r in rows]))
    (tmp_path/"05_seed_aggregated_metrics.json").write_text(json.dumps(list(summaries.values())))
    pairs=[{"left":a,"reference":b,"Harrell_delta":0.,"Uno60_delta":0.,
        "confidence_intervals_computed":False,"bootstrap_replicates":0,"p_value_computed":False}
        for a,b in audit.CORE_COMPARISONS]
    path=tmp_path/"06_paired_comparisons.json"
    path.write_text(json.dumps(pairs))
    assert audit.compare_report(tmp_path,rows,summaries)["required_comparisons_verified"]==12
    if corruption=="one_missing":pairs.pop()
    elif corruption=="duplicate":pairs[-1]=copy.deepcopy(pairs[0])
    elif corruption=="unexpected":pairs[-1]["left"]="B3"
    else:path.unlink()
    if corruption!="all_files_missing":path.write_text(json.dumps(pairs))
    with pytest.raises(ValueError,match="REPORT_REQUIRED_COMPARISON_SET_CHANGED"):
        audit.compare_report(tmp_path,rows,summaries)


def test_verifier_source_has_no_producer_scorer_reporter_or_neural_import():
    import ast
    tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):imports.append(node.module or "")
        elif isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
    assert not any(x.startswith(("torch","src.semantic_b0b7","scripts.semantic_b0b7")) for x in imports)

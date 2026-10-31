"""Qualification software only: no real patient payload and no optimizer fit."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import semantic_b0b7_p2_leakage_audit as audit


def qualification_configuration():
    master=json.loads((audit.ROOT/"configs/semantic_b0b7/p2_lock.json").read_text(encoding="utf-8"))
    return {"schema":"semantic_b0b7_p2_leakage_qualification_lock_v1","status":"LOCKED_BEFORE_TWO_AUDIT_FITS",
        "P2_master_lock_sha256":audit.MASTER_SHA,"formal_scientific_source_set_sha256":audit.canonical(master["frozen_sources"]),
        "outer":1,"fit_patients":409,"evaluation_patients":103,"seed":31,"family":"MAIN","fixed_epochs":50,
        "planned_audit_fits":2,"planned_audit_optimizer_updates":100,"heldout_permutation_seed":27131,
        "phase_a_definition":master["phase_a_families"]["MAIN"],"evaluation_metrics_used":False,
        "architecture_selection_performed":False,"DSS_values_never_passed_to_Phase_A_optimizer":True,
        "audit_sources":{name:audit.sha(audit.ROOT/name) for name in (audit.AUDIT_SCRIPT,audit.AUDIT_TEST)}}


def test_two_fit_private_body_requires_permission_before_any_input_read(monkeypatch):
    monkeypatch.setattr(audit,"sha",lambda *a:(_ for _ in ()).throw(AssertionError("must not read")))
    with pytest.raises(ValueError,match="AUTHORIZATION_REQUIRED_BEFORE_INPUT_READ"):
        audit.run_qualification("unread","unread","unread",assets_path="unread",legacy_root="unread",
            frozen_images_path="unread",formal_private="unread",private_output="unread",public_output="unread")


def test_derangement_is_fixed_nontrivial_and_does_not_use_outcomes():
    a=audit.derangement(103,27131);b=audit.derangement(103,27131)
    assert np.array_equal(a,b) and (a!=np.arange(103)).all() and set(a)==set(range(103))
    with pytest.raises(ValueError):audit.derangement(1,27131)


def synthetic_raw(n=20,g=20):
    rng=np.random.default_rng(2026)
    rna=rng.normal(size=(n,g));cna=rng.normal(size=(n,g))
    rna[::4,0]=np.nan;cna[::3,1]=np.nan
    clinical={"AGE":np.array([str(40+i) for i in range(n)]),"SEX":np.where(np.arange(n)%2,"MALE","FEMALE"),
        "AJCC_PATHOLOGIC_TUMOR_STAGE":np.where(np.arange(n)%2,"STAGE III","STAGE II"),
        "PATH_T_STAGE":np.where(np.arange(n)%2,"T3","T2"),"PATH_N_STAGE":np.where(np.arange(n)%2,"N1","N0"),
        "PATH_M_STAGE":np.array(["M0"]*n)}
    return {"rna":rna,"rna_measured":np.isfinite(rna),"cna":cna,"cna_measured":np.isfinite(cna),
        "platform":np.arange(n)%3,"clinical_raw":clinical,"folds":np.resize(np.arange(1,6),n),
        "cancer":np.array(["COAD"]*n),"patient_ids":np.array([f"synthetic_{i}" for i in range(n)]),
        "audit":{"cohort_sha256":"f"*64,"endpoint_read":False}}


def test_raw_perturbation_preserves_all_training_bytes_and_swaps_masks_together():
    raw=synthetic_raw();ev=np.arange(12,20);fit=np.arange(12)
    altered,perm,report=audit.perturb_raw(raw,ev,seed=27131)
    for key in ("rna","cna","rna_measured","cna_measured","platform"):
        assert audit.array_sha(altered[key][fit])==audit.array_sha(raw[key][fit])
        assert audit.array_sha(altered[key][ev])==audit.array_sha(raw[key][ev[perm]])
    for key in raw["clinical_raw"]:
        assert np.array_equal(altered["clinical_raw"][key][fit],raw["clinical_raw"][key][fit])
    assert report["rna_heldout_changed"] and report["cna_heldout_changed"] and report["clinical_fields_changed"]>0
    assert not report["selected_using_DSS_or_model_scores"]


def test_real_builder_on_synthetic_arrays_has_identical_train_and_changed_eval():
    from src.semantic_b0b7.p1r_data import build_partition_inputs,require_no_outcomes
    from scripts.semantic_b0b7_p2_matrix import training_input_binding
    from scripts.semantic_b0b7_p1r import validate_scope
    raw=synthetic_raw();n=len(raw["folds"]);fit=np.arange(12);ev=np.arange(12,n)
    altered,perm,_=audit.perturb_raw(raw,ev,seed=27131)
    names=["synthetic_program_"+str(i) for i in range(50)]
    frozen={"global_rows":np.arange(n),"program_names":names,"frozen_input_sha256":"e"*64,
        "image":{"patch_values":torch.randn(n,2,768),"patch_aligned":torch.randn(n,2,512),
                 "patch_mask":torch.ones(n,2,dtype=torch.bool),"patch_slide":torch.zeros(n,2,dtype=torch.long),
                 "titan_native":torch.randn(n,768),"titan_aligned":torch.randn(n,512),"titan_scores":torch.randn(n,32)},
        "conch_prototypes":torch.randn(8,512),"titan_prototypes":torch.randn(8,512),
        "relation_prior":torch.ones(8,50),"shuffled_relation_prior":torch.ones(8,50)}
    memberships=(names,np.ones((50,20),bool),np.ones((50,20),bool))
    a,_=build_partition_inputs(raw,frozen,fit,ev,memberships,split_name="synthetic_audit")
    b,_=build_partition_inputs(altered,frozen,fit,ev,memberships,split_name="synthetic_audit")
    assert audit.payload_fingerprints(a,a["train_indices"].numpy())==audit.payload_fingerprints(b,b["train_indices"].numpy())
    assert training_input_binding(a,"e"*64)==training_input_binding(b,"e"*64)
    assert audit.payload_fingerprints(a,a["valid_indices"].numpy())!=audit.payload_fingerprints(b,b["valid_indices"].numpy())
    dss={"t":np.arange(n).astype(float),"e":(np.arange(n)%3==0).astype(float),"valid":np.ones(n,bool)}
    changed=audit.perturb_dss(dss,ev,perm)
    assert np.array_equal(changed["t"][fit],dss["t"][fit]) and not np.array_equal(changed["t"][ev],dss["t"][ev])
    with pytest.raises(ValueError,match="ENDPOINT_BEARING_MAPPING"):
        require_no_outcomes({**raw,**changed})
    with pytest.raises(ValueError,match="OUTCOME_BEARING_PHASE_A_PAYLOAD"):
        validate_scope({**b,"e":changed["e"]},b["train_indices"],None,evaluate_valid=False)


def test_recursive_comparison_is_byte_exact_including_signed_zero_and_optimizer():
    a={"model":{"w":torch.tensor([1.,0.])},"optimizer":{"step":torch.tensor(50.),"exp_avg":torch.tensor([.1,.2])},
       "rng":torch.tensor([1,2,3],dtype=torch.uint8),"numpy":np.array([1,2]),"metadata":(None,31,.1)}
    assert audit.exact_tree(a,copy.deepcopy(a))["equal"]
    b=copy.deepcopy(a);b["model"]["w"][1]=-0.
    assert not audit.exact_tree(a,b)["equal"]
    b=copy.deepcopy(a);b["optimizer"]["exp_avg"][0]+=.01
    assert not audit.exact_tree(a,b)["equal"]
    b=copy.deepcopy(a);b["rng"][0]=0
    assert not audit.exact_tree(a,b)["equal"]


def checkpoint():
    return {"model":{"weight":torch.tensor([1.,2.])},"optimizer":{"state":{0:{"step":torch.tensor(50.),"exp_avg":torch.tensor([.1,.2])}}},
            "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.tensor([1,2],dtype=torch.uint8),
            "expected_replay":{"prediction":torch.ones(3,50),"pair_valid":torch.ones(3,3,dtype=torch.bool)},
            "epoch":50,"input_sha256":"a"*64,"validation":None}


def test_checkpoint_container_hash_may_differ_but_every_decoded_field_must_match(tmp_path):
    first=tmp_path/"first.pt";second=tmp_path/"second.pt";state=checkpoint()
    torch.save(state,first);torch.save(state,second)
    result=audit.compare_checkpoints(first,second)
    assert result["entire_decoded_checkpoint_exact"] and result["components"]["optimizer"]["equal"]
    assert result["new_model_forwards_for_comparison"]==0
    changed=copy.deepcopy(state);changed["cpu_rng"][0]=changed["cpu_rng"][0]^1;torch.save(changed,second)
    with pytest.raises(ValueError,match="MUTATION_CHANGED_DECODED"):
        audit.compare_checkpoints(first,second)


def test_prelock_budget_and_sources_are_checked(tmp_path):
    lock=qualification_configuration();path=tmp_path/"qualification.json";path.write_text(json.dumps(lock))
    audit.validate_qualification_lock(audit.ROOT/"configs/semantic_b0b7/p2_lock.json",path,audit.sha(path))
    lock["fixed_epochs"]=100;path.write_text(json.dumps(lock))
    with pytest.raises(ValueError,match="UNLOCKED_QUALIFICATION_BUDGET"):
        audit.validate_qualification_lock(audit.ROOT/"configs/semantic_b0b7/p2_lock.json",path,audit.sha(path))


def test_active_formal_guard_stops_before_gpu_probe(tmp_path,monkeypatch):
    (tmp_path/"ACTIVE_SCHEDULER_OWNER.private.json").write_text("{}")
    monkeypatch.setattr(audit.subprocess,"run",lambda *a,**kw:(_ for _ in ()).throw(AssertionError("no GPU probe")))
    with pytest.raises(ValueError,match="SCHEDULER_STILL_ACTIVE"):
        audit.assert_formal_finished(tmp_path,{})


def test_facts_use_existing_real_receipts_and_label_synthetic_vs_pending(tmp_path):
    lock=qualification_configuration();path=tmp_path/"qualification.json";path.write_text(json.dumps(lock))
    result=audit.build_fact_reports(audit.ROOT/"configs/semantic_b0b7/p2_lock.json",path,
        asset_binding=audit.ROOT/"tmp/SEMANTIC_B0B7_P2_ASSET_BINDING_20261031.json",
        profile=audit.ROOT/"tmp/SEMANTIC_B0B7_P2_GPU_PROFILE_V3_20261031.json",
        api_receipt=audit.ROOT/"reports/semantic_b0b7_p2/OFFICIAL_SKSURV_API_QUALIFICATION.json",
        e0_receipt=audit.ROOT/"reports/semantic_b0b7_p2/E0_ANCHOR_PROFILE_AUDIT.json",
        parent_binding=audit.ROOT/"reports/semantic_b0b7_p2/00_PARENT_PUBLICATION_BINDING.json",
        software_junit=audit.ROOT/"tmp/semantic_b0b7_p2_risk_tests.xml",output=tmp_path/"facts")
    assert len(result["files"])==8
    report=json.loads((tmp_path/"facts/03_NO_DSS_PRETRAIN_AND_CLINICAL_CROSSFIT_AUDIT.json").read_text(encoding="utf-8"))
    assert report["live_heldout_mutation_qualification"]=="PRELOCKED_NOT_RUN"
    assert report["software_tests"]["tests"]==21 and "synthetic" in report["software_tests"]["scope"]
    assert report["real_E0"]["Cox_fits"]==52 and report["real_E0"]["scalar_fits"]==4
    assert report["DSS_predictive_gain_inferred"] is False

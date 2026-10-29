"""Anonymous administrative boundary tests; no experiment or Git operations."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts import parallel_evidence_publication as publisher


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding="utf-8")


@pytest.fixture
def protocol(tmp_path,monkeypatch):
    root=tmp_path/"repo";root.mkdir();private=tmp_path/"private";private.mkdir()
    groups=dict(patient=["FAKE_SUBJECT_"+str(i) for i in range(512)],
                donor=["FAKE_DONOR_"+str(i) for i in range(6)],
                section=["FAKE_SECTION_"+str(i) for i in range(12)],
                barcodes=["FAKE_SPOT_"+str(i) for i in range(1001)])
    cohort="a"*64;stage="b"*64
    patient=hashlib.sha256("\n".join(groups["patient"]).encode()).hexdigest()
    st=publisher.canonical({k:groups[k] for k in ("donor","section","barcodes")})
    for name,value in (("CANONICAL_COHORT_SHA256",cohort),("CANONICAL_PATIENT_SHA256",patient),
                       ("CANONICAL_ST_PREPARED_SHA256",stage),("CANONICAL_ST_IDENTIFIER_SHA256",st)):
        monkeypatch.setattr(publisher.inherited,name,value)
    deny=private/"deny.json";put(deny,dict(schema="bgb_en_private_publication_denylist_v1",
        cohort_sha256=cohort,patient_order_sha256=patient,stage_b_sha256=stage,groups=groups))
    code="src/spgel/models.py";(root/code).parent.mkdir(parents=True);(root/code).write_text("# anonymous frozen code\n")
    lock=dict(schema="spgel_source_run_lock_v1",status="LOCKED_BEFORE_SPGEL_FITS",
        sources={code:publisher.sha256(root/code)},pilot=dict(logical_cells=72,actual_saved_states=192),
        s4=dict(neural_fits=2,actual_saved_states=6))
    lock["contract_sha256"]=publisher.canonical(lock)
    lock_path=private/"SOURCE_RUN_LOCK.json";put(lock_path,lock)
    public_lock="reports/spgel/SOURCE_RUN_LOCK.json"
    note="reports/spgel/S0_PROTOCOL.md";(root/note).parent.mkdir(parents=True);(root/note).write_text("RUNNING; no completion claim\n")
    historical=root/"reports/bgb_en_r7/RESULTS.md";historical.parent.mkdir(parents=True);historical.write_text("unchanged previous work\n")
    names=[code,note,public_lock]
    rows=[dict(file=n,sha256=publisher.sha256(lock_path if n==public_lock else root/n),
               bytes=(lock_path if n==public_lock else root/n).stat().st_size) for n in names]
    files=private/"files.json";put(files,dict(files=rows))
    return dict(root=root,private=private,deny=deny,lock=lock_path,files=files,note=note,rows=rows,
                historical=historical,secret=groups["barcodes"][17])


def run(f,stage="protocol"):
    return publisher.publish(f["root"],f["files"],"spgel",stage,f["lock"],f["deny"],
                             private=f["private"],copy_public_lock=True)


def test_protocol_manifest_binds_exact_bytes_includes_audit_and_excludes_self(protocol):
    f=protocol;history=publisher.sha256(f["historical"]);result=run(f)
    assert result["stage"]=="protocol" and result["git_operations"]==result["model_loads"]==0
    manifest=json.loads((f["root"]/"reports/spgel/PUBLICATION_MANIFEST.json").read_text())
    names={r["file"] for r in manifest["files"]}
    assert "reports/spgel/SOURCE_PUBLICATION_AUDIT.json" in names
    assert "reports/spgel/PUBLICATION_MANIFEST.json" not in names
    assert publisher.sha256(f["historical"])==history
    assert (f["root"]/"reports/spgel/SOURCE_RUN_LOCK.json").read_bytes()==f["lock"].read_bytes()


def test_complete_cannot_be_claimed_from_protocol_or_counts_only(protocol):
    with pytest.raises((publisher.inherited.PublicationError,FileNotFoundError)):
        run(protocol,"complete")
    assert not (protocol["root"]/"reports/spgel/PUBLICATION_MANIFEST.json").exists()


@pytest.mark.parametrize("mutation",["unknown_file","cross_line","binary_model","wrong_sha"])
def test_only_explicit_line_scoped_frozen_text_files_can_be_published(protocol,mutation):
    f=protocol;rows=list(f["rows"])
    if mutation=="wrong_sha":rows[0]=dict(rows[0],sha256="f"*64)
    else:
        relative={"unknown_file":"untracked.txt","cross_line":"reports/bgb_en_r8/secret.md",
                  "binary_model":"reports/spgel/model.pt"}[mutation]
        path=f["root"]/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("anonymous")
        rows.append(dict(file=relative,sha256=publisher.sha256(path),bytes=path.stat().st_size))
    put(f["files"],dict(files=rows))
    with pytest.raises(publisher.inherited.PublicationError):run(f)


def test_exact_identifier_is_rejected_without_printing_its_value(protocol):
    f=protocol;(f["root"]/f["note"]).write_text("must reject "+f["secret"])
    for row in f["rows"]:
        if row["file"]==f["note"]:row.update(sha256=publisher.sha256(f["root"]/f["note"]),bytes=(f["root"]/f["note"]).stat().st_size)
    put(f["files"],dict(files=f["rows"]))
    with pytest.raises(publisher.inherited.PublicationError) as error:run(f)
    assert f["secret"] not in str(error.value)
    assert not (f["root"]/"reports/spgel/PUBLICATION_MANIFEST.json").exists()


def test_source_drift_is_rejected_before_public_lock_copy(protocol):
    f=protocol;(f["root"]/"src/spgel/models.py").write_text("changed source")
    with pytest.raises(publisher.inherited.PublicationError,match="LOCKED_SCIENTIFIC_SOURCE_BYTES_CHANGED"):
        run(f)
    assert not (f["root"]/"reports/spgel/SOURCE_RUN_LOCK.json").exists()


def source_acceptance_fixture(f):
    root,private=f["root"],f["private"]
    script="scripts/check_spgel_frozen_delivery.py"
    (root/script).parent.mkdir(parents=True,exist_ok=True);(root/script).write_text("# synthetic independent acceptance\n")
    audits=[]
    for stage,units,states in (("pilot",72,192),("s4",2,6)):
        path=private/stage/"REPLAY_AUDIT.json";put(path,dict(synthetic=True))
        audits.append(dict(stage=stage,SHA256=publisher.sha256(path),replayed_models=units,replayed_states=states))
    exports=[];rows=[]
    for arm in ("S_A5","S_A1"):
        folder=private/"s4/export"/f"{arm}_seed31";folder.mkdir(parents=True)
        put(folder/"source_encoder_manifest.json",dict(synthetic=True))
        (folder/"encoder_state.pt").write_bytes(b"NOT_A_LOADABLE_MODEL")
        checkpoint=private/"s4/cells"/f"{arm}_full_seed31"/"checkpoint_epoch40.pt"
        checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b"NOT_A_LOADABLE_CHECKPOINT")
        manifest_sha=publisher.sha256(folder/"source_encoder_manifest.json")
        encoder_sha=publisher.sha256(folder/"encoder_state.pt")
        exports.append(dict(arm=arm,manifest_sha256=manifest_sha,encoder_state_sha256=encoder_sha))
        rows.append(dict(arm=arm,source_manifest_SHA256=manifest_sha,exported_encoder_SHA256=encoder_sha,
            original_checkpoint_SHA256=publisher.sha256(checkpoint),image_sections=12,image_spots=13011,
            image_API_gene_forward_calls=0,all_exported_parameters_frozen=True,new_optimizer_fits=0,new_optimizer_updates=0,
            actual_TCGA_bulk_qualification="NOT_RUN_SOURCE_SPOT_VALUES_NOT_PASSED_AS_BULK",
            maximum_errors=dict(image_prediction_z=0.,image_embedding=0.,image_program_map=0.),
            source_conditional_maximum_errors=({k:0. for k in ("source_gene_token","conditional_image_read",
                "conditional_gene_read","paired_scores","route_original_window")} if arm=="S_A5" else {}),
            source_conditional_batches=26 if arm=="S_A5" else 0,
            padding_multislide=dict(batch_rows=2,complete_sections=2,padding_positions=3,maximum_error=0.)))
    exported=dict(status="COMPLETE_TWO_REPLAYED_SOURCE_INTERFACES",exports=exports)
    put(private/"s4/EXPORT_COMPLETE.private.json",exported)
    lock=json.loads(f["lock"].read_text())
    value=dict(schema="spgel_actual_frozen_source_delivery_acceptance_v1",
        status="PASS_TWO_ACTUAL_S4_ENCODERS_FULL_SOURCE_CONTEXT_AND_BOUNDARIES",
        source_contract_SHA256=lock["contract_sha256"],acceptance_script_SHA256=publisher.sha256(root/script),
        source_export_complete_SHA256=publisher.sha256(private/"s4/EXPORT_COMPLETE.private.json"),
        source_encoders_count=2,source_sections_count=12,source_spots_count=13011,
        new_optimizer_fits=0,new_optimizer_updates=0,TCGA_loaded=False,DSS_or_PFI_read=False,
        source_or_target_labels_selected_model=False,source_ST_measurements_used_only_for_source_acceptance=True,
        public_contains_spot_predictions_coordinates_or_weights=False,
        TCGA_bulk_qualification="NOT_RUN_REQUIRES_SEPARATE_TRAIN_ONLY_DOMAIN_CONTRACT",
        prior_actual_replay_audits=audits,rows=rows)
    name="reports/spgel/S4_ACCEPTANCE.json"
    return lock,exported,value,{name,script},name


def seal_acceptance(f,value,name):
    put(f["root"]/name,value)
    put(f["private"]/"s4/frozen_delivery_acceptance_v1/COMPLETE.private.json",value)


def test_real_source_interface_acceptance_does_not_require_new_tcga_bulk_qualification(protocol):
    f=protocol;lock,exported,value,listed,name=source_acceptance_fixture(f)
    seal_acceptance(f,value,name)
    result=publisher.validate_source_delivery_acceptance(f["root"],f["private"],lock,listed,exported)
    assert result["actual_interface_encoders"]==2
    assert result["new_TCGA_bulk_qualification_required_for_source_publication"] is False
    assert publisher.allowed_namespace("scripts/check_spgel_frozen_delivery.py","spgel",lock)
    assert publisher.allowed_namespace("scripts/verify_spgel_aggregate_tables.py","spgel",lock)
    assert not publisher.allowed_namespace("scripts/unregistered_spgel_method.py","spgel",lock)


def test_export_files_alone_are_not_actual_interface_acceptance(protocol):
    f=protocol;lock,exported,_,listed,name=source_acceptance_fixture(f)
    listed.remove(name)
    with pytest.raises(publisher.inherited.PublicationError,match="ACTUAL_INTERFACE_ACCEPTANCE_REQUIRED"):
        publisher.validate_source_delivery_acceptance(f["root"],f["private"],lock,listed,exported)


@pytest.mark.parametrize("mutation",["unfrozen","gene_in_image","bad_forward","missing_reverse",
                                   "changed_weight_bytes","survival_read"])
def test_actual_interface_acceptance_scope_and_artifacts_are_fail_closed(protocol,mutation):
    f=protocol;lock,exported,value,listed,name=source_acceptance_fixture(f)
    if mutation=="unfrozen":value["rows"][0]["all_exported_parameters_frozen"]=False
    if mutation=="gene_in_image":value["rows"][0]["image_API_gene_forward_calls"]=1
    if mutation=="bad_forward":value["rows"][0]["maximum_errors"]["image_prediction_z"]=1e-3
    if mutation=="missing_reverse":value["rows"][0]["source_conditional_maximum_errors"].pop("conditional_gene_read")
    if mutation=="survival_read":value["DSS_or_PFI_read"]=True
    seal_acceptance(f,value,name)
    if mutation=="changed_weight_bytes":
        (f["private"]/"s4/export/S_A5_seed31/encoder_state.pt").write_bytes(b"CHANGED_UNLOADABLE_BYTES")
    with pytest.raises(publisher.inherited.PublicationError):
        publisher.validate_source_delivery_acceptance(f["root"],f["private"],lock,listed,exported)


@pytest.mark.parametrize("update_reference",[False,True])
def test_lossless_diagnostics_cannot_change_even_after_rehashing_its_reference(tmp_path,update_reference):
    from scripts import shard_spgel_public_report as sharder
    root=tmp_path/"repo";directory=root/"reports/spgel";directory.mkdir(parents=True)
    script="scripts/shard_spgel_public_report.py"
    (root/script).parent.mkdir(parents=True)
    shutil.copyfile(Path(sharder.__file__),root/script)
    source=dict(schema="spgel_source_metadata_report_v1",status="COMPLETE_REGISTERED_SOURCE_PILOT_AND_REPLAY",
        contract_sha256="a"*64,counters=dict(actual_saved_state_replays=192),
        donor_rows=[dict(anonymous_donor_code=i) for i in range(72)],
        fixed_checkpoint_diagnostics=[dict(arm="ANONYMOUS",heldout_donor_code=0,seed=31,
            snapshot="checkpoint"+str(i),scalar_loss=.5) for i in range(192)])
    original=tmp_path/"original.private.json";original.write_bytes(sharder.pretty(source))
    sharder.split(original,directory)
    listed={script,*{"reports/spgel/"+n for n in (sharder.MAIN,sharder.DIAGNOSTIC,sharder.AUDIT)}}
    restored,audit=publisher.load_lossless_source_report(root,listed)
    assert restored==source and audit["original_producer_sha256"]==publisher.sha256(original)
    diagnostic=json.loads((directory/sharder.DIAGNOSTIC).read_text())
    diagnostic["records"][0]["scalar_loss"]+=.1
    (directory/sharder.DIAGNOSTIC).write_bytes(sharder.compact(diagnostic))
    if update_reference:
        main=json.loads((directory/sharder.MAIN).read_text())
        main["fixed_checkpoint_diagnostics"]["sha256"]=publisher.sha256(directory/sharder.DIAGNOSTIC)
        (directory/sharder.MAIN).write_bytes(sharder.compact(main))
    expected="EXACT_ORIGINAL_PRODUCER_RECONSTRUCTION_FAILED" if update_reference else "DIAGNOSTIC_BYTES_CHANGED"
    with pytest.raises(ValueError,match=expected):
        publisher.load_lossless_source_report(root,listed)

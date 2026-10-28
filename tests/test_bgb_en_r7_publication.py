"""R7 publication guards on wholly fabricated roots; no real publication."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("r7_publication_tested",ROOT/"scripts/bgb_en_r7_publication.py")
publisher=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(publisher)
D="d"*64
NO_IDS=re.compile("NO_SYNTHETIC_IDENTIFIER_WILL_MATCH_991913")


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def qualification(root):
    for name in publisher.PROTOCOL_REQUIRED:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix==".json":write(path,{"synthetic":True})
        else:path.write_text("# synthetic placeholder; not a scientific implementation\n",encoding="utf-8")
    config=dict(trainable_arms=list(publisher.ARMS),primary_candidate="GATE1",GATE2_used_for_model_selection=False)
    config_sha=write(root/"configs/bgb_en/r7_inner_v1.json",config)
    source_names=set(publisher.CORE)|{publisher.BASE+"PROTOCOL.md"}
    i=0
    while len(source_names)<40:
        name=f"src/synthetic_locked_source_{i:02d}.py";i+=1;source_names.add(name)
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("# synthetic immutable source\n",encoding="utf-8")
    lock=dict(schema="bgb_en_r7_target_protocol_lock_v1",status="LOCKED_BEFORE_R7_FITS",planned_neural_fits=99,
        planned_target_optimizer_updates=7920,teacher_refits=0,source_refits=0,outer_test_access=False,
        config_sha256=config_sha,sources={name:publisher.inherited.sha256(root/name) for name in source_names},
        calibration_receipt_sha256={f"p{p}_seed{s}":D for p in (0,1,2) for s in (31,47,83)})
    lock["contract_sha256"]=publisher.canonical(lock);write(root/publisher.BASE/"PROTOCOL_LOCK.json",lock)
    records=[]
    for arm in publisher.ARMS:
        controls={"wrong_query":True,"mean_query":True,"interaction_off":True}
        if arm not in ("ADD","RAW"):controls["wrong_path"]=True
        records.append(dict(arm=arm,repeated_forward_max_error=0,optimizer_updates=0,own_evidence_invariants=controls,
            frozen_responses={name:dict(risk_max_absolute_change=0.,frozen_not_retrained=True) for name in controls},
            gradient_groups={"synthetic":dict(parameters=1,gradient_L2_squared=.1,finite=True)},
            derivative_probe="fixed_training_row_linear_risk_function_no_outcomes",explicit_backward_passes=1,
            complete_model_forward_calls=10,active_parameters=dict(registered=1,requires_grad=1)))
    q=dict(schema="bgb_en_r7_real_data_qualification_v1",status="PASS_R7_REAL_DATA_AND_READONLY_R6_TRANSFER",
        contract_sha256=lock["contract_sha256"],records=records,optimizer_instances=0,optimizer_updates=0,
        distinct_train_RMS_calibrations=9,scalar_scales_per_calibration=6,RAW_vs_R6_SCALARS_max_error=0,
        source_refits=0,teacher_refits=0,scalar_beta_refits=0,baseline_main_initialization_equal=True,
        qualification_weights_saved=False,calibration_fits_are_unsupervised_not_neural_fits=True,
        upstream_replay_audit_sha256=D,outer_test_access=False,new_survival_performance_estimate=False)
    write(root/publisher.BASE/"R7_0_REAL_DATA_QUALIFICATION.json",q)
    return lock


@pytest.fixture
def qualified(tmp_path):
    root=tmp_path/"synthetic_root";root.mkdir();lock=qualification(root)
    return root,lock


def seal_lock_and_qualification(root,lock):
    lock.pop("contract_sha256",None);lock["contract_sha256"]=publisher.canonical(lock)
    write(root/publisher.BASE/"PROTOCOL_LOCK.json",lock)
    path=root/publisher.BASE/"R7_0_REAL_DATA_QUALIFICATION.json";q=read(path);q["contract_sha256"]=lock["contract_sha256"];write(path,q)


def figure_metadata(root):
    directory=root/publisher.BASE;base=directory/"figures";base.mkdir(parents=True,exist_ok=True)
    source_names={"R7_0_R6_INTERFACE_AUDIT.json","R7_RESULTS.json"}|{Path(n).name for n in publisher.DIAGNOSTIC_PARTS}
    for name in source_names:write(directory/name,dict(synthetic_anonymous_aggregate=True))
    generator=root/"scripts/bgb_en_r7_plot_diagnostics.py";generator.parent.mkdir(parents=True,exist_ok=True)
    generator.write_text("# synthetic plot-generator bytes, never executed\n",encoding="utf-8")
    maps=dict(interface_trajectory={"R7_0_R6_INTERFACE_AUDIT.json"},stage_Harrell={"R7_RESULTS.json"},
        stage_Uno60={"R7_RESULTS.json"},query_pairing_deltas={"R7_RESULTS.json"},
        interaction_risk_SD={"R7_RESULTS.json"}|{Path(n).name for n in publisher.DIAGNOSTIC_PARTS},
        program_cancellation={"R7_RESULTS.json"}|{Path(n).name for n in publisher.DIAGNOSTIC_PARTS})
    figures=[]
    for name in publisher.FIGURE_NAMES:
        outputs=[]
        for suffix,content,format in ((".svg",b'<svg xmlns="http://www.w3.org/2000/svg"><defs><path id="glyph" d="M0 0"/></defs><use href="#glyph"/></svg>',"SVG"),
            (".png",b'\x89PNG\r\n\x1a\nSYNTHETIC_HEADER_ONLY_NOT_A_RENDERED_RESULT',"PNG")):
            path=base/(name+suffix);path.write_bytes(content)
            outputs.append(dict(file=path.name,format=format,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        figures.append(dict(basename=name,source_files=sorted(maps[name]),one_axes=True,uncertainty_intervals=False,pvalues=False,outputs=outputs))
    binding=dict(schema="bgb_en_r7_figure_bindings_v1",status="COMPLETE_6_ANONYMOUS_DIAGNOSTIC_FIGURES",figure_count=6,
        generator=dict(file="scripts/bgb_en_r7_plot_diagnostics.py",sha256=publisher.inherited.sha256(generator)),
        source_files=[dict(file=n,bytes=(directory/n).stat().st_size,sha256=publisher.inherited.sha256(directory/n)) for n in sorted(source_names)],figures=figures)
    write(base/"R7_FIGURE_BINDINGS.json",binding)
    return base,binding


def rebind_image(base,binding,*,svg=None,png=None):
    figure=binding["figures"][0]
    for suffix,content in ((".svg",svg),(".png",png)):
        if content is None:continue
        item=next(o for o in figure["outputs"] if o["file"].endswith(suffix));(base/item["file"]).write_bytes(content)
        item.update(bytes=len(content),sha256=hashlib.sha256(content).hexdigest())
    write(base/"R7_FIGURE_BINDINGS.json",binding)


def complete_metadata(root,lock,*,checkpoint_count=297):
    units=[]
    for arm in publisher.ARMS:
        for part in (0,1,2):
            for seed in (31,47,83):
                units.append(dict(unit=f"{arm}_p{part}_seed{seed}",arm=arm,partition=part,seed=seed,status="PASS",
                    all_checkpoints_actual_forward_replayed=True,actual_checkpoint_count=3,changed_pair_credits=0,
                    source_frozen_weight_hash_replayed=True,saved_query_mapping_recomputed=True,new_optimizer_updates=0,
                    checkpoints=[dict(epoch=e,status="PASS",actual_state_dict_forward_replayed=True,new_optimizer_updates=0,
                        changed_pair_credits=0,maximum_risk_error=1e-8,checkpoint_sha256=D) for e in (20,40,80)]))
    replay=dict(status="PASS_COMPLETE_R7_ALL_CHECKPOINT_REPLAY",contract_sha256=lock["contract_sha256"],expected_units=99,
        available_units=99,passed_units=99,failed_units=0,expected_checkpoints=297,passed_checkpoints=checkpoint_count,
        missing_or_unreplayed_units=[],units=units)
    replay_sha=write(root/publisher.BASE/"R7_CHECKPOINT_REPLAY_AUDIT.json",replay)
    desc=[]
    for path in publisher.DIAGNOSTIC_PARTS:
        digest=write(root/path,dict(schema="synthetic_diagnostic",rows=[dict(synthetic=True) for _ in range(3)]))
        desc.append(dict(file=Path(path).name,bytes=(root/path).stat().st_size,sha256=digest,rows=3))
    decision=dict(status="CURRENT_GATE1_NOT_SUPPORTED",synthetic_metadata_only=True)
    result=dict(status="COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS",contract_sha256=lock["contract_sha256"],
        replay_sha256=replay_sha,decision=decision,diagnostic_index=dict(parts=desc))
    write(root/publisher.BASE/"R7_RESULTS.json",result);write(root/publisher.BASE/"decision.json",decision)
    return result


def test_qualification_accepts_exact_zero_step_11_arm_40_source_9_cache_contract(qualified):
    root,lock=qualified
    assert publisher.qualification_guard(root)==lock
    assert len(lock["sources"])==40 and len(lock["calibration_receipt_sha256"])==9


@pytest.mark.parametrize("mutation",("main_source","source_count","cache_count","cache_keys","cache_sha","raw_error","optimizer_step","missing_arm","empty_invariants","false_invariant"))
def test_qualification_corruption_is_rejected(qualified,mutation):
    root,lock=qualified;qpath=root/publisher.BASE/"R7_0_REAL_DATA_QUALIFICATION.json";q=read(qpath)
    if mutation=="main_source":
        path=root/"src/bgb_en/r7_training.py";path.write_bytes(path.read_bytes()+b'# changed\n')
    elif mutation=="source_count":
        lock["sources"].pop(next(iter(lock["sources"])));seal_lock_and_qualification(root,lock)
    elif mutation in ("cache_count","cache_keys","cache_sha"):
        if mutation=="cache_count":lock["calibration_receipt_sha256"].pop("p0_seed31")
        elif mutation=="cache_keys":lock["calibration_receipt_sha256"]={f"bad_key_{i}":D for i in range(9)}
        else:lock["calibration_receipt_sha256"]["p0_seed31"]="not-a-hash"
        seal_lock_and_qualification(root,lock)
    else:
        if mutation=="raw_error":q["RAW_vs_R6_SCALARS_max_error"]=1e-8
        elif mutation=="optimizer_step":q["optimizer_updates"]=1
        elif mutation=="missing_arm":q["records"].pop()
        elif mutation=="empty_invariants":q["records"][0]["own_evidence_invariants"]={}
        else:q["records"][0]["own_evidence_invariants"]["wrong_query"]=False
        write(qpath,q)
    with pytest.raises(ValueError):publisher.qualification_guard(root)


def test_figure_guard_exact_35_sources_six_figures_twelve_pairs(qualified):
    root,_=qualified;figure_metadata(root)
    rows=publisher.figure_guard(root,NO_IDS)
    assert len(rows)==len({r["file"] for r in rows})==12


@pytest.mark.parametrize("kind",("empty_sources","duplicate_sources","wrong_chart_sources","duplicate_pair","source_bytes","png_magic","generator_bytes"))
def test_figure_manifest_and_bytes_corruption_rejected(qualified,kind):
    root,_=qualified;base,binding=figure_metadata(root)
    if kind=="empty_sources":binding["source_files"]=[]
    elif kind=="duplicate_sources":binding["source_files"].append(copy.deepcopy(binding["source_files"][0]))
    elif kind=="wrong_chart_sources":binding["figures"][0]["source_files"]=["R7_RESULTS.json"]
    elif kind=="duplicate_pair":binding["figures"][0]["outputs"].append(copy.deepcopy(binding["figures"][0]["outputs"][0]))
    elif kind=="source_bytes":
        path=root/publisher.BASE/"R7_RESULTS.json";path.write_bytes(path.read_bytes()+b" ")
    elif kind=="png_magic":rebind_image(base,binding,png=b'NOT_A_PNG_BUT_MATCHING_DECLARED_SHA')
    else:
        path=root/"scripts/bgb_en_r7_plot_diagnostics.py";path.write_bytes(path.read_bytes()+b'# changed\n')
    write(base/"R7_FIGURE_BINDINGS.json",binding)
    with pytest.raises(ValueError):publisher.figure_guard(root,NO_IDS)


@pytest.mark.parametrize("inside",(
    '<use href = "https://example.invalid/external.svg"/>',
    '<style>.x { fill: url("https://example.invalid/paint"); }</style>',
    '<style>@import "https://example.invalid/style.css";</style>',
    '<path d="M0 0" onclick="alert(1)"/>',
    '<script>alert(1)</script>',
    '<image href="#glyph"/>',
))
def test_svg_external_active_and_raster_content_rejected_even_after_sha_rebound(qualified,inside):
    root,_=qualified;base,binding=figure_metadata(root)
    content=('<svg xmlns="http://www.w3.org/2000/svg">'+inside+'</svg>').encode()
    rebind_image(base,binding,svg=content)
    with pytest.raises(ValueError):publisher.figure_guard(root,NO_IDS)


def synthetic_denylist(root,monkeypatch):
    inherited=publisher.inherited
    groups=dict(patient=[f"FAKE_SUBJECT_{i:04d}" for i in range(512)],donor=[f"FAKE_DONOR_{i}" for i in range(6)],
        section=[f"FAKE_SECTION_{i}" for i in range(12)],barcodes=[f"FAKE_BARCODE_{i:04d}" for i in range(1001)])
    patient_sha=hashlib.sha256('\n'.join(groups['patient']).encode()).hexdigest()
    st={key:groups[key] for key in ('donor','section','barcodes')}
    st_sha=publisher.canonical(st)
    monkeypatch.setattr(inherited,'CANONICAL_PATIENT_SHA256',patient_sha)
    monkeypatch.setattr(inherited,'CANONICAL_ST_IDENTIFIER_SHA256',st_sha)
    value=dict(schema='bgb_en_private_publication_denylist_v1',cohort_sha256=inherited.CANONICAL_COHORT_SHA256,
        patient_order_sha256=patient_sha,stage_b_sha256=inherited.CANONICAL_ST_PREPARED_SHA256,groups=groups)
    path=root/'synthetic_denylist.json';write(path,value);return path


def test_protocol_stage_does_not_require_or_claim_future_results_and_count_keys_are_safe(qualified,monkeypatch):
    root,_=qualified;denylist=synthetic_denylist(root,monkeypatch)
    assert publisher.PROTOCOL_REQUIRED<=publisher.FINAL_REQUIRED
    assert not (root/publisher.BASE/'R7_RESULTS.json').exists()
    manifest=publisher.publish(root,denylist,'protocol')
    assert manifest['scientific_run_state']=='LOCKED_RECIPE_ELIGIBILITY_PASSED_COMPLETION_NOT_CLAIMED'
    assert manifest['stage']=='protocol' and manifest['git_operations_performed']==0
    assert {r['file'] for r in manifest['files']}==set(publisher.PROTOCOL_REQUIRED)|{publisher.AUDIT}
    assert not any(r['file'] in publisher.FINAL_REPORTS or r['file'] in publisher.FIGURE_FILES for r in manifest['files'])
    audit=read(root/publisher.AUDIT)
    assert audit['formal_completion_claimed'] is False
    assert set(audit['identifier_group_counts'])=={'patient_count','donor_count','section_count','barcodes_count'}
    publisher.inherited.validate_public_json(audit)
    bad=copy.deepcopy(audit);bad['identifier_group_counts']['barcodes']=1
    with pytest.raises(ValueError,match='private row-level JSON field'):publisher.inherited.validate_public_json(bad)


def test_final_guard_requires_all_297_actual_checkpoint_records(qualified):
    root,lock=qualified;complete_metadata(root,lock,checkpoint_count=296)
    with pytest.raises(ValueError,match='R7_COMPLETE_297_CHECKPOINT_REPLAY_REQUIRED'):publisher.final_guard(root,lock)
    complete_metadata(root,lock,checkpoint_count=297)
    assert publisher.final_guard(root,lock)['status']=='COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS'


def test_complete_stage_cannot_publish_protocol_only_or_296_checkpoint_fixture(qualified,monkeypatch):
    root,lock=qualified;denylist=synthetic_denylist(root,monkeypatch)
    complete_metadata(root,lock,checkpoint_count=296)
    with pytest.raises(ValueError,match='R7_COMPLETE_297_CHECKPOINT_REPLAY_REQUIRED'):publisher.publish(root,denylist,'complete')
    assert not (root/publisher.MANIFEST).exists()


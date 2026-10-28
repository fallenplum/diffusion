"""Synthetic metadata-only source report qualification; no real fits/arrays."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bgb_en_r6_report_source.py"
SPEC = importlib.util.spec_from_file_location("bgb_en_r6_report_source_test", SCRIPT)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)

CANARY = "PRIVATE_SPOT_IDENTIFIER_AND_RAW_VALUE_MUST_NEVER_APPEAR"
FAMILY_VALUES = {
    ("S0", "MATCHED"): (2., .1), ("S1", "MATCHED"): (1.8, .15),
    ("S2", "MATCHED"): (1.6, .2), ("S3", "MATCHED"): (1.4, .3),
    ("S4", "MATCHED"): (1., .5), ("S5", "WRONG_DONOR"): (1.5, .2),
    ("S4", "WITHIN_SECTION_TISSUE_DEPTH"): (1.3, .3), ("S4", "RANDOM_GATE"): (1.2, .35),
    ("S6", "MATCHED"): (.9, .55), ("S6", "WRONG_NEIGHBOR"): (1.2, .3),
    ("S7", "MATCHED"): (.8, .6), ("S7", "WRONG_SPOT_ALIGNMENT"): (1.1, .4),
}
SOURCE_CODE = ("scripts/bgb_en_r6_pretrain.py", "scripts/bgb_en_r6_schedule_source.py",
    "src/bgb_en/r6_source.py", "src/bgb_en/r6_source_model.py", "src/bgb_en/r6_omics.py", "src/bgb_en/external.py")


def put(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def metadata_fixture(tmp_path, *, full=False, target_constant=None):
    cells = []
    for arm, control in r.FAMILIES:
        phase = "SPATIAL" if arm == "S6" else "SPOT_ALIGNMENT" if arm == "S7" else "PRIMARY"
        cells.extend(dict(arm=arm, control=control, heldout_donor=donor, seed=seed, schedule_phase=phase)
                     for donor in range(6) for seed in (31, 47))
    cells.extend(dict(arm="S7", control=control, heldout_donor=-1, seed=31, schedule_phase="FULL_TRANSFER")
                 for control in ("MATCHED", "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH"))
    config = dict(schema="bgb_en_r6_source_pretrain_lock_v1", status="LOCKED_BEFORE_SOURCE_FITS",
        cells=cells, source_code_sha256={p: r.sha256(r.ROOT / p) for p in SOURCE_CODE},
        fixed_defaults=dict(epochs=80, checkpoint_epochs=[20, 40, 80]),
        source_cache_manifest_sha256="1" * 64, fresh_certificate_sha256="2" * 64)
    config_path = tmp_path / "source.lock.json"; put(config_path, config)
    eligibility = dict(schema="bgb_en_r6_source_target_eligibility_v1", status="LOCKED_MODEL_INDEPENDENT_SCORING_SUPPORT",
        source_cache_manifest_sha256=config["source_cache_manifest_sha256"], fresh_certificate_sha256=config["fresh_certificate_sha256"],
        groups=[dict(heldout_donor=donor, program=program, tissue_class=tissue, support=4,
            target_nonconstant=(donor, program, tissue) != target_constant,
            eligible=(donor, program, tissue) != target_constant)
            for donor in range(6) for program in r.PROGRAMS for tissue in range(3)])
    eligibility_path = tmp_path / "eligibility.json"; put(eligibility_path, eligibility)
    private = tmp_path / "private"; private.mkdir()
    if full:
        for cell in cells:
            write_unit(private, cell, config, r.sha256(config_path), target_constant=target_constant)
    return config, config_path, private, eligibility_path


def score(cell, *, target_constant=None, null_prediction_group=None, reverse=False):
    arm, control = cell["arm"], cell["control"]
    value, rho = FAMILY_VALUES.get((arm, control), (.8, .6))
    if reverse and (arm, control) == ("S4", "MATCHED"):
        value, rho = (2.5, -.4)
    value += .005 * max(cell["heldout_donor"], 0) + (.01 if cell["seed"] == 47 else 0)
    programs = []
    for k, program in enumerate(r.PROGRAMS):
        groups = []
        for tissue in range(3):
            key = (cell["heldout_donor"], program, tissue)
            groups.append(dict(tissue_class=tissue, support=4, Huber=value + .01 * k,
                Spearman=None if key in (target_constant, null_prediction_group) else rho))
        programs.append(dict(program=program, support=12, Huber=value + .01 * k,
            Spearman=rho, within_tissue=groups, log_library_Spearman=.2))
    return dict(spots=12, program_Huber=sum(p["Huber"] for p in programs) / 4,
        tissue_BA=.75, tissue_CE=.5, per_program=programs,
        true_labels_used_for_scoring_only=True, visual_program_forward_is_gene_and_tissue_truth_free=True)


def write_unit(private, cell, config, config_sha, *, target_constant=None, null_prediction_group=None, reverse=False):
    directory = private / "source_cells" / r.cell_name(cell)
    directory.mkdir(parents=True, exist_ok=True)
    metrics = dict(train=score(cell), heldout=score(cell, target_constant=target_constant,
        null_prediction_group=null_prediction_group, reverse=reverse))
    if cell["heldout_donor"] < 0:
        metrics["heldout"] = dict(status="NOT_APPLICABLE_FULL_SOURCE_NO_HELDOUT")
    epochs = [80] if cell["arm"] == "S0" else [20, 40, 80]
    for epoch in epochs:
        state = directory / f"state_epoch{epoch:03d}.private.pt"
        state.write_bytes((CANARY + str(epoch)).encode())
        raw = directory / f"outputs_epoch{epoch:03d}.private.npz"
        raw.write_bytes(CANARY.encode())
        snapshot = dict(status="COMPLETE_SOURCE_SNAPSHOT", cell=cell, epoch=epoch,
            optimizer_updates=0 if cell["arm"] == "S0" else epoch,
            config_sha256=config_sha, source_code_sha256=config["source_code_sha256"], metrics=metrics,
            artifact_sha256={state.name: r.sha256(state), raw.name: r.sha256(raw)})
        put(directory / f"SNAPSHOT_{epoch:03d}.json", snapshot)
    if cell["arm"] != "S0":
        curve = [dict(epoch=epoch, optimizer_updates=epoch, complete_source_updates_represented=1,
            negative_key_gradients_included=True, objective=1 / epoch, parameter_L2=.01,
            replay_max_abs_error=0., preclip_gradient_norm=.2,
            component_audit=dict(program_huber=1 / epoch, tissue_CE=.5, alignment=.1, neighborhood=.2))
            for epoch in range(1, 81)]
        put(directory / "TRAINING_CURVE.json", curve)
    put(directory / "FIT_CONTRACT.json", dict(private_path=CANARY, private_row=CANARY))
    receipt = dict(status="COMPLETE_R6_RANDOM_REFERENCE_NO_OPTIMIZER" if cell["arm"] == "S0" else "COMPLETE_R6_SOURCE_FIT",
        cell=cell, config_sha256=config_sha, source_code_sha256=config["source_code_sha256"], epochs=80,
        optimizer_updates=0 if cell["arm"] == "S0" else 80, actual_new_optimizer_fits=0 if cell["arm"] == "S0" else 1,
        elapsed_seconds=1., metrics=metrics, full_source_no_heldout=cell["heldout_donor"] < 0,
        parameter_budget=dict(registered=1234, requires_grad=0 if cell["arm"] == "S0" else 1111),
        private_path=CANARY, artifact_sha256={file.name: r.sha256(file) for file in directory.iterdir()
            if file.name not in ("COMPLETE.json", "REPLAY.json")})
    put(directory / "COMPLETE.json", receipt)
    replay = dict(status="PASS_COMPLETE_R6_SOURCE_CHECKPOINT_REPLAY", cell=cell, config_sha256=config_sha,
        source_code_sha256=config["source_code_sha256"], completion_sha256=r.sha256(directory / "COMPLETE.json"),
        optimizer_updates=0, new_optimizer_fits=0, model_frozen=True,
        image_only_heldout_program_prediction=True, original_fit_artifacts_modified=False, metrics=metrics,
        forward_max_abs_errors={split: dict(program_predictions=0., tissue_logits=0.)
            for split in (["train"] if cell["heldout_donor"] < 0 else ["train", "heldout"])})
    put(directory / "REPLAY.json", replay)
    return directory


def test_missing_matrix_has_no_complete_mean_or_fabricated_zero_cost(tmp_path):
    _, config, private, eligibility = metadata_fixture(tmp_path)
    report, rows = r.build_report(config, private, target_eligibility=eligibility)
    assert report["status"] == "PARTIAL_SOURCE_MATRIX" and report["counts"] == {"MISSING": 147}
    assert report["decision"]["status"] == "INCONCLUSIVE" and report["unknown_update_cost_units"] == 147
    assert all(f["program_Huber"] is None for f in report["complete_family_summaries"])
    assert not rows


def test_complete_matrix_is_donor_then_seed_equal_weighted_with_three_full_models_separate(tmp_path):
    _, config, private, eligibility = metadata_fixture(tmp_path, full=True)
    report, rows = r.build_report(config, private, target_eligibility=eligibility)
    assert report["status"] == "COMPLETE_147_SOURCE_UNITS_AND_REPLAY"
    assert report["confirmed_formal_optimizer_fits"] == 135
    assert report["confirmed_formal_optimizer_updates"] == 10800
    assert report["unknown_update_cost_units"] == 0 and len(rows) == 144 * 4 * 3
    assert report["decision"]["status"] == "SUPPORTED" and len(report["full_source_transfer_units"]) == 3
    assert report["decision"]["all_locked_source_fits_still_required"]
    family = next(f for f in report["complete_family_summaries"] if (f["arm"], f["control"]) == ("S4", "MATCHED"))
    assert family["program_Huber"] == pytest.approx(1 + .015 + .0125 + .005)
    assert family["within_tissue_Spearman"] == .5
    for key in r.CORE_COMPARISONS:
        assert report["comparisons"][key]["complete_pairs"] == 12
        assert len(report["comparisons"][key]["pairs"]) == 12
    assert all(row["heldout_donor"] >= 0 for row in rows)
    assert all(not row["training_budget"]["epoch_selected_using_heldout"] for row in report["ledger"] if row["arm"] != "S0")


def test_target_constant_groups_are_excluded_identically_without_model_dependent_dropping(tmp_path):
    fixed_constant = (0, r.PROGRAMS[0], 0)
    _, config, private, eligibility = metadata_fixture(tmp_path, full=True, target_constant=fixed_constant)
    report, rows = r.build_report(config, private, target_eligibility=eligibility)
    assert report["decision"]["status"] == "SUPPORTED"
    excluded = [row for row in rows if (row["heldout_donor"], row["program"], row["tissue_class"]) == fixed_constant]
    assert excluded and all(row["eligibility_reason"] == "TARGET_CONSTANT" for row in excluded)
    assert all(row["eligible_for_within_tissue_rho"] is False for row in excluded)


def test_undefined_prediction_rho_in_eligible_group_makes_required_pair_inconclusive(tmp_path):
    config, config_path, private, eligibility = metadata_fixture(tmp_path, full=True)
    cell = next(c for c in config["cells"] if r.identity(c) == ("S4", "MATCHED", 0, 31))
    write_unit(private, cell, config, r.sha256(config_path), null_prediction_group=(0, r.PROGRAMS[0], 0))
    report, _ = r.build_report(config_path, private, target_eligibility=eligibility)
    assert report["status"] == "COMPLETE_147_SOURCE_UNITS_AND_REPLAY"
    assert report["decision"]["status"] == "INCONCLUSIVE"
    comparison = report["comparisons"][r.CORE_COMPARISONS[0]]
    assert comparison["complete_pairs"] == 12 and comparison["within_tissue_rho_true_minus_negative"] is None
    assert comparison["Huber_negative_minus_true"] > 0


def test_one_missing_cell_never_produces_a_partial_overall_or_pair_mean(tmp_path):
    config, config_path, private, eligibility = metadata_fixture(tmp_path, full=True)
    cell = next(c for c in config["cells"] if r.identity(c) == ("S5", "WRONG_DONOR", 5, 47))
    (private / "source_cells" / r.cell_name(cell) / "COMPLETE.json").unlink()
    report, _ = r.build_report(config_path, private, target_eligibility=eligibility)
    family = next(f for f in report["complete_family_summaries"] if f["arm"] == "S5")
    assert family["complete_replayed_units"] == 11 and family["program_Huber"] is None
    comparison = report["comparisons"][r.CORE_COMPARISONS[0]]
    assert comparison["complete_pairs"] == 11 and comparison["Huber_negative_minus_true"] is None
    assert report["decision"]["status"] == "INCONCLUSIVE"


def test_complete_negative_directions_are_not_supported_without_stopping_other_families(tmp_path):
    config, config_path, private, eligibility = metadata_fixture(tmp_path, full=True)
    for cell in config["cells"]:
        if (cell["arm"], cell["control"]) == ("S4", "MATCHED"):
            write_unit(private, cell, config, r.sha256(config_path), reverse=True)
    report, _ = r.build_report(config_path, private, target_eligibility=eligibility)
    assert report["decision"]["status"] == "NOT_SUPPORTED"
    assert not report["decision"]["source_score_dependent_execution_gate"]
    assert all(f["complete_replayed_units"] == 12 for f in report["complete_family_summaries"])


def test_binary_artifact_tampering_or_wrong_seed_replay_is_rejected_without_loading_binary(tmp_path):
    config, config_path, private, eligibility = metadata_fixture(tmp_path)
    cell = config["cells"][0]
    directory = write_unit(private, cell, config, r.sha256(config_path))
    (directory / "state_epoch080.private.pt").write_bytes(b"changed")
    report, _ = r.build_report(config_path, private, target_eligibility=eligibility)
    assert report["counts"]["INVALID"] == 1
    assert report["ledger"][0]["validation_failure_code"] == "SOURCE_ARTIFACT_SHA_CHANGED"
    directory = write_unit(private, cell, config, r.sha256(config_path))
    replay = r._load(directory / "REPLAY.json"); replay["cell"] = dict(cell, seed=47)
    put(directory / "REPLAY.json", replay)
    report, _ = r.build_report(config_path, private, target_eligibility=eligibility)
    assert report["ledger"][0]["status"] == "INVALID"
    assert not report["ledger"][0]["independent_replay_passed"]


def test_fixed_target_eligibility_provenance_coverage_and_group_support_are_verified(tmp_path):
    config, config_path, private, eligibility = metadata_fixture(tmp_path)
    original = r._load(eligibility)
    changed = deepcopy(original); changed["groups"].pop(); put(eligibility, changed)
    with pytest.raises(ValueError, match="72_GROUPS"):
        r.build_report(config_path, private, target_eligibility=eligibility)
    changed = deepcopy(original); changed["source_cache_manifest_sha256"] = "9" * 64; put(eligibility, changed)
    with pytest.raises(ValueError, match="PROVENANCE_CHANGED"):
        r.build_report(config_path, private, target_eligibility=eligibility)
    put(eligibility, original)
    cell = config["cells"][0]
    directory = write_unit(private, cell, config, r.sha256(config_path))
    receipt = r._load(directory / "COMPLETE.json")
    receipt["metrics"]["heldout"]["per_program"][0]["within_tissue"][0]["support"] = 3
    receipt["metrics"]["heldout"]["per_program"][0]["support"] = 11
    put(directory / "COMPLETE.json", receipt)
    report, _ = r.build_report(config_path, private, target_eligibility=eligibility)
    assert report["ledger"][0]["validation_failure_code"] == "SOURCE_TARGET_ELIGIBILITY_SUPPORT_CHANGED"


def test_public_exports_are_allowlisted_and_qualification_updates_are_not_formal_fits(tmp_path):
    _, config, private, eligibility = metadata_fixture(tmp_path, full=True)
    qualification = tmp_path / "qualification.private.json"
    put(qualification, dict(optimizer_updates=2, actual_new_optimizer_fits=1, private_error=CANARY, private_rows=CANARY))
    report, rows = r.build_report(config, private, target_eligibility=eligibility, qualification_receipt=qualification)
    assert report["confirmed_formal_optimizer_updates"] == 10800
    assert report["qualification_cost_separate_from_formal_matrix"]["optimizer_updates"] == 2
    output = tmp_path / "public"
    output.mkdir(); put(output / "SOURCE_AUDIT.json", dict(other_stage=dict(preserved=True)))
    r.write_report(report, rows, output)
    assert len(list(output.iterdir())) == 9
    for path in output.iterdir():
        text = path.read_text(encoding="utf-8-sig")
        assert CANARY not in text
        assert str(private) not in text and str(tmp_path) not in text
    audit = r._load(output / "SOURCE_AUDIT.json")
    assert audit["other_stage"]["preserved"]
    assert audit["r6_source_supervision"]["target_eligibility_sha256"] == r.sha256(eligibility)
    with pytest.raises(ValueError, match="HISTORICAL_REPORT_NAMESPACES"):
        r.write_report(report, rows, r.ROOT / "reports/bgb_en")


def test_error_ledger_does_not_publish_private_tracebacks_or_fake_zero_updates(tmp_path):
    config, config_path, private, eligibility = metadata_fixture(tmp_path)
    cell = config["cells"][12]
    directory = private / "source_cells" / r.cell_name(cell); directory.mkdir(parents=True)
    put(directory / "ERROR.json", dict(status="ERROR_SOURCE_FIT", completed_optimizer_updates=7,
        actual_new_optimizer_fits=1, error=CANARY, traceback=CANARY, cell=cell,
        config_sha256=r.sha256(config_path), source_code_sha256=config["source_code_sha256"]))
    report, _ = r.build_report(config_path, private, target_eligibility=eligibility)
    assert report["confirmed_formal_optimizer_updates"] == 7 and report["unknown_update_cost_units"] == 146
    assert CANARY not in json.dumps(report)
    assert report["counts"]["ERROR"] == 1


def test_metadata_report_has_no_tensor_array_or_raw_dataset_imports():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "import torch" not in text and "import numpy" not in text and "import h5py" not in text
    assert "torch.load" not in text and "np.load" not in text


def test_foreign_error_receipt_has_unknown_cost_and_failed_forward_error_audit_is_rejected(tmp_path):
    config, config_path, private, eligibility = metadata_fixture(tmp_path)
    cell = config["cells"][12]
    directory = private / "source_cells" / r.cell_name(cell); directory.mkdir(parents=True)
    put(directory / "ERROR.json", dict(status="ERROR_SOURCE_FIT", cell=dict(cell, seed=999),
        config_sha256=r.sha256(config_path), source_code_sha256=config["source_code_sha256"],
        completed_optimizer_updates=7, actual_new_optimizer_fits=1, error=CANARY))
    report, _ = r.build_report(config_path, private, target_eligibility=eligibility)
    assert report["counts"]["INVALID_ERROR_RECEIPT"] == 1
    assert report["confirmed_formal_optimizer_updates"] == 0 and report["unknown_update_cost_units"] == 147
    first = config["cells"][0]
    directory = write_unit(private, first, config, r.sha256(config_path))
    replay = r._load(directory / "REPLAY.json")
    replay["forward_max_abs_errors"]["heldout"]["program_predictions"] = 1.
    put(directory / "REPLAY.json", replay)
    report, _ = r.build_report(config_path, private, target_eligibility=eligibility)
    assert report["ledger"][0]["validation_failure_code"] == "INDEPENDENT_SOURCE_REPLAY_FORWARD_ERROR_AUDIT_MISSING_OR_FAILED"


def test_real_compute_qualification_fields_are_separate_from_135_formal_fits(tmp_path):
    _, config, private, eligibility = metadata_fixture(tmp_path)
    qualification = tmp_path / "compute_qualification.json"
    put(qualification, dict(qualification_updates=2, formal_source_fits=0,
        separate_qualification_optimizer_instances=1, saved_qualification_weights=False))
    report, _ = r.build_report(config, private, target_eligibility=eligibility, qualification_receipt=qualification)
    assert report["confirmed_formal_optimizer_fits"] == report["confirmed_formal_optimizer_updates"] == 0
    separate = report["qualification_cost_separate_from_formal_matrix"]
    assert separate["optimizer_updates"] == 2 and separate["optimizer_instances"] == 1
    assert separate["formal_source_fits"] == 0

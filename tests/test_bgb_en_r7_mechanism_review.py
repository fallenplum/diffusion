"""Independent anonymous R7 mechanism fixtures; no ML/data/model imports.

All COMPLETE/PASS strings and numerical cells here are fabricated metadata.
They never represent a formal source/target fit or patient-level evidence.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bgb_en_r7_mechanism_review.py"
ARMS = ("ADD", "RAW", "NORM", "INDEP", "SLOT", "GATE1", "GATE2", "SLOT_WRONG_RNA",
        "GATE1_WRONG_RNA", "GATE1_WRONG_PATH", "GATE2_WRONG_RNA")
SEEDS, PARTS, EPOCHS = (31, 47, 83), (0, 1, 2), (20, 40, 60, 80)
PATHS = ("cross_delta", "risk", "path_cross_delta", "gene_cross_delta", "total_delta_L2", "cross_delta_L2", "parameter_L2")
PAIRS = (("NORM", "RAW"), ("INDEP", "NORM"), ("SLOT", "INDEP"), ("GATE1", "SLOT"),
         ("GATE2", "GATE1"), ("GATE1", "ADD"), ("GATE1", "B1"),
         ("GATE1", "GATE1_WRONG_RNA"), ("GATE1", "GATE1_WRONG_PATH"),
         ("SLOT", "SLOT_WRONG_RNA"), ("GATE2", "GATE2_WRONG_RNA"))
CONTRACT = "c" * 64
CANARY = "SYNTHETIC_IDENTIFIER_MUST_NOT_BE_EXPORTED"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def put(path, value, *, allow_nan=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=allow_nan), encoding="utf-8")
    return digest(path)


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_review():
    assert SCRIPT.is_file(), "Owner must finish the new metadata-only script before tests run"
    spec = importlib.util.spec_from_file_location("r7_mechanism_independent_tested", SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def score(arm, part, seed):
    base = dict(B0=(.68, .66), B1=(.70, .68), ADD=(.72, .70), RAW=(.73, .705),
        NORM=(.735, .71), INDEP=(.75, .72), SLOT=(.76, .73), GATE1=(.74, .71), GATE2=(.745, .715),
        SLOT_WRONG_RNA=(.755, .725), GATE1_WRONG_RNA=(.75, .72), GATE1_WRONG_PATH=(.735, .705),
        GATE2_WRONG_RNA=(.742, .712))
    h, u = base[arm]
    shift = (-.02, .01, .03)[part] + (0. if arm in ("B0", "B1") else {31: -.001, 47: 0., 83: .002}[seed])
    return metric(h + shift, u + shift, part)


def metric(h, u, part):
    hd, ud = (10., 30., 90.)[part], (5., 25., 120.)[part]
    return dict(Harrell=h, Uno60=u, Harrell_num=h * hd, Harrell_den=hd, Uno_num=u * ud, Uno_den=ud)


def with_credit(candidate, reference, part):
    result = deepcopy(candidate)
    # Nonzero damage makes it impossible to invent these from delta alone.
    for name, key in (("", "Harrell"), ("Uno_", "Uno60")):
        damage = .20 + part * .01
        result[name + "damage"] = damage
        result[name + "repair"] = damage + candidate[key] - reference[key]
    return result


def summary(arm):
    seeds = []
    for seed in SEEDS:
        rows = [score(arm, part, seed) for part in PARTS]
        nums = {key: math.fsum(row[key] for row in rows) for key in ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")}
        seeds.append(dict(seed=seed, Harrell=nums["Harrell_num"] / nums["Harrell_den"],
                          Uno60=nums["Uno_num"] / nums["Uno_den"], **nums))
    return dict(arm=arm, expected_cells=9, complete_cells=9, status="COMPLETE_NINE_CELLS", per_seed=seeds,
        Harrell=statistics.mean(row["Harrell"] for row in seeds), Uno60=statistics.mean(row["Uno60"] for row in seeds),
        aggregation="within_partition_components_sum_per_seed_then_equal_three_seed_mean", cross_partition_risk_ranking=False)


def stats(value, n=272):
    return dict(n=n, mean=value, sd=value, min=0., q10=0., median=value, q90=value, max=2 * value)


def branch(arm, part, epoch):
    n = (272, 273, 273)[part]
    ratio = None if arm in ("ADD", "RAW", "NORM", "INDEP") else .6
    slots = {} if arm in ("ADD", "RAW", "NORM", "INDEP") else {
        key: dict(per_program=[dict(program=k, **stats(.01 * (k + 1), n)) for k in range(4)],
            sum=stats(.01, n), sum_absolute=stats(.02, n), nonzero_absolute_sum_patients=n,
            absolute_sum_over_sum_absolute_mean=ratio, cancellation_is_descriptive_not_performance=True)
        for key in ("slot_path_contributions", "slot_gene_contributions")}
    return dict(SD_total_delta_over_offset=.03, SD_cross_delta_over_offset=0. if arm == "ADD" else .01,
        quantities={"delta": stats(.03, n), "main_delta": stats(.025, n), "cross_delta": stats(.01, n),
            "path_cross_delta": stats(.008, n), "gene_cross_delta": stats(.002, n)},
        linear_readout_has_no_tanh_saturation_bound=True, slot_contributions=slots)


def qsummary(value):
    return dict(valid_items=32, patients_with_valid_items=8, mean=value, sd=0.)


def query_probe(arm, epoch):
    tags = ("matched_query", "wrong_query") + (() if arm in ("RAW", "ADD") else ("wrong_path",))
    controls = {}
    for tag in tags:
        interfaces = {key: qsummary(value) for key, value in dict(raw_G0=.25, RNA_context=1., base_query=1.,
            condition_query=1.02, query_delta=.02, P0=1., P_update=.01, reverse_query=1.,
            G_reverse_base=.3, G_update=.002, null=.1).items()}
        interfaces["raw_patch_mean_L2"] = dict(patients=8, mean=6., sd=.1)
        rounds = []
        for round_ in range(2 if arm.startswith("GATE2") else 0 if arm in ("RAW", "ADD") else 1):
            fields = {key: qsummary(1.) for key in ("P_before", "P_after", "G_before", "G_after", "RNA_route", "PATH_route")}
            fields.update(raw_delta_P=qsummary(.01), raw_delta_G=qsummary(.002), path_gate=qsummary(1.), gene_gate=qsummary(1.))
            rounds.append(dict(round=round_ + 1, **fields))
        controls[tag] = dict(interfaces=interfaces, rounds=rounds, query_source_valid_programs=32, own_valid_programs=32)
    return dict(scope="fixed_first_eight_current_train_rows_no_validation_no_outcome_selection", patients=8, controls=controls,
        matched_vs_controls={tag: dict(patch_plus_null_TV=qsummary(.004 * epoch / 80), risk_difference_SD=.02,
            own_values_main_offset_exact=True, frozen_not_retrained=True, RNA_query_wrong_is_not_reverse_path_wrong=True) for tag in tags[1:]},
        diagnostic_forward_calls=len(tags), new_backward_calls=0, new_optimizer_updates=0,
        fixed_six_RMS={name: .2 for name in ("context_rms", "patch_rms", "query_rms", "gene_key_rms", "evidence_rms", "reverse_query_rms")},
        fixed_initial_full_train_calibration=dict(train_only=True, outcomes_accessed=False, new_optimizer_updates=0),
        no_patient_identifiers_or_raw_attention_exported=True)


def diagnostic(arm, part, epoch):
    n = (272, 273, 273)[part]
    magnitudes = dict(cross_delta=4., risk=9., path_cross_delta=1., gene_cross_delta=.25,
                      total_delta_L2=.04, cross_delta_L2=.01, parameter_L2=.09)
    groups = {name: {"core.query_condition": dict(tensors=1, numel=64, nonzero=1, L2_squared=value),
                     "core.path_main": dict(tensors=1, numel=64, nonzero=0, L2_squared=0.)} for name, value in magnitudes.items()}
    return dict(epoch=epoch, optimizer_updates_completed=epoch, train=branch(arm, part, epoch),
        validation="NOT_EVALUATED_AT_DIAGNOSTIC_ONLY_EPOCH" if epoch == 60 else branch(arm, part, epoch),
        gradient_probe=dict(scope="complete_train_riskset_Cox_vector_accumulated_all_microbatches", patients=n,
            microbatches=math.ceil(n / 8), groups_by_path=groups, cox_loss=.9, parameter_regularizer=.01,
            parameter_L2_uses_actual_slot_one_quarter_geometry=True, risk_penalties_excluded_from_pure_Cox_paths=True,
            pure_Cox_paths=list(PATHS[:4]), residual_penalty_paths=list(PATHS[4:6]), new_optimizer_updates=0,
            norm_differences_are_not_vector_gradient_differences=True),
        query_probe=query_probe(arm, epoch), parameter_budget=dict(registered=150000, requires_grad=35652),
        validation_ranking_used_for_selection=False)


def frozen(arm, part, seed):
    full, reference = score(arm, part, seed), score("B1", part, seed)
    result = {}
    tags = ("matched_query", "wrong_query", "mean_query", "interaction_off", "main_off") + (() if arm in ("RAW", "ADD") else ("wrong_path",))
    for tag in tags:
        if tag == "matched_query" or tag == "interaction_off" and arm == "ADD":
            control = deepcopy(full)
        elif tag == "interaction_off":
            # FULL-OFF is negative while OFF-ADD is positive for GATE1.
            control = metric(full["Harrell"] + .03, full["Uno60"] + .02, part)
        else:
            control = metric(full["Harrell"] - .004, full["Uno60"] - .003, part)
        result[tag] = dict(metrics_vs_base=with_credit(control, reference, part),
            relative_to_trained=with_credit(full, control, part), risk_SD_ratio=.01,
            frozen_not_retrained=True, whole_modality_replaced=False,
            intervention_scope="path_query_only" if tag == "wrong_path" else tag)
    return result


def make_public(root):
    root = Path(root); root.mkdir(parents=True)
    replay = dict(status="PASS_COMPLETE_R7_ALL_CHECKPOINT_REPLAY", contract_sha256=CONTRACT,
        qualification_sha256="e" * 64, expected_units=99, available_units=99, passed_units=99, failed_units=0,
        expected_checkpoints=297, passed_checkpoints=297, missing_or_unreplayed_units=[], new_optimizer_fits=0,
        new_preprocessing_fits=0, primary_checkpoint_epoch=80, diagnostic_checkpoints_not_selected=True,
        patient_arrays_exported=False, units=[], zero_fit_baseline_reference_replays=[], paired_retrained_controls=[])
    parts = []
    for arm in ARMS:
        for part in PARTS:
            rows = []
            for seed in SEEDS:
                name = f"{arm}_p{part}_seed{seed}"
                replay["units"].append(dict(unit=name, arm=arm, partition=part, seed=seed, status="PASS",
                    contract_sha256=CONTRACT, actual_checkpoint_count=3, changed_pair_credits=0, new_optimizer_updates=0,
                    completion_sha256="b" * 64, artifact_sha256="b" * 64, qualification_sha256="e" * 64,
                    upstream_replay_audit_sha256="a" * 64, replay_script_sha256="a" * 64, maximum_risk_error=0.,
                    all_checkpoints_actual_forward_replayed=True, actual_state_dict_forward_replayed=True,
                    source_frozen_weight_hash_replayed=True, saved_query_mapping_recomputed=True,
                    checkpoints=[dict(epoch=e, status="PASS", actual_state_dict_forward_replayed=True,
                        maximum_risk_error=0., changed_pair_credits=0, new_optimizer_updates=0, checkpoint_sha256="d" * 64) for e in (20, 40, 80)]))
                rows.append(dict(arm=arm, partition=part, seed=seed,
                    checkpoint_diagnostics={str(e): diagnostic(arm, part, e) for e in EPOCHS},
                    final_risk_diagnostics=dict(train=branch(arm, part, 80), validation=branch(arm, part, 80)),
                    frozen_controls=frozen(arm, part, seed)))
            filename = f"R7_DIAGNOSTICS_{arm}_p{part}.json"; path = root / filename
            put(path, dict(schema="bgb_en_r7_diagnostic_shard_v1", rows=rows))
            parts.append(dict(file=filename, sha256=digest(path), bytes=path.stat().st_size, rows=3))
    for arm in ("B0", "B1"):
        for part in PARTS:
            for seed in SEEDS:
                replay["zero_fit_baseline_reference_replays"].append(dict(arm=arm, partition=part, seed=seed, status="PASS",
                    independent_fixed_anchor_metrics_recomputed=True, new_optimizer_fits=0, receipt_sha256="f" * 64))
    for candidate, reference in PAIRS:
        for part in PARTS:
            for seed in SEEDS:
                replay["paired_retrained_controls"].append(dict(candidate=candidate, reference=reference, partition=part, seed=seed,
                    candidate_vs_reference=with_credit(score(candidate, part, seed), score(reference, part, seed), part),
                    risk_SD_ratio=.02, direct_risk_pair_credits=True, bootstrap_or_CI_computed=False, primary_checkpoint_epoch=80))
    replay_path = root / "R7_CHECKPOINT_REPLAY_AUDIT.json"; put(replay_path, replay)
    report = dict(schema="bgb_en_r7_complete_report_v1", status="COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS",
        contract_sha256=CONTRACT, config_sha256="a" * 64, qualification_sha256="e" * 64, replay_sha256=digest(replay_path),
        primary_DSS={arm: summary(arm) for arm in ("B0", "B1", *ARMS)},
        secondary_PFI_frozen_DSS_risk={arm: summary(arm) for arm in ("B0", "B1", *ARMS)},
        budget=dict(formal_neural_fits=99, formal_optimizer_updates=7920, source_refits=0, teacher_or_beta_refits=0,
                    unsupervised_train_RMS_calibrations=9, zero_fit_baseline_reference_rows=18, diagnostics_optimizer_updates=0),
        diagnostic_index=dict(parts=parts, logical_cells=99, epochs_per_cell=list(EPOCHS)),
        paired_comparisons=[dict(candidate=candidate, reference=reference, question="synthetic_prelocked_comparison")
                            for candidate, reference in PAIRS],
        no_cross_partition_risk_ranking=True, seed_means_are_not_independent_cohorts=True,
        decision=dict(primary_candidate="GATE1", outer_results="NOT_RUN", CI_or_pvalues_computed=False, PFI_used_for_selection=False))
    report["synthetic_unselected_field"] = CANARY
    put(root / "R7_RESULTS.json", report)
    fold_rows = [dict(arm=arm, partition=part, seed=seed, **score(arm, part, seed))
                 for arm in ("B0", "B1", *ARMS) for part in PARTS for seed in SEEDS]
    write_csv(root / "R7_FOLD_METRICS.csv", fold_rows)
    return root


@pytest.fixture(scope="module")
def public_template(tmp_path_factory):
    return make_public(tmp_path_factory.mktemp("r7_mechanism_template") / "public")


@pytest.fixture
def public(public_template, tmp_path):
    target = tmp_path / "anonymous"; shutil.copytree(public_template, target); return target


def mutate_shard(public, change, *, rebind=True, name="R7_DIAGNOSTICS_GATE1_p0.json", allow_nan=False):
    path = public / name; value = read(path); change(value); put(path, value, allow_nan=allow_nan)
    if rebind:
        report = read(public / "R7_RESULTS.json"); entry = next(row for row in report["diagnostic_index"]["parts"] if row["file"] == name)
        entry.update(sha256=digest(path), bytes=path.stat().st_size); put(public / "R7_RESULTS.json", report)


def mutate_replay(public, change):
    path = public / "R7_CHECKPOINT_REPLAY_AUDIT.json"; value = read(path); change(value); put(path, value)
    report = read(public / "R7_RESULTS.json"); report["replay_sha256"] = digest(path); put(public / "R7_RESULTS.json", report)


def test_complete_scope_arithmetic_decomposition_and_actual_CLI(public, tmp_path):
    reviewer = load_review(); output = tmp_path / "review"
    result = reviewer.build(public, output)
    assert result["schema"] == "bgb_en_r7_mechanism_review_v1"
    assert result["status"] == "COMPLETE_METADATA_ONLY_99_CELLS_396_DIAGNOSTICS"
    assert result["model_cells"] == 99 and result["diagnostic_cell_epochs"] == 396
    assert result["new_model_forwards"] == result["new_backward_calls"] == result["new_optimizer_fits"] == 0
    assert len(result["arm_epoch_summaries"]) == 44 and all(row["cells"] == 9 for row in result["arm_epoch_summaries"])
    assert len(result["cell_epoch_index"]) == 396
    decomposition = result["main_output_decomposition"]
    assert decomposition["identity_verified"] is True
    assert len(decomposition["per_cell"]) == 99 and len(decomposition["per_seed"]) == 33 and len(decomposition["per_arm"]) == 11
    gate = next(row for row in decomposition["per_arm"] if row["arm"] == "GATE1")
    assert gate["Harrell"]["Cfull"] == pytest.approx(summary("GATE1")["Harrell"])
    for metric_name, full_add, full_off, off_add in (("Harrell", .02, -.03, .05), ("Uno60", .01, -.02, .03)):
        values = gate[metric_name]
        assert values["delta_full_add"] == pytest.approx(full_add)
        assert values["delta_full_off"] == pytest.approx(full_off)
        assert values["delta_off_add"] == pytest.approx(off_add)
        assert values["delta_full_add"] == pytest.approx(values["delta_full_off"] + values["delta_off_add"])
    assert all(row["off_add_direct_pair_credit"] is None for row in decomposition["per_cell"])
    hd_weighted_damage = math.fsum(damage * weight for damage, weight in zip((.20, .21, .22), (10, 30, 90))) / 130
    assert gate["direct_full_off_credit"]["damage"] == pytest.approx(hd_weighted_damage)
    assert gate["direct_full_off_credit"]["repair"] == pytest.approx(hd_weighted_damage - .03)
    assert len(result["paired_retrained_credit_rows"]) == 99
    assert result["scope_definitions"] and "FT_GRAD" in result["scope_definitions"]
    serialized = json.dumps(result, allow_nan=False)
    assert CANARY not in serialized and str(tmp_path) not in serialized
    # Replaying exactly the original credit must preserve damage as well as delta.
    credit = next(row for row in result["paired_retrained_credit_rows"] if row["candidate"] == "GATE1"
                  and row["reference"] == "GATE1_WRONG_RNA" and row["partition"] == 0 and row["seed"] == 31)
    assert credit["source"] == "actual_replay_direct_risk_pair_credits"
    assert credit["repair"] == pytest.approx(.19) and credit["damage"] == pytest.approx(.20)
    off_credit = next(row for row in result["frozen_credit_rows"] if row["arm"] == "GATE1"
                      and row["control"] == "interaction_off" and row["partition"] == 0 and row["seed"] == 31)
    assert off_credit["source"] == "saved_frozen_direct_risk_credits_authenticated_by_actual_replay"
    assert off_credit["repair"] - off_credit["damage"] == pytest.approx(-.03)
    epoch = next(row for row in result["arm_epoch_summaries"] if row["arm"] == "GATE1" and row["epoch"] == 20)
    grad = next(row for row in epoch["measurements"] if row.get("scope") == "FT_GRAD" and row.get("path") == "cross_delta"
                and row.get("group") == "core.query_condition")
    assert grad["mean"] == pytest.approx(2.) and grad["available_cells"] == result["expected_measurement_cells"] == epoch["cells"] == 9
    assert all(row["query_probe_patients"] == 8 and row["full_train_gradient_patients"] > 8 for row in result["cell_epoch_index"])
    cli_output = tmp_path / "cli_review"
    done = subprocess.run([sys.executable, str(SCRIPT), "--reports", str(public), "--output", str(cli_output)],
        text=True, capture_output=True, cwd=ROOT, timeout=60)
    assert done.returncode == 0, done.stderr
    assert list(cli_output.glob("*.json")) and list(cli_output.glob("*.md")) and list(cli_output.glob("*.csv"))


@pytest.mark.parametrize("tamper", ["shard_sha", "path_escape", "replay_hash"])
def test_sha_and_path_provenance_fail_closed(public, tmp_path, tamper):
    if tamper == "shard_sha":
        mutate_shard(public, lambda shard: shard["rows"][0]["checkpoint_diagnostics"]["20"].update(epoch=21), rebind=False)
    else:
        report = read(public / "R7_RESULTS.json")
        if tamper == "path_escape":
            original = report["diagnostic_index"]["parts"][0]
            outside = public.parent / "outside.json"
            outside.write_bytes((public / original["file"]).read_bytes())
            report["diagnostic_index"]["parts"][0]["file"] = "../outside.json"
            report["diagnostic_index"]["parts"][0].update(sha256=digest(outside), bytes=outside.stat().st_size)
        else:
            report["replay_sha256"] = "0" * 64
        put(public / "R7_RESULTS.json", report)
    with pytest.raises((ValueError, FileNotFoundError)):
        load_review().build(public, tmp_path / "rejected")


@pytest.mark.parametrize("tamper", ["missing_cell", "missing_epoch", "missing_replay", "epoch60_validation"])
def test_partial_or_wrong_scope_never_claims_complete(public, tmp_path, tamper):
    if tamper == "missing_cell":
        mutate_shard(public, lambda shard: shard["rows"].pop())
    elif tamper == "missing_epoch":
        mutate_shard(public, lambda shard: shard["rows"][0]["checkpoint_diagnostics"].pop("60"))
    elif tamper == "missing_replay":
        mutate_replay(public, lambda audit: audit.update(passed_checkpoints=296))
    else:
        mutate_shard(public, lambda shard: shard["rows"][0]["checkpoint_diagnostics"]["60"].update(validation={"SD": .1}))
    with pytest.raises(ValueError):
        load_review().build(public, tmp_path / "rejected")


@pytest.mark.parametrize("tamper", ["nonfinite_gradient", "fake_pair_credit", "unfrozen_off_credit", "fake_fulltrain_scope"])
def test_nonfinite_gradient_and_credit_or_scope_labels_are_not_silently_trusted(public, tmp_path, tamper):
    if tamper == "nonfinite_gradient":
        mutate_shard(public, lambda shard: shard["rows"][0]["checkpoint_diagnostics"]["20"]["gradient_probe"]
            ["groups_by_path"]["risk"]["core.query_condition"].update(L2_squared=float("nan")), allow_nan=True)
    elif tamper == "fake_pair_credit":
        mutate_replay(public, lambda audit: audit["paired_retrained_controls"][0].update(direct_risk_pair_credits=False))
    elif tamper == "unfrozen_off_credit":
        mutate_shard(public, lambda shard: shard["rows"][0]["frozen_controls"]["interaction_off"].update(frozen_not_retrained=False))
    else:
        mutate_shard(public, lambda shard: shard["rows"][0]["checkpoint_diagnostics"]["20"]["gradient_probe"].update(patients=8))
    with pytest.raises(ValueError):
        load_review().build(public, tmp_path / "rejected")


def test_zero_slot_sum_keeps_source_None_and_no_model_library_imports(public, tmp_path):
    def no_slot_signal(shard):
        for row in shard["rows"]:
            for epoch in EPOCHS:
                train = row["checkpoint_diagnostics"][str(epoch)]["train"]
                slots = train["slot_contributions"]
                for value in slots.values():
                    value["absolute_sum_over_sum_absolute_mean"] = None
                    value["nonzero_absolute_sum_patients"] = 0
                    value["sum"] = value["sum_absolute"] = stats(0., (272, 273, 273)[row["partition"]])
                    value["per_program"] = [dict(program=k, **stats(0., (272, 273, 273)[row["partition"]])) for k in range(4)]
                train["SD_cross_delta_over_offset"] = 0.
                train["SD_total_delta_over_offset"] = .025
                for name in ("cross_delta", "path_cross_delta", "gene_cross_delta"):
                    train["quantities"][name] = stats(0., (272, 273, 273)[row["partition"]])
                train["quantities"]["delta"] = stats(.025, (272, 273, 273)[row["partition"]])
    mutate_shard(public, no_slot_signal)
    result = load_review().build(public, tmp_path / "review")
    assert result["status"] == "COMPLETE_METADATA_ONLY_99_CELLS_396_DIAGNOSTICS"
    epoch = next(row for row in result["arm_epoch_summaries"] if row["arm"] == "GATE1" and row["epoch"] == 20)
    retained = next(row for row in epoch["measurements"] if row.get("scope") == "FT_RISK"
        and row.get("metric") == "slot_path_contributions.retained_absolute_fraction")
    assert retained["available_cells"] == 6 and result["expected_measurement_cells"] == epoch["cells"] == 9
    assert retained["mean"] == pytest.approx(.6)
    long_rows = read_csv(tmp_path / "review/R7_MECHANISM_REVIEW_GATE1.csv")
    source_none = [row for row in long_rows if row["scope"] == "FT_RISK" and row["partition"] == "0"
        and row["section"] == "program_cancellation" and row["metric"].endswith("retained_absolute_fraction")]
    assert len(source_none) == 3 * 4 * 2 and all(row["value"] == "" for row in source_none)
    # No .pt/.npz/.npy artifact is present or needed anywhere in the fixture.
    assert not any(public.rglob("*.pt")) and not any(public.rglob("*.npz")) and not any(public.rglob("*.npy"))
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imported.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom): imported.add((node.module or "").split(".")[0])
    assert not imported.intersection({"torch", "numpy", "scipy", "src"})


def test_maximum_allowed_fourteen_gradient_groups_fit_public_artifact_caps(public, tmp_path):
    reviewer = load_review()
    assert len(reviewer.GROUPS) == 14
    report = read(public / "R7_RESULTS.json")
    for entry in report["diagnostic_index"]["parts"]:
        path = public / entry["file"]; shard = read(path)
        for row in shard["rows"]:
            for epoch in EPOCHS:
                probe = row["checkpoint_diagnostics"][str(epoch)]["gradient_probe"]
                for path_name in PATHS:
                    probe["groups_by_path"][path_name] = {group: dict(tensors=1, numel=64, nonzero=1, L2_squared=.04)
                        for group in sorted(reviewer.GROUPS)}
        put(path, shard); entry.update(sha256=digest(path), bytes=path.stat().st_size)
    put(public / "R7_RESULTS.json", report)
    output = tmp_path / "cap_review"; result = reviewer.build(public, output)
    assert result["status"] == "COMPLETE_METADATA_ONLY_99_CELLS_396_DIAGNOSTICS"
    for path in output.iterdir():
        assert path.stat().st_size <= 2 * 1024**2, path.name
    for record in result["csv_artifacts"]:
        artifact = output / record["file"]
        assert digest(artifact) == record["sha256"] and artifact.stat().st_size == record["bytes"]
        assert len(read_csv(artifact)) == record["rows"]

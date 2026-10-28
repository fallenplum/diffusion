"""Anonymous source-only metadata report; never deserialize models or arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = ("iCMS2", "iCMS3", "FAP_fibroblast_associated_top25", "MARCO_macrophage_associated_top25")
FAMILIES = (("S0", "MATCHED"), ("S1", "MATCHED"), ("S2", "MATCHED"), ("S3", "MATCHED"),
    ("S4", "MATCHED"), ("S5", "WRONG_DONOR"), ("S4", "WITHIN_SECTION_TISSUE_DEPTH"),
    ("S4", "RANDOM_GATE"), ("S6", "MATCHED"), ("S6", "WRONG_NEIGHBOR"),
    ("S7", "MATCHED"), ("S7", "WRONG_SPOT_ALIGNMENT"))
CORE_COMPARISONS = ("S4_vs_S5_WRONG_DONOR", "S4_vs_S4_WITHIN_SECTION_TISSUE_DEPTH")
SOURCE_CODE_FILES = {"scripts/bgb_en_r6_pretrain.py", "scripts/bgb_en_r6_schedule_source.py",
    "src/bgb_en/r6_source.py", "src/bgb_en/r6_source_model.py", "src/bgb_en/r6_omics.py", "src/bgb_en/external.py"}
HASH_GATE_ALGORITHM = "unit_UNI2_float32_little_endian_row_SHA256_first64little_mod6_class_permutation_v1"
PERMUTATION_NAMES = ("012", "021", "102", "120", "201", "210")


def require(value, code):
    if not value:
        raise ValueError(code)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json(path, value):
    temp = Path(path).with_name(Path(path).name + ".writing")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _finite(value, *, none=False):
    return (value is None and none) or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))


def identity(cell):
    return (cell["arm"], cell["control"], cell["heldout_donor"], cell["seed"])


def cell_name(cell):
    return f"{cell['arm']}_{cell['control']}_h{cell['heldout_donor']}_seed{cell['seed']}"


def expected_matrix():
    units = {(arm, control, donor, seed) for arm, control in FAMILIES for donor in range(6) for seed in (31, 47)}
    units |= {("S7", control, -1, 31) for control in ("MATCHED", "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH")}
    return units


def validate_matrix(config):
    require(config.get("schema") == "bgb_en_r6_source_pretrain_lock_v1"
            and config.get("status") == "LOCKED_BEFORE_SOURCE_FITS", "SOURCE_CONFIG_NOT_LOCKED")
    cells = config.get("cells", [])
    require(len(cells) == 147 and len({identity(c) for c in cells}) == 147
            and {identity(c) for c in cells} == expected_matrix(), "FINITE_147_SOURCE_MATRIX_CHANGED")
    require(config.get("fixed_defaults", {}).get("epochs") == 80
            and config.get("fixed_defaults", {}).get("checkpoint_epochs") == [20, 40, 80],
            "SOURCE_EPOCH_BUDGET_CHANGED")
    require(isinstance(config.get("source_code_sha256"), dict)
            and set(config["source_code_sha256"]) == SOURCE_CODE_FILES, "SOURCE_CODE_HASH_LOCK_MISSING")
    for relative, digest in config["source_code_sha256"].items():
        path = (ROOT / relative).resolve()
        require(path.is_relative_to(ROOT) and path.is_file() and sha256(path) == digest, "SOURCE_CODE_HASH_CHANGED")
    return cells


def verify_artifacts(directory, hashes, verified=None):
    require(isinstance(hashes, dict) and hashes, "SOURCE_ARTIFACT_HASH_SET_EMPTY")
    safe = {}
    for filename, digest in hashes.items():
        path = (directory / filename).resolve()
        require(path.parent == directory.resolve() and path.is_file() and len(digest) == 64,
                "SOURCE_ARTIFACT_PATH_OR_HASH_INVALID")
        stat = path.stat()
        cache_key = (stat.st_size, stat.st_mtime_ns)
        cached = verified.get(path) if verified is not None else None
        actual = cached[1] if cached is not None and cached[0] == cache_key else sha256(path)
        if verified is not None:
            verified[path] = (cache_key, actual)
        require(actual == digest, "SOURCE_ARTIFACT_SHA_CHANGED")
        safe[filename] = digest
    return dict(verified_artifact_files=len(safe), artifact_manifest_sha256=canonical_hash(safe))


def _metric_difference(actual, expected):
    if isinstance(actual, dict):
        require(isinstance(expected, dict) and set(actual) == set(expected), "REPLAY_METRIC_KEYS_CHANGED")
        return max((_metric_difference(actual[key], expected[key]) for key in actual), default=0.)
    if isinstance(actual, list):
        require(isinstance(expected, list) and len(actual) == len(expected), "REPLAY_METRIC_LENGTH_CHANGED")
        return max((_metric_difference(a, b) for a, b in zip(actual, expected)), default=0.)
    if actual is None or isinstance(actual, (str, bool)):
        require(actual == expected, "REPLAY_METRIC_NONNUMERIC_CHANGED")
        return 0.
    require(_finite(actual) and _finite(expected), "REPLAY_METRIC_NONFINITE")
    return abs(float(actual) - float(expected))


def gate_metadata(value):
    """Keep only anonymous distribution aggregates; no row probabilities."""
    if not isinstance(value, dict) or value.get("status") == "NOT_RECORDED":
        return dict(status="NOT_RECORDED")
    require(value.get("schema") == "bgb_en_r6_row_gate_audit_v1" and value.get("status") == "RECORDED"
            and type(value.get("rows")) is int and value["rows"] > 0
            and value.get("algorithm") in ("predicted_identity", HASH_GATE_ALGORITHM), "SOURCE_GATE_METADATA_SCHEMA_CHANGED")
    scalar_keys = ("row_probability_multiset_max_abs_difference", "original_entropy_mean", "used_entropy_mean",
                   "row_entropy_max_abs_difference", "identity_rate")
    require(all(_finite(value.get(key)) and value[key] >= 0 for key in scalar_keys), "SOURCE_GATE_AUDIT_NONFINITE")
    counts = value.get("permutation_counts", {})
    require(set(counts) == set(PERMUTATION_NAMES) and all(type(v) is int and v >= 0 for v in counts.values())
            and sum(counts.values()) == value["rows"] and value.get("unique_permutations") == sum(v > 0 for v in counts.values())
            and abs(value["identity_rate"] - counts["012"] / value["rows"]) <= 1e-12, "SOURCE_GATE_PERMUTATION_COUNTS_CHANGED")
    for key in ("original_perclass_probability_mean", "used_perclass_probability_mean"):
        vector = value.get(key, [])
        require(isinstance(vector, list) and len(vector) == 3 and all(_finite(v) and 0 <= v <= 1 for v in vector)
                and abs(sum(vector) - 1.) <= 3e-6, "SOURCE_GATE_MARGINAL_PROBABILITY_INVALID")
    bool_keys = ("row_probability_multiset_exact", "used_probabilities_follow_reported_permutation",
        "permutation_uses_labels", "permutation_uses_patient_or_spot_ids", "permutation_uses_learned_features",
        "between_model_distributions_exactly_matched_claimed", "confidence_information_removed_claimed")
    require(all(isinstance(value.get(key), bool) for key in bool_keys), "SOURCE_GATE_BOOLEAN_AUDIT_MISSING")
    require(all(value[key] is False for key in bool_keys[2:]), "SOURCE_RANDOM_GATE_SCOPE_OR_PRIVACY_CLAIM_CHANGED")
    return {key: value[key] for key in ("schema", "status", "rows", "algorithm", *scalar_keys,
        "original_perclass_probability_mean", "used_perclass_probability_mean", "permutation_counts", "unique_permutations", *bool_keys)}


def cell_gate_metadata(metrics, cell):
    result = {}
    for split in ("train", "heldout"):
        scored = metrics.get(split, {})
        if scored.get("status") == "NOT_APPLICABLE_FULL_SOURCE_NO_HELDOUT":
            continue
        total = gate_metadata(scored.get("gate_diagnostics"))
        by_donor = []
        for entry in scored.get("gate_diagnostics_by_donor", []):
            require(type(entry.get("donor_code")) is int and 0 <= entry["donor_code"] <= 5, "SOURCE_GATE_DONOR_CODE_INVALID")
            by_donor.append(dict(donor_code=entry["donor_code"], diagnostics=gate_metadata(entry.get("diagnostics"))))
        if total["status"] == "RECORDED":
            require(total["rows"] == scored.get("spots") and len({r["donor_code"] for r in by_donor}) == len(by_donor)
                    and all(r["diagnostics"]["status"] == "RECORDED" for r in by_donor)
                    and sum(r["diagnostics"]["rows"] for r in by_donor) == total["rows"], "SOURCE_GATE_DONOR_SUPPORT_CHANGED")
            if cell["heldout_donor"] >= 0 and split == "heldout":
                require([r["donor_code"] for r in by_donor] == [cell["heldout_donor"]], "SOURCE_GATE_HELDOUT_DONOR_SCOPE_CHANGED")
            if cell["heldout_donor"] >= 0 and split == "train":
                require(all(r["donor_code"] != cell["heldout_donor"] for r in by_donor), "SOURCE_GATE_TRAIN_INCLUDED_HELDOUT_DONOR")
            if cell["control"] == "RANDOM_GATE":
                require(total["algorithm"] == HASH_GATE_ALGORITHM and total["row_probability_multiset_exact"]
                        and total["used_probabilities_follow_reported_permutation"], "SOURCE_RANDOM_GATE_ROW_MATCHING_NOT_VERIFIED")
        result[split] = dict(total=total, by_donor=by_donor)
    return result


def read_target_eligibility(path, config):
    evidence = _load(path)
    require(evidence.get("schema") == "bgb_en_r6_source_target_eligibility_v1"
            and evidence.get("status") == "LOCKED_MODEL_INDEPENDENT_SCORING_SUPPORT"
            and evidence.get("source_cache_manifest_sha256") == config["source_cache_manifest_sha256"]
            and evidence.get("fresh_certificate_sha256") == config["fresh_certificate_sha256"],
            "SOURCE_TARGET_ELIGIBILITY_PROVENANCE_CHANGED")
    rows = evidence.get("groups", [])
    expected = {(donor, program, tissue) for donor in range(6) for program in PROGRAMS for tissue in range(3)}
    keys = [(r["heldout_donor"], r["program"], r["tissue_class"]) for r in rows]
    require(len(rows) == 72 and len(set(keys)) == 72 and set(keys) == expected,
            "SOURCE_TARGET_ELIGIBILITY_NOT_COMPLETE_72_GROUPS")
    for row in rows:
        require(isinstance(row.get("support"), int) and row["support"] >= 0
                and isinstance(row.get("target_nonconstant"), bool)
                and row.get("eligible") is (row["support"] >= 3 and row["target_nonconstant"]),
                "SOURCE_TARGET_ELIGIBILITY_RULE_CHANGED")
    return {key: row for key, row in zip(keys, rows)}, sha256(path)


def _score_metadata(metrics, eligibility, heldout_donor):
    require(isinstance(metrics, dict) and metrics.get("true_labels_used_for_scoring_only") is True
            and metrics.get("visual_program_forward_is_gene_and_tissue_truth_free") is True,
            "SOURCE_VISUAL_INFERENCE_CONTRACT_MISSING")
    programs = metrics.get("per_program", [])
    require(len(programs) == 4 and [r.get("program") for r in programs] == list(PROGRAMS),
            "SOURCE_PROGRAM_ORDER_CHANGED")
    require(_finite(metrics.get("tissue_BA")) and 0 <= metrics["tissue_BA"] <= 1,
            "SOURCE_TISSUE_BA_INVALID")
    global_huber, group_rho, eligible_keys = [], [], []
    rows, reasons = [], []
    eligibility_unknown = False
    rho_unavailable = False
    for k, program in enumerate(programs):
        require(_finite(program.get("Huber")) and program["Huber"] >= 0
                and isinstance(program.get("support"), int), "SOURCE_PROGRAM_HUBER_INVALID")
        global_huber.append(program["Huber"])
        groups = program.get("within_tissue", [])
        require(len(groups) == 3 and [g.get("tissue_class") for g in groups] == [0, 1, 2], "SOURCE_TISSUE_GROUP_ORDER_CHANGED")
        require(sum(g["support"] for g in groups) == program["support"], "SOURCE_GROUP_SUPPORT_MISMATCH")
        for group in groups:
            n, rho, huber = group.get("support"), group.get("Spearman"), group.get("Huber")
            require(isinstance(n, int) and n >= 0 and _finite(rho, none=True) and _finite(huber, none=True), "SOURCE_GROUP_METRIC_INVALID")
            fixed_group = eligibility[(heldout_donor, PROGRAMS[k], group["tissue_class"])]
            require(n == fixed_group["support"], "SOURCE_TARGET_ELIGIBILITY_SUPPORT_CHANGED")
            target_nonconstant = fixed_group["target_nonconstant"]
            eligible = n >= 3 and target_nonconstant is True
            reason = "ELIGIBLE" if eligible else ("SUPPORT_BELOW_THREE" if n < 3 else
                "TARGET_CONSTANT" if target_nonconstant is False else "TARGET_ELIGIBILITY_UNKNOWN")
            if eligible:
                eligible_keys.append((k, group["tissue_class"], n))
                if rho is None:
                    rho_unavailable = True
                    reasons.append("ELIGIBLE_GROUP_HAS_UNDEFINED_PREDICTION_CORRELATION")
                else:
                    group_rho.append(float(rho))
            rows.append(dict(program=PROGRAMS[k], tissue_class=group["tissue_class"], support=n,
                Huber=huber, Spearman=rho, program_global_Huber=program["Huber"],
                tissue_BA=metrics["tissue_BA"], target_nonconstant=target_nonconstant,
                prediction_nonconstant=group.get("prediction_nonconstant"),
                eligible_for_within_tissue_rho=eligible if target_nonconstant is not None or n < 3 else None,
                eligibility_reason=reason))
    within = None if eligibility_unknown or rho_unavailable or not eligible_keys else statistics.mean(group_rho)
    require(_finite(metrics.get("program_Huber")) and abs(statistics.mean(global_huber) - metrics["program_Huber"]) <= 1e-12,
            "SOURCE_GLOBAL_HUBER_NOT_FOUR_PROGRAM_MEAN")
    return dict(program_Huber=statistics.mean(global_huber), within_tissue_Spearman=within,
        tissue_BA=float(metrics["tissue_BA"]), eligible_groups=len(eligible_keys),
        target_eligibility_signature=canonical_hash(eligible_keys) if not eligibility_unknown else None,
        rho_status="AVAILABLE" if within is not None else "INCONCLUSIVE",
        rho_unavailability_reasons=sorted(set(reasons))), rows


def _curve_metadata(directory, cell, receipt, config_sha, source_hashes, verified):
    if cell["arm"] == "S0":
        return dict(status="RANDOM_REFERENCE_ZERO_UPDATES", optimizer_updates=0, checkpoints=[80], curve_points=0), []
    path = directory / "TRAINING_CURVE.json"
    require(path.name in receipt["artifact_sha256"] and path.is_file(), "SOURCE_TRAINING_CURVE_MISSING")
    curve = _load(path)
    require(len(curve) == 80 and [r.get("epoch") for r in curve] == list(range(1, 81))
            and [r.get("optimizer_updates") for r in curve] == list(range(1, 81)), "SOURCE_ACTUAL_UPDATE_CURVE_NOT_EIGHTY")
    for record in curve:
        require(record.get("complete_source_updates_represented") == 1
                and record.get("negative_key_gradients_included") is True
                and _finite(record.get("objective")) and _finite(record.get("parameter_L2"))
                and _finite(record.get("replay_max_abs_error")), "SOURCE_CACHE_UPDATE_AUDIT_MISSING")
    result = []
    for epoch in (20, 40, 80):
        snapshot_path = directory / f"SNAPSHOT_{epoch:03d}.json"
        require(snapshot_path.name in receipt["artifact_sha256"], "SOURCE_BUDGET_SNAPSHOT_NOT_HASH_BOUND")
        snapshot = _load(snapshot_path)
        require(snapshot.get("status") == "COMPLETE_SOURCE_SNAPSHOT" and snapshot.get("cell") == cell
                and snapshot.get("epoch") == snapshot.get("optimizer_updates") == epoch
                and snapshot.get("config_sha256") == config_sha and snapshot.get("source_code_sha256") == source_hashes,
                "SOURCE_BUDGET_SNAPSHOT_LINEAGE_CHANGED")
        verify_artifacts(directory, snapshot.get("artifact_sha256"), verified)
        record = curve[epoch - 1]
        small = dict(epoch=epoch, objective=float(record["objective"]), parameter_L2=float(record["parameter_L2"]),
            preclip_gradient_norm=record.get("preclip_gradient_norm"), replay_max_abs_error=record["replay_max_abs_error"])
        components = record.get("component_audit", {})
        for key in ("program_huber", "tissue_CE", "alignment", "neighborhood"):
            small[key] = components.get(key)
        for split in ("train", "heldout"):
            source = snapshot.get("metrics", {}).get(split, {})
            small[split + "_program_Huber"] = source.get("program_Huber")
            small[split + "_tissue_BA"] = source.get("tissue_BA")
        require(all(_finite(v, none=True) for k, v in small.items() if k != "epoch"), "SOURCE_BUDGET_METRIC_NONFINITE")
        result.append(small)
    return dict(status="COMPLETE_FIXED_EIGHTY_UPDATE_BUDGET", optimizer_updates=80,
        curve_points=80, checkpoints=[20, 40, 80],
        objective_80_minus_20=result[-1]["objective"] - result[0]["objective"],
        epoch_selected_using_heldout=False, convergence_proven=False), result


def read_unit(private_root, cell, config, config_sha, eligibility):
    directory = Path(private_root) / "source_cells" / cell_name(cell)
    row = dict(arm=cell["arm"], control=cell["control"], heldout_donor=cell["heldout_donor"],
        seed=cell["seed"], planned_optimizer_updates=0 if cell["arm"] == "S0" else 80,
        planned_optimizer_fits=0 if cell["arm"] == "S0" else 1, status="MISSING",
        actual_optimizer_updates=None, actual_optimizer_fits=None, metadata_validated=False,
        independent_replay_passed=False, full_source=cell["heldout_donor"] < 0,
        elapsed_seconds=None, receipt_sha256=None, replay_sha256=None,
        verified_artifact_files=0, artifact_manifest_sha256=None)
    error = directory / "ERROR.json"
    complete = directory / "COMPLETE.json"
    if not complete.exists():
        if error.exists():
            try:
                evidence = _load(error)
                require(evidence.get("status") == "ERROR_SOURCE_FIT" and evidence.get("cell") == cell
                        and evidence.get("config_sha256") == config_sha
                        and evidence.get("source_code_sha256") == config["source_code_sha256"],
                        "SOURCE_ERROR_RECEIPT_LINEAGE_CHANGED")
                row["status"] = "ERROR"
                updates = evidence.get("completed_optimizer_updates")
                fits = evidence.get("actual_new_optimizer_fits")
                if type(updates) is int and 0 <= updates <= row["planned_optimizer_updates"]:
                    row["actual_optimizer_updates"] = updates
                if type(fits) is int and fits in (0, 1):
                    row["actual_optimizer_fits"] = fits
            except (ValueError, KeyError, TypeError, OSError):
                row.update(status="INVALID_ERROR_RECEIPT", validation_failure_code="SOURCE_ERROR_RECEIPT_LINEAGE_OR_JSON_INVALID")
            # Error text and tracebacks remain private.
        elif directory.exists() and any(directory.iterdir()):
            row["status"] = "INCOMPLETE_PRESERVED"
        return row, [], None
    try:
        require(not error.exists(), "CONFLICTING_ERROR_AND_COMPLETE_RECEIPTS")
        evidence = _load(complete)
        require(evidence.get("cell") == cell and evidence.get("config_sha256") == config_sha
                and evidence.get("source_code_sha256") == config["source_code_sha256"], "SOURCE_COMPLETION_LINEAGE_CHANGED")
        status = "COMPLETE_R6_RANDOM_REFERENCE_NO_OPTIMIZER" if cell["arm"] == "S0" else "COMPLETE_R6_SOURCE_FIT"
        require(evidence.get("status") == status and evidence.get("epochs") == 80
                and evidence.get("optimizer_updates") == row["planned_optimizer_updates"]
                and evidence.get("actual_new_optimizer_fits") == row["planned_optimizer_fits"], "SOURCE_COMPLETION_BUDGET_CHANGED")
        verified = {}
        row.update(verify_artifacts(directory, evidence.get("artifact_sha256"), verified))
        row.update(elapsed_seconds=evidence.get("elapsed_seconds"), receipt_sha256=sha256(complete))
        require(_finite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0, "SOURCE_COST_NONFINITE")
        budget, budget_rows = _curve_metadata(directory, cell, evidence, config_sha, config["source_code_sha256"], verified)
        row.update(actual_optimizer_updates=evidence["optimizer_updates"], actual_optimizer_fits=evidence["actual_new_optimizer_fits"])
        row["training_budget"] = budget
        row["training_budget_snapshots"] = budget_rows
        row["parameter_budget"] = {k: evidence.get("parameter_budget", {}).get(k) for k in ("registered", "requires_grad")}
        row["gate_metadata"] = cell_gate_metadata(evidence.get("metrics", {}), cell)
        scores, metrics = None, []
        if cell["heldout_donor"] < 0:
            require(evidence.get("full_source_no_heldout") is True
                    and evidence.get("metrics", {}).get("heldout", {}).get("status") == "NOT_APPLICABLE_FULL_SOURCE_NO_HELDOUT",
                    "FULL_SOURCE_HAS_FAKE_HELDOUT_METRICS")
        else:
            scores, metrics = _score_metadata(evidence.get("metrics", {}).get("heldout"), eligibility, cell["heldout_donor"])
        row["metadata_validated"] = True
        replay_path = directory / "REPLAY.json"
        if not replay_path.exists():
            row["status"] = "FIT_COMPLETE_REPLAY_MISSING"
            return row, [], scores
        replay = _load(replay_path)
        require(replay.get("status") == "PASS_COMPLETE_R6_SOURCE_CHECKPOINT_REPLAY"
                and replay.get("cell") == cell and replay.get("config_sha256") == config_sha
                and replay.get("source_code_sha256") == config["source_code_sha256"]
                and replay.get("completion_sha256") == sha256(complete) and replay.get("optimizer_updates") == 0
                and replay.get("new_optimizer_fits") == 0 and replay.get("model_frozen") is True
                and replay.get("image_only_heldout_program_prediction") is True
                and replay.get("original_fit_artifacts_modified") is False,
                "INDEPENDENT_SOURCE_REPLAY_LINEAGE_OR_STATUS_FAILED")
        require(_metric_difference(replay.get("metrics"), evidence.get("metrics")) <= 2e-6,
                "INDEPENDENT_SOURCE_REPLAY_METRICS_CHANGED")
        forward_errors = replay.get("forward_max_abs_errors", {})
        expected_splits = {"train"} if cell["heldout_donor"] < 0 else {"train", "heldout"}
        require(set(forward_errors) == expected_splits and all(isinstance(v, dict) and v
                and all(_finite(error) and 0 <= error <= 2e-6 for error in v.values()) for v in forward_errors.values()),
                "INDEPENDENT_SOURCE_REPLAY_FORWARD_ERROR_AUDIT_MISSING_OR_FAILED")
        row.update(status="PASS_COMPLETE_AND_REPLAYED", independent_replay_passed=True,
                   replay_sha256=sha256(replay_path))
        return row, [dict(arm=cell["arm"], control=cell["control"], heldout_donor=cell["heldout_donor"],
                         seed=cell["seed"], status=row["status"], **record) for record in metrics], scores
    except (ValueError, KeyError, TypeError, OSError) as error_:
        row.update(status="INVALID", validation_failure_code=str(error_) if isinstance(error_, ValueError) else type(error_).__name__)
        # All ValueError messages above are fixed codes; never publish OS errors.
        if not row["validation_failure_code"].replace("_", "").isalnum():
            row["validation_failure_code"] = type(error_).__name__
        return row, [], None


def _mean(values):
    return statistics.mean(values) if values and all(v is not None and _finite(v) for v in values) else None


def family_summary(family, ledger, scores):
    arm, control = family
    units = [r for r in ledger if (r["arm"], r["control"]) == family and not r["full_source"]]
    good = [r for r in units if r["status"] == "PASS_COMPLETE_AND_REPLAYED"]
    summary = dict(arm=arm, control=control, expected_units=12, complete_replayed_units=len(good),
        status="COMPLETE_12_LODO_UNITS" if len(good) == 12 else "INCOMPLETE_NO_OVERALL_MEAN",
        program_Huber=None, within_tissue_Spearman=None, tissue_BA=None, per_seed=[])
    if len(good) != 12:
        return summary
    for seed in (31, 47):
        ordered = [scores[(arm, control, donor, seed)] for donor in range(6)]
        summary["per_seed"].append(dict(seed=seed, donor_count=6,
            program_Huber=_mean([r["program_Huber"] for r in ordered]),
            within_tissue_Spearman=_mean([r["within_tissue_Spearman"] for r in ordered]),
            tissue_BA=_mean([r["tissue_BA"] for r in ordered])))
    for metric in ("program_Huber", "within_tissue_Spearman", "tissue_BA"):
        summary[metric] = _mean([seed[metric] for seed in summary["per_seed"]])
    summary["aggregation"] = "each_cell_equal_four_program_global_Huber_and_target_eligible_tissue_group_rho_then_equal_six_donors_then_equal_two_seeds"
    return summary


def paired_comparison(name, true_family, negative_family, ledger, scores):
    result = dict(name=name, true_arm=true_family[0], true_control=true_family[1],
        reference_arm=negative_family[0], reference_control=negative_family[1],
        expected_pairs=12, complete_pairs=0, status="INCONCLUSIVE", per_seed=[], pairs=[],
        Huber_negative_minus_true=None, within_tissue_rho_true_minus_negative=None)
    valid = {identity(r): r for r in ledger if r["status"] == "PASS_COMPLETE_AND_REPLAYED"}
    eligibility_mismatch = False
    for seed in (31, 47):
        pairs = []
        for donor in range(6):
            a, b = (*true_family, donor, seed), (*negative_family, donor, seed)
            if a not in valid or b not in valid:
                continue
            left, right = scores[a], scores[b]
            same_eligibility = left["target_eligibility_signature"] is not None and left["target_eligibility_signature"] == right["target_eligibility_signature"]
            eligibility_mismatch |= not same_eligibility
            rho_delta = left["within_tissue_Spearman"] - right["within_tissue_Spearman"] if same_eligibility and left["within_tissue_Spearman"] is not None and right["within_tissue_Spearman"] is not None else None
            pair = dict(seed=seed, heldout_donor=donor, Huber_negative_minus_true=right["program_Huber"] - left["program_Huber"],
                        within_tissue_rho_true_minus_negative=rho_delta, model_independent_eligible_groups_match=same_eligibility)
            pairs.append(pair); result["pairs"].append(pair)
        result["per_seed"].append(dict(seed=seed, complete_donors=len(pairs),
            Huber_negative_minus_true=_mean([r["Huber_negative_minus_true"] for r in pairs]) if len(pairs) == 6 else None,
            within_tissue_rho_true_minus_negative=_mean([r["within_tissue_rho_true_minus_negative"] for r in pairs]) if len(pairs) == 6 else None))
    result["complete_pairs"] = len(result["pairs"])
    if result["complete_pairs"] != 12:
        result["reason"] = "MISSING_REQUIRED_COMPLETE_REPLAYED_SOURCE_UNITS"
        return result
    result["Huber_negative_minus_true"] = _mean([r["Huber_negative_minus_true"] for r in result["per_seed"]])
    result["within_tissue_rho_true_minus_negative"] = _mean([r["within_tissue_rho_true_minus_negative"] for r in result["per_seed"]])
    if eligibility_mismatch or result["within_tissue_rho_true_minus_negative"] is None:
        result["reason"] = "TARGET_ELIGIBILITY_MISSING_MISMATCHED_OR_ELIGIBLE_RHO_UNDEFINED"
        return result
    result["status"] = "SUPPORTED" if result["Huber_negative_minus_true"] > 0 and result["within_tissue_rho_true_minus_negative"] > 0 else "NOT_SUPPORTED"
    result["reason"] = "PRELOCKED_DIRECTIONS_ONLY_NO_NEW_THRESHOLD_OR_SIGNIFICANCE_CLAIM"
    return result


def source_decision(comparisons):
    required = [comparisons[key]["status"] for key in CORE_COMPARISONS]
    status = "INCONCLUSIVE" if "INCONCLUSIVE" in required else "SUPPORTED" if all(s == "SUPPORTED" for s in required) else "NOT_SUPPORTED"
    return dict(status=status, required_comparisons=list(CORE_COMPARISONS),
        rule="both_comparisons_have_positive_equal_donor_seed_Huber_negative_minus_true_and_within_tissue_rho_true_minus_negative",
        source_mechanism_only=True, causality_or_cell_source_truth_established=False,
        source_score_dependent_execution_gate=False, all_locked_source_fits_still_required=True,
        survival_benefit_established=False, target_outer_evaluation_authorized_by_this_report=False,
        confidence_statement="该状态描述预锁方向证据；未计算置信区间、p值或独立外部泛化证据。")


def build_report(config_path, private_root, *, target_eligibility, qualification_receipt=None):
    config = _load(config_path)
    cells = validate_matrix(config)
    config_sha = sha256(config_path)
    eligibility, eligibility_sha = read_target_eligibility(target_eligibility, config)
    ledger, metric_rows, scores = [], [], {}
    for cell in cells:
        record, metrics, score = read_unit(private_root, cell, config, config_sha, eligibility)
        ledger.append(record); metric_rows.extend(metrics)
        if score is not None:
            scores[identity(cell)] = score
    families = [family_summary(f, ledger, scores) for f in FAMILIES]
    specs = (
        ("S4_vs_S5_WRONG_DONOR", ("S4", "MATCHED"), ("S5", "WRONG_DONOR")),
        ("S4_vs_S4_WITHIN_SECTION_TISSUE_DEPTH", ("S4", "MATCHED"), ("S4", "WITHIN_SECTION_TISSUE_DEPTH")),
        ("S4_vs_S3_ADDITIVE", ("S4", "MATCHED"), ("S3", "MATCHED")),
        ("S4_vs_S4_RANDOM_GATE", ("S4", "MATCHED"), ("S4", "RANDOM_GATE")),
        ("S6_vs_S4_NO_NEIGHBOR", ("S6", "MATCHED"), ("S4", "MATCHED")),
        ("S6_vs_S6_WRONG_NEIGHBOR", ("S6", "MATCHED"), ("S6", "WRONG_NEIGHBOR")),
        ("S7_vs_S4_NO_GENE_ALIGNMENT", ("S7", "MATCHED"), ("S4", "MATCHED")),
        ("S7_vs_S7_WRONG_SPOT_ALIGNMENT", ("S7", "MATCHED"), ("S7", "WRONG_SPOT_ALIGNMENT")))
    comparisons = {name: paired_comparison(name, a, b, ledger, scores) for name, a, b in specs}
    counts = {status: sum(r["status"] == status for r in ledger) for status in sorted({r["status"] for r in ledger})}
    qualification = dict(provided=False, optimizer_updates=None, optimizer_instances=None, formal_source_fits=None)
    if qualification_receipt is not None:
        receipt = _load(qualification_receipt)
        qualification.update(provided=True, receipt_sha256=sha256(qualification_receipt),
            optimizer_updates=receipt.get("qualification_updates", receipt.get("optimizer_updates", receipt.get("actual_optimizer_updates"))),
            optimizer_instances=receipt.get("separate_qualification_optimizer_instances", receipt.get("actual_new_optimizer_fits", receipt.get("new_optimizer_fits"))),
            formal_source_fits=receipt.get("formal_source_fits"))
        for key in ("optimizer_updates", "optimizer_instances", "formal_source_fits"):
            require(qualification[key] is None or isinstance(qualification[key], int) and qualification[key] >= 0,
                    "SOURCE_QUALIFICATION_COST_INVALID")
        require(qualification["formal_source_fits"] in (None, 0), "QUALIFICATION_MUST_NOT_BE_A_FORMAL_SOURCE_FIT")
    report = dict(schema="bgb_en_r6_source_supervision_report_v1", source_config_sha256=config_sha,
        report_code_sha256=sha256(Path(__file__)), source_code_sha256=config["source_code_sha256"],
        source_cache_manifest_sha256=config["source_cache_manifest_sha256"],
        fresh_certificate_sha256=config["fresh_certificate_sha256"],
        target_eligibility_sha256=eligibility_sha, target_eligible_groups=sum(r["eligible"] for r in eligibility.values()),
        eligibility_is_model_independent_source_scoring_metadata=True,
        status="COMPLETE_147_SOURCE_UNITS_AND_REPLAY" if counts.get("PASS_COMPLETE_AND_REPLAYED") == 147 else "PARTIAL_SOURCE_MATRIX",
        planned_logical_units=147, planned_formal_optimizer_fits=135, planned_formal_optimizer_updates=135 * 80,
        planned_random_references=12, counts=counts,
        confirmed_formal_optimizer_fits=sum(r["actual_optimizer_fits"] for r in ledger if r["actual_optimizer_fits"] is not None),
        confirmed_formal_optimizer_updates=sum(r["actual_optimizer_updates"] for r in ledger if r["actual_optimizer_updates"] is not None),
        unknown_fit_cost_units=sum(r["actual_optimizer_fits"] is None for r in ledger),
        unknown_update_cost_units=sum(r["actual_optimizer_updates"] is None for r in ledger),
        qualification_cost_separate_from_formal_matrix=qualification,
        complete_family_summaries=families, comparisons=comparisons, decision=source_decision(comparisons),
        ledger=ledger, full_source_transfer_units=[r for r in ledger if r["full_source"]],
        no_partial_overall_means=True, no_epoch_selection=True,
        no_patient_or_spot_rows_loaded=True, no_model_or_array_deserialization=True,
        no_CI_pvalue_or_new_numerical_mechanism_threshold=True,
        source_prediction_not_DSS_increment=True,
        source_programs_are_not_measured_cell_intrinsic_activity=True)
    return report, metric_rows


def write_report(report, metric_rows, output):
    output = Path(output).resolve()
    protected = (ROOT / "reports/bgb_en", ROOT / "reports/bgb_en_r5")
    require(not any(output == path.resolve() or output.is_relative_to(path.resolve()) for path in protected),
            "HISTORICAL_REPORT_NAMESPACES_PRESERVED")
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "R6_1_SOURCE_SUPERVISION.json", report)
    _json(output / "R6_SOURCE_DECISION.json", report["decision"])
    source_audit_path = output / "SOURCE_AUDIT.json"
    source_audit = _load(source_audit_path) if source_audit_path.exists() else {}
    source_audit["r6_source_supervision"] = {key: report[key] for key in (
        "source_config_sha256", "report_code_sha256", "source_code_sha256", "source_cache_manifest_sha256",
        "fresh_certificate_sha256", "target_eligibility_sha256", "target_eligible_groups",
        "eligibility_is_model_independent_source_scoring_metadata", "no_patient_or_spot_rows_loaded",
        "no_model_or_array_deserialization")}
    _json(source_audit_path, source_audit)
    _json(output / "R6_SOURCE_REPLAY_AUDIT.json", dict(status=report["status"], source_config_sha256=report["source_config_sha256"],
        expected_units=147, passed_units=report["counts"].get("PASS_COMPLETE_AND_REPLAYED", 0),
        source_code_sha256=report["source_code_sha256"], metadata_only_report=True,
        rows=[{key: r[key] for key in ("arm", "control", "heldout_donor", "seed", "status", "metadata_validated",
            "independent_replay_passed", "receipt_sha256", "replay_sha256", "verified_artifact_files", "artifact_manifest_sha256")} for r in report["ledger"]]))
    _csv(output / "R6_SOURCE_CELL_METRICS.csv", metric_rows,
        ["arm", "control", "heldout_donor", "seed", "status", "program", "tissue_class", "support", "Huber", "Spearman",
         "program_global_Huber", "tissue_BA", "target_nonconstant", "prediction_nonconstant", "eligible_for_within_tissue_rho", "eligibility_reason"])
    _csv(output / "R6_SOURCE_COSTS.csv", report["ledger"],
        ["arm", "control", "heldout_donor", "seed", "status", "full_source", "planned_optimizer_fits", "actual_optimizer_fits",
         "planned_optimizer_updates", "actual_optimizer_updates", "elapsed_seconds", "independent_replay_passed"])
    gate_rows = []
    for cell in report["ledger"]:
        for split, metadata in cell.get("gate_metadata", {}).items():
            scoped = [("split_total", None, metadata["total"])] + [("donor", r["donor_code"], r["diagnostics"]) for r in metadata["by_donor"]]
            for scope, donor, data in scoped:
                record = {key: cell[key] for key in ("arm", "control", "heldout_donor", "seed")}
                record.update(split=split, scope=scope, donor_code=donor, status=data["status"])
                if data["status"] == "RECORDED":
                    record.update({key: data[key] for key in ("rows", "algorithm", "row_probability_multiset_exact",
                        "row_probability_multiset_max_abs_difference", "row_entropy_max_abs_difference",
                        "original_entropy_mean", "used_entropy_mean", "identity_rate", "unique_permutations")})
                    record.update({"permutation_" + key: value for key, value in data["permutation_counts"].items()})
                    for prefix, vector in (("original_p", data["original_perclass_probability_mean"]), ("used_p", data["used_perclass_probability_mean"])):
                        record.update({prefix + str(i): value for i, value in enumerate(vector)})
                gate_rows.append(record)
    _csv(output / "R6_SOURCE_GATE_AUDIT.csv", gate_rows, ["arm", "control", "heldout_donor", "seed", "split", "scope", "donor_code", "status",
        "rows", "algorithm", "row_probability_multiset_exact", "row_probability_multiset_max_abs_difference", "row_entropy_max_abs_difference",
        "original_entropy_mean", "used_entropy_mean", "identity_rate", "unique_permutations", *["permutation_" + name for name in PERMUTATION_NAMES],
        "original_p0", "original_p1", "original_p2", "used_p0", "used_p1", "used_p2"])
    family_rows = [{key: row.get(key) for key in ("arm", "control", "status", "expected_units", "complete_replayed_units",
        "program_Huber", "within_tissue_Spearman", "tissue_BA")} for row in report["complete_family_summaries"]]
    _csv(output / "R6_1_SOURCE_SUPERVISION.csv", family_rows,
         ["arm", "control", "status", "expected_units", "complete_replayed_units", "program_Huber", "within_tissue_Spearman", "tissue_BA"])
    lines = ["# R6 源监督与独立重放审计", "", f"矩阵状态：**{report['status']}**；来源机制方向：**{report['decision']['status']}**。", "",
        f"预锁147个逻辑单元（135个正式拟合、12个零优化随机参考）；已确认{report['confirmed_formal_optimizer_fits']}个拟合、{report['confirmed_formal_optimizer_updates']}次完整源更新。",
        f"尚有{report['unknown_update_cost_units']}个单元更新数未知；资格验证成本独立列账。", "",
        "## 供体隔离结果", "", "Huber每单元先等权四程序，再等权六供体、两seed。相关系数先等权模型无关的有效组织组；不使用spot数加权供体。缺单元不输出完整均值。", "",
        "| 模型 | 控制 | 完成/12 | Huber | 组织内ρ | 组织BA |", "|---|---|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.6f}"
    for row in family_rows:
        lines.append(f"| {row['arm']} | {row['control']} | {row['complete_replayed_units']} | {fmt(row['program_Huber'])} | {fmt(row['within_tissue_Spearman'])} | {fmt(row['tissue_BA'])} |")
    lines += ["", "## 预锁方向比较", "", "正向定义：Huber为参照减真实模型，组织内ρ为真实模型减参照。两个量分别报告；没有CI、p值或事后效应阈值。", "",
              "| 比较 | 完整配对 | ΔHuber | Δρ | 状态 |", "|---|---:|---:|---:|---|"]
    for comparison in report["comparisons"].values():
        lines.append(f"| {comparison['name']} | {comparison['complete_pairs']}/12 | {fmt(comparison['Huber_negative_minus_true'])} | {fmt(comparison['within_tissue_rho_true_minus_negative'])} | {comparison['status']} |")
    lines += ["", "## 训练预算与边界", "", "20/40/80轮诊断保存在JSON各单元的training_budget_snapshots中。最终轮固定80；不按已见heldout结果选epoch，不凭训练损失下降宣称收敛。秒数来自fit收据，不含独立重放、排队或资格验证。", "",
        "随机gate由原始冻结UNIT UNI2逐行float32字节SHA决定类别置换，再重排该模型自身的预测概率。逐行概率集合与熵保持；列边际与两个已训练模型的概率分布不保证相同，置信度信息仍被保留。逐单元、逐供体诊断见R6_SOURCE_GATE_AUDIT.csv；identity不重抽，单一置换退化不能当成充分的语义null。", "",
        "完整源S7的三个迁移模型没有heldout结果，单独列账，不进入LODO性能均值。空间sham只打乱同切片几何，不使用组织真值或宣称组成匹配。", "",
        "优势：逐单元核对实际产物SHA、完整状态独立重放收据与供体隔离聚合。风险：源预测能力不能证明细胞来源正确，也不能证明TCGA DSS条件增量；源供体仍只有六个。", "",
        "## 拓展思考", "", "仅在源对应证据与独立患者级生存增量都成立时，才讨论可迁移机制。", "",
        "## 你可能想知道", "", "报告只读取匿名JSON及训练曲线，并对二进制文件计算SHA；没有反序列化权重、数组或读取原始spot记录。", ""]
    (output / "R6_1_SOURCE_SUPERVISION.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bgb_en_r6")
    parser.add_argument("--qualification-receipt", type=Path)
    parser.add_argument("--target-eligibility", type=Path, required=True)
    args = parser.parse_args()
    report, metrics = build_report(args.config, args.private_root, target_eligibility=args.target_eligibility,
                                   qualification_receipt=args.qualification_receipt)
    write_report(report, metrics, args.output)
    print(json.dumps(dict(status=report["status"], decision=report["decision"]["status"], counts=report["counts"])))


if __name__ == "__main__":
    main()

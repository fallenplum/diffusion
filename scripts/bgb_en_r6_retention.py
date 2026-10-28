"""Finite RNA-only retention controls; source/R5 gates precede target access.

Importing this module does not read cohorts, load weights, create an optimizer,
or fit anything. Private transformation caches are prepared before the separate
retention lock. Convex fits require that lock and never retry failed cells.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/bgb_en/r6_retention_v1.json"
SCRIPT = "scripts/bgb_en_r6_retention.py"
REPRESENTATIONS = ("FIXED_HALLMARK_SUM50", "FIXED_HALLMARK_MEAN50", "ST_GENE_UNION_RAW760",
                   "ST_SIGNED_PROGRAM_MEAN4", "FROZEN_S7_FLAT256", "FROZEN_S7_MEAN64")
DIMENSIONS = dict(zip(REPRESENTATIONS, (50, 50, 760, 4, 256, 64)))
REFERENCE_NAMES = ("LEGACY_RNA512_RIDGE", "LEGACY_FULL_RNA20511_RIDGE", "LEGACY_RNA512_PCA256_UNWHITENED_GENE_L2")
SOLVER = dict(family="L-BFGS-B", objective="full_train_riskset_Efron_mean_event_loss_plus_lambda_over2_coefL2",
              lambda_L2=1., initial_coefficient=0., maxiter=2000, maxls=50, ftol=1e-15, gtol=1e-9,
              accepted_gradient_max=1e-6, automatic_retry=False, hyperparameter_search=False)
SOURCE_FILES = (SCRIPT, CONFIG, "scripts/bgb_en_r6_common.py", "scripts/bgb_en_r6_pretrain.py",
    "src/bgb_en/r6_source.py", "src/bgb_en/r6_source_model.py", "src/bgb_en/r6_omics.py",
    "src/bgb_en/r6_pathology.py", "src/bgb_en/external.py", "src/bgb_en/replay.py",
    "src/phase10_trimodal/math.py", "src/phase10_trimodal/reference.py",
    "configs/sami_surv_stage_b_programs_v1.json")


def require(value, code):
    if not bool(value):
        raise ValueError(code)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def exclusive_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def private_root(path, *, readonly_roots=()):
    target = Path(path).resolve()
    require(not any((parent / ".git").exists() for parent in (target, *target.parents))
            and not target.is_relative_to(ROOT), "RETENTION_PRIVATE_DIRECTORY_MUST_BE_OUTSIDE_GIT")
    require(all(not target.is_relative_to(Path(parent).resolve()) for parent in readonly_roots),
            "RETENTION_CANNOT_WRITE_IN_SOURCE_OR_HISTORICAL_DATA")
    return target


def validate_recipe(config):
    require(config.get("schema") == "bgb_en_r6_retention_v1"
            and config.get("status") == "RECIPE_LOCKED_BEFORE_ANY_RETENTION_FIT"
            and tuple(config.get("representations", ())) == REPRESENTATIONS
            and config.get("partitions") == [0, 1, 2] and config.get("outer_fold") == 1,
            "RETENTION_FINITE_RECIPE_CHANGED")
    require(config.get("maximum_logical_Cox_cells") == config.get("maximum_new_optimizer_fits") == 18
            and config.get("convex_seed_replicates") == 1 and config.get("solver") == SOLVER
            and config.get("RNA_only") is True and config.get("CNA_enabled") is False
            and config.get("Clinical_or_pathology_inputs") is False
            and config.get("Set_or_Mamba_new_training_allowed") is False,
            "RETENTION_NO_EXTRA_CANDIDATES_OR_TUNING")
    require(config.get("source_control") == "MATCHED" and config.get("source_full_seed") == 31
            and config.get("source_arm") == "S7" and config.get("source_gene_topk") is None
            and config.get("source_full_three_controls_required") is True
            and config.get("launch_gate_required_before_target_label_loading") is True,
            "RETENTION_SOURCE_FAMILY_OR_GATE_CHANGED")
    require(config.get("scaler") == dict(imputation="current_training_observed_median_with_missing_mask_preserved",
            moments="current_training_observed_values_only_population_SD", minimum_SD=1e-8,
            constant_or_train_unobserved_columns_scale=1., unmeasured_numeric_storage_after_scaling=0.),
            "RETENTION_FOLD_SCALER_RECIPE_CHANGED")
    require(config.get("equivalence") == dict(families=list(REPRESENTATIONS[:2]),
            rule="exact_array_equal_scaled_train_and_validation_features_and_masks",
            representative=REPRESENTATIONS[0], approximate_equality_or_score_based_reuse=False),
            "RETENTION_EXACT_REUSE_RULE_CHANGED")
    require(config.get("replay") == dict(risk_max_abs_error=1e-12, independent_gradient_max=1e-6,
            independent_objective_max_abs_error=1e-10,
            readout="saved_float64_coefficients_with_independent_event_time_Efron_gradient"),
            "RETENTION_REPLAY_GATE_CHANGED")
    require(config.get("evaluation") == dict(primary_endpoint="DSS", tau=60., within_partition_Harrell_and_Uno60=True,
            aggregate="sum_within_partition_numerators_denominators_no_cross_partition_risk_sorting",
            bootstrap=False, confidence_intervals=False, p_values=False, outer_test_evaluated=False)
            and config["normalization_geometry"].get("learned_tokens_are_original_genes") is False,
            "RETENTION_ENDPOINT_OR_INFORMATION_CLAIM_CHANGED")
    refs = config.get("historical_references", [])
    require(len(refs) == 9 and {(r["reference"], r["partition"]) for r in refs}
            == {(name, part) for name in REFERENCE_NAMES for part in (0, 1, 2)},
            "RETENTION_EXACT_NINE_REFERENCE_IDENTITIES_REQUIRED")
    return config


def validate_launch_gate(path, expected_sha):
    require(path is not None and expected_sha is not None and sha256(path) == expected_sha,
            "R5_LAUNCH_GATE_REQUIRED_BEFORE_TARGET_LABEL_ACCESS")
    gate = read(path)
    require(gate.get("schema") == "bgb_en_r6_source_launch_gate_v1"
            and gate.get("r6_formal_training_allowed") is True
            and gate.get("R5_decision") in ("R5_READOUT_STOP_R6_ELIGIBLE", "R5_PERFORMANCE_ONLY_NOT_MECHANISM")
            and set(gate.get("R5_public_evidence_sha256", {})) == {
                "decision.json", "R5_REPORT_AUDIT.json", "R5_CHECKPOINT_REPLAY_AUDIT.json", "PROTOCOL_LOCK.json"},
            "COMPLETE_R5_GATE_DOES_NOT_AUTHORIZE_RETENTION")
    return gate


def masked_feature_scale(values, measured, train):
    """Fit only observed training entries; preserve explicit missingness."""
    x, mask, idx = np.asarray(values, np.float64), np.asarray(measured), np.asarray(train)
    require(x.ndim == 2 and mask.shape == x.shape and mask.dtype == bool
            and idx.ndim == 1 and idx.dtype.kind in "iu" and len(idx) > 1
            and len(np.unique(idx)) == len(idx) and ((idx >= 0) & (idx < len(x))).all()
            and np.isfinite(x[mask]).all(), "INVALID_RETENTION_FEATURE_SCALE_INPUT")
    count = mask[idx].sum(0)
    safe = np.where(mask[idx], x[idx], 0.)
    mean = safe.sum(0) / np.maximum(count, 1)
    sd = np.sqrt(np.where(mask[idx], (safe - mean)**2, 0.).sum(0) / np.maximum(count, 1))
    constant = sd <= 1e-8
    sd = np.where(constant, 1., sd)
    median = np.zeros(x.shape[1])
    for column in np.flatnonzero(count):
        median[column] = np.median(x[idx[mask[idx, column]], column])
    valid = mask & (count[None] > 0)
    imputed = np.where(mask, x, median)
    scaled = np.where(valid, (imputed - mean) / sd, 0.)
    require(np.isfinite(scaled).all() and (scaled[~valid] == 0).all(), "RETENTION_MISSING_STORAGE_NOT_ZERO")
    state = dict(feature_mean=mean, feature_sd=sd, median_imputation=median, train_observed_count=count)
    audit = dict(dimension=x.shape[1], active_rows=len(x), training_rows=len(idx), validation_rows=len(x) - len(idx),
        training_index_sha256=array_hash(idx),
        train_unobserved_columns=int((count == 0).sum()), train_observed_count_min=int(count.min()),
        train_observed_count_max=int(count.max()), train_constant_or_small_SD_columns=int(constant.sum()),
        effective_training_feature_columns=int((np.ptp(scaled[idx], axis=0) > 1e-12).sum()),
        masked_entries=int((~valid).sum()), observed_feature_entries=int(valid.sum()),
        observed_feature_fraction=float(valid.mean()), train_only_fit=True, source_ST_scaler_used=False,
        missing_mask_preserved=True, scaler_state_sha256=canonical({k: array_hash(v) for k, v in state.items()}),
        penalty_unit="L2_in_this_training_standardized_representation_not_general_geneL2_equivalence")
    return scaled, valid, state, audit


def fixed_pool(values, measured, membership, *, mean=False):
    x, mask, member = np.asarray(values, np.float64), np.asarray(measured, bool), np.asarray(membership, bool)
    require(x.ndim == 2 and x.shape == mask.shape and member.ndim == 2 and member.shape[1] == x.shape[1]
            and np.isfinite(x[mask]).all(), "INVALID_FIXED_PATHWAY_POOL")
    count = mask.astype(np.int64) @ member.astype(np.int64).T
    total = np.where(mask, x, 0.) @ member.astype(np.float64).T
    if mean:
        total = total / np.maximum(count, 1)
    return np.where(count > 0, total, 0.), count > 0, count


def signed_program_mean(values, measured, edge_program, edge_gene, edge_sign, *, programs=4):
    x, mask = np.asarray(values, np.float64), np.asarray(measured, bool)
    ep, eg, sign = np.asarray(edge_program), np.asarray(edge_gene), np.asarray(edge_sign)
    require(x.ndim == 2 and x.shape == mask.shape and ep.shape == eg.shape == sign.shape
            and np.isfinite(x[mask]).all() and np.isin(sign, [-1, 1]).all(), "INVALID_SIGNED_PROGRAM_INPUT")
    output = np.zeros((len(x), programs)); valid = np.zeros(output.shape, bool)
    counts = np.zeros((len(x), programs, 2), np.int64)
    for k in range(programs):
        for part, direction in enumerate((1, -1)):
            columns = eg[(ep == k) & (sign == direction)]
            if not len(columns):
                continue
            observed = mask[:, columns]
            count = observed.sum(1); counts[:, k, part] = count
            output[:, k] += direction * np.where(observed, x[:, columns], 0.).sum(1) / np.maximum(count, 1)
            valid[:, k] |= count > 0
    return output, valid, counts


def frozen_source_features(source_model, gene_values, gene_mask, mapping, *, microbatch=128, device="cpu"):
    """Reencode own fold-scaled genes, never target-trained latent or ST scale."""
    import torch
    from src.bgb_en.r6_omics import GeneValueBank
    from src.bgb_en.r6_pathology import source_model_state_sha256
    require(source_model.arm == "S7" and not source_model.training
            and not any(p.requires_grad for p in source_model.parameters()), "FROZEN_COMPLETE_S7_REQUIRED")
    require(source_model.gene_reader.mapping.gene_symbols == mapping.gene_symbols
            and microbatch == 128, "RETENTION_SOURCE_MEMBERS_OR_MICROBATCH_CHANGED")
    fingerprint = source_model_state_sha256(source_model)
    bank = GeneValueBank(mapping.gene_symbols, mapping.gene_symbols).to(device)
    values, masks = [], []
    with torch.no_grad():
        for start in range(0, len(gene_values), microbatch):
            measurement = bank(torch.as_tensor(gene_values[start:start + microbatch], dtype=torch.float32, device=device),
                               torch.as_tensor(gene_mask[start:start + microbatch], dtype=torch.bool, device=device))
            encoded = source_model.encode_gene_programs(measurement)
            require(encoded["all_members_used"] is True and encoded["original_gene_values_reencoded_this_call"] is True,
                    "RETENTION_SOURCE_MUST_READ_ALL_ORIGINAL_MEMBERS")
            values.append(encoded["tokens"].cpu().numpy()); masks.append(encoded["program_mask"].cpu().numpy())
    require(source_model_state_sha256(source_model) == fingerprint, "RETENTION_CHANGED_FROZEN_SOURCE_STATE")
    tokens, valid = np.concatenate(values).astype(np.float64), np.concatenate(masks)
    require(tokens.shape[1:] == (4, 64) and np.isfinite(tokens).all(), "INVALID_FROZEN_S7_TOKENS")
    flat, flat_mask = tokens.reshape(len(tokens), 256), np.repeat(valid, 64, axis=1)
    mean = np.where(valid[..., None], tokens, 0.).sum(1) / np.maximum(valid.sum(1), 1)[:, None]
    mean_mask = np.broadcast_to(valid.any(1)[:, None], mean.shape).copy()
    return (flat, flat_mask), (mean, mean_mask), fingerprint


def exact_pool_alias(scaled):
    first, second = (scaled[name] for name in REPRESENTATIONS[:2])
    identical = np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
    return {name: REPRESENTATIONS[0] if identical and name == REPRESENTATIONS[1] else name for name in REPRESENTATIONS}


def matrix_efron_objective(x, t, e, coefficient, lam=1.):
    """Same full-riskset Efron objective used by the existing Ridge Cox."""
    from src.phase10_trimodal.math import efron_weights
    x, t, e, w = np.asarray(x, float), np.asarray(t, float), np.asarray(e, int), np.asarray(coefficient, float)
    require(x.ndim == 2 and len(x) == len(t) and t.shape == e.shape and w.shape == (x.shape[1],)
            and np.isfinite(x).all() and np.isfinite(w).all(), "INVALID_RETENTION_COX_INPUT")
    with np.errstate(divide="ignore"):
        loga = np.log(efron_weights(t, e))
    risk = x @ w
    scores = loga + risk[None]
    logden = logsumexp(scores, axis=1)
    dr = (np.exp(scores - logden[:, None]).sum(0) - e) / e.sum()
    loss = (logden.sum() - risk[e == 1].sum()) / e.sum() + lam / 2 * (w @ w)
    return float(loss), x.T @ dr + lam * w


def independent_efron_objective(x, t, e, coefficient, lam=1.):
    """Independent event-time/risk-set loops, without the training weight matrix."""
    x, t, e, w = np.asarray(x, float), np.asarray(t, float), np.asarray(e, int), np.asarray(coefficient, float)
    require(x.ndim == 2 and len(x) == len(t) and t.shape == e.shape and w.shape == (x.shape[1],)
            and np.isfinite(x).all() and np.isfinite(t).all() and np.isfinite(w).all()
            and np.isin(e, [0, 1]).all() and e.sum() > 0, "INVALID_INDEPENDENT_EFRON_INPUT")
    risk = x @ w
    value, gradient = -float(risk[e == 1].sum()), -e.astype(float)
    for point in np.unique(t[e == 1]):
        deaths = np.flatnonzero((t == point) & (e == 1)); active = np.flatnonzero(t >= point)
        maximum = risk[active].max()
        active_exp, death_exp = np.exp(risk[active] - maximum), np.exp(risk[deaths] - maximum)
        for step in range(len(deaths)):
            fraction = step / len(deaths)
            denominator = active_exp.sum() - fraction * death_exp.sum()
            require(np.isfinite(denominator) and denominator > 0, "INVALID_INDEPENDENT_EFRON_DENOMINATOR")
            value += np.log(denominator) + maximum
            gradient[active] += active_exp / denominator
            gradient[deaths] -= fraction * death_exp / denominator
    return float(value / e.sum() + lam / 2 * (w @ w)), x.T @ (gradient / e.sum()) + lam * w


class RetentionSolverError(RuntimeError):
    def __init__(self, audit, coefficient):
        super().__init__("RETENTION_SOLVER_FAILED_LOCKED_GRADIENT_CONTRACT_NO_RETRY")
        self.audit, self.coefficient = audit, coefficient


def finite_solver_audit(audit):
    if audit is None:
        return None
    return {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
            for key, value in audit.items()}


def fit_ridge_once(x, t, e):
    started = time.monotonic()
    objective = lambda w: matrix_efron_objective(x, t, e, w, 1.)
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
        options=dict(maxiter=2000, maxls=50, ftol=1e-15, gtol=1e-9))
    coefficient = np.asarray(result.x, np.float64)
    loss, gradient = objective(coefficient)
    audit = dict(actual_optimizer_fits=1, iterations=int(result.nit), function_evaluations=int(result.nfev),
        solver_success=bool(result.success), gradient_max=float(np.max(np.abs(gradient))), loss=loss,
        elapsed_seconds=time.monotonic() - started, solver="L-BFGS-B", lambda_L2=1., automatic_retry=False)
    if not np.isfinite(loss) or not np.isfinite(coefficient).all() or audit["gradient_max"] > 1e-6:
        raise RetentionSolverError(audit, coefficient)
    check, independent_gradient = independent_efron_objective(x, t, e, coefficient)
    if abs(check - loss) > 1e-10 or np.max(np.abs(independent_gradient)) > 1e-6:
        raise RetentionSolverError(dict(audit, independent_gradient_max=float(np.max(np.abs(independent_gradient)))), coefficient)
    return coefficient, dict(audit, independent_objective_error=abs(check - loss),
                             independent_gradient_max=float(np.max(np.abs(independent_gradient))))


def _reference_paths(args, spec):
    containers = dict(formal=args.primary_private, pca=args.primary_private.parent / "pca_geometry_v1",
                      pca_repair=args.primary_private.parent / "pca_geometry_v1/solver_repair_v1")
    require(spec["container"] in containers and spec["cell"] not in (".", "..")
            and "/" not in spec["cell"] and "\\" not in spec["cell"],
            "INVALID_HISTORICAL_REFERENCE_CONTAINER")
    directory = containers[spec["container"]] / "cells" / spec["cell"]
    return directory / "COMPLETE.json", directory / "model.pt"


def historical_references(args, ctx, origins, config):
    rows = []
    for spec in config["historical_references"]:
        receipt, artifact = _reference_paths(args, spec)
        require(sha256(receipt) == spec["receipt_sha256"] and sha256(artifact) == spec["artifact_sha256"],
                "HISTORICAL_REFERENCE_EXACT_SHA_CHANGED")
        saved = read(receipt)
        if spec["container"] == "formal":
            require(saved.get("lineage") == ctx["lineage"] and saved.get("inner_partition") == spec["partition"]
                    and saved.get("seed") == 31 and saved.get("stage") == "R2", "HISTORICAL_REFERENCE_LINEAGE_CHANGED")
        else:
            require(saved["identity"]["partition"] == spec["partition"] and saved["identity"]["dimension"] == 256,
                    "HISTORICAL_PCA256_SOURCE_CHANGED")
        key = f"outer1_inner{spec['partition']}"
        require(origins[key]["prepared_artifact_sha256"] == config["prepared_artifact_sha256"][key],
                "HISTORICAL_REFERENCE_PREPARED_SCOPE_CHANGED")
        rows.append(dict(representation=spec["reference"], partition=spec["partition"], status="REUSED_EXACT_PRIOR_ACTUAL_REPLAY",
            metrics=saved["metrics"], parameters=spec["parameters"], lambda_L2=spec["lambda_L2"],
            actual_optimizer_fits=0, historical_optimizer_fits_referenced_not_new=1,
            receipt_sha256=spec["receipt_sha256"], artifact_sha256=spec["artifact_sha256"],
            PCA256_is_RNA512_not_fullRNA=spec["container"] != "formal"))
    require(len(rows) == 9, "EXACT_NINE_HISTORICAL_CONVEX_REFERENCES_REQUIRED")
    return rows


def real_context(args, *, with_prepared=False):
    """All source gates are checked before the common target loader is called."""
    config = validate_recipe(read(args.config))
    gate = validate_launch_gate(args.launch_gate, args.launch_gate_sha256)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts import bgb_en_r6_common as common
    from scripts.bgb_en_run import load_inputs
    from src.bgb_en.r6_omics import load_source_program_mapping
    common.configure_deterministic_runtime()
    mapping = load_source_program_mapping(ROOT / "configs/sami_surv_stage_b_programs_v1.json")
    sources, audits = common.source_foundations(args, mapping)  # Requires all three real complete/replayed full-source fits.
    original = argparse.Namespace(assets=args.assets, config=ROOT / "configs/bgb_en/prototype_v1.json")
    assets, _, raw, member, lineage = load_inputs(original)
    for name, digest in config["canonical_sources"].items():
        require(lineage.get(name) == digest, "RETENTION_CANONICAL_SOURCE_LINEAGE_CHANGED")
    ctx = dict(assets=assets, raw=raw, member=member, lineage=lineage, lock=None,
               mapping=mapping, sources=sources, source_audits=audits, config=config, gate=gate)
    prepared, origins = {}, {}
    if with_prepared:
        for part in (0, 1, 2):
            bundle, origin = common.qualified_prepared(args, ctx, part)
            stem = f"outer1_inner{part}"
            require(origin["prepared_artifact_sha256"] == config["prepared_artifact_sha256"][stem]
                    and origin["prepared_receipt_sha256"] == config["prepared_receipt_sha256"][stem],
                    "RETENTION_REQUIRES_EXACT_ORIGINAL_THREE_PARTITIONS")
            prepared[part], origins[stem] = bundle, origin
    return ctx, prepared, origins


def _cache_path(root, representation, partition):
    require(representation in REPRESENTATIONS and partition in (0, 1, 2), "UNKNOWN_RETENTION_CELL")
    return Path(root) / "features" / f"p{partition}" / (representation + ".private.npz")


def prepare_feature_caches(args, ctx, prepared):
    import torch
    from src.bgb_en.r6_omics import GeneValueBank
    entries = []
    for part in (0, 1, 2):
        bundle = prepared[part]
        train, valid = np.asarray(bundle["train"]), np.asarray(bundle["valid"])
        active = np.r_[train, valid]; local_train = np.arange(len(train))
        rna, measured = np.asarray(bundle["features"]["rna"])[active], np.asarray(bundle["features"]["rna_mask"])[active]
        hall = ctx["member"].cpu().numpy()
        raw = {}
        for name, is_mean in zip(REPRESENTATIONS[:2], (False, True)):
            values, mask, _ = fixed_pool(rna, measured, hall, mean=is_mean); raw[name] = (values, mask)
        mapping = ctx["mapping"]
        bank = GeneValueBank(ctx["raw"]["rna_symbols"].tolist(), mapping.gene_symbols)
        with torch.no_grad():
            own = bank(torch.as_tensor(rna), torch.as_tensor(measured))
        genes, masks = own.values[..., 0].numpy().astype(np.float64), own.measured_mask[..., 0].numpy()
        raw[REPRESENTATIONS[2]] = (genes, masks)
        signed, signed_mask, _ = signed_program_mean(genes, masks, mapping.edge_program.numpy(),
            mapping.edge_gene.numpy(), mapping.edge_sign.numpy())
        raw[REPRESENTATIONS[3]] = (signed, signed_mask)
        flat, mean, source_sha = frozen_source_features(ctx["sources"]["MATCHED"], genes, masks, mapping, device=args.device)
        require(source_sha == ctx["source_audits"]["MATCHED"]["complete_source_state_sha256"], "RETENTION_FROZEN_SOURCE_ID_CHANGED")
        raw[REPRESENTATIONS[4]], raw[REPRESENTATIONS[5]] = flat, mean
        scaled = {name: masked_feature_scale(*raw[name], local_train) for name in REPRESENTATIONS}
        aliases = exact_pool_alias(scaled)
        for name in REPRESENTATIONS:
            x, mask, state, audit = scaled[name]
            require(x.shape[1] == DIMENSIONS[name], "RETENTION_REPRESENTATION_DIMENSION_CHANGED")
            path = _cache_path(args.private, name, part); path.parent.mkdir(parents=True, exist_ok=True)
            require(not path.exists(), "EXISTING_RETENTION_FEATURE_CACHE_PRESERVED")
            np.savez_compressed(path, features=x, measured_mask=mask, train_rows=train, valid_rows=valid,
                                active_rows=active, **state)
            entries.append(dict(representation=name, partition=part, cache_sha256=sha256(path),
                feature_values_sha256=array_hash(x), measured_mask_sha256=array_hash(mask),
                train_rows_sha256=array_hash(train), valid_rows_sha256=array_hash(valid),
                optimizer_representation=aliases[name], normalization=audit,
                gene_registry_union=760 if name not in REPRESENTATIONS[:2] else None,
                registered_RNA_union_overlap=int(bank._rna_present.sum()) if name not in REPRESENTATIONS[:2] else None,
                source_ST_scaler_used=False, raw_gene_values_reencoded_for_frozen_tokens=name in REPRESENTATIONS[4:]))
    return entries


def create_lock(args):
    root = private_root(args.private, readonly_roots=(args.source_private, args.primary_private))
    require(not (root / "retentionLOCK.json").exists() and not (root / "CACHE_PREPARATION_STARTED.json").exists(),
            "EXISTING_RETENTION_PREPARATION_PRESERVED_NO_AUTOMATIC_RETRY")
    ctx, prepared, origins = real_context(args, with_prepared=True)
    refs = historical_references(args, ctx, origins, ctx["config"])
    for name, digest in ctx["config"]["historical_reference_audit_sha256"].items():
        require(sha256(ROOT / name) == digest, "HISTORICAL_REPLAY_AUDIT_CHANGED")
    exclusive_json(root / "CACHE_PREPARATION_STARTED.json", dict(status="DATA_TRANSFORM_ONLY_ZERO_FITS",
        config_sha256=sha256(args.config), launch_gate_sha256=args.launch_gate_sha256))
    entries = prepare_feature_caches(args, ctx, prepared)
    config_name = args.config.resolve().relative_to(ROOT).as_posix()
    files = set(ctx["lineage"]["source"]) | set(SOURCE_FILES) | {config_name}
    lock = dict(schema="bgb_en_r6_retention_lock_v1", status="LOCKED_BEFORE_ALL_RETENTION_FITS",
        config_sha256=sha256(args.config), source_config_sha256=sha256(args.source_config),
        assets_config_sha256=sha256(args.assets), launch_gate_sha256=args.launch_gate_sha256,
        source_full_states={k: ctx["source_audits"][k] for k in ("MATCHED", "WRONG_DONOR", "WITHIN_SECTION_TISSUE_DEPTH")},
        data_fingerprints={k: ctx["raw"]["audit"][k] for k in ("cohort_sha256", "endpoints_sha256")},
        prepared_origins=origins, sources={name: sha256(ROOT / name) for name in sorted(files)},
        representations=entries, historical_references=refs, planned_logical_cells=18,
        planned_actual_optimizer_fits=sum(r["optimizer_representation"] == r["representation"] for r in entries),
        new_teacher_or_PCA_fits=0, outer_test_evaluated=False, no_source_ST_scaler_for_TCGA=True)
    lock["contract_sha256"] = canonical(lock)
    exclusive_json(root / "retentionLOCK.json", lock)
    return lock


def verify_lock(args):
    private_root(args.private, readonly_roots=(args.source_private, args.primary_private))
    validate_launch_gate(args.launch_gate, args.launch_gate_sha256)
    validate_recipe(read(args.config))
    lock = read(args.private / "retentionLOCK.json")
    require(lock.get("schema") == "bgb_en_r6_retention_lock_v1" and lock.get("status") == "LOCKED_BEFORE_ALL_RETENTION_FITS"
            and canonical({k: v for k, v in lock.items() if k != "contract_sha256"}) == lock["contract_sha256"],
            "RETENTION_IMMUTABLE_LOCK_CHANGED")
    require(lock["config_sha256"] == sha256(args.config) and lock["source_config_sha256"] == sha256(args.source_config)
            and lock["assets_config_sha256"] == sha256(args.assets) and lock["launch_gate_sha256"] == args.launch_gate_sha256,
            "RETENTION_CONFIG_OR_GATE_CHANGED")
    require(all(sha256(ROOT / name) == digest for name, digest in lock["sources"].items()), "RETENTION_LOCKED_SOURCE_BYTES_CHANGED")
    for control, origin in lock["source_full_states"].items():
        directory = args.source_private / "source_cells" / f"S7_{control}_h-1_seed31"
        for filename, field in (("state_epoch080.private.pt", "checkpoint_sha256"), ("COMPLETE.json", "completion_sha256"), ("REPLAY.json", "replay_sha256")):
            require(sha256(directory / filename) == origin[field], "RETENTION_FULL_SOURCE_FILE_CHANGED")
    for stem, origin in lock["prepared_origins"].items():
        for suffix, field in ((".json", "prepared_receipt_sha256"), (".pt", "prepared_artifact_sha256")):
            require(sha256(args.primary_private / "prepared" / (stem + suffix)) == origin[field], "RETENTION_PREPARED_BYTES_CHANGED")
    for row in lock["representations"]:
        require(sha256(_cache_path(args.private, row["representation"], row["partition"])) == row["cache_sha256"], "RETENTION_FEATURE_CACHE_CHANGED")
    require(len(lock["representations"]) == 18 and {(r["representation"], r["partition"]) for r in lock["representations"]}
            == {(name, part) for name in REPRESENTATIONS for part in (0, 1, 2)}, "RETENTION_LOCKED_MATRIX_CHANGED")
    for row in lock["representations"]:
        require(row["normalization"]["dimension"] == DIMENSIONS[row["representation"]]
                and (row["optimizer_representation"] == row["representation"] or
                     row["representation"] == REPRESENTATIONS[1] and row["optimizer_representation"] == REPRESENTATIONS[0]),
                "RETENTION_LOCKED_DIMENSION_OR_ALIAS_CHANGED")
    for spec in read(args.config)["historical_references"]:
        receipt, artifact = _reference_paths(args, spec)
        require(sha256(receipt) == spec["receipt_sha256"] and sha256(artifact) == spec["artifact_sha256"],
                "RETENTION_REUSED_HISTORICAL_REFERENCE_CHANGED")
    return lock


def load_cached(args, entry):
    with np.load(_cache_path(args.private, entry["representation"], entry["partition"]), allow_pickle=False) as archive:
        result = {key: np.array(archive[key], copy=True) for key in archive.files}
    require(array_hash(result["features"]) == entry["feature_values_sha256"]
            and array_hash(result["measured_mask"]) == entry["measured_mask_sha256"]
            and array_hash(result["train_rows"]) == entry["train_rows_sha256"]
            and array_hash(result["valid_rows"]) == entry["valid_rows_sha256"], "RETENTION_CACHE_ROW_OR_TRANSFORM_CHANGED")
    require(np.array_equal(result["active_rows"], np.r_[result["train_rows"], result["valid_rows"]])
            and not np.intersect1d(result["train_rows"], result["valid_rows"]).size,
            "RETENTION_CACHE_TRAIN_VALID_OVERLAP_OR_ORDER")
    return result


def _directory(args, name, partition):
    require(name in REPRESENTATIONS and partition in (0, 1, 2), "UNKNOWN_RETENTION_UNIT")
    return args.private / "cells" / f"{name}_p{partition}"


def _contract(lock, entry):
    return dict(retention_contract_sha256=lock["contract_sha256"], representation=entry["representation"],
                partition=entry["partition"], feature_values_sha256=entry["feature_values_sha256"],
                measured_mask_sha256=entry["measured_mask_sha256"], optimizer_representation=entry["optimizer_representation"])


def _model_path(args, entry):
    return _directory(args, entry["optimizer_representation"], entry["partition"]) / "model.private.npz"


def validate_completion(args, lock, entry, value):
    alias = entry["optimizer_representation"] != entry["representation"]
    require(value.get("contract") == _contract(lock, entry) and value.get("status") == (
        "REUSED_EXACT_REPRESENTATION_FIT" if alias else "COMPLETE_RETENTION_FIT")
        and value.get("actual_optimizer_fits") == (0 if alias else 1) and value.get("lambda_L2") == 1.
        and sha256(_model_path(args, entry)) == value.get("artifact_sha256"), "RETENTION_COMPLETION_LINEAGE_OR_ARTIFACT_CHANGED")
    if alias:
        source = _directory(args, entry["optimizer_representation"], entry["partition"]) / "COMPLETE.json"
        require(sha256(source) == value.get("source_complete_sha256"), "RETENTION_ALIAS_SOURCE_RECEIPT_CHANGED")


def validate_training_labels(args, entry, cache, raw, value):
    """Alias fits inherit the original solver's exact training-label receipt."""
    expected = array_hash(np.c_[raw["t"][cache["train_rows"]], raw["e"][cache["train_rows"]]])
    owner = value
    if entry["optimizer_representation"] != entry["representation"]:
        owner = read(_directory(args, entry["optimizer_representation"], entry["partition"]) / "COMPLETE.json")
    require(owner.get("target_training_label_sha256") == expected, "RETENTION_EXACT_TRAINING_LABEL_BYTES_CHANGED")


def _evaluate(raw, cache, risk):
    from src.bgb_en.survival import evaluate
    train, valid = cache["train_rows"], cache["valid_rows"]
    return evaluate(raw["t"][train], raw["e"][train], raw["t"][valid], raw["e"][valid], risk)


def fit_unit(args, lock, entry, raw):
    directory = _directory(args, entry["representation"], entry["partition"])
    receipt = directory / "COMPLETE.json"; contract = _contract(lock, entry)
    if receipt.exists():
        previous = read(receipt)
        require(args.resume, "RETENTION_COMPLETE_REUSE_CHANGED_OR_NOT_EXPLICIT")
        validate_completion(args, lock, entry, previous)
        validate_training_labels(args, entry, load_cached(args, entry), raw, previous)
        return previous
    require(not directory.exists(), "RETENTION_INCOMPLETE_OR_FAILED_CELL_PRESERVED_NO_RETRY")
    directory.mkdir(parents=True)
    exclusive_json(directory / "START.json", dict(contract=contract, optimizer_started=False))
    started = False; audit = None
    try:
        cache = load_cached(args, entry); x = cache["features"]; n = len(cache["train_rows"])
        if entry["optimizer_representation"] != entry["representation"]:
            original = next(r for r in lock["representations"] if r["representation"] == entry["optimizer_representation"] and r["partition"] == entry["partition"])
            other = load_cached(args, original)
            require(np.array_equal(x, other["features"]) and np.array_equal(cache["measured_mask"], other["measured_mask"]),
                    "RETENTION_ALIAS_NOT_EXACT_NO_SCORE_BASED_REUSE")
            source = read(_directory(args, original["representation"], original["partition"]) / "COMPLETE.json")
            require(source.get("contract") == _contract(lock, original) and source.get("status") == "COMPLETE_RETENTION_FIT"
                    and sha256(_model_path(args, entry)) == source["artifact_sha256"], "RETENTION_ALIAS_SOURCE_NOT_COMPLETE")
            validate_training_labels(args, entry, cache, raw, source)
            with np.load(_model_path(args, entry), allow_pickle=False) as saved:
                risk = saved["validation_risk"]
            result = dict(status="REUSED_EXACT_REPRESENTATION_FIT", contract=contract, actual_optimizer_fits=0,
                optimizer_fits_reused=1, source_complete_sha256=sha256(_directory(args, original["representation"], original["partition"]) / "COMPLETE.json"),
                artifact_sha256=source["artifact_sha256"], metrics=_evaluate(raw, cache, risk),
                parameters=x.shape[1], lambda_L2=1., solver_iterations=0)
        else:
            started = True
            coef, audit = fit_ridge_once(x[:n], raw["t"][cache["train_rows"]], raw["e"][cache["train_rows"]])
            tr, va = x[:n] @ coef, x[n:] @ coef
            np.savez_compressed(directory / "model.private.npz", coefficient=coef, train_risk=tr,
                validation_risk=va, train_rows=cache["train_rows"], valid_rows=cache["valid_rows"])
            result = dict(status="COMPLETE_RETENTION_FIT", contract=contract, actual_optimizer_fits=1,
                artifact_sha256=sha256(directory / "model.private.npz"), metrics=_evaluate(raw, cache, va),
                parameters=x.shape[1], lambda_L2=1., solver_iterations=audit["iterations"], solver=audit,
                target_training_label_sha256=array_hash(np.c_[raw["t"][cache["train_rows"]], raw["e"][cache["train_rows"]]]))
        exclusive_json(receipt, result)
        return result
    except Exception as error:
        if isinstance(error, RetentionSolverError):
            audit = error.audit
            if np.isfinite(error.coefficient).all():
                np.savez_compressed(directory / "failed_coefficient.private.npz", coefficient=error.coefficient)
        exclusive_json(directory / "ERROR.json", dict(status="ERROR_RETENTION_PRESERVED_NO_AUTOMATIC_RETRY",
            contract=contract, actual_optimizer_fits=int(started), known_solver_audit=finite_solver_audit(audit),
            error_class=type(error).__name__, scientific_result_accepted=False))
        raise


def replay_unit(args, lock, entry, raw):
    directory = _directory(args, entry["representation"], entry["partition"])
    done = read(directory / "COMPLETE.json")
    validate_completion(args, lock, entry, done)
    cache = load_cached(args, entry); x = cache["features"]; n = len(cache["train_rows"])
    validate_training_labels(args, entry, cache, raw, done)
    with np.load(_model_path(args, entry), allow_pickle=False) as saved:
        coef = saved["coefficient"]
        require(np.array_equal(saved["train_rows"], cache["train_rows"]) and np.array_equal(saved["valid_rows"], cache["valid_rows"]),
                "RETENTION_REPLAY_TRAIN_VALID_ROWS_CHANGED")
        error = max(float(np.max(np.abs(x[:n] @ coef - saved["train_risk"]))),
                    float(np.max(np.abs(x[n:] @ coef - saved["validation_risk"]))))
    require(error <= 1e-12, "RETENTION_SAVED_COEFFICIENT_FORWARD_FAILED")
    loss, grad = independent_efron_objective(x[:n], raw["t"][cache["train_rows"]], raw["e"][cache["train_rows"]], coef)
    require(np.isfinite(loss) and np.max(np.abs(grad)) <= 1e-6, "RETENTION_INDEPENDENT_EFRON_GRADIENT_FAILED")
    training_value, _ = matrix_efron_objective(x[:n], raw["t"][cache["train_rows"]], raw["e"][cache["train_rows"]], coef)
    require(abs(training_value - loss) <= 1e-10, "RETENTION_INDEPENDENT_OBJECTIVE_FAILED")
    metric = _evaluate(raw, cache, x[n:] @ coef)
    for key in ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den", "Harrell", "Uno60"):
        require(metric[key] == done["metrics"][key], "RETENTION_INDEPENDENT_METRIC_REPLAY_FAILED")
    result = dict(status="PASS_RETENTION_COEFFICIENT_AND_INDEPENDENT_EFRON_REPLAY", contract=done["contract"],
        complete_sha256=sha256(directory / "COMPLETE.json"), risk_max_abs_error=error,
        independent_gradient_max=float(np.max(np.abs(grad))), independent_objective_error=abs(training_value - loss),
        metrics=metric, new_optimizer_fits=0, parameters=len(coef), lambda_L2=1.)
    atomic_json(directory / "REPLAY.json", result)
    return result


def write_report(args, lock):
    require(len(lock["representations"]) == 18 and {(r["representation"], r["partition"]) for r in lock["representations"]}
            == {(name, part) for name in REPRESENTATIONS for part in (0, 1, 2)}, "RETENTION_REPORT_EXACT_EIGHTEEN_CELLS_REQUIRED")
    output = ROOT / "reports/bgb_en_r6"; output.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in lock["representations"]:
        directory = _directory(args, entry["representation"], entry["partition"])
        row = dict(representation=entry["representation"], partition=entry["partition"], status="MISSING",
            actual_optimizer_fits=None, parameters=entry["normalization"]["dimension"], lambda_L2=1.,
            optimizer_representation=entry["optimizer_representation"], normalization=entry["normalization"])
        if (directory / "COMPLETE.json").is_file():
            done = read(directory / "COMPLETE.json")
            validate_completion(args, lock, entry, done)
            row.update(status="FIT_COMPLETE_REPLAY_MISSING", actual_optimizer_fits=done["actual_optimizer_fits"],
                       iterations=done.get("solver_iterations"), receipt_sha256=sha256(directory / "COMPLETE.json"))
            if (directory / "REPLAY.json").is_file():
                replay = read(directory / "REPLAY.json")
                require(replay.get("contract") == done["contract"] and replay.get("complete_sha256") == row["receipt_sha256"]
                        and replay.get("status") == "PASS_RETENTION_COEFFICIENT_AND_INDEPENDENT_EFRON_REPLAY"
                        and replay.get("new_optimizer_fits") == 0 and replay.get("risk_max_abs_error", 1.) <= 1e-12
                        and replay.get("independent_gradient_max", 1.) <= 1e-6, "RETENTION_REPORT_INDEPENDENT_REPLAY_CHANGED")
                row.update(status="COMPLETE_AND_REPLAYED", metrics=replay["metrics"], replay_sha256=sha256(directory / "REPLAY.json"))
        elif (directory / "ERROR.json").is_file():
            error = read(directory / "ERROR.json")
            require(error.get("contract") == _contract(lock, entry), "RETENTION_ERROR_RECEIPT_LINEAGE_CHANGED")
            row.update(status="ERROR_NO_RETRY", actual_optimizer_fits=error["actual_optimizer_fits"], error_class=error["error_class"],
                       iterations=(error.get("known_solver_audit") or {}).get("iterations"))
        rows.append(row)
    rows += lock["historical_references"]
    summaries = []
    for name in [*REPRESENTATIONS, *sorted({r["representation"] for r in lock["historical_references"]})]:
        cells = [r for r in rows if r["representation"] == name]
        good = [r for r in cells if r["status"] in ("COMPLETE_AND_REPLAYED", "REUSED_EXACT_PRIOR_ACTUAL_REPLAY")]
        summary = dict(representation=name, complete_partitions=len(good), expected_partitions=3,
                       status="COMPLETE" if len(good) == 3 else "INCONCLUSIVE", Harrell=None, Uno60=None)
        if len(good) == 3:
            metric = {key: float(sum(r["metrics"][key] for r in good)) for key in ("Harrell_num", "Harrell_den", "Uno_num", "Uno_den")}
            summary.update(Harrell=metric["Harrell_num"] / metric["Harrell_den"] if metric["Harrell_den"] else None,
                           Uno60=metric["Uno_num"] / metric["Uno_den"] if metric["Uno_den"] else None, counts=metric)
        summaries.append(summary)
    report = dict(schema="bgb_en_r6_retention_report_v1", retention_contract_sha256=lock["contract_sha256"],
        status="COMPLETE_18_LOGICAL_CELLS_AND_REPLAY" if all(r["status"] == "COMPLETE_AND_REPLAYED" for r in rows[:18]) else "INCONCLUSIVE_PARTIAL",
        planned_logical_cells=18, planned_actual_fits=lock["planned_actual_optimizer_fits"],
        confirmed_new_optimizer_fits=sum(r["actual_optimizer_fits"] for r in rows[:18] if r["actual_optimizer_fits"] is not None),
        unknown_fit_count_cells=sum(r["actual_optimizer_fits"] is None for r in rows[:18]), rows=rows, summaries=summaries,
        source_model_is_fixed_MATCHED_full_seed31=True, source_ST_scaler_used_for_TCGA=False,
        old_Set_Mamba_are_historical_supervised_bypass_diagnostics_only=True, new_Set_or_Mamba_training=0,
        general_geneL2_equivalence_claimed=False, historical_PCA256_is_RNA512_not_fullRNA=True,
        no_cross_partition_risk_concatenation=True, new_teacher_or_PCA_fits=0,
        DSS_conditional_increment_or_biological_source_truth_established=False,
        patient_risks_coefficients_genes_coordinates_or_runtime_paths_exported=False)
    atomic_json(output / "R6_2_OMICS_RETENTION.json", report)
    columns = ["representation", "partition", "status", "actual_optimizer_fits", "iterations", "parameters", "lambda_L2", "optimizer_representation",
               "Harrell", "Uno60", "Harrell_num", "Harrell_den", "Uno_num", "Uno_den"]
    with (output / "R6_2_OMICS_RETENTION.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row.get("metrics", {})})
    lines = ["# R6.2 RNA表征信息保真", "", f"状态：**{report['status']}**。逻辑18单元，确认新增凸拟合{report['confirmed_new_optimizer_fits']}。", "",
        "旧RNA512/fullRNA/PCA256按精确SHA及历史实际重放引用，0重拟合。新表征先用当前训练分区拟合观测尺度，未测值mask保留且数值存0。", "",
        "| 表示 | 完整分区 | Harrell | Uno60 |", "|---|---:|---:|---:|"]
    for row in summaries:
        fmt = lambda x: "—" if x is None else f"{x:.6f}"
        lines.append(f"| {row['representation']} | {row['complete_partitions']}/3 | {fmt(row['Harrell'])} | {fmt(row['Uno60'])} |")
    lines += ["", "所有新Cox固定lambda1，无超参搜索；sum/mean只在标准化后全部训练/验证值与mask逐元素相同时复用一次拟合。", "",
        "PCA256旧gene-L2证明只适用于RNA512的正交未whiten坐标。新非线性token的同lambda不代表相同gene-L2几何。", "",
        "优势是分离支持集、固定聚合与冻结源编码；风险是既有队列反复开发及读出正则几何。结果不能单独证明Clinical+Pathology条件增量或细胞来源。", ""]
    (output / "R6_2_OMICS_RETENTION.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "private", "primary-private", "source-private", "config", "source-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--launch-gate", type=Path, required=True)
    parser.add_argument("--launch-gate-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mode", choices=("lock", "fit", "replay", "report"), required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS)
    parser.add_argument("--partition", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    if args.mode == "lock":
        result = create_lock(args)
        print(json.dumps(dict(status=result["status"], planned_actual_fits=result["planned_actual_optimizer_fits"], new_optimizer_fits=0)))
        return
    lock = verify_lock(args)
    if args.mode == "report":
        result = write_report(args, lock)
        print(json.dumps(dict(status=result["status"], confirmed_new_optimizer_fits=result["confirmed_new_optimizer_fits"])))
        return
    ctx, _, _ = real_context(args)
    require(all(lock["data_fingerprints"][k] == ctx["raw"]["audit"][k] for k in lock["data_fingerprints"]), "RETENTION_DATA_CHANGED")
    cells = [r for r in lock["representations"] if (args.representation is None or args.representation == r["representation"])
             and (args.partition is None or args.partition == r["partition"])]
    completed = []
    for entry in cells:
        result = fit_unit(args, lock, entry, ctx["raw"]) if args.mode == "fit" else replay_unit(args, lock, entry, ctx["raw"])
        completed.append(dict(representation=entry["representation"], partition=entry["partition"], status=result["status"]))
    print(json.dumps(dict(status="COMPLETED_REQUESTED_RETENTION_ACTION", mode=args.mode, units=completed)))


if __name__ == "__main__":
    main()

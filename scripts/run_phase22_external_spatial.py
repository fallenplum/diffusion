"""Qualified, finite A0/A1/A2 external-ST fits. Never starts TCGA survival.

Prepare one locked dictionary with --prepare-only, then run nine independent
--arm/--seed cells. --smoke performs one in-memory update and writes nothing.
All outputs are private runtime artifacts; never publish checkpoints or arrays.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.phase22_spatial_bridge.molecular_dictionary import (
    DictionaryTrainingContract, counts_sha256, sequence_sha256, fit_molecular_dictionary,
)
from src.phase22_spatial_bridge.measurement_scale import (
    SCALE_PROTOCOL_SHA, SCALE_SOURCE, MeasurementScaleTrainingContract,
    fit_train_measurement_scale, verify_measurement_scale,
)
from src.phase22_spatial_bridge.spatial_encoder import LinearCompositionEncoder, LocalSpatialEncoder
from src.phase22_spatial_bridge.strank_objective import spatial_objective

ARMS = ("A0", "A1", "A2")
SEEDS = (31, 47, 83)
WEIGHT_SHA = "6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e"
MODEL_REVISION = "d517a8dd47902dd7c308b3c36f63bce47e7b9a43"
DICTIONARY_SCHEMA = "phase22_train_dictionary_with_discrepancy_scale_v2"
SCHEDULE = dict(max_epochs=100, min_epochs=10, patience=15, learning_rate=.0003,
                weight_decay=.01, regions_per_donor=8, region_spots=256,
                radius_um=300., neighbor_cap=16, threads=2,
                relative_training_domain="sampled_256_spot_same_section_region",
                relative_validation_domain="full_section",
                epoch_selection="minimum_validation_donor_equal_arm_objective")


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def append_journal(path, value):
    """Private append-only attempt/update evidence, including resumed work."""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def source_hashes():
    paths = ["scripts/run_phase22_external_spatial.py", "scripts/evaluate_phase22_external_spatial.py",
             "src/phase22_spatial_bridge/spatial_encoder.py",
             "src/phase22_spatial_bridge/molecular_dictionary.py",
             "src/phase22_spatial_bridge/measurement_scale.py",
             "src/phase22_spatial_bridge/strank_objective.py"]
    return {p: file_sha256(REPO / p) for p in paths}


def private_output(path):
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO / "reports") or resolved.is_relative_to(REPO / "src"):
        raise ValueError("patient/spot-level runtime outputs cannot be stored in public source/report directories")
    return resolved


@dataclass
class Section:
    section_id: str
    donor_id: str
    study_id: str
    split: str
    assay: str
    counts: np.ndarray
    gene_names: np.ndarray
    coords_um: np.ndarray
    barcodes: np.ndarray
    measured: np.ndarray
    features: np.ndarray | None
    evidence: dict


def qualification_rows(data_root, split_lock, qualification, *, roles, with_features, section_ids=None):
    root = Path(data_root)
    lock, marker = read_json(split_lock), read_json(qualification)
    lock_sha = file_sha256(split_lock)
    if marker.get("schema_version") != "phase22_external_st_qualification_v1" or marker.get("qualification_status") != "PASS_EXTERNAL_ST_DATA_CONTRACT":
        raise ValueError("external-ST data qualification has not passed")
    if marker.get("locked_split_sha") != lock_sha:
        raise ValueError("qualification does not bind the exact study/donor split lock")
    rows = lock["rows"]
    if len(rows) != 16 or len({r["section_id_private"] for r in rows}) != 16:
        raise ValueError("finite formal CRC16 lock requires exactly 16 unique sections")
    donor_roles = {}
    for row in rows:
        donor, role = row["donor_id_private"], row["split"]
        if role not in {"train", "validation", "test"} or (donor in donor_roles and donor_roles[donor] != role):
            raise ValueError("a donor crosses study/donor partitions or has an unknown split")
        donor_roles[donor] = role
    ready = marker["sections_ready"]
    if not isinstance(ready, list) or len(ready) != 16:
        raise ValueError("all finite raw sections must be data-qualified")
    evidence = {r["section_id_private"]: r for r in ready}
    if set(evidence) != {r["section_id_private"] for r in rows}:
        raise ValueError("qualified section set does not match locked finite cohort")
    result = []
    if section_ids is not None and not set(section_ids).issubset({r["section_id_private"] for r in rows if r["split"] in roles}):
        raise ValueError("requested sections fall outside the locked role")
    for row in rows:
        if row["split"] not in roles:
            continue
        if section_ids is not None and row["section_id_private"] not in section_ids:
            continue
        section = row["section_id_private"]
        item = evidence[section]
        for key in ("section_id_private", "study_id", "donor_id_private", "split", "assay"):
            if item.get(key) != row[key]:
                raise ValueError("qualified source identity disagrees with frozen split")
        counts_path = root / "processed/sections" / (section + ".npz")
        if file_sha256(counts_path) != item["npz_sha256"]:
            raise ValueError("processed count/coordinate file SHA mismatch")
        metadata_path = root / item["metadata_path"]
        if file_sha256(metadata_path) != item["metadata_sha256"]:
            raise ValueError("processed physical/measurement qualification metadata SHA mismatch")
        section_qualification = read_json(metadata_path)
        if (section_qualification.get("qualification_status") != "PASS_SECTION_DATA_CONTRACT" or
            section_qualification.get("section_id_private") != section or
            section_qualification.get("coordinate_unit") != "um" or
            section_qualification.get("npz_sha256") != item["npz_sha256"] or
            section_qualification.get("cache_signature", {}).get("locked_split_sha") != lock_sha or
            section_qualification.get("source_files", {}).get("patches", {}).get("sha256") != item["patch_sha256"]):
            raise ValueError("section physical/measurement qualification lineage mismatch")
        footprint = section_qualification["audit"]["image_footprint"]
        actual_view = float(footprint["actual_native_crop_view_um"])
        source_mpp = float(section_qualification["audit"]["coordinates"]["mpp_x"])
        if (not np.isfinite(actual_view) or not np.isfinite(source_mpp) or source_mpp <= 0 or
            abs(actual_view - 112.) > .5 * source_mpp + 1e-6 or footprint["target_pixels"] != 224):
            raise ValueError("physical image footprint violates the finite source-pixel rounding contract")
        with np.load(counts_path, allow_pickle=False) as data:
            counts = data["counts"].copy()
            genes = data["gene_names"].astype(str)
            coords = data["coords_um"].astype(np.float64)
            barcodes = data["barcodes"].astype(str)
            measured = data["measured_mask"].astype(bool)
        if counts.ndim != 2 or counts.shape != (len(barcodes), len(genes)) or measured.shape != (len(genes),):
            raise ValueError("count/source array dimensions disagree")
        if coords.shape != (len(counts), 2) or len(np.unique(barcodes)) != len(barcodes) or len(set(genes)) != len(genes):
            raise ValueError("malformed physical coordinate/barcode/gene contract")
        if not np.isfinite(counts).all() or np.any(counts < 0) or not np.array_equal(counts, counts.round()):
            raise ValueError("processed expression is not nonnegative raw integer counts")
        if not np.isfinite(coords).all():
            raise ValueError("physical coordinates must be finite")
        features = None
        item = dict(item, source_counts_npz_sha256=file_sha256(counts_path))
        if with_features:
            feature_path = root / "features/uni2-h" / (section + ".npz")
            feature_receipt_path = feature_path.with_suffix(".json")
            receipt = read_json(feature_receipt_path)
            expected = {"schema_version": "phase22_uni2_feature_section_v1", "status": "COMPLETE",
                        "section_id_private": section, "model": "UNI2-h", "model_sha256": WEIGHT_SHA,
                        "model_revision": MODEL_REVISION, "counts_npz_sha256": item["npz_sha256"],
                        "split_lock_sha256": lock_sha, "feature_dimension": 1536,
                        "image_view_um_nominal": 112., "image_mpp_nominal": .5, "image_pixels": 224,
                        "source_patch_sha256": item["patch_sha256"], "frozen": True,
                        "qualification_metadata_sha256": item["metadata_sha256"],
                        "actual_native_crop_view_um": actual_view, "exact_112um": footprint["exact_112um"],
                        "extractor_sha256": file_sha256(REPO / "scripts/extract_phase22_uni2.py"),
                        "normalization": "official_RGB_uint8_div255_ImageNet_mean_std; no_resize_for_224square"}
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise ValueError("frozen UNI2 extraction/footprint receipt mismatch")
            if receipt.get("features_sha256") != file_sha256(feature_path):
                raise ValueError("frozen UNI2 feature cache SHA mismatch")
            if receipt.get("precision") not in {"fp32", "bf16"}:
                raise ValueError("unqualified UNI2 feature precision")
            if receipt["precision"] == "bf16":
                parity = receipt.get("precision_parity", {})
                stats = [parity.get("cosine_min"), parity.get("cosine_mean"), parity.get("relative_L2_max")]
                if (parity.get("status") != "PASS_BF16_FP32_PARITY" or
                    any(type(v) not in (int, float) or not np.isfinite(v) for v in stats) or
                    stats[0] < .995 or stats[1] < .999 or stats[2] > .05):
                    raise ValueError("UNI2 BF16 numerical parity qualification failed")
            elif receipt.get("precision_parity", {}).get("status") != "FP32_NO_REDUCED_PRECISION_GATE":
                raise ValueError("FP32 extraction must explicitly register its full-precision qualification")
            with np.load(feature_path, allow_pickle=False) as visual:
                features = visual["features"].astype(np.float32)
                if not np.array_equal(visual["barcodes"].astype(str), barcodes) or not np.array_equal(visual["coords_um"], coords):
                    raise ValueError("UNI2 features are not elementwise paired to qualified counts/coordinates")
            if features.shape != (len(counts), 1536) or not np.isfinite(features).all():
                raise ValueError("invalid frozen UNI2 feature values")
            item.update(features_sha256=receipt["features_sha256"],
                        features_path=str(feature_path), feature_receipt_sha256=file_sha256(feature_receipt_path))
        result.append(Section(section, row["donor_id_private"], row["study_id"], row["split"],
                              row["assay"], counts, genes, coords, barcodes, measured, features, item))
    if not result:
        raise ValueError("no qualified sections in requested locked role")
    return result, lock_sha, donor_roles


def align_genes(section, gene_names):
    lookup = {g: i for i, g in enumerate(section.gene_names)}
    aligned = np.zeros((len(section.counts), len(gene_names)), dtype=np.float64)
    measured = np.zeros(len(gene_names), dtype=bool)
    for j, gene in enumerate(gene_names):
        if gene in lookup and section.measured[lookup[gene]]:
            aligned[:, j] = section.counts[:, lookup[gene]]
            measured[j] = True
    if not measured.any():
        raise ValueError("section has no measured dictionary genes")
    return aligned, measured


def oracle_coefficients(counts, dictionary, *, max_iter=1000, tolerance=1e-8):
    """Fixed-D KL composition oracle; true expression used only where declared.

    This is an expression-reconstruction reference, not a bound on Spearman,
    and its coefficients must never become image-predictor inputs.
    """
    y = np.asarray(counts, dtype=np.float64)
    d = np.asarray(dictionary, dtype=np.float64)
    mass = y.sum(1)
    positive = mass > 0
    target = np.divide(y, mass[:, None], out=np.zeros_like(y), where=mass[:, None] > 0)
    pi = np.full((len(y), len(d)), 1. / len(d), dtype=np.float64)
    previous = np.inf
    converged = False
    objective = np.nan
    for iteration in range(1, max_iter + 1):
        mu = np.maximum(pi @ d, np.finfo(np.float64).tiny)
        ratio = target / mu
        update = pi * (ratio @ d.T)
        denom = update.sum(1, keepdims=True)
        update = np.divide(update, denom, out=pi.copy(), where=denom > 0)
        update[~positive] = pi[~positive]
        pi = update
        if iteration % 10 == 0 or iteration == max_iter:
            mu = np.maximum(pi @ d, np.finfo(np.float64).tiny)
            p = target[positive]
            m = mu[positive]
            term = np.where(p > 0, p * (np.log(np.maximum(p, 1e-300)) - np.log(m)), 0.)
            objective = float(term.sum(1).mean()) if positive.any() else 0.
            if np.isfinite(previous) and abs(previous - objective) <= tolerance * max(1., abs(previous)):
                converged = True
                break
            previous = objective
    return pi, {"iterations": iteration, "converged": converged, "mean_KL": objective,
                "zero_profiles": int((~positive).sum()), "expression_oracle_not_image_inference": True}


def dictionary_paths(output_root):
    folder = Path(output_root) / "dictionary"
    return folder / "dictionary_private.npz", folder / "DICT_RECEIPT_PRIVATE.json"


def prepare_dictionary(args):
    sections, lock_sha, donor_roles = qualification_rows(args.data_root, args.split_lock, args.qualification,
                                                        roles=("train",), with_features=False)
    target_record = read_json(args.target_measured_genes)
    target = target_record["gene_symbols_private"]
    common = set(target)
    for section in sections:
        common &= set(section.gene_names[section.measured])
    genes = sorted(common)
    if not genes:
        raise ValueError("no common measured training/target genes")
    arrays, donors = [], []
    for section in sections:
        count, measured = align_genes(section, genes)
        if not measured.all():
            raise ValueError("dictionary input unexpectedly contains an unmeasured training gene")
        arrays.append(count.astype(np.int64))
        donors.extend([section.donor_id] * len(count))
    counts = np.concatenate(arrays)
    contract = DictionaryTrainingContract("ST-train", lock_sha, counts_sha256(counts), sequence_sha256(genes),
        sequence_sha256(donors), sequence_sha256(genes), sequence_sha256(sorted(target)), tuple(sorted(donor_roles.items())))
    path, receipt_path = dictionary_paths(args.output_root)
    identity = dict(split_lock_sha256=lock_sha, data_qualification_sha256=file_sha256(args.qualification),
                    target_measurement_file_sha256=file_sha256(args.target_measured_genes),
                    input_count_sha256=contract.counts_sha256, source_hashes=source_hashes(), seed=31,
                    dictionary_schema_version=DICTIONARY_SCHEMA,
                    measurement_scale_protocol_sha256=SCALE_PROTOCOL_SHA)
    if path.exists() or receipt_path.exists():
        if not path.exists() or not receipt_path.exists():
            raise ValueError("partial dictionary artifact exists; do not silently refit/overwrite")
        old = read_json(receipt_path)
        if old.get("identity") != identity or old.get("dictionary_npz_sha256") != file_sha256(path):
            raise ValueError("dictionary identity/source/input drift; locked dictionary cannot be overwritten")
        print("DICTIONARY_REUSED_VERIFIED", flush=True)
        return
    fit = fit_molecular_dictionary(counts, genes, donors, st_measured_genes=genes,
        target_measured_genes=target, contract=contract, seed=31, n_states=32, gene_cap=1024,
        max_iter=1000)
    def qualification_failed(stage, audit):
        failure_path = path.with_name("DICTIONARY_QUALIFICATION_FAILED_PRIVATE.json")
        failure = dict(status="FAIL_DICTIONARY_QUALIFICATION", failed_stage=stage,
                       schema_version=DICTIONARY_SCHEMA, identity=identity,
                       dictionary_fit_audit=fit.metadata, failed_stage_audit=audit,
                       qualified_dictionary_or_model_written=False, formal_spatial_model_fits=0)
        if failure_path.exists() and read_json(failure_path) != failure:
            raise ValueError("existing dictionary failure receipt differs; preserve it and use a new private output directory")
        write_json(failure_path, failure)
    if not fit.metadata["converged"]:
        qualification_failed("KL_NMF_DICTIONARY", {"reason": "not_converged_under_locked_budget"})
        raise ValueError("dictionary did not converge under locked budget; qualification cannot silently pass")
    # Balanced training profiles only. This auxiliary fixed-D solve determines
    # the train-only population prior u0, never a heldout patient's state.
    sampled = fit.sampled_spot_indices_private
    selected_counts = counts[sampled][:, fit.selected_gene_indices_private]
    pi, oracle_audit = oracle_coefficients(selected_counts, fit.dictionary)
    if oracle_audit.get("converged") is not True:
        qualification_failed("TRAIN_FIXED_DICTIONARY_ORACLE", oracle_audit)
        raise ValueError("training dictionary oracle did not converge; u0/error scale cannot qualify")
    sampled_donors = np.asarray(donors)[sampled]
    u0 = np.mean([pi[sampled_donors == donor].mean(0) for donor in sorted(set(sampled_donors))], axis=0)
    u0 = np.maximum(u0, 1e-8)
    u0 /= u0.sum()
    scale_contract = MeasurementScaleTrainingContract("ST-train", lock_sha,
        counts_sha256(selected_counts), sequence_sha256(sampled_donors.tolist()),
        counts_sha256(fit.dictionary), tuple(sorted(donor_roles.items())))
    try:
        scale = fit_train_measurement_scale(selected_counts, pi, fit.dictionary, sampled_donors.tolist(),
                                            oracle_converged=True, contract=scale_contract)
    except ValueError as error:
        qualification_failed("TRAIN_DISCREPANCY_SCALE", {"reason": str(error), "oracle_audit": oracle_audit})
        raise
    omega = scale.omega
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, D=fit.dictionary, gene_names=np.asarray(fit.gene_names_private),
                        selected_gene_indices=fit.selected_gene_indices_private,
                        sampled_spot_indices=sampled, u0=u0, omega=omega,
                        sigma=scale.sigma, scale_residual_rms=scale.residual_rms,
                        scale_within_variance=scale.within_variance)
    temporary.replace(path)
    receipt = dict(fit.metadata, identity=identity, schema_version=DICTIONARY_SCHEMA,
                   dictionary_npz_sha256=file_sha256(path), u0=u0.tolist(), omega=omega.tolist(),
                   u0_source="donor_equal_fixed_dictionary_KL_coefficients_on_balanced_ST_train_profiles",
                   u0_oracle_audit=oracle_audit, u0_positive_floor=1e-8,
                   omega_source=SCALE_SOURCE, omega_calibration=scale.metadata,
                   labels_read="external_ST_training_counts_only_no_DSS_no_PFI")
    receipt["status"] = "PASS_TRAIN_ONLY_DICTIONARY"
    write_json(receipt_path, receipt)
    print("DICTIONARY_COMPLETE", fit.dictionary.shape, flush=True)


def load_dictionary(output_root):
    path, receipt_path = dictionary_paths(output_root)
    receipt = read_json(receipt_path)
    if receipt.get("status") != "PASS_TRAIN_ONLY_DICTIONARY" or receipt.get("dictionary_npz_sha256") != file_sha256(path):
        raise ValueError("dictionary qualification or exact artifact hash mismatch")
    if (receipt.get("schema_version") != DICTIONARY_SCHEMA or
        receipt.get("identity", {}).get("dictionary_schema_version") != DICTIONARY_SCHEMA or
        receipt.get("identity", {}).get("measurement_scale_protocol_sha256") != SCALE_PROTOCOL_SHA or
        receipt.get("omega_source") != SCALE_SOURCE or receipt.get("u0_oracle_audit", {}).get("converged") is not True):
        raise ValueError("dictionary/error-scale qualification schema or oracle convergence mismatch")
    if receipt["identity"]["source_hashes"] != source_hashes():
        raise ValueError("dictionary source lineage drift")
    with np.load(path, allow_pickle=False) as data:
        d, genes = data["D"].copy(), data["gene_names"].astype(str)
        verify_measurement_scale(d, data["sigma"], data["omega"], data["scale_residual_rms"],
            data["scale_within_variance"], receipt["omega_calibration"],
            split_lock_sha256=receipt["identity"]["split_lock_sha256"])
    if d.shape != (32, len(genes)) or len(genes) > 1024:
        raise ValueError("locked 32-state dictionary dimension mismatch")
    return d, genes, receipt, file_sha256(path)


def build_model(arm, dictionary):
    if arm not in ARMS:
        raise ValueError("only the prelocked three A arms are permitted")
    if arm == "A0":
        return LinearCompositionEncoder(dictionary)
    return LocalSpatialEncoder(dictionary, radius_um=SCHEDULE["radius_um"], neighbor_cap=SCHEDULE["neighbor_cap"])


def load_frozen_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "phase22_external_spatial_checkpoint_v1":
        raise ValueError("unsupported qualified model checkpoint")
    if checkpoint["identity"]["source_hashes"] != source_hashes() or checkpoint["schedule"] != SCHEDULE:
        raise ValueError("model source/schedule lineage mismatch")
    model = build_model(checkpoint["arm"], checkpoint["dictionary"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, checkpoint


def infer_arrays(model, features, coords_um, section_id, indices=None, device="cuda"):
    if indices is not None:
        if not np.issubdtype(np.asarray(indices).dtype, np.integer):
            raise ValueError("subset physical-position indices must be integers")
        index = np.asarray(indices, dtype=np.int64)
        if len(np.unique(index)) != len(index) or (index < 0).any() or (index >= len(features)).any():
            raise ValueError("subset inference requires unique valid physical positions")
        features, coords_um = np.asarray(features)[index], np.asarray(coords_um)[index]
    with torch.inference_mode():
        prediction = model(torch.as_tensor(features, dtype=torch.float32, device=device),
                           torch.as_tensor(coords_um, dtype=torch.float64, device=device), [section_id] * len(features))
    return dict(pi0=prediction.states.cpu().numpy(), mu=prediction.mu.cpu().numpy(),
                log_mu=prediction.log_mu.cpu().numpy(),
                context_scope="ONLY_SUPPLIED_INDICES_REBUILT_PHYSICAL_GRAPH" if indices is not None else "FULL_SECTION_GRAPH")


def loss_for_arrays(log_mu, counts, measured, section_id, donor_id, arm, device):
    if counts.sum() <= 0:
        raise ValueError("sampled region has no positive measured raw counts")
    return spatial_objective(log_mu, torch.as_tensor(counts, dtype=torch.float64, device=device),
        [section_id] * len(counts), [donor_id] * len(counts),
        measured_mask=torch.as_tensor(np.broadcast_to(measured, counts.shape).copy(), device=device),
        lambda_relative=1. if arm == "A2" else 0.)


def validation(model, sections, genes, arm, device):
    model.eval()
    by_donor = {}
    # Import avoids a duplicated metric implementation and does not read test.
    from scripts.evaluate_phase22_external_spatial import section_metrics
    with torch.inference_mode():
        for section in sections:
            count, measured = align_genes(section, genes)
            prediction = infer_arrays(model, section.features, section.coords_um, section.section_id, device=device)
            losses = loss_for_arrays(torch.as_tensor(prediction["log_mu"], device=device), count, measured,
                                    section.section_id, section.donor_id, arm, device)
            metrics = section_metrics(prediction["mu"], count, measured)
            row = dict(objective=float(losses.loss), composition=float(losses.composition),
                       relative=float(losses.relative), **metrics)
            by_donor.setdefault(section.donor_id, []).append(row)
    donor_rows = [{key: float(np.mean([r[key] for r in rows])) for key in rows[0]}
                  for rows in by_donor.values()]
    result = {key: float(np.mean([r[key] for r in donor_rows])) for key in donor_rows[0]}
    result.update(donors=len(by_donor), sections=len(sections), context_scope="FULL_SECTION_GRAPH")
    return result


def region_schedule(sections, seed, epoch):
    # A private sampler RNG is unaffected by model dropout or global RNG use.
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 2201]))
    grouped = {}
    for section in sections:
        grouped.setdefault(section.donor_id, []).append(section)
    for donor in rng.permutation(sorted(grouped)):
        for _ in range(SCHEDULE["regions_per_donor"]):
            options = sorted(grouped[donor], key=lambda s: s.section_id)
            section = options[int(rng.integers(len(options)))]
            center = int(rng.integers(len(section.counts)))
            distance = np.square(section.coords_um - section.coords_um[center]).sum(1)
            # Coordinates break equal-distance ties, not arbitrary array order.
            idx = np.lexsort((section.coords_um[:, 1], section.coords_um[:, 0], distance))[:SCHEDULE["region_spots"]]
            yield section, idx


def train_cell(args):
    if args.smoke:
        # One qualified cached training section only. A smoke test does not
        # require validation features and must not persist any model/results.
        candidates, _, _ = qualification_rows(args.data_root, args.split_lock, args.qualification,
            roles=("train",), with_features=False)
        candidates = sorted([s for s in candidates if
            (Path(args.data_root) / "features/uni2-h" / (s.section_id + ".npz")).exists() and
            (Path(args.data_root) / "features/uni2-h" / (s.section_id + ".json")).exists()], key=lambda s: s.section_id)
        if not candidates:
            raise ValueError("smoke requires one qualified frozen-feature training section")
        sections, lock_sha, _ = qualification_rows(args.data_root, args.split_lock, args.qualification,
            roles=("train",), with_features=True, section_ids=[candidates[0].section_id])
    else:
        sections, lock_sha, _ = qualification_rows(args.data_root, args.split_lock, args.qualification,
            roles=("train", "validation"), with_features=True)
    train = [s for s in sections if s.split == "train"]
    val = [s for s in sections if s.split == "validation"]
    d, genes, dictionary_receipt, d_sha = load_dictionary(args.output_root)
    if dictionary_receipt["identity"]["split_lock_sha256"] != lock_sha:
        raise ValueError("dictionary/fitting split mismatch")
    if args.arm not in ARMS or args.seed not in SEEDS:
        raise ValueError("fit cell outside locked three-arm/three-seed matrix")
    if not args.smoke and not torch.cuda.is_available():
        raise RuntimeError("formal external fits require the authorized CUDA host")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(SCHEDULE["threads"])
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = build_model(args.arm, d).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SCHEDULE["learning_rate"], weight_decay=SCHEDULE["weight_decay"])
    if args.smoke:
        section, idx = next(region_schedule(train, args.seed, 0))
        count, measured = align_genes(section, genes)
        model.train()
        prediction = model(torch.as_tensor(section.features[idx], device=device),
                           torch.as_tensor(section.coords_um[idx], device=device), [section.section_id] * len(idx))
        losses = loss_for_arrays(prediction.log_mu, count[idx], measured, section.section_id, section.donor_id, args.arm, device)
        optimizer.zero_grad(set_to_none=True)
        losses.loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("smoke gradient is nonfinite")
        optimizer.step()
        print("SMOKE_IN_MEMORY_NOT_SAVED", args.arm, args.seed, len(idx), float(losses.loss.detach()), flush=True)
        return
    cell = private_output(args.output_root) / "fits" / args.arm / ("seed%03d" % args.seed)
    cell.mkdir(parents=True, exist_ok=True)
    identity = dict(arm=args.arm, seed=args.seed, split_lock_sha256=lock_sha,
        data_qualification_sha256=file_sha256(args.qualification), dictionary_npz_sha256=d_sha,
        source_hashes=source_hashes(), source_sections={s.section_id: s.evidence for s in sections})
    done_path, best_path, last_path = cell / "DONE_PRIVATE.json", cell / "model.pt", cell / "last.pt"
    if done_path.exists():
        done = read_json(done_path)
        if done.get("identity") != identity or done.get("model_sha256") != file_sha256(best_path):
            raise ValueError("completed fit artifact drift; cannot overwrite")
        print("COMPLETED_CELL_VERIFIED", args.arm, args.seed, flush=True)
        return
    curves, best, best_epoch, steps, start_epoch = [], np.inf, 0, 0, 1
    if last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=True)
        if last.get("identity") != identity or last.get("schedule") != SCHEDULE:
            raise ValueError("resumable checkpoint lineage drift")
        model.load_state_dict(last["state_dict"], strict=True)
        optimizer.load_state_dict(last["optimizer_state"])
        torch.set_rng_state(last["torch_rng_state"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([v.cpu() for v in last["cuda_rng_state"]])
        curves, best, best_epoch, steps = last["curves"], last["best"], last["best_epoch"], last["steps"]
        start_epoch = last["epoch"] + 1
    write_json(cell / "RUN_PRIVATE.json", dict(status="RUNNING", identity=identity, schedule=SCHEDULE,
               resumed_from_epoch=start_epoch - 1, formal_fit_cell=True, smoke=False))
    attempt_id = str(uuid.uuid4())
    append_journal(cell / "ATTEMPTS_PRIVATE.jsonl", dict(event="PROCESS_START", attempt_id=attempt_id,
        resumed_from_epoch=start_epoch - 1, committed_optimizer_steps=steps, time_unix=time.time()))
    started = time.monotonic()
    already_stopped = bool(curves) and (len(curves) >= SCHEDULE["max_epochs"] or
        (curves[-1]["epoch"] >= SCHEDULE["min_epochs"] and curves[-1]["epoch"] - best_epoch >= SCHEDULE["patience"]))
    for epoch in (() if already_stopped else range(start_epoch, SCHEDULE["max_epochs"] + 1)):
        model.train()
        epoch_values, skipped = [], 0
        for section, idx in region_schedule(train, args.seed, epoch):
            count, measured = align_genes(section, genes)
            if count[idx].sum() <= 0:
                skipped += 1
                continue
            prediction = model(torch.as_tensor(section.features[idx], device=device),
                               torch.as_tensor(section.coords_um[idx], device=device), [section.section_id] * len(idx))
            losses = loss_for_arrays(prediction.log_mu, count[idx], measured, section.section_id, section.donor_id, args.arm, device)
            optimizer.zero_grad(set_to_none=True)
            losses.loss.backward()
            if not torch.isfinite(losses.loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("formal gradient/loss nonfinite; no silent retry with altered schedule")
            optimizer.step()
            steps += 1
            append_journal(cell / "STEP_EVENTS_PRIVATE.jsonl", dict(event="OPTIMIZER_STEP_COMPLETED",
                attempt_id=attempt_id, epoch=epoch, committed_sequence=steps,
                note="resumed_partial_epoch_may_repeat_committed_sequence; count_lines_for_actual_recorded_updates"))
            epoch_values.append([float(losses.loss.detach()), float(losses.composition.detach()), float(losses.relative.detach())])
        if not epoch_values:
            raise ValueError("epoch has no valid count-supervised region")
        metric = validation(model, val, genes, args.arm, device)
        row = dict(epoch=epoch, train_objective=float(np.mean(epoch_values, axis=0)[0]),
                   train_composition=float(np.mean(epoch_values, axis=0)[1]),
                   train_relative=float(np.mean(epoch_values, axis=0)[2]),
                   validation=metric, optimizer_steps=len(epoch_values), zero_count_regions_skipped=skipped)
        curves.append(row)
        if metric["objective"] < best:
            best, best_epoch = metric["objective"], epoch
            write_torch(best_path, dict(schema_version="phase22_external_spatial_checkpoint_v1",
                arm=args.arm, seed=args.seed, selected_epoch=epoch, selection_validation=metric,
                identity=identity, schedule=SCHEDULE, dictionary=torch.from_numpy(d),
                gene_names=genes.tolist(), state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                trainable_parameters=sum(p.numel() for p in model.parameters()), context_scope="FULL_SECTION_VALIDATION_GRAPH"))
        write_torch(last_path, dict(identity=identity, schedule=SCHEDULE, epoch=epoch,
            state_dict=model.state_dict(), optimizer_state=optimizer.state_dict(), torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state_all() if device == "cuda" else [],
            curves=curves, best=float(best), best_epoch=best_epoch, steps=steps))
        write_json(cell / "TRAINING_CURVES_PRIVATE.json", dict(identity=identity, curves=curves))
        print("FORMAL_EPOCH", args.arm, args.seed, epoch, metric["objective"], metric["gene_spearman"], flush=True)
        if epoch >= SCHEDULE["min_epochs"] and epoch - best_epoch >= SCHEDULE["patience"]:
            break
    selected = torch.load(best_path, map_location="cpu", weights_only=True)
    append_journal(cell / "ATTEMPTS_PRIVATE.jsonl", dict(event="PROCESS_COMPLETED", attempt_id=attempt_id,
        completed_epoch_count=len(curves), committed_optimizer_steps=steps, time_unix=time.time()))
    write_json(done_path, dict(status="COMPLETE", identity=identity, schedule=SCHEDULE,
        actual_epochs=len(curves), selected_epoch=best_epoch, optimizer_steps=steps,
        validation=selected["selection_validation"], model_sha256=file_sha256(best_path),
        formal_completed_fit_cells=1, stopped_at_budget=len(curves) == SCHEDULE["max_epochs"],
        seconds_this_process=time.monotonic() - started,
        test_labels_or_counts_read=False, labels_read="ST_train_and_validation_only",
        update_count_scope="optimizer_steps is committed trajectory; private append-only journals additionally retain interrupted/repeated recorded updates",
        dictionary_fit_reference=d_sha, dictionary_refit=False))
    print("FORMAL_CELL_COMPLETE", args.arm, args.seed, len(curves), best_epoch, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-measured-genes", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    private_output(args.output_root)
    with threadpool_limits(limits=SCHEDULE["threads"]):
        if args.prepare_only:
            if args.smoke or args.arm or args.seed or args.target_measured_genes is None:
                parser.error("dictionary preparation requires target measurement list and no cell/smoke flags")
            prepare_dictionary(args)
        else:
            if args.arm is None or args.seed is None:
                parser.error("one formal/smoke cell requires arm and seed")
            train_cell(args)


if __name__ == "__main__":
    main()

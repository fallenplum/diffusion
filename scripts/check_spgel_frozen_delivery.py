"""Administrative acceptance of the two fixed, actually replayed S4 exports.

This adds no fit and changes no locked source code. Source spot measurements
may validate source gene/conditional paths; they never impersonate TCGA bulk.
Coordinates, spot predictions and measured labels remain in a private dump.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.spgel.contracts import (PROGRAMS, SourceStatistics, array_hash,
    canonical_hash, require, sha256)
from src.spgel.geometry import build_neighbors
from src.spgel.interface import load_frozen_interface
from src.spgel.training import (configure_runtime, load_assets, model_for,
    read_json, source_hashes, validate_lock, write_json)


ARMS = ("S_A5", "S_A1")
TOLERANCE = 2e-6


def tensor_state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), "NON_TENSOR_LEARNED_STATE")
        x = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(x.dtype).encode())
        digest.update(str(tuple(x.shape)).encode())
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()


def statistics_sha256(stats):
    return canonical_hash({key: dict(array_sha256=array_hash(value)) if isinstance(value, np.ndarray)
        else value for key, value in asdict(stats).items()})


def same_statistics(expected, actual):
    require(set(expected) == set(actual), "EXPORTED_SCALER_FIELD_SET_CHANGED")
    for key, value in expected.items():
        equal = np.array_equal(value, actual[key]) if isinstance(value, np.ndarray) else value == actual[key]
        require(equal, "EXPORTED_SCALER_VALUES_CHANGED")


def maximum_error(a, b):
    x, y = np.asarray(a), np.asarray(b)
    require(x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(),
            "ACCEPTANCE_FORWARD_SHAPE_OR_NONFINITE")
    return float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)), initial=0.))


def check_close(a, b, code):
    error = maximum_error(a, b)
    require(error <= TOLERANCE, code)
    return error


def expect_value_error(call, code):
    try:
        call()
    except ValueError as error:
        require(code in str(error), "WRONG_INTERFACE_REJECTION_REASON")
        return
    raise ValueError("UNSUPPORTED_OR_WRONG_DOMAIN_INTERFACE_WAS_ACCEPTED")


def private_npz(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def full_section_graph(source, rows):
    """The API must reconstruct the SAME complete graph, not a sampled graph."""
    local = build_neighbors(source.xy_um[rows], source.section[rows],
                            radius_um=300., maximum_neighbors=16)
    global_graph = source.neighbors[rows]
    present = global_graph >= 0
    lookup = np.full(len(source.features), -1, np.int64)
    lookup[rows] = np.arange(len(rows))
    expected = np.where(present, lookup[np.maximum(global_graph, 0)], -1)
    require(np.all(expected[present] >= 0) and np.array_equal(local, expected),
            "FULL_SECTION_API_GRAPH_DIFFERS_FROM_ORIGINAL_CHECKPOINT_CONTEXT")
    return local


def verify_replay_gate(private, lock):
    audits = []
    for stage, status, units, states in (
        ("pilot", "PASS_COMPLETE_SPGEL_PILOT_72_FITS_192_REPLAYS", 72, 192),
        ("s4", "PASS_COMPLETE_SPGEL_S4_2_FITS_6_REPLAYS", 2, 6)):
        path = private / stage / "REPLAY_AUDIT.json"
        value = read_json(path)
        require(value["status"] == status and value["contract_sha256"] == lock["contract_sha256"]
            and value["passed_units"] == len(value["units"]) == units and value["failed_units"] == 0
            and value["passed_states"] == states and value["new_optimizer_fits"] == 0
            and value["new_optimizer_updates"] == 0, "FIXED_SOURCE_FITS_AND_198_REPLAYS_NOT_COMPLETE")
        require(sum(len(row["snapshots"]) for row in value["units"]) == states
            and all(row["status"] == "PASS" and row["actual_saved_state_loaded"] is True
                and all(snapshot["actual_state_forward_replayed"] is True
                    for snapshot in row["snapshots"]) for row in value["units"]),
            "SOURCE_ACTUAL_REPLAY_FLAGS_OR_STATE_INVENTORY_CHANGED")
        audits.append(dict(stage=stage, SHA256=sha256(path), replayed_models=units, replayed_states=states))
    export_path = private / "s4" / "EXPORT_COMPLETE.private.json"
    exported = read_json(export_path)
    require(exported["status"] == "COMPLETE_TWO_REPLAYED_SOURCE_INTERFACES"
        and exported["contract_sha256"] == lock["contract_sha256"]
        and {row["arm"] for row in exported["exports"]} == set(ARMS)
        and len(exported["exports"]) == 2 and exported["new_optimizer_fits"] == 0,
        "FIXED_TWO_SOURCE_EXPORTS_NOT_COMPLETE")
    return audits, {row["arm"]: row for row in exported["exports"]}, sha256(export_path)


def original_image_forward(model, source, rows, all_rows, device, batch_size):
    pieces = []
    latent = []
    with torch.no_grad():
        for begin in range(0, len(rows), batch_size):
            batch = source.image_batch(rows[begin:begin + batch_size], allowed_rows=all_rows, device=device)
            output = model.image_only_predict(batch)
            pieces.append(output["program_predictions"].cpu().numpy())
            tokens = output.get("visual_program_tokens")
            if tokens is None:
                tokens = output["image_embedding"][:, None, :].expand(-1, 4, -1)
            latent.append(tokens.cpu().numpy())
    return np.concatenate(pieces), np.concatenate(latent)


def source_conditional_path(frozen, original, source, stats, rows, all_rows, device, batch_size):
    """Real ST genes validate frozen SOURCE paths; this is not the bulk API."""
    errors = dict(source_gene_token=0., conditional_image_read=0., conditional_gene_read=0.,
                  paired_scores=0., route_original_window=0.)
    batches = 0
    with torch.no_grad():
        for begin in range(0, len(rows), batch_size):
            selected = rows[begin:begin + batch_size]
            image = source.image_batch(selected, allowed_rows=all_rows, device=device)
            genes = stats.genes_z(source, selected, device)
            own = torch.arange(len(selected), device=device)[:, None]
            output_a = original.image_only_predict(image)
            output_b = frozen.model.image_only_predict(image)
            gene_a = original.encode_gene_programs(genes, paired_mode=True)
            gene_b = frozen.model.encode_gene_programs(genes, paired_mode=True)
            errors["source_gene_token"] = max(errors["source_gene_token"],
                check_close(gene_a["tokens"].cpu(), gene_b["tokens"].cpu(), "FROZEN_SOURCE_GENE_FORWARD_CHANGED"))
            require(torch.equal(gene_a["program_mask"], gene_b["program_mask"]), "SOURCE_GENE_MASK_CHANGED")
            a = original.paired_scores(output_a, genes, own, own, paired_mode=True, return_details=True)
            b = frozen.model.paired_scores(output_b, genes, own, own, paired_mode=True, return_details=True)
            for key, name in (("conditioned_image_tokens", "conditional_image_read"),
                              ("conditioned_gene_tokens", "conditional_gene_read"), ("scores", "paired_scores")):
                errors[name] = max(errors[name], check_close(a[key].cpu(), b[key].cpu(),
                    "FROZEN_SOURCE_ORIGINAL_EVIDENCE_REREAD_CHANGED"))
            # Pass the complete registered window from the exported image model,
            # including its zero-masked self duplicate and all original neighbors.
            coordinates = np.concatenate((np.zeros((len(selected), 1, 2), np.float32),
                                            image.neighbor_xy_um.cpu().numpy()), axis=1)
            routed = frozen.route_bulk_to_wsi(gene_b["tokens"], output_b["window_tokens"],
                                               coordinates, output_b["window_mask"])
            expected_image_mask = output_b["window_mask"].any(-1).cpu().numpy()[:, None].repeat(4, 1)
            require(routed["program_embeddings"].shape == (len(selected), 4, 128)
                and routed["attention"].shape == (len(selected), 4, output_b["window_tokens"].shape[1])
                and np.array_equal(routed["mask"], expected_image_mask),
                "CONDITIONAL_ROUTE_INTERFACE_SHAPE_OR_MASK")
            errors["route_original_window"] = max(errors["route_original_window"],
                check_close(routed["program_embeddings"], a["conditioned_image_tokens"][:, 0].cpu(),
                    "SOURCE_ROUTE_API_DIFFERS_FROM_ORIGINAL_COMPLETE_WINDOW"))
            batches += 1
    return errors, batches


def boundary_checks(frozen, source, section_rows, batch_size):
    # Both complete sections remain present. Padding never removes valid spots
    # or changes any patch's original neighborhood.
    left, right = section_rows[0], section_rows[1]
    joined = np.concatenate((left, right))
    padding = 3
    x = np.full((2, len(joined) + padding, 1536), np.nan, np.float32)
    xy = np.full((2, len(joined) + padding, 2), np.nan, np.float64)
    slide = np.full((2, len(joined) + padding), -1, np.int64)
    mask = np.zeros(slide.shape, bool)
    for b, rows in enumerate((joined, left)):
        x[b, :len(rows)] = source.features[rows]
        xy[b, :len(rows)] = source.xy_um[rows]
        slide[b, :len(rows)] = source.section[rows]
        mask[b, :len(rows)] = True
    output = frozen.encode_image_spots(x, xy, slide, mask, batch_size=batch_size)
    errors = []
    for b, rows in enumerate((joined, left)):
        reference = frozen.encode_image_spots(source.features[rows][None], source.xy_um[rows][None],
            source.section[rows][None], np.ones((1, len(rows)), bool), batch_size=batch_size)
        for key in ("program_embeddings", "program_pred_z", "program_pred"):
            errors.append(check_close(output[key][b, :len(rows)], reference[key][0],
                "PADDING_OR_BATCH_BOUNDARY_CHANGED_FULL_SECTION_OUTPUT"))
    # Separate-section encoding must match joint multi-slide encoding. This
    # detects accidental neighbor communication across section boundaries.
    for rows, offset in ((left, 0), (right, len(left))):
        reference = frozen.encode_image_spots(source.features[rows][None], source.xy_um[rows][None],
            source.section[rows][None], np.ones((1, len(rows)), bool), batch_size=batch_size)
        errors.append(check_close(output["program_pred_z"][0, offset:offset + len(rows)], reference["program_pred_z"][0],
                                  "SLIDE_BOUNDARY_CHANGED_ORIGINAL_CONTEXT"))
    require(np.array_equal(output["mask"], mask)
        and np.count_nonzero(output["program_embeddings"][~mask]) == 0
        and np.count_nonzero(output["program_pred_z"][~mask]) == 0
        and np.count_nonzero(output["program_pred"][~mask]) == 0, "PADDING_BECAME_IMAGE_EVIDENCE")
    return dict(maximum_error=max(errors, default=0.), batch_rows=2,
                complete_sections=2, padding_positions=int((~mask).sum()))


def verify_arm(arm, source, config, private, output, export_receipt, device, batch_size,
               *, source_contract_sha256, source_config_sha256):
    export = private / "s4" / "export" / f"{arm}_seed31"
    manifest_path = export / "source_encoder_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["arm"] == arm and manifest["seed"] == 31
        and manifest["input_dim"] == 1536 and manifest["output_width"] == 128
        and manifest["program_order"] == list(PROGRAMS)
        and manifest["registered_gene_union"] == len(source.gene_symbols)
        and manifest["source_gene_order_sha256"] == canonical_hash(source.gene_symbols.tolist())
        and manifest["source_data_cache_manifest_sha256"] == config["source_binding"]["cache_manifest_sha256"]
        and manifest["source_requalification_sha256"] == config["source_binding"]["fresh_requalification_sha256"]
        and manifest["S4_replay_sha256"] == sha256(private / "s4" / "REPLAY_AUDIT.json")
        and manifest["native_learned_program_slots"] == (arm == "S_A5")
        and manifest["molecular_and_conditional_interface_supported"] == (arm == "S_A5"),
        "EXPORTED_MANIFEST_DIMENSIONS_OR_SOURCE_LINEAGE_CHANGED")
    require(sha256(manifest_path) == export_receipt["manifest_sha256"]
        and sha256(export / "encoder_state.pt") == export_receipt["encoder_state_sha256"],
        "EXPORTED_INTERFACE_BYTES_CHANGED_AFTER_EXPORT")
    unit = private / "s4" / "cells" / f"{arm}_full_seed31"
    complete = read_json(unit / "COMPLETE.json")
    checkpoint = unit / "checkpoint_epoch40.pt"
    require(sha256(unit / "COMPLETE.json") == manifest["source_full_fit_COMPLETE_sha256"]
        and sha256(checkpoint) == manifest["source_full_checkpoint_sha256"]
        and complete["status"] == "COMPLETE_SPGEL_SOURCE_FIT"
        and complete["contract_sha256"] == source_contract_sha256
        and complete["config_sha256"] == source_config_sha256
        and complete["arm"] == arm and complete["seed"] == 31 and complete["heldout_donor"] == -1
        and complete["epochs"] == 40 and complete["optimizer_updates"] == 1040,
        "ORIGINAL_FIXED_SOURCE_CHECKPOINT_LINEAGE_CHANGED")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    exported = torch.load(export / "encoder_state.pt", map_location="cpu", weights_only=False)
    all_rows = np.arange(len(source.features), dtype=np.int64)
    require(state["arm"] == arm and state["seed"] == 31 and state["heldout_donor"] == -1
        and state["stage"] == "s4" and state["epoch"] == 40 and state["optimizer_updates"] == 1040
        and state["contract_sha256"] == source_contract_sha256 and state["config_sha256"] == source_config_sha256
        and state["train_rows_sha256"] == array_hash(all_rows)
        and state["heldout_rows_sha256"] == array_hash(np.empty(0, np.int64)),
        "EXPORTED_ORIGINAL_CHECKPOINT_IS_NOT_FIXED_FULL_SOURCE_EPOCH40")
    stats = SourceStatistics.fit(source, all_rows)
    same_statistics(asdict(stats), state["stats"])
    same_statistics(state["stats"], exported["source_statistics"])
    membership = canonical_hash(dict(program_order=list(PROGRAMS), gene_order=source.gene_symbols.tolist(),
        up_array_sha256=array_hash(source.up), down_array_sha256=array_hash(source.down)))
    require(np.array_equal(exported["program_up"], source.up)
        and np.array_equal(exported["program_down"], source.down)
        and exported["gene_symbols"] == source.gene_symbols.tolist()
        and manifest["source_gene_membership_sha256"] == manifest["source_ST_gene_membership_SHA"] == membership,
        "SOURCE_GENE_MEMBERSHIP_OR_ORDER_CHANGED")
    reference = model_for(config, source, stats, arm, 31, device).eval().requires_grad_(False)
    reference.load_state_dict(state["state_dict"], strict=True)
    frozen = load_frozen_interface(export, device=device)
    require(not frozen.model.training and not any(module.training for module in frozen.model.modules())
        and not any(p.requires_grad for p in frozen.model.parameters()), "EXPORTED_MODEL_NOT_FROZEN_EVAL")
    learned_sha = tensor_state_sha256(state["state_dict"])
    require(learned_sha == tensor_state_sha256(exported["state_dict"])
        == tensor_state_sha256(frozen.model.state_dict()), "EXPORTED_LEARNED_STATE_DIFFERS_FROM_ORIGINAL")
    require("rna" not in inspect.signature(frozen.encode_image_spots).parameters
        and "tissue" not in inspect.signature(frozen.encode_image_spots).parameters
        and "genes" not in inspect.signature(frozen.image_only_program_map).parameters,
        "IMAGE_API_ACCEPTS_FORBIDDEN_CONDITION_INPUT")

    sections = [np.flatnonzero(source.section == code) for code in np.unique(source.section)]
    errors = dict(image_prediction_z=0., image_embedding=0., image_program_map=0.)
    conditional_errors = {}
    gene_calls_in_image_api = 0
    conditional_batches = 0
    private_dumps = []
    for i, rows in enumerate(sections):
        full_section_graph(source, rows)
        called = [0]
        def observe_gene_call(*_):
            called[0] += 1
        hook = frozen.model.gene_encoder.register_forward_pre_hook(observe_gene_call) if frozen.model.has_genes else None
        try:
            image = frozen.encode_image_spots(source.features[rows][None], source.xy_um[rows][None],
                source.section[rows][None], np.ones((1, len(rows)), bool), batch_size=batch_size)
            program_map = frozen.image_only_program_map(source.features[rows][None], source.xy_um[rows][None],
                source.section[rows][None], np.ones((1, len(rows)), bool), batch_size=batch_size)
        finally:
            if hook is not None:hook.remove()
        gene_calls_in_image_api += called[0]
        require(called[0] == 0 and image["transcriptome_used"] is False
            and image["tissue_truth_used"] is False and program_map["lesion_segmentation_truth_claimed"] is False,
            "GENE_OR_TRUE_TISSUE_ENTERED_IMAGE_ONLY_INFERENCE")
        pz, latent = original_image_forward(reference, source, rows, all_rows, device, batch_size)
        require(image["program_embeddings"].shape == (1, len(rows), 4, 128)
            and image["program_pred_z"].shape == image["program_pred"].shape == (1, len(rows), 4)
            and image["native_learned_program_slots"] == (arm == "S_A5")
            and program_map["program_order"] == list(PROGRAMS), "ACTUAL_IMAGE_INTERFACE_SHAPES_OR_SEMANTICS")
        errors["image_prediction_z"] = max(errors["image_prediction_z"],
            check_close(image["program_pred_z"][0], pz, "IMAGE_API_CHANGED_ORIGINAL_FULL_CONTEXT_PREDICTION"))
        errors["image_embedding"] = max(errors["image_embedding"],
            check_close(image["program_embeddings"][0], latent, "IMAGE_API_CHANGED_ORIGINAL_EMBEDDING"))
        errors["image_program_map"] = max(errors["image_program_map"],
            check_close(program_map["program_pred"], image["program_pred"], "IMAGE_MAP_AND_ENCODING_API_DIFFER"))
        path = output / f"{arm}_section{i:02d}.private.npz"
        private_npz(path, source_rows=rows, section=source.section[rows], xy_um=source.xy_um[rows],
            program_order=np.asarray(PROGRAMS), program_embeddings=image["program_embeddings"][0],
            program_pred_z=image["program_pred_z"][0], program_pred=image["program_pred"][0],
            measured_program=source.targets[rows], measured_program_mask=source.target_mask[rows])
        private_dumps.append(dict(SHA256=sha256(path), rows=len(rows)))
        if arm == "S_A5":
            current, count = source_conditional_path(frozen, reference, source, stats, rows, all_rows, device, batch_size)
            for key, value in current.items():conditional_errors[key] = max(conditional_errors.get(key, 0.), value)
            conditional_batches += count

    boundary = boundary_checks(frozen, source, sections, batch_size)
    # These are domain/unsupported guards only. In particular, do not pass ST
    # source Z values under a fabricated current-R8-train bulk contract.
    probe = sections[0][:min(8, len(sections[0]))]
    if arm == "S_A1":
        expect_value_error(lambda: frozen.encode_bulk_genes(source.genes[probe], source.gene_mask[probe],
            membership, {}), "NO_TRAINED_MOLECULAR_ENCODER")
        expect_value_error(lambda: frozen.route_bulk_to_wsi(np.zeros((1, 4, 128), np.float32),
            np.zeros((1, 1, 128), np.float32), np.zeros((1, 1, 2)), np.ones((1, 1), bool)),
            "NO_TRAINED_CONDITIONAL_IMAGE_QUERY")
    else:
        source_z = stats.genes_z(source, probe).values.numpy()
        expect_value_error(lambda: frozen.encode_bulk_genes(source_z, source.gene_mask[probe], membership,
            dict(schema="spgel_bulk_scale_contract_v1", fit_scope="source_full_ST",
                uses_source_spot_scaler=True, values_are_current_train_gene_Z=False)),
            "SEPARATE_TRAIN_ONLY_SCALE_CONTRACT")
    require(learned_sha == tensor_state_sha256(frozen.model.state_dict())
        and not any(p.requires_grad for p in frozen.model.parameters()), "ACCEPTANCE_MUTATED_FROZEN_STATE")
    same_statistics(asdict(stats), asdict(frozen.stats))
    return dict(arm=arm, source_manifest_SHA256=sha256(manifest_path), exported_encoder_SHA256=sha256(export / "encoder_state.pt"),
        original_checkpoint_SHA256=sha256(checkpoint), learned_state_SHA256=learned_sha,
        source_scaler_SHA256=statistics_sha256(stats), source_membership_SHA256=membership,
        image_sections=len(sections), image_spots=sum(map(len, sections)), private_map_dumps=private_dumps,
        image_API_gene_forward_calls=gene_calls_in_image_api, maximum_errors=errors,
        source_conditional_maximum_errors=conditional_errors, source_conditional_batches=conditional_batches,
        padding_multislide=boundary, all_exported_parameters_frozen=True,
        source_image_map_scope="FULL_SOURCE_FIT_QA_NOT_HELDOUT",
        actual_TCGA_bulk_qualification="NOT_RUN_SOURCE_SPOT_VALUES_NOT_PASSED_AS_BULK",
        new_optimizer_fits=0, new_optimizer_updates=0)


def run(args):
    configure_runtime()
    private = args.private.resolve()
    require(not private.is_relative_to(ROOT), "PRIVATE_SOURCE_ROOT_INSIDE_CHECKOUT")
    require(args.public_output.resolve().is_relative_to(ROOT) and args.public_output.suffix == ".json",
            "ANONYMOUS_PUBLIC_AUDIT_MUST_BE_JSON_INSIDE_CHECKOUT")
    config = read_json(args.config)
    lock = validate_lock(private, args.config, config)
    source_files = source_hashes()
    audits, exports, export_sha = verify_replay_gate(private, lock)
    source = load_assets(private / "SOURCE_ASSETS.private.json", config)
    output = private / "s4" / "frozen_delivery_acceptance_v1"
    require(not output.exists(), "EXISTING_FROZEN_DELIVERY_ACCEPTANCE_PRESERVED_NO_AUTORETRY")
    require(not args.public_output.exists(), "EXISTING_PUBLIC_ACCEPTANCE_PRESERVED")
    output.mkdir(mode=0o700)
    try:
        rows = [verify_arm(arm, source, config, private, output, exports[arm], args.device, args.batch_size,
            source_contract_sha256=lock["contract_sha256"], source_config_sha256=sha256(args.config)) for arm in ARMS]
        require(source_hashes() == source_files, "LOCKED_SOURCE_CHANGED_DURING_DELIVERY_ACCEPTANCE")
        receipt = dict(schema="spgel_actual_frozen_source_delivery_acceptance_v1",
            status="PASS_TWO_ACTUAL_S4_ENCODERS_FULL_SOURCE_CONTEXT_AND_BOUNDARIES",
            acceptance_script_SHA256=sha256(__file__), source_contract_SHA256=lock["contract_sha256"],
            source_export_complete_SHA256=export_sha, prior_actual_replay_audits=audits,
            source_encoders_count=2, source_sections_count=len(np.unique(source.section)),
            source_spots_count=len(source.features), rows=rows, new_optimizer_fits=0, new_optimizer_updates=0,
            TCGA_loaded=False, DSS_or_PFI_read=False, source_or_target_labels_selected_model=False,
            source_ST_measurements_used_only_for_source_acceptance=True,
            full_source_image_maps_are_heldout=False,
            public_contains_spot_predictions_coordinates_or_weights=False,
            TCGA_bulk_qualification="NOT_RUN_REQUIRES_SEPARATE_TRAIN_ONLY_DOMAIN_CONTRACT")
        write_json(output / "COMPLETE.private.json", receipt)
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.public_output, receipt)
        return receipt
    except BaseException as error:
        write_json(output / "ERROR_RECEIPT.private.json", dict(status="FAILED_FROZEN_DELIVERY_ACCEPTANCE_NO_AUTOMATIC_RETRY",
            failure_type=type(error).__name__, new_optimizer_fits=0, new_optimizer_updates=0))
        (output / "error.private.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def run_heldout_maps(args):
    """Render every fixed LODO heldout section from saved predictions, no model calls."""
    configure_runtime()
    private = args.private.resolve()
    require(not private.is_relative_to(ROOT), "PRIVATE_SOURCE_ROOT_INSIDE_CHECKOUT")
    require(args.public_output.resolve().is_relative_to(ROOT) and args.public_output.suffix == ".json",
            "ANONYMOUS_PUBLIC_AUDIT_MUST_BE_JSON_INSIDE_CHECKOUT")
    config = read_json(args.config)
    lock = validate_lock(private, args.config, config)
    science = source_hashes()
    audits, _, _ = verify_replay_gate(private, lock)
    source = load_assets(private / "SOURCE_ASSETS.private.json", config)
    pilot = read_json(private / "pilot" / "REPLAY_AUDIT.json")
    replays = {(row["arm"], row["heldout_donor"], row["seed"]): row for row in pilot["units"]}
    output = private / "heldout_maps"
    require(not output.exists() and not args.public_output.exists(),
            "EXISTING_HELDOUT_MAPS_OR_PUBLIC_RECEIPT_PRESERVED")
    output.mkdir(mode=0o700)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from PIL import Image
        ordinals = {int(code): i for i, code in enumerate(np.unique(source.section))}
        records = []
        prediction_array_hashes = []
        for arm in ARMS:
            for donor in range(6):
                train, held = source.split(donor)
                stats = SourceStatistics.fit(source, train, heldout_donors=[donor])
                unit = private / "pilot" / "cells" / f"{arm}_d{donor}_seed31"
                done = read_json(unit / "COMPLETE.json")
                replay = replays[(arm, donor, 31)]
                require(done["status"] == "COMPLETE_SPGEL_SOURCE_FIT"
                    and done["contract_sha256"] == lock["contract_sha256"]
                    and done["config_sha256"] == sha256(args.config)
                    and replay["completion_sha256"] == sha256(unit / "COMPLETE.json")
                    and replay["artifact_sha256"] == done["artifact_sha256"] == sha256(unit / "model.pt"),
                    "SAVED_HELDOUT_SOURCE_MODEL_OR_REPLAY_BINDING_CHANGED")
                checkpoint = unit / "checkpoint_epoch40.pt"
                snapshot = [row for row in done["snapshots"] if row["label"] == "epoch40"]
                require(len(snapshot) == 1 and sha256(checkpoint) == snapshot[0]["checkpoint_sha256"]
                    and any(row["label"] == "epoch40" and row["actual_state_forward_replayed"] is True
                        and row["checkpoint_sha256"] == sha256(checkpoint) for row in replay["snapshots"]),
                    "HELDOUT_MAPS_REQUIRE_FIXED_EPOCH40_ACTUAL_REPLAY")
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                require(state["stage"] == "pilot" and state["epoch"] == 40 and state["arm"] == arm
                    and state["seed"] == 31 and state["heldout_donor"] == donor
                    and state["train_rows_sha256"] == array_hash(train)
                    and state["heldout_rows_sha256"] == array_hash(held),
                    "HELDOUT_MAP_CHECKPOINT_SPLIT_OR_EPOCH_CHANGED")
                same_statistics(asdict(stats), state["stats"])
                artifact = torch.load(unit / "model.pt", map_location="cpu", weights_only=False)
                same_statistics(asdict(stats), artifact["stats"])
                prediction = np.asarray(artifact["snapshots"]["epoch40"]["heldout_prediction"])
                require(prediction.shape == (len(held), 4) and np.isfinite(prediction).all(),
                        "SAVED_HELDOUT_PREDICTION_SHAPE_OR_VALUES")
                prediction_array_hashes.append(array_hash(prediction))
                truth = stats.targets_z(source, held)
                for section in np.unique(source.section[held]):
                    selected = source.section[held] == section
                    rows = held[selected]
                    ordinal = ordinals[int(section)]
                    stem = f"{arm}_anonymous_section{ordinal:02d}_heldout_source_proxy"
                    array_path = output / (stem + ".private.npz")
                    private_npz(array_path, source_rows=rows, xy_um=source.xy_um[rows],
                        measured_program_raw=source.targets[rows], measured_program_Z=truth[selected],
                        measured_mask=source.target_mask[rows], image_only_prediction_Z=prediction[selected],
                        program_order=np.asarray(PROGRAMS))
                    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
                        fig, axes = plt.subplots(4, 2, figsize=(9, 13), constrained_layout=True)
                        fig.suptitle(f"{arm} | anonymous section {ordinal:02d} | LODO heldout | epoch 40\n"
                            "Source-program proxy, not cell truth; fixed training-Z color limits [-3, 3]", fontsize=10)
                        for k, program in enumerate(PROGRAMS):
                            for column, (label, values) in enumerate((("Measured source proxy", truth[selected, k]),
                                                                    ("Image-only prediction", prediction[selected, k]))):
                                ax = axes[k, column]
                                measured = source.target_mask[rows, k]
                                xy = source.xy_um[rows]
                                ax.scatter(xy[:, 0], xy[:, 1], c="#d9d9d9", marker="s", s=6, linewidths=0.)
                                present = measured if column == 0 else np.ones(len(rows), bool)
                                image = ax.scatter(xy[present, 0], xy[present, 1], c=values[present],
                                    vmin=-3., vmax=3., cmap="viridis", marker="s", s=6, linewidths=0.)
                                ax.set(title=f"{program}\n{label}", xlabel="x (um)", ylabel="y (um)")
                                ax.set_aspect("equal", adjustable="box");ax.invert_yaxis()
                            fig.colorbar(image, ax=axes[k, :], label="Training-partition Z", shrink=.7, ticks=[-3, 0, 3])
                        png = output / (stem + ".private.png")
                        require(not png.exists(), "EXISTING_PRIVATE_HELDOUT_PNG_PRESERVED")
                        fig.savefig(png, dpi=160, metadata={"Title": f"Anonymous section {ordinal:02d}; LODO heldout",
                            "Description": "Fixed epoch40 source-program proxy; not cellular ground truth.",
                            "Software": "SPGEL private source QA"})
                        plt.close(fig)
                    with Image.open(png) as decoded:
                        metadata = {key: value for key, value in decoded.info.items() if isinstance(value, str)}
                        for value in metadata.values():
                            require(not any(token in value for token in (str(private), str(ROOT), "TCGA-", "/root/", "\\Users\\")),
                                    "PRIVATE_PNG_METADATA_CONTAINS_PATH_OR_IDENTIFIERS")
                    records.append(dict(arm=arm, donor=donor, section_ordinal=ordinal, rows=len(rows),
                        PNG=png.name, PNG_SHA256=sha256(png), array_file=array_path.name, array_SHA256=sha256(array_path),
                        checkpoint_SHA256=sha256(checkpoint), saved_model_artifact_SHA256=done["artifact_sha256"]))
                del artifact, state
        require(len(records) == 24 and len({(row["arm"], row["section_ordinal"]) for row in records}) == 24
            and all(sum(row["rows"] for row in records if row["arm"] == arm) == len(source.features) for arm in ARMS),
            "ALL_FIXED_HELDOUT_SECTIONS_MUST_BE_RENDERED_ONCE_PER_MODEL")
        require(source_hashes() == science, "LOCKED_SOURCE_CHANGED_DURING_HELDOUT_MAP_RENDER")
        write_json(output / "MAP_INDEX.private.json", dict(schema="spgel_private_fixed_LODO_map_index_v1",
            status="COMPLETE_ALL_24_FIXED_HELDOUT_SOURCE_PROXY_MAPS", source_contract_SHA256=lock["contract_sha256"],
            all_sections_included=True, model_or_section_selection=False, rows=records,
            new_optimizer_fits=0, new_optimizer_updates=0, new_model_forward_calls=0))
        receipt = dict(schema="spgel_anonymous_fixed_heldout_map_audit_v1",
            status="PASS_ALL_12_SECTIONS_TWO_FIXED_MODELS_SAVED_LODO_EPOCH40_MAPS",
            rendering_script_SHA256=sha256(__file__), source_contract_SHA256=lock["contract_sha256"],
            prior_actual_replay_audits=audits, fixed_source_models_count=2, fixed_epoch=40,
            source_sections_count=12, source_spots_per_model=len(source.features), private_PNG_count=24,
            private_array_count=24, saved_prediction_arrays_used=12,
            PNG_SHA256=[row["PNG_SHA256"] for row in records], saved_prediction_array_SHA256=prediction_array_hashes,
            fixed_training_Z_color_limits=[-3, 3], all_sections_included=True,
            model_or_epoch_or_section_selected_from_heldout=False,
            source_ST_program_proxy_is_cell_truth=False, source_full_fit_maps_are_not_used_as_heldout=True,
            independent_source_confirmation=False, public_contains_coordinates_images_or_spot_predictions=False,
            PNG_text_metadata_contains_paths_or_identifiers=False, TCGA_loaded=False, DSS_or_PFI_read=False,
            new_optimizer_fits=0, new_optimizer_updates=0, new_model_forward_calls=0)
        args.public_output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.public_output, receipt)
        return receipt
    except BaseException as error:
        write_json(output / "ERROR_RECEIPT.private.json", dict(status="FAILED_HELDOUT_MAP_RENDER_NO_AUTOMATIC_RETRY",
            failure_type=type(error).__name__, new_optimizer_fits=0, new_optimizer_updates=0, new_model_forward_calls=0))
        (output / "error.private.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/spgel/pilot_v1.json")
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mode", choices=("acceptance", "heldout_maps"), default="acceptance")
    args = parser.parse_args()
    require(1 <= args.batch_size <= 512, "ACCEPTANCE_BATCH_SIZE_OUTSIDE_MEMORY_BOUND")
    result = run(args) if args.mode == "acceptance" else run_heldout_maps(args)
    print(json.dumps({k:result[k] for k in ("status", "source_encoders_count", "fixed_source_models_count",
        "source_sections_count", "source_spots_count", "private_PNG_count", "new_optimizer_fits", "new_optimizer_updates") if k in result}))


if __name__ == "__main__":
    main()

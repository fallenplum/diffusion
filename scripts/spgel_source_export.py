"""Export only the two prelocked, fully fitted and replayed source encoders."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.spgel.contracts import PROGRAMS, canonical_hash, sha256, require, array_hash
from src.spgel.training import read_json, validate_lock, write_json, torch_write, load_assets


def export_interfaces(config_path, private):
    config = read_json(config_path); private = Path(private)
    lock = validate_lock(private, config_path, config)
    source = load_assets(private / "SOURCE_ASSETS.private.json", config)
    replay = read_json(private / "s4/REPLAY_AUDIT.json")
    require(replay.get("status") == "PASS_COMPLETE_SPGEL_S4_2_FITS_6_REPLAYS"
            and replay.get("contract_sha256") == lock["contract_sha256"]
            and replay.get("passed_units") == 2 and replay.get("passed_states") == 6
            and replay.get("failed_units") == 0, "EXPORT_REQUIRES_BOTH_ACTUAL_S4_MODEL_REPLAYS")
    pilot = read_json(private / "pilot/REPLAY_AUDIT.json")
    require(pilot.get("status") == "PASS_COMPLETE_SPGEL_PILOT_72_FITS_192_REPLAYS"
            and pilot.get("contract_sha256") == lock["contract_sha256"], "EXPORT_REQUIRES_COMPLETE_SOURCE_PILOT")
    output = private / "s4/export"
    require(not output.exists(), "EXISTING_SOURCE_EXPORT_PRESERVED")
    output.mkdir(parents=True, mode=0o700)
    exports = []
    for arm in ("S_A5", "S_A1"):
        unit = private / "s4/cells" / f"{arm}_full_seed31"
        done = read_json(unit / "COMPLETE.json")
        cp = unit / "checkpoint_epoch40.pt"
        primary = [s for s in done["snapshots"] if s["label"] == "epoch40"]
        require(len(primary) == 1 and sha256(cp) == primary[0]["checkpoint_sha256"]
                and done["contract_sha256"] == lock["contract_sha256"], "SOURCE_EXPORT_CHECKPOINT_LINEAGE")
        state = torch.load(cp, map_location="cpu", weights_only=False)
        up, down, symbols = source.up, source.down, source.gene_symbols.tolist()
        membership_hash = canonical_hash(dict(program_order=list(PROGRAMS), gene_order=symbols,
            up_array_sha256=array_hash(up), down_array_sha256=array_hash(down)))
        held = []
        for donor in range(6):
            cell = read_json(private / "pilot/cells" / f"{arm}_d{donor}_seed31" / "COMPLETE.json")
            snapshot = [s for s in cell["snapshots"] if s["label"] == "epoch40"][0]
            held.append(dict(heldout_donor=donor, seed=31, COMPLETE_sha256=sha256(
                private / "pilot/cells" / f"{arm}_d{donor}_seed31" / "COMPLETE.json"),
                source_spots_count=cell["heldout_rows"], task_A=snapshot["heldout"]["task_A"],
                task_C=snapshot["heldout"]["task_C"]))
        manifest = dict(schema="spgel_frozen_source_encoder_manifest_v1",
            status="COMPLETE_REPLAYED_SOURCE_ENCODER_INTERFACE", arm=arm, seed=31,
            source_version="SPGEL_PILOT_V1_DEVELOPMENT_FULL_FIT", input_dim=1536, output_width=128,
            input_L2_normalization="float32_per_row_unit_L2_reject_norm_le_1e-12",
            output_embedding_unit="trainable_value_coordinates_not_claimed_unit_L2",
            source_ST_gene_membership_SHA=membership_hash, source_gene_membership_sha256=membership_hash,
            source_gene_order_sha256=canonical_hash(symbols), program_order=list(PROGRAMS),
            registered_gene_union=len(symbols), measurement_masks_required=True,
            program_prediction_units="source_train_Z_and_raw_signed_log1p_CP10K",
            adapter_shape=[128, 128] if arm == "S_A5" else None,
            bulk_gene_scaler="R8_current_train_only_no_automatic_source_spot_scaler_reuse",
            geometry="same_slide_real_um_neighbors_radius300_max16_no_implicit_TCGA_support_equivalence",
            source_data_cache_manifest_sha256=config["source_binding"]["cache_manifest_sha256"],
            source_requalification_sha256=config["source_binding"]["fresh_requalification_sha256"],
            source_full_fit_COMPLETE_sha256=sha256(unit / "COMPLETE.json"),
            source_full_checkpoint_sha256=sha256(cp), S4_replay_sha256=sha256(private / "s4/REPLAY_AUDIT.json"),
            heldout_source_results=held, independent_source_confirmation=False,
            checkpoint_selected_using_DSS=False, R8_requires_explicit_new_source_version="R8_SOURCE_SPGEL_V1",
            native_learned_program_slots=arm == "S_A5", molecular_and_conditional_interface_supported=arm == "S_A5")
        core_hash = canonical_hash(manifest)
        manifest["source_manifest_core_sha256"] = core_hash
        folder = output / f"{arm}_seed31"; folder.mkdir(mode=0o700)
        encoder_state = dict(schema="spgel_frozen_source_encoder_state_v1", source_manifest_core_sha256=core_hash,
            arm=arm, state_dict=state["state_dict"], source_statistics=state["stats"],
            program_up=up, program_down=down, gene_symbols=symbols)
        torch_write(folder / "encoder_state.pt", encoder_state)
        manifest["encoder_state_sha256"] = sha256(folder / "encoder_state.pt")
        write_json(folder / "source_encoder_manifest.json", manifest)
        exports.append(dict(arm=arm, manifest_sha256=sha256(folder / "source_encoder_manifest.json"),
            encoder_state_sha256=manifest["encoder_state_sha256"], new_optimizer_fits=0))
    result = dict(schema="spgel_source_interface_exports_v1", status="COMPLETE_TWO_REPLAYED_SOURCE_INTERFACES",
        contract_sha256=lock["contract_sha256"], exports=exports, new_optimizer_fits=0,
        no_source_model_selection_by_R8_or_DSS=True)
    write_json(private / "s4/EXPORT_COMPLETE.private.json", result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--private", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/spgel/pilot_v1.json")
    args = p.parse_args()
    result = export_interfaces(args.config, args.private)
    print(json.dumps(dict(status=result["status"], exported_encoders=len(result["exports"]), new_optimizer_fits=0)))


if __name__ == "__main__":
    main()

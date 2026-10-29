"""Frozen SPGEL evidence interface with separate spot and bulk scale contracts."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch

from .contracts import GeneBatch, ImageBatch, SourceStatistics, canonical_hash, require, sha256, PROGRAMS
from .geometry import build_neighbors
from .models import SpatialProgramEncoder


def as_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


class FrozenSpatialProgramInterface:
    def __init__(self, model, stats: SourceStatistics, *, gene_symbols, source_manifest):
        self.model, self.stats = model.eval().requires_grad_(False), stats
        self.gene_symbols = tuple(map(str, gene_symbols))
        self.source_manifest = dict(source_manifest)
        self.device = next(model.parameters()).device

    def encode_image_spots(self, uni2_features, xy_um, slide_ids, masks, *, batch_size=512):
        """Batched, same-slide real-coordinate images; no transcriptome input.

        Input rows must already have the registered stateless unit-L2 transform.
        Absolute coordinates/slide IDs construct neighborhoods but are not model
        features. Padding output is zero and carries an explicit false mask.
        """
        x = as_numpy(uni2_features).astype(np.float32, copy=False)
        xy, slide, valid = as_numpy(xy_um).astype(float), as_numpy(slide_ids), as_numpy(masks).astype(bool)
        require(x.ndim == 3 and x.shape[-1] == 1536 and xy.shape == (*x.shape[:2], 2)
                and slide.shape == valid.shape == x.shape[:2] and batch_size >= 1,
                "FROZEN_IMAGE_SPOT_BATCH_SHAPES")
        b, n = valid.shape
        embeddings, prediction_z = np.zeros((b, n, 4, 128), np.float32), np.zeros((b, n, 4), np.float32)
        for patient in range(b):
            rows = np.flatnonzero(valid[patient])
            if not len(rows):
                continue
            feature = x[patient, rows]
            coordinate, section = xy[patient, rows], slide[patient, rows]
            neighbors = build_neighbors(coordinate, section, radius_um=300., maximum_neighbors=16)
            with torch.no_grad():
                for start in range(0, len(rows), batch_size):
                    local = np.arange(start, min(start + batch_size, len(rows)))
                    graph = neighbors[local]; present = graph >= 0; safe = np.maximum(graph, 0)
                    batch = ImageBatch(torch.as_tensor(feature[local], device=self.device),
                        torch.as_tensor(np.where(present[..., None], feature[safe], 0.).astype(np.float32), device=self.device),
                        torch.as_tensor(np.where(present[..., None], coordinate[safe] - coordinate[local, None], 0.).astype(np.float32), device=self.device),
                        torch.as_tensor(present, device=self.device))
                    output = self.model.image_only_predict(batch)
                    latent = output.get("visual_program_tokens")
                    if latent is None:
                        # A1 has no learned program slots; do not invent four learned semantics.
                        latent = output["image_embedding"][:, None, :].expand(-1, 4, -1)
                    embeddings[patient, rows[local]] = latent.cpu().numpy()
                    prediction_z[patient, rows[local]] = output["program_predictions"].cpu().numpy()
        prediction = prediction_z * self.stats.target_sd + self.stats.target_mean
        prediction = np.where(valid[..., None], prediction, 0.)
        return dict(program_embeddings=embeddings, program_pred_z=prediction_z,
            program_pred=prediction, mask=valid, native_learned_program_slots=self.model.has_slots,
            source_manifest_sha256=canonical_hash(self.source_manifest), transcriptome_used=False,
            tissue_truth_used=False, absolute_coordinates_are_model_inputs=False)

    def image_only_program_map(self, uni2_features, xy_um, slide_ids, masks, **kwargs):
        output = self.encode_image_spots(uni2_features, xy_um, slide_ids, masks, **kwargs)
        return dict(program_pred=output["program_pred"], program_pred_z=output["program_pred_z"],
            mask=output["mask"], program_order=list(PROGRAMS), lesion_segmentation_truth_claimed=False)

    def encode_bulk_genes(self, rna, rna_mask, fixed_membership, source_scaler_contract):
        """Bulk values are caller train-standardized; source spot scales are never applied."""
        require(self.model.has_genes, "BASELINE_HAS_NO_TRAINED_MOLECULAR_ENCODER")
        contract = dict(source_scaler_contract)
        require(contract.get("schema") == "spgel_bulk_scale_contract_v1"
                and contract.get("fit_scope") == "current_R8_train_only"
                and contract.get("uses_source_spot_scaler") is False
                and contract.get("values_are_current_train_gene_Z") is True,
                "BULK_MUST_HAVE_SEPARATE_TRAIN_ONLY_SCALE_CONTRACT")
        require(fixed_membership == self.source_manifest["source_gene_membership_sha256"],
                "BULK_FIXED_MEMBERSHIP_CHANGED")
        values, observed = as_numpy(rna).astype(np.float32, copy=False), as_numpy(rna_mask).astype(bool)
        require(values.ndim == 2 and observed.shape == values.shape and values.shape[1] == len(self.gene_symbols),
                "BULK_VALUES_REQUIRE_EXPLICIT_SOURCE_GENE_ORDER_AND_MASK")
        require(contract.get("gene_order_sha256") == canonical_hash(list(self.gene_symbols)),
                "BULK_GENE_ORDER_CHANGED")
        with torch.no_grad():
            batch = GeneBatch(torch.as_tensor(np.where(observed, values, 0.), device=self.device),
                torch.as_tensor(observed, device=self.device), torch.arange(values.shape[1], device=self.device))
            result = self.model.encode_gene_programs(batch, paired_mode=True)
        return dict(program_tokens=result["tokens"].cpu().numpy(), mask=result["program_mask"].cpu().numpy(),
            coverage=result["coverage"].cpu().numpy(), source_spot_scaler_applied=False,
            bulk_and_spot_measurements_equivalent_claimed=False, coverage_is_calibrated_confidence=False)

    def route_bulk_to_wsi(self, program_token, patch_emb, xy_um, masks):
        require(self.model.bidirectional, "BASELINE_HAS_NO_TRAINED_CONDITIONAL_IMAGE_QUERY")
        token = torch.as_tensor(program_token, dtype=torch.float32, device=self.device)
        patch = torch.as_tensor(patch_emb, dtype=torch.float32, device=self.device)
        valid = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        xy = as_numpy(xy_um).astype(float)
        require(patch.ndim == 3 and patch.shape[-1] == 128 and token.shape == (len(patch), 4, 128)
                and valid.shape == patch.shape[:2] and xy.shape == (*patch.shape[:2], 2)
                and np.isfinite(xy[as_numpy(masks).astype(bool)]).all()
                and torch.isfinite(patch[valid]).all() and torch.isfinite(token).all(),
                "WSI_PREENCODED_VALUE_OR_GEOMETRY_SHAPE")
        with torch.no_grad():
            q = self.model.window.queries + self.model.gene_to_image(token)
            result = self.model.window(patch, valid, query=q, return_attention=True)
        return dict(program_embeddings=result["tokens"].cpu().numpy(),
            attention=result["attention"].cpu().numpy(), mask=result["program_mask"].cpu().numpy(),
            absolute_coordinates_used_as_content=False, semantic_correspondence_is_not_histology_truth=True)


def load_frozen_interface(directory, *, device="cpu"):
    root = Path(directory)
    manifest = json.loads((root / "source_encoder_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema") == "spgel_frozen_source_encoder_manifest_v1"
            and manifest.get("status") == "COMPLETE_REPLAYED_SOURCE_ENCODER_INTERFACE"
            and manifest.get("checkpoint_selected_using_DSS") is False
            and manifest.get("R8_requires_explicit_new_source_version") == "R8_SOURCE_SPGEL_V1",
            "FROZEN_SOURCE_INTERFACE_MANIFEST_INVALID")
    require(canonical_hash({k: v for k, v in manifest.items()
            if k not in ("source_manifest_core_sha256", "encoder_state_sha256")}) == manifest["source_manifest_core_sha256"],
            "FROZEN_SOURCE_MANIFEST_CORE_CHANGED")
    file = root / "encoder_state.pt"
    require(sha256(file) == manifest["encoder_state_sha256"], "FROZEN_ENCODER_STATE_SHA_CHANGED")
    state = torch.load(file, map_location="cpu", weights_only=False)
    require(state.get("schema") == "spgel_frozen_source_encoder_state_v1"
            and state["source_manifest_core_sha256"] == manifest["source_manifest_core_sha256"],
            "FROZEN_ENCODER_STATE_MANIFEST_BINDING")
    stats = SourceStatistics(**state["source_statistics"])
    model = SpatialProgramEncoder(state["arm"], state["program_up"], state["program_down"], state["gene_symbols"],
        width=128, hidden=128, image_mlp_hidden=527, tissue_program_mean=stats.tissue_program_mean, seed=31).to(device)
    model.load_state_dict(state["state_dict"], strict=True)
    return FrozenSpatialProgramInterface(model, stats, gene_symbols=state["gene_symbols"], source_manifest=manifest)

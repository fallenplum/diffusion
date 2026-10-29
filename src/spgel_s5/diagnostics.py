"""Small frozen TASK-B interventions, with no target/tissue/depth access."""
from __future__ import annotations

import numpy as np
import torch

from ..spgel.contracts import PROGRAMS, array_hash, canonical_hash, require


def _tensor_fingerprint(value):
    return array_hash(value.detach().cpu().numpy())


def _state_fingerprint(model):
    return canonical_hash({name: _tensor_fingerprint(value)
                           for name, value in model.state_dict().items()})


def _gradient_fingerprint(model):
    return canonical_hash({name: None if value.grad is None else _tensor_fingerprint(value.grad)
                           for name, value in model.named_parameters()})


def _rms_difference(first, second, valid):
    require(first.shape == second.shape and first.shape[:2] == valid.shape
            and first.ndim == 3 and first.shape[1] == 4, "S5_DIAGNOSTIC_TOKEN_SHAPES")
    delta = (first - second).to(torch.float64)
    require(torch.isfinite(delta[valid]).all(), "S5_DIAGNOSTIC_NONFINITE_TOKEN_DIFFERENCE")
    def value(mask):
        return float(delta[mask].square().mean().sqrt().item()) if mask.any() else None
    return dict(RMS=value(valid), valid_spot_programs=int(valid.sum().item()),
        per_program=[dict(program_index=k, valid_spots=int(valid[:, k].sum().item()),
            RMS=value(valid & (torch.arange(4, device=valid.device)[None, :] == k))) for k in range(4)])


def _attention_difference(first, second, valid):
    require(first.shape == second.shape and first.shape[:2] == valid.shape
            and first.ndim == 3, "S5_DIAGNOSTIC_ATTENTION_SHAPES")
    delta = (first - second).to(torch.float64)
    require(torch.isfinite(delta[valid]).all(), "S5_DIAGNOSTIC_NONFINITE_ATTENTION_DIFFERENCE")
    l1 = delta.abs().sum(-1)
    def mean(mask):
        return float(l1[mask].mean().item()) if mask.any() else None
    return dict(mean_row_L1=mean(valid), valid_spot_programs=int(valid.sum().item()),
        maximum_absolute_difference=float(delta[valid].abs().max().item()) if valid.any() else None,
        per_program=[dict(program_index=k, valid_spots=int(valid[:, k].sum().item()),
            mean_row_L1=mean(valid & (torch.arange(4, device=valid.device)[None, :] == k)))
            for k in range(4)])


def _scores(own, other):
    valid = own["valid"][:, 0] & other["valid"][:, 0]
    delta = (other["scores"][:, 0] - own["scores"][:, 0]).to(torch.float64)
    require(torch.isfinite(delta[valid]).all(), "S5_DIAGNOSTIC_NONFINITE_SCORE_DIFFERENCE")
    return dict(valid_pairs=int(valid.sum().item()),
        mean_difference=float(delta[valid].mean().item()) if valid.any() else None,
        mean_absolute_difference=float(delta[valid].abs().mean().item()) if valid.any() else None,
        RMS=float(delta[valid].square().mean().sqrt().item()) if valid.any() else None)


def paired_diagnostics(model, source, rows, stats, device="cpu"):
    """Three frozen candidate-specific pair runs; no optimizer or backward.

    Only features, measured genes/masks, relative window geometry and supplied
    training-fitted gene scaling are used. No target/tissue/depth is inspected.
    Index hashes are provenance only and never enter the model as features.
    """
    rows = np.asarray(rows)
    require(rows.ndim == 1 and rows.dtype.kind in "iu" and len(rows) >= 2
            and len(np.unique(rows)) == len(rows)
            and ((rows >= 0) & (rows < len(source.features))).all(),
            "S5_DIAGNOSTIC_REQUIRES_DISTINCT_PARTITION_ROWS")
    positions = np.linspace(0, len(rows) - 1, min(128, len(rows)), dtype=np.int64)
    probe = np.asarray(rows[positions], dtype=np.int64)
    require(len(np.unique(probe)) == len(probe), "S5_DIAGNOSTIC_DUPLICATE_PROBE")
    n = len(probe)
    own = torch.arange(n, dtype=torch.long, device=device)[:, None]
    rolled = own.roll(1, 0)
    require(not torch.equal(own, rolled), "S5_DIAGNOSTIC_ROLL_MUST_CHANGE_PAIRS")
    state_before, gradients_before = _state_fingerprint(model), _gradient_fingerprint(model)
    training_before = {name: module.training for name, module in model.named_modules()}
    model.eval()
    try:
        with torch.inference_mode():
            image = source.image_batch(probe, allowed_rows=rows, device=device)
            genes = stats.genes_z(source, probe, device=device)
            output = model.image_only_predict(image, return_attention=True)
            molecular = model.gene_encoder(genes, return_attention=True)
            def pair(ir, gr):
                return model.paired_scores(output, genes, ir, gr, paired_mode=True,
                    return_details=True, _gene_base=molecular)
            matched = pair(own, own)
            wrong_gene = pair(own, rolled)
            wrong_image = pair(rolled, own)
            base_v, base_g = output["visual_program_tokens"], molecular["tokens"]
            image_valid = output["window_mask"].any(-1)[:, None].expand(n, 4)
            gene_valid = molecular["program_mask"]
            both = image_valid & gene_valid
            gene_change_valid = both & gene_valid[rolled[:, 0]]
            image_change_valid = both & image_valid[rolled[:, 0]]
            def token(run, name):
                return run[name][:, 0]
            def attention(run, name, baseline):
                found = run.get(name)
                return baseline if found is None else found[:, 0]
            base_ia = output["program_attention"]
            base_gu, base_gd = molecular["up_attention"], molecular["down_attention"]
            matched_ia = attention(matched, "image_attention", base_ia)
            matched_gu = attention(matched, "gene_up_attention", base_gu)
            matched_gd = attention(matched, "gene_down_attention", base_gd)
            result = dict(schema="spgel_s5_frozen_pair_intervention_v1", status="COMPLETE_THREE_FROZEN_TASK_B_RUNS",
                probe_count=n, partition_rows_count=len(rows),
                selection="whole_partition_row_equidistant_at_most_128_without_labels",
                partition_rows_sha256=array_hash(np.asarray(rows, dtype=np.int64)),
                probe_rows_sha256=array_hash(probe), rolled_local_mapping_sha256=_tensor_fingerprint(rolled),
                program_order=list(PROGRAMS),
                gene_to_image=dict(
                    base_to_conditioned=_rms_difference(token(matched, "conditioned_image_tokens"), base_v, both),
                    changed_gene_context=_rms_difference(token(wrong_gene, "conditioned_image_tokens"),
                        token(matched, "conditioned_image_tokens"), gene_change_valid),
                    base_attention_change=_attention_difference(matched_ia, base_ia, both),
                    changed_gene_attention=_attention_difference(
                        attention(wrong_gene, "image_attention", base_ia), matched_ia, gene_change_valid)),
                image_to_gene=dict(
                    base_to_conditioned=_rms_difference(token(matched, "conditioned_gene_tokens"), base_g, both),
                    changed_image_context=_rms_difference(token(wrong_image, "conditioned_gene_tokens"),
                        token(matched, "conditioned_gene_tokens"), image_change_valid),
                    base_up_attention_change=_attention_difference(matched_gu, base_gu, both),
                    base_down_attention_change=_attention_difference(matched_gd, base_gd, both),
                    changed_image_up_attention=_attention_difference(
                        attention(wrong_image, "gene_up_attention", base_gu), matched_gu, image_change_valid),
                    changed_image_down_attention=_attention_difference(
                        attention(wrong_image, "gene_down_attention", base_gd), matched_gd, image_change_valid)),
                paired_score_change=dict(rolled_genes=_scores(matched, wrong_gene),
                                         rolled_images=_scores(matched, wrong_image)),
                image_only_forward_calls=1, gene_base_forward_calls=1, paired_score_calls=3,
                optimizer_instances=0, optimizer_updates=0, backward_calls=0,
                target_tissue_library_or_survival_reads=False,
                matching_is_tissue_or_depth_controlled=False,
                medical_or_cell_source_ground_truth_claimed=False,
                pair_score_change_is_not_isolated_routing_effect=True,
                raw_identifiers_embeddings_attention_or_predictions_returned=False,
                undefined_measurements_are_null_not_imputed=True,
                finite_observed_differences=True)
    finally:
        for name, module in model.named_modules():
            module.training = training_before[name]
        require(state_before == _state_fingerprint(model), "S5_DIAGNOSTIC_MUTATED_MODEL_STATE")
        require(gradients_before == _gradient_fingerprint(model), "S5_DIAGNOSTIC_MUTATED_EXISTING_GRADIENTS")
    result.update(model_state_sha256=state_before, model_state_unchanged=True,
                  existing_gradients_unchanged=True, training_modes_restored=True)
    return result

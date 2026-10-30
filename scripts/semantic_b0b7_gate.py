"""Real-input, zero-optimizer P0 group gate for the first P1 qualification.

This gate never authorizes a survival fit. It proves bounded input/interface
and gradient properties; it is not evidence of prognostic usefulness.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.semantic_b0b7.models import ARMS, ImageInputs, ModelConfig, OmicsInputs, SemanticRepresentation
from src.semantic_b0b7.pretext import PairMetadata, auxiliary_loss, matched_derangement, negative_support


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


SOURCE_FILES = (
    "src/semantic_b0b7/__init__.py", "src/semantic_b0b7/models.py", "src/semantic_b0b7/pretext.py",
    "src/semantic_b0b7/data.py", "src/semantic_b0b7/survival.py", "scripts/semantic_b0b7_gate.py",
    "scripts/semantic_b0b7_prepare.py", "scripts/semantic_b0b7_audit.py", "scripts/semantic_b0b7_p1.py",
    "configs/semantic_b0b7/p0_p1_v1.json", "configs/semantic_b0b7/allowed_relations_v1.json",
    "configs/phase6a_titan_crc_morphology_concepts.json",
    "scripts/semantic_b0b7_titan_worker.py",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True, type=Path)
    p.add_argument("--expected-input-sha256", required=True)
    p.add_argument("--data-audit", required=True, type=Path)
    p.add_argument("--alignment-audit", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise ValueError("do not overwrite a previous gate")
    assert sha(args.inputs) == args.expected_input_sha256
    source = {rel: sha(ROOT / rel) for rel in SOURCE_FILES}
    source_digest = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    data_audit = json.loads(args.data_audit.read_text())
    assert data_audit["status"] == "PASS_DATA_BYTE_INTEGRITY_ONLY"
    assert data_audit["original_CONCH_H5_verified"] == 519
    alignment = json.loads(args.alignment_audit.read_text())
    assert alignment["status"] == "COMPLETE_FROZEN_INPUTS_ONLY", alignment["status"]
    assert alignment["conch_projection"]["real_slides_checked"] >= 32
    assert alignment["conch_projection"]["official_vs_literal_max_abs_error"] <= 1e-6
    assert alignment["titan_zero_shot"]["real_slide_embeddings_checked"] >= 32
    assert alignment["titan_zero_shot"]["official_zero_shot_max_abs_error"] <= 2e-6
    value = torch.load(args.inputs, map_location="cpu", weights_only=False)
    forbidden = {"DSS", "DSS_time", "DSS_event", "PFI", "t", "e", "time", "event", "outcomes"}
    assert not forbidden.intersection(value)
    train, valid = np.asarray(value["train_indices"], int), np.asarray(value["valid_indices"], int)
    assert len(train) + len(valid) == 409 and not np.intersect1d(train, valid).size
    assert set(np.r_[train, valid]) == set(range(409))
    assert value["target"].shape == value["target_mask"].shape == (409, 50)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(31)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    def tensor(x):
        return torch.as_tensor(x, device=args.device)
    batch_image = ImageInputs(**{k: tensor(v)[train] for k, v in value["image"].items()})
    batch_genes = OmicsInputs(**{k: tensor(v)[train] for k, v in value["omics"].items()})
    target, measured = tensor(value["target"])[train], tensor(value["target_mask"])[train]
    metadata = PairMetadata(**{k: np.asarray(v)[train] for k, v in value["metadata"].items()})
    negatives, support = negative_support(metadata, mode="matched")
    wrong, pairing = matched_derangement(metadata, seed=103103)
    assert (wrong != np.arange(len(train))).all()
    # Wrong auxiliary positives cannot silently remove additional negatives
    # in B6a alone. Use this same train-only support for every probe arm.
    negatives[np.arange(len(train)), wrong] = False
    negatives = tensor(negatives)
    config = ModelConfig(clinical_dim=batch_image.clinical.shape[1])
    parameters, controls, initial = {}, [], {}
    for arm in ARMS:
        torch.manual_seed(31)
        model = SemanticRepresentation(arm, tensor(value["conch_prototypes"]), tensor(value["titan_prototypes"]),
            tensor(value["relation_prior"]), config=config, seed=31).to(args.device)
        parameters[arm] = model.parameter_counts()
        if arm in ("B3", "B4"):
            initial[arm] = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        image = replace(batch_image, patch_values=batch_image.patch_values.detach().clone().requires_grad_(),
                        clinical=batch_image.clinical.detach().clone().requires_grad_())
        genes = OmicsInputs(batch_genes.values.detach().clone().requires_grad_(), batch_genes.measured)
        result = model(image, genes, stage="pretext")
        loss = auxiliary_loss(result, target, measured, negatives,
                              positive=tensor(wrong).long() if arm == "B6a" else None)
        assert torch.isfinite(loss["loss"])
        loss["loss"].backward()
        missing_grad = [name for name, parameter in model.named_parameters()
                        if parameter.requires_grad and (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                                                       or not bool(parameter.grad.abs().sum() > 0))]
        assert not missing_grad, (arm, missing_grad)
        gradients = {"patch": float(image.patch_values.grad.norm()),
                     "omics": float(genes.values.grad.norm()), "clinical": float(image.clinical.grad.norm())}
        assert all(np.isfinite(v) and v > 0 for v in gradients.values()), (arm, gradients)
        model.eval()
        with torch.no_grad():
            normal = model(image, genes)
            swapped = model(image, OmicsInputs(genes.values.flip(0), genes.measured.flip(0)))
            oracle_error = float((normal["program_prediction"] - swapped["program_prediction"]).abs().max())
            query_probe = None
            if arm == "B4":
                saved = model.conch_prototypes.detach().clone()
                model.conch_prototypes[0].copy_(saved[1])
                changed = model(image, genes)
                query_probe = {"operation": "replace_one_CONCH_group_with_another_real_group_only",
                               "patch_attention_max_delta": float((changed["patch_attention"] - normal["patch_attention"]).abs().max()),
                               "program_prediction_max_delta": float((changed["program_prediction"] - normal["program_prediction"]).abs().max()),
                               "TITAN_query_evidence_max_delta": float((changed["titan_query_evidence"] - normal["titan_query_evidence"]).abs().max()),
                               "omics_token_max_delta": float((changed["gene_tokens"] - normal["gene_tokens"]).abs().max())}
                model.conch_prototypes.copy_(saved)
                assert query_probe["patch_attention_max_delta"] > 0 and query_probe["program_prediction_max_delta"] > 0
                assert query_probe["TITAN_query_evidence_max_delta"] == query_probe["omics_token_max_delta"] == 0
                del changed
        assert oracle_error == 0, arm
        controls.append({"arm": arm, "parameters": parameters[arm], "input_gradient_norms": gradients,
                         "unused_or_nonfinite_gradient_tensors": len(missing_grad), "image_only_omics_swap_error": oracle_error,
                         "text_query_real_input_perturbation": query_probe,
                         "matched_loss_anchors": loss["pair"]["image_anchors_used"]})
        del result, loss, normal, swapped, model, image, genes
    assert set(initial["B3"]) == set(initial["B4"])
    assert all(torch.equal(initial["B3"][name], initial["B4"][name]) for name in initial["B3"])
    assert parameters["B3"] == parameters["B4"]
    assert parameters["B5"] == parameters["B7"]
    assert parameters["B4"]["total"] == parameters["B2_MATCHED_1H"]["total"] == parameters["B2_MATCHED_4H"]["total"]
    evidence = {"optimizer_steps": 0, "fits": 0, "real_forward_backward_models": len(controls),
                "actual_forward_backward": True, "gradients_finite": True,
                "input_sha256": args.expected_input_sha256, "source_sha256": source_digest,
                "initial_shared_parameter_identity_B3_B4": True, "controls": controls,
                "matched_negative_support": support, "wrong_positive_mapping": pairing,
                "GPU_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0,
                "seconds": time.monotonic() - started}
    gate = {"schema": "semantic_b0b7_p0_gate_v1", "status": "PASS", "scope": "P1_AUXILIARY_QUALIFICATION_ONLY",
            "input_sha256": args.expected_input_sha256, "frozen_sources": source, "source_sha256": source_digest,
            "hard_gates": {name: {"status": "PASS"} for name in (
                "data_endpoint_fold_alignment", "official_text_alignment", "prototype_and_relation_contract",
                "real_forward_backward", "outcome_blind_scope")}, "probe_evidence": evidence,
            "data_audit_sha256": sha(args.data_audit), "alignment_audit_sha256": sha(args.alignment_audit),
            "P2_DSS_authorized": False, "P2_anchor_rebuild": "DEFERRED_NO_REAL_DSS_FIT_THIS_ROUND",
            "scientific_mechanism_supported": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": gate["status"], "scope": gate["scope"], "models": len(controls),
                      "optimizer_steps": 0, "seconds": evidence["seconds"], "gate_sha256": sha(args.output)}), flush=True)


if __name__ == "__main__":
    main()

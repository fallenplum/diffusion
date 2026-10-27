"""Outcome-free external tissue/program supervision for UNI2-h projection.

The four spatial program scores are measured local expression proxies. Tissue
categories are independently annotated tissue classes, not cell-source truth.
Only a 1536->64 Linear/LayerNorm/GELU projection transfers into the TCGA model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.nn import functional as F


PROGRAMS = ("iCMS2", "iCMS3", "FAP_fibroblast_associated_top25", "MARCO_macrophage_associated_top25")
INPUT_NORMALIZATION = "float32_per_row_L2_norm_division; reject_norm_le_1e-12"
PROTOCOL = dict(schema="bgb_en_external_projection_v2_unitnorm", input_dim=1536, width=64, programs=4,
                tissues=3, epochs=20, optimizer="Adam", learning_rate=.001, weight_decay=.0001,
                program_loss="natural-weighted mean four-target Huber, delta=1",
                tissue_loss="source-fit-derived class-balanced cross entropy", tissue_weight=.1,
                sample_weight="equal donor, then equal section, then equal spot",
                target_scaling="natural-weighted source-fit mean and population SD",
                batch="complete current source-fit", epoch_selection="none; fixed final epoch20",
                supervised_fits=7, transfer="projection only", neighbor_features_used=False,
                input_normalization=INPUT_NORMALIZATION,
                TCGA_DSS_read=False, original_ST_test_read=False)


def require(value: Any, message: str) -> None:
    if not bool(value):
        raise ValueError(message)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str(array.dtype).encode()+str(array.shape).encode()+array.tobytes()).hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def natural_weights(donor: np.ndarray, section: np.ndarray) -> np.ndarray:
    donor, section = np.asarray(donor).astype(str), np.asarray(section).astype(str)
    require(donor.ndim == 1 and section.shape == donor.shape and len(donor) > 0, "nonempty donor/section vectors required")
    result = np.zeros(len(donor), dtype=np.float64)
    for identity in np.unique(donor):
        rows = np.flatnonzero(donor == identity)
        sections = np.unique(section[rows])
        for slide in sections:
            indices = rows[section[rows] == slide]
            result[indices] = 1/(len(np.unique(donor))*len(sections)*len(indices))
    require(np.isclose(result.sum(), 1) and (result > 0).all(), "invalid hierarchical source weights")
    return result


def fit_target_scaler(target: np.ndarray, weight: np.ndarray) -> dict[str, np.ndarray]:
    y, w = np.asarray(target, np.float64), np.asarray(weight, np.float64)
    require(y.ndim == 2 and y.shape[1] == 4 and w.shape == (len(y),), "four targets and one weight per source row required")
    require(np.isfinite(y).all() and np.isfinite(w).all() and (w > 0).all() and np.isclose(w.sum(), 1), "invalid targets or weights")
    mean = np.sum(y*w[:, None], axis=0)
    scale = np.sqrt(np.sum((y-mean)**2*w[:, None], axis=0))
    require((scale > 1e-12).all() and np.isfinite(scale).all(), "source-fit contains constant/nonfinite program target")
    return dict(mean=mean, scale=scale)


def balanced_class_weights(tissue: np.ndarray, weight: np.ndarray) -> np.ndarray:
    y, w = np.asarray(tissue), np.asarray(weight, np.float64)
    require(y.shape == w.shape and np.isin(y, [0, 1, 2]).all(), "three canonical tissue labels required")
    mass = np.bincount(y.astype(int), weights=w, minlength=3)
    require((mass > 0).all(), "source-fit must contain all three tissue classes")
    return 1/(3*mass)


class ExternalProgramModel(nn.Module):
    def __init__(self, input_dim: int = 1536, width: int = 64):
        super().__init__()
        # Exactly matches PathologyEncoder(P1).project. Source and target loaders
        # both perform the same float32 per-row L2 normalization before it.
        self.project = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        self.program = nn.Linear(width, 4)
        self.tissue = nn.Linear(width, 3)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.project(features)
        return self.program(embedding), self.tissue(embedding)


def source_loss(program_pred: torch.Tensor, tissue_logits: torch.Tensor,
                target_z: torch.Tensor, tissue: torch.Tensor,
                sample_weight: torch.Tensor, class_weight: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    program_loss = (F.huber_loss(program_pred, target_z, delta=1., reduction="none").mean(-1)*sample_weight).sum()
    tissue_loss = (F.cross_entropy(tissue_logits, tissue, weight=class_weight, reduction="none")*sample_weight).sum()
    return program_loss+.1*tissue_loss, dict(program_huber=program_loss, tissue_CE=tissue_loss)


def normalize_source_features(features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Same operation order/dtype as read_h5_slide(normalize=True); raw untouched."""
    raw = np.asarray(features)
    require(raw.ndim == 2 and raw.dtype == np.float32 and np.isfinite(raw).all(),
            "source feature normalization requires finite float32 rows")
    norm = np.linalg.norm(raw, axis=1)
    require((norm > 1e-12).all() and np.isfinite(norm).all(), "source feature has zero/nonfinite L2 norm")
    normalized = raw/norm[:, None]
    result_norm = np.linalg.norm(normalized, axis=1)
    require(np.allclose(result_norm, 1., atol=2e-6, rtol=0), "derived source L2 normalization failed")
    normalized.setflags(write=False)
    return normalized, dict(definition=INPUT_NORMALIZATION, raw_median_L2=float(np.median(norm)),
                            raw_min_L2=float(norm.min()), raw_max_L2=float(norm.max()),
                            normalized_median_L2=float(np.median(result_norm)),
                            normalized_max_abs_unit_error=float(np.max(np.abs(result_norm-1))),
                            raw_array_modified=False, normalization_fitted_to_any_labels=False,
                            derived_normalized_feature_sha256=array_hash(normalized))


def load_qualified_source(data_path: str | Path, manifest_path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only the exact Stage-B prepared source; no TCGA or held ST assets."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    require(manifest.get("schema_version") == "sami_stage_b_prepared_v1"
            and manifest.get("status") == "PASS_PREPARED_SIGNED_PROGRAMS", "source manifest is not qualified Stage B")
    require(manifest.get("original_ST_test_read") is False and manifest.get("TCGA_DSS_read") is False,
            "source manifest does not exclude TCGA/DSS and historical ST-test")
    require(tuple(manifest.get("program_order", ())) == PROGRAMS, "fixed external program order changed")
    digest = sha256(data_path)
    require(manifest.get("prepared_data_sha256") == digest and manifest.get("data_sha256") == digest,
            "source prepared data SHA differs from manifest")
    with np.load(data_path, allow_pickle=False) as archive:
        required = ("X", "program_y", "tissue_y", "donor", "section")
        require(set(required) <= set(archive.files), "source archive missing required arrays")
        # No endpoint, barcode, coordinate or neighbor-X column is materialized.
        data = {key: np.array(archive[key], copy=True) for key in required}
    x, y, tissue = data["X"], data["program_y"], data["tissue_y"]
    require(x.shape == (13011, 1536) and x.dtype == np.float32 and np.isfinite(x).all(), "source UNI2-h feature contract differs")
    require(y.shape == (len(x), 4) and np.isfinite(y).all(), "source programs invalid")
    require(tissue.shape == (len(x),) and np.issubdtype(tissue.dtype, np.integer)
            and set(np.unique(tissue)) == {0, 1, 2}, "source tissue classes invalid")
    for key in ("donor", "section"):
        require(data[key].shape == (len(x),) and data[key].dtype.kind in "US", "source donor/section vector invalid")
    require(len(np.unique(data["donor"])) == 6 and len(np.unique(data["section"])) == 12, "source donor/section coverage changed")
    for section in np.unique(data["section"]):
        require(len(np.unique(data["donor"][data["section"] == section])) == 1, "source section crosses donors")
    require(manifest.get("n_spots") == len(x) and manifest.get("donors") == 6 and manifest.get("sections") == 12
            and manifest.get("feature_dimension") == 1536, "source manifest counts disagree")
    data["X"], norm_audit = normalize_source_features(x)
    audit = dict(status="PASS_EXTERNAL_SOURCE_QUALIFICATION", source_data_sha256=digest,
                 source_manifest_sha256=sha256(manifest_path), spots=len(x), feature_dimension=1536,
                 donors=6, sections=12, program_order=list(PROGRAMS),
                 tissue_counts=np.bincount(tissue, minlength=3).tolist(),
                 source_program_definition_sha256=manifest.get("program_definition_sha256"),
                 TCGA_DSS_read=False, original_ST_test_read=False, neighbor_X_used=False,
                 actual_supervised_fits=0, true_cell_source_labels_available=False,
                 input_normalization=norm_audit,
                 output_boundary="tissue-class and local program supervision; no survival or true cell-source claim")
    return data, audit


def source_partitions(data: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    donors = np.unique(data["donor"])
    require(len(donors) == 6, "exact six source donors required")
    result = []
    for index, donor in enumerate(donors):
        result.append(dict(fit_index=index, role="leave_one_donor_out",
                           train=np.flatnonzero(data["donor"] != donor),
                           valid=np.flatnonzero(data["donor"] == donor)))
    result.append(dict(fit_index=6, role="full_source_transfer", train=np.arange(len(data["X"])),
                       valid=np.empty(0, np.int64)))
    return result


def _metrics(data: dict[str, np.ndarray], index: np.ndarray, program: np.ndarray,
             logits: np.ndarray, scaler: dict[str, np.ndarray]) -> dict[str, Any]:
    weight = natural_weights(data["donor"][index], data["section"][index])
    true_z = (data["program_y"][index]-scaler["mean"])/scaler["scale"]
    difference = np.abs(program-true_z)
    huber = np.where(difference <= 1., .5*difference**2, difference-.5)
    pred = logits.argmax(-1)
    tissue = data["tissue_y"][index]
    recall = [float(np.mean(pred[tissue == label] == label)) if np.any(tissue == label) else None for label in range(3)]
    correlations = []
    within_tissue = []
    for k in range(4):
        truth, estimate = true_z[:, k], program[:, k]
        correlations.append(float(spearmanr(truth, estimate).statistic) if np.ptp(truth) > 0 and np.ptp(estimate) > 0 else 0.)
        by_tissue = []
        for label in np.unique(tissue):
            by_section = []
            for section in np.unique(data["section"][index][tissue == label]):
                choose = (tissue == label) & (data["section"][index] == section)
                if choose.sum() >= 20 and np.ptp(truth[choose]) > 0:
                    by_section.append(float(spearmanr(truth[choose], estimate[choose]).statistic) if np.ptp(estimate[choose]) > 0 else 0.)
            if by_section:
                by_tissue.append(float(np.mean(by_section)))
        within_tissue.append(float(np.mean(by_tissue)) if by_tissue else None)
    return dict(program_standardized_Huber=float(np.dot(weight, huber.mean(-1))),
                program_standardized_MSE=float(np.dot(weight, ((program-true_z)**2).mean(-1))),
                per_program_Spearman=correlations,
                per_program_within_section_tissue_Spearman=within_tissue,
                tissue_balanced_accuracy=float(np.mean([v for v in recall if v is not None])),
                tissue_recall=recall, rows=len(index))


def fit_source_partition(data: dict[str, np.ndarray], train: np.ndarray, valid: np.ndarray,
                         *, device: str = "cuda", seed: int = 9033011,
                         epochs: int = 20) -> dict[str, Any]:
    """Exactly 20 complete-source-fit optimizer steps; no validation selection."""
    require(epochs == 20, "source budget is fixed at 20 epochs; validation selection is prohibited")
    train, valid = np.asarray(train, np.int64), np.asarray(valid, np.int64)
    require(len(train) > 0 and len(np.unique(train)) == len(train) and not np.intersect1d(train, valid).size,
            "source fit/heldout row overlap")
    require(not np.intersect1d(data["donor"][train], data["donor"][valid]).size, "source fit/heldout donor overlap")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        require(torch.cuda.is_available(), "CUDA requested but unavailable")
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model = ExternalProgramModel(input_dim=data["X"].shape[1]).to(device)
    weight = natural_weights(data["donor"][train], data["section"][train])
    scaler = fit_target_scaler(data["program_y"][train], weight)
    cw = balanced_class_weights(data["tissue_y"][train], weight)
    x = torch.as_tensor(data["X"][train], device=device)
    y = torch.as_tensor((data["program_y"][train]-scaler["mean"])/scaler["scale"], device=device, dtype=torch.float32)
    tissue = torch.as_tensor(data["tissue_y"][train], device=device, dtype=torch.long)
    w = torch.as_tensor(weight, device=device, dtype=torch.float32)
    class_weight = torch.as_tensor(cw, device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    start = time.monotonic()
    curve = []
    for epoch in range(1, epochs+1):
        optimizer.zero_grad(set_to_none=True)
        program, logits = model(x)
        loss, components = source_loss(program, logits, y, tissue, w, class_weight)
        require(torch.isfinite(loss), "nonfinite source training loss")
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                "nonfinite or missing source training gradient")
        optimizer.step()
        curve.append(dict(epoch=epoch, loss=float(loss.detach()),
                          program_huber=float(components["program_huber"].detach()),
                          tissue_CE=float(components["tissue_CE"].detach())))
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic()-start
    model.eval()
    with torch.no_grad():
        program, logits = model(x)
        train_metric = _metrics(data, train, program.cpu().numpy(), logits.cpu().numpy(), scaler)
        valid_metric, valid_pred = None, None
        if len(valid):
            vp, vt = model(torch.as_tensor(data["X"][valid], device=device))
            valid_pred = dict(program_standardized=vp.cpu().numpy(), tissue_logits=vt.cpu().numpy())
            valid_metric = _metrics(data, valid, valid_pred["program_standardized"], valid_pred["tissue_logits"], scaler)
    return dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                project_state_dict={k: v.detach().cpu().clone() for k, v in model.project.state_dict().items()},
                target_scaler=scaler, class_weights=cw, train_metrics=train_metric,
                valid_metrics=valid_metric, valid_prediction=valid_pred, curve=curve,
                registered_parameters=sum(p.numel() for p in model.parameters()), epochs=20,
                optimizer_updates=20, elapsed_seconds=elapsed,
                peak_GPU_bytes=torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else 0)


def initialize_pathology_project(encoder: nn.Module, artifact_path: str | Path,
                                expected_sha256: str | None = None) -> dict[str, Any]:
    """Load only the full-six-donor source projection, leaving all heads intact."""
    if expected_sha256 is not None:
        require(sha256(artifact_path) == expected_sha256, "external transfer artifact SHA mismatch")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    require(artifact.get("schema") == "bgb_en_external_projection_transfer_v2_unitnorm"
            and artifact.get("role") == "full_source_transfer" and artifact.get("epochs") == 20
            and artifact.get("source_donors") == 6,
            "external transfer artifact schema/fit role differs")
    require(artifact.get("input_normalization") == INPUT_NORMALIZATION,
            "external projection requires the matching target float32 per-row L2 input normalization")
    require(artifact.get("TCGA_DSS_read") is False and artifact.get("original_ST_test_read") is False,
            "external transfer artifact does not exclude target outcomes")
    expected = encoder.project.state_dict()
    state = artifact.get("project_state_dict", {})
    require(set(state) == set(expected) and all(state[k].shape == expected[k].shape for k in expected),
            "external projection does not match target visual feature space/dimensions")
    require(all(torch.isfinite(v).all() for v in state.values()), "external projection contains nonfinite weights")
    encoder.project.load_state_dict(state, strict=True)
    return dict(status="PASS_PROJECT_ONLY_INITIALIZATION", artifact_sha256=sha256(artifact_path),
                source_data_sha256=artifact["source_data_sha256"], transferred_parameters=sum(v.numel() for v in state.values()),
                source_donors=6, source_epochs=20, target_DSS_used_for_pretraining=False,
                input_normalization=INPUT_NORMALIZATION,
                target_heads_replaced=False, source_truth="tissue labels plus four local expression program proxies")

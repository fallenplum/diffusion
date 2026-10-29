"""Leakage-safe source data and separate image / paired gene input types.

The old qualified R6 cache is an immutable input, not a new independent cohort.
No TCGA, survival endpoint, clinical label or patient risk is imported here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
import torch


PROGRAMS = ("iCMS2", "iCMS3", "FAP_fibroblast_associated_top25", "MARCO_macrophage_associated_top25")
IMAGE_UNIT = "float32_per_row_L2_norm_division_reject_norm_le_1e-12"
GENE_UNIT = "log1p_CP10K_from_full_source_count_library"


def require(value: Any, code: str) -> None:
    if not bool(value):
        raise ValueError(code)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


@dataclass(frozen=True)
class ImageBatch:
    """Only image evidence and relative geometry can enter image-only inference.

    Absolute positions, donor IDs, tissue truth, counts and depth are absent.
    The data adapter must establish same-slide neighbors before constructing it.
    """
    features: torch.Tensor
    neighbor_features: torch.Tensor
    neighbor_xy_um: torch.Tensor
    neighbor_mask: torch.Tensor

    def validate(self, *, input_dim: int = 1536, radius_um: float = 300.) -> "ImageBatch":
        x, n, xy, mask = self.features, self.neighbor_features, self.neighbor_xy_um, self.neighbor_mask
        require(x.ndim == 2 and len(x) > 0 and x.shape[1] == input_dim and x.dtype == torch.float32,
                "IMAGE_REQUIRES_FINITE_FLOAT32_UNIT_ROWS")
        require(n.ndim == 3 and n.shape[0] == len(x) and n.shape[2] == input_dim and 1 <= n.shape[1] <= 16,
                "IMAGE_NEIGHBOR_FEATURE_SHAPE")
        require(xy.shape == (*n.shape[:2], 2) and mask.shape == n.shape[:2] and mask.dtype == torch.bool,
                "IMAGE_RELATIVE_GEOMETRY_SHAPE")
        require(n.dtype == xy.dtype == torch.float32 and x.device == n.device == xy.device == mask.device,
                "IMAGE_GEOMETRY_DTYPE_OR_DEVICE")
        require(torch.isfinite(x).all() and torch.isfinite(n[mask]).all() and torch.isfinite(xy[mask]).all(),
                "IMAGE_OBSERVED_FEATURE_NONFINITE")
        require((x.norm(dim=-1) - 1).abs().max() <= 3e-6, "IMAGE_ROWS_NOT_UNIT_L2")
        if mask.any():
            require((n[mask].norm(dim=-1) - 1).abs().max() <= 3e-6, "IMAGE_NEIGHBORS_NOT_UNIT_L2")
            require(xy[mask].square().sum(-1).max() <= radius_um**2 + 1e-2,
                    "IMAGE_EDGE_EXCEEDS_REGISTERED_RADIUS")
        return self


@dataclass(frozen=True)
class GeneBatch:
    """A single original-gene bank; masking acts before all membership copies."""
    values: torch.Tensor
    observed: torch.Tensor
    gene_ids: torch.Tensor

    def validate(self, genes: int) -> "GeneBatch":
        require(self.values.ndim == 2 and self.values.shape[1] == genes and len(self.values) > 0,
                "GENES_REQUIRE_ORIGINAL_B_BY_UNION_VALUES")
        require(self.values.dtype == torch.float32 and self.observed.shape == self.values.shape
                and self.observed.dtype == torch.bool and self.observed.device == self.values.device,
                "GENE_VALUES_OR_MASK_DTYPE_DEVICE")
        require(self.gene_ids.shape == (genes,) and self.gene_ids.dtype == torch.long
                and self.gene_ids.device == self.values.device
                and torch.equal(self.gene_ids.sort().values, torch.arange(genes, device=self.values.device)),
                "GENE_ID_VALUES_MASK_MUST_BE_JOINT_PERMUTATION")
        require(torch.isfinite(self.values[self.observed]).all(), "OBSERVED_GENE_NONFINITE")
        return self

    def masked(self, hide: torch.Tensor) -> "GeneBatch":
        require(hide.shape == self.observed.shape and hide.dtype == torch.bool
                and hide.device == self.observed.device and not (hide & ~self.observed).any(),
                "ONLY_MEASURED_GENES_CAN_BE_MASK_TARGETS")
        observed = self.observed & ~hide
        return GeneBatch(torch.where(observed, self.values, torch.zeros_like(self.values)), observed, self.gene_ids)


def program_scores(values: np.ndarray, observed: np.ndarray, up: np.ndarray, down: np.ndarray,
                   *, minimum_genes: int = 5, minimum_coverage: float = .95):
    x, mask = np.asarray(values, float), np.asarray(observed, bool)
    up, down = np.asarray(up, bool), np.asarray(down, bool)
    require(x.ndim == 2 and mask.shape == x.shape and up.shape == down.shape == (4, x.shape[1]),
            "FOUR_PROGRAM_SCORE_SHAPES")
    require(np.isfinite(x[mask]).all() and not (up & down).any() and up.any(1).all(),
            "SIGNED_PROGRAM_VALUES_OR_MEMBERSHIP")
    scores = np.zeros((len(x), 4), np.float64)
    valid = np.zeros_like(scores, bool)
    for k in range(4):
        pu, pd = mask & up[k], mask & down[k]
        cu, cd = pu.sum(1), pd.sum(1)
        coverage = (cu + cd) / int(up[k].sum() + down[k].sum())
        ok = (cu >= minimum_genes) & (coverage >= minimum_coverage)
        if down[k].any():
            ok &= cd >= minimum_genes
        u = np.where(pu, x, 0.).sum(1) / np.maximum(cu, 1)
        d = np.where(pd, x, 0.).sum(1) / np.maximum(cd, 1)
        scores[:, k], valid[:, k] = np.where(ok, u - d, 0.), ok
    return scores, valid


@dataclass(repr=False)
class SpatialSource:
    features: np.ndarray
    genes: np.ndarray
    gene_mask: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    tissue: np.ndarray
    xy_um: np.ndarray
    section: np.ndarray
    donor: np.ndarray
    library: np.ndarray
    neighbors: np.ndarray
    up: np.ndarray
    down: np.ndarray
    gene_symbols: np.ndarray
    audit: dict[str, Any]

    def __repr__(self):
        return f"SpatialSource(rows={len(self.features)}, donors={len(np.unique(self.donor))}, programs=4)"

    def validate(self) -> "SpatialSource":
        x, n, u = self.features, len(self.features), self.genes.shape[1]
        require(x.ndim == 2 and x.shape[1] == 1536 and x.dtype == np.float32 and np.isfinite(x).all(),
                "SOURCE_UNIT_UNI2_FEATURE_SHAPE")
        require(np.max(np.abs(np.linalg.norm(x, axis=1) - 1)) <= 3e-6, "SOURCE_UNIT_UNI2_NORMALIZATION")
        require(self.genes.shape == self.gene_mask.shape == (n, u) and self.gene_mask.dtype == bool
                and np.isfinite(self.genes[self.gene_mask]).all(), "SOURCE_REAL_GENE_MEASUREMENTS")
        require(self.targets.shape == self.target_mask.shape == (n, 4) and self.target_mask.dtype == bool
                and np.isfinite(self.targets[self.target_mask]).all(), "SOURCE_PROGRAM_TARGETS")
        require(self.tissue.shape == self.section.shape == self.donor.shape == self.library.shape == (n,)
                and np.isin(self.tissue, (0, 1, 2)).all() and np.isfinite(self.library).all()
                and (self.library > 0).all(), "SOURCE_METADATA_SHAPES")
        require(self.xy_um.shape == (n, 2) and np.isfinite(self.xy_um).all()
                and self.up.shape == self.down.shape == (4, u), "SOURCE_GEOMETRY_MEMBERSHIP_SHAPES")
        require(len(self.gene_symbols) == u and len(set(self.gene_symbols.tolist())) == u,
                "SOURCE_UNIQUE_GENE_VOCABULARY")
        require(self.neighbors.ndim == 2 and self.neighbors.shape[0] == n and 1 <= self.neighbors.shape[1] <= 16
                and self.neighbors.dtype.kind in "iu" and ((self.neighbors >= -1) & (self.neighbors < n)).all(),
                "SOURCE_GRAPH_RANGE")
        observed = self.neighbors >= 0
        safe = np.maximum(self.neighbors, 0)
        for codes in (self.section, self.donor):
            require(np.all(codes[safe][observed] == np.broadcast_to(codes[:, None], safe.shape)[observed]),
                    "SOURCE_GRAPH_CROSSES_SLIDE_OR_DONOR")
        delta = self.xy_um[safe] - self.xy_um[:, None, :]
        require(np.max(np.where(observed, np.sum(delta**2, axis=-1), 0.)) <= 300.**2 + 1e-5,
                "SOURCE_GRAPH_EXCEEDS_300UM")
        for section in np.unique(self.section):
            rows = self.section == section
            require(len(np.unique(self.donor[rows])) == 1 and len(np.unique(self.xy_um[rows], axis=0)) == rows.sum(),
                    "SOURCE_SLIDE_IDENTITY_OR_COORDINATE_DUPLICATION")
        y, valid = program_scores(self.genes, self.gene_mask, self.up, self.down)
        require(np.array_equal(valid, self.target_mask)
                and np.max(np.abs(y[valid] - self.targets[valid]), initial=0.) <= 2e-12,
                "SOURCE_PROGRAM_VALUES_NOT_REPLAYED_FROM_MEASURED_GENES")
        return self

    def split(self, heldout_donor: int):
        require(heldout_donor in np.unique(self.donor), "HELDOUT_DONOR_NOT_IN_SOURCE")
        train = np.flatnonzero(self.donor != heldout_donor)
        test = np.flatnonzero(self.donor == heldout_donor)
        require(len(train) > 1 and len(test) > 1 and not np.intersect1d(self.donor[train], self.donor[test]).size,
                "DONOR_HELDOUT_SPLIT_FAILED")
        return train, test

    def image_batch(self, rows, *, allowed_rows, device="cpu") -> ImageBatch:
        rows, scope = np.asarray(rows, np.int64), np.asarray(allowed_rows, np.int64)
        require(rows.ndim == scope.ndim == 1 and len(rows) and len(np.unique(scope)) == len(scope)
                and np.isin(rows, scope).all(), "IMAGE_ROWS_OUTSIDE_REGISTERED_PARTITION")
        neighbors = self.neighbors[rows]
        valid = neighbors >= 0
        require(np.isin(neighbors[valid], scope).all(), "NEIGHBOR_CROSSES_CURRENT_FIT_OR_EVALUATION_SCOPE")
        safe = np.maximum(neighbors, 0)
        features = np.where(valid[..., None], self.features[safe], 0.).astype(np.float32)
        xy = np.where(valid[..., None], self.xy_um[safe] - self.xy_um[rows, None], 0.).astype(np.float32)
        return ImageBatch(torch.as_tensor(self.features[rows], device=device),
            torch.as_tensor(features, device=device), torch.as_tensor(xy, device=device),
            torch.as_tensor(valid, device=device)).validate()


def load_registered_r6_cache(directory: str | Path, *, expected_manifest_sha256: str) -> SpatialSource:
    """Read only the two named immutable source payloads; no hidden dataset discovery."""
    root = Path(directory)
    manifest_path = root / "manifest.private.json"
    require(sha256(manifest_path) == expected_manifest_sha256, "SOURCE_CACHE_MANIFEST_SHA_CHANGED")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "bgb_en_r6_private_source_cache_v1"
            and manifest.get("status") == "COMPLETE_PRIVATE_DERIVED_CACHE"
            and set(manifest.get("files", {})) == {"gene_counts.private.npz", "source_arrays.private.npz"},
            "SOURCE_CACHE_SCHEMA_OR_FILE_LIST")
    for name, digest in manifest["files"].items():
        require((root / name).is_file() and not (root / name).is_symlink() and sha256(root / name) == digest,
                "SOURCE_CACHE_BYTES_CHANGED")
    with np.load(root / "source_arrays.private.npz", allow_pickle=False) as archive:
        a = {key: np.array(archive[key], copy=True) for key in archive.files}
    counts = sparse.load_npz(root / "gene_counts.private.npz").astype(np.float64)
    require(np.isfinite(counts.data).all() and (counts.data >= 0).all()
            and np.equal(counts.data, np.floor(counts.data)).all(), "SOURCE_RAW_COUNTS_NONINTEGER")
    library = np.asarray(a["library_size"], np.float64)
    require(counts.shape == a["log1p_cp10k"].shape and (np.asarray(counts.sum(1)).ravel() <= library).all(),
            "GENE_UNION_SUM_IS_NOT_FULL_LIBRARY")
    replay = np.log1p(counts.toarray() * (10000. / library[:, None]))
    require(np.max(np.abs(replay - a["log1p_cp10k"])) <= 2e-12, "SOURCE_FULL_LIBRARY_LOGCP10K_REPLAY")
    y, valid = program_scores(a["log1p_cp10k"], a["measured_mask"], a["program_up"], a["program_down"])
    audit = json.loads(json.dumps(manifest["source_audit"]))
    audit.update(spgel_cache_manifest_sha256=expected_manifest_sha256, source_cache_is_historically_used=True,
                 source_task_only=True, survival_files_loaded=False,
                 program_score_replay_max_abs_error=float(np.max(np.abs(y[valid] - a["program_targets"][valid]))))
    return SpatialSource(a["features_unit"], a["log1p_cp10k"], a["measured_mask"], a["program_targets"], valid,
        a["tissue_y"], a["coords_um"], a["section_codes"], a["donor_codes"], library, a["neighbors"],
        a["program_up"], a["program_down"], a["gene_symbols"], audit).validate()


@dataclass(frozen=True)
class SourceStatistics:
    target_mean: np.ndarray
    target_sd: np.ndarray
    gene_mean: np.ndarray
    gene_sd: np.ndarray
    class_weights: np.ndarray
    tissue_program_mean: np.ndarray
    high_program_threshold: np.ndarray
    log_depth_quartiles: np.ndarray
    fit_rows_sha256: str
    fit_donor_codes: tuple[int, ...]

    @classmethod
    def fit(cls, source: SpatialSource, rows, *, heldout_donors=()):
        rows = np.asarray(rows, np.int64)
        require(rows.ndim == 1 and len(rows) > 1 and len(np.unique(rows)) == len(rows)
                and ((rows >= 0) & (rows < len(source.features))).all(), "SOURCE_SCALER_FIT_ROWS")
        require(not np.intersect1d(source.donor[rows], np.asarray(heldout_donors)).size,
                "HELDOUT_DONOR_ENTERED_SOURCE_STATISTICS")
        def moments(values, mask):
            count = mask.sum(0)
            mean = np.where(mask, values, 0.).sum(0) / np.maximum(count, 1)
            sd = np.sqrt(np.where(mask, (values - mean)**2, 0.).sum(0) / np.maximum(count, 1))
            return mean, np.where(sd > 1e-12, sd, 1.), count
        ym, ys, yc = moments(source.targets[rows], source.target_mask[rows])
        gm, gs, _ = moments(source.genes[rows], source.gene_mask[rows])
        require((yc >= 2).all(), "SOURCE_PROGRAM_TRAIN_SUPPORT")
        mass = np.bincount(source.tissue[rows], minlength=3)
        require((mass > 0).all(), "ALL_THREE_TISSUES_REQUIRED_IN_TRAIN")
        tissue_mean = np.empty((3, 4), np.float64)
        for label in range(3):
            selected = rows[source.tissue[rows] == label]
            mean, _, count = moments(source.targets[selected], source.target_mask[selected])
            require((count > 0).all(), "TISSUE_PROGRAM_TRAIN_SUPPORT")
            tissue_mean[label] = (mean - ym) / ys
        threshold = np.array([np.quantile(source.targets[rows, k][source.target_mask[rows, k]], .9) for k in range(4)])
        return cls(ym, ys, gm, gs, len(rows) / (3. * mass), tissue_mean, threshold,
            np.quantile(np.log1p(source.library[rows]), [.25, .5, .75]), array_hash(rows),
            tuple(map(int, np.unique(source.donor[rows]))))

    def targets_z(self, source: SpatialSource, rows):
        return np.where(source.target_mask[rows], (source.targets[rows] - self.target_mean) / self.target_sd, 0.)

    def genes_z(self, source: SpatialSource, rows, device="cpu") -> GeneBatch:
        values = np.where(source.gene_mask[rows], (source.genes[rows] - self.gene_mean) / self.gene_sd, 0.).astype(np.float32)
        return GeneBatch(torch.as_tensor(values, device=device),
            torch.as_tensor(source.gene_mask[rows], device=device),
            torch.arange(values.shape[1], device=device)).validate(values.shape[1])

    def inverse_targets(self, prediction):
        return np.asarray(prediction) * self.target_sd + self.target_mean

    def audit(self):
        return dict(scope="current_source_training_donors_only", fit_rows_sha256=self.fit_rows_sha256,
                    training_donor_count=len(self.fit_donor_codes), high_activity_quantile=.9,
                    official_HVG_processing_is_a_separate_contract=True,
                    feature_normalization=IMAGE_UNIT, source_program_unit=GENE_UNIT,
                    heldout_statistics_used=False, TCGA_or_DSS_used=False)

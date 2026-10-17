"""Train-only discrepancy scales for the fixed molecular dictionary.

These are in-sample dictionary reconstruction error proxies, not estimates of
technical measurement noise. A fixed train-variation floor bounds inverse gene
weights. No validation/test profile, model selection, or endpoint is accepted.
"""
from dataclasses import dataclass
import hashlib
import json
import re

import numpy as np

from .molecular_dictionary import counts_sha256, sequence_sha256


SCALE_SCHEMA = "phase22_train_dictionary_discrepancy_scale_v1"
SCALE_SOURCE = "train_in_sample_dictionary_residual_proxy_not_measurement_noise"
SCALE_PROTOCOL = {
    "schema_version": SCALE_SCHEMA,
    "partition_role": "ST-train",
    "residual_RMS": "sqrt(mean_donor(mean_profile((count/library_exposure-oracle_pi@D)^2)))",
    "within_variance": "mean_donor(population_variance_profile(count/library_exposure))",
    "pooled_scale": "sqrt(mean_gene(within_variance))",
    "sigma": "max(residual_RMS,0.1*sqrt(within_variance),0.1*pooled_scale)",
    "floor_ratio": 0.1,
    "omega": "1/(sqrt(G)*sigma)",
    "source": SCALE_SOURCE,
    "zero_pooled_scale": "FAIL_NO_TRAINING_VARIATION",
    "oracle_nonconvergence": "FAIL_NO_CALIBRATION",
    "hyperparameter_search": False,
}
SCALE_PROTOCOL_SHA = hashlib.sha256(json.dumps(SCALE_PROTOCOL, sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MeasurementScaleTrainingContract:
    partition_role: str
    split_lock_sha256: str
    counts_sha256: str
    donor_sequence_sha256: str
    dictionary_sha256: str
    donor_roles: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MeasurementScaleFit:
    sigma: np.ndarray
    omega: np.ndarray
    residual_rms: np.ndarray
    within_variance: np.ndarray
    metadata: dict


def fit_train_measurement_scale(counts, oracle_pi, dictionary, donor_ids, *,
                                oracle_converged: bool,
                                contract: MeasurementScaleTrainingContract) -> MeasurementScaleFit:
    if not isinstance(contract, MeasurementScaleTrainingContract) or contract.partition_role != "ST-train":
        raise ValueError("measurement scale requires an ST-train contract")
    if oracle_converged is not True:
        raise ValueError("nonconverged dictionary oracle cannot calibrate measurement scales")
    raw, pi, d = np.asarray(counts), np.asarray(oracle_pi, dtype=np.float64), np.asarray(dictionary, dtype=np.float64)
    donors = list(donor_ids)
    if raw.ndim != 2 or min(raw.shape) < 1 or len(donors) != len(raw):
        raise ValueError("training counts and donor rows must be nonempty and aligned")
    if np.any(~np.isfinite(raw)) or np.any(raw < 0) or np.any(raw != np.floor(raw)):
        raise ValueError("calibration requires nonnegative integer raw counts")
    if d.ndim != 2 or d.shape[1] != raw.shape[1] or pi.shape != (len(raw), len(d)):
        raise ValueError("oracle coefficients and dictionary dimensions disagree")
    for name, x in (("dictionary", d), ("oracle_pi", pi)):
        if np.any(~np.isfinite(x)) or np.any(x < 0) or not np.allclose(x.sum(1), 1., atol=1e-10, rtol=0):
            raise ValueError(name + " must have finite simplex rows")
    if any(not isinstance(donor, str) or not donor for donor in donors):
        raise ValueError("known private training donor IDs are required")
    roles = dict(contract.donor_roles)
    if len(roles) != len(contract.donor_roles) or any(role not in {"train", "validation", "test"} for role in roles.values()):
        raise ValueError("unique, known donor roles are required")
    if any(roles.get(donor) != "train" for donor in donors):
        raise ValueError("calibration received a non-train or unknown donor")
    if set(donors) != {donor for donor, role in roles.items() if role == "train"}:
        raise ValueError("calibration must contain exactly the locked training donors")
    expected_hashes = dict(counts_sha256=counts_sha256(raw),
                           donor_sequence_sha256=sequence_sha256(donors),
                           dictionary_sha256=counts_sha256(d))
    for name, actual in expected_hashes.items():
        if getattr(contract, name) != actual:
            raise ValueError("measurement scale input hash mismatch: " + name)
    if not re.fullmatch("[0-9a-f]{64}", contract.split_lock_sha256):
        raise ValueError("measurement scale requires the exact external split-lock SHA")
    exposure = raw.sum(1, dtype=np.float64)
    if np.any(exposure <= 0) or np.any(~np.isfinite(exposure)):
        raise ValueError("calibration cannot use zero/invalid-exposure profiles")
    y = raw / exposure[:, None]
    reconstruction = pi @ d
    donor_array = np.asarray(donors)
    errors, variances = [], []
    for donor in sorted(set(donors)):
        index = donor_array == donor
        errors.append(np.square(y[index] - reconstruction[index]).mean(0))
        # Translation leaves variance unchanged and makes identical profiles
        # exactly zero rather than inventing a ~1e-32 rounding variance floor.
        centered_at_first = y[index] - y[index][0]
        variances.append(centered_at_first.var(0, ddof=0))
    residual_rms = np.sqrt(np.mean(errors, axis=0))
    within_variance = np.mean(variances, axis=0)
    pooled_scale = float(np.sqrt(within_variance.mean()))
    if not np.isfinite(pooled_scale) or pooled_scale <= 0:
        raise ValueError("FAIL_NO_TRAINING_VARIATION: pooled within-donor scale must be positive")
    gene_floor = .1 * np.sqrt(within_variance)
    pooled_floor = .1 * pooled_scale
    sigma = np.maximum(np.maximum(residual_rms, gene_floor), pooled_floor)
    omega = 1. / (np.sqrt(raw.shape[1]) * sigma)
    metadata = {
        "schema_version": SCALE_SCHEMA, "status": "PASS_TRAIN_ONLY_DISCREPANCY_SCALE",
        "protocol": SCALE_PROTOCOL, "protocol_sha256": SCALE_PROTOCOL_SHA,
        "source": SCALE_SOURCE, "partition_role": "ST-train",
        "split_lock_sha256": contract.split_lock_sha256, **expected_hashes,
        "training_donors": len(set(donors)), "training_profiles": len(raw), "genes": raw.shape[1],
        "oracle_converged": True, "pooled_within_donor_scale": pooled_scale,
        "oracle_coefficients_sha256": counts_sha256(pi),
        "floor_ratio": .1, "pooled_absolute_floor": pooled_floor,
        "residual_below_gene_variation_floor_genes": int(np.sum(residual_rms < gene_floor)),
        "residual_below_pooled_floor_genes": int(np.sum(residual_rms < pooled_floor)),
        "any_floor_active_genes": int(np.sum(sigma > residual_rms)),
        "sigma_min": float(sigma.min()), "sigma_max": float(sigma.max()),
        "omega_min": float(omega.min()), "omega_max": float(omega.max()),
        "inverse_weight_upper_bound": float(1. / (np.sqrt(raw.shape[1]) * pooled_floor)),
        "validation_or_test_profiles_read": False,
    }
    for name, x in (("sigma", sigma), ("omega", omega), ("residual_rms", residual_rms), ("within_variance", within_variance)):
        metadata[name + "_sha256"] = counts_sha256(x)
    return MeasurementScaleFit(sigma, omega, residual_rms, within_variance, metadata)


def verify_measurement_scale(dictionary, sigma, omega, residual_rms, within_variance,
                             receipt, *, split_lock_sha256):
    """Fail closed when loaded private scale arrays or train lineage drift."""
    if (receipt.get("schema_version") != SCALE_SCHEMA or
        receipt.get("status") != "PASS_TRAIN_ONLY_DISCREPANCY_SCALE" or
        receipt.get("protocol") != SCALE_PROTOCOL or receipt.get("protocol_sha256") != SCALE_PROTOCOL_SHA or
        receipt.get("partition_role") != "ST-train" or receipt.get("source") != SCALE_SOURCE or
        receipt.get("split_lock_sha256") != split_lock_sha256 or receipt.get("oracle_converged") is not True or
        receipt.get("validation_or_test_profiles_read") is not False):
        raise ValueError("unqualified train-only measurement scale receipt")
    d = np.asarray(dictionary, dtype=np.float64)
    if receipt.get("dictionary_sha256") != counts_sha256(d):
        raise ValueError("measurement scale dictionary lineage mismatch")
    arrays = {name: np.asarray(x, dtype=np.float64) for name, x in
              (("sigma", sigma), ("omega", omega), ("residual_rms", residual_rms), ("within_variance", within_variance))}
    for name, x in arrays.items():
        if x.shape != (d.shape[1],) or not np.isfinite(x).all() or np.any(x < 0):
            raise ValueError("invalid measurement scale array: " + name)
        if receipt.get(name + "_sha256") != counts_sha256(x):
            raise ValueError("measurement scale array hash mismatch: " + name)
    pooled = float(np.sqrt(arrays["within_variance"].mean()))
    if not np.isfinite(pooled) or pooled <= 0:
        raise ValueError("FAIL_NO_TRAINING_VARIATION")
    expected_sigma = np.maximum(np.maximum(arrays["residual_rms"], .1 * np.sqrt(arrays["within_variance"])), .1 * pooled)
    if not np.array_equal(arrays["sigma"], expected_sigma) or not np.array_equal(arrays["omega"], 1. / (np.sqrt(d.shape[1]) * expected_sigma)):
        raise ValueError("loaded measurement scale violates the locked formula")
    if receipt.get("genes") != d.shape[1] or receipt.get("pooled_within_donor_scale") != pooled:
        raise ValueError("loaded measurement scale dimensions/statistics disagree")

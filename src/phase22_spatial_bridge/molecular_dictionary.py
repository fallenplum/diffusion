"""Train-only, donor-balanced KL-NMF molecular dictionary qualification.

The contract binds the exact count/gene/donor/measurement inputs and an external
partition lock. Validation/test donors are rejected rather than silently
included or internally repartitioned. No survival outcome is an input.
"""
from dataclasses import dataclass
import hashlib
import json
import re
import warnings

import numpy as np
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning


def sequence_sha256(values) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def counts_sha256(values) -> str:
    """Hash exact dense values or canonical sparse CSR arrays and dtype/shape."""
    digest = hashlib.sha256()
    if sparse.issparse(values):
        x = values.tocsr(copy=True)
        x.sum_duplicates()
        x.sort_indices()
        digest.update(b"csr-v1")
        for part in (np.asarray(x.shape, dtype=np.int64), x.indptr, x.indices, x.data):
            array = np.ascontiguousarray(part)
            digest.update(array.dtype.str.encode())
            digest.update(array.tobytes())
    else:
        array = np.ascontiguousarray(values)
        digest.update(b"dense-v1")
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DictionaryTrainingContract:
    partition_role: str
    split_lock_sha256: str
    counts_sha256: str
    gene_names_sha256: str
    donor_sequence_sha256: str
    st_measured_genes_sha256: str
    target_measured_genes_sha256: str
    # Source IDs and group assignments are private runtime data, not reports.
    donor_roles: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MolecularDictionaryFit:
    dictionary: np.ndarray
    gene_names_private: tuple[str, ...]
    selected_gene_indices_private: np.ndarray
    sampled_spot_indices_private: np.ndarray
    metadata: dict


def _validate_contract(raw_counts, genes, donors, st_measured_genes,
                       target_measured_genes, contract: DictionaryTrainingContract):
    if not isinstance(contract, DictionaryTrainingContract) or contract.partition_role != "ST-train":
        raise ValueError("an ST-train DictionaryTrainingContract is required")
    expected = {
        "counts_sha256": counts_sha256(raw_counts),
        "gene_names_sha256": sequence_sha256(genes),
        "donor_sequence_sha256": sequence_sha256(donors),
        "st_measured_genes_sha256": sequence_sha256(sorted(st_measured_genes)),
        "target_measured_genes_sha256": sequence_sha256(sorted(target_measured_genes)),
    }
    for key, actual in expected.items():
        locked = getattr(contract, key)
        if not re.fullmatch("[0-9a-f]{64}", locked) or actual != locked:
            raise ValueError(f"dictionary input hash mismatch: {key}")
    if not re.fullmatch("[0-9a-f]{64}", contract.split_lock_sha256):
        raise ValueError("external split-lock SHA256 is required")
    roles = dict(contract.donor_roles)
    if len(roles) != len(contract.donor_roles) or any(r not in {"train", "validation", "test"} for r in roles.values()):
        raise ValueError("donor split roles must be unique and explicitly known")
    if not donors or any(not isinstance(d, str) or not d for d in donors):
        raise ValueError("private donor IDs must be nonempty strings")
    if any(roles.get(d) != "train" for d in donors):
        raise ValueError("dictionary received non-train or unknown donor")
    if set(donors) != {d for d, role in roles.items() if role == "train"}:
        raise ValueError("dictionary input must contain exactly the locked training donors")


def fit_molecular_dictionary(raw_counts, gene_names, donor_ids, *,
                             st_measured_genes, target_measured_genes,
                             contract: DictionaryTrainingContract,
                             seed: int = 31, n_states: int = 32, gene_cap: int = 1024,
                             profiles_per_donor_cap: int = 2048,
                             max_iter: int = 500, tolerance: float = 1e-5) -> MolecularDictionaryFit:
    """Fit row-simplex D on donor-balanced positive composition profiles.

    Common measured genes with positive train counts are ranked by donor-equal
    composition variance if there are more than ``gene_cap``. Each donor then
    contributes the same number of nonzero profiles, sampled without
    replacement by the fixed initialization seed. Measured zero is a count;
    unmeasured genes cannot enter the intersection. Empty profiles are
    excluded and counted explicitly, not transformed into uniform biology.
    This deterministic procedure is not selected against TCGA outcomes.
    """
    genes, donors = list(gene_names), list(donor_ids)
    st_measured_genes, target_measured_genes = set(st_measured_genes), set(target_measured_genes)
    if len(set(genes)) != len(genes) or any(not isinstance(g, str) or not g for g in genes):
        raise ValueError("gene names must be unique, resolved and nonempty")
    if getattr(raw_counts, "ndim", None) != 2 or raw_counts.shape != (len(donors), len(genes)):
        raise ValueError("raw_counts must have [training spot, named gene] shape")
    if min(raw_counts.shape) < 1 or not 1 <= gene_cap <= 1024 or n_states < 1:
        raise ValueError("nonempty data, positive states and 1 <= gene_cap <= 1024 required")
    if profiles_per_donor_cap < 1 or max_iter < 1 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("profile/iteration budgets and tolerance must be positive")
    values = raw_counts.data if sparse.issparse(raw_counts) else np.asarray(raw_counts)
    if not np.isfinite(values).all() or np.any(values < 0) or not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
        raise ValueError("dictionary requires finite nonnegative integer raw counts")
    _validate_contract(raw_counts, genes, donors, st_measured_genes, target_measured_genes, contract)
    x = raw_counts.tocsr().astype(np.float64) if sparse.issparse(raw_counts) else np.asarray(raw_counts, dtype=np.float64)
    common = [i for i, name in enumerate(genes) if name in st_measured_genes and name in target_measured_genes]
    if not common:
        raise ValueError("no common measured ST-train/target genes")
    counts = x[:, common]
    positive_genes = np.asarray(counts.sum(axis=0)).ravel() > 0
    selected = np.asarray(common, dtype=np.int64)[positive_genes]
    if not len(selected):
        raise ValueError("all common measured training genes have zero counts")
    counts = x[:, selected]
    lib = np.asarray(counts.sum(axis=1)).ravel()
    positive_spots = lib > 0
    donor_array = np.asarray(donors)
    grouped = {d: np.flatnonzero((donor_array == d) & positive_spots) for d in sorted(set(donors))}
    if any(len(idx) == 0 for idx in grouped.values()):
        raise ValueError("a training donor has no positive common-gene profiles")
    # Equal donor moments prevent a large section from controlling gene choice.
    if len(selected) > gene_cap:
        mean, second = [], []
        for idx in grouped.values():
            profile = counts[idx].toarray() if sparse.issparse(counts) else counts[idx].copy()
            profile /= lib[idx, None]
            mean.append(profile.mean(0))
            second.append(np.square(profile).mean(0))
        variance = np.mean(second, axis=0) - np.square(np.mean(mean, axis=0))
        # Stable original gene order breaks exact variance ties.
        keep = np.argsort(-variance, kind="stable")[:gene_cap]
        selected = np.sort(selected[keep])
        counts = x[:, selected]
        lib = np.asarray(counts.sum(axis=1)).ravel()
        positive_spots = lib > 0
        grouped = {d: np.flatnonzero((donor_array == d) & positive_spots) for d in sorted(set(donors))}
        if any(len(idx) == 0 for idx in grouped.values()):
            raise ValueError("gene selection removed all positive profiles from a donor")
    per_donor = min(profiles_per_donor_cap, *(len(idx) for idx in grouped.values()))
    rng = np.random.default_rng(seed)
    sampled = np.concatenate([np.sort(rng.choice(idx, per_donor, replace=False)) for idx in grouped.values()])
    profile = counts[sampled].toarray() if sparse.issparse(counts) else counts[sampled].copy()
    profile /= lib[sampled, None]
    estimator = NMF(n_components=n_states, init="random", solver="mu",
                    beta_loss="kullback-leibler", random_state=seed,
                    max_iter=max_iter, tol=tolerance)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit_transform(profile)
    d = np.asarray(estimator.components_, dtype=np.float64)
    row_mass = d.sum(1)
    if not np.isfinite(d).all() or np.any(d < 0) or np.any(row_mass <= 0):
        raise ValueError("KL-NMF produced an invalid or empty dictionary state")
    d = d / row_mass[:, None]
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
    metadata = {
        "status": "FITTED" if converged else "FITTED_NOT_CONVERGED",
        "method": "sklearn_KL_NMF_mu_on_composition_profiles",
        "partition_role": "ST-train", "split_lock_sha256": contract.split_lock_sha256,
        "input_count_sha256": contract.counts_sha256,
        "gene_sequence_sha256": contract.gene_names_sha256,
        "target_measured_genes_sha256": contract.target_measured_genes_sha256,
        "selected_gene_sequence_sha256": sequence_sha256([genes[i] for i in selected]),
        "sampled_profile_indices_sha256": counts_sha256(sampled),
        "dictionary_sha256": counts_sha256(d),
        "seed": int(seed), "states": int(n_states), "genes": int(len(selected)),
        "training_donors": len(grouped), "profiles_per_donor": int(per_donor),
        "total_profiles": int(len(sampled)), "zero_common_gene_profiles_excluded": int((~positive_spots).sum()),
        "iterations": int(estimator.n_iter_), "max_iterations": int(max_iter),
        "tolerance": float(tolerance), "converged": bool(converged),
        "reconstruction_error": float(estimator.reconstruction_err_),
        "donor_equal_sampling": True, "survival_labels_read": False,
    }
    return MolecularDictionaryFit(d, tuple(genes[i] for i in selected), selected, sampled, metadata)

"""Outcome-blind, immutable patient reference bank for frozen CDEI kernels.

Only explicitly selected fit-patient rows are inspected or retained. Encoded
tokens must already share a frozen coordinate system. This module reads no
endpoints. Its state_dict contains protected data and must remain private.
Training omics references exclude self; the clinical reference is full-fit mean.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import NamedTuple

import torch
from torch import nn


class ReferenceMean(NamedTuple):
    tokens: torch.Tensor
    mask: torch.Tensor
    counts: torch.Tensor
    clinical_mean: torch.Tensor


def _tensor_hash(value: torch.Tensor) -> str:
    item = value.detach().cpu().contiguous()
    header = json.dumps({"dtype": str(item.dtype), "shape": list(item.shape)},
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\n" + item.view(torch.uint8).numpy().tobytes()).hexdigest()


def _indices(value, *, name: str, size: int, nonempty: bool,
             unique: bool) -> torch.Tensor:
    try:
        index = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be a one-dimensional integer index") from error
    if index.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer index")
    if index.numel() == 0:
        if nonempty:
            raise ValueError(f"{name} must be nonempty")
        return torch.empty(0, dtype=torch.int64)
    if index.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must be a one-dimensional integer index")
    index = index.detach().to(device="cpu", dtype=torch.int64).clone()
    if bool((index < 0).any()) or bool((index >= size).any()):
        raise ValueError(f"{name} contains an unknown or out-of-bounds patient index")
    if unique and torch.unique(index).numel() != index.numel():
        raise ValueError(f"{name} must contain unique patients")
    return index


class FrozenPatientReferenceBank(nn.Module):
    """Patient-weighted slot means from exactly the declared fit donors.

    encoded_tokens is floating [N,24,32], measured_mask is Boolean [N,24],
    clinical is floating [N,Dc], and fit_indices is nonempty/unique. An optional
    donor_indices contract must have exactly the same set as fit_indices.
    One row is one patient, never one slide. Only fit rows are checked for
    finiteness, so heldout rows may contain poison/NaN. Unmeasured fit values
    are zeroed with explicit masks. Private provenance uses a canonical JSON
    string and a copy-on-read property; public reports receive its hash only.

    All buffers are detached clones. Returned values and state snapshots do
    not alias them. Direct private-buffer tampering is detected before use.
    """

    def __init__(self, encoded_tokens, measured_mask, clinical, fit_indices,
                 *, donor_indices=None, provenance=None):
        super().__init__()
        tokens = torch.as_tensor(encoded_tokens)
        available = torch.as_tensor(measured_mask)
        clinical_values = torch.as_tensor(clinical)
        if (tokens.ndim != 3 or tuple(tokens.shape[1:]) != (24, 32)
                or not tokens.is_floating_point() or tokens.is_complex()
                or tokens.shape[0] < 1):
            raise ValueError("encoded_tokens must be floating [patients,24,32]")
        if available.shape != tokens.shape[:2] or available.dtype != torch.bool:
            raise ValueError("measured_mask must be Boolean [patients,24]")
        if (clinical_values.ndim != 2 or clinical_values.shape[0] != len(tokens)
                or clinical_values.shape[1] < 1 or not clinical_values.is_floating_point()
                or clinical_values.is_complex()):
            raise ValueError("clinical must be floating [patients,clinical_dim]")
        if available.device != tokens.device or clinical_values.device != tokens.device:
            raise ValueError("all reference inputs must share one device")
        self._cohort_size = int(len(tokens))
        fit = _indices(fit_indices, name="fit_indices", size=len(tokens), nonempty=True, unique=True)
        donors = fit if donor_indices is None else _indices(
            donor_indices, name="donor_indices", size=len(tokens), nonempty=True, unique=True)
        if not torch.equal(torch.sort(fit).values, torch.sort(donors).values):
            raise ValueError("reference donor set must equal the current fit patient set")
        if provenance is not None and not isinstance(provenance, Mapping):
            raise ValueError("provenance must be JSON-compatible metadata")
        try:
            context = json.dumps(dict(provenance or {}), sort_keys=True,
                                 separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("provenance must be JSON-compatible metadata") from error
        self._provenance_json = context
        self._provenance_sha256 = hashlib.sha256(context.encode("utf-8")).hexdigest()
        self._metadata_snapshot = (self._cohort_size, self._provenance_json,
                                   self._provenance_sha256)
        # Fit selection precedes finite checks; no heldout value is retained.
        index = fit.to(tokens.device)
        donor_mask = available.index_select(0, index).detach().clone()
        selected = tokens.index_select(0, index).detach().clone()
        selected_clinical = clinical_values.index_select(0, index).detach().clone()
        if not bool(torch.isfinite(selected[donor_mask]).all()):
            raise ValueError("measured fit tokens must be finite")
        if not bool(torch.isfinite(selected_clinical).all()):
            raise ValueError("fit clinical values must be finite")
        selected = torch.where(donor_mask[..., None], selected, torch.zeros_like(selected))
        self.register_buffer("_fit_indices", index.detach().clone())
        self.register_buffer("_encoded_tokens", selected)
        self.register_buffer("_measured_mask", donor_mask)
        self.register_buffer("_clinical", selected_clinical)
        self.register_buffer("_slot_sums", selected.double().sum(dim=0))
        self.register_buffer("_slot_counts", donor_mask.sum(dim=0, dtype=torch.int64))
        self.register_buffer("_clinical_fit_mean",
                             selected_clinical.double().mean(dim=0).to(clinical_values.dtype))
        self._integrity_hashes = {name: _tensor_hash(value) for name, value in self.named_buffers()}

    def _check_integrity(self):
        if (self._cohort_size, self._provenance_json,
                self._provenance_sha256) != self._metadata_snapshot:
            raise ValueError("immutable reference metadata changed")
        current = {name: _tensor_hash(value) for name, value in self.named_buffers()}
        if current != self._integrity_hashes:
            raise ValueError("immutable reference bank buffer changed")
        if any(value.requires_grad for value in self.buffers()):
            raise ValueError("immutable reference buffers cannot require gradients")

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self._check_integrity()
        for name, buffer in self._buffers.items():
            if buffer is not None and name not in self._non_persistent_buffers_set:
                destination[prefix + name] = buffer.detach().clone()
        destination[prefix + "_extra_state"] = self.get_extra_state()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                             missing_keys, unexpected_keys, error_msgs):
        self._check_integrity()
        changed = []
        for name, buffer in self._buffers.items():
            key = prefix + name
            if key in state_dict:
                item = state_dict[key]
                if (not isinstance(item, torch.Tensor) or item.dtype != buffer.dtype
                        or item.shape != buffer.shape
                        or not torch.equal(item.detach().to(buffer.device), buffer)):
                    changed.append(key)
        extra_key = prefix + "_extra_state"
        if extra_key in state_dict and state_dict[extra_key] != self.get_extra_state():
            changed.append(extra_key)
        if changed:
            error_msgs.append("immutable reference bank state mismatch: " + ",".join(changed))
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def get_extra_state(self):
        return {"schema": "phase23_frozen_patient_reference_v1",
                "cohort_size": self._cohort_size,
                "provenance_json": self._provenance_json,
                "provenance_sha256": self._provenance_sha256}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("immutable reference metadata changed")

    @property
    def provenance(self):
        """Fresh private metadata dictionary; callers cannot mutate the bank."""
        self._check_integrity()
        return json.loads(self._provenance_json)

    @torch.no_grad()
    def query_reference_mean(self, query_indices, additional_exclusions=None) -> ReferenceMean:
        """Exclude the union of self and optional per-query patient IDs.

        additional_exclusions is an integer [queries,M] array. Repeated IDs
        are excluded once; IDs outside fit but inside the cohort have no
        effect. Both matched/wrong-omics branches can exclude their common
        {own patient, wrong patient} union, avoiding donor-mask confounding.
        An empty additional list or [queries,0] array is permitted.
        """
        self._check_integrity()
        query = _indices(query_indices, name="query_indices", size=self._cohort_size,
                         nonempty=False, unique=False).to(self._fit_indices.device)
        exclude = query[:, None] == self._fit_indices[None, :]
        if additional_exclusions is not None:
            try:
                extra = torch.as_tensor(additional_exclusions)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError("additional_exclusions must be integer [queries,M]") from error
            if extra.ndim == 1 and extra.numel() == 0:
                extra = torch.empty((len(query), 0), dtype=torch.int64)
            if extra.ndim != 2 or extra.shape[0] != len(query):
                raise ValueError("additional_exclusions must have shape [queries,M]")
            extra_indices = _indices(extra.reshape(-1), name="additional_exclusions",
                                     size=self._cohort_size, nonempty=False, unique=False)
            extra = extra_indices.reshape(extra.shape).to(self._fit_indices.device)
            exclude = exclude | (extra[:, :, None] == self._fit_indices[None, None, :]).any(dim=1)
        # CUDA need not implement int64 GEMM. Binary donor counts are exact
        # in float64 at this finite cohort size, then restored to int64.
        excluded_counts = (exclude.double() @ self._measured_mask.double()).to(torch.int64)
        excluded_sum = (exclude.double() @ self._encoded_tokens.double().flatten(start_dim=1))
        excluded_sum = excluded_sum.reshape(len(query), 24, 32)
        counts = self._slot_counts[None, :] - excluded_counts
        sums = self._slot_sums[None, :, :] - excluded_sum
        valid = counts > 0
        means = sums / counts.clamp_min(1)[..., None]
        means = torch.where(valid[..., None], means, torch.zeros_like(means))
        return ReferenceMean(means.to(self._encoded_tokens.dtype), valid, counts,
                             self._clinical_fit_mean[None, :].expand(len(query), -1).clone())

    def public_summary(self):
        """Anonymous counts/hashes only; no donors, raw tokens or masks."""
        self._check_integrity()
        combined = hashlib.sha256(json.dumps(self._integrity_hashes, sort_keys=True,
                                  separators=(",", ":")).encode()).hexdigest()
        return {"schema": "phase23_frozen_patient_reference_public_v1",
                "cohort_size": self._cohort_size,
                "fit_patient_count": int(len(self._fit_indices)),
                "token_slots": 24, "embedding_dimension": 32,
                "clinical_dimension": int(self._clinical.shape[1]),
                "observed_slots": int((self._slot_counts > 0).sum()),
                "measured_patient_slot_count": int(self._measured_mask.sum()),
                "fit_indices_sha256": self._integrity_hashes["_fit_indices"],
                "encoded_tokens_sha256": self._integrity_hashes["_encoded_tokens"],
                "measured_mask_sha256": self._integrity_hashes["_measured_mask"],
                "clinical_fit_sha256": self._integrity_hashes["_clinical"],
                "provenance_sha256": self._provenance_sha256,
                "bank_state_sha256": combined}

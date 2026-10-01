"""M7A fold-local data adapter.

M7A uses only the audited six MCAT signature groups and the existing 512/256
fold-local numeric arrays.  Reusing the phase-2 preprocessing keeps the
endpoint, clinical columns, standardization, and signature intersections
identical to the established MCAT baseline.
"""
from src.phase_m2.data import FoldArrays, RawInputs, load_raw_inputs, prepare_path_cache, preprocess_fold

M7AFoldArrays = FoldArrays

__all__ = ["FoldArrays", "M7AFoldArrays", "RawInputs", "load_raw_inputs", "prepare_path_cache", "preprocess_fold"]

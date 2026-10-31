import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "omics_provenance", Path(__file__).parents[1] / "scripts" / "semantic_b0b7_p2_omics_provenance.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_missing_duplicate_is_recovered_only_by_observed_same_symbol():
    raw = np.asarray([[np.nan, 2., np.nan, 5.], [np.nan, np.nan, np.nan, 7.]], np.float32)
    symbols = np.asarray(["DUP", "DUP", "", "UNIQUE"])
    rebuilt, valid, locations = audit.collapse_symbols(raw, symbols, False)
    assert valid.tolist() == ["DUP", "UNIQUE"]
    np.testing.assert_array_equal(rebuilt, [[2., 5.], [np.nan, 7.]])
    stats = audit.collapse_missing_audit(raw, rebuilt, symbols, locations, valid)
    assert stats["raw_nonfinite_cells"] == 5
    assert stats["canonical_nonfinite_cells"] == 1
    assert stats["blank_symbol_nonfinite_cells_removed"] == 2
    assert stats["duplicate_symbol_missing_raw_cells_with_observed_same_symbol"] == 1
    assert stats["canonical_missing_in_duplicate_symbol_columns"] == 1


def test_rna_transform_precedes_symbol_mean_and_preserves_natural_missing():
    raw = np.asarray([[3., 15.], [np.nan, np.nan]], np.float32)
    rebuilt, _, _ = audit.collapse_symbols(raw, np.asarray(["A", "A"]), True)
    np.testing.assert_array_equal(rebuilt, [[3.], [np.nan]])
    assert rebuilt.dtype == np.float64


def test_rna_invalid_log_domain_is_rejected():
    with pytest.raises(ValueError, match="domain"):
        audit.collapse_symbols(np.asarray([[-1.]], np.float32), np.asarray(["A"]), True)


def test_inheritance_comparison_rejects_dtype_or_value_changes():
    a = np.asarray([[1., np.nan]], np.float32)
    assert audit.same_array(a, a.copy())
    assert not audit.same_array(a, a.astype(np.float64))
    assert not audit.same_array(a, np.asarray([[2., np.nan]], np.float32))

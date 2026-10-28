"""Pure synthetic source-depth diagnostics; no cache/model/TCGA access."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/bgb_en_r6_source_depth_audit.py"
spec = importlib.util.spec_from_file_location("r6_source_depth_audit_tests", SCRIPT)
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


def test_tied_midranks_and_spearman_are_hand_verified_without_pvalues():
    assert r.average_ranks([4, 1, 1, 3]).tolist() == [4., 1.5, 1.5, 3.]
    result = r.spearman([1, 1, 3, 4], [1, 2, 2, 4])
    assert result["support"] == 4 and result["status"] == "AVAILABLE"
    assert result["Spearman"] == pytest.approx(5 / 6)
    assert r.spearman([1, 2, 3], [3, 2, 1])["Spearman"] == pytest.approx(-1.)
    assert "pvalue" not in result


def test_insufficient_support_and_constant_inputs_keep_undefined_values():
    for x, y, code in (([1, np.nan, 3], [1, 2, 3], "INSUFFICIENT_SUPPORT"),
            ([2, 2, 2], [1, 2, 3], "CONSTANT_PROGRAM_VALUE"),
            ([1, 2, 3], [2, 2, 2], "CONSTANT_TECHNICAL_VALUE"),
            ([1, 1, 1], [2, 2, 2], "BOTH_CONSTANT")):
        result = r.spearman(x, y)
        assert result["Spearman"] is None and result["status"] == code


def test_true_source_group_correlations_are_anonymous_not_prediction_or_causal_claims():
    n = 72
    library = np.arange(1., n + 1)
    targets = np.column_stack((library, -library, np.ones(n), np.log1p(library)))
    data = SimpleNamespace(program_targets=targets, library_size=library, n_detected_genes=library,
        donor_codes=np.repeat(np.arange(6), 12), section_codes=np.repeat(np.arange(12), 6), tissue_y=np.arange(n) % 3,
        private_spot_ids="DO_NOT_EXPORT_SYNTHETIC_IDENTIFIER", features_unit="UNUSED_VISUAL_ARRAY")
    rows = r.target_correlations(data)
    assert len(rows) == 304
    full = [row for row in rows if row["scope"] == "ALL" and row["tissue_class"] == -1]
    assert full[0]["log_library_Spearman"] == pytest.approx(1.)
    assert full[1]["n_detected_genes_Spearman"] == pytest.approx(-1.)
    assert full[2]["log_library_Spearman"] is None
    assert all(row["measure"] == "TRUE_MEASURED_PROGRAM" for row in rows)
    assert all(type(row["group_code"]) is int and type(row["tissue_class"]) is int for row in rows)
    assert "DO_NOT_EXPORT" not in json.dumps(rows) and "features" not in json.dumps(rows)
    # Every within-section tissue has two spots, so it must remain undefined.
    tiny = [row for row in rows if row["scope"] == "SECTION" and row["tissue_class"] == 0]
    assert tiny and all(row["log_library_Spearman"] is None and row["support"] == 2 for row in tiny)

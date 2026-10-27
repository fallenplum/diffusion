"""Anonymous support-set audit guards; synthetic inputs only."""
import json

import numpy as np
import pytest

from scripts.bgb_en_checkpoint_audit import AuditError
from scripts.bgb_en_input_coverage import support_counts, union_counts


def test_support_separates_uncovered_genes_from_multipathway_membership():
    member = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]], bool)
    old = np.array([1, 1, 0, 1, 0], bool)
    row = support_counts(np.array([0, 1, 3]), member, old)
    assert row['token_supported_selected_gene_count'] == 2
    assert row['token_supported_selected_gene_fraction'] == pytest.approx(2 / 3)
    assert row['selected_membership_count_histogram'] == {'0': 1, '1': 1, '2': 1}
    assert row['selected_legacy_but_not_token_supported_count'] == 1
    assert row['selected_outside_legacy_union_count'] == 0
    assert 'indices' not in json.dumps(row)


def test_true_and_permuted_support_are_not_interchangeable():
    member = np.array([[1, 1, 0, 0]], bool)
    old = np.ones(4, bool)
    assert support_counts(np.array([0, 1]), member, old)['token_supported_selected_gene_count'] == 2
    assert support_counts(np.array([0, 1]), member[:, ::-1], old)['token_supported_selected_gene_count'] == 0


def test_union_denominators_have_explicit_direction():
    member = np.array([[1, 1, 1, 0]], bool)
    old = np.array([1, 0, 0, 1], bool)
    count = union_counts(member, old)
    assert count['legacy_union_covered_by_Hallmark50_fraction'] == .5
    assert count['Hallmark50_union_covered_by_legacy_fraction'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('selected', [np.array([1, 1]), np.array([-1]), np.array([4]), np.array([.5])])
def test_invalid_selection_fails_closed(selected):
    with pytest.raises(AuditError):
        support_counts(selected, np.array([[1, 0, 0, 0]], bool), np.ones(4, bool))

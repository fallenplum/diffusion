"""Support/mask invariants and an actual semantic change, no training."""
import numpy as np
import pytest
import torch

from scripts.bgb_en_checkpoint_audit import AuditError
from scripts.bgb_en_matched_support import base_model, matched_support_null


def fixture():
    member = np.array([[1, 1, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 0]], bool)
    rna = np.ones((6, 6), bool); cna = rna.copy()
    rna[0, 2:4] = False
    return member, rna, cna, np.array([0, 1, 2, 3])


def test_null_preserves_union_row_sizes_overlap_and_exact_train_mask_coverage():
    member, rna, cna, train = fixture()
    null, perm, audit = matched_support_null(member, rna, cna, train, seed=270199)
    null = null.numpy()
    assert not np.array_equal(null, member)
    assert np.array_equal(null.any(0), member.any(0))
    assert np.array_equal(null.sum(1), member.sum(1))
    assert np.array_equal(null.astype(int) @ null.astype(int).T, member.astype(int) @ member.astype(int).T)
    assert np.array_equal(rna[train].astype(int) @ null.astype(int).T, rna[train].astype(int) @ member.astype(int).T)
    assert np.array_equal(cna[train].astype(int) @ null.astype(int).T, cna[train].astype(int) @ member.astype(int).T)
    assert np.array_equal(perm.numpy()[4:], [4, 5])
    assert audit['changed_gene_annotation_count'] > 0
    assert audit['validation_masks_used'] is False


def test_validation_masks_cannot_change_null_or_source_permutation():
    member, rna, cna, train = fixture()
    a, p, _ = matched_support_null(member, rna, cna, train, seed=270199)
    rna[4:] = ~rna[4:]; cna[4:] = ~cna[4:]
    b, q, _ = matched_support_null(member, rna, cna, train, seed=270199)
    assert torch.equal(a, b) and torch.equal(p, q)


def test_RNA_and_CNA_mask_signatures_both_constrain_permutation():
    member, rna, cna, train = fixture()
    cna[1, 0] = False
    _, p, _ = matched_support_null(member, rna, cna, train, seed=270199)
    assert p[0].item() == 0  # unique CNA signature must be left fixed


def test_values_are_not_an_argument_and_metadata_have_no_selected_indices():
    member, rna, cna, train = fixture()
    _, _, audit = matched_support_null(member, rna, cna, train, seed=270199)
    assert audit['values_permuted'] is False and audit['gene_IDs_permuted'] is False
    assert set(audit) >= {'actual_membership_sha256', 'actual_permutation_sha256'}


def test_no_semantic_change_is_rejected_instead_of_claimed_as_a_null():
    with pytest.raises(AuditError, match='NO_SEMANTIC'):
        matched_support_null(np.ones((2, 4), bool), np.ones((3, 4), bool), np.ones((3, 4), bool), np.arange(3), seed=1)


@pytest.mark.parametrize('stage,variant,expected', [('R2', 'O4_MATCHED_SUPPORT', 'O4'), ('R3', 'F5_MATCHED_SUPPORT', 'F5')])
def test_constructor_dispatches_true_base_arm_and_does_not_apply_original_shuffle(stage, variant, expected):
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs)); return 'created'
    membership = torch.tensor([[True, False]])
    assert base_model(fake, stage, variant, {}, membership, None, 'cpu', gamma=.1) == 'created'
    assert calls[0][0][1] == expected
    assert calls[0][0][3] is membership


def test_wrong_stage_is_rejected():
    with pytest.raises(AuditError, match='STAGE'):
        base_model(lambda *a: None, 'R3', 'O4_MATCHED_SUPPORT', {}, torch.ones(1, 1), None, 'cpu')

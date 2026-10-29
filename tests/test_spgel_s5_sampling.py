"""S5 controls must preserve partition boundaries, support and real evidence."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from src.spgel.contracts import array_hash
from src.spgel_s5.sampling import (make_wrong_pairs, make_batch_candidates,
                                  permute_program_membership)


def source_fixture():
    n = 10
    source = SimpleNamespace(
        tissue=np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 2, 2], np.int64),
        section=np.arange(n, dtype=np.int64),
        donor=np.asarray([0, 0, 1, 1, 0, 0, 1, 1, 2, 2], np.int64),
        library=np.asarray([100, 101, 102, 103, 500, 501, 502, 503, 1000, 1001], float),
        xy_um=np.zeros((n, 2), dtype=np.float64),
        targets=np.arange(n * 4, dtype=float).reshape(n, 4))
    rows = np.arange(8, dtype=np.int64)
    stats = SimpleNamespace(fit_rows_sha256=array_hash(rows), fit_donor_codes=(0, 1),
        log_depth_quartiles=np.log1p([200., 400., 700.]))
    return source, rows, stats


class WrongPairTest(unittest.TestCase):
    def test_fixed_partition_only_bijection_and_no_source_mutation(self):
        source, rows, stats = source_fixture()
        original = deepcopy(source)
        out = make_wrong_pairs(source, rows, stats, seed=31)
        repeated = make_wrong_pairs(source, rows, stats, seed=31)
        p, a = out["partner_global_rows"], out["active"]
        self.assertTrue(a[rows].all())
        self.assertTrue(np.array_equal(np.sort(p[rows]), rows))
        self.assertTrue((p[rows] != rows).all())
        self.assertTrue(np.all(p[8:] == -1))
        self.assertTrue(np.array_equal(source.tissue[rows], source.tissue[p[rows]]))
        self.assertTrue(np.array_equal(p, repeated["partner_global_rows"]))
        for name in vars(source):
            self.assertTrue(np.array_equal(getattr(source, name), getattr(original, name)))
        # Poisoning heldout metadata cannot change the training remapping.
        source.library[8:] = np.nan
        source.xy_um[8:] = np.nan
        source.tissue[8:] = -999
        after = make_wrong_pairs(source, rows, stats, seed=31)
        self.assertTrue(np.array_equal(p, after["partner_global_rows"]))

    def test_infeasible_whole_stratum_is_inactive_not_relaxed(self):
        source, rows, stats = source_fixture()
        source.section[:4] = 0
        # Same-slide centers coincide, so all four candidate pairs are illegal.
        out = make_wrong_pairs(source, rows, stats, seed=47)
        self.assertFalse(out["active"][:4].any())
        self.assertTrue(np.array_equal(out["partner_global_rows"][:4], rows[:4]))
        self.assertTrue(out["active"][4:8].all())
        self.assertEqual(out["audit"]["ITC_active_rows"], 4)

    def test_sampler_statistics_wrong_scope_is_rejected(self):
        source, rows, stats = source_fixture()
        stats.fit_rows_sha256 = array_hash(rows[:-1])
        with self.assertRaisesRegex(ValueError, "STATISTICS_TRAIN_SCOPE"):
            make_wrong_pairs(source, rows, stats, seed=31)


class BatchCandidateTest(unittest.TestCase):
    def test_shared_original_bank_negatives_exclude_both_sources(self):
        source, rows, stats = source_fixture()
        remap = make_wrong_pairs(source, rows, stats, seed=31)
        ni, ng, active, audit = make_batch_candidates(source, rows,
            remap["partner_global_rows"][rows], remap["active"][rows], stats, seed=91)
        self.assertTrue(np.array_equal(ni, ng))
        self.assertEqual(ni.shape, (8, 8))
        self.assertTrue(active.all())
        for i in range(len(rows)):
            valid = ni[i][ni[i] >= 0]
            self.assertEqual(len(valid), len(set(valid)))
            self.assertTrue((valid < len(rows)).all())
            self.assertNotIn(rows[i], rows[valid])
            self.assertNotIn(remap["partner_global_rows"][rows[i]], rows[valid])
        self.assertTrue(audit["separate_directional_CE_required"])

    def test_near_either_source_excluded_and_no_negative_disables(self):
        source, _, stats = source_fixture()
        rows = np.asarray([0, 2, 4], np.int64)
        partners = np.asarray([1, 3, 5], np.int64)
        # For anchor0: original row2 is near source0, and original row4 is near
        # wrong source1. Distances and section identities, not expression, rule.
        source.section[0] = source.section[2] = 0
        source.section[1] = source.section[4] = 1
        source.xy_um[[0, 2]] = [[0, 0], [100, 0]]
        source.xy_um[[1, 4]] = [[500, 0], [600, 0]]
        ni, ng, active, audit = make_batch_candidates(source, rows, partners,
            np.ones(3, bool), stats, seed=31)
        self.assertFalse(active[0])
        self.assertTrue((ni[0] == -1).all())
        self.assertTrue(np.array_equal(ni, ng))
        self.assertGreaterEqual(audit["no_negative_anchors_disabled_in_all_arms"], 1)

    def test_heldout_partner_rejected_even_if_inactive(self):
        source, _, stats = source_fixture()
        with self.assertRaisesRegex(ValueError, "CROSSES_TRAIN_DONORS"):
            make_batch_candidates(source, np.asarray([0, 2]), np.asarray([8, 3]),
                np.asarray([False, True]), stats, seed=31)


class MembershipTest(unittest.TestCase):
    def test_column_null_changes_membership_not_truth_and_ignores_heldout(self):
        source, rows, stats = source_fixture()
        genes = 64
        rng = np.random.default_rng(3)
        source.genes = rng.uniform(.1, 2., (10, genes))
        source.gene_mask = np.ones((10, genes), bool)
        source.up = np.zeros((4, genes), bool)
        source.down = np.zeros((4, genes), bool)
        for i in range(genes):
            source.up[i % 4, i] = True
        source.down[0, 1:8:4] = True
        before = deepcopy(source)
        out = permute_program_membership(source, rows, stats, seed=31)
        p = out["gene_permutation"]
        self.assertTrue(np.array_equal(np.sort(p), np.arange(genes)))
        self.assertGreater(out["audit"]["changed_signed_membership_cells"], 0)
        self.assertTrue(np.array_equal(source.up.sum(1), out["up"].sum(1)))
        self.assertTrue(np.array_equal(source.down.sum(1), out["down"].sum(1)))
        old = np.concatenate((source.up, source.down)).astype(int)
        new = np.concatenate((out["up"], out["down"])).astype(int)
        self.assertTrue(np.array_equal(old @ old.T, new @ new.T))
        for name in vars(source):
            self.assertTrue(np.array_equal(getattr(source, name), getattr(before, name)))
        source.genes[8:] = np.nan
        source.gene_mask[8:] = False
        again = permute_program_membership(source, rows, stats, seed=31)
        self.assertTrue(np.array_equal(p, again["gene_permutation"]))

    def test_no_effective_edge_change_is_rejected(self):
        source, rows, stats = source_fixture()
        source.genes = np.zeros((10, 8), float)
        source.gene_mask = np.ones((10, 8), bool)
        source.up = np.ones((4, 8), bool)
        source.down = np.zeros((4, 8), bool)
        with self.assertRaisesRegex(ValueError, "DID_NOT_CHANGE_ANY_EDGE"):
            permute_program_membership(source, rows, stats, seed=31)


if __name__ == "__main__":
    unittest.main()

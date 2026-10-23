"""Synthetic signed-program and real-coordinate neighbor preparation tests."""
from __future__ import annotations

import copy
import unittest

import numpy as np
from scipy import sparse

from scripts.prepare_sami_stage_b import signed_program_scores, spatial_neighbors, validate_programs


def program(up, down=(), identifier="synthetic_program"):
    return dict(id=identifier, genes=list(up) + list(down), up_genes=list(up), down_genes=list(down),
                source_url="https://example.org/original-table", source_table="synthetic-table",
                source_file_sha256="synthetic_sha", source_hypothesis="associated_not_true_source",
                limitations="synthetic_only")


class SignedProgramTests(unittest.TestCase):
    def setUp(self):
        self.genes = np.asarray([f"gene_{i}" for i in range(12)])
        self.program = program(self.genes[:5], self.genes[5:10])
        self.counts = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 500, 1000],
                                  [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 250, 100]], dtype=np.float32)

    def test_float64_full_library_signed_reference(self):
        y, lib, detected, coverage = signed_program_scores(self.counts, self.genes, [self.program])
        reference_lib = self.counts.astype(np.float64).sum(axis=1)
        log_counts = np.log1p(self.counts.astype(np.float64) / reference_lib[:, None] * 10000.)
        expected = log_counts[:, :5].mean(1) - log_counts[:, 5:10].mean(1)
        np.testing.assert_allclose(y[:, 0], expected, rtol=0, atol=2e-15)
        np.testing.assert_array_equal(lib, reference_lib)
        np.testing.assert_array_equal(detected, [11, 12])
        self.assertEqual(y.dtype, np.float64)
        self.assertEqual(coverage[0]["measured_genes"], 10)

    def test_sparse_dense_parity_including_explicit_zeros(self):
        dense = signed_program_scores(self.counts, self.genes, [self.program])
        matrix = sparse.csr_matrix(self.counts)
        # Explicit zero storage must not increase detected-gene counts.
        matrix[0, 0] = 0
        actual = signed_program_scores(matrix, self.genes, [self.program])
        for left, right in zip(dense[:3], actual[:3]):
            np.testing.assert_array_equal(left, right)

    def test_zero_counts_remain_measured(self):
        counts = self.counts.copy()
        counts[:, :10] = 0
        y, _, _, coverage = signed_program_scores(counts, self.genes, [self.program])
        np.testing.assert_array_equal(y, np.zeros((2, 1)))
        self.assertEqual(coverage[0]["measured_coverage"], 1.)

    def test_unsigned_program_is_mean_up(self):
        p = program(self.genes[:5])
        y, lib, _, _ = signed_program_scores(self.counts, self.genes, [p])
        expected = np.log1p(self.counts[:, :5].astype(np.float64) * (10000. / lib[:, None])).mean(1)
        np.testing.assert_array_equal(y[:, 0], expected)

    def test_sign_reversal_changes_target_sign(self):
        p = program(self.genes[5:10], self.genes[:5])
        y = signed_program_scores(self.counts, self.genes, [self.program])[0]
        swapped = signed_program_scores(self.counts, self.genes, [p])[0]
        np.testing.assert_array_equal(y, -swapped)

    def test_measurement_threshold_and_measured_denominator(self):
        genes = np.asarray([f"gene_{i}" for i in range(19)])
        p = program([f"gene_{i}" for i in range(20)])
        counts = np.arange(1, 20, dtype=np.float64)[None, :]
        y, lib, _, audit = signed_program_scores(counts, genes, [p])
        self.assertEqual(audit[0]["measured_coverage"], .95)
        expected = np.log1p(counts * (10000. / lib[:, None])).mean(1)
        np.testing.assert_array_equal(y[:, 0], expected)
        with self.assertRaisesRegex(ValueError, "coverage"):
            signed_program_scores(counts[:, :-1], genes[:-1], [p])

    def test_nonempty_direction_minimum_measured_genes(self):
        genes = np.asarray([f"g{i}" for i in range(100)])
        # High total coverage cannot hide a tiny signed direction.
        p = program(list(genes[:95]), list(genes[95:99]))
        with self.assertRaisesRegex(ValueError, "direction"):
            signed_program_scores(np.ones((1, 100)), genes, [p])

    def test_invalid_counts_and_zero_library_fail(self):
        for value in (-1, .5, np.nan, np.inf):
            counts = self.counts.copy()
            counts[0, 0] = value
            with self.assertRaisesRegex(ValueError, "integer counts"):
                signed_program_scores(counts, self.genes, [self.program])
        with self.assertRaisesRegex(ValueError, "library"):
            signed_program_scores(np.zeros_like(self.counts), self.genes, [self.program])

    def test_gene_duplicates_and_program_overlap_fail(self):
        genes = self.genes.copy()
        genes[1] = genes[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            signed_program_scores(self.counts, genes, [self.program])
        p = program(self.genes[:5], self.genes[:5])
        with self.assertRaisesRegex(ValueError, "gene set"):
            signed_program_scores(self.counts, self.genes, [p])

    def test_external_program_schema_exact_direction_and_provenance(self):
        record = dict(schema_version="sami_surv_stage_b_programs_v1", programs=[self.program])
        validate_programs(record, require_locked_order=False)
        for mutation in ("genes", "source_table", "duplicate", "short_down"):
            altered = copy.deepcopy(record)
            p = altered["programs"][0]
            if mutation == "genes": p["genes"] = list(reversed(p["genes"]))
            if mutation == "source_table": p["source_table"] = ""
            if mutation == "duplicate": altered["programs"].append(copy.deepcopy(p))
            if mutation == "short_down": p["down_genes"] = p["down_genes"][:4]
            with self.assertRaises(ValueError):
                validate_programs(altered, require_locked_order=False)


class SpatialNeighborTests(unittest.TestCase):
    def test_same_section_even_when_other_slide_spatially_identical(self):
        X = np.asarray([[1, 2], [3, 4], [100, 200], [300, 400]], dtype=np.float32)
        coords = np.asarray([[0, 0], [100, 0], [0, 0], [100, 0]], dtype=np.float64)
        result, indices, counts, audit = spatial_neighbors(X, coords, ["a", "a", "b", "b"])
        np.testing.assert_array_equal(result, [[2, 3], [2, 3], [200, 300], [200, 300]])
        np.testing.assert_array_equal(counts, [2, 2, 2, 2])
        self.assertEqual(audit["cross_section_edges"], 0)
        self.assertEqual(indices[0, :2].tolist(), [0, 1])

    def test_distance_boundary_self_and_stable_global_row_tie(self):
        X = np.arange(5, dtype=np.float32)[:, None]
        coords = np.asarray([[0, 0], [100, 0], [-100, 0], [300, 0], [300.001, 0]])
        result, indices, _, _ = spatial_neighbors(X, coords, ["s"] * 5, maximum=3)
        self.assertEqual(indices[0].tolist(), [0, 1, 2])
        self.assertEqual(float(result[0, 0]), 1.)
        _, indices, counts, _ = spatial_neighbors(X, coords, ["s"] * 5, maximum=16)
        self.assertEqual(indices[0, :counts[0]].tolist(), [0, 1, 2, 3])

    def test_neighbor_cap_is_sixteen_including_self(self):
        X = np.arange(21, dtype=np.float32)[:, None]
        coords = np.column_stack([np.arange(21), np.zeros(21)])
        result, indices, counts, _ = spatial_neighbors(X, coords, ["s"] * 21)
        self.assertEqual(indices[0, :16].tolist(), list(range(16)))
        self.assertTrue(np.all(counts == 16))
        self.assertEqual(float(result[0, 0]), 7.5)

    def test_isolated_spot_preserves_own_value(self):
        X = np.asarray([[4.], [9.]], dtype=np.float32)
        result, _, counts, audit = spatial_neighbors(X, [[0, 0], [301, 0]], ["s", "s"])
        np.testing.assert_array_equal(result, X)
        np.testing.assert_array_equal(counts, [1, 1])
        self.assertEqual(audit["self_only_rows"], 2)

    def test_duplicate_coordinates_or_nonfinite_features_fail(self):
        X = np.asarray([[4.], [9.]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "duplicate physical"):
            spatial_neighbors(X, [[0, 0], [0, 0]], ["s", "s"])
        X[0] = np.nan
        with self.assertRaisesRegex(ValueError, "feature"):
            spatial_neighbors(X, [[0, 0], [100, 0]], ["s", "s"])

    def test_full_float64_mean_then_frozen_float32_output(self):
        X = np.asarray([[2**24], [1], [-2**24]], dtype=np.float32)
        result, _, _, _ = spatial_neighbors(X, [[0, 0], [1, 0], [2, 0]], ["s"] * 3)
        np.testing.assert_array_equal(result, np.full((3, 1), np.float32(1 / 3)))


if __name__ == "__main__":
    unittest.main()

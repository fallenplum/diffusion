import unittest

import numpy as np
import torch

from src.spgel_hest_neural.graph import build_graph


class GraphTests(unittest.TestCase):
    def test_complete_ties_axis_swap_translation_stable(self):
        coords = np.asarray([(x, y) for x in range(-3, 4) for y in range(-3, 4)], dtype=float)
        specimens = np.zeros(len(coords), dtype=np.int64)
        original = build_graph(coords, specimens)
        swapped = build_graph(coords[:, ::-1], specimens)
        translated = build_graph(coords + [100002, -100004], specimens)
        np.testing.assert_array_equal(original.edge_index, swapped.edge_index)
        np.testing.assert_array_equal(original.edge_index, translated.edge_index)
        row = 24
        selected = original.edge_index[1, original.edge_index[0] == row]
        other = np.delete(np.arange(len(coords)), row)
        d2 = np.sum((coords[other] - coords[row])**2, axis=1)
        expected = np.r_[row, other[np.lexsort((other, d2))][:15]]
        np.testing.assert_array_equal(selected, expected)

    def test_duplicate_coordinates_self_always_included_and_real_counts(self):
        coords = np.zeros((18, 2))
        graph = build_graph(coords, np.zeros(18, dtype=np.int64))
        for row in range(18):
            selected = graph.edge_index[1, graph.edge_index[0] == row]
            self.assertEqual(selected[0], row)
            self.assertEqual(len(set(selected)), 16)
            expected = [row] + [j for j in range(18) if j != row][:15]
            self.assertEqual(selected.tolist(), expected)

    def test_no_cross_slide_even_when_locations_overlap(self):
        coords = np.asarray([[0, 0], [1, 0], [0, 0], [1, 0], [2, 0]], dtype=float)
        specimen = np.asarray([0, 0, 1, 1, 1])
        graph = build_graph(coords, specimen)
        self.assertTrue(np.all(specimen[graph.edge_index[0]] == specimen[graph.edge_index[1]]))
        np.testing.assert_array_equal(graph.degree, [2, 2, 3, 3, 3])
        torch.testing.assert_close(graph.torch_adjacency().to_dense().sum(1), torch.ones(5))

    def test_identity_exact_and_sparse(self):
        graph = build_graph(np.arange(10).reshape(5, 2), np.asarray([0, 0, 1, 1, 1]), identity=True)
        np.testing.assert_array_equal(graph.edge_index, np.stack([np.arange(5), np.arange(5)]))
        np.testing.assert_array_equal(graph.degree, np.ones(5))
        self.assertEqual(graph.torch_adjacency().layout, torch.sparse_coo)
        torch.testing.assert_close(graph.torch_adjacency().to_dense(), torch.eye(5))

    def test_nonfinite_and_bad_alignment_rejected(self):
        for coords, specimens in [(np.array([[np.nan, 0]]), [0]), (np.zeros((2, 2)), [0]),
                                   (np.zeros((1, 2)), [-1]), (np.zeros((1, 2)), [0.5])]:
            with self.assertRaises(ValueError):
                build_graph(coords, specimens)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch

from src.spgel_hest_neural.graph import build_graph
from src.spgel_hest_neural.models import ARMS, build_model, parameter_count


class ModelTests(unittest.TestCase):
    def test_initialization_common_projections_and_graph_all_parameters_equal(self):
        models = [build_model(arm, seed=1) for arm in ARMS]
        for model in models[1:]:
            for name in ("input_projection.weight", "input_projection.bias", "output_projection.weight", "output_projection.bias"):
                torch.testing.assert_close(model.state_dict()[name], models[0].state_dict()[name], rtol=0, atol=0)
                self.assertNotEqual(model.state_dict()[name].data_ptr(), models[0].state_dict()[name].data_ptr())
        for key in models[1].state_dict():
            torch.testing.assert_close(models[1].state_dict()[key], models[2].state_dict()[key], rtol=0, atol=0)
        self.assertEqual([parameter_count(model) for model in models], [55858, 72242, 72242])

    def test_exact_graph_formula_and_identity_neighbour_active_gradient(self):
        torch.manual_seed(8)
        x = torch.randn(20, 256)
        coords = np.column_stack([np.arange(20), np.zeros(20)])
        graph = build_graph(coords, np.zeros(20, dtype=np.int64), identity=True)
        model = build_model("IDENTITY_GNN")
        prediction = model(x, graph)
        h = torch.relu(model.input_projection(x))
        expected = model.output_projection(torch.relu(model.self_projection(h) + model.neighbour_projection(h) + model.hidden_bias))
        torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
        prediction.square().mean().backward()
        for name, value in model.named_parameters():
            self.assertIsNotNone(value.grad, name)
            self.assertTrue(torch.isfinite(value.grad).all(), name)
            self.assertGreater(value.grad.abs().sum().item(), 0, name)

    def test_image_predict_has_no_target_input_or_batch_statistics(self):
        torch.manual_seed(5)
        x = torch.randn(32, 256)
        model = build_model("MLP")
        self.assertFalse(any(isinstance(m, (torch.nn.BatchNorm1d, torch.nn.Dropout)) for m in model.modules()))
        whole = model(x)
        separate = torch.cat([model(x[:11]), model(x[11:])])
        torch.testing.assert_close(whole, separate, rtol=1e-5, atol=1e-6)
        self.assertEqual(whole.shape, (32, 50))

    def test_factory_preserves_global_rng_and_arms_train_independently(self):
        torch.manual_seed(42)
        before = torch.random.get_rng_state().clone()
        graph = build_model("SPATIAL_GNN")
        identity = build_model("IDENTITY_GNN")
        torch.testing.assert_close(before, torch.random.get_rng_state(), rtol=0, atol=0)
        with torch.no_grad():
            graph.input_projection.weight.add_(1)
        self.assertFalse(torch.equal(graph.input_projection.weight, identity.input_projection.weight))


if __name__ == "__main__":
    unittest.main()

"""Fixed two-ReLU controls; Graph/Identity differ only in their adjacency."""
from __future__ import annotations

import hashlib

import torch
from torch import nn

from .graph import Graph

ARMS = ("MLP", "SPATIAL_GNN", "IDENTITY_GNN")


def _module_seed(seed, label):
    payload = f"spgel_hest_neural_v1:{seed}:{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _linear(in_features, out_features, seed, label, bias=True):
    # Common module labels deliberately give identical input/output weights
    # across arms. fork_rng preserves the caller's random state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(_module_seed(seed, label))
        return nn.Linear(in_features, out_features, bias=bias)


class ImageOnlyBaseline(nn.Module):
    def __init__(self, arm, seed=1):
        super().__init__()
        if arm not in ARMS:
            raise ValueError("UNKNOWN_PRELOCKED_ARM")
        self.arm = arm
        self.input_projection = _linear(256, 128, seed, "input")
        self.output_projection = _linear(128, 50, seed, "output")
        if arm == "MLP":
            self.hidden_projection = _linear(128, 128, seed, "mlp_hidden")
        else:
            self.self_projection = _linear(128, 128, seed, "graph_self", bias=False)
            self.neighbour_projection = _linear(128, 128, seed, "graph_neighbour", bias=False)
            self.hidden_bias = nn.Parameter(torch.zeros(128))

    def forward(self, x, graph=None):
        if x.ndim != 2 or x.shape[1] != 256:
            raise ValueError("IMAGE_PCA256_MATRIX_REQUIRED")
        h = torch.relu(self.input_projection(x))
        if self.arm == "MLP":
            hidden = self.hidden_projection(h)
        else:
            if isinstance(graph, Graph):
                graph = graph.torch_adjacency(device=x.device, dtype=x.dtype)
            if graph is None or graph.layout != torch.sparse_coo or tuple(graph.shape) != (len(x), len(x)):
                raise ValueError("ALIGNED_SPARSE_ADJACENCY_REQUIRED")
            message = torch.sparse.mm(graph, h)
            hidden = self.self_projection(h) + self.neighbour_projection(message) + self.hidden_bias
        return self.output_projection(torch.relu(hidden))


def build_model(arm, seed=1):
    return ImageOnlyBaseline(arm, seed=seed)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""Exact fixed recipe and a separate scientific code inventory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn
import torch

SOURCE_FILES = (
    "src/spgel_hest_neural/__init__.py",
    "src/spgel_hest_neural/graph.py",
    "src/spgel_hest_neural/models.py",
    "src/spgel_hest_neural/recipe.py",
    "src/spgel_hest_neural/neural_execution.py",
    "configs/spgel_hest_neural/pilot_v1.json",
    "scripts/spgel_hest_neural.py",
    "tests/test_spgel_hest_neural_graph.py",
    "tests/test_spgel_hest_neural_models.py",
    "tests/test_spgel_hest_neural_recipe.py",
    "tests/test_spgel_hest_neural_execution.py",
)
REPO = Path(__file__).resolve().parents[2]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe(path):
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError("DUPLICATE_RECIPE_JSON_KEY")
            output[key] = value
        return output
    result = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_RECIPE")))
    canonical = json.loads((REPO / "configs/spgel_hest_neural/pilot_v1.json").read_text(encoding="utf-8"))
    if result != canonical:
        raise ValueError("REGISTERED_NEURAL_RECIPE_CHANGED")
    if result["budget"] != {
        "neural_fits": 12, "epochs_per_fit": 40, "optimizer_updates_per_fit": 40,
        "optimizer_updates_total": 480, "snapshot_epochs": [10, 20, 40],
        "actual_state_replays": 36, "new_unsupervised_PCA_reconstructions": 4,
        "new_Ridge_fits": 0, "automatic_retry_fits": 0,
    } or result["tasks"] != ["COAD", "READ"] or result["arms"] != ["MLP", "SPATIAL_GNN", "IDENTITY_GNN"]:
        raise ValueError("REGISTERED_NEURAL_BUDGET_CHANGED")
    return result


def source_hashes():
    return {name: file_sha256(REPO / name) for name in SOURCE_FILES}


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__, "device": "CPU",
            "threads_per_process": 2, "maximum_parallel_workers": 2}

"""Fixed finite S5-B1 contract; no official-test model selection."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import scipy
import sklearn
import torch
from ..spgel_hest.assets import read_json, sha256, canonical_hash
from ..spgel_hest.official import require

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("MLP", "GRAPHSAGE", "IDENTITY", "QUERY4", "SPGEL50_ADAPT")
CHECKPOINTS = (80, 160, 320, 640, 1280)
SOURCES = (
    "src/spgel_s5_hest/__init__.py", "src/spgel_s5_hest/protocol.py",
    "src/spgel_s5_hest/data.py", "src/spgel_s5_hest/models.py",
    "src/spgel_s5_hest/execution.py", "scripts/spgel_s5_hest.py",
    "configs/spgel_s5_hest/shared_feature_v1.json",
    "tests/test_spgel_s5_hest.py",
)
INHERITED = ("src/spgel_hest/assets.py", "src/spgel_hest/official.py",
    "src/spgel_hest_neural/graph.py", "src/spgel_hest_neural/models.py")

def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

def recipe(path):
    value = read_json(path)
    expected = read_json(ROOT / "configs/spgel_s5_hest/shared_feature_v1.json")
    require(value == expected and value["arms"] == list(ARMS)
        and value["development_checkpoints"] == list(CHECKPOINTS)
        and value["development_max_epochs"] == 1280 and value["development_saved_states"]==100
        and value["final_saved_states"]==20 and value["paired_anchors_maximum"]==512
        and value["paired_negative_candidates"]==8 and value["paired_temperature"]==0.1
        and value["paired_objective_weight"]==0.1 and value["paired_MSE_or_gene_reconstruction_objective"] is False
        and value["development_fits"] == value["final_fits"] == 20
        and value["selection_scope"] == "each_outer_fold_own_inner_validation_only",
        "FIXED_S5_SHARED_FEATURE_RECIPE_REQUIRED")
    return value

def source_hashes():
    return {name: sha256(ROOT / name) for name in (*SOURCES, *INHERITED)}

def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, torch=torch.__version__, device="CPU", threads=2)

def private_root(path):
    p = Path(path).resolve()
    require(not p.is_relative_to(ROOT) and not any((a / ".git").exists() for a in (p, *p.parents)),
        "PRIVATE_S5_NAMESPACE_OUTSIDE_SOURCE_REPOSITORY_REQUIRED")
    p.mkdir(parents=True, exist_ok=True)
    return p

def array_sha(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def name(dataset, fold, arm=None):
    require(dataset in ("COAD", "READ") and fold in (0, 1), "INVALID_S5_OFFICIAL_FOLD")
    require(arm is None or arm in ARMS, "INVALID_S5_ARM")
    return f"{dataset}_fold{fold}" + ("_" + arm if arm else "")

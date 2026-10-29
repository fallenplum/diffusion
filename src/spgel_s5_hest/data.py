"""Inner-only PCA and graphs; outer-test expression stays sealed during fitting."""
from __future__ import annotations
from pathlib import Path
import h5py
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from ..spgel_hest.assets import bound_file
from ..spgel_hest.official import require
from ..spgel_hest_neural.graph import build_graph
from .protocol import array_sha, write_json

def coordinates(assets, dataset, partition):
    pieces = []
    for i, specimen in enumerate(partition.specimen_ids):
        with h5py.File(bound_file(assets["tasks"][dataset]["samples"][specimen]["features"]), "r") as f:
            bs = f["barcodes"][:].flatten().astype(str)
            xy = f["coords"][:]
        mask = partition.specimen_index == i
        require(np.array_equal(bs, partition.barcodes[mask])
            and xy.shape == (int(mask.sum()), 2) and np.isfinite(xy).all(),
            "S5_COORDINATE_AND_FEATURE_BARCODE_ORDER_CHANGED")
        pieces.append(xy)
    return np.concatenate(pieces)

def inner_indices(coords, specimen_index):
    """No outcome-dependent split; one slide held out or spatial blocks."""
    xy, slide = np.asarray(coords), np.asarray(specimen_index)
    require(xy.ndim == 2 and xy.shape[1] == 2 and len(slide) == len(xy)
        and np.isfinite(xy).all(), "S5_INNER_SPLIT_GEOMETRY")
    unique = np.unique(slide)
    if len(unique) > 1:
        val = slide == unique[0]
        method = "FIRST_OFFICIAL_TRAIN_SLIDE_HELD_OUT_NOT_ALWAYS_DONOR_DISJOINT"
        block_sha = None
    else:
        low, span = xy.min(0), np.ptp(xy, axis=0)
        require(np.all(span > 0), "S5_SINGLE_SLIDE_NEEDS_NONDEGENERATE_COORDINATES")
        grid = np.minimum(np.floor((xy-low)/span*8).astype(np.int64), 7)
        val = (grid[:, 0] + 2*grid[:, 1]) % 5 == 0
        method = "SINGLE_TRAIN_SLIDE_FIXED_EIGHT_BY_EIGHT_SPATIAL_BLOCKS"
        block_sha = array_sha(grid)
    train_ids, val_ids = np.flatnonzero(~val), np.flatnonzero(val)
    require(len(train_ids) >= 256 and len(val_ids) >= 2
        and not np.intersect1d(train_ids, val_ids).size, "S5_INNER_TRAIN_PCA256_OR_VALIDATION_INFEASIBLE")
    return train_ids, val_ids, dict(method=method, train_rows=len(train_ids), validation_rows=len(val_ids),
        train_index_sha256=array_sha(train_ids), validation_index_sha256=array_sha(val_ids),
        block_assignment_sha256=block_sha, uses_expression_or_outcomes=False,
        graph_cross_inner_partition_edges=0, donor_heldout_validation_claimed=False,
        spatial_buffer_between_inner_train_and_validation=False,
        single_slide_spatial_autocorrelation_is_not_removed=True)

def fit_inputs(train_raw, other_raw):
    require(min(train_raw.shape) >= 256, "S5_TRAIN_ONLY_PCA256_INFEASIBLE")
    pipe = Pipeline([("scaler", StandardScaler()), ("PCA", PCA(n_components=256, random_state=1))])
    with threadpool_limits(limits=2):
        left = np.asarray(pipe.fit_transform(train_raw), np.float32)
        right = np.asarray(pipe.transform(other_raw), np.float32) if other_raw is not None else None
    require(np.isfinite(left).all() and (right is None or np.isfinite(right).all()), "S5_NONFINITE_PCA")
    return pipe, left, right

def save_cache(folder, payload, pipeline, metadata):
    folder = Path(folder);folder.mkdir(parents=True, exist_ok=False)
    with (folder/"inputs.private.npz").open("xb") as f:
        np.savez_compressed(f, **payload)
    joblib.dump(pipeline, folder/"pipeline.private.joblib", compress=0)
    from ..spgel_hest.assets import sha256
    meta = dict(metadata, schema="s5_hest_input_cache_v1", status="PASS",
        arrays_sha256={k: array_sha(v) for k,v in payload.items()},
        input_file_sha256=sha256(folder/"inputs.private.npz"),
        pipeline_sha256=sha256(folder/"pipeline.private.joblib"))
    write_json(folder/"CACHE.private.json", meta)
    return meta

def load_cache(folder, expected):
    from ..spgel_hest.assets import sha256, read_json
    folder = Path(folder)
    require(sha256(folder/"CACHE.private.json") == expected["cache_metadata_sha256"], "S5_CACHE_METADATA_CHANGED")
    meta=read_json(folder/"CACHE.private.json")
    require(sha256(folder/"inputs.private.npz") == meta["input_file_sha256"]
        and sha256(folder/"pipeline.private.joblib") == meta["pipeline_sha256"], "S5_CACHE_FILE_CHANGED")
    with np.load(folder/"inputs.private.npz", allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    require({k:array_sha(v) for k,v in arrays.items()} == meta["arrays_sha256"], "S5_CACHE_ARRAY_CHANGED")
    return arrays,meta

def graph_payload(xy, specimen):
    graph=build_graph(xy, specimen)
    # Graph rows have true degrees; padding is masked and never an extra neighbour.
    indices=np.zeros((len(xy),16),dtype=np.int64);valid=np.zeros_like(indices,dtype=bool)
    order=np.argsort(graph.edge_index[0],kind="stable")
    columns=graph.edge_index[1,order]
    starts=np.concatenate(([0],np.cumsum(graph.degree)))
    for row in range(len(xy)):
        cols=columns[starts[row]:starts[row+1]]
        indices[row,:len(cols)]=cols;valid[row,:len(cols)]=True
    return graph,indices,valid

import copy
import numpy as np
import pytest
import torch

from src.spgel.contracts import SourceStatistics, SpatialSource, program_scores
from src.spgel.geometry import build_neighbors, hard_negative_rows, localize_negative_indices
from src.spgel.metrics import regression_metrics, retrieval_metrics, localization_metrics


def source():
    rng = np.random.default_rng(148)
    n, u = 36, 24
    x = rng.normal(size=(n, 1536)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    counts = rng.poisson(3., size=(n, u)).astype(float)
    library = counts.sum(1) + 1000.
    genes = np.log1p(counts * (10000. / library[:, None]))
    observed = np.ones_like(genes, bool)
    up, down = np.zeros((4, u), bool), np.zeros((4, u), bool)
    for k in range(4):
        up[k, (np.arange(6) + k * 3) % u] = True
        down[k, (np.arange(6) + k * 3 + 8) % u] = True
    targets, valid = program_scores(genes, observed, up, down)
    donor = np.repeat(np.arange(6), 6)
    section = np.repeat(np.arange(12), 3)
    xy = np.c_[np.tile(np.arange(3) * 120., 12), np.zeros(n)]
    graph = build_neighbors(xy, section)
    return SpatialSource(x, genes, observed, targets, valid, np.arange(n) % 3,
        xy, section, donor, library, graph, up, down, np.array([f"SYN_GENE_{k}" for k in range(u)]), {}).validate()


def test_source_statistics_cannot_fit_heldout_and_do_not_depend_on_it():
    a = source(); train, test = a.split(3)
    stats = SourceStatistics.fit(a, train, heldout_donors=[3])
    b = copy.deepcopy(a)
    b.targets[test] += 5000; b.genes[test] *= 1000; b.library[test] *= 100
    other = SourceStatistics.fit(b, train, heldout_donors=[3])
    for name in ("target_mean", "target_sd", "gene_mean", "gene_sd", "high_program_threshold", "log_depth_quartiles"):
        assert np.array_equal(getattr(stats, name), getattr(other, name))
    with pytest.raises(ValueError, match="HELDOUT_DONOR"):
        SourceStatistics.fit(a, np.r_[train, test[:1]], heldout_donors=[3])


def test_neighbor_geometry_same_slide_and_partition():
    a = source(); train, test = a.split(0)
    assert np.all(a.section[np.maximum(a.neighbors, 0)][a.neighbors >= 0]
                  == np.broadcast_to(a.section[:, None], a.neighbors.shape)[a.neighbors >= 0])
    a.image_batch(test, allowed_rows=test)
    bad = copy.deepcopy(a); bad.neighbors[train[0], 0] = test[0]
    with pytest.raises(ValueError, match="NEIGHBOR_CROSSES"):
        bad.image_batch(train[:1], allowed_rows=train)
    with pytest.raises(ValueError, match="SOURCE_GRAPH_CROSSES"):
        bad.validate()


def test_hard_negatives_are_scoped_exclude_nearby_and_use_train_depth():
    a = source(); train, test = a.split(0)
    stats = SourceStatistics.fit(a, train, heldout_donors=[0])
    neg, audit = hard_negative_rows(section=a.section, tissue=a.tissue, library=a.library,
        xy_um=a.xy_um, rows=train, train_depth_quartiles=stats.log_depth_quartiles, count=4, seed=47)
    assert audit["expression_targets_used_for_selection"] is False
    local = localize_negative_indices(neg, train)
    assert (local >= 0).any() and not np.isin(neg[train][neg[train] >= 0], test).any()
    for row in train:
        other = neg[row][neg[row] >= 0]
        assert row not in other
        assert np.all((a.section[other] != a.section[row]) | (np.square(a.xy_um[other] - a.xy_um[row]).sum(1) > 300.**2))


def test_constant_correlations_and_missing_support_are_not_zero():
    target = np.c_[np.arange(7), np.ones(7), np.arange(7)]
    pred = target.copy(); mask = np.ones_like(target, bool); mask[:, 2] = False
    metric = regression_metrics(pred, target, mask)
    assert metric["per_output"][0]["Pearson"] == pytest.approx(1.)
    assert metric["per_output"][1]["Pearson"] is None and metric["per_output"][2]["Pearson"] is None
    assert metric["per_output"][2]["MAE"] is None and metric["defined_Pearson_count"] == 1


def test_retrieval_ties_receive_fractional_expected_credit():
    metric = retrieval_metrics(np.ones(3), np.ones((3, 8)), np.ones((3, 8), bool))
    assert metric["Recall_at_1"] == pytest.approx(1 / 9)
    assert metric["Recall_at_5"] == pytest.approx(5 / 9)
    assert metric["median_rank"] == 5 and metric["pair_AUC"] == .5
    no_support = retrieval_metrics(np.ones(3), np.ones((3, 8)), np.zeros((3, 8), bool))
    assert no_support["Recall_at_1"] is None and no_support["anchors_count"] == 0


def test_high_activity_is_evaluated_at_given_train_threshold():
    target = np.arange(20).reshape(10, 2)
    prediction = target.copy()
    result = localization_metrics(prediction, target, np.ones_like(target, bool), np.array([8., 9.]))
    assert result["Mean_AUPRC"] == 1 and result["threshold_fit_scope"] == "source_training_rows_only"
    assert result["per_output"][0]["high_activity_threshold_from_train"] == 8.

"""Source-task metrics with support counts, undefined values and tie handling."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from .contracts import require


def correlation(x, y, *, rank=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    require(x.shape == y.shape and x.ndim == 1 and np.isfinite(x).all() and np.isfinite(y).all(),
            "CORRELATION_FINITE_PAIRED_VECTORS")
    if len(x) < 2:
        return None
    if rank:
        x, y = rankdata(x), rankdata(y)
    a, b = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator <= 1e-15 else float(np.clip(a @ b / denominator, -1, 1))


def regression_metrics(prediction, target, observed=None):
    prediction, target = np.asarray(prediction, float), np.asarray(target, float)
    mask = np.isfinite(target) if observed is None else np.asarray(observed, bool)
    require(prediction.ndim == 2 and prediction.shape == target.shape == mask.shape
            and np.isfinite(prediction).all() and np.isfinite(target[mask]).all(), "REGRESSION_METRIC_SHAPES")
    per = []
    for column in range(prediction.shape[1]):
        selected = mask[:, column]
        p, y = prediction[selected, column], target[selected, column]
        absolute = np.abs(p - y)
        huber = np.where(absolute <= 1., .5 * absolute**2, absolute - .5)
        per.append(dict(column=column, observed_count=int(selected.sum()),
            Pearson=correlation(p, y), Spearman=correlation(p, y, rank=True),
            MAE=float(np.abs(p - y).mean()) if len(p) else None,
            MSE=float(np.square(p - y).mean()) if len(p) else None,
            Huber_delta1=float(huber.mean()) if len(p) else None))
    def mean_defined(name):
        values = [x[name] for x in per if x[name] is not None]
        return float(np.mean(values)) if values else None
    pcc = [x["Pearson"] for x in per if x["Pearson"] is not None]
    return dict(per_output=per, Mean_Pearson=mean_defined("Pearson"),
        Median_Pearson=float(np.median(pcc)) if pcc else None,
        Mean_Spearman=mean_defined("Spearman"), Mean_MAE=mean_defined("MAE"), Mean_MSE=mean_defined("MSE"),
        Mean_Huber_delta1=mean_defined("Huber_delta1"),
        defined_Pearson_count=len(pcc), output_count=len(per),
        undefined_correlation_is_not_zero=True, independent_patients_claimed=False)


def image_prediction_metrics(prediction_z, target_native, observed, target_mean, target_sd):
    mean, sd = np.asarray(target_mean, float), np.asarray(target_sd, float)
    prediction_z, target_native = np.asarray(prediction_z, float), np.asarray(target_native, float)
    require(prediction_z.shape == target_native.shape and target_native.shape[1] == 4
            and mean.shape == sd.shape == (4,) and (sd > 0).all(), "IMAGE_PROGRAM_EVALUATION_SCALE")
    native = regression_metrics(prediction_z * sd + mean, target_native, observed)
    standardized = regression_metrics(prediction_z, (target_native - mean) / sd, observed)
    return dict(task="IMAGE_ONLY_ST_PREDICTION", native=native, train_Z=standardized,
        primary_donor_four_program_Pearson=native["Mean_Pearson"] if native["defined_Pearson_count"] == 4 else None,
        program_count=4, defined_program_count=native["defined_Pearson_count"],
        scale="raw_signed_log1pCP10K_program_and_current_train_Z",
        primary_requires_all_four_defined=True, true_RNA_or_tissue_not_prediction_input=True)


def localization_metrics(prediction, target, observed, train_threshold):
    prediction, target, observed = np.asarray(prediction, float), np.asarray(target, float), np.asarray(observed, bool)
    threshold = np.asarray(train_threshold, float)
    require(prediction.shape == target.shape == observed.shape and threshold.shape == (target.shape[1],)
            and np.isfinite(threshold).all(), "LOCALIZATION_TRAIN_THRESHOLD_SHAPES")
    base = regression_metrics(prediction, target, observed)
    for k, row in enumerate(base["per_output"]):
        selected = observed[:, k]
        labels = target[selected, k] >= threshold[k]
        positives = int(labels.sum())
        row.update(high_activity_count=positives,
            AUPRC=float(average_precision_score(labels, prediction[selected, k])) if positives and len(labels) > positives else None,
            prevalence=float(labels.mean()) if len(labels) else None,
            high_activity_threshold_from_train=float(threshold[k]))
    values = [r["AUPRC"] for r in base["per_output"] if r["AUPRC"] is not None]
    base.update(Mean_AUPRC=float(np.mean(values)) if values else None,
        primary_four_program_AUPRC=float(np.mean(values)) if len(values) == 4 else None,
        defined_AUPRC_count=len(values), primary_requires_all_four_defined=True,
        threshold_fit_scope="source_training_rows_only", task="PROGRAM_TO_REGION_LOCALIZATION",
        lesion_segmentation_truth_claimed=False)
    return base


def retrieval_metrics(positive_score, negative_score, valid_negative):
    pos, neg, mask = np.asarray(positive_score, float), np.asarray(negative_score, float), np.asarray(valid_negative, bool)
    require(pos.ndim == 1 and neg.ndim == 2 and mask.shape == neg.shape and len(pos) == len(neg)
            and np.isfinite(pos).all() and np.isfinite(neg[mask]).all(), "RETRIEVAL_SCORE_SHAPES")
    active = mask.any(1)
    if not active.any():
        return dict(anchors_count=0, Recall_at_1=None, Recall_at_5=None, median_rank=None, pair_AUC=None,
                    scope="positive_versus_registered_hard_negative_candidate_sets")
    p, n, m = pos[active, None], neg[active], mask[active]
    better = ((n > p) & m).sum(1)
    ties = ((n == p) & m).sum(1)
    count = m.sum(1)
    ranks = 1 + better + .5 * ties
    # Expected credit from uniform tie-breaking; no model-specific index favors the positive.
    recall = lambda k: np.clip((k - better) / (1 + ties), 0., 1.)
    auc = (((n < p) & m).sum(1) + .5 * ties) / count
    enough5 = count + 1 >= 5
    return dict(anchors_count=int(active.sum()), Recall_at_1=float(recall(1).mean()),
        Recall_at_5=float(recall(5)[enough5].mean()) if enough5.any() else None,
        Recall_at_5_eligible_count=int(enough5.sum()), median_rank=float(np.median(ranks)), pair_AUC=float(auc.mean()),
        negative_candidate_count_min=int(count.min()), negative_candidate_count_max=int(count.max()),
        tie_rule="expected_uniform_credit_average_rank", aggregation="equal_anchor_not_independent_patient",
        scope="positive_versus_registered_hard_negative_candidate_sets")


def stratified_within_slide(prediction, target, observed, section, tissue):
    p, y, mask = np.asarray(prediction), np.asarray(target), np.asarray(observed, bool)
    section, tissue = np.asarray(section), np.asarray(tissue)
    require(section.shape == tissue.shape == (len(y),), "STRATIFIED_EVALUATION_METADATA_SHAPES")
    groups = []
    for slide in np.unique(section):
        for label in np.unique(tissue[section == slide]):
            rows = (section == slide) & (tissue == label)
            groups.append(dict(section_code=int(slide), tissue_code=int(label), spots_count=int(rows.sum()),
                               metrics=regression_metrics(p[rows], y[rows], mask[rows])))
    return dict(task="within_slide_tissue_stratified_prediction", groups=groups,
        tissue_truth_used_for_evaluation_groups_only=True, labels_used_as_model_inputs=False)

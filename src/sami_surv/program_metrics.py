"""Independent, identity-free metrics for the locked SAMI Stage B audit.

Inputs ``y`` and ``pred`` are in ORIGINAL program-score units.  Training-only
``ymean`` and ``yscale`` are supplied by the caller; this module never fits them.
There is no model fitting, source-label inference, or survival evaluation here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


N_PROGRAMS = 4
MIN_GROUP_N = 20


def _mean_available(values):
    valid = [float(v) for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def _mean_complete(values):
    """Do not silently replace the fixed four programs or donors by a subset."""
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def _corr(truth, pred, coverage):
    coverage['candidate_groups'] += 1
    if len(truth) < MIN_GROUP_N:
        coverage['below_minimum_n_groups'] += 1
        return None
    if np.ptp(truth) == 0:
        coverage['constant_truth_groups'] += 1
        return None
    coverage['eligible_groups'] += 1
    if np.ptp(pred) == 0:
        coverage['constant_prediction_groups'] += 1
        return 0.0
    result = float(spearmanr(truth, pred).statistic)
    if not np.isfinite(result):
        raise ValueError('Unexpected nonfinite Spearman for nonconstant finite data')
    return result


def _coverage():
    return dict(candidate_groups=0, eligible_groups=0, below_minimum_n_groups=0,
                constant_truth_groups=0, constant_prediction_groups=0)


def _group_spearman(y, pred, donor, section, tissue, within_tissue):
    """Each donor: sections within tissue, tissues, then four programs."""
    coverage = [_coverage() for _ in range(N_PROGRAMS)]
    donor_results = []
    for d in np.unique(donor):
        di = np.flatnonzero(donor == d)
        program_results = []
        for k in range(N_PROGRAMS):
            if within_tissue:
                class_results = []
                for c in np.unique(tissue[di]):
                    ci = di[tissue[di] == c]
                    section_results = []
                    for s in np.unique(section[ci]):
                        idx = ci[section[ci] == s]
                        section_results.append(_corr(y[idx, k], pred[idx, k], coverage[k]))
                    class_results.append(_mean_available(section_results))
                program_results.append(_mean_available(class_results))
            else:
                section_results = []
                for s in np.unique(section[di]):
                    idx = di[section[di] == s]
                    section_results.append(_corr(y[idx, k], pred[idx, k], coverage[k]))
                program_results.append(_mean_available(section_results))
        donor_results.append(program_results)
    donor_aggregates = [_mean_complete(row) for row in donor_results]
    return dict(
        aggregate=_mean_complete(donor_aggregates),
        available_donor_description=_mean_available(donor_aggregates),
        complete_donors=sum(v is not None for v in donor_aggregates),
        per_program=[_mean_complete([row[k] for row in donor_results]) for k in range(N_PROGRAMS)],
        complete_donors_per_program=[sum(row[k] is not None for row in donor_results) for k in range(N_PROGRAMS)],
        coverage=coverage,
    )


def _natural_mean(values, donor, section):
    donor_values = []
    for d in np.unique(donor):
        di = np.flatnonzero(donor == d)
        donor_values.append(np.mean([
            np.mean(values[di[section[di] == s]], axis=0)
            for s in np.unique(section[di])
        ], axis=0))
    return np.mean(donor_values, axis=0)


def _qc(values, nuisance, donor, section):
    """Descriptive within-section rank association; no residualization or fitting."""
    repeated = np.repeat(np.log1p(nuisance)[:, None], N_PROGRAMS, axis=1)
    dummy_tissue = np.zeros(len(nuisance), dtype=np.int64)
    return _group_spearman(repeated, values, donor, section, dummy_tissue, False)


def evaluate_programs(y, pred, donor, section, tissue, ymean, yscale,
                      library_size=None, n_detected_genes=None):
    """Evaluate ORIGINAL-unit predictions without returning any donor/spot IDs.

    Huber delta is exactly 1 in training-standardized target units. Spearman is
    calculated within groups of at least 20 observations; undefined true-target
    groups are excluded with coverage. If a program has no eligible group for a
    donor, the strict four-program aggregate is None rather than a changed task.
    """
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != N_PROGRAMS or len(y) == 0 or pred.shape != y.shape:
        raise ValueError('y and pred must be nonempty matching [N,4] original-unit arrays')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Targets and predictions must be finite')
    donor, section, tissue = (np.asarray(a) for a in (donor, section, tissue))
    if any(a.ndim != 1 or len(a) != len(y) for a in (donor, section, tissue)):
        raise ValueError('Grouping arrays must be one-dimensional and aligned')
    ymean, yscale = np.asarray(ymean, dtype=np.float64), np.asarray(yscale, dtype=np.float64)
    if ymean.shape != (N_PROGRAMS,) or yscale.shape != (N_PROGRAMS,):
        raise ValueError('Training target mean and scale must have shape [4]')
    if not np.isfinite(ymean).all() or not np.isfinite(yscale).all() or np.any(yscale <= 0):
        raise ValueError('Training target mean and positive scale must be finite')
    # Mirror the training-standardized operation, not a refitted evaluation scale.
    residual = (pred-ymean)/yscale - (y-ymean)/yscale
    absolute = np.abs(residual)
    huber = np.where(absolute <= 1.0, 0.5*residual**2, absolute-0.5)
    mse = residual**2
    if not np.isfinite(huber).all() or not np.isfinite(mse).all():
        raise ValueError('Nonfinite standardized prediction errors')
    hp, mp = _natural_mean(huber, donor, section), _natural_mean(mse, donor, section)
    wt = _group_spearman(y, pred, donor, section, tissue, True)
    ws = _group_spearman(y, pred, donor, section, tissue, False)
    result = dict(
        huber=float(np.mean(hp)), mse=float(np.mean(mp)),
        within_tissue_spearman=wt['aggregate'], within_section_spearman=ws['aggregate'],
        n_spots=int(len(y)), n_donors=int(len(np.unique(donor))),
        n_sections=int(sum(len(np.unique(section[donor == d])) for d in np.unique(donor))),
        minimum_group_n=MIN_GROUP_N, prediction_units='original_program_score',
        incomplete_primary_policy='null_if_any_donor_lacks_any_of_four_program_correlations',
        within_tissue_complete_donors=wt['complete_donors'],
        within_section_complete_donors=ws['complete_donors'],
        available_donor_description=dict(within_tissue_spearman=wt['available_donor_description'],
                                         within_section_spearman=ws['available_donor_description']),
        per_program=[],
    )
    for k in range(N_PROGRAMS):
        result['per_program'].append(dict(
            program_index=k, huber=float(hp[k]), mse=float(mp[k]),
            within_tissue_spearman=wt['per_program'][k],
            within_section_spearman=ws['per_program'][k],
            eligible_groups=wt['coverage'][k]['eligible_groups'],
            within_tissue_coverage=wt['coverage'][k],
            within_section_coverage=ws['coverage'][k],
            within_tissue_complete_donors=wt['complete_donors_per_program'][k],
            within_section_complete_donors=ws['complete_donors_per_program'][k],
        ))
    qc = {}
    for name, nuisance in [('library_size', library_size), ('n_detected_genes', n_detected_genes)]:
        if nuisance is None:
            continue
        nuisance = np.asarray(nuisance, dtype=np.float64)
        if nuisance.shape != (len(y),) or not np.isfinite(nuisance).all() or np.any(nuisance < 0):
            raise ValueError('QC nuisance must be finite nonnegative and row-aligned')
        qc[name] = dict(
            transform='log1p', metric='within_section_Spearman_descriptive_only',
            truth=_qc(y, nuisance, donor, section), prediction=_qc(pred, nuisance, donor, section),
        )
    result['qc_associations'] = qc
    return result

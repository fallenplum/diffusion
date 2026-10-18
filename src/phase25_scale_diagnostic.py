"""Pure numerical summaries for frozen risk-scale diagnostics.

No file, endpoint, predictor or fitting API is imported or called. Population
SD uses ddof=0. Undefined correlations and ratios return None with a reason.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _finite_array(value, name):
    array = np.asarray(value)
    if (not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array)
            or array.size == 0):
        raise ValueError(name + ' must be a nonempty real numeric array')
    result = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(name + ' must contain finite float64 values')
    return result


def _finite_scalar(value, name, *, nonnegative=False):
    if (not np.isscalar(value) or isinstance(value, (bool, np.bool_))
            or not np.issubdtype(np.asarray(value).dtype, np.number) or np.iscomplexobj(value)):
        raise ValueError(name + ' must be a finite real scalar')
    result = float(value)
    if not np.isfinite(result) or (nonnegative and result < 0):
        raise ValueError(name + ' must be finite' + (' and nonnegative' if nonnegative else ''))
    return result


def _pair(first, second, names):
    first, second = _finite_array(first, names[0]), _finite_array(second, names[1])
    if first.shape != second.shape:
        raise ValueError('paired risk arrays must have identical shapes')
    return first, second


def _checked_calculation(function, name):
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            result = function()
    except (FloatingPointError, OverflowError) as error:
        raise ValueError(name + ' is not representable as finite float64') from error
    if not np.isfinite(result).all():
        raise ValueError(name + ' is not representable as finite float64')
    return result


def scaled_risk(base, trained_risk, alpha):
    """Return base + alpha*(trained_risk-base), preserving alpha=0/1 exactly.

For alpha in [0,1], the convex-combination expression avoids an unnecessary
overflow in the risk difference. No risk centering or standardization occurs.
"""
    base, trained = _pair(base, trained_risk, ('base', 'trained_risk'))
    alpha = _finite_scalar(alpha, 'alpha')
    if alpha == 0.:
        return base.copy()
    if alpha == 1.:
        return trained.copy()
    if 0. <= alpha <= 1.:
        return _checked_calculation(lambda: (1. - alpha) * base + alpha * trained, 'scaled risk')
    return _checked_calculation(lambda: base + alpha * (trained - base), 'scaled risk')


def moments(values):
    """Anonymous finite-array moments; quantiles use linear interpolation."""
    values = _finite_array(values, 'values').reshape(-1)
    magnitude = float(np.max(np.abs(values)))
    if magnitude == 0.:
        return {key: len(values) if key == 'count' else 0.
                for key in ('count', 'mean', 'SD', 'RMS', 'median', 'IQR', 'q05', 'q95', 'abs_q95')}
    normalized = values / magnitude
    q05, q25, median, q75, q95 = np.quantile(normalized, [.05, .25, .5, .75, .95], method='linear')
    answer = {'count': len(values),
              'mean': float(normalized.mean() * magnitude),
              'SD': float(normalized.std(ddof=0) * magnitude),
              'RMS': float(np.sqrt(np.mean(normalized * normalized)) * magnitude),
              'median': float(median * magnitude),
              'IQR': float(_checked_calculation(lambda: (q75 - q25) * magnitude, 'IQR')),
              'q05': float(q05 * magnitude), 'q95': float(q95 * magnitude),
              'abs_q95': float(np.quantile(np.abs(normalized), .95, method='linear') * magnitude)}
    if not all(np.isfinite(value) for value in answer.values()):
        raise ValueError('moment is not representable as finite float64')
    return answer


def _pearson(first, second):
    first, second = np.asarray(first).reshape(-1), np.asarray(second).reshape(-1)
    first_constant, second_constant = bool(np.all(first == first[0])), bool(np.all(second == second[0]))
    if first_constant or second_constant:
        which = 'both' if first_constant and second_constant else ('first' if first_constant else 'second')
        return None, 'UNDEFINED_ZERO_SD_' + which.upper()
    # Independent rescaling makes covariance arithmetic safe without changing
    # Pearson's coefficient or reporting a falsely defined zero correlation.
    x = first / np.max(np.abs(first)); y = second / np.max(np.abs(second))
    x, y = x - x.mean(), y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.:
        return None, 'UNDEFINED_NUMERICAL_ZERO_SD'
    return float(np.clip(np.dot(x, y) / denominator, -1., 1.)), 'DEFINED'


def _ratio(numerator, denominator, name):
    if denominator == 0.:
        return None, 'UNDEFINED_ZERO_' + name
    value = float(_checked_calculation(lambda: numerator / denominator, name + ' ratio'))
    return value, 'DEFINED'


def paired_distributions(cf, full):
    """Compare same-patient risks. Gap is cf-full; no labels are consumed."""
    cf, full = _pair(cf, full, ('cf', 'full'))
    cf_summary, full_summary = moments(cf), moments(full)
    gap = _checked_calculation(lambda: cf - full, 'cf-full gap')
    gap_summary = moments(gap)
    # Population SD of the gap is exactly its centered RMS. Avoid explicitly
    # subtracting a very large gap mean and losing numerical precision.
    centered_rms = gap_summary['SD']
    pearson, pearson_status = _pearson(cf, full)
    spearman, spearman_status = _pearson(rankdata(cf.reshape(-1), method='average'),
                                       rankdata(full.reshape(-1), method='average'))
    ratio, ratio_status = _ratio(cf_summary['SD'], full_summary['SD'], 'FULL_SD')
    centered_ratio, centered_status = _ratio(centered_rms, full_summary['SD'], 'FULL_SD')
    return {'cf': cf_summary, 'full': full_summary, 'gap_direction': 'cf_minus_full',
            'Pearson': pearson, 'Spearman': spearman,
            'correlation_status': {'Pearson': pearson_status, 'Spearman': spearman_status},
            'zero_SD': {'cf': cf_summary['SD'] == 0., 'full': full_summary['SD'] == 0.},
            'SD_ratio_cf_over_full': ratio,
            'gap_raw_RMS': gap_summary['RMS'], 'gap_centered_RMS': centered_rms,
            'gap_centered_RMS_over_full_SD': centered_ratio,
            'ratio_status': {'SD_ratio_cf_over_full': ratio_status,
                             'gap_centered_RMS_over_full_SD': centered_status}}


def residual_summary(base, trained_risk, reference_sd):
    """Summarize a saved correction trained_risk-base without refitting."""
    base, trained = _pair(base, trained_risk, ('base', 'trained_risk'))
    reference_sd = _finite_scalar(reference_sd, 'reference_sd', nonnegative=True)
    delta = _checked_calculation(lambda: trained - base, 'risk correction')
    base_summary, delta_summary = moments(base), moments(delta)
    sd_ratio, sd_status = _ratio(delta_summary['SD'], base_summary['SD'], 'BASE_SD')
    rms_ratio, rms_status = _ratio(delta_summary['RMS'], reference_sd, 'REFERENCE_SD')
    correlation, status = _pearson(delta, base)
    return {'delta': delta_summary, 'reference_SD': reference_sd,
            'SD_delta_over_SD_base': sd_ratio, 'RMS_delta_over_reference_SD': rms_ratio,
            'Pearson_delta_base': correlation, 'Pearson_delta_base_status': status,
            'zero_SD': {'base': base_summary['SD'] == 0., 'delta': delta_summary['SD'] == 0.,
                        'reference': reference_sd == 0.},
            'ratio_status': {'SD_delta_over_SD_base': sd_status, 'RMS_delta_over_reference_SD': rms_status}}

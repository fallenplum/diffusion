"""无第三方评分器依赖的Efron风险集、患者对与IPCW核验实现。"""
from __future__ import annotations
import numpy as np


def efron_weights(time, event):
    t, e = np.asarray(time, float), np.asarray(event, int)
    if t.ndim != 1 or e.shape != t.shape or not np.isfinite(t).all() or not np.isin(e, [0, 1]).all() or not e.sum():
        raise ValueError('生存结局无效，或者训练集没有观察到事件')
    rows = []
    for u in np.unique(t[e == 1]):
        deaths = (t == u) & (e == 1)
        d = int(deaths.sum())
        for j in range(d):
            rows.append((t >= u).astype(float) - j / d * deaths)
    return np.stack(rows)


def pairs(t, e):
    t, e = np.asarray(t), np.asarray(e)
    left, right = [], []
    for i in range(len(t)):
        if e[i] != 1:
            continue
        js = np.flatnonzero((t > t[i]) | ((t == t[i]) & (e == 0)))
        left.extend([i] * len(js)); right.extend(js.tolist())
    return np.asarray(left, int), np.asarray(right, int)


def censor_probability(train_time, train_event, query):
    t, e = np.asarray(train_time, float), np.asarray(train_event, int)
    times = np.unique(t)
    probability, g = 1.0, []
    for u in times:
        cens = int(((t == u) & (e == 0)).sum())
        at_risk = int((t >= u).sum() - ((t == u) & (e == 1)).sum())
        if cens:
            if at_risk <= 0:
                raise ValueError('删失分布支持集不足')
            probability *= 1 - cens / at_risk
        g.append(probability)
    pos = np.searchsorted(times, np.asarray(query), side='right') - 1
    return np.where(pos < 0, 1.0, np.asarray(g)[np.maximum(pos, 0)])


def evaluate(train_t, train_e, t, e, risk, tau=60.):
    """输出折内Harrell/Uno60。Uno权重由训练集删失过程估计。"""
    r, t, e = np.asarray(risk), np.asarray(t), np.asarray(e)
    if not np.isfinite(r).all():
        raise ValueError('预测风险包含非有限值')
    i, j = pairs(t, e)
    if not len(i):
        return {'Harrell': None, 'Uno60': None, 'comparable_pairs': 0}
    credit = np.where(np.abs(r[i] - r[j]) <= 1e-8, .5, (r[i] > r[j]).astype(float))
    eligible = (e == 1) & (t < tau)
    weight = np.zeros(len(t), float)
    g = censor_probability(train_t, train_e, t[eligible])
    if np.any(g <= 0):
        raise ValueError('Uno评估的删失支持集不足')
    weight[eligible] = 1 / (g ** 2)
    den = float(weight[i].sum())
    return {'Harrell': float(credit.mean()), 'Uno60': float(np.dot(weight[i], credit) / den) if den > 0 else None,
            'comparable_pairs': int(len(i))}

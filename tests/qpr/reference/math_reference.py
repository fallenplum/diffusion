#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QPR-Fusion 独立、无需患者数据的 NumPy 数学资格参考。

这不是模型实现、不训练神经网络，不能被报告成 QPR 科学效果。
运行: python 04_QPR数学参考测试.py
"""
from __future__ import annotations
import math
import numpy as np


def logsumexp(z):
    z = np.asarray(z, np.float64)
    m = float(np.max(z))
    return m + math.log(float(np.exp(z - m).sum()))


def full_answer(q, k, v, w, tau=1.0):
    """返回真实关系边上的软注意力向量答案及 logZ。"""
    q, k, v, w = (np.asarray(a, np.float64) for a in (q, k, v, w))
    assert np.isclose(w.sum(), 1) and (w > 0).all()
    logits = tau * (k @ q) + np.log(w)
    lse = logsumexp(logits)
    att = np.exp(logits - lse)
    return att @ v, lse


def compress(k, v, w, S):
    """给定行随机分配 S [E,M]，计算质量π与均值KV。"""
    k, v, w, S = (np.asarray(a, np.float64) for a in (k, v, w, S))
    assert k.ndim == v.ndim == S.ndim == 2 and k.shape[0] == v.shape[0] == len(w) == len(S)
    assert np.all(S >= -1e-14) and np.allclose(S.sum(axis=1), 1, atol=1e-12)
    pi = w @ S
    kb = np.zeros((len(pi), k.shape[-1]), np.float64)
    vb = np.zeros((len(pi), v.shape[-1]), np.float64)
    active = pi > 1e-14
    kb[active] = ((w[:, None] * S).T @ k)[active] / pi[active, None]
    vb[active] = ((w[:, None] * S).T @ v)[active] / pi[active, None]
    return pi, kb, vb


def compressed_answer(q, pi, kb, vb, tau=1.0, use_mass=True):
    q, pi, kb, vb = (np.asarray(a, np.float64) for a in (q, pi, kb, vb))
    active = pi > 1e-14
    assert active.any()
    logits = tau * (kb[active] @ q)
    if use_mass:
        logits += np.log(pi[active])
    else:
        # no-mass 对照沿每个非空中心等权softmax，归一化常数只是任意中心归一形式
        logits -= math.log(int(active.sum()))
    lse = logsumexp(logits)
    a = np.exp(logits - lse)
    return a @ vb[active], lse


def distortion(k, v, w, S, kb, vb):
    dk = np.sum(w[:, None] * S * np.linalg.norm(k[:, None, :] - kb[None, :, :], axis=-1))
    dv = np.sum(w[:, None] * S * np.linalg.norm(v[:, None, :] - vb[None, :, :], axis=-1))
    return dk, dv


def safe_norm_rows(x, radius=1.0):
    x = np.asarray(x, np.float64)
    return x / np.maximum(radius, np.linalg.norm(x, axis=1, keepdims=True)) * radius


def efron_event_mean_nll(time, event, risk):
    """无外部依赖的Efron ties检查，event-mean Cox negative partial loglikelihood。"""
    t, e, r = np.asarray(time, float), np.asarray(event, int), np.asarray(risk, float)
    assert t.ndim == e.ndim == r.ndim == 1 and len(t) == len(e) == len(r)
    assert np.all(np.isfinite(t)) and np.all(np.isfinite(r)) and np.all(np.isin(e, [0, 1]))
    n_ev = int(e.sum())
    assert n_ev > 0
    total = 0.0
    for ti in np.unique(t[e == 1]):
        deaths = (t == ti) & (e == 1)
        d = int(deaths.sum())
        at_risk = t >= ti
        ref = float(np.max(r[at_risk]))
        sum_r = float(np.exp(r[at_risk] - ref).sum())
        sum_d = float(np.exp(r[deaths] - ref).sum())
        total -= float(r[deaths].sum())
        for l in range(d):
            den = sum_r - (l / d) * sum_d
            assert den > 0
            total += ref + math.log(den)
    return total / n_ev


def test_chain_spatial_counterexample():
    a, b = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    assert sorted(a) == sorted(b)
    e_a = int(np.square(np.diff(a)).sum())
    e_b = int(np.square(np.diff(b)).sum())
    assert (e_a, e_b) == (1, 3)
    return (e_a, e_b)


def test_mass_exactness():
    # cluster1占90%原始边质量、cluster2占10%；如果忽略质量则答案有明显偏差。
    k = np.array([[.2, 0], [.2, 0], [-.3, .1], [-.3, .1]])
    v = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    w = np.array([.70, .20, .07, .03])
    S = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    pi, kb, vb = compress(k, v, w, S)
    for q in (np.array([0., 0.]), np.array([.4, .5]), np.array([-.6, .3])):
        x, lz = full_answer(q, k, v, w)
        y, ly = compressed_answer(q, pi, kb, vb)
        assert np.allclose(x, y, atol=1e-13), (x, y)
        assert abs(lz - ly) < 1e-13
    q = np.zeros(2)
    x, _ = full_answer(q, k, v, w)
    wrong, _ = compressed_answer(q, pi, kb, vb, use_mass=False)
    assert np.linalg.norm(x - wrong) > .5
    return float(np.linalg.norm(x - wrong))


def test_random_bounds():
    worst_violation = float('-inf')
    max_error = 0.0
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        E, M, dimk, dimv = 35, 7, 8, 10
        k = safe_norm_rows(rng.normal(size=(E, dimk)))
        v = safe_norm_rows(rng.normal(size=(E, dimv)))
        q = rng.normal(size=dimk)
        q = q / max(1.0, np.linalg.norm(q))
        w = rng.dirichlet(np.ones(E))
        S = rng.dirichlet(np.ones(M), size=E)
        pi, kb, vb = compress(k, v, w, S)
        out, _ = full_answer(q, k, v, w)
        approx, _ = compressed_answer(q, pi, kb, vb)
        dk, dv = distortion(k, v, w, S, kb, vb)
        Q, K, V, tau = 1.0, 1.0, 1.0, 1.0
        bound = math.exp(2 * tau * Q * K) * (dv + 2 * V * tau * Q * dk)
        error = float(np.linalg.norm(out - approx))
        worst_violation = max(worst_violation, error - bound)
        max_error = max(max_error, error)
        assert error <= bound + 1e-11, (seed, error, bound)
    return max_error, worst_violation


def test_efron_loss_bound():
    rng = np.random.default_rng(2342)
    t = np.array([1., 1., 2., 2., 3., 4., 5., 5., 6., 7., 9., 10.])
    e = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0])
    r0 = rng.normal(size=len(t))
    perturb = rng.uniform(-.01, .01, size=len(t))
    a, b = efron_event_mean_nll(t, e, r0), efron_event_mean_nll(t, e, r0 + perturb)
    eps = float(np.max(np.abs(perturb)))
    assert abs(a - b) <= 2 * eps + 1e-12, (a, b, eps)
    return a, b, abs(a-b), 2*eps


def test_adjacency_permutation_invariance():
    xy = np.array([(i * 256, j * 256) for i in range(4) for j in range(4)], np.int64)
    feat = np.arange(16, dtype=np.int64)
    def paired_edges(xy, feat):
        lookup = {tuple(x): int(f) for x, f in zip(xy, feat)}
        out = []
        for (x, y), f in lookup.items():
            for dx, dy in ((256, 0), (0, 256), (256, 256), (256, -256)):
                j = lookup.get((x + dx, y + dy))
                if j is not None:
                    out.append(tuple(sorted((f, j))))
        return sorted(out)
    base = paired_edges(xy, feat)
    perm = np.random.default_rng(101).permutation(len(xy))
    assert base == paired_edges(xy[perm], feat[perm])
    assert len(base) > 0
    return len(base)


def main():
    result = {
        '1_chain_values': test_chain_spatial_counterexample(),
        '2_wrong_mass_error': test_mass_exactness(),
        '3_random_attention_bound': test_random_bounds(),
        '4_efron_loss_bound': test_efron_loss_bound(),
        '5_coordinate_pairing_permutation_invariance_edges': test_adjacency_permutation_invariance(),
    }
    print('PASS: 5 independent numerical qualification groups (synthetic only; no TCGA data)')
    for k, v in result.items():
        print(k, v)
    return result


if __name__ == '__main__':
    main()

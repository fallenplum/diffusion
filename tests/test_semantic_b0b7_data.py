import numpy as np
import pytest

from src.semantic_b0b7.data import CLINICAL_FIELDS, ObservedScale, ProgramTransform, background_crossfit, clinical_design, pretext_split


def test_observed_scale_ignores_evaluation_and_preserves_missing():
    x = np.array([[1, np.nan], [3, np.nan], [9, 100]], float)
    state = ObservedScale.fit(x, [0, 1])
    z, mask = state.transform(x)
    assert np.array_equal(state.mean, [2, 0])
    assert np.all(z[:, 1] == 0) and not mask[:, 1].any()
    x[2] = [1e10, -2e10]
    other = ObservedScale.fit(x, [0, 1])
    assert np.array_equal(other.mean, state.mean)
    assert np.array_equal(other.sd, state.sd)


def test_gain_and_loss_split_before_observed_scaling():
    r = np.arange(24, dtype=float).reshape(4, 6)
    c = np.array([[1, -1, 1, -1, np.nan, 0], [0, -2, 2, 0, np.nan, 1],
                  [10, -10, 0, 1, 1, 1], [0, 1, 0, 1, 1, 1]], float)
    membership = np.ones((50, 6), bool)
    fit = ProgramTransform.fit(r, c, [0, 1], membership, membership, min_genes=2)
    assert np.array_equal(fit.gain_scale.mean, [0.5, 0, 1.5, 0, 0, 0.5])
    assert np.array_equal(fit.loss_scale.mean, [0, 1.5, 0, 0.5, 0, 0])
    out = fit.transform(r, c)
    assert out["values"].shape == (4, 50, 3)
    assert out["measured"].all()
    assert np.allclose(out["coverage"][:, :, 1], 5 / 6)


def test_program_all_missing_is_not_observed_zero():
    r = np.full((4, 6), np.nan)
    c = np.zeros_like(r)
    m = np.ones((50, 6), bool)
    fit = ProgramTransform.fit(r, c, [0, 1], m, m)
    out = fit.transform(r, c)
    assert not out["measured"][:, :, 0].any()
    assert out["measured"][:, :, 1:].all()
    assert np.isfinite(out["values"]).all()


def test_pretext_split_never_accepts_or_requires_endpoint():
    folds = np.repeat(np.arange(1, 6), 12)
    train, valid, test = pretext_split(folds)
    assert set(train) | set(valid) == set(np.flatnonzero(folds != 1))
    assert not set(train) & set(valid) and not set(train) & set(test)
    with pytest.raises(TypeError):
        pretext_split(folds, event=np.ones(60))


def test_background_uses_single_encoder_coordinate_and_excludes_evaluation():
    rng = np.random.default_rng(2)
    c, z = rng.normal(size=(24, 3)), rng.normal(size=(24, 2, 4))
    tr, ev = np.arange(18), np.arange(18, 24)
    first = background_crossfit(c, z, tr, ev)
    z2 = z.copy(); z2[ev] += 1e8
    other = background_crossfit(c, z2, tr, ev)
    assert np.array_equal(first["mean"], other["mean"])
    assert all(x["fit_n"] == 12 and x["held_n"] == 6 for x in first["audit"]["folds"])
    assert np.isfinite(first["mean"]).all()
    with pytest.raises(ValueError, match="overlaps"):
        background_crossfit(c, z, tr, [1])


def test_background_raw_clinical_preprocessing_is_refit_in_every_subtrain():
    raw = {k: np.array(["A"] * 24, dtype=object) for k in CLINICAL_FIELDS}
    raw["AGE"] = np.array([str(40 + i) for i in range(24)], dtype=object)
    raw["AGE"][2] = "[Not Available]"
    raw["SEX"][18:] = "NEVER_SEEN_IN_TRAIN"
    c, st = clinical_design(raw, np.arange(18))
    assert "NEVER_SEEN_IN_TRAIN" not in st["levels"]["SEX"]
    z = np.random.default_rng(9).normal(size=(24, 2, 4))
    result = background_crossfit(c, z, np.arange(18), np.arange(18, 24), clinical_raw=raw)
    assert all(x["clinical_preprocessing_fit_hash"] == x["fit_hash"] for x in result["audit"]["folds"])
    assert result["audit"]["clinical_preprocessing"] == "raw_six_field_refit_in_each_background_fit"

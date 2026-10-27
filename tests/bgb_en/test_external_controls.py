import numpy as np
from scripts.bgb_en_external_controls import frozen_true_scaler, target_remapping
from src.bgb_en import external


def data():
    donors = np.repeat(np.array(list('abcdef')), 12)
    sections = np.concatenate([np.repeat([name+'1', name+'2'], 6) for name in 'abcdef'])
    tissue = np.tile(np.repeat(np.arange(3), 2), 12)
    return dict(donor=donors, section=sections, tissue_y=tissue)


def test_wrong_donor_remapping_preserves_tissue_and_really_changes_donor():
    source = data()
    index, audit = target_remapping(source, 'WRONG_DONOR', 9033021)
    assert np.all(source['donor'][index] != source['donor'])
    assert np.array_equal(source['tissue_y'][index], source['tissue_y'])
    assert audit['cross_donor_rows'] == len(index) and audit['self_target_rows'] == 0
    assert np.array_equal(index, target_remapping(source, 'WRONG_DONOR', 9033021)[0])


def test_spatial_remapping_bijection_within_section_and_tissue():
    source = data()
    index, audit = target_remapping(source, 'SPATIAL_SHUFFLE', 9033022)
    assert np.array_equal(np.sort(index), np.arange(len(index)))
    assert np.array_equal(source['donor'][index], source['donor'])
    assert np.array_equal(source['section'][index], source['section'])
    assert np.array_equal(source['tissue_y'][index], source['tissue_y'])
    assert audit['self_target_rows'] == 0


def test_original_true_target_scaler_is_frozen_and_dependency_restored():
    rng = np.random.default_rng(23)
    target = rng.normal(size=(30, 4))
    weight = np.full(30, 1/30)
    scaler = external.fit_target_scaler(target, weight)
    original_function = external.fit_target_scaler
    with frozen_true_scaler(scaler):
        rematched = external.fit_target_scaler(target[:1].repeat(30, axis=0), weight)
        assert np.array_equal(rematched['mean'], scaler['mean'])
        assert np.array_equal(rematched['scale'], scaler['scale'])
    assert external.fit_target_scaler is original_function

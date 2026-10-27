import importlib.util
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[2]/'scripts/bgb_en_path_sensitivity.py'
SPEC=importlib.util.spec_from_file_location('bgb_path_sensitivity_tested',PATH)
sensitivity=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(sensitivity)


def test_disk_estimate_counts_final_arrays_without_raw_h5_copy():
    rows=[{'patient':0,'n':2000},{'patient':1,'n':5000}]
    estimate=sensitivity.estimate_cache_disk(rows,1024,n_patients=2)
    assert estimate['array_bytes']['features']==2*1024*768*4
    assert estimate['array_bytes']['region_ids']==2*1024*8
    assert estimate['selected_patches']==2048
    assert estimate['peak_additional_disk_bytes']>sum(estimate['array_bytes'].values())
    assert estimate['full_raw_H5_copied'] is False
    assert estimate['second_complete_cache_disk_copy'] is False


def test_reserve_gate_fails_before_cache_allocation():
    with pytest.raises(ValueError,match='reserved free disk'):
        sensitivity.check_reserve(5000,2001,3000)
    assert sensitivity.check_reserve(5000,2000,3000)['guaranteed_remaining_free_bytes']==3000


def test_uni2_128_reuses_existing_source_and_1024_is_not_implicitly_added(tmp_path):
    existing=tmp_path/'existing_uni2_128'
    assert sensitivity.cache_path({'patch_uni2':str(existing)},tmp_path,'UNI2',128)==existing
    with pytest.raises(ValueError,match='prelocked only'):
        sensitivity.cache_path({'patch_uni2':str(existing)},tmp_path,'UNI2',1024)
    assert sensitivity.cache_path({},tmp_path,'CONCH',1024).name=='conch_regions1024_seed270003'

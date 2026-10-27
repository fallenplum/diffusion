"""CPU scheduling guards, no preparation or model fitting."""
import json

import pytest

from scripts.bgb_en_checkpoint_audit import AuditError, sha256
from scripts.bgb_en_precompute_nested_prepare import prep_paths, remaining_storage_bytes, validate_existing


def test_storage_forecast_counts_only_unwritten_bytes_of_locked_twelve(tmp_path):
    assert remaining_storage_bytes(tmp_path, 1000) == 12000
    path, receipt = prep_paths(tmp_path, 2, 0)
    path.parent.mkdir(); path.write_bytes(b'x' * 1000); receipt.write_text('{}')
    temp = prep_paths(tmp_path, 2, 1)[0].with_suffix('.pt.tmp'); temp.write_bytes(b'x' * 400)
    assert remaining_storage_bytes(tmp_path, 1000) == 10600


def test_existing_bundle_requires_both_receipt_and_verified_lineage_and_hash(tmp_path):
    path, receipt = prep_paths(tmp_path, 2, 0); path.parent.mkdir()
    assert validate_existing(path, receipt, {'source': 'correct'}) is None
    path.write_bytes(b'pretend saved preparation')
    with pytest.raises(AuditError, match='INTERRUPTED'):
        validate_existing(path, receipt, {'source': 'correct'})
    row = {'status': 'COMPLETE', 'lineage': {'source': 'correct'}, 'sha256': sha256(path),
           'upstream_optimizer_fits': 13, 'upstream_preprocessing_fits': 4}
    receipt.write_text(json.dumps(row))
    assert validate_existing(path, receipt, row['lineage']) == row
    with pytest.raises(AuditError, match='LINEAGE'):
        validate_existing(path, receipt, {'source': 'wrong'})
    path.write_bytes(b'changed')
    with pytest.raises(AuditError, match='HASH'):
        validate_existing(path, receipt, row['lineage'])

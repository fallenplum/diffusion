"""Actual model dispatch plus frozen stage-gamma path; no model fitting."""
import copy
import json

import pytest

from scripts.bgb_en_checkpoint_audit import AuditError
from scripts.bgb_en_matched_support_plan import ROOT, verify_constructor_and_gamma


def test_both_nulls_keep_base_architecture_and_gamma_without_double_shuffle():
    config = json.loads((ROOT / 'configs/bgb_en/prototype_v1.json').read_text())
    result = verify_constructor_and_gamma(config)
    assert [(r['actual_base_arm'], r['gamma']) for r in result] == [('O4', .3), ('F5', .1)]
    assert all(r['same_membership_object_forwarded'] for r in result)
    assert not any(r['original_O9_or_SHUFFLED_constructor_invoked'] for r in result)


def test_changed_gamma_is_rejected():
    config = json.loads((ROOT / 'configs/bgb_en/prototype_v1.json').read_text())
    config = copy.deepcopy(config); config['R2']['gamma'] = .1
    with pytest.raises(AuditError, match='GAMMA'):
        verify_constructor_and_gamma(config)

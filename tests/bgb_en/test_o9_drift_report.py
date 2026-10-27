import copy
import json
from pathlib import Path
import pytest
from scripts.bgb_en_report_o9_drift import aggregate_documents, markdown


def artifacts():
    base = Path(__file__).resolve().parents[2]/'reports/bgb_en'
    return (json.loads((base/'CHECKPOINT_AUDIT.json').read_text()),
        [json.loads((base/name).read_text()) for name in ('O9_FLOAT32_DRIFT_DEFAULT.json', 'O9_FLOAT32_DRIFT_DETERMINISTIC.json')])


def test_actual_thirty_replays_are_append_only_and_do_not_promote_original_failure():
    original, documents = artifacts()
    result = aggregate_documents(original, documents)
    assert result['status'] == 'FLOAT32_DRIFT_WITH_UNCHANGED_RANK_DIAGNOSIS'
    assert result['fixed_complete_validation_replays'] == 30
    assert result['original_audit']['passed_units'] == 184 and result['original_audit']['failed_units'] == 2
    assert result['original_audit']['promoted_to_PASS'] is False
    assert len(result['cells']) == 6 and all(len(cell['all_five_repetitions']) == 5 for cell in result['cells'])


def test_invalid_gamma_or_changed_rank_cannot_be_described_as_unchanged_drift():
    original, documents = artifacts()
    wrong = copy.deepcopy(documents); wrong[0]['cells'][0]['gamma'] = .1
    with pytest.raises(ValueError, match='Invalid diagnostic gamma'):
        aggregate_documents(original, wrong)
    wrong = copy.deepcopy(documents); wrong[0]['cells'][0]['repeats'][0]['delta_Harrell_vs_saved'] = .001
    changed = aggregate_documents(original, wrong)
    assert changed['status'] == 'FLOAT32_DRIFT_WITH_RANK_CHANGES_DIAGNOSIS'
    with pytest.raises(ValueError, match='ranking changed'):
        markdown(changed)

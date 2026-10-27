"""Per-job I/O destinations remove shared temporary files without scientific edits."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path

import pytest

from scripts.bgb_en_checkpoint_audit import AuditError
from scripts.bgb_en_matched_support_io_plan import derive_plan
from scripts.bgb_en_run import write


def plan():
    return {'status': 'LOCKED_BEFORE_ADDITIONAL_FITS', 'outer_test_evaluated': False,
        'tasks': [{'label': f'job_{i}', 'script': 'scripts/bgb_en_matched_support.py',
                   'args': ['--stage', 'train', '--seed', '31', '--private', 'private']} for i in range(12)]}


def test_only_output_is_appended_and_original_plan_is_not_mutated(tmp_path):
    original = plan(); before = copy.deepcopy(original)
    runtime, rows = derive_plan(original, tmp_path, 'abc')
    assert original == before and len(rows) == 12
    outputs = set()
    for old, new in zip(original['tasks'], runtime['tasks']):
        assert new['script'] == old['script'] and new['label'] == old['label']
        assert new['args'][:-2] == old['args'] and new['args'][-2] == '--output'
        destination = Path(new['args'][-1]); outputs.add(destination)
        assert destination.parent == tmp_path.resolve() / 'unit_public_views'
    assert len(outputs) == 12


def test_same_lock_and_temporary_basenames_can_be_written_concurrently_in_unique_directories(tmp_path):
    runtime, _ = derive_plan(plan(), tmp_path, 'abc')
    value = {'anonymous': 'unchanged_lock', 'fixed': [1, 2, 3]}
    def write_unit(task):
        output = Path(task['args'][-1]) / 'MATCHED_SUPPORT_NULL_LOCK.json'
        write(output, value)
        return output, json.loads(output.read_text())
    with ThreadPoolExecutor(max_workers=12) as pool:
        result = list(pool.map(write_unit, runtime['tasks']))
    assert len(result) == 12 and all(row == value for _, row in result)
    assert len({path.parent for path, _ in result}) == 12
    assert not (tmp_path / 'MATCHED_SUPPORT_NULL_LOCK.json').exists()


@pytest.mark.parametrize('explicit', [['--output', 'already'], ['--output=already']])
def test_existing_output_is_rejected_instead_of_shadowed(tmp_path, explicit):
    value = plan(); value['tasks'][0]['args'] += explicit
    with pytest.raises(AuditError, match='ALREADY_EXPLICIT'):
        derive_plan(value, tmp_path, 'abc')


def test_changed_script_is_rejected(tmp_path):
    value = plan(); value['tasks'][0]['script'] = 'scripts/bgb_en_other.py'
    with pytest.raises(AuditError, match='UNEXPECTED_SCIENTIFIC_DRIVER'):
        derive_plan(value, tmp_path, 'abc')


def test_path_traversal_label_is_rejected(tmp_path):
    value = plan(); value['tasks'][0]['label'] = '../escape'
    with pytest.raises(ValueError):
        derive_plan(value, tmp_path, 'abc')

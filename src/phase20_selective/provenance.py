"""Explicit, SHA-bound compatibility for the Phase20 anchor-only audit repair.

Original checkpoints and receipts retain their actual D1 training source.  Only
enumerated immutable artifacts can be carried; accepting a source string alone
is forbidden.  Fresh runs using the current source need no legacy inventory.
"""
import hashlib
import json
from pathlib import Path

LEGACY_SOURCE = 'd1d7f8176f18417b8107767700f68ec0292181c3a0e76fce37894e06cc0c3192'


def _api():
    from . import data
    return data


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical_attempt_sha(attempt):
    return hashlib.sha256(json.dumps(attempt,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def repair_inventory():
    data = _api(); identity = data.source()
    path = data.P/'ANCHOR_AUDIT_REPAIR_COMPATIBILITY.json'
    public = _read(data.R/'AUDIT_REPAIR_COMPATIBILITY.json')
    lock = _read(data.R/'AUDIT_REPAIR_LOCK.json')
    inventory = _read(path)
    if (inventory['status'] != 'PASS' or inventory['source'] != identity or
            inventory['legacy_source'] != LEGACY_SOURCE or public['status'] != 'PASS' or
            public['source'] != identity or public['inventory_sha256'] != _sha(path) or
            lock['status'] != 'LOCKED_BEFORE_RESTART_AFTER_AUDIT_ONLY_REPAIR' or
            lock['source'] != identity or lock['training_files'] != data.source_files() or
            lock['compatibility_audit_sha256'] != _sha(data.R/'AUDIT_REPAIR_COMPATIBILITY.json') or
            lock['qualification_sha256'] != _sha(data.R/'AUDIT_REPAIR_TESTS.json')):
        raise ValueError('anchor-only repair provenance is incomplete or changed')
    return inventory


def require_record(path, checkpoint_path=None):
    """Return the untouched receipt, enforcing actual source and exact bytes."""
    data = _api(); path = Path(path).resolve(); record = _read(path)
    origin = record.get('source',record.get('training_source'))
    if origin == data.source():
        pass
    elif origin == LEGACY_SOURCE:
        inventory = repair_inventory()
        if inventory['legacy_records'].get(str(path)) != _sha(path):
            raise ValueError('D1 record is not in the frozen repair inventory: '+str(path))
        if checkpoint_path is not None:
            ckpath = Path(checkpoint_path).resolve()
            if inventory['legacy_checkpoints'].get(str(ckpath)) != _sha(ckpath):
                raise ValueError('D1 checkpoint is not in the frozen repair inventory')
    else:
        raise ValueError('unqualified actual training source: '+str(origin))
    if checkpoint_path is not None and record.get('checkpoint_sha256') != _sha(checkpoint_path):
        raise ValueError('checkpoint bytes differ from original receipt')
    return record


def require_attempt(attempt, path, index):
    data = _api(); origin = attempt.get('source')
    if origin == data.source():
        return attempt
    if origin != LEGACY_SOURCE:
        raise ValueError('unknown optimizer-attempt source')
    inventory = repair_inventory()
    expected = inventory['legacy_attempt_digests'].get(str(Path(path).resolve()),[])
    if index < 0 or index >= len(expected) or canonical_attempt_sha(attempt) != expected[index]:
        raise ValueError('original failed/successful/interrupted attempt was modified')
    return attempt


def verify_qualified_tests(gate):
    """Bind initial test evidence to D1 and the actual repair suite to D2."""
    data = _api(); identity = data.source()
    origin = gate.get('source',gate.get('training_source'))
    if origin == identity:
        if any(_sha(data.ROOT/relative) != expected for relative,expected in gate['test_files'].items()):
            raise ValueError('current qualified test bytes changed')
        return
    inventory = repair_inventory()
    if origin != LEGACY_SOURCE or any(inventory['legacy_source_files'].get(relative) != expected
                                     for relative,expected in gate['test_files'].items()):
        raise ValueError('initial test record is not bound to D1 bytes')
    qualification = _read(data.R/'AUDIT_REPAIR_TESTS.json')
    if (qualification['status'] != 'PASS' or qualification['source'] != identity or
            qualification['actual_training_tests'] != 64 or qualification['actual_math_tests'] != 7 or
            qualification['failures_errors_skips'] != 0 or
            qualification['training_source_files'] != data.source_files() or
            any(_sha(data.ROOT/relative) != expected for relative,expected in qualification['test_files'].items())):
        raise ValueError('actual current-source repair qualification is missing or stale')

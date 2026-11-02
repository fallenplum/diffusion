"""Every public stage API enforces the same fail-closed prerequisites."""
import json
from pathlib import Path


def require_stage(stage,report_root):
    required={'gate1':['gate0'],'gate2':['gate0'],'gate3':['gate0','gate1','gate2'],'gate4':['gate0','gate1','gate2','gate3']}
    if stage not in required:raise ValueError('UNKNOWN_TRAINING_STAGE')
    for previous in required[stage]:
        p=Path(report_root)/previous/'decision.json'
        if not p.is_file():raise ValueError('MISSING_PRIOR_GATE:'+previous)
        d=json.loads(p.read_text(encoding='utf-8'))
        field={'gate0':'pass','gate1':'fidelity_go','gate2':'relation_go','gate3':'compression_go'}[previous]
        if d.get(field) is not True:raise ValueError('PRIOR_GATE_NOT_PASSED:'+previous)


def run_stage(stage,*,report_root,lock):
    require_stage(stage,report_root)
    frozen=json.loads(Path(lock).read_text(encoding='utf-8'))
    if frozen.get('status')!='LOCKED_BEFORE_FORMAL_FITS':raise ValueError('FORMAL_PRECOMMIT_NOT_LOCKED')
    raise NotImplementedError('QUALIFIED_STAGE_TRAINER_NOT_IMPLEMENTED_YET')

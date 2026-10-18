"""Immutable training identity, explicit gates, and private append-only paths."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from src.phase24_cdei_g1.runtime import atomic_json, sha

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=Path('/root/autodl-tmp/phase25_private/b')
REPORT=ROOT/'reports/phase25_b'
CONFIG=ROOT/'configs/phase25_local_routing.json'
PARENT='58eaf1db947593cf8ff872d8ac2644b01b191441'
ARMS=('R0','R1','R2','R3')
NEURAL_ARMS=ARMS[1:]


def cfg():return json.loads(CONFIG.read_text(encoding='utf-8'))


def source_files():
    from src.phase19_attention.data import _import_closure
    initial=[ROOT/f'src/phase25_local/{n}.py' for n in ('__init__','runtime','data','model','train')]
    initial += [CONFIG,ROOT/'scripts/run_phase25.py',ROOT/'scripts/prepare_phase25.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(_import_closure(initial))}


def source():
    from src.phase24_cdei_g1.runtime import source as inherited
    if inherited()!='8234bca2b7fad388d3862d4d7be940f7e76f291bac516754b1e45fa5639bd18b':
        raise ValueError('published Phase24 ancestor changed')
    if cfg()['parent_commit']!=PARENT:raise ValueError('parent commit changed')
    return hashlib.sha256(json.dumps(source_files(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def cell_dir(stage,seed,fold):
    if stage not in ('prototype','outer'):raise ValueError('unknown stage')
    return PRIVATE/stage/f'seed{int(seed)}_fold{int(fold)}'


def authorize_outer(identity):
    p=PRIVATE/'INNER_GATE.json'
    if not p.is_file():raise ValueError('outer gate missing')
    gate=json.loads(p.read_text())
    if gate.get('source')!=identity or gate.get('decision')!='GO':raise ValueError('outer not admitted')
    # Outer evaluator/refit scheduling must be independently locked after GO.
    raise ValueError('outer protocol implementation requires post-screen lock; no implicit launch')


def authorization(stage,identity):
    if stage=='outer':authorize_outer(identity)
    a=json.loads((ROOT/'reports/phase25_a/decision.json').read_text())
    if a.get('status')!='AUDIT_COMPLETE_WITH_EXPLICIT_IDENTIFICATION_LIMITS' or not a.get('small_prototype_allowed'):
        raise ValueError('Phase25-A audit prerequisite missing')
    q=json.loads((PRIVATE/'REAL_QUALIFICATION_PRIVATE.json').read_text())
    if q.get('status')!='PASS' or q.get('source')!=identity:raise ValueError('real qualification missing/drift')


def verify_done(path,identity,stage,seed,fold):
    path=Path(path);d=json.loads((path/'DONE_PRIVATE.json').read_text())
    expected={'status':'COMPLETE','source':identity,'stage':stage,'seed':int(seed),'fold':int(fold),
              'model_slots':4,'neural_models':3,'epochs':80,'optimizer_steps':240}
    if any(d.get(k)!=v for k,v in expected.items()) or sha(path/'checkpoint.pt')!=d.get('checkpoint_sha256'):
        raise ValueError('completed cell source/budget/hash mismatch')
    return d

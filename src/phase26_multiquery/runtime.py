"""Immutable Phase26 identities and prototype-only authorization."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from src.phase25_local.runtime import atomic_json, sha

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path('/root/autodl-tmp/phase26_multiquery_private')
REPORT = ROOT/'reports/phase26_multiquery'
CONFIG = ROOT/'configs/phase26_multiquery.json'
PARENT = '91032779b2df9e5e6dea668274a76607247810db'
LEGACY_SOURCE = '91cad3e88814d92d9f43b3ebf8c0962d11098b3fa80a8ea81c275a6b83f2884c'
NEURAL_ARMS = ('QUERY', 'ATTN', 'COMP')


def cfg(): return json.loads(CONFIG.read_text(encoding='utf-8'))


def source_files():
    names = [f'src/phase26_multiquery/{n}.py' for n in ('__init__','runtime','data','model','train')]
    names += ['configs/phase26_multiquery.json', 'scripts/prepare_phase26.py', 'scripts/run_phase26.py']
    return {n: sha(ROOT/n) for n in names}


def source():
    from src.phase25_local.runtime import source as old_source
    if old_source() != LEGACY_SOURCE or cfg()['parent_commit'] != PARENT:
        raise ValueError('immutable Phase25 source/parent changed')
    payload = {'legacy_source': LEGACY_SOURCE, 'files': source_files()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def cell_dir(seed, fold): return PRIVATE/'prototype'/f'seed{int(seed)}_fold{int(fold)}'


def authorization(identity):
    p = PRIVATE/'REAL_QUALIFICATION_PRIVATE.json'
    if not p.is_file(): raise ValueError('real qualification missing')
    q = json.loads(p.read_text())
    if q.get('status') != 'PASS' or q.get('source') != identity:
        raise ValueError('real qualification source mismatch')


def verify_done(path, identity, seed, fold):
    path = Path(path); d = json.loads((path/'DONE_PRIVATE.json').read_text())
    fields = {'status':'COMPLETE','source':identity,'stage':'prototype','seed':int(seed),'fold':int(fold),
              'model_slots':4,'neural_models':3,'epochs':80,'optimizer_steps':240}
    if any(d.get(k) != v for k,v in fields.items()) or sha(path/'checkpoint.pt') != d.get('checkpoint_sha256'):
        raise ValueError('Phase26 completed cell identity/budget/hash mismatch')
    return d

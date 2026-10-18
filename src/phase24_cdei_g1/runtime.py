"""Explicit new training identity and append-only private experiment I/O."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from src.phase23_cdei.runtime import atomic_json, sha

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = Path('/root/autodl-tmp/phase24_cdei_g1_private')
REPORT = ROOT/'reports/phase24_cdei_g1'
CONFIG = ROOT/'configs/phase24_cdei_g1.json'
PARENT = 'c85520243e65a3884d6f9122cb733ac8dba4ae7a'
G0_SOURCE = 'f6659b92737042032a0c64bc681ca108814c6aeb2e811bff4d486655a1573f0e'
ARMS = ('REL','POOL','LOCAL','CTRL')


def cfg():
    return json.loads(CONFIG.read_text(encoding='utf-8'))


def source_files():
    paths = [ROOT/f'src/phase24_cdei_g1/{name}.py' for name in ('model','train','runtime')]
    paths += [CONFIG, ROOT/'scripts/run_phase24_cdei_g1.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


def source():
    from src.phase23_cdei.runtime import source as g0
    inherited = g0()
    if inherited != G0_SOURCE:
        raise ValueError('published G0 ancestor source changed')
    return hashlib.sha256(json.dumps({'inherited_G0':inherited,'files':source_files()},
                                    sort_keys=True,separators=(',',':')).encode()).hexdigest()


def cell_dir(role,seed,fold):
    if role not in ('inner','outer'):raise ValueError('invalid role')
    return PRIVATE/role/f'seed{int(seed)}_fold{int(fold)}'


def verify_done(path,identity,role,seed,fold):
    path=Path(path)
    done=json.loads((path/'DONE_PRIVATE.json').read_text())
    expected={'status':'COMPLETE','source':identity,'role':role,'seed':int(seed),'fold':int(fold),'arms':4,'epochs':80}
    if any(done.get(k)!=v for k,v in expected.items()) or sha(path/'checkpoint.pt')!=done['checkpoint_sha256']:
        raise ValueError('cell receipt, source or checkpoint hash mismatch')
    return done


def authorization(role,identity):
    q=json.loads((PRIVATE/'REAL_DATA_QUALIFICATION.json').read_text())
    if q['status']!='PASS' or q['source']!=identity:raise ValueError('real-data qualification missing/drift')
    if role=='outer':
        gate=json.loads((PRIVATE/'INNER_GATE.json').read_text())
        if gate['decision']!='GO' or gate['source']!=identity:raise ValueError('outer gate not admitted')

"""Read-only Phase20 inputs and an independent Phase21 training source lock."""
import hashlib
import json
from pathlib import Path
from src.phase20_selective import data as prior
from src.phase10_trimodal.data import digest

ROOT=prior.ROOT
P=Path('/root/autodl-tmp/phase21_relative_private')
R=ROOT/'reports/phase21_relative_attention'
CONFIG=ROOT/'configs/phase21_relative_attention.json'
Data=prior.Data
write=prior.write

def cfg(): return json.loads(CONFIG.read_text(encoding='utf-8'))

def source_files():
    files=dict(prior.source_files())
    initial=[CONFIG]+[ROOT/f'scripts/{name}_phase21.py' for name in ['run','prepare','schedule','start']]
    initial+=list((ROOT/'src/phase21_relative').glob('*.py'))
    initial+=list((ROOT/'tests').glob('test_phase21_*.py'))
    files.update({p.relative_to(ROOT).as_posix():digest(p) for p in prior.prior._import_closure(initial)})
    return files

def source():
    return hashlib.sha256(json.dumps(source_files(),sort_keys=True,separators=(',',':')).encode()).hexdigest()

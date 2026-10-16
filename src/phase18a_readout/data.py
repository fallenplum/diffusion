import json,hashlib
from pathlib import Path
from src.phase17r_ssd import data as old
from src.phase17_ssd.train import seed_all
from src.phase10_trimodal.data import digest

ROOT=Path(__file__).resolve().parents[2];P=Path('/root/autodl-tmp/phase18a_readout_private');R=ROOT/'reports/phase18a_local_state_readout';CONFIG=ROOT/'configs/phase18a_local_state_readout.json'
write=old.write
def cfg():return json.loads(CONFIG.read_text())

def source():
    parent=old.source()
    if parent!=cfg()['parent_training_source']:raise ValueError('Phase17R training source changed')
    paths=[CONFIG]+sorted((ROOT/'src/phase18a_readout').glob('*.py'))+[ROOT/'scripts/run_phase18a.py',ROOT/'tests/test_phase18a_readout.py',ROOT/'tests/test_phase18a_model.py']
    return hashlib.sha256((parent+''.join(p.relative_to(ROOT).as_posix()+digest(p) for p in paths)).encode()).hexdigest()

class Data(old.Data):
    def __init__(self,fold,role,seed):
        assert old.source()==cfg()['parent_training_source']
        super().__init__(fold,role,seed)


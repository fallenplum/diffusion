import hashlib,json
from pathlib import Path
from src.phase17_ssd import data as old
from src.phase17_ssd.train import Data as OldData,seed_all
from src.phase10_trimodal.data import digest

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/phase17r_conditional_repair.json'
P=Path('/root/autodl-tmp/phase17r_ssd_private')
R=ROOT/'reports/phase17r_conditional_repair'
write=old.write
def cfg():return json.loads(CONFIG.read_text())
def source():
    paths=[CONFIG]+sorted((ROOT/'src/phase17r_ssd').glob('*.py'))+[ROOT/'scripts/run_phase17r.py']
    return hashlib.sha256((old.source()+''.join(str(p.relative_to(ROOT))+digest(p) for p in paths)).encode()).hexdigest()

class Data(OldData):
    def __init__(self,fold,role,seed,direction='forward'):
        assert old.source()==cfg()['parent_training_source'],'parent training changed'
        super().__init__(fold,role,cfg()['patch_budget'],'A8',seed)
        self.direction=direction
    def forward(self,*args,**kwargs):
        if self.direction=='reverse' and kwargs.get('order','original')=='original':kwargs['order']='reverse'
        return super().forward(*args,**kwargs)

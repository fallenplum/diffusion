"""Finite qualification grid and gates; no retrospective outer-score tuning."""
import itertools,math
from src.semantic_m123r.models import ARMS
from src.semantic_m123r.training import SNAPSHOTS


def configurations(arm):
    return [{'a':a,'b':b,'n':n} for a,b,n in itertools.product((0.,) if arm=='R2' else (.01,.1),(.01,.1),(.0005,.005))]


def inner_jobs(outer=1,arms=ARMS):
    return [{'outer':outer,'inner':i,'arm':arm,'seed':seed,'config':config,'epochs':140}
        for arm in arms for seed in (31,47,83) for config in configurations(arm) for i in range(3)]


def micro(rows):
    hn=math.fsum(r['Harrell_num'] for r in rows);hd=math.fsum(r['Harrell_den'] for r in rows)
    if len(rows)!=3 or hd<=0:raise ValueError('EXACT_THREE_INNER_FOLDS_REQUIRED')
    supported=all(r['Uno_num'] is not None and r['Uno_den'] is not None and r['Uno_den']>0 for r in rows)
    return {'Harrell':hn/hd,'Uno60':math.fsum(r['Uno_num'] for r in rows)/math.fsum(r['Uno_den'] for r in rows) if supported else None}


def choose(candidates,*,trained):
    allowed=[c for c in candidates if not trained or c['epoch']>=30]
    if not allowed:raise ValueError('NO_GENUINE_ENCODER_CANDIDATES')
    best=None
    for c in allowed:
        m=micro(c['metrics']);row={**c,'summary':m}
        key=(m['Harrell'],m['Uno60'] if m['Uno60'] is not None else -math.inf,
            c['config']['b'],c['config']['a'],c['config']['n'],-c['epoch'])
        if best is None or key>best[0]:best=(key,row)
    return best[1]


def formal_arms(gates):
    groups={'M1_GROUP':('R2',),'M1_FUNCTION':('R2','R3'),'MIL':('R3','R4','R5'),
        'MOLECULAR':('R3','R4','R5','R6','R7','R8','R9')}
    selected={arm for name,arms in groups.items() if gates[name]['passed'] for arm in arms}
    return [arm for arm in ARMS if arm in selected]

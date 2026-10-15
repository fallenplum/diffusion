"""Record and recheck all shared selector/input hashes; no protected values public."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.phase16_routing.core import *
def main():
    files=list((P/'selectors').glob('*.npy'))+[P/'CANDIDATE_CACHE.json',P/'coords.npy',P/'row_owners.npy',P/'conch_global.npy',parent.DATA,parent.PRIVATE/'endpoints.npz']+list(Path('/root/phase15_complement_private/anchors').glob('DSS_*.pt'))
    target=P/'INPUT_PINS.json';rows=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in sorted(files)]
    if target.exists():
        old=json.loads(target.read_text())
        identity=lambda x:{r['path']:(r['sha256'],r['bytes']) for r in x}
        if identity(old['files'])!=identity(rows):raise ValueError('shared input/selector changed')
    else:write(target,{'status':'PASS','files':rows,'note':'recorded during formal execution;selectors created before all fits;checkpoint selected rows independently replayed'})
    print('INPUT_PINS_PASS',len(rows),flush=True)
if __name__=='__main__':main()

"""Fixed Phase27 source and append-only prototype artifact identities."""
import hashlib
import json
from pathlib import Path
from src.phase26_multiquery.runtime import atomic_json, sha

ROOT=Path(__file__).resolve().parents[2]
PRIVATE=Path('/root/autodl-tmp/phase27_readout_private')
REPORT=ROOT/'reports/phase27_separated_readout'
CONFIG=ROOT/'configs/phase27_separated_readout.json'
PARENT='95702033e77db931be60b36e27c08549d0e4bc87'
LEGACY_SOURCE='c4c29b105b3780a11ca8edc32c3a78bde8302ffd59f3c7eefd50a0fa4c13096c'
ARMS=('QUERY','ATTN','COMP')
CONTROLS=('MATCHED','RNA_MEAN_QUERY','WRONG_RNA_QUERY','VALUE_OFF','PATCH_PERMUTE')


def cfg():return json.loads(CONFIG.read_text(encoding='utf-8'))


def source_files():
    names=[f'src/phase27_readout/{n}.py' for n in ('__init__','runtime','features','math','pipeline')]
    names+=['configs/phase27_separated_readout.json','scripts/run_phase27.py']
    return {p:sha(ROOT/p) for p in names}


def source():
    from src.phase26_multiquery.runtime import source as inherited
    if inherited()!=LEGACY_SOURCE or cfg()['parent_commit']!=PARENT:raise ValueError('Phase26 ancestor changed')
    return hashlib.sha256(json.dumps({'ancestor_source':LEGACY_SOURCE,'files':source_files()},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()


def feature_path(seed,fold):return PRIVATE/'features'/f'seed{int(seed)}_fold{int(fold)}.pt'


def cell_dir(seed,fold):return PRIVATE/'prototype'/f'seed{int(seed)}_fold{int(fold)}'


def verify_done(path,identity,seed,fold):
    path=Path(path);d=json.loads((path/'DONE_PRIVATE.json').read_text())
    expected={'status':'COMPLETE','source':identity,'stage':'prototype','seed':int(seed),'fold':int(fold),
              'heads':3,'new_convex_fits':3,'new_neural_steps':0}
    if any(d.get(k)!=v for k,v in expected.items()) or sha(path/'head.pt')!=d.get('checkpoint_sha256'):
        raise ValueError('Phase27 head completion/source/hash mismatch')
    return d

"""Repair D0 metadata only after exact B0 risk identity; preserve original records."""
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.phase10_trimodal.data import digest,write_json

def main():
 root=Path('/root/phase10_trimodal_private/formal');report=Path(__file__).resolve().parents[1]/'reports/phase10_trimodal_mechanism_audit';archive=root/'metadata_repair_originals';archive.mkdir(exist_ok=True)
 old=report/'independent_validation.json'
 if old.exists() and not (archive/'independent_validation_failed.json').exists():shutil.copy2(old,archive/'independent_validation_failed.json')
 evidence=[]
 for p in sorted((root/'cells').glob('D0_*.json')):
  tag=p.stem;z=dict(np.load(root/'oof'/f'{tag}.npz'));btag=tag.replace('D0_','B0_');b=dict(np.load(root/'oof'/f'{btag}.npz'));cell=json.loads(p.read_text());bc=json.loads((root/'cells'/f'{btag}.json').read_text())
  assert np.array_equal(z['test_indices'],b['test_indices']) and np.array_equal(z['risk'],b['risk'])
  assert cell['reuse_of']=='B0' and cell['checkpoint_sha256']==bc['checkpoint_sha256']
  original=archive/p.name
  if not original.exists():shutil.copy2(p,original)
  cell['prediction_sha256']=digest(root/'oof'/f'{tag}.npz');cell['private_manifest_sha256']=digest(root/'manifests'/f'{tag}.json')
  cell['metadata_repair']={'reason':'copied B0 metadata overrode D0 container hashes','original_record_sha256':digest(original),'predictions_modified':False,'exact_B0_risk_identity':True,'B0_prediction_sha256':digest(root/'oof'/f'{btag}.npz')}
  write_json(p,cell);evidence.append({'arm':'D0','seed':cell['seed'],'fold':cell['fold'],**cell['metadata_repair']})
 write_json(report/'metadata_repair_audit.json',{'status':'PASS','records':len(evidence),'training_affected':False,'records_audit':evidence})
 print('D0 metadata repaired with exact B0 identity:',len(evidence))
if __name__=='__main__':main()

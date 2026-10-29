"""Fetch public immutable primary-code snapshots required by the extractor."""
import argparse,hashlib,json
from pathlib import Path
from urllib.request import urlopen,Request
FILES={
 'benchmark.py':('https://raw.githubusercontent.com/mahmoodlab/HEST/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/bench/benchmark.py','02db96fe0cbe08202b22577cda97fd55820dfa97d359e7f9cb53cf88dc7b77d3'),
 'st_dataset.py':('https://raw.githubusercontent.com/mahmoodlab/HEST/3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3/src/hest/bench/st_dataset.py','cf0886ba5435c83e7bd4a36d6819e6be824c8c0de1393ba2be2c8b53422133f2'),
 'trident_load.py':('https://raw.githubusercontent.com/mahmoodlab/TRIDENT/bdeeb74b7eaad8d48bda232bde7de66cae980918/trident/patch_encoder_models/load.py','74189dfd5f1ec6ec46c87e830a2e1699c0f3f4717b0994a960a6915521a274fd')}
def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);rows=[]
 for name,(url,expected) in FILES.items():
  target=a.output/name
  if target.exists():data=target.read_bytes()
  else:
   with urlopen(Request(url,headers={'User-Agent':'HEST-fixed-source-reproduction'}),timeout=60) as stream:data=stream.read()
   if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('PUBLIC_PRIMARY_CODE_SHA_MISMATCH')
   with target.open('xb') as stream:stream.write(data)
  if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('EXISTING_PRIMARY_CODE_PRESERVED_BUT_SHA_MISMATCH')
  rows.append(dict(file=name,sha256=expected,bytes=len(data)))
 print(json.dumps(dict(status='PASS_THREE_PINNED_PUBLIC_PRIMARY_CODE_SNAPSHOTS',files=rows)))
if __name__=='__main__':main()

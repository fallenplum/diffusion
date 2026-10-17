"""Download a pinned, explicitly enumerated public CRC subset to local E disk.

No full-repository snapshots, credential persistence or patient data publication.
The original HEST gated metadata is a separate, explicitly authorized request;
the public benchmark never substitutes for approval to the gated repository.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time
from huggingface_hub import HfApi,get_token,hf_hub_download

REPO='MahmoodLab/hest-bench'
REVISION='b039b5965ed7092f280d4f336da7972ef9a10e1a'
ROOT=Path(r'E:\diffusion\dataset\spatial_bridge')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--patches',action='store_true');p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if not 1<=args.workers<=4:raise ValueError('finite download concurrency')
    ROOT.mkdir(parents=True,exist_ok=True)
    info=HfApi(token=get_token()).dataset_info(REPO,revision=REVISION,files_metadata=True)
    chosen=[f for f in info.siblings if f.rfilename.split('/')[0] in ['COAD','READ'] and
            ('/adata/' in f.rfilename or '/splits/' in f.rfilename or
             (args.patches and '/patches/' in f.rfilename and f.rfilename.endswith('.h5')))]
    plan={'repo':REPO,'revision':REVISION,'public_subset':True,'patches_included':args.patches,
          'file_count':len(chosen),'bytes':sum(f.size for f in chosen),
          'files':[{'path':f.rfilename,'bytes':f.size,'sha256':f.lfs.sha256 if f.lfs else None} for f in chosen]}
    (ROOT/'metadata').mkdir(exist_ok=True)
    (ROOT/'metadata'/('CRC_PATCH_DOWNLOAD_PLAN.json' if args.patches else 'CRC_EXPRESSION_DOWNLOAD_PLAN.json')).write_text(json.dumps(plan,indent=2)+'\n')
    print('LOCKED_SUBSET',len(chosen),plan['bytes'],flush=True)
    def download(f):
        start=time.monotonic()
        target=Path(hf_hub_download(REPO,f.rfilename,repo_type='dataset',revision=REVISION,token=get_token(),local_dir=ROOT/'raw'/'hest_bench'))
        actual=sha(target);expected=f.lfs.sha256 if f.lfs else None
        if target.stat().st_size!=f.size or (expected and actual!=expected):raise ValueError('download size/SHA mismatch')
        print('VERIFIED',f.rfilename,f.size,round(time.monotonic()-start,2),flush=True)
        return {'path':f.rfilename,'bytes':f.size,'sha256':actual,'LFS_SHA_verified':expected is not None,'seconds':time.monotonic()-start}
    done=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(download,f) for f in chosen]
        for job in as_completed(futures):
            done.append(job.result())
            receipt=ROOT/'metadata'/('CRC_PATCH_DOWNLOAD_DONE.json' if args.patches else 'CRC_EXPRESSION_DOWNLOAD_DONE.json')
            tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps({'status':'COMPLETE' if len(done)==len(chosen) else 'RUNNING',
                'repo':REPO,'revision':REVISION,'expected_files':len(chosen),'verified_files':len(done),'files':done},indent=2)+'\n');tmp.replace(receipt)
    print('PHASE22_SUBSET_COMPLETE',len(done),flush=True)

if __name__=='__main__':main()

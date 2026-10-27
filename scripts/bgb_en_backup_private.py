"""Stream protected experiment files to approved local storage, outside Git.

Derived cache feature blobs are reproducible from original H5 plus saved row
indices and are excluded. Patient indices, teachers, predictions and weights
stay in this private archive; this tool never creates a public Git artifact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shlex
import subprocess
import time


def remote_command(roots):
    for root in roots:
        path=PurePosixPath(root)
        if (not path.is_absolute() or '..' in path.parts or not path.name.startswith('bgb_en')
            or not str(path).startswith('/root/')):raise ValueError('Backup is restricted to explicitly named BG-BEN private directories')
    command=['tar','--create','--gzip','--file=-',
             '--exclude=*/assets/features.npy','--exclude=*/assets/*/features.npy','--directory=/']
    command+=[str(PurePosixPath(root)).lstrip('/') for root in roots]
    return shlex.join(command)


def main():
    p=argparse.ArgumentParser();p.add_argument('--host',required=True);p.add_argument('--port',required=True,type=int)
    p.add_argument('--user',default='root');p.add_argument('--ssh-key',required=True,type=Path);p.add_argument('--known-hosts',required=True,type=Path)
    p.add_argument('--remote-root',action='append',required=True);p.add_argument('--target',required=True,type=Path)
    args=p.parse_args();command=remote_command(args.remote_root);target=args.target.resolve()
    if target.suffixes[-2:]!=['.tar','.gz']:raise ValueError('Backup target must be a private tar.gz archive')
    if any(parent.name=='.git' for parent in target.parents):raise ValueError('Private backup cannot be placed in Git metadata')
    repository=Path(__file__).resolve().parents[1]
    if target.is_relative_to(repository):raise ValueError('Use approved private storage outside the public checkout')
    partial=target.with_name(target.name+'.partial')
    if target.exists() or partial.exists():raise ValueError('Existing backup or partial file requires explicit operator review')
    target.parent.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-F','NUL' if os.name=='nt' else '/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=20',
        '-i',str(args.ssh_key),'-o','UserKnownHostsFile='+str(args.known_hosts),'-p',str(args.port),args.user+'@'+args.host,command]
    start=time.monotonic();digest=hashlib.sha256();size=0
    process=subprocess.Popen(ssh,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with partial.open('xb') as handle:
            while True:
                block=process.stdout.read(4*1024**2)
                if not block:break
                handle.write(block);digest.update(block);size+=len(block)
        stderr=process.stderr.read();code=process.wait()
        if code!=0:raise RuntimeError('Private stream transfer failed; retained partial archive for operator review')
        partial.replace(target)
        record=dict(status='COMPLETE_PRIVATE_ARCHIVE_STREAM',filename=target.name,bytes=size,sha256=digest.hexdigest(),
            elapsed_seconds=time.monotonic()-start,source_directories=args.remote_root,
            excluded_reproducible_cache_payload=['*/assets/features.npy','*/assets/*/features.npy'],
            contains_protected_arrays_and_weights=True,public_upload_authorized=False,
            transport_stderr_bytes=len(stderr),transport_stderr_sha256=hashlib.sha256(stderr).hexdigest())
        target.with_suffix(target.suffix+'.private.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({key:record[key] for key in ('status','bytes','sha256','elapsed_seconds')}))
    finally:
        if process.poll() is None:process.terminate()


if __name__=='__main__':main()

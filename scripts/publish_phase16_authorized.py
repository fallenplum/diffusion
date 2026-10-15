"""Publish only the reviewed anonymous manifest; verify remote commit and blobs."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase16_semantic.core import P, R, base, sha, write

BRANCH = 'codex/phase16-semantic-roi-routing'
SSH = 'ssh -i /root/.ssh/id_ed25519_github -o StrictHostKeyChecking=accept-new'
PATTERN = re.compile(rb'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA )?PRIVATE KEY-----')


def git(*args):
    return subprocess.check_output(['git', '-c', 'core.sshCommand=' + SSH, *args], cwd=ROOT)


def main():
    if git('branch', '--show-current').decode().strip() != BRANCH:
        raise RuntimeError('Unexpected branch; refusing publication.')
    if git('remote', 'get-url', 'origin').decode().strip() != 'git@github.com:fallenplum/diffusion.git':
        raise RuntimeError('Unexpected origin; refusing publication.')
    manifest_path = R / 'ARTIFACT_MANIFEST.json'
    delivery_path = R / 'DELIVERY.json'
    manifest = json.loads(manifest_path.read_text())
    delivery = json.loads(delivery_path.read_text())
    if manifest['status'] != 'PASS' or delivery['status'] != 'PASS' or delivery['manifest_sha256'] != sha(manifest_path):
        raise ValueError('Delivery checksum/gate failed.')
    old = json.loads((base.R / 'ARTIFACT_MANIFEST.json').read_text())
    for row in old['public_files']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise ValueError('Historical file changed: ' + row['path'])
    rows = manifest['public_files'] + [
        {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}
        for path in [manifest_path, delivery_path]
    ]
    paths = [row['path'] for row in rows]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate manifest path.')
    for row in rows:
        relative = PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix not in {'.py', '.json', '.csv', '.md'}:
            raise ValueError('Unsafe public path: ' + str(relative))
        path = ROOT.joinpath(*relative.parts)
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('Manifest mismatch: ' + row['path'])
        if PATTERN.search(data):
            raise ValueError('Protected content: ' + row['path'])
    staged = set(git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0')) - {''}
    if staged - set(paths):
        raise RuntimeError('Unrelated staged files; refusing commit.')
    git('add', '--', *paths)
    staged = set(git('diff', '--cached', '--name-only', '-z').decode().strip('\0').split('\0')) - {''}
    if staged - set(paths):
        raise RuntimeError('Unexpected file entered index.')
    git('diff', '--cached', '--check')
    if staged:
        print(git('commit', '-m', 'Complete Phase16 semantic routing, omics and MIL mechanism audit').decode(), flush=True)
    commit = git('rev-parse', 'HEAD').decode().strip()
    print(git('push', 'origin', 'HEAD:refs/heads/' + BRANCH).decode(), flush=True)
    remote = git('ls-remote', 'origin', 'refs/heads/' + BRANCH).decode().split()[0]
    if remote != commit:
        raise RuntimeError('Remote branch did not reach the reviewed commit.')
    for row in rows:
        blob = git('show', commit + ':' + row['path'])
        if hashlib.sha256(blob).hexdigest() != row['sha256']:
            raise ValueError('Committed blob mismatch: ' + row['path'])
    result = {'status': 'PASS', 'branch': BRANCH, 'commit': commit, 'remote_commit': remote,
              'verified_public_blobs': len(rows), 'previous_public_files_unchanged': len(old['public_files']),
              'patient_data_uploaded': False, 'model_weights_uploaded': False,
              'results_url': 'https://github.com/fallenplum/diffusion/blob/' + BRANCH +
                             '/reports/phase16_semantic_routing/semantic_completion/RESULTS.md'}
    write(P / 'GITHUB_PUSH_VERIFIED.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

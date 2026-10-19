"""Publish only anonymous Phase28 source/reports after immutable-parent audit."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = Path('/root/autodl-tmp/phase28_reread_private')
REPORT = ROOT/'reports/phase28_bidir_reread'
BRANCH = 'research/phase28-bidirectional-evidence-reread'
PARENT = 'd8f837bb97cdccea8960b0a4b1648a1a56b55a14'


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args]).decode().strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert git('branch', '--show-current') == BRANCH
    assert git('remote', 'get-url', 'origin') == 'git@github.com:fallenplum/diffusion.git'
    old = json.loads((PRIVATE/'PARENT_FILES_PRIVATE.json').read_text())
    assert len(old) == 2738 and all(sha(ROOT/p) == value for p, value in old.items())
    assert not git('diff', '--name-only', PARENT)
    files = sorted([*ROOT.glob('src/phase28_reread/*.py'), *ROOT.glob('tests/test_phase28*.py'),
                    *ROOT.glob('scripts/*phase28*.py'), ROOT/'configs/phase28_bidir_reread.json',
                    *REPORT.glob('*.json'), *REPORT.glob('*.md')])
    files = [p for p in files if p.name != 'PUBLIC_MANIFEST.json']
    rows = []
    forbidden = re.compile(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|hf_[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----')
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        assert relative not in old and path.suffix in ('.py', '.json', '.md') and not path.is_symlink()
        content = path.read_text(encoding='utf-8')
        assert not forbidden.search(content), ('protected identity or credential pattern', relative)
        if path.suffix == '.json':
            obj = json.loads(content)
            def check(value, trail=()):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in ('patient_ids', 'fit_risk', 'valid_risks', 'model_state', 'gene_symbols', 'coordinates'):
                            # A read-access counter is a count, not the protected
                            # array itself. Permit only this exact audited path.
                            allowed_counter = (key == 'patient_ids' and trail == ('central_feature_array_read_counts',)
                                               and type(item) is int and item >= 0)
                            assert allowed_counter, ('protected JSON value', relative, trail+(key,))
                        check(item, trail+(key,))
                elif isinstance(value, list):
                    assert not (len(value) > 20 and all(isinstance(v, (float, int)) for v in value))
                    for item in value: check(item, trail+('[]',))
            check(obj)
        rows.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    manifest = {'status': 'PASS', 'parent_commit': PARENT, 'branch': BRANCH,
                'inherited_files_verified_unchanged': len(old), 'files': rows,
                'contains_patient_arrays_weights_or_identifiers': False,
                'licensed_GMT_added': False, 'scope': 'no-semantic core implementation and qualification only'}
    target = REPORT/'PUBLIC_MANIFEST.json'
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    files.append(target)
    paths = [p.relative_to(ROOT).as_posix() for p in files]
    subprocess.check_call(['git', '-C', str(ROOT), 'add', '--', *paths])
    staged = git('diff', '--cached', '--name-only').splitlines()
    assert set(staged) == set(paths), 'staged paths outside exact public manifest'
    subprocess.check_call(['git', '-C', str(ROOT), 'commit', '-m', 'Add Phase28 bidirectional original-evidence reread qualification'])
    commit = git('rev-parse', 'HEAD')
    assert git('rev-parse', 'HEAD^') == PARENT
    subprocess.check_call(['git', '-C', str(ROOT), 'push', '-u', 'origin', BRANCH])
    remote = git('ls-remote', 'origin', 'refs/heads/'+BRANCH).split()[0]
    assert remote == commit
    print(json.dumps({'status': 'PUSH_VERIFIED', 'commit': commit, 'branch': BRANCH,
                      'new_public_files': len(paths), 'historical_files_unchanged': len(old)}))


if __name__ == '__main__':
    main()

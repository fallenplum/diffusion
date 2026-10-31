import json,re
import pytest
from scripts import semantic_m123r_publication as pub


@pytest.mark.parametrize('name',[
 'src/semantic_m123r/io.py','tests/test_semantic_m123r.py','reports/semantic_m123r/qualification/RESULTS.md',
 'configs/semantic_m123r/master_public.json'])
def test_repair_only_text_namespace(name):assert pub.allowed(name)


@pytest.mark.parametrize('name',[
 '../secret.json','configs/semantic_m123r/master_lock.json','reports/semantic_m123r/x.npz',
 'reports/semantic_m123/RESULTS.md','reports/semantic_m123r/a/b/c.csv'])
def test_no_private_lock_binary_parent_or_escape(name):assert not pub.allowed(name)


def test_original_raw_risk_array_rule_is_unchanged(tmp_path):
    name='reports/semantic_m123r/x.json';p=tmp_path/name;p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'risk':[.1,.2]}));old=pub.old.original.ALLOWLIST
    with pytest.raises(ValueError,match='raw array'):
        pub.old.scan(tmp_path,[pub.row(p,name)],re.compile('(?!)'))
    assert pub.old.original.ALLOWLIST==old
    assert pub.sha(pub.old.original.__file__)==pub.old.SCANNER


def test_projection_removes_only_machine_paths_and_keeps_full_science():
    path=pub.ROOT/'configs/semantic_m123r/master_lock.json';raw=pub.read(path)
    public=pub.scientific_projection(path)
    assert public['private_execution_lock_sha256']==pub.MASTER
    assert public['scientific_master']=={k:v for k,v in raw.items() if k not in ('parent_master','previous_study_private')}
    assert set(public['machine_path_placeholders'])=={'parent_master','previous_study_private'}

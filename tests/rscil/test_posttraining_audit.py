import importlib.util
from pathlib import Path
import pytest
from src.rscil.protocol import synthetic_gate


def test_no_go_if_only_one_comparator_wins_or_loss_worsens():
    rows=[]
    for sc in ('CORRELATED_INTERACTION','NULL','REDUNDANT'):
        for seed in range(12):
            for arm in ('S0','S1','S2','S3','S1S'):
                rows.append({'n':512,'scenario':sc,'seed':seed,'arm':arm,'Harrell':.704 if arm=='S3' else .7,'Cox_NLL':5.})
    assert synthetic_gate(rows)['passed']
    for r in rows:
        if r['scenario']=='CORRELATED_INTERACTION' and r['arm']=='S2':r['Harrell']=.705
    assert not synthetic_gate(rows)['passed']
    for r in rows:
        if r['scenario']=='CORRELATED_INTERACTION' and r['arm']=='S2':r['Harrell']=.7
        if r['scenario']=='CORRELATED_INTERACTION' and r['arm']=='S3':r['Cox_NLL']=5.001
    assert not synthetic_gate(rows)['passed']


def test_publication_scope_cannot_include_binary_parent_or_escape():
    path=Path(__file__).resolve().parents[2]/'scripts/rscil/publication.py'
    spec=importlib.util.spec_from_file_location('publication_rscil_test',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.allowed('reports/rscil/decision.json')
    for s in ('../secret.json','reports/old.json','reports/rscil/weights.pt','reports/rscil/../../raw.json'):
        assert not m.allowed(s)

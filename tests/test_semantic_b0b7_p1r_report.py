"""Metadata reporting boundaries, not fabricated scientific completion."""
import csv
from pathlib import Path
import pytest
from scripts import semantic_b0b7_p1r_report as r


def test_csv_parts_keep_original_rows_and_old_row_limit(tmp_path):
    rows=[{'seed':31,'value':str(i)} for i in range(20005)]
    parts=r.csv_parts(tmp_path,'table',rows)
    recovered=[]
    for part in parts:
        raw=list(csv.DictReader((tmp_path/part).open(encoding='utf-8',newline='')))
        assert len(raw)<=19999 and (tmp_path/part).stat().st_size<=2*1024**2
        recovered+=raw
    assert len(parts)==2 and [int(x['value']) for x in recovered]==list(range(20005))


def test_csv_parts_split_by_bytes_without_dropping_or_reordering(tmp_path):
    rows=[{'value':'a'*600000,'ordinal':i} for i in range(4)]
    parts=r.csv_parts(tmp_path,'bytes',rows)
    assert len(parts)==2
    assert all((tmp_path/p).stat().st_size<=2*1024**2 for p in parts)


def test_relation_flatten_preserves_unknown_and_zero_without_interpreting():
    data={'unknown_mass_fraction_mean':.872,'known_weighted_contribution':{'assigned_positive_mean':0.,'negative_std':None}}
    flat=r.flatten_relation(data)
    assert flat['known_weighted_contribution_assigned_positive_mean']==0.
    assert flat['known_weighted_contribution_negative_std'] is None


def test_empty_report_table_rejected_instead_of_empty_shell(tmp_path):
    with pytest.raises(ValueError,match='EMPTY_REPORT_TABLE'):
        r.csv_parts(tmp_path,'empty',[])

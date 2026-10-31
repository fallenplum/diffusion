"""Read-only qualification of root-owned P2 split and endpoint boundary."""
import hashlib
import numpy as np
import pytest
from src.semantic_b0b7 import p2_data


def folds_fixture():
    f=np.concatenate([np.full(n,i) for i,n in enumerate([103,103,102,102,102],1)])
    e=np.zeros(512,int)
    for value,events in zip(range(1,6),[13,14,12,13,13]):
        e[np.flatnonzero(f==value)[:events]]=1
    return f,e


def test_outer_test_event_perturbation_does_not_change_its_inner_assignments():
    folds,event=folds_fixture();first=p2_data.nested_split_plan(folds,event)
    altered=event.copy();altered[folds==3]=1-altered[folds==3]
    other=p2_data.nested_split_plan(folds,altered)
    for key in first:
        if key.startswith("outer3_"):
            assert first[key]==other[key]


def test_every_inner_is_disjoint_and_excludes_its_fixed_outer_test():
    folds,event=folds_fixture();plan=p2_data.nested_split_plan(folds,event)
    assert len(plan)==20
    for row in plan.values():
        fit,ev=set(row["fit"]),set(row["evaluation"])
        assert not fit&ev
        if row["role"]=="inner":
            assert fit|ev==set(np.flatnonzero(folds!=row["outer"]))
            assert not (fit|ev)&set(np.flatnonzero(folds==row["outer"]))


def test_dss_reader_preserves_canonical_original_dtype_bytes_and_never_reads_pfi(tmp_path,monkeypatch):
    ids=np.array(["PERSON"+str(i) for i in range(512)],dtype="U20")
    path=tmp_path/"synthetic_endpoints.npz"
    event=np.zeros(512);event[:65]=1
    np.savez(path,patient_hash=hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest(),
             DSS_time=np.linspace(1,100,512),DSS_event=event,DSS_valid=np.ones(512,bool))
    monkeypatch.setattr(p2_data,"ENDPOINT_SHA256",p2_data.file_sha(path))
    raw,audit=p2_data.read_dss_only(path,ids)
    assert raw["e"].sum()==65 and audit["PFI_OS_read"] is False
    with pytest.raises(ValueError,match="PATIENT_ORDER"):
        p2_data.read_dss_only(path,ids.astype("U12"))

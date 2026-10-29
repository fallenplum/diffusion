"""Anonymous administrative boundary tests; no experiment or Git operations."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import parallel_evidence_publication as publisher


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False),encoding="utf-8")


@pytest.fixture
def protocol(tmp_path,monkeypatch):
    root=tmp_path/"repo";root.mkdir();private=tmp_path/"private";private.mkdir()
    groups=dict(patient=["FAKE_SUBJECT_"+str(i) for i in range(512)],
                donor=["FAKE_DONOR_"+str(i) for i in range(6)],
                section=["FAKE_SECTION_"+str(i) for i in range(12)],
                barcodes=["FAKE_SPOT_"+str(i) for i in range(1001)])
    cohort="a"*64;stage="b"*64
    patient=hashlib.sha256("\n".join(groups["patient"]).encode()).hexdigest()
    st=publisher.canonical({k:groups[k] for k in ("donor","section","barcodes")})
    for name,value in (("CANONICAL_COHORT_SHA256",cohort),("CANONICAL_PATIENT_SHA256",patient),
                       ("CANONICAL_ST_PREPARED_SHA256",stage),("CANONICAL_ST_IDENTIFIER_SHA256",st)):
        monkeypatch.setattr(publisher.inherited,name,value)
    deny=private/"deny.json";put(deny,dict(schema="bgb_en_private_publication_denylist_v1",
        cohort_sha256=cohort,patient_order_sha256=patient,stage_b_sha256=stage,groups=groups))
    code="src/spgel/models.py";(root/code).parent.mkdir(parents=True);(root/code).write_text("# anonymous frozen code\n")
    lock=dict(schema="spgel_source_run_lock_v1",status="LOCKED_BEFORE_SPGEL_FITS",
        sources={code:publisher.sha256(root/code)},pilot=dict(logical_cells=72,actual_saved_states=192),
        s4=dict(neural_fits=2,actual_saved_states=6))
    lock["contract_sha256"]=publisher.canonical(lock)
    lock_path=private/"SOURCE_RUN_LOCK.json";put(lock_path,lock)
    public_lock="reports/spgel/SOURCE_RUN_LOCK.json"
    note="reports/spgel/S0_PROTOCOL.md";(root/note).parent.mkdir(parents=True);(root/note).write_text("RUNNING; no completion claim\n")
    historical=root/"reports/bgb_en_r7/RESULTS.md";historical.parent.mkdir(parents=True);historical.write_text("unchanged previous work\n")
    names=[code,note,public_lock]
    rows=[dict(file=n,sha256=publisher.sha256(lock_path if n==public_lock else root/n),
               bytes=(lock_path if n==public_lock else root/n).stat().st_size) for n in names]
    files=private/"files.json";put(files,dict(files=rows))
    return dict(root=root,private=private,deny=deny,lock=lock_path,files=files,note=note,rows=rows,
                historical=historical,secret=groups["barcodes"][17])


def run(f,stage="protocol"):
    return publisher.publish(f["root"],f["files"],"spgel",stage,f["lock"],f["deny"],
                             private=f["private"],copy_public_lock=True)


def test_protocol_manifest_binds_exact_bytes_includes_audit_and_excludes_self(protocol):
    f=protocol;history=publisher.sha256(f["historical"]);result=run(f)
    assert result["stage"]=="protocol" and result["git_operations"]==result["model_loads"]==0
    manifest=json.loads((f["root"]/"reports/spgel/PUBLICATION_MANIFEST.json").read_text())
    names={r["file"] for r in manifest["files"]}
    assert "reports/spgel/SOURCE_PUBLICATION_AUDIT.json" in names
    assert "reports/spgel/PUBLICATION_MANIFEST.json" not in names
    assert publisher.sha256(f["historical"])==history
    assert (f["root"]/"reports/spgel/SOURCE_RUN_LOCK.json").read_bytes()==f["lock"].read_bytes()


def test_complete_cannot_be_claimed_from_protocol_or_counts_only(protocol):
    with pytest.raises((publisher.inherited.PublicationError,FileNotFoundError)):
        run(protocol,"complete")
    assert not (protocol["root"]/"reports/spgel/PUBLICATION_MANIFEST.json").exists()


@pytest.mark.parametrize("mutation",["unknown_file","cross_line","binary_model","wrong_sha"])
def test_only_explicit_line_scoped_frozen_text_files_can_be_published(protocol,mutation):
    f=protocol;rows=list(f["rows"])
    if mutation=="wrong_sha":rows[0]=dict(rows[0],sha256="f"*64)
    else:
        relative={"unknown_file":"untracked.txt","cross_line":"reports/bgb_en_r8/secret.md",
                  "binary_model":"reports/spgel/model.pt"}[mutation]
        path=f["root"]/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text("anonymous")
        rows.append(dict(file=relative,sha256=publisher.sha256(path),bytes=path.stat().st_size))
    put(f["files"],dict(files=rows))
    with pytest.raises(publisher.inherited.PublicationError):run(f)


def test_exact_identifier_is_rejected_without_printing_its_value(protocol):
    f=protocol;(f["root"]/f["note"]).write_text("must reject "+f["secret"])
    for row in f["rows"]:
        if row["file"]==f["note"]:row.update(sha256=publisher.sha256(f["root"]/f["note"]),bytes=(f["root"]/f["note"]).stat().st_size)
    put(f["files"],dict(files=f["rows"]))
    with pytest.raises(publisher.inherited.PublicationError) as error:run(f)
    assert f["secret"] not in str(error.value)
    assert not (f["root"]/"reports/spgel/PUBLICATION_MANIFEST.json").exists()


def test_source_drift_is_rejected_before_public_lock_copy(protocol):
    f=protocol;(f["root"]/"src/spgel/models.py").write_text("changed source")
    with pytest.raises(publisher.inherited.PublicationError,match="LOCKED_SCIENTIFIC_SOURCE_BYTES_CHANGED"):
        run(f)
    assert not (f["root"]/"reports/spgel/SOURCE_RUN_LOCK.json").exists()

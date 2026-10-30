"""Synthetic metadata counterexamples; no private registry or model is loaded."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import zipfile
import copy

import pytest

from scripts import semantic_b0b7_repairs_publication as pub


def record(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    path.write_bytes(raw)
    return {"file": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


@pytest.fixture
def preservation_root(tmp_path, monkeypatch):
    root = tmp_path / "preserved"
    history = {}
    for i in range(121):
        row = record(root, f"reports/spgel_s5/OLD_{i:03d}.json", "{}\n")
        history[row["file"]] = row["sha256"]
    audit = {"historical_public_source_sha256": history, "historical_files_unchanged": 121,
        "original_privacy_scanner_sha256": pub.SCANNER_SHA, "established_denylist_sha256": pub.DENYLIST_SHA}
    audit_row = record(root, pub.OLD_AUDIT, json.dumps(audit))
    rows = [record(root, f"reports/semantic_b0b7/OLD_{i:03d}.json", "{}\n") for i in range(77)] + [audit_row]
    manifest = {"files": rows, "file_count_including_audit_excluding_manifest": 78, "manifest_does_not_hash_itself": True}
    manifest_row = record(root, pub.OLD_MANIFEST, json.dumps(manifest))
    monkeypatch.setattr(pub, "OLD_MANIFEST_SHA", manifest_row["sha256"])
    return root


def plan_fixture(root, preserved):
    row = record(root, "reports/semantic_b0b7_p1r/SYNTHETIC.json", '{"status":"synthetic_aggregate"}\n')
    plan = pub.build_plan(root, preserved, [row], phase="P1R", base_commit=pub.P1R_BASE)
    path = root / "plan_metadata.json"
    path.write_bytes(pub.encode(plan))
    return plan, path


@pytest.mark.parametrize("name,phase", [
    ("../escape.json", "P1R"), ("src/semantic_b0b7/models.py", "P1R"),
    ("reports/semantic_b0b7/OLD.json", "P1R"), ("tmp/aggregate.json", "P1R"),
    ("reports/semantic_b0b7_p1r/checkpoint.pt", "P1R"),
    ("src/semantic_b0b7/p2_tiny.py", "P1R"), ("src/semantic_b0b7/p1r_models.py", "P2"),
    ("reports/semantic_b0b7_p2/PUBLICATION_MANIFEST.json", "P2"),
    ("reports/semantic_b0b7_p2/nested/aggregate.json", "P2"),
    ("scripts/semantic_b0b7_p2.py\\escape", "P2")])
def test_only_new_phase_paths_and_no_generated_inputs(name, phase):
    with pytest.raises(ValueError):
        pub.safe_name(name, phase)


def test_old79_and_history121_are_all_byte_verified_and_changed_old_file_fails(preservation_root):
    snapshot = pub.snapshot_original(preservation_root)
    assert len(snapshot["original_79"]) == 79 and len(snapshot["history_121"]) == 121
    (preservation_root / "reports/spgel_s5/OLD_000.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="SHA_CHANGED"):
        pub.snapshot_original(preservation_root)


@pytest.mark.parametrize("payload", [
    '{"patient_ids":["synthetic_person"]}', '{"nested":{"prediction":[0.1,0.2]}}',
    json.dumps({"description": "/root/" + "private_asset"}),
    "a" * (2 * 1024**2 + 1), b"\xff\xfe\x00\x00",
], ids=["identifier_field", "nested_prediction_array", "path", "oversize", "binary"])
def test_original_privacy_size_utf8_and_raw_json_rules_are_not_relaxed(tmp_path, payload):
    row = record(tmp_path, "reports/semantic_b0b7_p1r/SYNTHETIC.json", payload)
    saved = pub.original.ALLOWLIST
    with pytest.raises(ValueError):
        pub.scan_exact(tmp_path, [row], re.compile(r"(?!x)x"))
    assert pub.original.ALLOWLIST is saved


def test_actual_identifier_pattern_and_csv_limits_still_apply(tmp_path):
    token = "synthetic_private_identifier_for_test"
    row = record(tmp_path, "reports/semantic_b0b7_p1r/SYNTHETIC.md", token)
    with pytest.raises(ValueError, match="actual private"):
        pub.scan_exact(tmp_path, [row], re.compile(token))
    row = record(tmp_path, "reports/semantic_b0b7_p1r/SYNTHETIC.csv", "aggregate\n" + "1\n"*20_000)
    with pytest.raises(ValueError, match="oversized aggregate CSV"):
        pub.scan_exact(tmp_path, [row], re.compile(r"(?!x)x"))
    row = record(tmp_path, "reports/semantic_b0b7_p1r/SYNTHETIC.csv", "aggregate\n" + "1\n"*19_999)
    assert pub.scan_exact(tmp_path, [row], re.compile(r"(?!x)x")) == [row]


def test_json1001_rows_remains_blocked(tmp_path):
    row = record(tmp_path, "reports/semantic_b0b7_p1r/SYNTHETIC.json", json.dumps([{"aggregate":1}]*1001))
    with pytest.raises(ValueError, match="unbounded public JSON array"):
        pub.scan_exact(tmp_path, [row], re.compile(r"(?!x)x"))


def test_explicit_unique_file_sha_size_inventory_required(tmp_path, preservation_root):
    row = record(tmp_path, "src/semantic_b0b7/p1r_tiny.py", "aggregate = 1\n")
    with pytest.raises(ValueError, match="UNIQUE"):
        pub.build_plan(tmp_path, preservation_root, [row,row], phase="P1R", base_commit=pub.P1R_BASE)
    changed = {**row, "bytes": row["bytes"]+1}
    with pytest.raises(ValueError, match="SIZE_CHANGED"):
        pub.build_plan(tmp_path, preservation_root, [changed], phase="P1R", base_commit=pub.P1R_BASE)
    with pytest.raises(ValueError, match="BASE_CHANGED"):
        pub.build_plan(tmp_path, preservation_root, [row], phase="P1R", base_commit="a"*40)
    with pytest.raises(ValueError, match="FULL_P1R_COMMIT"):
        pub.validate_base("P2", "a"*40)
    pub.validate_base("P2", "a"*40, "a"*40)


def test_stage_zip_contains_exact_metadata_and_verified_utf8_rows(tmp_path, preservation_root):
    root = tmp_path / "source"
    plan, path = plan_fixture(root, preservation_root)
    archive = tmp_path / "stage.zip"
    receipt = pub.stage(root, preservation_root, path, pub.sha(path), archive)
    assert receipt["established_private_identifier_scan_performed"] is False
    assert receipt["candidate_files"] == 1 and receipt["zip_members"] == 2
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {plan["files"][0]["file"], pub.PLAN_ZIP_MEMBER}
    destination = tmp_path / "unpacked"
    restored = pub.unpack_stage(archive, pub.sha(archive), path, pub.sha(path), destination)
    assert restored == plan
    assert pub.sha(destination/plan["files"][0]["file"]) == plan["files"][0]["sha256"]
    with pytest.raises(ValueError, match="EXISTING_STAGE"):
        pub.stage(root, preservation_root, path, pub.sha(path), archive)


@pytest.mark.parametrize("violation", ["extra", "duplicate", "symlink", "payload_changed"])
def test_zip_extra_duplicate_symlink_and_hash_changes_fail_before_extract(tmp_path, preservation_root, violation):
    root = tmp_path / "source"
    plan,path = plan_fixture(root,preservation_root)
    archive=tmp_path/"forged.zip";row=plan["files"][0]
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr(pub.PLAN_ZIP_MEMBER,path.read_bytes())
        info,raw=pub.zip_entry(row["file"],(root/row["file"]).read_bytes())
        if violation=="symlink":
            info.external_attr=(stat.S_IFLNK|0o777)<<16
        if violation=="payload_changed":
            raw=raw.replace(b"aggregate",b"changedxx")
        z.writestr(info,raw)
        if violation=="duplicate":
            z.writestr(info,raw)
        if violation=="extra":
            z.writestr("../escape.py",b"x=1")
    destination=tmp_path/"not_created"
    with pytest.raises(ValueError):
        pub.unpack_stage(archive,pub.sha(archive),path,pub.sha(path),destination)
    assert not destination.exists()


def simulated_final_scan(tmp_path, preservation_root, monkeypatch, *, content=None):
    """Only an artificial test registry, never the real server private registry."""
    root=tmp_path/"source";plan,path=plan_fixture(root,preservation_root)
    if content is not None:
        row=record(root,plan["files"][0]["file"],content)
        plan=pub.build_plan(root,preservation_root,[row],phase="P1R",base_commit=pub.P1R_BASE)
        path.write_bytes(pub.encode(plan))
    archive=tmp_path/"stage.zip"
    pub.stage(root,preservation_root,path,pub.sha(path),archive)
    denylist=tmp_path/"synthetic_registry.json"
    denylist.write_text(json.dumps({"groups":{"synthetic":["synthetic_identifier_must_not_escape"]}}))
    monkeypatch.setattr(pub,"DENYLIST_SHA",pub.sha(denylist))
    # Rebuild metadata after changing the synthetic constant. Original source
    # bytes and original identifier_pattern/scan_file remain unchanged.
    monkeypatch.setattr(pub.original,"validate_denylist",lambda x:x)
    old_plan_verify=pub.verify_plan
    def verify(*args,**kwargs):
        # Original preservation fixture embeds the real pinned denylist digest.
        with monkeypatch.context() as m:
            m.setattr(pub,"DENYLIST_SHA",plan["established_denylist_sha256"])
            return old_plan_verify(*args,**kwargs)
    monkeypatch.setattr(pub,"verify_plan",verify)
    old_preserve=pub.preservation
    def preserve(*args,**kwargs):
        with monkeypatch.context() as m:
            m.setattr(pub,"DENYLIST_SHA",plan["established_denylist_sha256"])
            return old_preserve(*args,**kwargs)
    monkeypatch.setattr(pub,"preservation",preserve)
    return pub.final_scan(archive,pub.sha(archive),path,pub.sha(path),preservation_root,denylist,tmp_path/"final"),plan


def test_synthetic_server_scan_is_non_circular_and_p2_requires_preserved_p1r(tmp_path,preservation_root,monkeypatch):
    binding,plan=simulated_final_scan(tmp_path,preservation_root,monkeypatch)
    root=tmp_path/"final/public";receipt_name,manifest_name=pub.publication_paths("P1R")
    manifest=json.loads((root/manifest_name).read_text())
    receipt=json.loads((root/receipt_name).read_text())
    assert binding["publication_manifest_sha256"]==pub.sha(root/manifest_name)
    assert binding["scan_receipt_sha256"]==pub.sha(root/receipt_name)==manifest["source_scan_receipt_sha256"]
    assert manifest_name not in {r["file"] for r in manifest["files"]}
    assert "publication_manifest_sha256" not in receipt and receipt["manifest_hash_is_not_embedded_here"]
    assert binding["published_files_including_manifest_and_receipt"]==3
    assert receipt["Git_operations"]==0 and receipt["patient_mapping_created"] is False
    for path in root.rglob("*"):
        if path.is_file():
            target=preservation_root/path.relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    previous=pub.snapshot_prior_p1r(preservation_root,root/manifest_name,pub.sha(root/manifest_name))
    assert len(previous)==3
    new=record(tmp_path/"newp2","src/semantic_b0b7/p2_tiny.py","aggregate=1\n")
    p2plan=pub.build_plan(tmp_path/"newp2",preservation_root,[new],phase="P2",base_commit="a"*40,
        prior_p1r_commit="a"*40,prior_manifest_path=root/manifest_name,expected_prior_sha=pub.sha(root/manifest_name))
    assert p2plan["explicit_prior_p1r_commit"]=="a"*40
    assert p2plan["prior_p1r_manifest_sha256"]==pub.sha(root/manifest_name)
    assert p2plan["preserved_snapshots"]["prior_p1r"]["file_count"]==3
    with pytest.raises(ValueError,match="SHA_CHANGED"):
        (preservation_root/plan["files"][0]["file"]).write_text("changed")
        pub.snapshot_prior_p1r(preservation_root,root/manifest_name,pub.sha(root/manifest_name))


def test_synthetic_actual_registry_match_is_rejected_not_exported(tmp_path,preservation_root,monkeypatch):
    with pytest.raises(ValueError,match="actual private"):
        simulated_final_scan(tmp_path,preservation_root,monkeypatch,
            content=json.dumps({"description":"synthetic_identifier_must_not_escape"}))
    assert not (tmp_path/"final/FINAL_BINDING.json").exists()


def test_wrong_established_denylist_is_rejected_before_any_extract(tmp_path):
    deny=tmp_path/"synthetic.json";deny.write_text("{}")
    with pytest.raises(ValueError,match="ESTABLISHED_PRIVATE_DENYLIST_CHANGED"):
        pub.final_scan("unused","unused","unused","unused","unused",deny,tmp_path/"not_created")
    assert not (tmp_path/"not_created").exists()


def test_helper_and_counterexample_sources_pass_original_local_text_rules():
    names=(pub.HELPER,pub.HELPER_TEST)
    rows=[{"file":name,"sha256":pub.sha(pub.ROOT/name),"bytes":(pub.ROOT/name).stat().st_size} for name in names]
    assert pub.scan_exact(pub.ROOT,rows,re.compile(r"(?!x)x"))==sorted(rows,key=lambda x:x["file"])


def synthetic_prior_scan(root,monkeypatch):
    """Build the real R0 receipt schema with entirely artificial text bytes."""
    receipt_name="reports/semantic_b0b7_p1r/R0_T2_PUBLIC_REPORT_SCAN_RECEIPT.json"
    specification=copy.deepcopy(pub.PRIOR_REAL_SCANS[receipt_name])
    outputs=[record(root,f"reports/semantic_b0b7_p1r/SYNTHETIC_{i:02d}.json","{}\n") for i in range(21)]
    reporter=record(root,"scripts/semantic_b0b7_p1r_diagnostic_report.py","aggregate=1\n")
    lock=record(root,"configs/semantic_b0b7/p1r_t2_lock.json","{}\n")
    dependencies=[reporter,lock,
        record(root,"scripts/semantic_b0b7_p1r_diagnostics.py","aggregate=2\n"),
        record(root,"tests/test_semantic_b0b7_p1r_diagnostics.py","aggregate=3\n")]
    build={"schema":"semantic_b0b7_p1r_r0_t2_report_build_v1","optimizer_updates":0,"model_forwards":0,
        "reporter_source_sha256":reporter["sha256"],"T2_lock_sha256":lock["sha256"],
        "source_R0_index_sha256":specification["provenance"]["source_R0_index_sha256"],
        "source_T2_sha256":specification["provenance"]["source_T2_sha256"],
        "historical_state_fields_changed":False,"raw_patient_arrays_or_weights_loaded":False,
        "files":[{**x,"file":Path(x["file"]).name} for x in outputs]}
    build_row=record(root,"reports/semantic_b0b7_p1r/R0_T2_REPORT_BUILD_AUDIT.json",json.dumps(build))
    candidates=outputs+dependencies+[build_row]
    receipt={"schema":specification["schema"],"status":specification["status"],
        "original_scanner_sha256":pub.SCANNER_SHA,"established_denylist_sha256":pub.DENYLIST_SHA,
        "Git_writes":0,"new_fits_by_reporter":0,"new_model_forwards_by_reporter":0,
        "patient_level_arrays_or_weights_exported":False,"private_identifiers_exported":False,
        "files":candidates,**specification["provenance"]}
    row=record(root,receipt_name,json.dumps(receipt))
    specification["sha256"]=row["sha256"]
    monkeypatch.setattr(pub,"PRIOR_REAL_SCANS",{receipt_name:specification})
    return candidates+[row],receipt_name,receipt


def test_exact_prior_real_scan_evidence_reuses_only_matching_bytes(tmp_path,monkeypatch):
    rows,name,receipt=synthetic_prior_scan(tmp_path,monkeypatch)
    original=pub.original.scan_file;calls=[]
    def tracked(root,relative,pattern):
        calls.append(relative)
        return original(root,relative,pattern)
    monkeypatch.setattr(pub.original,"scan_file",tracked)
    actual,audit=pub.scan_with_exact_prior_evidence(tmp_path,rows,re.compile("synthetic_secret_identifier"))
    assert actual==sorted(rows,key=lambda x:x["file"])
    assert audit["reused_exact_prior_scan_count"]==26 and audit["scanned_now_count"]==1
    assert calls==[name]  # the evidence receipt does not attest to its own bytes
    assert len(audit["coverage"])==27 and not audit["new_regex_scan_of_all_files_claimed"]
    target=rows[0]["file"]
    changed=record(tmp_path,target,"{ }\n")
    changed_rows=[changed if x["file"]==target else x for x in rows]
    calls.clear()
    _,audit=pub.scan_with_exact_prior_evidence(tmp_path,changed_rows,re.compile("synthetic_secret_identifier"))
    assert audit["reused_exact_prior_scan_count"]==25 and audit["scanned_now_count"]==2
    assert set(calls)=={target,name}


@pytest.mark.parametrize("field,value",[("original_scanner_sha256","b"*64),
    ("established_denylist_sha256","c"*64),("schema","wrong_schema"),
    ("source_R0_index_sha256","d"*64),("Git_writes",1)])
def test_different_scanner_denylist_schema_or_provenance_never_reuses_receipt(tmp_path,monkeypatch,field,value):
    rows,name,receipt=synthetic_prior_scan(tmp_path,monkeypatch)
    receipt[field]=value
    changed=record(tmp_path,name,json.dumps(receipt))
    # Even an explicitly supplied updated checksum cannot relax metadata checks.
    pub.PRIOR_REAL_SCANS[name]["sha256"]=changed["sha256"]
    rows=[changed if x["file"]==name else x for x in rows]
    _,audit=pub.scan_with_exact_prior_evidence(tmp_path,rows,re.compile(r"(?!x)x"))
    assert audit["reused_exact_prior_scan_count"]==0 and audit["scanned_now_count"]==27
    assert len(audit["unqualified_prior_receipts_not_reused"])==1


def test_one_byte_receipt_change_and_missing_provenance_force_real_rescan(tmp_path,monkeypatch):
    rows,name,receipt=synthetic_prior_scan(tmp_path,monkeypatch)
    changed=record(tmp_path,name,(tmp_path/name).read_bytes()+b" ")
    rows=[changed if x["file"]==name else x for x in rows]
    _,audit=pub.scan_with_exact_prior_evidence(tmp_path,rows,re.compile(r"(?!x)x"))
    assert audit["reused_exact_prior_scan_count"]==0
    receipt.pop("new_fits_by_reporter")
    changed=record(tmp_path,name,json.dumps(receipt));pub.PRIOR_REAL_SCANS[name]["sha256"]=changed["sha256"]
    rows=[changed if x["file"]==name else x for x in rows]
    _,audit=pub.scan_with_exact_prior_evidence(tmp_path,rows,re.compile(r"(?!x)x"))
    assert audit["reused_exact_prior_scan_count"]==0 and audit["scanned_now_count"]==27

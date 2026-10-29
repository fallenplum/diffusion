"""Anonymous administrative adversaries; no model, patient or source arrays."""
import copy
import json
from pathlib import Path
import re
import shutil

import pytest

from scripts import hest_official_publication as pub


@pytest.mark.parametrize("name", ["../private.json", "E:/data.json",
    "reports/spgel/S3_RESULTS.md", "reports/bgb_en_r8/R8_RESULTS.json", "src/spgel_hest/model.pt"])
def test_historical_binary_and_escaping_inputs_are_not_publication_candidates(name):
    with pytest.raises(ValueError):
        pub.safe_name(name)


def listed(tmp_path, name, text):
    path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding="utf-8")
    return dict(file=name,sha256=pub.sha(path),bytes=path.stat().st_size)


def test_explicit_file_sha_is_checked_even_if_size_stays_the_same(tmp_path):
    row=listed(tmp_path,"src/spgel_hest/official.py","safe")
    allow=tmp_path/"allow.json";allow.write_text(json.dumps(dict(files=[row])))
    pub.inventory(tmp_path,allow)
    (tmp_path/row["file"]).write_text("fake")
    with pytest.raises(ValueError,match="SHA_OR_SIZE_CHANGED"):
        pub.inventory(tmp_path,allow)


def test_actual_geo_count_only_projection_recovers_original_byte_identity():
    path=Path(__file__).parents[1]/"reports/spgel/INDEPENDENT_CRC_SOURCE_QUALIFICATION_PUBLIC_PROJECTION_20261029.json"
    receipt=pub.restore_geo_projection(pub.read(path))
    assert receipt["byte_exact_reconstruction"] and receipt["original_raw_sha256"]==pub.GEO_ORIGINAL_SHA


@pytest.mark.parametrize("change_lineage", [False,True])
def test_changed_aggregate_count_cannot_hide_behind_projection_lineage(change_lineage):
    path=Path(__file__).parents[1]/"reports/spgel/INDEPENDENT_CRC_SOURCE_QUALIFICATION_PUBLIC_PROJECTION_20261029.json"
    value=copy.deepcopy(pub.read(path))
    value["new_studies"]["GSE267401"]["official_attachment_groups"]["barcode_attachment_summary"]["files"]-=1
    if change_lineage:
        value["public_projection_lineage"]["original_canonical_semantic_sha256"]="0"*64
    with pytest.raises(ValueError,match="PROJECTION_LOSS_OR_ORIGINAL_BYTES_CHANGED"):
        pub.restore_geo_projection(value)


def test_existing_exact_identifier_rule_is_not_bypassed_by_new_allowlist(tmp_path,monkeypatch):
    row=listed(tmp_path,"reports/spgel/HEST_SYNTHETIC_20261029.md","SYNTHETIC_FORBIDDEN_TOKEN")
    deny=tmp_path/"synthetic_deny.json";deny.write_text(json.dumps(dict(groups={"patient":["SYNTHETIC_FORBIDDEN_TOKEN"]})))
    monkeypatch.setattr(pub,"DENY_SHA",pub.sha(deny))
    monkeypatch.setattr(pub.inherited,"validate_denylist",lambda x:x)
    with pytest.raises(ValueError,match="actual private patient/ST identifier"):
        pub.checked_scan(tmp_path,[row],deny)


def test_raw_prediction_array_remains_blocked_by_inherited_scanner(tmp_path,monkeypatch):
    row=listed(tmp_path,"reports/spgel/HEST_SYNTHETIC_20261029.json",json.dumps({"predictions":[1,2,3]}))
    deny=tmp_path/"synthetic_deny.json";deny.write_text(json.dumps(dict(groups={"patient":["SYNTHETIC_NOT_PRESENT"]})))
    monkeypatch.setattr(pub,"DENY_SHA",pub.sha(deny))
    monkeypatch.setattr(pub.inherited,"validate_denylist",lambda x:x)
    with pytest.raises(ValueError,match="private row-level JSON field"):
        pub.checked_scan(tmp_path,[row],deny)


def test_partial_completion_never_writes_source_audit_or_manifest(tmp_path,monkeypatch):
    row=listed(tmp_path,"src/spgel_hest/official.py","anonymous")
    allow=tmp_path/"allow.json";allow.write_text(json.dumps(dict(files=[row])))
    bundle=tmp_path/"incomplete.json";bundle.write_text(json.dumps(dict(
        schema="hest_official_local_metadata_evidence_v1",status="PASS_LOCAL_PRIVATE_JSON_AND_INDEPENDENT_ROOT_REVIEW_BINDINGS",
        dataset_revision=pub.DATA_REVISION,source_canonical_sha256=pub.RIDGE_SOURCE_SHA,
        actual_features=8,actual_feature_patches=24058,actual_Ridge_fits=4,actual_model_replays=3,
        new_fits_by_publication=0,model_or_prediction_arrays_read_by_publication=False)))
    monkeypatch.setattr(pub,"checked_scan",lambda root,rows,deny:(rows,re.compile(r"(?!x)x")))
    with pytest.raises(ValueError,match="COMPLETE_LOCAL_METADATA_EVIDENCE_REQUIRED"):
        pub.publish_manifest(tmp_path,allow,tmp_path/"unused",bundle)
    assert not (tmp_path/pub.AUDIT).exists() and not (tmp_path/pub.MANIFEST).exists()


def test_actual_feature_projection_audit_binds_original_frozen_feature_qualification():
    root=Path(__file__).parents[1]
    names=projection_files(root)
    bundle={"feature_qualification_original_sha256":"f923dd1d0fa9cc9e0e14f217a06e100d9381b6c67f93375a53111dad0d8dc30d"}
    check=pub.verify_feature_public_projections(root,names,bundle)
    assert check["actual_byte_exact_restores"]==3 and check["projection_is_not_executable_science"]
    bundle["feature_qualification_original_sha256"]="0"*64
    with pytest.raises(ValueError,match="PROJECTION_OR_ACTUAL_RESTORE_BYTES_CHANGED"):
        pub.verify_feature_public_projections(root,names,bundle)


def test_feature_projection_modified_after_actual_restore_is_rejected(tmp_path):
    root=Path(__file__).parents[1]
    names=projection_files(root)
    for name in names:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/name,target)
    manifest=pub.read(tmp_path/pub.PROJECTION_MANIFEST)
    target=tmp_path/manifest["artifacts"][0]["projection_file"]
    target.write_bytes(target.read_bytes()+b"changed")
    bundle={"feature_qualification_original_sha256":"f923dd1d0fa9cc9e0e14f217a06e100d9381b6c67f93375a53111dad0d8dc30d"}
    with pytest.raises(ValueError,match="PROJECTION_OR_ACTUAL_RESTORE_BYTES_CHANGED"):
        pub.verify_feature_public_projections(tmp_path,names,bundle)


def projection_files(root):
    manifest=pub.read(root/pub.PROJECTION_MANIFEST)
    return {pub.PROJECTION_MANIFEST,pub.PROJECTION_RESTORE,pub.MATERIALIZER,pub.MATERIALIZER_TEST,
            *(r["projection_file"] for r in manifest["artifacts"])}

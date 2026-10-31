"""Synthetic publication counterexamples; no Git writes or private data reads."""
from copy import deepcopy
import hashlib
import io
import csv
import json
from pathlib import Path
import re
import zipfile

import pytest

from scripts import semantic_b0b7_p2s_publication as pub


def put(root, name, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + "\n").encode()
    path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    return {"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "source"; root.mkdir()
    old = []
    for i in range(472):
        name = "scripts/verify_semantic_b0b7_p2.py" if i == 0 else f"protected/old_{i:03d}.py"
        row = put(root, name, f"# anonymous old source {i}\n".encode())
        raw = (root / name).read_bytes()
        row["git_blob_sha1"] = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(); old.append(row)
    b = {"schema": "semantic_b0b7_p2s_published_parent472_baseline_v1", "commit": pub.BASE_COMMIT,
        "tree": pub.BASE_TREE, "file_count": 472, "all_local_bytes_and_Git_blob_SHA1_verified": True,
        "total_bytes": sum(r["bytes"] for r in old), "files": old}
    baseline = tmp_path / "baseline.json"; baseline.write_bytes(pub.encode(b))
    monkeypatch.setattr(pub, "BASELINE_SHA", pub.sha(baseline))
    rows = []
    for name in (pub.HELPER, pub.TEST, pub.VERIFIER, pub.VERIFIER_TEST,
                 "src/semantic_b0b7/p2s_demo.py", "scripts/semantic_b0b7_p2s_schedule.py", "scripts/semantic_b0b7_p2s_report.py"):
        raw = Path(pub.__file__).read_bytes() if name == pub.HELPER else b"# anonymous synthetic source\n"
        rows.append(put(root, name, raw))
    indexed = {r["file"]: r for r in rows}
    sources = {"src/semantic_b0b7/p2s_demo.py": indexed["src/semantic_b0b7/p2s_demo.py"]["sha256"],
               old[0]["file"]: old[0]["sha256"]}
    study = {"schema": "semantic_b0b7_p2s_lock_v1", "status": "LOCKED_BEFORE_P2S_OPTIMIZER",
             "parent_git_commit": pub.BASE_COMMIT, "frozen_sources": sources}
    sr = put(root, "configs/semantic_b0b7/p2s_lock.json", study); rows.append(sr)
    monkeypatch.setattr(pub, "STUDY_SHA", sr["sha256"])
    execution = {"schema": "semantic_b0b7_p2s_execution_lock_v1", "status": "LOCKED_BEFORE_P2S_SCHEDULER_LAUNCH",
        "parent_pretraining_lock_sha256": pub.STUDY_SHA,
        "execution_sources": {"scripts/semantic_b0b7_p2s_schedule.py": indexed["scripts/semantic_b0b7_p2s_schedule.py"]["sha256"]}}
    er = put(root, "configs/semantic_b0b7/p2s_execution_lock.json", execution); rows.append(er)
    monkeypatch.setattr(pub, "EXECUTION_SHA", er["sha256"])
    readout = {"schema": "semantic_b0b7_p2s_readout_lock_v1", "status": "LOCKED_BEFORE_P2S_DSS_FITS",
        "parent_study_lock_sha256": pub.STUDY_SHA, "expected_feature_states": 72, "expected_Cox_fits": 2376,
        "primary_ridge_lambda": 1., "ridge_candidates": [.1, 1., 10.], "outer_evaluation": False, "readout_sources": sources}
    dr = put(root, "configs/semantic_b0b7/p2s_readout_lock.json", readout); rows.append(dr)
    prefix = "reports/semantic_b0b7_p2s/"
    costs = {"new_optimizer_updates": 21600, "actual_checkpoint_replays": 72, "new_snapshots": 54,
             "reused_original400_snapshots": 18, "actual_Cox_fits": 0, "explicit_NA_head_cells": 2376}
    cr = put(root, prefix + "09_COMPLETION.json", {"schema": "semantic_p2s_reporting_completion_v1",
        "scientific_mechanism_decision": "NOT_AUTOMATICALLY_ASSIGNED", "epoch_pair_weight_or_ridge_winner_selected": False,
        "CIs_p_values_or_bootstrap_computed": False, "costs": costs}); rows.append(cr)
    table = put(root, prefix + "TABLE.csv", b"split,value\ninner,1\n"); rows.append(table)
    ar = put(root, prefix + "AUDIT.csv", b"status,positions\nNA,2376\n"); rows.append(ar)
    index = {"schema": "semantic_p2s_receipt_report_index_v1", "status": "COMPLETE_FIXED_GRID_WITH_EXPLICIT_NA",
        "pretraining_lock_sha256": pub.STUDY_SHA, "readout_lock_sha256": dr["sha256"], "outer_results_loaded": False,
        "new_fits_or_forwards": 0, "self_hash_cycle": False, "readout_manifest_sha256": "f" * 64,
        "source_sha256": indexed["scripts/semantic_b0b7_p2s_report.py"]["sha256"], "frozen_scientific_sources": sources,
        "readout_source_closure": sources, "files": [{**r, "file": Path(r["file"]).name} for r in (cr, table)]}
    ir = put(root, prefix + "INDEX.json", index); rows.append(ir)
    audit = {"schema": "semantic_p2s_independent_postsave_Z_coefficient_audit_v1", "status": "PASS_WITH_EXPLICIT_RANK_NA",
        "readout_jobs": 72, "registered_head_positions": 2376, "study_lock_sha256": pub.STUDY_SHA,
        "readout_lock_sha256": dr["sha256"], "manifest_sha256": "f" * 64,
        "report_arithmetic": {"status": "PASS_264_CONFIGURATION_AGGREGATE_ARITHMETIC", "configuration_summaries_checked": 264},
        "model_predict_calls": 0, "producer_scorer_calls": 0, "new_optimizer_fits": 0, "new_neural_forwards": 0,
        "outer_predictions_read": False, "scientific_mechanism_decision_assigned": False, "fitted_heads_audited": 0,
        "explicit_rank_NA": 2376, "verifier_source_sha256": indexed[pub.VERIFIER]["sha256"],
        "independent_helper_source_sha256": old[0]["sha256"], "files": [{**ar, "file": "AUDIT.csv"}]}
    vr = put(root, prefix + "INDEPENDENT_SUMMARY.json", audit); rows.append(vr)
    proofs = dict(zip(("study_lock", "execution_lock", "readout_lock", "report_index", "independent", "completion"), (sr, er, dr, ir, vr, cr)))
    plan = pub.build_plan(root, root, baseline, pub.BASELINE_SHA, rows, proofs)
    plan_path = tmp_path / "plan.json"; plan_path.write_bytes(pub.encode(plan))
    return dict(root=root, baseline=baseline, old=old, rows=rows, proofs=proofs, plan=plan, plan_path=plan_path, tmp=tmp_path)


def test_fixed_branch_parent_tree_and_actual_local472_bytes(bundle):
    assert bundle["plan"]["base_git_commit"] == pub.BASE_COMMIT and bundle["plan"]["base_git_tree"] == pub.BASE_TREE
    assert bundle["plan"]["branch"] == "research/semantic-b0b7-p2-sufficiency"
    assert bundle["plan"]["preservation"]["protected_files"] == 472
    assert bundle["plan"]["private_identifier_scan_performed"] is False


@pytest.mark.parametrize("name", ["scripts/semantic_b0b7_p2_publication.py", "reports/semantic_b0b7_p2/RESULTS.md",
    "src/semantic_b0b7/models.py", "reports/semantic_b0b7_p2s/a.private.json", "reports/semantic_b0b7_p2s/raw.npz",
    "../escape.py", "scripts/verify_semantic_b0b7_p2.py", pub.MANIFEST, pub.SCAN_RECEIPT])
def test_old_namespace_binary_and_generated_candidates_are_blocked(name):
    with pytest.raises(ValueError): pub.safe_name(name)


@pytest.mark.parametrize("tamper", ["source_missing", "report_part_missing", "audit_part_missing", "verifier_missing", "helper_missing", "proof_missing", "duplicate", "old_candidate"])
def test_complete_reproduction_closure_is_required(bundle, tamper):
    rows, proofs = deepcopy(bundle["rows"]), deepcopy(bundle["proofs"])
    target = {"source_missing": "src/semantic_b0b7/p2s_demo.py", "report_part_missing": "TABLE.csv",
              "audit_part_missing": "AUDIT.csv", "verifier_missing": pub.VERIFIER, "helper_missing": pub.HELPER}
    if tamper in target: rows = [r for r in rows if not r["file"].endswith(target[tamper])]
    elif tamper == "proof_missing": del proofs["independent"]
    elif tamper == "duplicate": rows.append(rows[0])
    else: rows.append({k: bundle["old"][0][k] for k in ("file", "bytes", "sha256")})
    with pytest.raises(ValueError):
        pub.build_plan(bundle["root"], bundle["root"], bundle["baseline"], pub.BASELINE_SHA, rows, proofs)


def test_changed_old_published_file_prevents_staging(bundle):
    p = bundle["root"] / bundle["old"][2]["file"]; p.write_bytes(b"# changed old file\n")
    with pytest.raises(ValueError):
        pub.stage(bundle["root"], bundle["root"], bundle["baseline"], pub.BASELINE_SHA,
            bundle["plan_path"], pub.sha(bundle["plan_path"]), bundle["tmp"] / "must_not_exist.zip")


@pytest.mark.parametrize("bad", [list(range(129)), {"prediction": [1, 2]}, {"patient_id": "anonymous"}, [None] * 1001])
def test_original_json_limits_are_not_relaxed(bad):
    with pytest.raises(ValueError): pub.encode(bad)


def test_original_scanner_enforces_actual_private_pattern_utf8_and_csv_limit(tmp_path):
    root = tmp_path; name = "reports/semantic_b0b7_p2s/TEST.md"
    needle = "".join(("PRIVACY", "NEEDLE", "047"))
    row = put(root, name, needle.encode())
    original_allowlist = pub.original.ALLOWLIST
    with pytest.raises(ValueError): pub.scan_exact(root, [row], re.compile(needle))
    assert pub.original.ALLOWLIST is original_allowlist
    name = "reports/semantic_b0b7_p2s/TEST.csv"
    row = put(root, name, ("value\n" + "0\n" * 20000).encode())
    with pytest.raises(ValueError): pub.scan_exact(root, [row], re.compile(r"(?!x)x"))
    row = put(root, "reports/semantic_b0b7_p2s/BAD.md", b"\xff")
    with pytest.raises(ValueError): pub.scan_exact(root, [row], re.compile(r"(?!x)x"))


def test_metadata_stage_rejects_extra_zip_payload(bundle):
    archive = bundle["tmp"] / "stage.zip"
    pub.stage(bundle["root"], bundle["root"], bundle["baseline"], pub.BASELINE_SHA,
        bundle["plan_path"], pub.sha(bundle["plan_path"]), archive)
    with zipfile.ZipFile(archive, "a") as z: z.writestr("unexpected.pt", b"model")
    with pytest.raises(ValueError):
        pub.unpack_stage(archive, pub.sha(archive), bundle["plan_path"], pub.sha(bundle["plan_path"]),
            bundle["baseline"], pub.BASELINE_SHA, bundle["tmp"] / "unpack")


def test_actual_scan_rescans_every_candidate_controls_and_generated_without_cycles(bundle, monkeypatch):
    archive = bundle["tmp"] / "stage.zip"
    pub.stage(bundle["root"], bundle["root"], bundle["baseline"], pub.BASELINE_SHA,
        bundle["plan_path"], pub.sha(bundle["plan_path"]), archive)
    deny = bundle["tmp"] / "deny.json"; deny.write_text('{"groups":{"test":["unused_guard_needle"]}}', encoding="utf-8")
    monkeypatch.setattr(pub, "DENYLIST_SHA", pub.sha(deny))
    # Rebind the synthetic plan to this synthetic registry; real API pins d73b.
    plan = deepcopy(bundle["plan"]); plan["established_denylist_sha256"] = pub.DENYLIST_SHA
    bundle["plan_path"].write_bytes(pub.encode(plan)); archive.unlink()
    pub.stage(bundle["root"], bundle["root"], bundle["baseline"], pub.BASELINE_SHA,
        bundle["plan_path"], pub.sha(bundle["plan_path"]), archive)
    monkeypatch.setattr(pub.original, "validate_denylist", lambda x: x)
    git_calls = []
    def fake_git(repo, b):
        git_calls.append(1)
        return {"protected_files_verified": 472, "Git_write_operations": 0}
    monkeypatch.setattr(pub, "verify_git_parent", fake_git)
    calls = []; scanner = pub.original.scan_file
    def spy(*args): calls.append(args[1]); return scanner(*args)
    monkeypatch.setattr(pub.original, "scan_file", spy)
    out = bundle["tmp"] / "final"
    binding = pub.final_scan(archive, pub.sha(archive), bundle["plan_path"], pub.sha(bundle["plan_path"]),
        bundle["baseline"], pub.BASELINE_SHA, "synthetic_git", deny, out)
    assert len(git_calls) == 2 and binding["Git_write_operations"] == 0
    assert binding["candidate_reused_prior_scan_count"] == 0
    assert set(calls) == {r["file"] for r in bundle["rows"]} | {pub.PLAN_MEMBER, pub.BASELINE_MEMBER, pub.SCAN_RECEIPT, pub.MANIFEST}
    assert len(calls) == len(bundle["rows"]) + 4
    manifest = pub.read(out / "public" / pub.MANIFEST); receipt = pub.read(out / "public" / pub.SCAN_RECEIPT)
    assert pub.MANIFEST not in {r["file"] for r in manifest["files"]}
    assert manifest["source_scan_receipt_sha256"] == pub.sha(out / "public" / pub.SCAN_RECEIPT)
    assert "publication_manifest_sha256" not in receipt
    assert len(binding["files"]) == len(bundle["rows"]) + 2
    for row in binding["files"]:
        raw = (out / "public" / row["file"]).read_bytes()
        assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == row["git_blob_sha1"]


@pytest.mark.parametrize("tamper", [None, "tree", "blob", "index"])
def test_readonly_git_object_verifier_checks_all472_without_export(bundle, monkeypatch, tamper):
    b = pub.read(bundle["baseline"]); commands = []
    controls = iter([{"HEAD": "h", "index": "i"}, {"HEAD": "h", "index": "changed" if tamper == "index" else "i"}])
    monkeypatch.setattr(pub, "git_control_hashes", lambda unused_repo: next(controls))
    def command(repo, *args):
        commands.append(args)
        if args[:2] == ("cat-file", "-t"): return b"commit\n"
        if args[0] == "rev-parse": return (("0" * 40 if tamper == "tree" else pub.BASE_TREE) + "\n").encode()
        assert args[:2] == ("ls-tree", "-rz")
        return b"".join(f"100644 blob {r['git_blob_sha1']}\t{r['file']}\0".encode() for r in b["files"])
    monkeypatch.setattr(pub, "git_command", command)
    raw = b""
    for i, r in enumerate(b["files"]):
        content = (bundle["root"] / r["file"]).read_bytes()
        if tamper == "blob" and i == 0: content = b"X" + content[1:]
        raw += f"{r['git_blob_sha1']} blob {r['bytes']}\n".encode() + content + b"\n"
    class FakeBatch:
        stdin = None
        def __init__(self, argv, **kwargs):
            assert argv[-2:] == ["cat-file", "--batch"]
            assert kwargs["env"]["GIT_NO_REPLACE_OBJECTS"] == "1"
            self.stdin, self.stdout, self.stderr = io.BytesIO(), io.BytesIO(raw), io.BytesIO()
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(pub.subprocess, "Popen", FakeBatch)
    if tamper:
        with pytest.raises(ValueError): pub.verify_git_parent("synthetic_git", b)
    else:
        result = pub.verify_git_parent("synthetic_git", b)
        assert result["protected_files_verified"] == 472 and result["Git_write_operations"] == 0
        assert result["checkout_or_BASE_export_performed"] is False
    assert all(c[0] in ("cat-file", "rev-parse", "ls-tree") for c in commands)


def projected_bundle(bundle):
    root = bundle["root"]; rows = deepcopy(bundle["rows"]); proofs = deepcopy(bundle["proofs"])
    old_proof = proofs["report_index"]; source = pub.read(root / old_proof["file"])
    rows = [r for r in rows if r["file"] != old_proof["file"]]
    prefix = "reports/semantic_b0b7_p2s/"; projections = []; physical_parts = []
    headers = [f"column_{i:03d}" for i in range(154)]
    for number in range(1, 4):
        data = [[f"source{number}_row{r}_col{c}" for c in range(154)] for r in range(3)]
        data[0][0] = "quoted,value"; data[1][1] = 'a"quote'
        out = io.StringIO(newline=""); writer = csv.writer(out, lineterminator="\n")
        writer.writerow(headers); writer.writerows(data); raw = out.getvalue().encode()
        name = f"04_all_2376_readout_cells_{number:02d}.csv"
        original = {"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        source["files"].append(original)
        parts = []
        for group, columns, ranges in ((1, headers[:149], ((0, 2), (2, 3))), (2, headers[149:], ((0, 3),))):
            for vertical, (start, stop) in enumerate(ranges):
                partname = f"PROJECTED_{number}_{group}_{vertical}.csv"
                buf = io.StringIO(newline=""); w = csv.writer(buf, lineterminator="\n")
                w.writerow(["aggregate_row_ordinal"] + columns)
                w.writerows([[r] + [data[r][headers.index(c)] for c in columns] for r in range(start, stop)])
                row = put(root, prefix + partname, buf.getvalue().encode()); rows.append(row)
                metadata = {**row, "file": partname}; physical_parts.append(metadata)
                parts.append({**metadata, "columns": columns, "row_start": start, "row_stop_exclusive": stop})
        projections.append({"source_file": name, "source_bytes": len(raw), "source_sha256": original["sha256"],
            "source_headers": headers, "source_rows": 3, "ordinal_field": "aggregate_row_ordinal", "ordinal_start": 0, "parts": parts})
    sr = put(root, prefix + "SOURCE_REPORT_INDEX.json", source); rows.append(sr)
    source_code = put(root, "scripts/semantic_b0b7_p2s_project_report.py", b"# independent projection source\n")
    test_code = put(root, "tests/test_semantic_b0b7_p2s_project_report.py", b"# projection tests\n")
    rows.extend((source_code, test_code))
    public_index = {**source, "schema": "semantic_p2s_lossless_publication_report_index_v1",
        "files": [r for r in source["files"] if not r["file"].startswith("04_all_2376_readout_cells_")] +
            [{**sr, "file": "SOURCE_REPORT_INDEX.json"}] + physical_parts,
        "source_report_index_file": "SOURCE_REPORT_INDEX.json", "source_report_index_sha256": sr["sha256"],
        "projection_source": {k: source_code[k] for k in ("file", "sha256")},
        "projection_test": {k: test_code[k] for k in ("file", "sha256")}, "column_projection": projections}
    row = put(root, prefix + "PUBLICATION_REPORT_INDEX.json", public_index); rows.append(row)
    proofs["report_index"] = row
    return rows, proofs, public_index


def test_three_overwide_sources_reconstruct_byte_exact_from_safe_public_parts(bundle):
    rows, proofs, public_index = projected_bundle(bundle)
    plan = pub.build_plan(bundle["root"], bundle["root"], bundle["baseline"], pub.BASELINE_SHA, rows, proofs)
    projection = plan["metadata_proof_closure"]["lossless_column_projection"]
    assert projection["virtual_source_files"] == 3 and projection["all_source_CSV_bytes_independently_reconstructed_exact"] is True
    assert projection["original_csv_files_written_or_published"] is False
    assert not any(Path(r["file"]).name in {p["source_file"] for p in public_index["column_projection"]} for r in rows)
    for part in public_index["column_projection"]:
        assert part["source_rows"] == 3 and len(part["source_headers"]) == 154


@pytest.mark.parametrize("tamper", ["missing_part", "duplicated_cells", "wrong_ordinal", "altered_value", "fourth_virtual", "normal_file_missing"])
def test_projection_cannot_drop_duplicate_modify_or_generalize_missing_sources(bundle, tamper):
    rows, proofs, index = projected_bundle(bundle)
    if tamper == "missing_part":
        index["column_projection"][0]["parts"].pop()
    elif tamper == "duplicated_cells":
        extra = deepcopy(index["column_projection"][0]["parts"][0])
        extra["file"] = "DUPLICATE.csv"
        oldpart = bundle["root"] / "reports/semantic_b0b7_p2s/" / index["column_projection"][0]["parts"][0]["file"]
        row = put(bundle["root"], "reports/semantic_b0b7_p2s/DUPLICATE.csv", oldpart.read_bytes()); rows.append(row)
        index["files"].append({**row, "file": "DUPLICATE.csv"}); index["column_projection"][0]["parts"].append(extra)
    elif tamper in ("wrong_ordinal", "altered_value"):
        part = index["column_projection"][0]["parts"][0]
        name = "reports/semantic_b0b7_p2s/" + part["file"]
        parsed = list(csv.reader(io.StringIO((bundle["root"] / name).read_text())))
        parsed[1][0 if tamper == "wrong_ordinal" else 1] = "wrong"
        out = io.StringIO(newline=""); w = csv.writer(out, lineterminator="\n"); w.writerows(parsed)
        row = put(bundle["root"], name, out.getvalue().encode())
        rows = [row if r["file"] == name else r for r in rows]
        part.update(bytes=row["bytes"], sha256=row["sha256"])
        index["files"] = [{**row, "file": part["file"]} if r["file"] == part["file"] else r for r in index["files"]]
    elif tamper == "fourth_virtual":
        index["column_projection"].append(deepcopy(index["column_projection"][0]))
    else:
        index["files"] = [r for r in index["files"] if r["file"] != "TABLE.csv"]
    proof = put(bundle["root"], proofs["report_index"]["file"], index)
    rows = [proof if r["file"] == proof["file"] else r for r in rows]; proofs["report_index"] = proof
    with pytest.raises(ValueError):
        pub.build_plan(bundle["root"], bundle["root"], bundle["baseline"], pub.BASELINE_SHA, rows, proofs)

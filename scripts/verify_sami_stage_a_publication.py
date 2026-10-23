"""Finite, fail-closed SAMI Stage A anonymous publication audit; never pushes.

Private inputs supply an exact identity denylist only.  They are never copied
into publication artifacts.  --self-test uses synthetic inputs exclusively.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]
REPORT_ROOT = "reports/sami_surv_stage_a/"
CORE_FILES = (
    "src/sami_surv/annotations.py", "src/sami_surv/morphology.py",
    "scripts/prepare_sami_stage_a.py", "scripts/run_sami_stage_a.py",
    "scripts/audit_sami_stage_a.py", "scripts/summarize_sami_stage_a.py",
    "scripts/verify_sami_stage_a_publication.py",
    "tests/test_sami_annotations.py", "tests/test_sami_morphology.py",
    "tests/test_sami_independent_audit.py", "configs/sami_surv_stage_a_v1.json",
)
REQUIRED_REPORTS = (
    "PROTOCOL.md", "DATA_AUDIT.json", "SOURCE_LOCK_GUARD_REPAIR.json", "RESULTS.json",
    "INDEPENDENT_AUDIT.json", "EXECUTION_AUDIT.json", "RESULTS.md", "decision.json",
    "ANONYMOUS_SECTION_METRICS.json",
)
OPTIONAL_REPORTS = ("AUDIT_IMPLEMENTATION_REPAIR.json", "BACKUP_RECEIPT.json")
GENERATED_REPORTS = ("PUBLIC_MANIFEST.json", "PUBLICATION_AUDIT.json")
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".h5", ".h5ad", ".pkl", ".pickle", ".bin"}
SENSITIVE_JSON_KEYS = {
    "patient_id", "patient_ids", "donor_id_private", "section_id_private", "patient_source_field_verified",
    "barcodes", "barcode", "source_sample", "annotation_member", "image_filename_private",
    "X", "y", "features", "coords_um", "raw_labels", "weights", "state_dict", "model_state_dict",
    "oof_probabilities", "oof_predictions", "risk_vector", "gene_expression", "raw_counts",
}
CREDENTIAL_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    re.compile(r"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    re.compile(r'''(?i)(?:password|passwd|secret|api[_-]?key)\s*[:=]\s*["'][A-Za-z0-9/+_.=!-]{8,}["']'''),
    re.compile(r"(?i)https?://[^/\s:@]+:[^/\s@]+@"),
)
TCGA_PATTERN = re.compile(r"\bTCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}(?:-[A-Za-z0-9-]+)?\b")
MAX_PUBLIC_FILE_BYTES = 2 * 1024 * 1024


class PublicationAuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise PublicationAuditError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(text):
    def pairs(items):
        value = {}
        for k, v in items:
            require(k not in value, "duplicate JSON key")
            value[k] = v
        return value

    def invalid(_):
        raise PublicationAuditError("nonfinite JSON constant")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def approved_paths(report_names):
    names = list(report_names)
    require(len(names) == len(set(names)), "duplicate report path")
    require(set(REQUIRED_REPORTS).issubset(names), "required anonymous report missing from publication list")
    require(set(names).issubset(set(REQUIRED_REPORTS) | set(OPTIONAL_REPORTS)), "unapproved report path")
    # No globbing: every byte scanned must be bound to an explicit public path.
    return list(CORE_FILES) + [REPORT_ROOT + n for n in names]


def checked_public_path(root, relative):
    require(isinstance(relative, str) and "\\" not in relative and ":" not in relative,
            "noncanonical public path")
    p = PurePosixPath(relative)
    require(not p.is_absolute() and all(part not in {"", ".", ".."} for part in p.parts)
            and p.as_posix() == relative, "unsafe public path")
    require(p.suffix not in FORBIDDEN_SUFFIXES and p.suffix in {".py", ".json", ".md"},
            "forbidden public artifact type")
    require(not any("private" in part.lower() or part.lower() in {"raw", "weights"} for part in p.parts),
            "private path cannot be published")
    root = Path(root).resolve()
    path = root.joinpath(*p.parts)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root),
            "public file absent or path escaped")
    require(0 < path.stat().st_size <= MAX_PUBLIC_FILE_BYTES, "public file size outside allowed bounds")
    return path


def identity_denylist(prepared, manifest):
    """Load exactly the identity arrays, never feature, label, or risk arrays."""
    record = strict_json(Path(manifest).read_text(encoding="utf-8"))
    require(record.get("schema_version") == "sami_stage_a_prepared_v1"
            and record.get("status") == "PASS_PREPARED_FROZEN_TISSUE3", "unqualified private manifest")
    require(record.get("data_sha256") == sha(prepared), "private prepared SHA mismatch")
    required_sections = record.get("sections")
    require(isinstance(required_sections, list) and len(required_sections) == 12, "invalid private section manifest")
    identities, counts = set(), {}

    def add(value):
        require(isinstance(value, str) and len(value) >= 4, "invalid private identity field")
        identities.add(value)

    with np.load(prepared, allow_pickle=False) as arr:
        for key in ("donor", "section", "barcodes"):
            value = arr[key]
            require(value.ndim == 1 and value.dtype.kind in "US", "invalid private identity array")
            unique = set(value.astype(str))
            counts[key] = len(unique)
            for identity in unique:
                add(identity)
    require(counts["donor"] == 6 and counts["section"] == 12, "private identity coverage mismatch")
    for source in required_sections:
        # Only these exact fields are identities. Study names, normalizations,
        # revisions, label names and ordinary source properties are not IDs.
        for key in ("section_id_private", "donor_id_private", "patient_source_field_verified",
                    "source_sample", "annotation_member"):
            add(source[key])
        if "image_filename_private" in source:
            add(source["image_filename_private"])
        sample = source["source_sample"]
        if sample.startswith("Count_"):
            add(sample.removeprefix("Count_"))
        member = source["annotation_member"]
        prefix = "Pathology_SpotAnnotations/Pathologist_Annotations_"
        require(member.startswith(prefix) and member.endswith(".csv"), "invalid private annotation lineage")
        add(PurePosixPath(member).name)
        add(member[len(prefix):-4])
    require(isinstance(record.get("outer_donor_folds"), list) and len(record["outer_donor_folds"]) == 6,
            "invalid private donor folds")
    for fold in record["outer_donor_folds"]:
        add(fold["heldout_donor"])
        require(len(fold["train_donors"]) == 5, "invalid private training donor set")
        for donor in fold["train_donors"]:
            add(donor)
    public_counts = dict(unique_donor_count=counts["donor"], unique_section_count=counts["section"],
                         unique_barcode_count=counts["barcodes"])
    return identities, dict(identity_field_unique_counts=public_counts, exact_forbidden_strings=len(identities))


def inspect_json_structure(value):
    if isinstance(value, dict):
        for key, child in value.items():
            require(key not in SENSITIVE_JSON_KEYS, "sensitive spot-level JSON field")
            inspect_json_structure(child)
    elif isinstance(value, list):
        require(len(value) <= 128, "large numeric or per-case array is not an aggregate")
        for child in value:
            inspect_json_structure(child)


def inspect_payload(payload, suffix, identities):
    require(b"\x00" not in payload, "binary payload under text extension")
    text = payload.decode("utf-8-sig", errors="strict")
    hits = sum(1 for identity in identities if identity in text)
    require(hits == 0, "private identity string detected")
    require(TCGA_PATTERN.search(text) is None, "TCGA identifier detected")
    require(not any(pattern.search(text) for pattern in CREDENTIAL_PATTERNS), "credential pattern detected")
    if suffix == ".json":
        inspect_json_structure(strict_json(text))
    elif suffix == ".py":
        ast.parse(text)
    elif suffix != ".md":
        raise PublicationAuditError("unexpected text type")
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), identity_hits=0,
                credential_hits=0, TCGA_identifier_hits=0)


def audit_files(root, relative_paths, identities):
    require(len(relative_paths) == len(set(relative_paths)), "duplicate public file path")
    allowed = set(CORE_FILES) | {REPORT_ROOT + n for n in REQUIRED_REPORTS + OPTIONAL_REPORTS}
    require(set(relative_paths).issubset(allowed), "path outside exact publication whitelist")
    rows, failures = [], []
    for relative in relative_paths:
        try:
            path = checked_public_path(root, relative)
            evidence = inspect_payload(path.read_bytes(), path.suffix, identities)
            rows.append(dict(path=relative, **evidence))
        except (OSError, ValueError, SyntaxError, UnicodeError) as error:
            # Do not emit source text, decoded data, identity values or exceptions
            # with private paths. Only the approved public path and coarse class.
            reason = str(error) if isinstance(error, PublicationAuditError) else type(error).__name__
            failures.append(dict(path=relative, category=reason))
    return rows, failures


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def run(args):
    root = args.root.resolve()
    paths = approved_paths(args.reports)
    identities, identity_audit = identity_denylist(args.prepared, args.prepared_manifest)
    rows, failures = audit_files(root, paths, identities)
    output = root / REPORT_ROOT
    audit_path = output / "PUBLICATION_AUDIT.json"
    manifest_path = output / "PUBLIC_MANIFEST.json"
    require(not audit_path.exists() and not manifest_path.exists(), "publication receipts already exist")
    report = dict(schema_version="sami_stage_a_publication_audit_v1", status="FAIL" if failures else "PASS",
                  publication_performed=False, exact_allowlist=True, expected_files=len(paths), scanned_files=len(rows),
                  failures=failures, failed_files=len(failures), **identity_audit,
                  private_manifest_sha256=sha(args.prepared_manifest), private_prepared_sha256=sha(args.prepared),
                  arrays_accessed=["donor", "section", "barcodes"], feature_label_risk_arrays_accessed=False,
                  private_arrays_exported=False, raw_patients_or_weights_exported=False,
                  receipt_recursion_exclusion=list(GENERATED_REPORTS))
    if not failures:
        manifest = dict(schema_version="sami_stage_a_public_file_manifest_v1", status="PASS_CONTENT_AUDIT",
                        files=[{k: row[k] for k in ("path", "bytes", "sha256")} for row in rows],
                        recursion_excluded_receipts=list(GENERATED_REPORTS),
                        source_verifier_sha256=sha(root / "scripts/verify_sami_stage_a_publication.py"),
                        public_files_only=True)
        write_json_atomic(manifest_path, manifest)
        report["public_manifest_sha256"] = sha(manifest_path)
    write_json_atomic(audit_path, report)
    return report


class SyntheticPublicationTests(unittest.TestCase):
    def test_approved_default_list_is_finite(self):
        self.assertEqual(len(approved_paths(REQUIRED_REPORTS)), 20)

    def test_unknown_or_recursive_or_incomplete_reports_rejected(self):
        for names in (REQUIRED_REPORTS + ("raw.json",), REQUIRED_REPORTS + GENERATED_REPORTS,
                      REQUIRED_REPORTS[:-1], REQUIRED_REPORTS + ("RESULTS.json",)):
            with self.assertRaises(PublicationAuditError):
                approved_paths(names)

    def test_every_optional_report_passes_path_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            allowed = approved_paths(REQUIRED_REPORTS + OPTIONAL_REPORTS)
            for name in OPTIONAL_REPORTS:
                relative = REPORT_ROOT + name
                self.assertIn(relative, allowed)
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"synthetic_aggregate":true}', encoding="utf-8")
                self.assertEqual(checked_public_path(root, relative), path)
            old_name = REPORT_ROOT + "PRIVATE_BACKUP_RECEIPT.json"
            with self.assertRaisesRegex(PublicationAuditError, "private path"):
                checked_public_path(root, old_name)

    def test_valid_aggregate_and_source(self):
        inspect_payload(b'{"n_spots":13011,"class_recall":[0.5,0.6,0.7]}', ".json", {"secret-donor"})
        inspect_payload(b'labels_are_not_truth = True\n', ".py", {"secret-donor"})

    def test_exact_private_ids_blocked_without_echo(self):
        for payload in (b'{"cohort":"secret-donor"}', b'# secret-donor\n'):
            with self.assertRaises(PublicationAuditError) as error:
                inspect_payload(payload, ".md", {"secret-donor"})
            self.assertNotIn("secret-donor", str(error.exception))

    def test_credentials_and_TCGA_blocked(self):
        synthetic = ["hf_" + "q" * 25, "ghp_" + "q" * 30,
                     "TCGA-" + "ZZ-1234", "-----BEGIN " + "PRIVATE KEY-----",
                     "password=" + repr("synthetic-long-password")]
        for value in synthetic:
            with self.assertRaises(PublicationAuditError):
                inspect_payload(value.encode(), ".md", set())

    def test_sensitive_json_or_array_blocked(self):
        for value in ({"nested": {"barcodes": ["example"]}}, {"features": [1, 2]},
                      {"aggregate": list(range(129))}):
            with self.assertRaises(PublicationAuditError):
                inspect_payload(json.dumps(value).encode(), ".json", set())

    def test_json_duplicates_nonfinite_and_binary_fail(self):
        for value in (b'{"a":1,"a":2}', b'{"a":NaN}', b'\x00binary'):
            with self.assertRaises(PublicationAuditError):
                inspect_payload(value, ".json", set())

    def test_invalid_python_fails(self):
        with self.assertRaises(SyntaxError):
            inspect_payload(b'def incomplete(', ".py", set())

    def test_path_escape_binary_and_private_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for path in ("../source.py", "/source.py", "C:/source.py", "a\\source.py", "raw/source.py", "model.pt"):
                with self.assertRaises(PublicationAuditError):
                    checked_public_path(temp, path)

    def test_paths_outside_allowlist_fail_without_opening(self):
        with self.assertRaises(PublicationAuditError):
            audit_files(Path.cwd(), ["src/sami_surv/private.py"], set())

    def test_identity_arrays_only_and_normal_properties_not_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared.npz"
            donors = np.asarray([f"synthetic-donor-{i//2}" for i in range(12)])
            sections = np.asarray([f"synthetic-section-{i}" for i in range(12)])
            np.savez(prepared, donor=donors, section=sections,
                     barcodes=np.asarray([f"synthetic-barcode-{i}" for i in range(12)]),
                     X=np.asarray([object()], dtype=object), y=np.asarray([object()], dtype=object))
            rows = [dict(section_id_private=sections[i], donor_id_private=donors[i],
                         patient_source_field_verified=f"synthetic-patient-{i//2}", source_sample=f"Count_synthetic-sample-{i}",
                         annotation_member=f"Pathology_SpotAnnotations/Pathologist_Annotations_synthetic-sample-{i}.csv",
                         normal_property="normalization_is_not_an_identity") for i in range(12)]
            unique_donors = sorted(set(donors))
            manifest = root / "manifest.json"
            record = dict(schema_version="sami_stage_a_prepared_v1", status="PASS_PREPARED_FROZEN_TISSUE3",
                          data_sha256=sha(prepared), sections=rows,
                          outer_donor_folds=[dict(heldout_donor=d, train_donors=[x for x in unique_donors if x != d])
                                             for d in unique_donors])
            manifest.write_text(json.dumps(record), encoding="utf-8")
            deny, counts = identity_denylist(prepared, manifest)
            self.assertIn("synthetic-sample-0", deny)
            self.assertNotIn("normalization_is_not_an_identity", deny)
            self.assertEqual(counts["identity_field_unique_counts"]["unique_section_count"], 12)
            record["data_sha256"] = "wrong"
            manifest.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(PublicationAuditError, "SHA mismatch"):
                identity_denylist(prepared, manifest)

    def test_failure_output_does_not_echo_private_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            relative = REPORT_ROOT + "RESULTS.md"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text("secret-donor", encoding="utf-8")
            rows, failures = audit_files(root, [relative], {"secret-donor"})
            self.assertFalse(rows)
            self.assertEqual(len(failures), 1)
            self.assertNotIn("secret-donor", json.dumps(failures))
            self.assertEqual(failures[0]["path"], relative)

    def test_full_run_generated_receipts_pass_same_payload_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, private = base / "public-repo", base / "private-runtime"
            private.mkdir()
            prepared = private / "prepared.npz"
            donors = np.asarray([f"synthetic-donor-{i//2}" for i in range(12)])
            sections = np.asarray([f"synthetic-section-{i}" for i in range(12)])
            np.savez(prepared, donor=donors, section=sections,
                     barcodes=np.asarray([f"synthetic-barcode-{i}" for i in range(12)]),
                     X=np.asarray([object()], dtype=object), y=np.asarray([object()], dtype=object))
            unique_donors = sorted(set(donors))
            record = dict(schema_version="sami_stage_a_prepared_v1", status="PASS_PREPARED_FROZEN_TISSUE3",
                          data_sha256=sha(prepared), sections=[
                              dict(section_id_private=sections[i], donor_id_private=donors[i],
                                   patient_source_field_verified=f"synthetic-patient-{i//2}",
                                   source_sample=f"Count_synthetic-sample-{i}",
                                   annotation_member=f"Pathology_SpotAnnotations/Pathologist_Annotations_synthetic-sample-{i}.csv")
                              for i in range(12)],
                          outer_donor_folds=[dict(heldout_donor=d, train_donors=[x for x in unique_donors if x != d])
                                             for d in unique_donors])
            manifest = private / "manifest.json"
            manifest.write_text(json.dumps(record), encoding="utf-8")
            for relative in approved_paths(REQUIRED_REPORTS):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = "pass\n" if path.suffix == ".py" else "{}\n" if path.suffix == ".json" else "Synthetic aggregate\n"
                path.write_text(payload, encoding="utf-8")
            args = argparse.Namespace(root=root, prepared=prepared, prepared_manifest=manifest,
                                      reports=list(REQUIRED_REPORTS))
            result = run(args)
            self.assertEqual(result["status"], "PASS")
            deny, _ = identity_denylist(prepared, manifest)
            for filename in GENERATED_REPORTS:
                path = root / REPORT_ROOT / filename
                inspected = inspect_payload(path.read_bytes(), ".json", deny)
                self.assertEqual(inspected["identity_hits"], 0)
            self.assertEqual(result["identity_field_unique_counts"],
                             {"unique_donor_count": 6, "unique_section_count": 12, "unique_barcode_count": 12})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--reports", nargs="+", default=list(REQUIRED_REPORTS))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticPublicationTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.prepared is None or args.prepared_manifest is None:
        parser.error("--prepared and --prepared-manifest are required")
    try:
        result = run(args)
    except (OSError, ValueError, KeyError, UnicodeError):
        # Raw private-data exception messages and local source paths must not leak.
        print(json.dumps({"status": "FAIL_BEFORE_PUBLICATION", "publication_performed": False}))
        raise SystemExit(2)
    print(json.dumps({k: result[k] for k in ("status", "expected_files", "scanned_files", "failed_files", "publication_performed")}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

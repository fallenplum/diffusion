"""Audit a finite SAMI Stage B public file set; never stage, copy, or publish data.

Only exact donor/section/barcode arrays and the private provenance manifest are
read. External gene-symbol lists receive a narrowly scoped, hash-bound exception
to the aggregate-array limit. --self-test uses synthetic private inputs only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.verify_sami_stage_a_publication import (
    CREDENTIAL_PATTERNS, TCGA_PATTERN, PublicationAuditError,
    checked_public_path, require, sha, strict_json,
)

REPORT_ROOT = "reports/sami_surv_stage_b/"
PROGRAM_FILE = "configs/sami_surv_stage_b_programs_v1.json"
CONFIG_FILE = "configs/sami_surv_stage_b_v1.json"
SELF_FILE = "scripts/prepare_sami_stage_b_publication.py"
HELPER_FILE = "scripts/verify_sami_stage_a_publication.py"
HELPER_SHA256 = "b1715290c07a5918a9e6eca2b30668bfe9b4af09a93ed0dac26bf5d264ddf762"
CORE_FILES = (
    CONFIG_FILE, PROGRAM_FILE,
    "src/sami_surv/spatial_programs.py", "src/sami_surv/program_metrics.py",
    "scripts/audit_sami_stage_b_data.py", "scripts/prepare_sami_stage_b.py",
    "scripts/run_sami_stage_b.py", "scripts/audit_sami_stage_b_run.py", SELF_FILE,
    "tests/test_sami_stage_b_data.py", "tests/test_sami_spatial_programs.py",
    "tests/test_sami_program_metrics.py", "tests/test_sami_stage_b_runner.py",
    "tests/test_sami_stage_b_run_audit.py",
)
REQUIRED_REPORTS = (
    "PROTOCOL.md", "SOURCE_REPLAY_AUDIT.json", "DATA_FEASIBILITY.json", "DATA_AUDIT.json",
    "RESULTS.json", "RESULTS.md", "CHECKPOINT_AUDIT.json", "decision.json",
)
GENERATED_REPORTS = ("PUBLIC_MANIFEST.json", "PUBLICATION_AUDIT.json")
PROGRAM_ORDER = ("iCMS2", "iCMS3", "FAP_fibroblast_associated_top25",
                 "MARCO_macrophage_associated_top25")
PROGRAM_DIRECTION_COUNTS = ((308, 279), (74, 54), (25, 0), (25, 0))
RESULTS_FILE = REPORT_ROOT + "RESULTS.json"
QC_NAMES = {"library_size", "n_detected_genes"}
RESULT_ARMS = {"MEAN", "COMPOSITION", "ADDITIVE", "TISSUE_COND", "RANDOM_GATE",
               "WRONG_DONOR", "SPATIAL_SHUFFLE"}
SENSITIVE_KEYS = {
    "patient_id", "patient_ids", "donor_id", "donor_ids", "donor_id_private",
    "section_id", "section_ids", "section_id_private", "patient_source_field_verified",
    "barcodes", "barcode", "source_sample", "annotation_member", "image_filename_private",
    "X", "y", "neighbor_X", "tissue_y", "program_y", "features", "coords_um",
    "raw_labels", "weights", "state_dict", "model_state_dict", "coefficients",
    "oof_probabilities", "oof_predictions", "risk_vector", "gene_expression", "raw_counts",
    "library_size", "n_detected_genes", "neighbor_indices", "neighbor_counts",
}
# Absolute local paths do not belong in anonymous public payloads. Source code
# uses command-line arguments, relative repository paths, and synthetic fixtures.
LOCAL_PATH = re.compile(r"(?i)(?:\b[A-Z]:[/\\]|/(?:home|root|autodl-tmp|mnt|Users)/)")
GENE_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def approved_paths():
    return list(CORE_FILES) + [REPORT_ROOT + name for name in REQUIRED_REPORTS]


def identity_denylist(prepared, manifest):
    record = strict_json(Path(manifest).read_text(encoding="utf-8"))
    require(record.get("schema_version") == "sami_stage_b_prepared_v1"
            and record.get("status") == "PASS_PREPARED_SIGNED_PROGRAMS", "unqualified private manifest")
    digest = sha(prepared)
    require(record.get("data_sha256") == digest == record.get("prepared_data_sha256"),
            "private prepared SHA mismatch")
    identities, arrays = set(), {}
    with np.load(prepared, allow_pickle=False) as data:
        for key in ("donor", "section", "barcodes"):
            values = data[key]
            require(values.ndim == 1 and values.dtype.kind in "US", "invalid private identity array")
            arrays[key] = values.astype(str)
            for value in set(arrays[key]):
                require(len(value) >= 4, "invalid private identity field")
                identities.add(value)
    lengths = {len(x) for x in arrays.values()}
    require(len(lengths) == 1 and next(iter(lengths)) == record.get("n_spots"),
            "private identity row count mismatch")
    donors, sections = set(arrays["donor"]), set(arrays["section"])
    require(len(donors) == record.get("donors") == 6 and len(sections) == record.get("sections") == 12,
            "private identity coverage mismatch")
    pairs = list(zip(arrays["section"], arrays["barcodes"]))
    require(len(set(pairs)) == len(pairs), "duplicate within-section private identity")
    source_files = record.get("source_files")
    require(isinstance(source_files, list) and len(source_files) == 12, "invalid private source manifest")
    source_sections = set()
    for source in source_files:
        section, donor = source["section_id_private"], source["donor_id_private"]
        require(section in sections and donor in donors and section not in source_sections,
                "private source identity mismatch")
        source_sections.add(section)
        require(set(arrays["donor"][arrays["section"] == section]) == {donor},
                "cross-donor section identity mismatch")
        identities.update((section, donor))
    folds = record.get("outer_donor_folds")
    require(isinstance(folds, list) and len(folds) == 6, "invalid private donor folds")
    heldout = set()
    for fold in folds:
        donor = fold["heldout_donor"]
        require(donor in donors and donor not in heldout
                and len(fold["train_donors"]) == 5
                and set(fold["train_donors"]) == donors - {donor}, "invalid private fold identities")
        heldout.add(donor)
    counts = dict(unique_donor_count=len(donors), unique_section_count=len(sections),
                  unique_barcode_count=len(set(arrays["barcodes"])))
    return identities, record, dict(identity_field_unique_counts=counts,
                                    exact_forbidden_strings=len(identities))


def validate_gene_definition(value):
    require(value.get("schema_version") == "sami_surv_stage_b_programs_v1", "invalid gene-definition schema")
    programs = value.get("programs")
    require(isinstance(programs, list) and len(programs) == 4, "invalid external program count")
    require(value.get("program_order") == list(PROGRAM_ORDER), "invalid external program order")
    for program, name, counts in zip(programs, PROGRAM_ORDER, PROGRAM_DIRECTION_COUNTS):
        require(program.get("id") == name, "invalid external program identity")
        for field in ("genes", "up_genes", "down_genes"):
            genes = program.get(field)
            require(isinstance(genes, list)
                    and all(isinstance(g, str) and GENE_SYMBOL.fullmatch(g) for g in genes),
                    "external gene list must contain exact symbol strings")
        up, down = program["up_genes"], program["down_genes"]
        require((len(up), len(down)) == counts, "external program membership count mismatch")
        require(program["genes"] == up + down and len(set(up + down)) == len(up + down),
                "invalid external program membership")
        require(re.fullmatch(r"[0-9a-f]{64}", program.get("source_file_sha256", "")) is not None,
                "missing external program source SHA")
        for field in ("source_url", "source_table", "source_hypothesis"):
            require(isinstance(program.get(field), str) and len(program[field]) > 0,
                    "missing external program source field")
        limitations = program.get("limitations")
        require(isinstance(limitations, list) and 0 < len(limitations) <= 128
                and all(isinstance(x, str) and x for x in limitations),
                "invalid external program limitations")


def is_qc_location(location):
    return (len(location) == 6 and location[0] == "per_fold"
            and type(location[1]) is int and 0 <= location[1] < 6
            and location[2] == "arms" and location[3] in RESULT_ARMS
            and location[4:] == ("metrics", "qc_associations"))


def validate_qc_aggregate(value):
    """Accept only the four-program within-section association summary schema."""
    require(isinstance(value, dict) and set(value) == {"transform", "metric", "truth", "prediction"},
            "invalid aggregate QC schema")
    require(value["transform"] == "log1p"
            and value["metric"] == "within_section_Spearman_descriptive_only", "invalid aggregate QC metric")
    coverage_keys = {"candidate_groups", "eligible_groups", "below_minimum_n_groups",
                     "constant_truth_groups", "constant_prediction_groups"}
    group_keys = {"aggregate", "available_donor_description", "complete_donors", "per_program",
                  "complete_donors_per_program", "coverage"}

    def correlation(number):
        return number is None or (type(number) in (int, float) and np.isfinite(number)
                                  and -1.000000000001 <= number <= 1.000000000001)

    for name in ("truth", "prediction"):
        group = value[name]
        require(isinstance(group, dict) and set(group) == group_keys, "invalid aggregate QC group schema")
        require(correlation(group["aggregate"]) and correlation(group["available_donor_description"]),
                "invalid aggregate QC correlation")
        require(type(group["complete_donors"]) is int and group["complete_donors"] in (0, 1),
                "invalid per-fold QC donor count")
        for field in ("per_program", "complete_donors_per_program", "coverage"):
            require(isinstance(group[field], list) and len(group[field]) == 4,
                    "invalid aggregate QC four-program vector")
        require(all(correlation(v) for v in group["per_program"]), "invalid aggregate QC correlation vector")
        require(all(type(v) is int and v in (0, 1) for v in group["complete_donors_per_program"]),
                "invalid per-fold QC program donor counts")
        for coverage in group["coverage"]:
            require(isinstance(coverage, dict) and set(coverage) == coverage_keys,
                    "invalid aggregate QC coverage schema")
            require(all(type(v) is int and 0 <= v <= 12 for v in coverage.values()),
                    "invalid aggregate QC coverage count")
            require(coverage["candidate_groups"] == coverage["eligible_groups"]
                    + coverage["below_minimum_n_groups"] + coverage["constant_truth_groups"]
                    and coverage["constant_prediction_groups"] <= coverage["eligible_groups"],
                    "inconsistent aggregate QC coverage counts")


def inspect_json_structure(value, location=(), external_gene_lists=False, qc_aggregates=False):
    if isinstance(value, dict):
        exact_qc = qc_aggregates and is_qc_location(location)
        if exact_qc:
            require(set(value) == QC_NAMES, "invalid aggregate QC nuisance fields")
        for key, child in value.items():
            if exact_qc and key in QC_NAMES:
                validate_qc_aggregate(child)
            else:
                require(key not in SENSITIVE_KEYS, "sensitive spot-level JSON field")
            inspect_json_structure(child, location + (key,), external_gene_lists, qc_aggregates)
    elif isinstance(value, list):
        is_genes = (external_gene_lists and len(location) == 3 and location[0] == "programs"
                    and isinstance(location[1], int) and 0 <= location[1] < 4
                    and location[2] in {"genes", "up_genes", "down_genes"})
        require(is_genes or len(value) <= 128, "large numeric or per-case array is not an aggregate")
        for i, child in enumerate(value):
            inspect_json_structure(child, location + (i,), external_gene_lists, qc_aggregates)


def inspect_payload(payload, relative, identities, program_sha256=None):
    require(b"\x00" not in payload, "binary payload under text extension")
    text = payload.decode("utf-8-sig", errors="strict")
    require(not any(value in text for value in identities), "private identity string detected")
    require(TCGA_PATTERN.search(text) is None, "TCGA identifier detected")
    require(not any(pattern.search(text) for pattern in CREDENTIAL_PATTERNS), "credential pattern detected")
    require(LOCAL_PATH.search(text) is None, "private absolute path detected")
    digest = hashlib.sha256(payload).hexdigest()
    suffix = Path(relative).suffix
    if suffix == ".json":
        record = strict_json(text)
        special = relative == PROGRAM_FILE
        if special:
            require(program_sha256 is not None and digest == program_sha256,
                    "external gene definition SHA mismatch")
            validate_gene_definition(record)
        allow_qc = (relative == RESULTS_FILE and isinstance(record, dict)
                    and record.get("schema_version") == "sami_stage_b_results_v1")
        inspect_json_structure(record, external_gene_lists=special, qc_aggregates=allow_qc)
    elif suffix == ".py":
        ast.parse(text)
    else:
        require(suffix == ".md", "unexpected text type")
    return dict(bytes=len(payload), sha256=digest, identity_hits=0, credential_hits=0,
                TCGA_identifier_hits=0, private_path_hits=0)


def audit_files(root, paths, identities, program_sha256):
    require(len(paths) == len(set(paths)) and set(paths).issubset(approved_paths()),
            "path outside exact publication whitelist")
    rows, failures = [], []
    for relative in paths:
        try:
            path = checked_public_path(root, relative)
            rows.append(dict(path=relative, **inspect_payload(path.read_bytes(), relative, identities, program_sha256)))
        except (OSError, ValueError, SyntaxError, UnicodeError, KeyError) as error:
            reason = str(error) if isinstance(error, PublicationAuditError) else type(error).__name__
            failures.append(dict(path=relative, category=reason))
    return rows, failures


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def run(args):
    root = args.root.resolve()
    require(sha(REPO / HELPER_FILE) == HELPER_SHA256, "generic verifier dependency changed")
    require(sha(root / SELF_FILE) == sha(Path(__file__)), "publication verifier source differs from audited copy")
    identities, private, identity_audit = identity_denylist(args.prepared, args.prepared_manifest)
    config_path = checked_public_path(root, CONFIG_FILE)
    require(sha(config_path) == private["config_sha256"], "protocol SHA mismatch")
    config = strict_json(config_path.read_text(encoding="utf-8"))
    require(config.get("schema_version") == "sami_surv_stage_b_protocol_v1", "invalid protocol schema")
    require(config["data_contract"]["program_definition_sha256"] == private["program_definition_sha256"],
            "protocol and prepared program SHA mismatch")
    paths = approved_paths()
    rows, failures = audit_files(root, paths, identities, private["program_definition_sha256"])
    output = root / REPORT_ROOT
    for filename in GENERATED_REPORTS:
        require(not (output / filename).exists(), "publication receipts already exist")
    report = dict(schema_version="sami_stage_b_publication_audit_v1", status="FAIL" if failures else "PASS",
                  publication_performed=False, exact_allowlist=True, expected_files=len(paths), scanned_files=len(rows),
                  failures=failures, failed_files=len(failures), **identity_audit,
                  private_manifest_sha256=sha(args.prepared_manifest), private_prepared_sha256=sha(args.prepared),
                  config_sha256=private["config_sha256"], program_definition_sha256=private["program_definition_sha256"],
                  arrays_accessed=["donor", "section", "barcodes"], feature_label_expression_arrays_accessed=False,
                  private_arrays_exported=False, raw_patients_or_weights_exported=False,
                  external_gene_list_exception="exact_SHA_bound_program_config_and_three_named_membership_fields_only",
                  aggregate_QC_exception="RESULTS_per_fold_arms_metrics_qc_associations_exact_four_program_schema_only",
                  receipt_recursion_exclusion=list(GENERATED_REPORTS))
    outputs = {}
    if not failures:
        manifest = dict(schema_version="sami_stage_b_public_file_manifest_v1", status="PASS_CONTENT_AUDIT",
                        files=[{key: row[key] for key in ("path", "bytes", "sha256")} for row in rows],
                        recursion_excluded_receipts=list(GENERATED_REPORTS),
                        source_verifier_sha256=sha(root / SELF_FILE),
                        generic_path_and_credential_verifier_sha256=HELPER_SHA256, public_files_only=True)
        outputs["PUBLIC_MANIFEST.json"] = json_bytes(manifest)
        report["public_manifest_sha256"] = hashlib.sha256(outputs["PUBLIC_MANIFEST.json"]).hexdigest()
    outputs["PUBLICATION_AUDIT.json"] = json_bytes(report)
    # Receipts must pass the identical privacy rules before any receipt is saved.
    for filename, payload in outputs.items():
        inspect_payload(payload, REPORT_ROOT + filename, identities)
    output.mkdir(parents=True, exist_ok=True)
    for filename, payload in outputs.items():
        with (output / filename).open("xb") as stream:
            stream.write(payload)
    return report


def synthetic_fixture(base):
    root, private = base / "repo", base / "private-runtime"
    private.mkdir()
    donors = np.asarray([f"fixture-subject-{i // 2}" for i in range(12)])
    sections = np.asarray([f"synthetic-section-{i}" for i in range(12)])
    prepared = private / "prepared.npz"
    np.savez(prepared, donor=donors, section=sections,
             barcodes=np.asarray([f"synthetic-barcode-{i}" for i in range(12)]),
             X=np.asarray([object()], dtype=object), program_y=np.asarray([object()], dtype=object))
    for relative in approved_paths():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "pass\n" if path.suffix == ".py" else "{}\n" if path.suffix == ".json" else "Synthetic aggregate\n"
        path.write_text(text, encoding="utf-8")
    for relative in (PROGRAM_FILE, CONFIG_FILE, SELF_FILE):
        (root / relative).write_bytes((REPO / relative).read_bytes())
    unique = sorted(set(donors))
    record = dict(schema_version="sami_stage_b_prepared_v1", status="PASS_PREPARED_SIGNED_PROGRAMS",
                  data_sha256=sha(prepared), prepared_data_sha256=sha(prepared), n_spots=12, donors=6, sections=12,
                  config_sha256=sha(root / CONFIG_FILE), program_definition_sha256=sha(root / PROGRAM_FILE),
                  source_files=[dict(section_id_private=sections[i], donor_id_private=donors[i],
                                     normal_property="ordinary_source_attribute") for i in range(12)],
                  outer_donor_folds=[dict(heldout_donor=d, train_donors=[x for x in unique if x != d]) for d in unique])
    manifest = private / "manifest.json"
    manifest.write_bytes(json_bytes(record))
    return argparse.Namespace(root=root, prepared=prepared, prepared_manifest=manifest)


class SyntheticPublicationTests(unittest.TestCase):
    def test_allowlist_is_finite_and_complete(self):
        self.assertEqual(len(approved_paths()), 22)
        self.assertEqual(len(set(approved_paths())), 22)
        self.assertFalse(any(REPORT_ROOT + x in approved_paths() for x in GENERATED_REPORTS))

    def test_paths_outside_allowlist_fail_before_open(self):
        with self.assertRaises(PublicationAuditError):
            audit_files(REPO, ["reports/sami_surv_stage_b/raw.json"], set(), "unused")

    def test_actual_id_and_credential_never_echoed(self):
        for payload in ("synthetic-private-donor", "hf_" + "q" * 25, "TCGA-" + "ZZ-1234",
                        "E:" + "/local/file", "password=" + repr("synthetic-long-password")):
            with self.assertRaises(PublicationAuditError) as caught:
                inspect_payload(payload.encode(), "reports/example.md", {"synthetic-private-donor"})
            self.assertNotIn(payload, str(caught.exception))

    def test_row_level_fields_and_large_generic_arrays_rejected(self):
        for value in ({"barcodes": ["example"]}, {"program_y": [[1]]}, {"weights": [1]},
                      {"values": list(range(129))}, {"nested": {"neighbor_indices": []}}):
            with self.assertRaises(PublicationAuditError):
                inspect_payload(json_bytes(value), REPORT_ROOT + "RESULTS.json", set())

    def test_external_gene_exception_is_exact_path_and_hash(self):
        payload = (REPO / PROGRAM_FILE).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        inspect_payload(payload, PROGRAM_FILE, set(), digest)
        for path, expected in ((REPORT_ROOT + "RESULTS.json", digest), (PROGRAM_FILE, "wrong")):
            with self.assertRaises(PublicationAuditError):
                inspect_payload(payload, path, set(), expected)

    def test_external_gene_exception_does_not_allow_numeric_or_private_fields(self):
        for mutation in ("numeric", "sensitive", "unrelated_large", "duplicate_gene"):
            value = strict_json((REPO / PROGRAM_FILE).read_text(encoding="utf-8"))
            if mutation == "numeric":
                value["programs"][0]["up_genes"][0] = 123
            elif mutation == "sensitive":
                value["programs"][0]["barcodes"] = ["example"]
            elif mutation == "unrelated_large":
                value["programs"][0]["unapproved_data"] = list(range(129))
            else:
                value["programs"][0]["genes"][0] = value["programs"][0]["genes"][1]
            payload = json_bytes(value)
            with self.assertRaises(PublicationAuditError):
                inspect_payload(payload, PROGRAM_FILE, set(), hashlib.sha256(payload).hexdigest())

    def test_invalid_json_binary_and_source(self):
        for payload in (b'{"a":1,"a":2}', b'{"a":NaN}', b'\x00binary'):
            with self.assertRaises(PublicationAuditError):
                inspect_payload(payload, "example.json", set())
        with self.assertRaises(SyntaxError):
            inspect_payload(b"def incomplete(", "example.py", set())

    def test_identity_only_read_and_manifest_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            args = synthetic_fixture(Path(temp))
            deny, record, counts = identity_denylist(args.prepared, args.prepared_manifest)
            self.assertIn("fixture-subject-" + str(0), deny)
            self.assertNotIn("ordinary_source_attribute", deny)
            self.assertEqual(counts["identity_field_unique_counts"]["unique_barcode_count"], 12)
            record["prepared_data_sha256"] = "wrong"
            args.prepared_manifest.write_bytes(json_bytes(record))
            with self.assertRaisesRegex(PublicationAuditError, "SHA mismatch"):
                identity_denylist(args.prepared, args.prepared_manifest)

    def test_cross_donor_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args = synthetic_fixture(Path(temp))
            record = strict_json(args.prepared_manifest.read_text())
            record["source_files"][0]["donor_id_private"] = "fixture-subject-" + str(1)
            args.prepared_manifest.write_bytes(json_bytes(record))
            with self.assertRaisesRegex(PublicationAuditError, "cross-donor"):
                identity_denylist(args.prepared, args.prepared_manifest)

    def test_full_run_receipts_pass_same_payload_rules_and_no_recursion(self):
        with tempfile.TemporaryDirectory() as temp:
            args = synthetic_fixture(Path(temp))
            report = run(args)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["scanned_files"], 22)
            deny, _, _ = identity_denylist(args.prepared, args.prepared_manifest)
            for name in GENERATED_REPORTS:
                inspect_payload((args.root / REPORT_ROOT / name).read_bytes(), REPORT_ROOT + name, deny)
            manifest = strict_json((args.root / REPORT_ROOT / "PUBLIC_MANIFEST.json").read_text())
            self.assertEqual({x["path"] for x in manifest["files"]}, set(approved_paths()))
            with self.assertRaisesRegex(PublicationAuditError, "receipts already exist"):
                run(args)

    def test_missing_required_file_fails_without_success_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            args = synthetic_fixture(Path(temp))
            (args.root / REPORT_ROOT / "RESULTS.md").unlink()
            report = run(args)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["failed_files"], 1)
            self.assertFalse((args.root / REPORT_ROOT / "PUBLIC_MANIFEST.json").exists())

    def test_failure_report_never_contains_actual_private_id(self):
        with tempfile.TemporaryDirectory() as temp:
            args = synthetic_fixture(Path(temp))
            identity = "fixture-subject-" + str(0)
            (args.root / REPORT_ROOT / "RESULTS.md").write_text(identity)
            report = run(args)
            self.assertEqual(report["status"], "FAIL")
            self.assertNotIn(identity, json.dumps(report))

    def test_changed_protocol_fails_before_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            args = synthetic_fixture(Path(temp))
            with (args.root / CONFIG_FILE).open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(PublicationAuditError, "protocol SHA"):
                run(args)
            self.assertFalse((args.root / REPORT_ROOT / "PUBLICATION_AUDIT.json").exists())

    @staticmethod
    def qc_fixture():
        from src.sami_surv.program_metrics import evaluate_programs
        n = 80
        values = np.arange(n, dtype=float)
        target = np.stack([values, values[::-1], np.sin(values), np.cos(values)], axis=1)
        result = evaluate_programs(target, target * 0.8, np.full(n, "synthetic"),
                                   np.asarray(["section_a"] * 40 + ["section_b"] * 40),
                                   np.zeros(n, dtype=int), np.zeros(4), np.ones(4),
                                   library_size=100 + values, n_detected_genes=10 + values)
        return dict(schema_version="sami_stage_b_results_v1", per_fold=[
            dict(outer_index=0, arms={"MEAN": dict(metrics=result)})])

    def test_actual_metric_schema_aggregate_qc_is_allowed(self):
        record = self.qc_fixture()
        inspect_payload(json_bytes(record), RESULTS_FILE, set())

    def test_qc_exception_requires_exact_report_schema_and_location(self):
        record = self.qc_fixture()
        with self.assertRaises(PublicationAuditError):
            inspect_payload(json_bytes(record), REPORT_ROOT + "CHECKPOINT_AUDIT.json", set())
        record["schema_version"] = "other"
        with self.assertRaises(PublicationAuditError):
            inspect_payload(json_bytes(record), RESULTS_FILE, set())
        record = self.qc_fixture()
        record["unapproved"] = record["per_fold"][0]["arms"]["MEAN"]["metrics"]["qc_associations"]
        with self.assertRaises(PublicationAuditError):
            inspect_payload(json_bytes(record), RESULTS_FILE, set())

    def test_qc_exception_rejects_raw_lists_and_inexact_subschema(self):
        for mutation in ("raw_list", "extra_field", "wrong_metric", "array_in_scalar", "long_programs",
                         "string_in_correlation", "invalid_counter", "extra_nuisance"):
            record = self.qc_fixture()
            qc = record["per_fold"][0]["arms"]["MEAN"]["metrics"]["qc_associations"]
            if mutation == "raw_list":
                qc["library_size"] = [1, 2, 3]
            elif mutation == "extra_field":
                qc["library_size"]["raw_values"] = [1, 2, 3]
            elif mutation == "wrong_metric":
                qc["library_size"]["metric"] = "per_spot_measurement"
            elif mutation == "array_in_scalar":
                qc["library_size"]["truth"]["aggregate"] = [1, 2]
            elif mutation == "long_programs":
                qc["library_size"]["truth"]["per_program"] = [0.1] * 80
            elif mutation == "string_in_correlation":
                qc["library_size"]["truth"]["per_program"][0] = "not_a_metric"
            elif mutation == "invalid_counter":
                qc["library_size"]["truth"]["coverage"][0]["candidate_groups"] = 100
            else:
                qc["unknown_nuisance"] = {}
            with self.assertRaises(PublicationAuditError):
                inspect_payload(json_bytes(record), RESULTS_FILE, set())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-manifest", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticPublicationTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.prepared is None or args.prepared_manifest is None:
        parser.error("--prepared and --prepared-manifest are required")
    try:
        result = run(args)
    except (OSError, ValueError, KeyError, UnicodeError, TypeError):
        print(json.dumps(dict(status="FAIL_BEFORE_PUBLICATION", publication_performed=False)))
        raise SystemExit(2)
    print(json.dumps({key: result[key] for key in ("status", "expected_files", "scanned_files", "failed_files", "publication_performed")}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

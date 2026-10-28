"""R7 explicit anonymous publication, with separate protocol/final stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import bgb_en_publication as inherited

ARMS = ("ADD", "RAW", "NORM", "INDEP", "SLOT", "GATE1", "GATE2",
        "SLOT_WRONG_RNA", "GATE1_WRONG_RNA", "GATE1_WRONG_PATH", "GATE2_WRONG_RNA")
BASE = "reports/bgb_en_r7/"
CORE_SCRIPTS = tuple("scripts/bgb_en_r7_" + name + ".py" for name in
    ("common", "run", "replay", "schedule", "report", "qualify", "interface_audit", "publication", "delivery"))
CORE = CORE_SCRIPTS + ("src/bgb_en/r7_survival.py", "src/bgb_en/r7_training.py", "configs/bgb_en/r7_inner_v1.json")
CORE_TESTS = tuple("tests/test_bgb_en_r7_" + name + ".py" for name in
    ("survival", "engine", "report", "interface_audit", "delivery", "publication"))
EARLY_REPORTS = tuple(BASE + name for name in ("PROTOCOL.md", "PROTOCOL_LOCK.json", "REPRODUCE.md",
    "R7_0_REAL_DATA_QUALIFICATION.json", "R7_0_R6_INTERFACE_AUDIT.json", "R7_0_INTERFACE_TRAJECTORY.md",
    "R7_SCALE_CALIBRATION_AUDIT.json"))
POST_SCRIPTS = tuple("scripts/bgb_en_r7_" + name + ".py" for name in
    ("independent_review", "mechanism_review", "plot_diagnostics"))
POST_TESTS = tuple("tests/test_bgb_en_r7_" + name + ".py" for name in
    ("independent_review", "mechanism_review", "plot_diagnostics"))
FINAL_REPORTS = tuple(BASE + name for name in ("RESULTS.md", "R7_RESULTS.json", "decision.json",
    "R7_FOLD_METRICS.csv", "R7_COSTS.csv", "R7_LEARNING_CURVES.csv", "R7_PAIRED_COMPARISONS.csv",
    "R7_CHECKPOINT_REPLAY_AUDIT.json", "R7_DIRECT_PAIR_CREDITS.csv", "R7_SECONDARY_FOLD_METRICS.csv",
    "R7_POSTFIT_EXPORT_AUDIT.json", "R7_INDEPENDENT_REVIEW.json", "R7_INDEPENDENT_REVIEW.md",
    "R7_MECHANISM_REVIEW.json", "R7_MECHANISM_REVIEW.md", "R7_MECHANISM_REVIEW.csv",
    "R7_EXECUTION_LEDGER.json", "R7_RESOURCE_SUMMARY.json", "R7_CONCLUSION.md"))
DIAGNOSTIC_PARTS = tuple(BASE + f"R7_DIAGNOSTICS_{arm}_p{part}.json" for arm in ARMS for part in (0, 1, 2))
MECHANISM_PARTS = tuple(BASE + f"R7_MECHANISM_REVIEW_{arm}.csv" for arm in ARMS)
FIGURE_NAMES = ("interface_trajectory", "stage_Harrell", "stage_Uno60", "query_pairing_deltas",
                "interaction_risk_SD", "program_cancellation")
FIGURE_FILES = frozenset(BASE + "figures/" + name + ext for name in FIGURE_NAMES for ext in (".svg", ".png"))
FIGURE_TEXT = (BASE + "figures/FIGURE_README.md", BASE + "figures/R7_FIGURE_BINDINGS.json")
PROTOCOL_REQUIRED = frozenset(CORE + CORE_TESTS + EARLY_REPORTS)
FINAL_REQUIRED = PROTOCOL_REQUIRED | frozenset(POST_SCRIPTS + POST_TESTS + FINAL_REPORTS +
    DIAGNOSTIC_PARTS + MECHANISM_PARTS + FIGURE_TEXT) | FIGURE_FILES
ALLOWLIST = FINAL_REQUIRED
AUDIT, MANIFEST = BASE + "SOURCE_PUBLICATION_AUDIT.json", BASE + "PUBLICATION_MANIFEST.json"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def qualification_guard(root):
    lock = read(root / BASE / "PROTOCOL_LOCK.json")
    require = inherited.require
    require(lock.get("status") == "LOCKED_BEFORE_R7_FITS" and lock.get("planned_neural_fits") == 99
        and lock.get("planned_target_optimizer_updates") == 7920 and lock.get("teacher_refits") == 0
        and lock.get("source_refits") == 0 and lock.get("outer_test_access") is False,
        "R7_PUBLICATION_FINITE_BUDGET_CHANGED")
    caches = lock.get("calibration_receipt_sha256", {})
    expected_caches = {f"p{p}_seed{s}" for p in (0, 1, 2) for s in (31, 47, 83)}
    require(canonical({k:v for k,v in lock.items() if k != "contract_sha256"}) == lock["contract_sha256"]
        and len(lock.get("sources", {})) == 40 and set(caches) == expected_caches
        and all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in caches.values()),
        "R7_PUBLICATION_SCIENTIFIC_LOCK_CHANGED")
    for name, digest in lock["sources"].items():
        original = root / name
        require(not original.is_symlink() and original.resolve().is_relative_to(root)
            and inherited.sha256(original) == digest, "R7_PUBLICATION_LOCKED_SOURCE_BYTES_CHANGED")
    q = read(root / BASE / "R7_0_REAL_DATA_QUALIFICATION.json")
    require(q.get("status") == "PASS_R7_REAL_DATA_AND_READONLY_R6_TRANSFER"
        and q.get("contract_sha256") == lock["contract_sha256"] and len(q.get("records", [])) == 11
        and q.get("optimizer_instances") == q.get("optimizer_updates") == 0
        and q.get("distinct_train_RMS_calibrations") == 9 and q.get("scalar_scales_per_calibration") == 6
        and q.get("RAW_vs_R6_SCALARS_max_error") == 0
        and q.get("source_refits") == q.get("teacher_refits") == q.get("scalar_beta_refits") == 0,
        "R7_PUBLICATION_ACTUAL_REAL_DATA_QUALIFICATION_REQUIRED")
    require({r["arm"] for r in q["records"]} == set(ARMS)
        and all(r.get("repeated_forward_max_error") == 0 and r.get("optimizer_updates") == 0
                and set(r.get("own_evidence_invariants", {})) ==
                    ({"wrong_query", "mean_query", "interaction_off"} |
                     ({"wrong_path"} if r["arm"] not in ("ADD", "RAW") else set()))
                and all(v is True for v in r["own_evidence_invariants"].values()) for r in q["records"]),
        "R7_PUBLICATION_OWN_EVIDENCE_OR_SOURCE_ISOLATION_CHANGED")
    config = read(root / "configs/bgb_en/r7_inner_v1.json")
    require(inherited.sha256(root / "configs/bgb_en/r7_inner_v1.json") == lock["config_sha256"]
        and tuple(config["trainable_arms"]) == ARMS and config["primary_candidate"] == "GATE1"
        and config["GATE2_used_for_model_selection"] is False, "R7_PUBLICATION_RECIPE_CHANGED")
    return lock


def figure_guard(root, pattern):
    require = inherited.require
    base = root / BASE / "figures"
    binding = read(base / "R7_FIGURE_BINDINGS.json")
    require(binding.get("schema") == "bgb_en_r7_figure_bindings_v1"
        and binding.get("status") == "COMPLETE_6_ANONYMOUS_DIAGNOSTIC_FIGURES"
        and binding.get("figure_count") == 6, "R7_FIGURE_COUNT_OR_SCHEMA_CHANGED")
    generator = binding["generator"]
    require(generator["file"] == "scripts/bgb_en_r7_plot_diagnostics.py"
        and inherited.sha256(root / generator["file"]) == generator["sha256"], "R7_FIGURE_GENERATOR_CHANGED")
    diag_names = {Path(name).name for name in DIAGNOSTIC_PARTS}
    maps = dict(interface_trajectory={"R7_0_R6_INTERFACE_AUDIT.json"},
        stage_Harrell={"R7_RESULTS.json"}, stage_Uno60={"R7_RESULTS.json"},
        query_pairing_deltas={"R7_RESULTS.json"}, interaction_risk_SD={"R7_RESULTS.json"} | diag_names,
        program_cancellation={"R7_RESULTS.json"} | diag_names)
    expected = set().union(*maps.values())
    sources = binding.get("source_files", [])
    require(len(sources) == 35 and {s["file"] for s in sources} == expected,
            "R7_FIGURE_EXACT_35_SOURCE_BINDINGS_REQUIRED")
    for s in sources:
        require(Path(s["file"]).name == s["file"] and "/" not in s["file"] and "\\" not in s["file"],
                "R7_FIGURE_SOURCE_BASENAME_REQUIRED")
        p = root / BASE / s["file"]
        require(not p.is_symlink() and p.stat().st_size == s["bytes"] and inherited.sha256(p) == s["sha256"],
                "R7_FIGURE_SOURCE_BYTES_CHANGED")
    figures = binding.get("figures", [])
    require(len(figures) == 6 and {f["basename"] for f in figures} == set(FIGURE_NAMES),
            "R7_SIX_FIXED_FIGURES_REQUIRED")
    rows = []
    for fig in figures:
        require(len(fig["source_files"]) == len(maps[fig["basename"]])
            and set(fig["source_files"]) == maps[fig["basename"]]
            and fig.get("one_axes") is True and fig.get("pvalues") is False
            and fig.get("uncertainty_intervals") is False, "R7_FIGURE_SOURCE_OR_STATISTICAL_CONTRACT_CHANGED")
        require(len(fig["outputs"]) == 2 and {o["file"] for o in fig["outputs"]} ==
            {fig["basename"] + ".png", fig["basename"] + ".svg"}, "R7_PAIRED_PNG_SVG_REQUIRED")
        for item in fig["outputs"]:
            original = base / item["file"]
            require(not original.is_symlink() and original.resolve().parent == base.resolve(),
                    "R7_FIGURE_OUTSIDE_EXPLICIT_DIRECTORY")
            data = original.read_bytes()
            require(len(data) == item["bytes"] and len(data) <= 8 * 1024**2
                and hashlib.sha256(data).hexdigest() == item["sha256"], "R7_FIGURE_OUTPUT_BYTES_CHANGED")
            if original.suffix == ".png":
                require(data.startswith(b"\x89PNG\r\n\x1a\n"), "R7_INVALID_PNG")
            else:
                text = data.decode("utf-8"); xml = ET.fromstring(text)
                require(xml.tag.endswith("svg") and not any(n.tag.rsplit("}", 1)[-1].lower()
                    in ("script", "image", "foreignobject", "animate") for n in xml.iter()),
                    "R7_SVG_SCRIPT_OR_EMBEDDED_RASTER_FORBIDDEN")
                require(all(v.startswith("#") for n in xml.iter() for k, v in n.attrib.items()
                    if k.rsplit("}", 1)[-1].lower() == "href") and pattern.search(text) is None,
                    "R7_SVG_EXTERNAL_LINK_OR_PRIVATE_IDENTIFIER")
                require(not any(k.rsplit("}", 1)[-1].lower().startswith("on")
                    for n in xml.iter() for k in n.attrib) and "@import" not in text.lower(),
                    "R7_SVG_ACTIVE_STYLE_OR_EVENT_HANDLER")
                for match in re.finditer(r"url\(\s*([^)]*)\)", text, flags=re.I):
                    require(match.group(1).strip().strip("\"'").startswith("#"),
                            "R7_SVG_EXTERNAL_CSS_URL")
                for label, rule in inherited.PATTERNS.items():
                    require(rule.search(text) is None, "R7_FIGURE_PRIVACY_RULE_" + label)
            rows.append(dict(file=str(original.relative_to(root)).replace("\\", "/"),
                             bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return rows


def final_guard(root, lock):
    require = inherited.require
    result = read(root / BASE / "R7_RESULTS.json")
    require(result.get("status") == "COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS"
        and result.get("contract_sha256") == lock["contract_sha256"], "R7_FINAL_MATRIX_INCOMPLETE")
    from scripts.bgb_en_r7_report import validate_replay
    rp_path = root / BASE / "R7_CHECKPOINT_REPLAY_AUDIT.json"
    require(inherited.sha256(rp_path) == result["replay_sha256"], "R7_PUBLIC_REPLAY_BYTES_CHANGED")
    validate_replay(read(rp_path), lock)
    require(read(root / BASE / "decision.json") == result["decision"], "R7_FINAL_DECISION_CHANGED")
    expected = {Path(p).name for p in DIAGNOSTIC_PARTS}
    parts = result["diagnostic_index"]["parts"]
    require(len(parts) == 33 and {p["file"] for p in parts} == expected, "R7_33_DIAGNOSTIC_PARTS_REQUIRED")
    for p in parts:
        path = root / BASE / p["file"]
        require(inherited.sha256(path) == p["sha256"] and path.stat().st_size == p["bytes"]
                and len(read(path)["rows"]) == p["rows"] == 3, "R7_DIAGNOSTIC_BINDING_CHANGED")
    return result


def publish(root, denylist, stage):
    root = root.resolve()
    lock = qualification_guard(root)
    denied = inherited.validate_denylist(read(denylist))
    pattern = inherited.identifier_pattern(denied)
    required = FINAL_REQUIRED if stage == "complete" else PROTOCOL_REQUIRED
    if stage == "complete": final_guard(root, lock)
    snapshot = {str(p.relative_to(root)): inherited.sha256(p)
        for name in ("bgb_en", "bgb_en_r5", "bgb_en_r6")
        for p in (root / "reports" / name).rglob("*") if p.is_file()}
    previous = inherited.ALLOWLIST
    inherited.ALLOWLIST = ALLOWLIST | {AUDIT, MANIFEST}
    try:
        files = [inherited.scan_file(root, f, pattern) for f in sorted(required - FIGURE_FILES)]
        if stage == "complete": files += figure_guard(root, pattern)
        audit = dict(schema="bgb_en_r7_source_publication_audit_v1", status="PASS_EXPLICIT_ANONYMOUS_FILES_ONLY",
            stage=stage, contract_sha256=lock["contract_sha256"],
            formal_completion_claimed=stage == "complete", qualification_optimizer_updates=0,
            raw_patient_spot_arrays_coordinates_risks_models_credentials_included=False,
            identifier_group_counts={k + "_count": len(v) for k,v in denied["groups"].items()},
            scanner_sha256=inherited.sha256(Path(inherited.__file__)),
            files_count_before_audit=len(files), git_operations_performed=0)
        (root / AUDIT).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        files.append(inherited.scan_file(root, AUDIT, pattern)); files.sort(key=lambda f: f["file"])
        content = "".join(f["file"] + "\0" + f["sha256"] + "\0" + str(f["bytes"]) + "\n" for f in files)
        manifest = dict(schema="bgb_en_r7_publication_manifest_v1", status=audit["status"], stage=stage,
            scientific_run_state="COMPLETE_PRELOCKED_99_FIT_MATRIX" if stage == "complete" else
                                 "LOCKED_RECIPE_ELIGIBILITY_PASSED_COMPLETION_NOT_CLAIMED",
            contract_sha256=lock["contract_sha256"], files=files, files_count=len(files),
            total_bytes=sum(f["bytes"] for f in files), ordered_file_manifest_sha256=hashlib.sha256(content.encode()).hexdigest(),
            unknown_files_grant_publication_permission=False, raw_data_or_models_included=False,
            git_operations_performed=0)
        (root / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        inherited.scan_file(root, MANIFEST, pattern)
    finally: inherited.ALLOWLIST = previous
    after = {str(p.relative_to(root)): inherited.sha256(p)
        for name in ("bgb_en", "bgb_en_r5", "bgb_en_r6")
        for p in (root / "reports" / name).rglob("*") if p.is_file()}
    inherited.require(after == snapshot, "R7_PUBLICATION_CHANGED_HISTORICAL_REPORTS")
    return manifest


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--denylist", type=Path, required=True)
    parser.add_argument("--stage", choices=("protocol", "complete"), required=True)
    args = parser.parse_args()
    manifest = publish(args.root, args.denylist, args.stage)
    print(json.dumps(dict(status=manifest["status"], stage=args.stage, files=manifest["files_count"])))


if __name__ == "__main__": main()

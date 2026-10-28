"""Fail-closed publication of the seven anonymous, source-bound R6 plots."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


publisher = load("r6_figure_publication", ROOT / "scripts/bgb_en_r6_publication.py")
fixtures = load("r6_plot_fixtures", ROOT / "tests/bgb_en/test_r6_plot_diagnostics.py")
NO_IDENTIFIERS = re.compile(r"NEVER_A_REAL_IDENTIFIER_829187")


@pytest.fixture(scope="module")
def rendered_template(tmp_path_factory):
    temporary = tmp_path_factory.mktemp("r6_bound_figure_template")
    reports = fixtures.anonymous_reports.__wrapped__(temporary)
    root = temporary / "public"
    directory = root / "reports/bgb_en_r6"
    shutil.copytree(reports, directory)
    (root / "scripts").mkdir()
    shutil.copyfile(ROOT / "scripts/bgb_en_r6_plot_diagnostics.py", root / "scripts/bgb_en_r6_plot_diagnostics.py")
    fixtures.plot.render(directory, directory / "figures")
    return root


@pytest.fixture
def public_root(rendered_template, tmp_path):
    root = tmp_path / "public"
    shutil.copytree(rendered_template, root)
    return root


def binding_path(root):
    return root / "reports/bgb_en_r6/figures/R6_FIGURE_BINDINGS.json"


def read_binding(root):
    return json.loads(binding_path(root).read_text(encoding="utf-8"))


def write_binding(root, value):
    binding_path(root).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_valid_fixed_pairs_and_every_required_source_bound(public_root):
    rows = publisher.scan_bound_figures(public_root, NO_IDENTIFIERS)
    assert len(rows) == len({row["file"] for row in rows}) == 14
    assert {Path(row["file"]).suffix for row in rows} == {".svg", ".png"}


def test_external_xml_href_with_spaces_is_rejected_after_bytes_rebound(public_root):
    binding = read_binding(public_root)
    item = next(i for i in binding["figures"][0]["outputs"] if i["format"] == "SVG")
    path = public_root / "reports/bgb_en_r6/figures" / item["file"]
    content = path.read_bytes().replace(b"</svg>", b'<use href = "https://example.invalid/external.svg#node" /></svg>')
    # Confirm this valid XML spelling escaped the former regex-only check.
    assert re.search(r'(?:xlink:)?href=[\"\x27](?!#)', content.decode()) is None
    path.write_bytes(content); item.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    write_binding(public_root, binding)
    with pytest.raises(ValueError, match="SVG_EXTERNAL_HREF_FORBIDDEN"):
        publisher.scan_bound_figures(public_root, NO_IDENTIFIERS)


def test_duplicate_png_entry_does_not_count_as_exact_pair(public_root):
    binding = read_binding(public_root)
    binding["figures"][0]["outputs"].append(copy.deepcopy(binding["figures"][0]["outputs"][0]))
    write_binding(public_root, binding)
    with pytest.raises(ValueError, match="PAIRED_SVG_PNG_REQUIRED"):
        publisher.scan_bound_figures(public_root, NO_IDENTIFIERS)


@pytest.mark.parametrize("mutation", ("empty", "duplicate", "missing"))
def test_source_manifest_cannot_be_empty_duplicate_or_drop_a_source(public_root, mutation):
    binding = read_binding(public_root)
    if mutation == "empty":
        binding["source_files"] = []
    elif mutation == "duplicate":
        binding["source_files"].append(copy.deepcopy(binding["source_files"][0]))
    else:
        binding["source_files"].pop()
    write_binding(public_root, binding)
    with pytest.raises(ValueError, match="FIGURE_EXACT_NONEMPTY_SOURCE_SET_REQUIRED"):
        publisher.scan_bound_figures(public_root, NO_IDENTIFIERS)


def test_each_chart_must_use_its_fixed_source_mapping(public_root):
    binding = read_binding(public_root)
    binding["figures"][0]["source_files"] = ["R6_2_OMICS_RETENTION.json"]
    write_binding(public_root, binding)
    with pytest.raises(ValueError, match="FIGURE_PER_CHART_SOURCE_BINDING_CHANGED"):
        publisher.scan_bound_figures(public_root, NO_IDENTIFIERS)


def test_source_bytes_change_fails_even_figure_bytes_unchanged(public_root):
    path = public_root / "reports/bgb_en_r6/R6_SOURCE_CELL_METRICS.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="FIGURE_SOURCE_BYTES_OR_ALLOWED_PATH_CHANGED"):
        publisher.scan_bound_figures(public_root, NO_IDENTIFIERS)


def test_exact_risk_shards_supported_without_weakening_source_union(public_root):
    directory = public_root / "reports/bgb_en_r6"
    risk = json.loads((directory / "R6_RISK_DIAGNOSTICS.json").read_text(encoding="utf-8"))
    fixtures.risk_bundle(directory, risk)
    binding = read_binding(public_root)
    names = {"R6_RISK_DIAGNOSTICS.json"} | {"R6_RISK_DIAGNOSTICS_" + arm + ".json" for arm in publisher.DIAGNOSTIC_ARMS}
    binding["source_files"] = [row for row in binding["source_files"] if row["file"] not in names]
    for name in sorted(names):
        content = (directory / name).read_bytes()
        binding["source_files"].append(dict(file=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
    figure = next(row for row in binding["figures"] if row["basename"] == "risk_branch_SD")
    figure["source_files"] = sorted(names)
    write_binding(public_root, binding)
    assert len(publisher.scan_bound_figures(public_root, NO_IDENTIFIERS)) == 14


def test_audit_coverage_is_scalar_count_json_while_raw_identifier_arrays_stay_blocked():
    denied = dict(groups=dict(patient=['synthetic-patient'], donor=['synthetic-donor'],
        section=['synthetic-section'], barcodes=['synthetic-barcode-a','synthetic-barcode-b']))
    counts = publisher.identifier_group_counts(denied)
    assert counts == dict(patient_count=1, donor_count=1, section_count=1, barcodes_count=2)
    audit = dict(schema='bgb_en_r6_source_publication_audit_v1',
        status='PASS_EXPLICIT_ANONYMOUS_FILES_ONLY', actual_private_identifier_groups_scanned=counts,
        private_identifiers_coordinates_values_risks_and_weights_included=False)
    publisher.inherited.validate_public_json(audit)
    assert 'synthetic-' not in json.dumps(audit)
    for field in ('patient_ids', 'barcodes', 'risk_vector', 'RNA_matrix', 'state_dict'):
        with pytest.raises(publisher.inherited.PublicationError, match='private row-level JSON field'):
            publisher.inherited.validate_public_json({field:['synthetic-private-value']})


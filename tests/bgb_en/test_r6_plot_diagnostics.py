"""Anonymous report fixtures exercise aggregation, packaging and vector QA."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("r6_plot", ROOT / "scripts/bgb_en_r6_plot_diagnostics.py")
plot = importlib.util.module_from_spec(spec); spec.loader.exec_module(plot)
D = "d" * 64


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(value, denominator):
    return dict(Harrell=value, Harrell_num=value * denominator, Harrell_den=denominator,
        Uno60=value - .02, Uno_num=(value - .02) * denominator * 1.3, Uno_den=denominator * 1.3)


def attention_bundle(directory, value):
    logical = plot.canonical(value); descriptors = []
    for arm in plot.ARMS:
        indices = [i for i, row in enumerate(value["rows"]) if row["arm"] == arm]
        name = "R6_ATTENTION_DIAGNOSTICS_" + arm + ".json"
        digest = write(directory / name, dict(schema="bgb_en_r6_attention_diagnostics_arm_shard_v1", arm=arm,
            contract_sha256=D, logical_object_sha256=logical, rows=[value["rows"][i] for i in indices]))
        descriptors.append(dict(arm=arm, file=name, row_count=9, row_positions=indices,
            sha256=digest, bytes=(directory / name).stat().st_size))
    index = {k: v for k, v in value.items() if k != "rows"}
    index.update(storage_schema="bgb_en_r6_attention_diagnostics_sharded_index_v1", logical_object_sha256=logical,
        rows_count=90, rows_shards=descriptors)
    write(directory / "R6_ATTENTION_DIAGNOSTICS.json", index)


def risk_bundle(directory, value):
    descriptors = []
    for arm in plot.ARMS:
        name = "R6_RISK_DIAGNOSTICS_" + arm + ".json"
        digest = write(directory / name, dict(schema="bgb_en_r6_risk_diagnostic_arm_v1", arm=arm,
            contract_sha256=D, units=[r for r in value["units"] if r["arm"] == arm]))
        descriptors.append(dict(arm=arm, file=name, count=9, sha256=digest))
    write(directory / "R6_RISK_DIAGNOSTICS.json", dict(schema="bgb_en_r6_risk_diagnostic_index_v1",
        unit_count=90, parts=descriptors, logical_metadata={k: v for k, v in value.items() if k != "units"},
        logical_sha256=plot.canonical(value)))


@pytest.fixture
def anonymous_reports(tmp_path):
    directory = tmp_path / "anonymous"
    directory.mkdir()
    families, source_rows = [], []
    for fi, (arm, control) in enumerate(plot.FAMILIES):
        all_huber, all_rho = [], []
        for donor in range(6):
            for seed in (31, 47):
                hubers, rhos = [], []
                for pi, program in enumerate(plot.PROGRAMS):
                    huber = .2 + fi * .01 + donor * .005 + seed * .0001 + pi * .002
                    hubers.append(huber)
                    for tissue in range(3):
                        rho = .1 + fi * .004 + donor * .003 + seed * .0001 + pi * .002 + tissue * .001
                        rhos.append(rho)
                        source_rows.append(dict(arm=arm, control=control, heldout_donor=donor, seed=seed,
                            status="PASS_COMPLETE_AND_REPLAYED", program=program, tissue_class=tissue,
                            program_global_Huber=huber, Spearman=rho, eligible_for_within_tissue_rho=True))
                all_huber.append(statistics.mean(hubers)); all_rho.append(statistics.mean(rhos))
        families.append(dict(arm=arm, control=control, status="COMPLETE_12_LODO_UNITS", complete_replayed_units=12,
            program_Huber=statistics.mean(all_huber), within_tissue_Spearman=statistics.mean(all_rho)))
    source_sha = write(directory / "R6_1_SOURCE_SUPERVISION.json", dict(schema="bgb_en_r6_source_supervision_report_v1",
        status="COMPLETE_147_SOURCE_UNITS_AND_REPLAY", complete_family_summaries=families))
    with (directory / "R6_SOURCE_CELL_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(source_rows[0])); writer.writeheader(); writer.writerows(source_rows)
    rows, summaries = [], []
    for ri, representation in enumerate(plot.REPRESENTATIONS):
        scores = []
        for part, denominator in zip(plot.PARTITIONS, (1., 2., 5.)):
            score = metrics(.60 + ri * .003 + part * .04, denominator)
            scores.append(score)
            rows.append(dict(representation=representation, partition=part,
                status="REUSED_EXACT_PRIOR_ACTUAL_REPLAY" if representation.startswith("LEGACY_") else "COMPLETE_AND_REPLAYED",
                lambda_L2=10. if representation == "LEGACY_FULL_RNA20511_RIDGE" else 1., metrics=score))
        summaries.append(dict(representation=representation, status="COMPLETE", complete_partitions=3,
            Harrell=sum(x["Harrell_num"] for x in scores) / sum(x["Harrell_den"] for x in scores)))
    write(directory / "R6_2_OMICS_RETENTION.json", dict(schema="bgb_en_r6_retention_report_v1",
        status="COMPLETE_18_LOGICAL_CELLS_AND_REPLAY", rows=rows, summaries=summaries))
    arms = []
    for ai, arm in enumerate(plot.TARGET_ARMS):
        values = [dict(seed=seed, **metrics(.70 + ai * .001 + j * .002, 100 + j * 13)) for j, seed in enumerate(plot.SEEDS)]
        arms.append(dict(arm=arm, complete_cells=9, status="COMPLETE_NINE_CELLS", cross_partition_risk_ranking=False,
            per_seed=values, Harrell=statistics.mean(r["Harrell"] for r in values), Uno60=statistics.mean(r["Uno60"] for r in values)))
    target_sha = write(directory / "R6_4_INNER_MATRIX.json", dict(schema="bgb_en_r6_target_inner_report_v1",
        contract_sha256=D, source=dict(report_sha256=source_sha), actual_replay=dict(complete=True, sha256=D),
        counts={"PASS_FIT_AND_ACTUAL_REPLAY": 90, "PASS_ZERO_FIT_REFERENCE": 18},
        decision=dict(complete_source_and_target_evidence=True), primary_DSS=arms))
    units, attention_rows = [], []
    for ai, arm in enumerate(plot.ARMS):
        for part in plot.PARTITIONS:
            for sj, seed in enumerate(plot.SEEDS):
                main = .01 + ai * .001 + part * .0001 + sj * .00001
                cross = 0. if arm in ("B2", "B3") else .002 + ai * .0002 + part * .0001 + sj * .00001
                units.append(dict(arm=arm, partition=part, seed=seed, final_epoch=80,
                    cross_structurally_disabled=arm in ("B2", "B3"),
                    saved_scalar_distributions=dict(validation=dict(quantities=dict(main_delta=dict(sd=main), cross_delta=dict(sd=cross))))))
                attention_rows.append(dict(arm=arm, partition=part, seed=seed, fixed_validation_rows=8,
                    matched_wrong_patch_plus_null_TV=dict(n=8, mean=.02 + ai * .003 + part * .0005 + sj * .0001)))
    risk = dict(schema="bgb_en_r6_risk_diagnostics_v1", status="COMPLETE_SAVED_METADATA_DIAGNOSTICS",
        contract_sha256=D, report_sha256=target_sha, model_or_array_or_label_files_read=False, units=units,
        description="匿名数据保持非ASCII canonical兼容")
    write(directory / "R6_RISK_DIAGNOSTICS.json", risk)
    attention = dict(schema="bgb_en_r6_attention_diagnostics_v1", status="COMPLETE_90_FIRST8_INFERENCE_DIAGNOSTICS",
        contract_sha256=D, replay_audit_sha256=D, unique_probed_patients=24, repeated_patient_evaluations=720,
        independent_patient_count_is_not_720=True, fixed_first_valid_rows=8, rows=attention_rows,
        description="非ASCII逻辑对象")
    attention_bundle(directory, attention)
    return directory


def test_seven_same_figure_pairs_bind_sources_and_contain_only_static_vectors(anonymous_reports, tmp_path):
    output = tmp_path / "plots"
    result = plot.render(anonymous_reports, output)
    assert result["figure_count"] == 7
    assert {p.name for p in output.iterdir()} == {plot.MANIFEST_NAME, plot.README_NAME} | {
        name + "." + ext for name in plot.BASENAMES for ext in ("svg", "png")}
    assert result["rendering"]["svg_metadata_Date"] is None
    assert result["runtime"]["optimizer_fits"] == result["runtime"]["model_forwards"] == 0
    for source in result["source_files"]:
        assert source["sha256"] == hashlib.sha256((anonymous_reports / source["file"]).read_bytes()).hexdigest()
        assert Path(source["file"]).name == source["file"]
    for figure in result["figures"]:
        assert figure["one_axes"] and not figure["uncertainty_intervals"] and not figure["pvalues"]
        assert figure["axes"]["x_unit"] and figure["axes"]["y_unit"] and figure["axes"]["x_scale"] == "linear"
        for item in figure["outputs"]:
            path = output / item["file"]
            assert path.stat().st_size == item["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
            if item["format"] == "SVG":
                plot.validate_svg(path.read_bytes())
                tree = ET.fromstring(path.read_bytes())
                assert len([node for node in tree.iter() if node.attrib.get("id", "").startswith("axes_")]) == 1
                assert b"<dc:date>" not in path.read_bytes()
            else:
                with Image.open(path) as image:
                    assert image.format == "PNG" and image.width >= 1500 and image.height >= 1000
    text = (output / plot.MANIFEST_NAME).read_text(encoding="utf-8")
    assert str(tmp_path) not in text and "patient_ids" not in text
    with pytest.raises(ValueError, match="FIGURE_OUTPUTS_EXIST_NO_OVERWRITE"):
        plot.render(anonymous_reports, output)


def test_pooled_components_not_mean_cindex_and_donor_not_seed_dots(anonymous_reports):
    plans = plot.build_plans(plot.Reports(anonymous_reports))
    source = plans[0]
    assert len(source["groups"]) == 6
    assert all(len(values) == 1 for group in source["groups"] for values in group["values"])
    assert source["means"][0] == pytest.approx(.2 + 2.5 * .005 + 39 * .0001 + 1.5 * .002)
    retention = plans[2]
    assert retention["means"][0] == pytest.approx((.60 * 1 + .64 * 2 + .68 * 5) / 8)
    assert retention["means"][0] != pytest.approx(statistics.mean((.60, .64, .68)))
    assert "L2=10" in retention["labels"][7]
    harrell = plans[3]
    assert harrell["means"][1] == 0.
    assert harrell["means"][2] == pytest.approx(.001)
    assert len(harrell["groups"]) == 3


def test_partial_input_fails_before_output_created(anonymous_reports, tmp_path):
    path = anonymous_reports / "R6_4_INNER_MATRIX.json"
    value = json.loads(path.read_text()); value["actual_replay"]["complete"] = False; write(path, value)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="TARGET_MATRIX_NOT_COMPLETE_AND_REPLAYED"):
        plot.render(anonymous_reports, output)
    assert not output.exists()


def test_risk_bundle_nonascii_roundtrip_and_structural_zero_guard(anonymous_reports):
    path = anonymous_reports / "R6_RISK_DIAGNOSTICS.json"
    original = json.loads(path.read_text(encoding="utf-8")); risk_bundle(anonymous_reports, original)
    reports = plot.Reports(anonymous_reports)
    reconstructed, _ = plot.read_risk(reports)
    assert reconstructed == original
    plans = plot.build_plans(plot.Reports(anonymous_reports))
    assert plans[5]["groups"][1]["means"][:2] == [0., 0.]
    assert "linear scale" in plans[5]["xlabel"]
    original["units"][0]["saved_scalar_distributions"]["validation"]["quantities"]["cross_delta"]["sd"] = .001
    risk_bundle(anonymous_reports, original)
    with pytest.raises(ValueError, match="STRUCTURALLY_DISABLED_CROSS_SD_NONZERO"):
        plot.build_plans(plot.Reports(anonymous_reports))


def test_shard_bytes_and_source_duplicate_are_fail_closed(anonymous_reports):
    path = anonymous_reports / "R6_ATTENTION_DIAGNOSTICS_B7.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="ATTENTION_SHARD_HASH_CHANGED"):
        plot.build_plans(plot.Reports(anonymous_reports))
    # Check duplicate source rows separately, before the attention reader.
    source = anonymous_reports / "R6_SOURCE_CELL_METRICS.csv"
    lines = source.read_text(encoding="utf-8").splitlines()
    source.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_DUPLICATE_PROGRAM_TISSUE_ROW"):
        plot.build_plans(plot.Reports(anonymous_reports))


def test_svg_external_href_and_image_rejected_and_repeat_svg_bytes_stable(anonymous_reports, tmp_path):
    with pytest.raises(ValueError, match="SVG_EXTERNAL_HREF_FORBIDDEN"):
        plot.validate_svg(b'<svg xmlns="http://www.w3.org/2000/svg"><use href="https://example.invalid/a"/></svg>')
    with pytest.raises(ValueError, match="SVG_RASTER_IMAGE_FORBIDDEN"):
        plot.validate_svg(b'<svg xmlns="http://www.w3.org/2000/svg"><image/></svg>')
    item = plot.build_plans(plot.Reports(anonymous_reports))[3]
    contents = []
    for _ in range(2):
        figure = plot.draw(item)
        try:
            import io
            stream = io.BytesIO()
            figure.savefig(stream, format="svg", metadata={"Date": None, "Creator": "BG-BEN R6 diagnostic plotter"})
            contents.append(stream.getvalue())
        finally:
            plot.plt.close(figure)
    assert contents[0] == contents[1]


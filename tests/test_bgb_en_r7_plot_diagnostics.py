"""Synthetic-only R7 plots: arithmetic, missingness and vector/hash guards."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from PIL import Image
import pytest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location("r7_plot",ROOT/"scripts/bgb_en_r7_plot_diagnostics.py")
plot=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(plot)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def anonymous(tmp_path):
    root=tmp_path/"anonymous";rows=[]
    for part in plot.PARTITIONS:
        for sj,seed in enumerate(plot.SEEDS):
            for epoch in plot.EPOCHS:
                baseline=1.+part*.4+sj*.2
                factor=1.+epoch/80.*(1.+part*.2+sj*.1)
                quantities={name:dict(coordinate_RMS=baseline*(j+1)*(1.+(factor-1.)*(1.+j*.15)))
                    for j,name in enumerate(plot.SERIES) if name!="cross_risk_SD"}
                quantities["cross_risk_SD"]=baseline*.01*(1.+(factor-1.)*.45)
                rows.append(dict(partition=part,seed=seed,epoch=epoch,quantities=quantities))
    write(root/plot.INTERFACE,dict(schema="bgb_en_r7_readonly_r6_interface_audit_v1",status="PASS_9_R6_CELLS_36_STATE_FORWARDS",
        rows_count=36,actual_model_forwards=36,repeated_probes_not_independent_cohorts=True,
        validation_scores_computed=False,labels_used_for_probe_selection=False,new_optimizer_fits=0,new_backward_calls=0,rows=rows))
    primary={}
    for ai,arm in enumerate(("B0","B1",*plot.ARMS)):
        seeds=[]
        for sj,seed in enumerate(plot.SEEDS):
            h,u=.60+ai*.005+sj*.002,.58+ai*.004+sj*.003
            seeds.append(dict(seed=seed,Harrell=h,Uno60=u,Harrell_num=h*(100+sj),Harrell_den=100+sj,
                Uno_num=u*(120+sj),Uno_den=120+sj))
        primary[arm]=dict(arm=arm,complete_cells=9,status="COMPLETE_NINE_CELLS",per_seed=seeds,
            Harrell=statistics.mean(r["Harrell"] for r in seeds),Uno60=statistics.mean(r["Uno60"] for r in seeds))
    desc=[]
    for ai,arm in enumerate(plot.ARMS):
        for part in plot.PARTITIONS:
            models=[]
            for sj,seed in enumerate(plot.SEEDS):
                slots={}
                if arm!="ADD":
                    for j,key in enumerate(("slot_path_contributions","slot_gene_contributions")):
                        value=.05+ai*.01+part*.02+sj*.005+j*.1
                        if arm=="RAW" and part==0 and sj==0 and j==0:value=0.
                        if arm=="NORM" and part==0 and sj==0 and j==0:value=None
                        slots[key]=dict(absolute_sum_over_sum_absolute_mean=value,nonzero_absolute_sum_patients=0 if value is None else 10)
                models.append(dict(arm=arm,partition=part,seed=seed,
                    final_risk_diagnostics=dict(validation=dict(SD_cross_delta_over_offset=0. if arm=="ADD" else .2*ai+.01*part+.005*sj,
                        slot_contributions=slots)),checkpoint_diagnostics={str(e):dict(epoch=e) for e in (20,40,60,80)}))
            name=f"R7_DIAGNOSTICS_{arm}_p{part}.json";digest=write(root/name,dict(schema="bgb_en_r7_diagnostic_shard_v1",rows=models))
            desc.append(dict(file=name,sha256=digest,bytes=(root/name).stat().st_size,rows=3))
    pairs=[dict(candidate=a,reference=b,delta_Harrell=primary[a]["Harrell"]-primary[b]["Harrell"],
        delta_Uno60=primary[a]["Uno60"]-primary[b]["Uno60"]) for a,b in plot.PAIRS]
    write(root/plot.RESULTS,dict(schema="bgb_en_r7_complete_report_v1",status="COMPLETE_99_FITS_297_CHECKPOINT_REPLAYS",
        budget=dict(formal_neural_fits=99,formal_optimizer_updates=7920),decision=dict(complete_actual_checkpoint_replays=297),
        no_cross_partition_risk_ranking=True,primary_DSS=primary,paired_comparisons=pairs,
        diagnostic_index=dict(parts=desc,logical_cells=99,epochs_per_cell=[20,40,60,80])))
    return root


def rebind_shard(root,name,value):
    digest=write(root/name,value);result=read(root/plot.RESULTS)
    part=next(r for r in result["diagnostic_index"]["parts"] if r["file"]==name)
    part.update(sha256=digest,bytes=(root/name).stat().st_size);write(root/plot.RESULTS,result)


def test_all_six_pairs_bind_exact_35_sources_and_static_single_axes(anonymous,tmp_path):
    output=tmp_path/"figures";binding=plot.render(anonymous,output)
    assert binding["schema"]=="bgb_en_r7_figure_bindings_v1" and binding["figure_count"]==6
    assert binding["status"]=="COMPLETE_6_ANONYMOUS_DIAGNOSTIC_FIGURES"
    assert len(binding["source_files"])==35 and {r["file"] for r in binding["source_files"]}==plot.INPUTS
    assert set(p.name for p in output.iterdir())=={plot.README,plot.BINDING}|{n+"."+e for n in plot.BASENAMES for e in ("svg","png")}
    assert binding["rendering"]["svg_metadata_Date"] is None and binding["runtime"]["model_forwards"]==0
    for source in binding["source_files"]:
        assert source["sha256"]==hashlib.sha256((anonymous/source["file"]).read_bytes()).hexdigest()
    expected_sources=[[plot.INTERFACE],[plot.RESULTS],[plot.RESULTS],[plot.RESULTS],sorted([plot.RESULTS,*plot.SHARDS]),sorted([plot.RESULTS,*plot.SHARDS])]
    for i,figure in enumerate(binding["figures"]):
        assert figure["basename"]==plot.BASENAMES[i] and figure["source_files"]==expected_sources[i]
        assert figure["one_axes"] and not figure["uncertainty_intervals"] and not figure["pvalues"]
        assert figure["axes"]["y_scale"]==("log" if i==0 else "linear")
        for item in figure["outputs"]:
            path=output/item["file"];content=path.read_bytes()
            assert len(content)==item["bytes"] and hashlib.sha256(content).hexdigest()==item["sha256"]
            if item["format"]=="SVG":
                plot.validate_svg(content);xml=ET.fromstring(content)
                assert len([n for n in xml.iter() if n.attrib.get("id","").startswith("axes_")])==1
                assert b"<dc:date>" not in content
            else:
                with Image.open(path) as image:assert image.format=="PNG" and image.width>=1500 and image.height>=1000
    assert str(tmp_path) not in (output/plot.BINDING).read_text(encoding="utf-8")
    with pytest.raises(ValueError,match="PLOT_OUTPUT_EXISTS_NO_OVERWRITE"):plot.render(anonymous,output)


def test_interface_mean_then_normalize_and_log_has_no_pseudocount(anonymous):
    plans=plot.plans_from(plot.Reports(anonymous));trajectory=plans[0]
    rows=read(anonymous/plot.INTERFACE)["rows"]
    expected=statistics.mean(r["quantities"]["Base_query"]["coordinate_RMS"] for r in rows if r["epoch"]==80)/statistics.mean(
        r["quantities"]["Base_query"]["coordinate_RMS"] for r in rows if r["epoch"]==0)
    wrong=statistics.mean((1.+1.*(1.+p*.2+sj*.1)) for p in plot.PARTITIONS for sj in range(3))
    assert trajectory["series"][0]["values"][0]==1.
    assert trajectory["series"][0]["values"][-1]==pytest.approx(expected)
    assert abs(expected-wrong)>.01
    figure=plot.draw(trajectory)
    try:assert figure.axes[0].get_yscale()=="log"
    finally:plot.plt.close(figure)


def test_none_cancellation_is_not_zero_and_true_zero_is_retained(anonymous):
    plans=plot.plans_from(plot.Reports(anonymous));cancel=plans[-1]
    assert cancel["groups"][0]["values"][0]==[None]*9 and cancel["groups"][0]["means"][0] is None
    assert "P 0/9; G 0/9" in cancel["labels"][0]
    assert "P 8/9; G 9/9" in cancel["labels"][2]
    assert cancel["groups"][0]["values"][1][0]==0.
    assert len([v for v in cancel["groups"][0]["values"][1] if v is not None])==9
    assert plans[-2]["means"][0]==0. and all(v==0. for v in plans[-2]["groups"][0]["values"][0])
    assert plans[1]["labels"]==list(plot.STAGES) and len(plans[1]["groups"])==3
    assert plans[3]["groups"][0]["marker"]=="o" and plans[3]["groups"][1]["marker"]=="s"


@pytest.mark.parametrize("kind,code",(("log_zero","LOG_INTERFACE_ZERO_CANNOT_BE_REPLACED"),
    ("diag_sha","DIAGNOSTIC_SHARD_BYTES_OR_SHA_CHANGED"),("incomplete","COMPLETE_R7_ANONYMOUS_RESULTS_REQUIRED"),
    ("bad_cancellation","PROGRAM_CANCELLATION_RATIO_INVALID")))
def test_corrupt_input_stops_before_any_figures(anonymous,tmp_path,kind,code):
    if kind=="log_zero":
        path=anonymous/plot.INTERFACE;value=read(path);value["rows"][0]["quantities"]["Base_query"]["coordinate_RMS"]=0.;write(path,value)
    elif kind=="diag_sha":
        path=anonymous/plot.SHARDS[0];path.write_bytes(path.read_bytes()+b" ")
    elif kind=="incomplete":
        path=anonymous/plot.RESULTS;value=read(path);value["status"]="PARTIAL";write(path,value)
    else:
        name="R7_DIAGNOSTICS_RAW_p0.json";value=read(anonymous/name)
        value["rows"][0]["final_risk_diagnostics"]["validation"]["slot_contributions"]["slot_path_contributions"]["absolute_sum_over_sum_absolute_mean"]=1.01
        rebind_shard(anonymous,name,value)
    with pytest.raises(ValueError,match=code):plot.render(anonymous,tmp_path/"must_not_exist")
    assert not (tmp_path/"must_not_exist").exists()


def test_svg_rejects_raster_external_href_and_has_stable_hashsalt(anonymous):
    with pytest.raises(ValueError,match="SVG_EXTERNAL_HREF_FORBIDDEN"):
        plot.validate_svg(b'<svg xmlns="http://www.w3.org/2000/svg"><use href = "https://example.invalid/other" /></svg>')
    with pytest.raises(ValueError,match="SVG_RASTER_OR_ACTIVE_CONTENT_FORBIDDEN"):
        plot.validate_svg(b'<svg xmlns="http://www.w3.org/2000/svg"><image/></svg>')
    plan=plot.plans_from(plot.Reports(anonymous))[1];outputs=[]
    for _ in range(2):
        figure=plot.draw(plan)
        try:
            data=io.BytesIO();figure.savefig(data,format="svg",metadata={"Date":None,"Creator":"BG-BEN R7 diagnostic plotter"});outputs.append(data.getvalue())
        finally:plot.plt.close(figure)
    assert outputs[0]==outputs[1]


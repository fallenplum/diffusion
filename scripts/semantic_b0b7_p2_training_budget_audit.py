"""Read-only complete Phase-A curves and finite-budget diagnostics after P2.

Reads only source-bound Phase-A JSON receipts, never risk outputs or endpoints.
Full per-epoch scalar curves are ordinary UTF8 CSV with a lossless SHA index;
JSON summaries are readable aggregates, not encoded duplicate timeseries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(31,47,83)
CHECKPOINTS=(50,100,200,400)
MASTER_SHA="b9aae9582c78e80f26833b0e049956246f3c7d09723abe813a505acb543f949c"
MAX_BYTES=2*1024**2
MAX_CSV_TOTAL_ROWS=20000
EPSILON=1e-12  # numerical equality only, never a statistical/extension gate
DIRECTIONS=("image_to_gene","gene_to_image")
RETRIEVAL_FIELDS=("anchors","anchors_covered","coverage","mean_negative_count",
                  "mean_reciprocal_midrank","paired_gap_mean","top1_tie_averaged")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(8*1024**2),b""):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def safe_file(root,name,expected=None):
    p=PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and str(p)==name and "\\" not in name and ":" not in name,"UNSAFE_RELATIVE_ARTIFACT")
    base=Path(root).resolve();path=base.joinpath(*p.parts)
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base),"MISSING_OR_ESCAPING_ARTIFACT")
    require(not any(x.is_symlink() for x in path.parents if x.is_relative_to(base)),"SYMLINK_PARENT")
    if expected is not None:require(sha(path)==expected,"ARTIFACT_SHA_CHANGED")
    return path


def finite_scalar(value):
    require(value is None or (type(value) in (int,float,bool) and math.isfinite(value)),"NONFINITE_OR_NONSCALAR_CURVE_VALUE")
    return value


def hard_gap(validation):
    values=[validation["retrieval_hard"][d]["paired_gap_mean"] for d in DIRECTIONS]
    supported=[finite_scalar(x) for x in values if x is not None]
    return statistics.mean(supported) if supported else None


def flatten_epoch(source,epoch):
    row={k:source[k] for k in ("family","split","role","outer","inner","seed")}
    row.update(epoch=int(epoch["epoch"]),optimizer_step=int(epoch["optimizer_step"]))
    for key in ("train_program_loss","train_pair_loss","train_auxiliary_loss"):
        value=finite_scalar(epoch[key]);require(value is not None and value>=0,"INVALID_TRAINING_LOSS")
        row[key]=value
    gradients=list(epoch["gradient_norms"].values())
    require(gradients and all(v is not None and math.isfinite(v) and v>0 for v in gradients),"NONFINITE_OR_INACTIVE_PARAMETER_GRADIENT")
    row.update(active_parameter_elements=int(epoch["active_parameter_elements"]),
        gradient_norm_min=min(gradients),gradient_norm_max=max(gradients))
    v=epoch.get("validation")
    row["validation_available"]=v is not None
    row["validation_NA_reason"]=None if v is not None else ("OUTER_FIXED_REFIT_HAS_NO_VALIDATION" if source["role"]=="outer_final" else "NOT_A_PRELOCKED_CHECKPOINT")
    for field in ("auxiliary_loss","program_loss","pair_hard_loss","lambda_pair","median_prediction_to_target_sd_ratio"):
        row["val_"+field]=finite_scalar(v.get(field)) if v is not None else None
    for field in ("patient_equal_masked_huber","macro_program_pearson","macro_program_spearman",
                  "patient_equal_profile_pearson_secondary","patients_with_measurement","measured_target_cells","valid_pearson_programs","coverage_fraction"):
        row["val_"+field]=finite_scalar(v["program"].get(field)) if v is not None else None
    row["val_hard_gap_available_direction_mean"]=hard_gap(v) if v is not None else None
    row["val_hard_gap_supported_directions"]=sum(v["retrieval_hard"][d]["paired_gap_mean"] is not None for d in DIRECTIONS) if v is not None else None
    for kind in ("hard","random"):
        for direction in DIRECTIONS:
            for field in RETRIEVAL_FIELDS:
                row["val_"+kind+"_"+direction+"_"+field]=finite_scalar(v["retrieval_"+kind][direction].get(field)) if v is not None else None
    return row


def selection_epoch(receipt):
    values=[r for r in receipt["history"] if "validation" in r]
    minimum=min(r["validation"]["program"]["patient_equal_masked_huber"] for r in values)
    eligible=[r for r in values if r["validation"]["program"]["patient_equal_masked_huber"]<=minimum+receipt["program_huber_tolerance"]]
    return min(eligible,key=lambda r:(-(hard_gap(r["validation"]) if hard_gap(r["validation"]) is not None else -math.inf),r["epoch"]))["epoch"]


def change(first,last,*,lower_better):
    if first is None or last is None:return {"status":"NA_UNSUPPORTED_OR_NOT_RUN","raw_delta":None,"improvement":None,"relative_improvement":None}
    delta=last-first;improvement=-delta if lower_better else delta
    return {"status":"NUMERICALLY_FLAT" if abs(delta)<=EPSILON else ("IMPROVED_DIRECTION" if improvement>0 else "WORSE_DIRECTION"),
        "raw_delta":delta,"improvement":improvement,"relative_improvement":improvement/abs(first) if abs(first)>EPSILON else None}


def summarize_cell(source,receipt,*,receipt_sha):
    steps=receipt["optimizer_steps"];inner=source["role"]=="inner"
    require(steps in CHECKPOINTS and len(receipt["history"])==steps and
        [r["epoch"] for r in receipt["history"]]==list(range(1,steps+1)) and
        [r["optimizer_step"] for r in receipt["history"]]==list(range(1,steps+1)),"INCOMPLETE_OR_DUPLICATE_EPOCH_HISTORY")
    present={r["epoch"] for r in receipt["history"] if "validation" in r}
    if inner:
        require(steps==400 and present==set(CHECKPOINTS) and receipt["evaluate_valid"] and
                receipt["selected_epoch"]==selection_epoch(receipt),"INNER_CHECKPOINT_OR_SELECTION_CHANGED")
    else:
        require(not present and not receipt["evaluate_valid"] and receipt["valid_patients"]==0 and
                receipt["selected_epoch"]==receipt["fixed_epoch"]==steps,"OUTER_FIXED_REFIT_CONTAINS_VALIDATION")
    states=receipt["states"]
    require(len(states)==len({s["epoch"] for s in states}) and {s["epoch"] for s in states}=={e for e in CHECKPOINTS if e<=steps} and
            all(s["actual_replay"] and s["replay_max_error"]==0 for s in states),"RECORDED_CHECKPOINT_REPLAY_SET_CHANGED")
    checkpoint_rows={}
    # Every epoch is validated; only fixed summaries are retained in JSON.
    for epoch in receipt["history"]:
        row=flatten_epoch(source,epoch)
        if epoch["epoch"] in (100,200,400):checkpoint_rows[str(epoch["epoch"])]=row
    values=[r["validation"]["program"]["patient_equal_masked_huber"] for r in receipt["history"] if "validation" in r]
    raw_min=min(values) if values else None
    raw_best=[r["epoch"] for r in receipt["history"] if "validation" in r and r["validation"]["program"]["patient_equal_masked_huber"]==raw_min]
    result={**source,"receipt_sha256":receipt_sha,"optimizer_steps":steps,"selected_epoch":receipt["selected_epoch"],
        "history_records":len(receipt["history"]),"training_only":not receipt["evaluate_valid"],
        "recorded_saved_replays":len(states),"source_state_sha256":receipt["selected_checkpoint_sha256"],
        "program_huber_original_selection_tolerance":receipt["program_huber_tolerance"],
        "selected_epoch_at400":receipt["selected_epoch"]==400,"raw_Huber_minimum_epochs":raw_best,
        "raw_Huber_has_minimum_at400":400 in raw_best,"raw_Huber_strict_unique_minimum_at400":raw_best==[400],
        "fixed_checkpoints":checkpoint_rows,"changes":{},"budget_extension_automatically_triggered":False,
        "budget400_is_not_a_convergence_certificate":True}
    for start,end in ((100,200),(200,400)):
        a,b=checkpoint_rows.get(str(start)),checkpoint_rows.get(str(end));key=f"{start}_to_{end}"
        part={}
        for field,lower in (("train_program_loss",True),("train_pair_loss",True),("train_auxiliary_loss",True),
                           ("val_patient_equal_masked_huber",True),("val_hard_gap_available_direction_mean",False)):
            part[field]=change(a.get(field) if a else None,b.get(field) if b else None,lower_better=lower)
        improvement=part["val_patient_equal_masked_huber"]["improvement"]
        part["val_Huber_change_within_original_selection_band"]=(abs(improvement)<=receipt["program_huber_tolerance"]) if improvement is not None else None
        part["val_Huber_improvement_exceeds_original_selection_band"]=(improvement>receipt["program_huber_tolerance"]) if improvement is not None else None
        result["changes"][key]=part
    return result


class CsvShards:
    def __init__(self,out,*,byte_limit=MAX_BYTES,row_limit=MAX_CSV_TOTAL_ROWS):
        self.out=Path(out);self.byte_limit=byte_limit;self.row_limit=row_limit
        self.fields=None;self.buffer=None;self.rows=0;self.total=0;self.files=[];self.byte_count=0
    def encoded(self,row):
        stream=io.StringIO(newline="");writer=csv.DictWriter(stream,fieldnames=self.fields,lineterminator="\n")
        writer.writerow({k:("" if v is None else v) for k,v in row.items()})
        return stream.getvalue().encode("utf-8")
    def start(self):
        stream=io.StringIO(newline="");csv.writer(stream,lineterminator="\n").writerow(self.fields)
        self.buffer=io.BytesIO();self.buffer.write(stream.getvalue().encode("utf-8"))
        self.byte_count=self.buffer.tell();self.rows=0
    def append(self,row):
        require(all(v is None or type(v) in (str,int,float,bool) for v in row.values()),"CSV_NONSCALAR_VALUE")
        require(all(not isinstance(v,float) or math.isfinite(v) for v in row.values()),"CSV_NONFINITE_VALUE")
        if self.fields is None:
            self.fields=list(row);require(len(self.fields)<=150,"CSV_COLUMN_LIMIT");self.start()
        require(list(row)==self.fields,"CURVE_SCHEMA_CHANGED_ACROSS_CELLS")
        encoded=self.encoded(row)
        if self.rows and (self.rows+2>self.row_limit or self.byte_count+len(encoded)>self.byte_limit):self.flush();self.start()
        require(self.rows+2<=self.row_limit and self.byte_count+len(encoded)<=self.byte_limit,"SINGLE_CURVE_ROW_EXCEEDS_LIMIT")
        self.buffer.write(encoded);self.byte_count+=len(encoded);self.rows+=1;self.total+=1
    def flush(self):
        if not self.rows:return
        name=f"09_PHASE_A_FULL_HISTORY_{len(self.files)+1:03d}.csv";path=self.out/name
        with path.open("xb") as f:f.write(self.buffer.getvalue())
        self.files.append({"file":name,"sha256":sha(path),"bytes":self.byte_count,"data_records":self.rows,"total_rows_including_header":self.rows+1})
        self.rows=0
    def finish(self):self.flush();return self.files


def encoded_json(value):
    sys.path.insert(0,str(ROOT));from scripts import bgb_en_publication as old
    old.validate_public_json(value)
    raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=MAX_BYTES and not any(p.search(raw.decode()) for p in old.PATTERNS.values()),"PUBLIC_JSON_RULE_FAILED")
    return raw


def write_json(path,value):
    with Path(path).open("xb") as f:f.write(encoded_json(value))


def json_records(out,stem,records):
    chunks=[];current=[];indented_bytes=0
    for row in records:
        raw=encoded_json(row)
        row_bytes=len(raw)-1+2*raw.count(b"\n")
        fits=indented_bytes+row_bytes+2*(len(current)+1)+3<=MAX_BYTES and len(current)+1<=1000
        if not fits:
            require(bool(current),"SINGLE_JSON_RECORD_TOO_BIG");chunks.append(current);current=[row];indented_bytes=row_bytes
            require(row_bytes+5<=MAX_BYTES,"SINGLE_JSON_RECORD_TOO_BIG")
        else:current.append(row);indented_bytes+=row_bytes
    if current:chunks.append(current)
    files=[]
    for i,rows in enumerate(chunks):
        name=f"{stem}_{i+1:03d}.json";write_json(Path(out)/name,rows)
        files.append({"file":name,"sha256":sha(Path(out)/name),"records":len(rows)})
    return files


def grouped_summary(cells,keys):
    groups=defaultdict(list)
    for cell in cells:groups[tuple(cell[k] for k in keys)].append(cell)
    results=[]
    for group,items in sorted(groups.items()):
        frequency=Counter(x["selected_epoch"] for x in items)
        row={**dict(zip(keys,group)),"fits":len(items),"optimizer_steps":sum(x["optimizer_steps"] for x in items),
            "selected_epoch_frequency":{str(e):frequency[e] for e in CHECKPOINTS},
            "selected_epoch_at400_count":sum(x["selected_epoch_at400"] for x in items),
            "raw_Huber_unique_minimum_at400_count":sum(x["raw_Huber_strict_unique_minimum_at400"] for x in items)}
        for field in ("train_program_loss","val_patient_equal_masked_huber","val_hard_gap_available_direction_mean"):
            changes=[x["changes"]["200_to_400"][field]["improvement"] for x in items]
            valid=[v for v in changes if v is not None]
            row[field+"_200to400"]={"supported_fits":len(valid),"NA_fits":len(items)-len(valid),
                "mean_improvement":statistics.mean(valid) if valid else None,"min_improvement":min(valid) if valid else None,
                "max_improvement":max(valid) if valid else None,"improved_direction_count":sum(v>EPSILON for v in valid),
                "numerically_flat_count":sum(abs(v)<=EPSILON for v in valid),"worse_direction_count":sum(v < -EPSILON for v in valid)}
        row["val_Huber200to400_within_original_selection_band_count"]=sum(x["changes"]["200_to_400"]["val_Huber_change_within_original_selection_band"] is True for x in items)
        row["val_Huber200to400_improvement_exceeds_band_count"]=sum(x["changes"]["200_to_400"]["val_Huber_improvement_exceeds_original_selection_band"] is True for x in items)
        results.append(row)
    return results


def generate(private,lock_path,expected_lock_sha,output,*,allow_completed_formal_receipts=False):
    require(allow_completed_formal_receipts,"EXPLICIT_POST480_AUTHORIZATION_REQUIRED_BEFORE_INPUT_READ")
    require(sha(lock_path)==expected_lock_sha==MASTER_SHA,"MASTER_LOCK_CHANGED")
    master=read(lock_path)
    for name,digest in master["frozen_sources"].items():safe_file(ROOT,name,digest)
    require(not (Path(private)/"ACTIVE_SCHEDULER_OWNER.private.json").exists(),"FORMAL_SCHEDULER_STILL_ACTIVE")
    expected=20*len(master["family_readouts"])*len(SEEDS)
    require(expected==480,"FORMAL_SOURCE_CARDINALITY_CHANGED")
    # Verify every receipt before creating outputs. This reads no risk files.
    records=[]
    for outer in range(1,6):
        for inner in (*range(3),None):
            split=f"outer{outer}_"+("final" if inner is None else f"inner{inner}")
            role="outer_final" if inner is None else "inner"
            for family in master["family_readouts"]:
                for seed in SEEDS:
                    folder=Path(private)/"phase_a"/split/(family+"_seed"+str(seed))
                    path=safe_file(folder,"cell.private.json");receipt=read(path)
                    binding=read(safe_file(folder,"MATRIX_BINDING.private.json"))
                    require(receipt["status"]=="COMPLETE" and receipt["identity"]=="NEW_FIT" and receipt["DSS_OS_PFI_accessed"] is False and
                            receipt["master_lock_sha256"]==canonical(master["phase_a_lock"]) and
                            receipt["frozen_sources"]==master["phase_a_lock"]["frozen_sources"] and
                            binding["p2_lock_sha256"]==MASTER_SHA and binding["split"]==split and binding["family"]==family and
                            binding["seed"]==seed and binding["cell_receipt_sha256"]==sha(path),"ACTUAL_HISTORY_SOURCE_BINDING_CHANGED")
                    definition=master["phase_a_families"][family]
                    require(receipt["variant"]==definition["name"] and all(receipt["spec"][k]==v for k,v in definition.items() if k!="name"),
                            "ACTUAL_HISTORY_ARCHITECTURE_DEFINITION_CHANGED")
                    source={"family":family,"split":split,"role":role,"outer":outer,"inner":inner,"seed":seed}
                    summary=summarize_cell(source,receipt,receipt_sha=sha(path))
                    records.append((path,source,summary))
    require(len(records)==expected,"ALL480_SOURCE_RECEIPTS_REQUIRED")
    out=Path(output);require(not out.exists(),"NEW_OUTPUT_DIRECTORY_REQUIRED");out.mkdir(parents=True)
    writer=CsvShards(out);coverage=[];cells=[]
    for path,source,summary in records:
        require(sha(path)==summary["receipt_sha256"],"HISTORY_CHANGED_DURING_AUDIT")
        receipt=read(path);start=writer.total
        for epoch in receipt["history"]:writer.append(flatten_epoch(source,epoch))
        require(writer.total-start==summary["optimizer_steps"],"CURVE_COVERAGE_COUNT_CHANGED")
        require(sha(path)==summary["receipt_sha256"],"HISTORY_CHANGED_WHILE_EXPORTING")
        coverage.append({**source,"source_receipt_sha256":summary["receipt_sha256"],"curve_start_offset":start,"curve_records":writer.total-start})
        cells.append(summary)
    csv_files=writer.finish()
    require(writer.total==sum(x["optimizer_steps"] for x in cells)==sum(x["data_records"] for x in csv_files),"INCOMPLETE_EPOCH_EXPORT")
    cell_files=json_records(out,"09_TRAINING_BUDGET_CELLS",cells)
    index={"schema":"semantic_p2_complete_epoch_curve_index_v1","status":"COMPLETE_100_PERCENT_ACTUAL_EPOCH_SCALAR_CURVES",
        "master_lock_sha256":MASTER_SHA,"audit_source_sha256":sha(__file__),"source_fit_count":len(cells),
        "actual_epoch_records":writer.total,"CSV_NA_representation":"empty_cell_with_explicit_validation_available_and_NA_reason",
        "floats":"plain_CSV_python_roundtrip_decimal_no_quantization_or_sampling","curve_shards":csv_files,
        "source_coverage":coverage,"JSON_cell_summary_shards":cell_files,"all_source_gradients_finite_nonzero":True,
        "curve_scope":"all_epochs_all_training_losses_and_global_validation_scalar_metrics_no_sampling",
        "curve_format":"complete_ordinary_UTF8_CSV_with_readable_JSON_SHA_index_not_duplicate_encoded_arrays",
        "outer_risk_files_or_outcome_arrays_read":False,"new_model_forwards":0,"optimizer_updates_run_by_auditor":0}
    write_json(out/"09_TRAINING_BUDGET_CURVE_INDEX.json",index)
    summary={"schema":"semantic_p2_training_budget_summary_v1","status":"COMPLETE_FIXED_BUDGET_EXECUTION_AND_TREND_AUDIT_NOT_CONVERGENCE_CERTIFICATE",
        "master_lock_sha256":MASTER_SHA,"audit_source_sha256":sha(__file__),"actual_formal_source_fits":len(cells),
        "actual_formal_optimizer_steps":writer.total,"inner_source_fits":sum(x["role"]=="inner" for x in cells),
        "outer_final_source_fits":sum(x["role"]=="outer_final" for x in cells),
        "family_scope":grouped_summary(cells,("family","role")),"family_seed_scope":grouped_summary(cells,("family","role","seed")),
        "numerical_equality_epsilon":EPSILON,"Huber_flat_band":"original_train_median_Huber_times0.01_selection_tolerance_not_a_new_significance_threshold",
        "selected400_and_raw_Huber_min400_are_different_and_reported_separately":True,
        "checkpoint_replays_are_source_receipt_markers_not_new_weight_loading_or_model_forward":True,
        "outer_final_has_no_validation_or_outer_score_used_to_judge_epoch":True,
        "all_primary_fits_at_locked_epoch_budget_does_not_prove_sufficient_training":True,
        "automatic_extension_or_architecture_search":False,"outer_risk_files_or_outcome_arrays_read":False,
        "separate_P0_P1R_E0_or_two_fit_leakage_qualifier_not_counted_as_formal_sources":True,
        "confidence_intervals_computed":False,"p_values_computed":False,"new_neural_forwards":0,"new_optimizer_updates":0}
    write_json(out/"09_TRAINING_BUDGET_AUDIT.json",summary)
    return {"status":summary["status"],"actual_source_fits":len(cells),"actual_epoch_records":writer.total,
        "full_curve_CSV_shards":len(csv_files),"summary_sha256":sha(out/"09_TRAINING_BUDGET_AUDIT.json"),
        "curve_index_sha256":sha(out/"09_TRAINING_BUDGET_CURVE_INDEX.json")}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("private","lock","expected-lock-sha256","output-newdir"):p.add_argument("--"+key,required=True)
    p.add_argument("--allow-completed-formal-receipts",action="store_true");a=p.parse_args()
    print(json.dumps(generate(a.private,a.lock,a.expected_lock_sha256,a.output_newdir,
        allow_completed_formal_receipts=a.allow_completed_formal_receipts),sort_keys=True))


if __name__=="__main__":main()

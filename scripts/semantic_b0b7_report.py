"""Read-only P1 aggregate reporting and exact-byte public lossless packaging.

No model or tensor loader is imported. Input must already have passed the
server's original protected-identifier scan. Every shard is ordinary nested
JSON, so the original guard can inspect every field directly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


LIMIT = 2 * 1024 * 1024
ARMS = ("B3", "B4", "B6a", "B4_MAP_SHUFFLE")
SEEDS = (31, 47, 83)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def original_key_order_overrides(value):
    """Preserve source order where native numeric keys sorted before JSON.

    JSON turns integer dictionary keys into strings. A source map sorted as
    50/100/200/400 will sort differently after parsing. Explicit key order
    overrides preserve the original bytes without changing any JSON value.
    """
    overrides = []
    def visit(item, path):
        if isinstance(item, dict):
            keys = list(item)
            if keys != sorted(keys):
                overrides.append({"path": path, "keys": keys})
            for key, child in item.items():
                visit(child, [*path, key])
        elif isinstance(item, list):
            for ordinal, child in enumerate(item):
                visit(child, [*path, ordinal])
    visit(value, [])
    return overrides


def original_pretty_bytes(value, overrides):
    lookup = {tuple(x["path"]): x["keys"] for x in overrides}
    def ordered(item, path):
        if isinstance(item, dict):
            keys = lookup.get(tuple(path), sorted(item))
            if len(keys) != len(item) or set(keys) != set(item):
                raise ValueError("ORIGINAL_KEY_ORDER_SUPPORT_MISMATCH")
            return {key: ordered(item[key], [*path, key]) for key in keys}
        if isinstance(item, list):
            return [ordered(child, [*path, ordinal]) for ordinal, child in enumerate(item)]
        return item
    return (json.dumps(ordered(value, []), sort_keys=False, ensure_ascii=False,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_once(path, content):
    path = Path(path)
    if len(content) >= LIMIT:
        raise ValueError("PUBLIC_FILE_MUST_BE_SMALLER_THAN_2MIB:" + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError("EXISTING_PUBLIC_ARTIFACT_BYTES_DIFFER:" + path.name)
    else:
        with path.open("xb") as stream:
            stream.write(content)
    return {"file": path.name, "bytes": len(content), "sha256": digest(content)}


def lossless_package(raw, directory):
    """One readable JSON shard per fitted cell, plus remaining source fields."""
    source = json.loads(raw.decode("utf-8-sig"))
    normalized = canonical(source)
    order_overrides = original_key_order_overrides(source)
    if original_pretty_bytes(source, order_overrides) != raw:
        raise ValueError("SOURCE_DOES_NOT_USE_THE_PRELOCKED_CANONICAL_PRETTY_UTF8_FORMAT")
    directory = Path(directory)
    parts = []
    for ordinal, cell in enumerate(source["cells"]):
        item = {"schema": "semantic_b0b7_public_lossless_cell_v1", "ordinal": ordinal,
                "cell": cell}
        name = "04_PRETEXT_LOSSLESS_CELL_" + cell["arm"] + "_seed" + str(cell["seed"]).zfill(3) + ".json"
        record = write_once(directory / name, json_bytes(item))
        record.update(ordinal=ordinal, cell_canonical_sha256=digest(canonical(cell)),
                      arm=cell["arm"], seed=cell["seed"])
        parts.append(record)
    index = {"schema": "semantic_b0b7_public_lossless_index_v1", "encoding": "ordinary-structured-json-cells",
             "source_utf8_bytes": len(raw), "source_utf8_sha256": digest(raw),
             "canonical_json_bytes": len(normalized), "canonical_json_sha256": digest(normalized),
             "source_without_cells": {k: v for k, v in source.items() if k != "cells"},
             "source_key_order_overrides": order_overrides,
             "parts": parts, "all_original_json_fields_preserved": True,
             "original_whitespace_and_utf8_bytes_preserved": True,
             "serialization": {"original_native_sort_keys": True,
                               "reconstruct_original_key_order_then_dump_sort_keys": False,
                               "indent": 2, "ensure_ascii": False,
                               "allow_nan": False, "encoding": "utf-8", "terminal_newline": "LF"},
             "publisher_can_apply_original_guard_to_all_structured_fields": True,
             "restore_function": "scripts/semantic_b0b7_report.py:restore_public_report",
             "no_patient_array_or_weight_conversion_performed": True}
    name = "04_PRETEXT_LOSSLESS_INDEX.json"
    index_record = write_once(directory / name, json_bytes(index))
    restored, restored_bytes = restore_public_report(directory / name)
    if restored_bytes != raw or canonical(restored) != normalized:
        raise ValueError("LOSSLESS_PACKAGE_ACTUAL_ROUNDTRIP_FAILURE")
    return index, [*parts, index_record]


def restore_public_report(index_path):
    """Verify every plain JSON shard and reconstructed original/canonical SHA."""
    path = Path(index_path)
    index = json.loads(path.read_text(encoding="utf-8"))
    if index.get("schema") != "semantic_b0b7_public_lossless_index_v1":
        raise ValueError("UNRECOGNIZED_LOSSLESS_INDEX")
    if not 1 <= index["source_utf8_bytes"] <= 64 * 1024 * 1024:
        raise ValueError("UNSAFE_SOURCE_SIZE")
    parts = index["parts"]
    if [p["ordinal"] for p in parts] != list(range(len(parts))):
        raise ValueError("LOSSLESS_PART_ORDER")
    cells = []
    for record in parts:
        relative = Path(record["file"])
        if relative.name != str(relative) or relative.is_absolute():
            raise ValueError("LOSSLESS_PART_PATH_ESCAPE")
        content = (path.parent / relative).read_bytes()
        if len(content) != record["bytes"] or digest(content) != record["sha256"]:
            raise ValueError("LOSSLESS_PART_FILE_HASH")
        item = json.loads(content.decode("utf-8"))
        if (item.get("schema") != "semantic_b0b7_public_lossless_cell_v1"
                or item.get("ordinal") != record["ordinal"]):
            raise ValueError("LOSSLESS_PART_IDENTITY")
        cell = item["cell"]
        if (digest(canonical(cell)) != record["cell_canonical_sha256"]
                or cell["arm"] != record["arm"] or cell["seed"] != record["seed"]):
            raise ValueError("LOSSLESS_CELL_IDENTITY_OR_CANONICAL_HASH")
        cells.append(cell)
    value = {**index["source_without_cells"], "cells": cells}
    raw = original_pretty_bytes(value, index["source_key_order_overrides"])
    if len(raw) != index["source_utf8_bytes"] or digest(raw) != index["source_utf8_sha256"]:
        raise ValueError("LOSSLESS_ORIGINAL_BYTES_HASH")
    normalized = canonical(value)
    if len(normalized) != index["canonical_json_bytes"] or digest(normalized) != index["canonical_json_sha256"]:
        raise ValueError("LOSSLESS_CANONICAL_HASH")
    return value, raw


def validate_aggregate(report):
    if (report.get("schema") != "semantic_b0b7_p1_public_aggregate_v1"
            or report.get("status") != "P1_EVALUATED" or report.get("neural_fits") != 12
            or report.get("decoder_fits") != 3 or report.get("dss_models_fitted") != 0
            or report.get("dss_scores_selected") != 0 or report.get("P2_started") is not False):
        raise ValueError("NOT_THE_LOCKED_OUTCOME_FREE_P1_AGGREGATE")
    cells = report["cells"]
    if len(cells) != 12 or {(c["arm"], c["seed"]) for c in cells} != {(a, s) for a in ARMS for s in SEEDS}:
        raise ValueError("INCOMPLETE_4_BY_3_MATRIX")
    steps, states = 0, 0
    for cell in cells:
        checkpoints = {int(k): v for k, v in cell["checkpoint_validation"].items()}
        expected = {50, 100, 200, 400} | ({800} if cell["optimizer_steps"] == 800 else set())
        if set(checkpoints) != expected or cell["optimizer_steps"] not in (400, 800):
            raise ValueError("CHECKPOINT_BUDGET_MISMATCH")
        selected = min(checkpoints, key=lambda e: (checkpoints[e]["auxiliary_loss"], e))
        if (cell["selection_rule"] != "validation_Laux_then_lower_epoch_tie"
                or cell["selected_epoch"] != selected
                or canonical(cell["selected_validation"]) != canonical(checkpoints[selected])):
            raise ValueError("AUXILIARY_CHECKPOINT_SELECTION_MISMATCH")
        if cell["actual_replayed_states"] != len(checkpoints):
            raise ValueError("ACTUAL_REPLAY_COUNT_MISMATCH")
        if len(cell["train_learning_curve"]) != cell["optimizer_steps"]:
            raise ValueError("MISSING_EPOCH_CURVES")
        if [r["epoch"] for r in cell["train_learning_curve"]] != list(range(1, cell["optimizer_steps"] + 1)):
            raise ValueError("OPTIMIZER_EPOCH_LINEAGE")
        steps += cell["optimizer_steps"]
        states += len(checkpoints)
    if steps != report["optimizer_steps"] or states != report["actual_checkpoint_replays"]:
        raise ValueError("AGGREGATE_COMPLETION_COUNT_MISMATCH")
    votes = sum(c["checkpoint_validation"]["400"]["auxiliary_loss"] < c["checkpoint_validation"]["200"]["auxiliary_loss"] for c in cells)
    extension = report["common_extension"]
    if votes != extension["improved200to400_cells"] or extension["extension_required"] != (votes >= 6):
        raise ValueError("GROUP_EXTENSION_DECISION_MISMATCH")


def finite_mean(values):
    values = [float(v) for v in values if v is not None]
    return statistics.fmean(values) if values else None


def chance_lookup(chance):
    if chance is None:
        return {}
    result = {}
    for row in chance["rows"]:
        scope = row["scope"]
        group = "ALL" if scope == "global" else str(row["group"])
        key = (scope, group, row["mode"], row["direction"])
        if key in result:
            raise ValueError("DUPLICATE_EXACT_CHANCE_SCOPE")
        value = row["exact_chance_mean"]
        if row["anchors_covered"] > 0 and (value is None or not 0 < value <= 1):
            raise ValueError("INVALID_EXACT_CHANCE_VALUE")
        result[key] = row
    return result


def attach_chance(item, lookup, scope, group, mode, direction):
    chance = lookup.get((scope, str(group), mode, direction))
    if chance is None:
        return {"exact_chance_mean": None, "top1_minus_exact_chance": None,
                "chance_status": "NOT_INDEPENDENTLY_SUPPLIED_DO_NOT_INFER_FROM_MEAN_COUNT"}
    if item["anchors_covered"] != chance["anchors_covered"]:
        raise ValueError("CHANCE_SUPPORT_DIFFERS_FROM_ACTUAL_RETRIEVAL:" + str((scope, group, mode, direction)))
    value = chance["exact_chance_mean"]
    observed = item["top1_tie_averaged"]
    return {"exact_chance_mean": value,
            "top1_minus_exact_chance": observed - value if observed is not None and value is not None else None,
            "chance_status": "EXACT_PER_ANCHOR_RECIPROCAL_CANDIDATE_COUNT_MEAN"}


def seed_row(cell, chance=None):
    metrics = cell["selected_validation"]
    program = metrics["program"]
    row = {"arm": cell["arm"], "seed": cell["seed"], "selected_epoch": cell["selected_epoch"],
           "optimizer_steps": cell["optimizer_steps"], "validation_auxiliary_loss": metrics["auxiliary_loss"],
           "program_pearson": program["macro_program_pearson"],
           "program_spearman": program["macro_program_spearman"],
           "patient_equal_huber": program["patient_equal_masked_huber"],
           "patients": program["patients"], "valid_pearson_programs": program["valid_pearson_programs"]}
    for mode in ("hard", "random"):
        for direction, short in (("image_to_gene", "i2g"), ("gene_to_image", "g2i")):
            item = metrics["retrieval_" + mode][direction]
            for key in ("anchors_covered", "coverage", "mean_negative_count", "paired_gap_mean", "top1_tie_averaged", "mean_reciprocal_midrank"):
                row[mode + "_" + short + "_" + key] = item[key]
            comparison = attach_chance(item, chance or {}, "global", "ALL", mode, direction)
            for key in ("exact_chance_mean", "top1_minus_exact_chance"):
                row[mode + "_" + short + "_" + key] = comparison[key]
    return row


def aggregate_rows(seed_rows):
    numeric = [k for k in seed_rows[0] if k not in ("arm", "seed", "selected_epoch")]
    result = []
    for arm in ARMS:
        rows = [r for r in seed_rows if r["arm"] == arm]
        out = {"arm": arm, "seed_count": len(rows),
               "selected_epochs_seed31_47_83": ";".join(str(next(r["selected_epoch"] for r in rows if r["seed"] == s)) for s in SEEDS)}
        for key in numeric:
            values = [r[key] for r in rows if r[key] is not None]
            out[key + "_seed_mean"] = finite_mean(values)
            out[key + "_seed_min"] = min(values) if values else None
            out[key + "_seed_max"] = max(values) if values else None
        result.append(out)
    return result


def paired_rows(seed_rows):
    lookup = {(r["arm"], r["seed"]): r for r in seed_rows}
    metrics = ("program_pearson", "program_spearman", "patient_equal_huber", "validation_auxiliary_loss",
               "hard_i2g_top1_tie_averaged", "hard_g2i_top1_tie_averaged")
    rows = []
    for comparator in ("B3", "B6a", "B4_MAP_SHUFFLE"):
        for seed in SEEDS:
            a, b = lookup[("B4", seed)], lookup[(comparator, seed)]
            for metric in metrics:
                delta = a[metric] - b[metric] if a[metric] is not None and b[metric] is not None else None
                higher = metric not in ("patient_equal_huber", "validation_auxiliary_loss")
                rows.append({"comparison": "B4-minus-" + comparator, "seed": seed, "metric": metric,
                             "B4_selected_epoch": a["selected_epoch"], "control_selected_epoch": b["selected_epoch"],
                             "B4": a[metric], "control": b[metric], "delta_raw": delta,
                             "higher_is_better": higher,
                             "benefit_oriented_delta": delta if higher or delta is None else -delta,
                             "uncertainty": "same_validation_patients_three_seeds_descriptive_no_independent_sample_CI"})
    return rows


def metric_scopes(metrics):
    yield "global", "ALL", metrics
    for variable, groups in metrics["strata"].items():
        for group, value in groups.items():
            yield variable, group, value


def program_rows(report):
    for cell in report["cells"]:
        for epoch, metrics in sorted(cell["checkpoint_validation"].items(), key=lambda x: int(x[0])):
            for scope, group, item in metric_scopes(metrics):
                program = item["program"]
                for p in program["per_program"]:
                    yield {"arm": cell["arm"], "seed": cell["seed"], "epoch": int(epoch),
                           "selected": int(epoch) == cell["selected_epoch"], "scope": scope, "group": group,
                           "program_index": p["program_index"], "patients": p["patients"],
                           "pearson": p["pearson"], "spearman": p["spearman"]}
    for name, metrics in report["baseline_metrics"].items():
        for p in metrics["per_program"]:
            yield {"arm": "BASELINE_" + name, "seed": "DETERMINISTIC", "epoch": "NOT_APPLICABLE",
                   "selected": True, "scope": "global", "group": "ALL", "program_index": p["program_index"],
                   "patients": p["patients"], "pearson": p["pearson"], "spearman": p["spearman"]}


def retrieval_rows(report, chance=None):
    for cell in report["cells"]:
        for epoch, metrics in sorted(cell["checkpoint_validation"].items(), key=lambda x: int(x[0])):
            for scope, group, item in metric_scopes(metrics):
                for mode in ("hard", "random"):
                    for direction in ("image_to_gene", "gene_to_image"):
                        observed = item["retrieval_" + mode][direction]
                        yield {"arm": cell["arm"], "seed": cell["seed"], "epoch": int(epoch),
                               "selected": int(epoch) == cell["selected_epoch"], "scope": scope, "group": group,
                               "mode": mode, "direction": direction, **observed,
                               **attach_chance(observed, chance or {}, scope, group, mode, direction),
                               "chance_requirement": "exact mean of per-anchor 1/(n_negative+1), not reciprocal mean count"}


def csv_parts(directory, prefix, rows):
    rows = iter(rows)
    first = next(rows, None)
    if first is None:
        raise ValueError("NO_CSV_ROWS:" + prefix)
    columns = list(first)
    def encode(row, header=False):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        if header:
            writer.writeheader()
        else:
            writer.writerow(row)
        return stream.getvalue().encode("utf-8")
    header = encode(first, header=True)
    parts, current, count = [], bytearray(header), 0
    def finish():
        ordinal = len(parts) + 1
        record = write_once(Path(directory) / (prefix + "_" + str(ordinal).zfill(3) + ".csv"), bytes(current))
        record["data_rows"] = count
        parts.append(record)
    for row in (first, *rows):
        content = encode(row)
        if len(content) + len(header) >= LIMIT:
            raise ValueError("OVERSIZE_CSV_ROW")
        if len(current) + len(content) >= 1_750_000 or count >= 19_999:
            finish()
            current, count = bytearray(header), 0
        current.extend(content)
        count += 1
    finish()
    return parts


def number(value, places=6):
    return "NA" if value is None else f"{value:.{places}f}"


def markdown_report(report, means, comparisons, *, source_sha, context=None, chance=None):
    lines = ["# P1：无DSS的语义—分子对应资格实验", "",
             "**本轮完成12个神经拟合、3个Ridge decoder family；DSS/P2未启动。** 这是既有TCGA队列一个inner分区的内部开发结果，不是独立外部验证。", "",
             f"原外折1的409名训练患者中，本分区训练{report['train_patients']}人、验证{report['valid_patients']}人；本轮没有读取外折测试输入。", "",
             "## 实际完成与预算", "",
             f"- 神经矩阵：B3/B4/B6a/B4_MAP_SHUFFLE × seeds31/47/83；{report['optimizer_steps']}次Adam更新、{report['actual_checkpoint_replays']}个已保存状态实际前向回放。",
             "- 固定Ridge alpha=1；Clinical、Pathology、P+C三个decoder，每个含50个测量掩码内的程序求解，共150个标量右端系统；相同掩码可共享分解。训练患者中位数对照不拟合神经网络。",
             f"- {report['common_extension']['improved200to400_cells']}/12个拟合的验证辅助损失在200→400下降；共同800续训触发={report['common_extension']['extension_required']}，预算受限标志={report['budget_limited']}。这不等于已证明数学收敛。",
             "- checkpoint只按无DSS的validation L_aux最小选取；同值选较早epoch。所有候选和每epoch训练曲线均保留。", "",
             "## 三seed描述性结果", "",
             "以下是各seed独立按辅助损失选轮后的均值。三个seed共享同一训练/验证患者，不是三个独立队列，不计算伪独立置信区间。", "",
             "| 模型 | seed31/47/83选轮 | 程序Pearson | 程序Spearman | 患者等权Huber（低优） |",
             "|---|---|---:|---:|---:|"]
    for row in means:
        lines.append(f"| {row['arm']} | {row['selected_epochs_seed31_47_83']} | {number(row['program_pearson_seed_mean'])} | {number(row['program_spearman_seed_mean'])} | {number(row['patient_equal_huber_seed_mean'])} |")
    lines += ["", "## 固定Ridge与中位数参考", "",
              "| 参考 | 程序Pearson | 程序Spearman | 患者等权Huber（低优） |", "|---|---:|---:|---:|"]
    for name, item in report["baseline_metrics"].items():
        lines.append(f"| {name} | {number(item['macro_program_pearson'])} | {number(item['macro_program_spearman'])} | {number(item['patient_equal_masked_huber'])} |")
    lines += ["", "**相关性与预测误差必须分开判断。** 较高的预测相关性不意味着预测数值校准或Huber更好。训练中位数在每个程序内跨患者恒定，Pearson/Spearman未定义（NA），不能当成0再计算相关性增益。", "",
              "## 检索与精确chance", "",
              "| 模型 | Hard I→G Top1 | 精确chance | 差值 | Hard G→I Top1 | 精确chance | 差值 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in means:
        lines.append(f"| {row['arm']} | {number(row['hard_i2g_top1_tie_averaged_seed_mean'])} | {number(row['hard_i2g_exact_chance_mean_seed_mean'])} | {number(row['hard_i2g_top1_minus_exact_chance_seed_mean'])} | {number(row['hard_g2i_top1_tie_averaged_seed_mean'])} | {number(row['hard_g2i_exact_chance_mean_seed_mean'])} | {number(row['hard_g2i_top1_minus_exact_chance_seed_mean'])} |")
    lines += ["", "精确chance按每个有合适负例的anchor取1/(n_negative+1)后平均，并逐方向/分层核对实际覆盖；NA表示未提供独立精确计算，不用1/(平均候选数)填补。即使差值为正，也只是同一验证患者上的描述性结果，不能据此单独证明对应。", "",
              "## 同seed机制比较", "",
              "每项先在同seed相减，再描述三seed均值；不额外拟合，不构造置信区间。", "",
              "| 比较 | Pearson Δ均值 | Huber Δ均值（负优） | Pearson正向seed数 |", "|---|---:|---:|---:|"]
    for comparator in ("B3", "B6a", "B4_MAP_SHUFFLE"):
        p = [x["delta_raw"] for x in comparisons if x["comparison"] == "B4-minus-" + comparator and x["metric"] == "program_pearson"]
        h = [x["delta_raw"] for x in comparisons if x["comparison"] == "B4-minus-" + comparator and x["metric"] == "patient_equal_huber"]
        lines.append(f"| B4−{comparator} | {number(finite_mean(p))} | {number(finite_mean(h))} | {sum(x is not None and x > 0 for x in p)}/3 |")
    lines += ["", "## 科学诊断与限制", "",
              "1. 只能用上述真文本/伪文本、真实/错误辅助正对、真实/乱序关系的直接配对描述本实现。没有稳定优势时，不能宣称文本身份或真实配对已产生独特对应收益。",
              "2. 神经L_prog将8个slot先均值汇总为64维；Ridge保留8×768局部读取并拼接TITAN native/aligned/S32。维数、读出与正则先验并不相等，因此性能差异只能定位候选瓶颈，不能直接证明某个encoder必然错误。",
              "3. B6a与其他臂同seed共用实际负例候选：预先排除B6错误正对；本人图像、RNA/CNA、程序目标和临床不换。两人分层可能对所有臂都没有负例，覆盖已完整保留。",
              "4. 检索Top1需逐anchor精确比较 E[1/(n_negative+1)]；1/(平均候选数)不等价。正gap、非零梯度、回放成功都不能单独证明语义对应。独立chance诊断另列，未完成时不作高于chance的主张。",
              "5. 本轮没有DSS模型或外折预后结果，不能判断65个事件是否造成训练—验证分离，更不能从无DSS对应直接推导生存增量。",
              "6. Bulk RNA无局部spot真值，CONCH/TITAN语义分数不是组织学人工标签；单个inner分区和共享患者的seed变化不构成独立医学验证。",
              "", "## 可复核交付", "",
              "全50程序、所有低支持/无效相关、stage/missing/cohort分层、随机与困难负例、每checkpoint及训练曲线均在无损原报告中，不删除负结果。明文CSV方便查看；lossless index绑定原UTF8字节与canonical JSON，restore函数实际重组核验通过。",
              "", f"原聚合字节SHA256：`{source_sha}`。", "",
              "公开发布前对所有普通JSON分片与明文报告执行原私有denylist检查；完整重组字节也可按同一guard独立复核。"]
    if context is not None:
        lines += ["", "## 独立资源与输入核验凭据", "",
                  "附带上下文完整保存于单独JSON；以下资源数字仅在上下文提供明确测量字段时使用，不根据显存倒推并行或完工状态。"]
    return "\n".join(lines) + "\n"


def build(args):
    raw = Path(args.aggregate).read_bytes()
    if digest(raw) != args.expected_sha256:
        raise ValueError("PUBLIC_AGGREGATE_SHA_MISMATCH")
    report = json.loads(raw.decode("utf-8-sig"))
    validate_aggregate(report)
    directory = Path(args.output_dir)
    context = json.loads(Path(args.context).read_text(encoding="utf-8-sig")) if args.context else None
    chance = json.loads(Path(args.chance).read_text(encoding="utf-8-sig")) if args.chance else None
    if chance is not None and chance.get("input_sha256") != report["input_sha256"]:
        raise ValueError("EXACT_CHANCE_INPUT_BANK_SHA_DIFFERS")
    lookup = chance_lookup(chance)
    index, records = lossless_package(raw, directory)
    seeds = sorted((seed_row(c, lookup) for c in report["cells"]), key=lambda r: (ARMS.index(r["arm"]), r["seed"]))
    means = aggregate_rows(seeds)
    paired = paired_rows(seeds)
    records += csv_parts(directory, "04_PRETEXT_SEED_METRICS", seeds)
    records += csv_parts(directory, "04_PRETEXT_ARM_SUMMARY", means)
    records += csv_parts(directory, "04_PRETEXT_PAIRED_MECHANISM_DIFFERENCES", paired)
    records += csv_parts(directory, "04_PRETEXT_PROGRAM_METRICS", program_rows(report))
    records += csv_parts(directory, "04_PRETEXT_RETRIEVAL_METRICS", retrieval_rows(report, lookup))
    curve_rows = ({"arm": c["arm"], "seed": c["seed"], **r} for c in report["cells"] for r in c["train_learning_curve"])
    records += csv_parts(directory, "04_PRETEXT_TRAINING_CURVES", curve_rows)
    summary = {"schema": "semantic_b0b7_p1_read_only_summary_v1", "status": "P1_AUXILIARY_EVALUATED_DSS_NOT_STARTED",
               "source_utf8_sha256": digest(raw), "source_canonical_sha256": index["canonical_json_sha256"],
               "neural_fits": report["neural_fits"], "decoder_fits": report["decoder_fits"],
               "optimizer_steps": report["optimizer_steps"], "actual_checkpoint_replays": report["actual_checkpoint_replays"],
               "same_seed_rows": seeds, "three_seed_descriptive_means": means,
               "same_seed_differences": paired, "common_extension": report["common_extension"],
               "unmodified_source_bindings": report["frozen_sources"],
               "uncertainty_interval_invented": False, "DSS_or_P2_models_fitted": 0,
               "independent_patients_across_seeds": False,
               "original_all_fields_and_bytes_preserved": True,
               "additional_context": context, "independent_exact_chance": chance,
               "exact_chance_external_binding": {"input_sha256": report["input_sha256"],
                   "source_aggregate_sha256": digest(raw),
                   "chance_report_file_sha256": digest(Path(args.chance).read_bytes())} if chance else None}
    records.append(write_once(directory / "04_PRETEXT_PAIRED_RETENTION_RESULTS.json", json_bytes(summary)))
    markdown = markdown_report(report, means, paired, source_sha=digest(raw), context=context, chance=chance)
    records.append(write_once(directory / "04_PRETEXT_PAIRED_RETENTION_RESULTS.md", markdown.encode("utf-8")))
    audit = {"schema": "semantic_b0b7_p1_read_only_public_report_build_v1", "status": "PASS",
             "no_model_or_array_loader_imported": True, "optimizer_steps_run_by_reporter": 0,
             "original_utf8_roundtrip_exact": True, "canonical_roundtrip_exact": True,
             "source_utf8_sha256": digest(raw), "source_utf8_bytes": len(raw),
             "all_public_files_below_2MiB": True, "files": records,
             "publisher_must_validate_original_denylist_on_all_structured_fields": True}
    write_once(directory / "04_PRETEXT_REPORT_BUILD_AUDIT.json", json_bytes(audit))
    print(json.dumps({"status": "PASS", "files_without_self_audit": len(records),
                      "source_bytes": len(raw), "lossless_parts": len(index["parts"]),
                      "original_bytes_roundtrip_exact": True}))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--aggregate", required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--context")
    p.add_argument("--chance")
    r = sub.add_parser("restore")
    r.add_argument("--index", required=True)
    r.add_argument("--output")
    args = cli.parse_args()
    if args.command == "build":
        build(args)
    else:
        value, raw = restore_public_report(args.index)
        if args.output:
            path = Path(args.output)
            with path.open("xb") as stream:
                stream.write(raw)
        print(json.dumps({"status": "PASS", "source_bytes": len(raw), "source_utf8_sha256": digest(raw),
                          "canonical_json_sha256": digest(canonical(value))}))


if __name__ == "__main__":
    main()

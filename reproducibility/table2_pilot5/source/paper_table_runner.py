"""File adapters and reporting for the three project evaluation tables."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from paper_table_metrics import METRIC_KEYS, label_metrics, read_label_map, summarize


HEADERS = {
    1: ["Method", "Part Condition", "Proj. mIoU (%)", "Proj. Boundary F1 (%)", "Part Plausibility (1-5)", "Boundary Quality (1-5)"],
    2: ["Variant", "2D Matched mIoU (%)", "2D Boundary F1 (%)", "2D Part F1@0.5 (%)", "Proj. mIoU (%)"],
    3: ["Method", "Total Time (s/object)", "Peak GPU Memory (GB)", "Run Failure Rate (%)", "Collapse Rate (%)"],
}


def field(record, dotted_key, default=None):
    value = record
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def data_path(root, pattern, sample=None):
    text = str(pattern).format(**(sample or {}))
    path = Path(text)
    return path if path.is_absolute() else root / path


def read_config(config_path, root_override=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    root = Path(root_override or config["root"])
    if not root.is_absolute():
        root = config_path.parent / root
    root = root.resolve()
    with data_path(root, config["manifest"]).open(encoding="utf-8-sig", newline="") as handle:
        samples = list(csv.DictReader(handle))
    if not samples or any(not row.get("uid") or not row.get("group") for row in samples):
        raise ValueError("The manifest requires nonempty uid and group columns")
    keys = [(row["group"], row["uid"]) for row in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("The manifest contains duplicate (group, uid) records")
    expected_counts = config.get("expected_group_counts")
    if expected_counts is not None and dict(Counter(row["group"] for row in samples)) != expected_counts:
        raise ValueError("Manifest group counts do not match expected_group_counts; do not substitute a pilot")
    groups = config.get("groups", list(dict.fromkeys(row["group"] for row in samples)))
    if not groups or len(groups) != len(set(groups)) or any(row["group"] not in groups for row in samples):
        raise ValueError("Configured groups must be unique and include every manifest group")
    methods = config["methods"]
    for name in ("key", "name"):
        values = [method[name] for method in methods]
        if len(values) != len(set(values)):
            raise ValueError(f"Method {name} values must be unique")
    return config, root, samples, groups


def source_labels(root, spec, sample):
    if "labels" in spec:
        path = data_path(root, spec["labels"], sample)
    elif "annotation" in spec:
        annotation_path = data_path(root, spec["annotation"], sample)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
        label_path = field(annotation, spec.get("labels_key", "final_map.labels_path"))
        if not label_path:
            raise ValueError(f"No discrete-label path in {annotation_path}")
        path = data_path(root, label_path)
    else:
        raise ValueError("A label source needs labels or annotation")
    palette = spec.get("palette")
    if isinstance(palette, str):
        palette = json.loads(data_path(root, palette, sample).read_text(encoding="utf-8-sig"))
    if spec.get("palette_annotation"):
        if palette is not None:
            raise ValueError("Use palette or palette_annotation, not both")
        annotation = json.loads(data_path(root, spec["palette_annotation"], sample).read_text(encoding="utf-8-sig"))
        palette = {str(spec.get("background", 0)): spec.get("background_color", [0, 0, 0])}
        colors = {tuple(palette[str(spec.get("background", 0))])}
        for part in annotation["parts"]:
            label = str(int(part["label_id"]))
            color = tuple(part["color"])
            if label in palette or color in colors:
                raise ValueError("Annotation palette collision: label IDs and colors must be unique")
            palette[label] = list(color)
            colors.add(color)
    if spec.get("palette_format") == "index_array":
        if spec.get("label_semantics") != "palette_index":
            raise ValueError("index_array requires explicit label_semantics=palette_index; original part IDs are not recovered")
        colors = np.asarray(palette)
        if colors.ndim != 2 or colors.shape[0] < 2 or colors.shape[1] not in (3, 4):
            raise ValueError("An index-array palette must contain RGB/RGBA rows including background")
        if not np.issubdtype(colors.dtype, np.integer) or np.any(colors < 0) or np.any(colors > 255):
            raise ValueError("Index-array palette colors must be 0..255 integers")
        palette = {str(index): color[:3].tolist() for index, color in enumerate(colors)}
    if palette is not None and not isinstance(palette, dict):
        raise ValueError("A palette must be a label-to-color mapping, or an explicit index_array")
    return read_label_map(path, palette=palette)


def camera_parameters(path, index=0):
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        payload = payload[index]
    elif isinstance(payload, dict) and "frames" in payload:
        payload = {**{key: value for key, value in payload.items() if key != "frames"}, **payload["frames"][index]}
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported camera JSON in {path}")
    matrix_keys = ("transform_matrix", "camera_to_world", "world_to_camera", "extrinsic", "c2w")
    keys = matrix_keys + ("intrinsics", "K", "camera_angle_x", "camera_angle_y", "fl_x", "fl_y", "cx", "cy", "w", "h", "width", "height", "image_size")
    result = {key: payload[key] for key in keys if key in payload}
    if not any(key in result for key in matrix_keys):
        raise ValueError(f"No supported camera transform in {path}")
    return result


def check_camera(root, pred, ref, sample):
    if not pred.get("camera") or not ref.get("camera"):
        raise ValueError("Projection evaluation requires prediction and reference camera paths")
    left = camera_parameters(data_path(root, pred["camera"], sample), pred.get("camera_index", 0))
    right = camera_parameters(data_path(root, ref["camera"], sample), ref.get("camera_index", 0))
    if set(left) != set(right):
        raise ValueError("Prediction/reference camera fields differ; normalize the camera metadata first")
    for key in left:
        a, b = np.asarray(left[key], dtype=float), np.asarray(right[key], dtype=float)
        if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or not np.allclose(a, b, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Prediction/reference camera mismatch in {key}")


def evaluate_maps(config, root, samples, groups, method, kind, missing_policy):
    records = []
    reference_spec = config.get("reference", {}).get(kind)
    prediction_spec = method.get(kind)
    options = config.get("metrics", {})
    for sample in samples:
        row = {"method": method["key"], "group": sample["group"], "uid": sample["uid"], "stage": kind,
               "status": "not_configured", "error": None,
               "label_semantics": (prediction_spec or {}).get("label_semantics", "part_id"), **dict.fromkeys(METRIC_KEYS)}
        if reference_spec and prediction_spec:
            try:
                reference = source_labels(root, reference_spec, sample)
            except FileNotFoundError as exc:
                row.update(status="missing_reference", error=str(exc))
                records.append(row)
                continue
            except (ValueError, OSError, KeyError, IndexError) as exc:
                row.update(status="invalid_reference", error=str(exc))
                records.append(row)
                continue
            try:
                prediction = source_labels(root, prediction_spec, sample)
            except FileNotFoundError as exc:
                row.update(status="missing_prediction", error=str(exc))
                if missing_policy == "zero":
                    prediction = np.full_like(reference, options.get("background", 0))
                    row.update(label_metrics(prediction, reference, background=options.get("background", 0),
                                             ignore_labels=options.get("ignore_labels", [-1, 65535]),
                                             boundary_tolerance=options.get("boundary_tolerance_px", 2),
                                             part_threshold=options.get("part_iou_threshold", 0.5)))
                    row["status"] = "missing_prediction_zeroed"
                records.append(row)
                continue
            except (ValueError, OSError, KeyError, IndexError) as exc:
                row.update(status="invalid_prediction", error=str(exc))
                records.append(row)
                continue
            try:
                if kind == "projection":
                    check_camera(root, prediction_spec, reference_spec, sample)
                row.update(label_metrics(prediction, reference, background=options.get("background", 0),
                                         ignore_labels=options.get("ignore_labels", [-1, 65535]),
                                         boundary_tolerance=options.get("boundary_tolerance_px", 2),
                                         part_threshold=options.get("part_iou_threshold", 0.5)))
                row["status"] = "ok"
            except (ValueError, OSError, KeyError, IndexError) as exc:
                row.update(status="invalid_prediction", error=str(exc))
        records.append(row)
    return records, summarize(records, groups, METRIC_KEYS)


def human_scores(config, root, samples, groups, method):
    indexed = defaultdict(list)
    path = config.get("human_ratings")
    if path and data_path(root, path).is_file():
        with data_path(root, path).open(encoding="utf-8-sig", newline="") as handle:
            seen = set()
            allowed = {(row["group"], row["uid"]) for row in samples}
            for row in csv.DictReader(handle):
                if row["method"] not in (method["key"], method["name"]):
                    continue
                key = (row["group"], row["uid"])
                if key not in allowed:
                    raise ValueError(f"Human rating outside the manifest: {key}")
                identity = (*key, row["rater_id"])
                if identity in seen:
                    raise ValueError(f"Duplicate human rating: {identity}")
                seen.add(identity)
                scores = [float(row[key]) for key in ("plausibility", "boundary_quality")]
                if not all(math.isfinite(value) and 1 <= value <= 5 for value in scores):
                    raise ValueError("Human scores must lie between 1 and 5")
                indexed[key].append(scores)
    records = []
    for sample in samples:
        scores = indexed[(sample["group"], sample["uid"])]
        means = np.mean(scores, axis=0) if scores else [None, None]
        records.append({"group": sample["group"], "uid": sample["uid"], "n_raters": len(scores),
                        "plausibility": float(means[0]) if scores else None,
                        "boundary_quality": float(means[1]) if scores else None})
    return records, summarize(records, groups, ("plausibility", "boundary_quality"))


def _log_index(root, spec, samples):
    paths = sorted(glob.glob(str(data_path(root, spec["path"]))))
    allowed = {(row["group"], row["uid"]) for row in samples}
    records = {}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig") as handle:
            if Path(path).suffix == ".jsonl":
                loaded = [json.loads(line) for line in handle if line.strip()]
            else:
                loaded = json.load(handle)
                loaded = loaded if isinstance(loaded, list) else [loaded]
        for row in loaded:
            if any(field(row, key) != value for key, value in spec.get("filters", {}).items()):
                continue
            if spec.get("run_id") is not None and row.get("run_id") != spec["run_id"]:
                continue
            if field(row, spec.get("status_key", "status")) in {"skipped_existing", "skipped", "not_run", "cached", "waiting"}:
                continue
            identity = (field(row, spec.get("group_key", "group")), field(row, spec.get("uid_key", "uid")))
            if identity not in allowed:
                continue
            if identity in records:
                raise ValueError(f"Multiple measured log records for {identity}; select a unique run_id/log file")
            records[identity] = row
    return records


def _number(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("A numeric log measurement cannot be boolean")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Log measurements must be finite and nonnegative")
    return value


def run_measurements(config, root, samples, method):
    measurement_scope = method.get("measurement_scope", "end_to_end")
    if measurement_scope not in {"end_to_end", "segmentation_only"}:
        raise ValueError("measurement_scope must be end_to_end or segmentation_only")
    components = method.get("run_components", [])
    indexed = [(spec, _log_index(root, spec, samples)) for spec in components]
    records = []
    memory_scale = {"GB": 1.0, "GiB": 2**30 / 1e9, "MB": 1e6 / 1e9, "MiB": 2**20 / 1e9, "bytes": 1 / 1e9}
    for sample in samples:
        identity = (sample["group"], sample["uid"])
        states, times, memories, collapse_flags = [], [], [], []
        has_final_stage = False
        for spec, index in indexed:
            phase = spec.get("phase")
            if phase not in {"preprocessing", "guidance", "segmentation", "pipeline"}:
                raise ValueError("Each run component needs phase preprocessing/guidance/segmentation/pipeline")
            has_final_stage |= phase in {"segmentation", "pipeline"}
            log = index.get(identity)
            status = field(log or {}, spec.get("status_key", "status"))
            success_statuses = ["success", "partial", "ok"] if phase == "guidance" else ["success", "ok"]
            state = True if status in spec.get("success_statuses", success_statuses) else False if status in spec.get("failure_statuses", ["failed", "error"]) else None
            if phase in {"segmentation", "pipeline"} and field(log or {}, "stage_status.segvigen") == "not_run":
                state = None
            states.append(state)
            times.append(_number(field(log or {}, spec.get("time_key", "timing_seconds.total"))) if state is not None else None)
            memory = _number(field(log or {}, spec.get("memory_key", "peak_gpu_memory_gb"))) if state is not None else None
            unit = spec.get("memory_unit", "GB")
            if unit not in memory_scale:
                raise ValueError(f"Unsupported memory unit: {unit}")
            memories.append(memory * memory_scale[unit] if memory is not None else None)
            if phase in {"segmentation", "pipeline"} and state is True:
                flag = field(log or {}, spec.get("collapse_key", "collapsed"))
                if flag is not None and not isinstance(flag, bool):
                    raise ValueError("Collapse flags must be explicit booleans, not part counts or RGB colors")
                if flag is not None:
                    collapse_flags.append(flag)
        complete = bool(states) and has_final_stage and all(state is True for state in states)
        failed = bool(states) and any(state is False for state in states)
        failure = float(failed) if failed or complete else None
        elapsed = sum(times) if has_final_stage and times and all(value is not None for value in times) else None
        peak = max(memories) if has_final_stage and memories and all(value is not None for value in memories) else None
        if len(set(collapse_flags)) > 1:
            raise ValueError(f"Conflicting collapse flags for {identity}")
        collapsed = float(collapse_flags[0]) if complete and collapse_flags else None
        records.append({"method": method["key"], "group": sample["group"], "uid": sample["uid"], "stage": "runtime",
                        "status": "success" if complete else "failed" if failed else "unknown",
                        "measurement_scope": measurement_scope,
                        "measured_component_seconds": elapsed, "measured_component_peak_gpu_memory_gb": peak,
                        "total_seconds": elapsed if measurement_scope == "end_to_end" else None,
                        "peak_gpu_memory_gb": peak if measurement_scope == "end_to_end" else None, "run_failure": failure,
                        "collapsed": collapsed})
    expected = len(samples)
    def complete_mean(key):
        values = [row[key] for row in records]
        return float(np.mean(values)) if len(values) == expected and all(value is not None for value in values) else None
    peak_values = [row["peak_gpu_memory_gb"] for row in records]
    valid_outputs = [row for row in records if row["status"] == "success"]
    flags = [row["collapsed"] for row in valid_outputs]
    summary = {
        "measurement_scope": measurement_scope,
        "total_seconds": complete_mean("total_seconds"),
        "peak_gpu_memory_gb": max(peak_values) if peak_values and all(value is not None for value in peak_values) else None,
        "run_failure": complete_mean("run_failure"),
        "collapsed": float(np.mean(flags)) if flags and all(value is not None for value in flags) else None,
        "n_expected": expected, "n_valid_outputs": len(valid_outputs),
        "n_known_run_status": sum(row["run_failure"] is not None for row in records),
        "n_timed": sum(row["total_seconds"] is not None for row in records),
        "n_memory_measured": sum(row["peak_gpu_memory_gb"] is not None for row in records),
        "n_collapse_measured": sum(row["collapsed"] is not None for row in valid_outputs),
    }
    return records, summary


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_table(table_number, argv=None):
    parser = argparse.ArgumentParser(description=f"Evaluate the metrics for paper Table {table_number}; no model inference is run.")
    parser.add_argument("--config", required=True, help="JSON mapping the current experiment's manifest, labels, cameras, and logs")
    parser.add_argument("--root", help="Override the configured experiment root")
    parser.add_argument("--out", required=True, help="Directory for CSV and JSON reports")
    parser.add_argument("--missing-policy", choices=("unavailable", "zero"), default="unavailable",
                        help="Keep incomplete quality means TBD by default; zero only if missing outputs are confirmed failures")
    args = parser.parse_args(argv)
    try:
        config, root, samples, groups = read_config(args.config, args.root)
        if table_number == 2 and config.get("metrics", {}).get("part_iou_threshold", 0.5) != 0.5:
            raise ValueError("Table 2 uses Part F1@0.5; part_iou_threshold must be 0.5")
        table_rows, per_sample, methods = [], [], []
        for method in config["methods"]:
            if table_number not in method.get("tables", [1, 2, 3]):
                continue
            result = {"key": method["key"], "name": method["name"],
                      "label_semantics": {kind: method.get(kind, {}).get("label_semantics", "part_id")
                                          for kind in ("guidance", "projection")}}
            if table_number in (1, 2):
                projection_rows, projection = evaluate_maps(config, root, samples, groups, method, "projection", args.missing_policy)
                per_sample.extend(projection_rows)
                result["projection"] = projection
                p = projection["macro"]
                if table_number == 1:
                    human_rows, human = human_scores(config, root, samples, groups, method)
                    result["human"] = human
                    result["human_per_object"] = human_rows
                    values = [method["name"], p["matched_miou"], p["boundary_f1"], human["macro"]["plausibility"], human["macro"]["boundary_quality"]]
                    scales = [1, 100, 100, 1, 1]
                else:
                    guidance_rows, guidance = evaluate_maps(config, root, samples, groups, method, "guidance", args.missing_policy)
                    per_sample.extend(guidance_rows)
                    result["guidance"] = guidance
                    g = guidance["macro"]
                    values = [method["name"], g["matched_miou"], g["boundary_f1"], g["part_f1"], p["matched_miou"]]
                    scales = [1, 100, 100, 100, 100]
            else:
                runtime_rows, runtime = run_measurements(config, root, samples, method)
                per_sample.extend(runtime_rows)
                result["runtime"] = runtime
                values = [method["name"], runtime["total_seconds"], runtime["peak_gpu_memory_gb"], runtime["run_failure"], runtime["collapsed"]]
                scales = [1, 1, 1, 100, 100]
            formatted = [values[0]] + ["TBD" if value is None else f"{value * scale:.2f}" for value, scale in zip(values[1:], scales[1:])]
            headers = [header for header in HEADERS[table_number] if header != "Part Condition"]
            table_row = dict(zip(headers, formatted))
            if table_number == 1:
                table_row["Part Condition"] = method.get("part_condition", "TBD")
                result["part_condition"] = table_row["Part Condition"]
            table_rows.append(table_row)
            methods.append(result)
        if not methods:
            raise ValueError(f"No configured methods are assigned to Table {table_number}")
        errors = [row for row in per_sample if row.get("status", "").startswith("invalid_")]
        report = {
            "schema_version": "paper_table_evaluation@1", "table": table_number,
            "config": str(Path(args.config).resolve()), "root": str(root), "groups": groups,
            "n_manifest_objects": len(samples), "missing_policy": args.missing_policy,
            "metrics": config.get("metrics", {}), "quality_aggregation": "object mean within group, then equal group macro",
            "miou_protocol": "one-to-one IoU matching; unmatched reference parts score zero; normalized by reference-part count",
            "supplementary_gt_best_iou": "Each reference part independently takes its best predicted overlap; not one-to-one. No best-level search is performed.",
            "methods": methods, "input_errors": errors,
        }
        output = Path(args.out).resolve()
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / f"table{table_number}.csv", HEADERS[table_number], table_rows)
        sample_headers = list(dict.fromkeys(key for row in per_sample for key in row))
        write_csv(output / f"table{table_number}_per_sample.csv", sample_headers, per_sample)
        (output / f"table{table_number}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if table_number == 1 and not (output / "human_ratings_template.csv").exists():
            write_csv(output / "human_ratings_template.csv", ["method", "group", "uid", "rater_id", "plausibility", "boundary_quality"], [])
        print(f"Table {table_number}: {len(methods)} methods, {len(samples)} manifest objects; reports saved to {output}")
        if errors:
            print(f"Found {len(errors)} invalid inputs; affected means remain TBD. See input_errors in table{table_number}.json.")
        return 2 if errors else 0
    except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
        parser.exit(2, f"Evaluation input error: {exc}\n")

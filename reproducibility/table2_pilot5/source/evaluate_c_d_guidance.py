"""Compare existing C color guides directly with UID-matched D integer labels."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image


KEYS = ("matched_miou", "boundary_f1", "part_f1")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_metrics(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name("paper_table_metrics.py")
    spec = importlib.util.spec_from_file_location("direct_2d_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decode_colors(rgba, palette):
    """Exact visible color IDs, preserving disconnected same-color regions."""
    rgba = np.asarray(rgba)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected RGBA pixels")
    packed = (rgba[:, :, 0].astype(np.uint32) << 16) | (rgba[:, :, 1].astype(np.uint32) << 8) | rgba[:, :, 2].astype(np.uint32)
    packed[rgba[:, :, 3] == 0] = 0
    table = {0: 0}
    for index, color in enumerate(palette, 1):
        key = (int(color[0]) << 16) | (int(color[1]) << 8) | int(color[2])
        if key in table:
            raise ValueError("Palette must contain unique nonblack foreground colors")
        table[key] = index
    values, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
    bad = [(int(v), int(n)) for v, n in zip(values, counts) if int(v) not in table]
    if bad:
        raise ValueError(f"off-palette foreground pixels: {bad}")
    mapped = np.array([table[int(v)] for v in values], dtype=np.uint16)
    return mapped[inverse].reshape(packed.shape)


def aggregate(records):
    if not records:
        return {}
    values = np.array([[row["metrics"][key] for key in KEYS] for row in records])
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(records), size=(2000, len(records)))
    ci = np.quantile(values[indices].mean(axis=1), [0.025, 0.975], axis=0)
    return {key: {"mean": float(values[:, i].mean()), "ci95": ci[:, i].tolist()}
            for i, key in enumerate(KEYS)}


def evaluate(predictions, references, manifest_path, palette_path, output, *, progress=True,
             method="C_v5_full", rotation_k=0, metrics_path=None):
    predictions, references, manifest_path, palette_path, output = map(
        Path, (predictions, references, manifest_path, palette_path, output))
    started = time.monotonic()
    if rotation_k not in (0, 1, 2, 3):
        raise ValueError("rotation_k must be an explicit fixed quarter-turn count")
    metric = load_metrics(metrics_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group = manifest.get("group")
    if group not in ("assemblies", "solid_volume", "thin_fine_grained"):
        raise ValueError(f"Unsupported reference group: {group}")
    entries = manifest["per_uid"]
    if any(row.get("group", group) != group for row in entries):
        raise ValueError("Reference manifest mixes different groups")
    expected = {row["uid"] for row in entries}
    if not expected or len(entries) != len(expected):
        raise ValueError("Reference cohort must contain nonempty unique UIDs")
    colors = json.loads(palette_path.read_text(encoding="utf-8"))
    # The existing B/C generation reserves palette[0]. Its remaining 20 colors
    # are anonymous visible IDs here; no semantic name is inferred from a color.
    palette = [tuple(row[:3]) for row in colors[1:]]
    prediction_paths = {path.stem: path for path in predictions.glob("*.png")}
    available = expected & prediction_paths.keys()
    missing = sorted(expected - prediction_paths.keys())
    records, errors = [], []
    core_path = Path(metric.__file__)
    core_hash = digest(core_path)
    palette_hash = digest(palette_path)
    for index, entry in enumerate(entries, 1):
        uid = entry["uid"]
        if uid not in available:
            save_json(output / "per_object" / uid / "result.json",
                      {"uid": uid, "status": "missing_prediction", "metrics": None})
            continue
        try:
            ref_path = references / entry["labels_file"]
            if digest(ref_path) != entry["labels_sha256"]:
                raise ValueError("Reference hash mismatch against the designated manifest")
            reference = metric.read_label_map(ref_path)
            if not np.any(reference > 0) or np.any(reference == 65535):
                raise ValueError("Reference must have foreground and no undecided label 65535")
            pred_path = prediction_paths[uid]
            with Image.open(pred_path) as image:
                prediction = decode_colors(np.asarray(image.convert("RGBA")), palette)
            prediction = np.rot90(prediction, k=rotation_k)
            scores = metric.label_metrics(prediction, reference, boundary_tolerance=2, part_threshold=0.5)
            pred_mask, ref_mask = prediction > 0, reference > 0
            union = int(np.count_nonzero(pred_mask | ref_mask))
            foreground_iou = float(np.count_nonzero(pred_mask & ref_mask) / union)
            record = {
                "uid": uid, "group": group, "method": method, "status": "evaluated", "metrics": scores,
                "prediction_path": str(pred_path.resolve()), "prediction_sha256": digest(pred_path),
                "reference_path": str(ref_path.resolve()), "reference_sha256": entry["labels_sha256"],
                "foreground_iou": foreground_iou,
                "foreground_exact": bool(np.array_equal(pred_mask, ref_mask)),
                "canvas_hw": list(reference.shape), "rotation_ccw": 90 * rotation_k, "resized": False,
                "empty_prediction": scores["predicted_parts"] == 0,
                "palette_limit_reached": scores["predicted_parts"] >= len(palette),
                "prediction_label_semantics": "one visible exact RGB color is one anonymous part; disconnected same-color regions share an ID",
                "native_C_part_ids_available": False, "native_prediction_part_ids_available": False,
                "reference_origin": entry.get("origin"),
                "reference_source_status": entry.get("status"),
                "reference_source_ground_truth_ready": entry.get("ground_truth_ready"),
                "metrics_core_sha256": core_hash, "palette_sha256": palette_hash,
            }
            label_folder = "decoded_C_labels" if method == "C_v5_full" else "decoded_labels"
            label_path = output / label_folder / group / uid / "labels_uint16.png"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(prediction).save(label_path)
            record["decoded_prediction_sha256"] = digest(label_path)
            save_json(output / "per_object" / uid / "result.json", record)
            records.append(record)
            if progress and (len(records) % 10 == 0 or index == len(entries)):
                print(json.dumps({"evaluated": len(records), "available": len(available),
                                  "seconds": round(time.monotonic() - started, 1)}), flush=True)
        except Exception as exc:
            error = {"uid": uid, "error": repr(exc)}
            errors.append(error)
            save_json(output / "per_object" / uid / "result.json",
                      {**error, "status": "evaluation_error", "metrics": None})
            if progress:
                print(json.dumps(error), flush=True)
    records.sort(key=lambda row: row["uid"])
    summary = {
        "group": group, "method": method,
        "scope": f"direct 2D {method} color-guide agreement against user-designated D integer labels; no 3D projection",
        "n_reference_expected": len(expected), "n_prediction_files": len(prediction_paths),
        "n_available_pairs": len(available), "n_evaluated": len(records), "n_errors": len(errors),
        "complete_reference_cohort": len(records) == len(expected),
        "missing_prediction_uids": missing,
        "prediction_only_uids": sorted(prediction_paths.keys() - expected), "errors": errors,
        "metrics": aggregate(records),
        "foreground_exact_count": sum(row["foreground_exact"] for row in records),
        "foreground_iou_mean": float(np.mean([row["foreground_iou"] for row in records])) if records else None,
        "foreground_iou_min": min((row["foreground_iou"] for row in records), default=None),
        "palette_limit_uids": [row["uid"] for row in records if row["palette_limit_reached"]],
        "empty_prediction_uids": [row["uid"] for row in records if row["empty_prediction"]],
        "boundary_tolerance_pixels": 2, "part_iou_threshold": 0.5,
        "bootstrap": {"resamples": 2000, "seed": 42, "unit": "object"},
        "input_paths": {"C" if method == "C_v5_full" else method: str(predictions.resolve()), "D": str(references.resolve()),
                        "reference_manifest": str(manifest_path.resolve()), "palette": str(palette_path.resolve()),
                        "metrics_core": str(core_path.resolve())},
        "rotation_ccw": 90 * rotation_k,
        "reference_manifest_sha256": digest(manifest_path),
        "reference_manifest_ground_truth_ready": manifest.get("ground_truth_ready"),
        "reference_origin_counts": dict(Counter(row.get("origin", "unspecified") for row in entries)),
        "reference_source_status_counts": dict(Counter(row.get("status", "unspecified") for row in entries)),
        "metrics_core_sha256": core_hash, "palette_sha256": palette_hash,
        "native_C_part_ids_available": False,
        "color_reuse_limitation": "Only visible prediction colors are available. Distinct native parts sharing a color cannot be recovered or certified absent. No connected-component splitting is applied.",
        "seconds": time.monotonic() - started,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }
    save_json(output / "summary.json", summary)
    save_json(output / "per_object_metrics.json", records)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_object_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        columns = ["uid", *KEYS, "predicted_parts", "reference_parts", "foreground_iou", "foreground_exact", "palette_limit_reached"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({**{key: row[key] for key in ("uid", "foreground_iou", "foreground_exact", "palette_limit_reached")},
                             **{key: row["metrics"][key] for key in (*KEYS, "predicted_parts", "reference_parts")}})
    with (output / "missing_C_uids.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "status"])
        writer.writerows((uid, "C_file_not_present_locally; not scored as an algorithm failure") for uid in missing)
    coverage_note = ("已覆盖清单中的全部 D 参考。" if summary["complete_reference_cohort"] else
                     "当前均值仅覆盖已评估配对，未代表完整 D 参考集。")
    lines = [f"# {method} 与 D {group} 二维引导图评估", "",
             f"D 参考共 {len(expected)} 个 UID；本地预测图共 {len(prediction_paths)} 张；完成 {len(records)} 对。",
             f"缺少预测图 {len(missing)} 个；评估错误 {len(errors)} 个。{coverage_note}", "",
             "| 指标 | 均值（%） | 对象 bootstrap 95% 区间（%） |", "|---|---:|---:|"]
    for key, name in zip(KEYS, ("2D matched mIoU", "2D Boundary F1（2 px）", "Part F1@0.5")):
        if key in summary["metrics"]:
            value = summary["metrics"][key]
            lines.append(f"| {name} | {100 * value['mean']:.2f} | {100 * value['ci95'][0]:.2f}–{100 * value['ci95'][1]:.2f} |")
    lines += ["", f"前景逐像素一致：{summary['foreground_exact_count']}/{len(records)}。",
              f"预测/D 前景 IoU 均值：{100 * summary['foreground_iou_mean']:.4f}%。" if records else "没有可评分配对。", "",
              "预测彩色图按精确 RGB 转整数标签；同色区域使用同一编号，不拆连通分量。D 使用已有整数标签。",
              f"固定逆时针旋转 {90 * rotation_k} 度；没有缩放、镜像或根据 D 分数选择变换。空预测计零；缺少文件单列。",
              "预测的原生部件 ID 未提供；若不同部件复用同色，当前结果衡量其彩色图中的区域划分。不能据彩色 PNG 还原被合并的原生部件。",
              f"达到 {len(palette)} 色上限的样本有 {len(summary['palette_limit_uids'])} 个，详见 summary.json。",
              "", f"预测来源：`{predictions.resolve()}`", f"D 来源：`{references.resolve()}`",
              "", "附 per_object_metrics.csv、missing_C_uids.csv、逐对象记录和 decoded_C_labels。"]
    if any(row.get("origin") for row in entries):
        lines += ["", "D 参考来源按上传清单保留："]
        lines += [f"- `{origin}`：{count} 张。" for origin, count in summary["reference_origin_counts"].items()]
        lines += ["", "原有验收状态保留在逐对象记录中；本次计算不改变参考图的验收状态。"]
    (output / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--reference-manifest", required=True, type=Path)
    parser.add_argument("--palette", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", default="C_v5_full")
    parser.add_argument("--rotation-k", default=0, type=int, choices=(0, 1, 2, 3))
    parser.add_argument("--metrics-core", type=Path)
    args = parser.parse_args()
    result = evaluate(args.predictions, args.references, args.reference_manifest, args.palette, args.output,
                      method=args.method, rotation_k=args.rotation_k, metrics_path=args.metrics_core)
    print(json.dumps({"output": str(args.output.resolve()), "n_evaluated": result["n_evaluated"],
                      "n_errors": result["n_errors"], "metrics": result["metrics"]}), flush=True)

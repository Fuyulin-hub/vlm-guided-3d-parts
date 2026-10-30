"""Run the agreed B/C versus D, category-stratified, direct 2D comparison."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
from zipfile import ZipFile


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


direct = load_module(Path(__file__).with_name("evaluate_c_d_guidance.py"), "category_direct")
GROUPS = ("assemblies", "solid_volume", "thin_fine_grained")
METHODS = ("B_sam2", "C_v5_full")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def paired_summary(records, expected):
    expected = set(expected)
    lookup = {}
    for method in METHODS:
        rows = records[method]
        lookup[method] = {r["uid"]: r for r in rows}
        if len(rows) != len(lookup[method]):
            raise ValueError(f"Duplicate UID in {method}")
        if set(lookup[method]) - expected:
            raise ValueError(f"Unexpected UID in {method}")
    uids = sorted(expected.intersection(*(set(lookup[m]) for m in METHODS)))
    delta = [{"uid": u, "metrics": {k: lookup["C_v5_full"][u]["metrics"][k]
              - lookup["B_sam2"][u]["metrics"][k] for k in direct.KEYS}} for u in uids]
    return {"n_reference": len(expected), "n_paired": len(uids), "paired_uids": uids,
            "missing_by_method": {m: sorted(expected - lookup[m].keys()) for m in METHODS},
            "methods": {m: direct.aggregate([lookup[m][u] for u in uids]) for m in METHODS},
            "delta_C_minus_B": direct.aggregate(delta)}


def reference_entry(root, row, *, origin, receipt, subset=None):
    relative = row.get("labels_file") or row["paths"]["labels"]
    path = (root / relative).resolve()
    expected_hash = row.get("labels_sha256") or row["output_sha256"]["labels"]
    if direct.digest(path) != expected_hash:
        raise ValueError(f"Reference receipt hash mismatch: {path}")
    return {"uid": row["uid"], "labels_file": str(path), "labels_sha256": expected_hash,
            "origin": origin, "status": row.get("status"),
            "ground_truth_ready": row.get("ground_truth_ready", False),
            "source_receipt": str(receipt.resolve()), "source_receipt_sha256": direct.digest(receipt),
            "reference_subset": subset}


def prepare_references(workspace, output):
    downloads = workspace / "downloads"
    roots = {
        "assemblies": downloads / "assemblies_label_upload_repaired_20261023/assemblies",
        "solid_volume": downloads / "D_solid_completion_20261024/server_upload_solid_187_20261024_v2"}
    names = {"assemblies": "manifest.json", "solid_volume": "manifest_solid_repaired_20261024.json"}
    result = {}
    for group, root in roots.items():
        receipt = root / names[group]
        entries = [reference_entry(root, row, origin=row.get("origin", "designated_D_reference"),
                                   receipt=receipt) for row in read(receipt)["per_uid"]]
        result[group] = entries
    batch = downloads / "D_thin_continuation_20261024_batch01"
    entries = []
    for item in read(batch / "existing_63_inventory.json"):
        root = downloads / item["root"]
        receipt = root / item["receipt"]
        data = read(receipt)
        matches = [r for r in data.get("per_uid", data.get("cases", [])) if r["uid"] == item["uid"]]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous thin source: {item}")
        entries.append(reference_entry(root, matches[0], origin=item["root"], receipt=receipt,
                                       subset="existing_63"))
    receipt = batch / "delivery/manifest.json"
    for row in read(receipt)["cases"]:
        entries.append(reference_entry(receipt.parent, row, origin="new_ai_assisted_batch01",
                                       receipt=receipt, subset="new_21_requires_review"))
    result["thin_fine_grained"] = entries
    manifests = {}
    for group, rows in result.items():
        if len({r["uid"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate D reference UID in {group}")
        path = output / "inputs" / f"{group}_references.json"
        direct.save_json(path, {"group": group, "ground_truth_ready": False,
                               "per_uid": sorted(rows, key=lambda r: r["uid"])})
        manifests[group] = path
    return manifests


def prepare_code_and_b(code_zip, b_zip, output):
    code_root, b_root = output / "inputs/evaluation_code_clean", output / "inputs/B_original"
    code_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(code_zip, output / "inputs/evaluation_code_clean.zip")
    code_hashes = {}
    with ZipFile(code_zip) as archive:
        for name in ("evaluate.py", "projection.py", "summarize.py", "paper_table_metrics.py"):
            path = code_root / name
            path.write_bytes(archive.read("evaluation_code_clean/" + name))
            code_hashes[name] = direct.digest(path)
    with ZipFile(b_zip) as archive:
        prefix = "B_sam2_original_guidance_606_glb_paired_20261021/"
        rows = list(csv.DictReader(io.StringIO(archive.read(
            prefix + "manifest_b_sam2_guidance_606_glb_mapping.csv").decode("utf-8-sig"))))
        if len(rows) != 606 or len({(r["group"], r["uid"]) for r in rows}) != 606:
            raise ValueError("B source package must contain 606 unique group/UID rows")
        for row in rows:
            if row["rot90_alignment_transform"] != "CCW90_to_match_A_raw_rot90":
                raise ValueError("B package declares an unexpected orientation")
            group, uid = row["group"], row["uid"]
            if group not in GROUPS or not uid.isalnum():
                raise ValueError("Invalid B source identity")
            path = b_root / group / f"{uid}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(row["guidance_archive_path"]))
            if direct.digest(path) != row["guidance_sha256"]:
                raise ValueError(f"B package image hash mismatch: {uid}")
    return code_root / "paper_table_metrics.py", b_root, code_hashes


def report(output, summary):
    lines = ["# B/C 二维引导图按结构类别评估", "",
             "B：Automatic SAM2；C：用户指定的本地 C_v5_full；D：已有整数参考标签。",
             "本表评估二维引导图，不是 GLB 投影或三维分割指标。单位为 %；提升量为百分点。", "",
             "| 类别 | D 参考数 | B/C 共同样本 | 方法 | matched mIoU | Boundary F1 | Part F1@0.5 |",
             "|---|---:|---:|---|---:|---:|---:|"]
    csv_rows = []
    for group in GROUPS:
        result = summary["groups"][group]
        for method in (*METHODS, "delta_C_minus_B"):
            metrics = result["methods"][method] if method in METHODS else result[method]
            cells = [f"{100 * metrics[k]['mean']:.2f}" if metrics else "未测" for k in direct.KEYS]
            lines.append(f"| {group} | {result['n_reference']} | {result['n_paired']} | {method} | "
                         + " | ".join(cells) + " |")
            csv_rows.append({"group": group, "method": method, "n": result["n_paired"],
                             **{k + "_pct_or_pp": 100 * metrics[k]["mean"] if metrics else None
                                for k in direct.KEYS}})
    lines += ["", "## 口径与覆盖", "",
              "- 按每类 B/C/D 的共同 UID 配对；不将不同样本上的均值直接相减。缺文件单列，不冒充算法失败。",
              "- 按部件区域作一对一匹配，不要求 B/C 与 D 同色或同编号。同色但不相连的预测仍共用一个 ID。",
              "- B 按源包记录统一逆时针旋转 90°；C/D 不变。不按 D 分数选择变换，不缩放、镜像或修补。",
              "- B 原图实际包含原调色板索引 0 的红色，因此精确解码全部 21 个已声明前景色；C 保留原来的索引 1–20 规则。未修改任何输入像素。",
              "- Boundary F1 容差 2 px；Part F1 的 IoU 阈值 0.5；按对象等权平均。",
              "- 均值及 C−B 差值均附 2000 次对象配对 bootstrap 的 95% 区间，seed=42，详见 summary.json。",
              "- D 来源和未验收标记原样保留；thin 的 84 套含已有 63 套及新增 21 套待复核 AI 辅助候选，不能宣称全部为人工验收真值。",
              "- 未用 D 的 GLB 作为真值，未生成/修改 GLB/VXZ，也没有重新运行分割模型。",
              "", "## 文件", "", "- summary.json：分组配对分数、置信区间、提升量、缺失 UID。",
              "- inputs/*_references.json：冻结的标签路径、哈希、来源状态。",
              "- B_sam2/<类别> 与 C_v5_full/<类别>：逐对象指标、解码标签及独立覆盖情况。"]
    for group, result in summary["groups"].items():
        lines += ["", f"### {group}",
                  f"计划共同配对 {result['n_planned_pairs']}，成功配对 {result['n_paired']}；"
                  f"评估错误 {result['evaluation_errors']}。",
                  "缺失 UID：`" + json.dumps(result["missing_by_method"], ensure_ascii=False) + "`。"]
    if not summary["complete_on_available_pairs"]:
        lines += ["", "存在未完成计划配对或评估错误；当前表为部分结果。"]
    (output / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (output / "category_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def finalize(output, code_zip, b_zip, c_root):
    """Re-aggregate verified result files without rerunning models or scoring."""
    core = output / "inputs/evaluation_code_clean/paper_table_metrics.py"
    core_hash = direct.digest(core)
    groups = {}
    for group in GROUPS:
        entries = read(output / "inputs" / f"{group}_references.json")["per_uid"]
        expected = {r["uid"]: r for r in entries}
        records, availability, errors = {}, {}, 0
        for method in METHODS:
            result_root = output / method / group
            state = read(result_root / "summary.json")
            rows = read(result_root / "per_object_metrics.json")
            errors += state["n_errors"]
            prediction_root = (output / "inputs/B_original" if method == "B_sam2" else c_root) / group
            availability[method] = {p.stem for p in prediction_root.glob("*.png")} & expected.keys()
            for row in rows:
                if (row["group"] != group or row["method"] != method or row["uid"] not in expected
                        or row["metrics_core_sha256"] != core_hash
                        or row["reference_sha256"] != expected[row["uid"]]["labels_sha256"]
                        or row["rotation_ccw"] != (90 if method == "B_sam2" else 0)
                        or row["resized"]):
                    raise ValueError(f"Result protocol mismatch: {method}/{group}/{row['uid']}")
                for key in ("prediction", "reference"):
                    if direct.digest(row[key + "_path"]) != row[key + "_sha256"]:
                        raise ValueError(f"Result source changed: {method}/{group}/{row['uid']}")
                if direct.digest(state["input_paths"]["palette"]) != row["palette_sha256"]:
                    raise ValueError("Result palette changed")
            if len(rows) != state["n_evaluated"]:
                raise ValueError("Result count mismatch")
            records[method] = rows
        result = paired_summary(records, expected)
        planned = sorted(set.intersection(*map(set, availability.values())))
        result.update(n_planned_pairs=len(planned), planned_paired_uids=planned,
                      evaluation_errors=errors,
                      independent_method_coverage={m: len(records[m]) for m in METHODS},
                      reference_origin_counts=dict(Counter(r["origin"] for r in entries)))
        if group == "thin_fine_grained":
            existing = {r["uid"] for r in entries if r["reference_subset"] == "existing_63"}
            result["existing_63_sensitivity"] = paired_summary(
                {m: [r for r in records[m] if r["uid"] in existing] for m in METHODS}, existing)
        groups[group] = result
    frozen = read(output / "frozen_inputs.json")
    changed = [r["path"] for r in frozen if direct.digest(r["path"]) != r["sha256"]]
    summary = {"scope": "direct_2d_guidance_not_3d_projection", "groups": groups,
               "complete_on_available_pairs": all(r["paired_uids"] == r["planned_paired_uids"]
                                                  and not r["evaluation_errors"] for r in groups.values()),
               "source_files_unchanged": not changed, "changed_inputs": changed,
               "code_zip_path": str(code_zip.resolve()), "code_zip_sha256": direct.digest(code_zip),
               "supplied_code_sha256": {p.name: direct.digest(p) for p in core.parent.glob("*.py")},
               "B_zip_path": str(b_zip.resolve()), "B_zip_sha256": direct.digest(b_zip),
               "C_root": str(c_root.resolve()),
               "executed_direct_adapter_sha256": direct.digest(output / "inputs/direct_adapter_used.py")
                   if (output / "inputs/direct_adapter_used.py").exists() else direct.digest(Path(direct.__file__)),
               "driver_sha256": direct.digest(Path(__file__)),
               "bootstrap": {"resamples": 2000, "seed": 42, "paired": True},
               "foreground_palette_indices": {"B_sam2": [0, 20], "C_v5_full": [1, 20]},
               "B_rotation_ccw": 90, "C_and_D_rotation_ccw": 0, "resized": False,
               "D_ground_truth_ready_promoted": False}
    direct.save_json(output / "summary.json", summary)
    report(output, summary)
    if changed:
        raise RuntimeError("Inputs changed during scoring; inspect changed_inputs")
    print(json.dumps({"output": str(output.resolve()), "complete": summary["complete_on_available_pairs"],
                      "paired_counts": {g: r["n_paired"] for g, r in groups.items()}}), flush=True)
    return summary


def run(workspace, code_zip, b_zip, c_root, output, workers=3):
    output.mkdir(parents=True, exist_ok=True)
    manifests = prepare_references(workspace, output)
    core, b_root, code_hashes = prepare_code_and_b(code_zip, b_zip, output)
    palette = workspace / "downloads/D_human_fixed_bc20_294_20261023/colors_pure_bc.json"
    b_palette = output / "inputs/B_decoder_all_21_declared_colors.json"
    direct.save_json(b_palette, [[0, 0, 0, 255]] + read(palette))
    jobs, planned, frozen = {}, {}, []
    for group in GROUPS:
        expected = {r["uid"] for r in read(manifests[group])["per_uid"]}
        available = {}
        for method, root, rotation in (("B_sam2", b_root, 1), ("C_v5_full", c_root, 0)):
            files = {p.stem: p for p in (root / group).glob("*.png")}
            available[method] = expected & files.keys()
            frozen.extend({"path": str(files[u].resolve()), "sha256": direct.digest(files[u])}
                          for u in sorted(available[method]))
            jobs[(method, group)] = dict(predictions=root / group, references=workspace,
                manifest_path=manifests[group], palette_path=b_palette if method == "B_sam2" else palette,
                output=output / method / group,
                method=method, rotation_k=rotation, metrics_path=core)
        planned[group] = sorted(set.intersection(*map(set, available.values())))
    for manifest in manifests.values():
        frozen.extend({"path": row["labels_file"], "sha256": row["labels_sha256"]}
                      for row in read(manifest)["per_uid"])
    direct.save_json(output / "frozen_inputs.json", frozen)
    summaries = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(direct.evaluate, **args): key for key, args in jobs.items()}
        for future in as_completed(futures):
            method, group = futures[future]
            summaries[(method, group)] = future.result()
            print(json.dumps({"finished": method, "group": group,
                              "n": summaries[(method, group)]["n_evaluated"]}), flush=True)
    return finalize(output, code_zip, b_zip, c_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--code-zip", required=True, type=Path)
    parser.add_argument("--b-zip", required=True, type=Path)
    parser.add_argument("--c-root", type=Path, default=Path("D:/c_v5_full_guidance/C_v5_full"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        result = finalize(args.output.resolve(), args.code_zip, args.b_zip, args.c_root)
    else:
        result = run(args.workspace.resolve(), args.code_zip, args.b_zip, args.c_root,
                     args.output.resolve(), args.workers)
    raise SystemExit(0 if result["complete_on_available_pairs"] else 1)

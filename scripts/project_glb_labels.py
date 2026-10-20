#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 SegViGen 分割 GLB 投影为离散部件标签图与可复现相机文件。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# 必须在导入 pyrender / OpenGL 前指定无头 EGL 后端。
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np


GROUPS = ("assemblies", "solid_volume", "thin_fine_grained")
DEFAULT_ROOT = Path("/data2/codefile/fuyulin/experiments/objaverse_684")
DEFAULT_MANIFEST = DEFAULT_ROOT / "manifest_frozen.csv"
DEFAULT_PALETTE = Path("/data2/codefile/fuyulin/1trellis2/segvigen_inputs/colors_pure.json")
DEFAULT_REFERENCE_ROOT = Path("/data2/codefile/fuyulin/experiments/human_fixed_710/masks/C_qwen_sam2")


def parse_bool(value: str | bool) -> bool:
    """解析显式命令行布尔值，避免 bool("false") == True。"""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"无法解析布尔值：{value!r}")


def view_azimuths(views: int) -> list[float]:
    """从 45° 起按 360/N 均匀生成方位角。"""
    if not isinstance(views, int) or views < 1:
        raise ValueError("--views 必须是正整数")
    step = 360.0 / views
    return [float((45.0 + index * step) % 360.0) for index in range(views)]


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"无法归一化{name}")
    return vector / norm


def camera_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """按项目 OpenGL c2w 约定构造相机位姿。"""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if eye.shape != (3,) or target.shape != (3,) or not np.all(np.isfinite([eye, target])):
        raise ValueError("eye 和 target 必须是有限三维向量")
    forward = _unit(target - eye, "相机 forward")
    right = _unit(np.cross(forward, np.array([0.0, 0.0, 1.0])), "相机 right")
    up = _unit(np.cross(right, forward), "相机 up")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def camera_parameters(
    bounds: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    yfov_deg: float,
    size: int,
) -> dict[str, Any]:
    """由场景包围盒生成相机参数，严格沿用项目渲染口径。"""
    bounds = np.asarray(bounds, dtype=np.float64)
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[1] < bounds[0])
    ):
        raise ValueError("场景包围盒必须是有限且有序的 (2, 3) 数组")
    if not isinstance(size, int) or size < 1:
        raise ValueError("图像尺寸必须是正整数")
    if not np.isfinite(yfov_deg) or not 0.0 < yfov_deg < 180.0:
        raise ValueError("yfov 必须位于 (0, 180) 度")

    target = (bounds[0] + bounds[1]) * 0.5
    radius = float(np.linalg.norm(bounds[1] - bounds[0]) * 0.5)
    if not np.isfinite(radius) or radius <= 1e-12:
        raise ValueError("场景包围盒对角线长度必须非零")
    yfov = math.radians(float(yfov_deg))
    distance = radius / math.sin(yfov * 0.5) * 1.18
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    direction = np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )
    eye = target + distance * direction
    c2w = camera_pose(eye, target)
    znear = distance - radius * 2.2
    zfar = distance + radius * 2.2
    if not 0.0 < znear < zfar:
        raise ValueError("由指定相机公式得到的 znear/zfar 非法")
    focal = float(size) / (2.0 * math.tan(yfov * 0.5))
    intrinsic = np.array(
        [[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return {
        "azimuth_deg": float(azimuth_deg),
        "elevation_deg": float(elevation_deg),
        "yfov_deg": float(yfov_deg),
        "image_size": [int(size), int(size)],
        "distance": distance,
        "radius": radius,
        "target": target.tolist(),
        "eye": eye.tolist(),
        "c2w": c2w.tolist(),
        "K": intrinsic.tolist(),
        "znear": float(znear),
        "zfar": float(zfar),
    }


def load_palette(path: str | Path) -> np.ndarray:
    """读取 RGB/RGBA JSON 调色板，并返回 P×3 uint8；index 0 保留给背景。"""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    array = np.asarray(raw)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] not in (3, 4):
        raise ValueError(f"调色板必须是至少两行的 RGB/RGBA 数组：{path}")
    if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0) or np.any(array > 255):
        raise ValueError(f"调色板颜色必须是 0..255 整数：{path}")
    return array[:, :3].astype(np.uint8, copy=False)


def load_label_mapping(
    annotation_path: str | Path | None,
    palette: np.ndarray,
) -> tuple[dict[int, int], dict[str, Any]]:
    """建立 palette_index→label_id；annotation 缺失时退化为匿名调色板编号。"""
    palette = np.asarray(palette, dtype=np.uint8)
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < 2:
        raise ValueError("palette 必须为 P×3，且至少包含背景槽和一种前景色")
    path = Path(annotation_path) if annotation_path is not None else None
    if path is None or not path.is_file():
        return (
            {index: index for index in range(1, len(palette))},
            {
                "source": "palette_identity_fallback",
                "annotation": str(path) if path is not None else None,
                "reason": "annotation_missing",
                "mapping_semantics": "palette_index_surrogate",
                "palette_collisions": [],
            },
        )

    document = json.loads(path.read_text(encoding="utf-8"))
    parts = document.get("parts") or []
    if not parts:
        return (
            {index: index for index in range(1, len(palette))},
            {
                "source": "palette_identity_fallback",
                "annotation": str(path),
                "reason": "parts_empty",
                "mapping_semantics": "palette_index_surrogate",
                "palette_collisions": [],
            },
        )

    labels_by_palette: dict[int, set[int]] = {}
    for part in parts:
        label_id = int(part["label_id"])
        if not 1 <= label_id <= np.iinfo(np.uint16).max:
            raise ValueError(f"label_id 超出 uint16 正数范围：{label_id}")
        expected = (label_id - 1) % (len(palette) - 1) + 1
        palette_index = int(part.get("palette_index", expected))
        if palette_index != expected or not 1 <= palette_index < len(palette):
            raise ValueError(
                f"label_id={label_id} 的 palette_index={palette_index} 与公式期望 {expected} 不一致"
            )
        color = part.get("color")
        if color is not None:
            color_rgb = np.asarray(color, dtype=np.int64).reshape(-1)[:3]
            if color_rgb.shape != (3,) or not np.array_equal(color_rgb, palette[palette_index]):
                raise ValueError(f"label_id={label_id} 的 color 与 palette_index={palette_index} 不一致")
        labels_by_palette.setdefault(palette_index, set()).add(label_id)

    # 调色板循环复用时，同一颜色无法从渲染像素反推多个 label_id。
    # 为保证批量投影可完成，确定性地合并到最小 label_id，并把不可逆信息完整写入元数据。
    mapping = {
        palette_index: min(label_ids)
        for palette_index, label_ids in sorted(labels_by_palette.items())
    }
    collisions = [
        {
            "palette_index": int(palette_index),
            "label_ids": sorted(int(label_id) for label_id in label_ids),
            "selected_label_id": int(min(label_ids)),
        }
        for palette_index, label_ids in sorted(labels_by_palette.items())
        if len(label_ids) > 1
    ]
    return mapping, {
        "source": "annotation",
        "annotation": str(path),
        "reason": "palette_cycle_collision_merged" if collisions else None,
        "mapping_semantics": (
            "annotation_label_id_merged_by_palette_color"
            if collisions
            else "annotation_label_id"
        ),
        "palette_collisions": collisions,
    }


def quantize_palette(
    rgb: np.ndarray,
    foreground: np.ndarray,
    palette: np.ndarray,
    threshold: float,
    *,
    chunk_pixels: int = 262_144,
) -> tuple[np.ndarray, dict[str, Any]]:
    """把前景 RGB 确定性量化到 palette[1:]；超阈值与背景均输出 0。"""
    rgb = np.asarray(rgb)
    foreground = np.asarray(foreground, dtype=bool)
    palette = np.asarray(palette)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.shape[:2] != foreground.shape:
        raise ValueError("rgb 必须为 H×W×3/4，foreground 必须为同尺寸二维数组")
    if palette.ndim != 2 or palette.shape[1] != 3 or len(palette) < 2:
        raise ValueError("palette 必须为至少两行的 P×3 数组")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("颜色距离阈值必须是有限非负数")
    if chunk_pixels < 1:
        raise ValueError("chunk_pixels 必须为正整数")

    output = np.zeros(foreground.shape, dtype=np.uint16)
    flat_positions = np.flatnonzero(foreground.reshape(-1))
    colors = rgb[:, :, :3].reshape(-1, 3)[flat_positions].astype(np.int32, copy=False)
    candidates = palette[1:, :3].astype(np.int32, copy=False)
    threshold_sq = float(threshold) ** 2
    selected = np.zeros(len(flat_positions), dtype=np.uint16)
    for start in range(0, len(flat_positions), chunk_pixels):
        stop = min(start + chunk_pixels, len(flat_positions))
        diff = colors[start:stop, None, :] - candidates[None, :, :]
        distance_sq = np.sum(diff * diff, axis=2, dtype=np.int64)
        # np.argmin 在并列时返回第一个位置，因此固定选择更小的 palette_index。
        nearest_zero_based = np.argmin(distance_sq, axis=1)
        nearest_distance = distance_sq[np.arange(stop - start), nearest_zero_based]
        hit = nearest_distance <= threshold_sq
        selected[start:stop] = np.where(hit, nearest_zero_based + 1, 0).astype(np.uint16)
    output.reshape(-1)[flat_positions] = selected
    foreground_pixels = int(len(flat_positions))
    hit_pixels = int(np.count_nonzero(selected))
    if foreground_pixels and hit_pixels == 0:
        raise RuntimeError("非空渲染前景未命中任何调色板颜色，请检查颜色阈值或源纹理")
    return output, {
        "foreground_pixels": foreground_pixels,
        "hit_pixels": hit_pixels,
        "unknown_pixels": foreground_pixels - hit_pixels,
        "hit_rate": float(hit_pixels / foreground_pixels) if foreground_pixels else 0.0,
    }


def palette_indices_to_labels(
    palette_indices: np.ndarray,
    mapping: dict[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """按显式映射把 palette index 转为 uint16 label_id，未映射颜色写 0。"""
    palette_indices = np.asarray(palette_indices)
    if palette_indices.ndim != 2 or not np.issubdtype(palette_indices.dtype, np.integer):
        raise ValueError("palette_indices 必须是二维整数数组")
    labels = np.zeros(palette_indices.shape, dtype=np.uint16)
    for palette_index, label_id in sorted(mapping.items()):
        if not 1 <= int(palette_index) <= np.iinfo(np.uint16).max:
            raise ValueError(f"非法 palette_index：{palette_index}")
        if not 1 <= int(label_id) <= np.iinfo(np.uint16).max:
            raise ValueError(f"非法 label_id：{label_id}")
        labels[palette_indices == int(palette_index)] = np.uint16(label_id)
    palette_hit_pixels = int(np.count_nonzero(palette_indices))
    mapped_pixels = int(np.count_nonzero(labels))
    return labels, {
        "palette_hit_pixels": palette_hit_pixels,
        "mapped_pixels": mapped_pixels,
        "unmapped_pixels": palette_hit_pixels - mapped_pixels,
        "mapping_rate": float(mapped_pixels / palette_hit_pixels) if palette_hit_pixels else 0.0,
    }


def foreground_from_render(rgba: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """用深度确定可见几何前景，并按要求拒绝全空 alpha 渲染。"""
    rgba = np.asarray(rgba)
    depth = np.asarray(depth)
    if rgba.ndim != 3 or rgba.shape[2] != 4 or depth.shape != rgba.shape[:2]:
        raise ValueError("渲染结果必须是同尺寸 RGBA 与 depth")
    alpha_nonempty = rgba[:, :, 3] > 0
    # MSAA resolve 可能使轮廓像素的 alpha 与 depth 不完全同步；depth>0 才是几何前景。
    foreground = np.isfinite(depth) & (depth > 0)
    if not np.any(alpha_nonempty) or not np.any(foreground):
        raise RuntimeError("渲染得到全空 alpha/depth，无法生成标签图")
    return foreground


def _safe_component(value: str, field: str) -> str:
    value = str(value).strip()
    if (
        not value
        or value in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9._-]+", value)
        or ".." in value
    ):
        raise ValueError(f"{field} 不是安全的路径组件：{value!r}")
    return value


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """读取并校验固定测试集 manifest。"""
    path = Path(path)
    required = {"group", "uid", "input_rgba", "width", "height", "mode", "format"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"manifest 缺少列：{missing}")
        records: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            group = _safe_component(row["group"], "group")
            uid = _safe_component(row["uid"], "uid")
            if group not in GROUPS:
                raise ValueError(f"manifest 第 {row_number} 行 group 非法：{group}")
            key = (group, uid)
            if key in seen:
                raise ValueError(f"manifest 存在重复对象：{group}/{uid}")
            seen.add(key)
            records.append(dict(row, group=group, uid=uid))
    return records


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hungarian_maximize(weights: np.ndarray) -> list[tuple[int, int]]:
    """确定性的 O(n^3) Hungarian 最大权匹配，不依赖 SciPy。"""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or not np.all(np.isfinite(weights)):
        raise ValueError("匹配权重必须是有限二维数组")
    rows, cols = weights.shape
    if not rows or not cols:
        return []
    size = max(rows, cols)
    padded = np.zeros((size, size), dtype=np.float64)
    padded[:rows, :cols] = weights
    cost = float(np.max(padded)) - padded

    # 1-indexed Hungarian 最小费用实现；扫描顺序固定，平局时选择较小列。
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    p = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)
    for i in range(1, size + 1):
        p[0] = i
        minv = np.full(size + 1, np.inf, dtype=np.float64)
        used = np.zeros(size + 1, dtype=bool)
        j0 = 0
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = np.inf
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(int(p[j] - 1), j - 1) for j in range(1, size + 1)]
    return [(row, col) for row, col in pairs if row < rows and col < cols]


def reference_consistency(prediction: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """对编号无关的部件分区做 Hungarian IoU 一致性校验。"""
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if prediction.ndim != 2 or reference.ndim != 2:
        raise ValueError("预测和参考标签图必须是二维数组")
    if prediction.shape != reference.shape:
        return {
            "status": "incompatible_shape",
            "prediction_shape": list(prediction.shape),
            "reference_shape": list(reference.shape),
        }
    pred_fg = prediction > 0
    ref_fg = reference > 0
    intersection = int(np.count_nonzero(pred_fg & ref_fg))
    union = int(np.count_nonzero(pred_fg | ref_fg))
    foreground_iou = float(intersection / union) if union else 1.0
    pred_ids = np.unique(prediction[pred_fg])
    ref_ids = np.unique(reference[ref_fg])
    if not len(ref_ids):
        return {
            "status": "reference_has_no_parts",
            "foreground_iou": foreground_iou,
            "prediction_parts": int(len(pred_ids)),
            "reference_parts": 0,
        }
    matrix = np.zeros((len(pred_ids), len(ref_ids)), dtype=np.float64)
    for pred_index, pred_id in enumerate(pred_ids):
        pred_mask = prediction == pred_id
        for ref_index, ref_id in enumerate(ref_ids):
            ref_mask = reference == ref_id
            part_intersection = int(np.count_nonzero(pred_mask & ref_mask))
            part_union = int(np.count_nonzero(pred_mask | ref_mask))
            matrix[pred_index, ref_index] = part_intersection / part_union if part_union else 0.0
    pairs = _hungarian_maximize(matrix)
    matched_sum = float(sum(matrix[row, col] for row, col in pairs))
    return {
        "status": "ok",
        "protocol": "hungarian_max_iou_label_permutation_invariant",
        "foreground_iou": foreground_iou,
        "matched_miou": matched_sum / len(ref_ids),
        "prediction_parts": int(len(pred_ids)),
        "reference_parts": int(len(ref_ids)),
        "matched_pairs": [
            {
                "prediction_label": int(pred_ids[row]),
                "reference_label": int(ref_ids[col]),
                "iou": float(matrix[row, col]),
            }
            for row, col in pairs
            if matrix[row, col] > 0
        ],
    }


def _write_json_temp(path: Path, document: dict[str, Any]) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _write_png_temp(path: Path, labels: np.ndarray) -> Path:
    from PIL import Image

    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype != np.uint16:
        raise ValueError("标签图必须是二维 np.uint16")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    Image.fromarray(labels).save(temporary, format="PNG")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    return temporary


def write_projection_bundle(
    output_dir: str | Path,
    labels_by_view: dict[str, np.ndarray],
    cameras: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """原子提交一个对象的全部输出，并把 v00 作为最后缓存哨兵。"""
    output_dir = Path(output_dir)
    if "v00" not in labels_by_view:
        raise ValueError("输出至少需要 v00 标签图")
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_files: dict[Path, Path] = {}
    previous_sentinel: Path | None = None
    try:
        for view_id, labels in sorted(labels_by_view.items()):
            if not re.fullmatch(r"v\d{2,}", view_id):
                raise ValueError(f"非法 view_id：{view_id}")
            destination = output_dir / f"label_{view_id}.png"
            temporary_files[destination] = _write_png_temp(destination, labels)
        camera_path = output_dir / "cameras.json"
        meta_path = output_dir / "projection_meta.json"
        temporary_files[camera_path] = _write_json_temp(camera_path, cameras)
        temporary_files[meta_path] = _write_json_temp(meta_path, metadata)

        sentinel = output_dir / "label_v00.png"
        # 显式覆盖时先撤下旧缓存哨兵。若后续提交中断，下一次运行不会误判为完整缓存。
        if sentinel.exists():
            previous_sentinel = output_dir / f".{sentinel.name}.{uuid.uuid4().hex}.previous"
            os.replace(sentinel, previous_sentinel)
        for destination in sorted(path for path in temporary_files if path != sentinel):
            os.replace(temporary_files[destination], destination)
        expected_labels = {f"label_{view_id}.png" for view_id in labels_by_view}
        for stale_label in output_dir.glob("label_v*.png"):
            if stale_label.name not in expected_labels:
                stale_label.unlink()
        os.replace(temporary_files[sentinel], sentinel)
    finally:
        for temporary in temporary_files.values():
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        if previous_sentinel is not None:
            try:
                previous_sentinel.unlink(missing_ok=True)
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 SegViGen 分割 GLB 投影为离散部件标签图与相机文件"
    )
    parser.add_argument("--arm", required=True, help="结果臂，例如 B_sam2 或 C_v5_full")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow-overwrite", nargs="?", const=True, default=False, type=parse_bool
    )
    parser.add_argument(
        "--with-reference", nargs="?", const=True, default=False, type=parse_bool
    )
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT)
    parser.add_argument(
        "--color-threshold",
        type=float,
        default=64.0,
        help="RGB 欧氏距离阈值（默认 64，经真实 SegViGen 纹理校准）；超过阈值记为 0",
    )
    return parser


def configure_egl_device(device: str) -> int:
    """把 cuda:N 解析为 EGL 设备编号；必须在导入 pyrender 前调用。"""
    match = re.fullmatch(r"(?:cuda:)?(\d+)", str(device).strip().lower())
    if not match:
        raise ValueError(f"--device 必须形如 cuda:0 或 0：{device!r}")
    index = int(match.group(1))
    os.environ["EGL_DEVICE_ID"] = str(index)
    return index


def load_textured_scene(glb_path: str | Path):
    """读取 GLB，确认每个几何体都携带可投影的纹理视觉。"""
    import trimesh

    glb_path = Path(glb_path)
    loaded = trimesh.load(glb_path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.Scene(loaded)
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise ValueError(f"GLB 不含可渲染几何体：{glb_path}")
    for name, geometry in loaded.geometry.items():
        visual = getattr(geometry, "visual", None)
        if getattr(visual, "kind", None) != "texture" or getattr(visual, "uv", None) is None:
            raise ValueError(f"几何体 {name!r} 不是带 UV 的 TextureVisuals")
        material = getattr(visual, "material", None)
        texture = getattr(material, "baseColorTexture", None)
        if texture is None:
            texture = getattr(material, "image", None)
        if texture is None:
            raise ValueError(f"几何体 {name!r} 缺少 baseColorTexture")
    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    # 复用相机校验，提前拒绝 NaN/退化场景。
    camera_parameters(bounds, 45.0, 20.0, 42.0, 16)
    return loaded, bounds


def prepare_pyrender_scene(trimesh_scene):
    """把 trimesh.Scene 转为无光照纹理平涂场景，不修改源 GLB。"""
    import pyrender
    from pyrender.constants import GLTF

    scene = pyrender.Scene.from_trimesh_scene(
        trimesh_scene,
        bg_color=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        ambient_light=np.zeros(3, dtype=np.float32),
    )
    primitive_count = 0
    for node in scene.mesh_nodes:
        for primitive in node.mesh.primitives:
            primitive_count += 1
            material = primitive.material
            texture = getattr(material, "baseColorTexture", None)
            if texture is None:
                raise ValueError("pyrender 转换后丢失 baseColorTexture")
            if getattr(primitive, "color_0", None) is not None:
                raise ValueError("检测到会与 baseColorTexture 相乘的 vertex color，无法保证纯色标签")
            # flat.frag 会把 factor、纹理和顶点色相乘；固定为白色以直出纹理颜色。
            material.baseColorFactor = np.ones(4, dtype=np.float32)
            sampler = getattr(texture, "sampler", None)
            if sampler is None:
                texture.sampler = pyrender.Sampler(
                    magFilter=GLTF.NEAREST,
                    minFilter=GLTF.NEAREST,
                )
            else:
                # 仅禁用插值；保留 GLB 自带的 wrapS/wrapT，避免 UV 边缘串色。
                sampler.magFilter = GLTF.NEAREST
                sampler.minFilter = GLTF.NEAREST
    if not primitive_count:
        raise ValueError("pyrender 场景没有 mesh primitive")
    return scene


def create_renderer(size: int):
    """创建方形 EGL 离屏渲染器。"""
    if not isinstance(size, int) or size < 1:
        raise ValueError("--size 必须是正整数")
    import pyrender

    return pyrender.OffscreenRenderer(viewport_width=size, viewport_height=size)


@contextmanager
def object_renderer(size: int):
    """每个对象独立持有 EGL context，避免跨对象纹理显存累积。"""
    renderer = create_renderer(size)
    try:
        yield renderer
    finally:
        renderer.delete()


def render_flat(renderer, scene, camera: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """以 FLAT|RGBA 渲染带 baseColorTexture 的场景。"""
    import pyrender

    perspective = pyrender.PerspectiveCamera(
        yfov=math.radians(float(camera["yfov_deg"])),
        aspectRatio=1.0,
        znear=float(camera["znear"]),
        zfar=float(camera["zfar"]),
    )
    camera_node = scene.add(perspective, pose=np.asarray(camera["c2w"], dtype=np.float64))
    try:
        color, depth = renderer.render(
            scene,
            flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA,
        )
    finally:
        scene.remove_node(camera_node)
    return np.asarray(color), np.asarray(depth)


def query_gpu_memory_mb(gpu_index: int) -> int | None:
    """单次查询 GPU 总显存占用；不轮询、不找卡、不据此启动任务。"""
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={int(gpu_index)}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return int(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _camera_json_record(view_id: str, camera: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "azimuth_deg",
        "elevation_deg",
        "yfov_deg",
        "image_size",
        "distance",
        "target",
        "eye",
        "c2w",
        "K",
        "znear",
        "zfar",
    )
    return {"view_id": view_id, **{key: camera[key] for key in keys}}


def _read_reference(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"参考标签图必须是二维整数 PNG：{path}")
    return array.astype(np.uint16, copy=False)


def process_one(
    record: dict[str, str],
    args: argparse.Namespace,
    palette: np.ndarray,
) -> dict[str, Any]:
    """处理单个对象并返回一条事件日志；异常交给主循环记录 failed。"""
    started = time.perf_counter()
    group = _safe_component(record["group"], "group")
    uid = _safe_component(record["uid"], "uid")
    arm = _safe_component(args.arm, "arm")
    result_dir = Path(args.root) / "results" / arm / group / uid
    glb_path = result_dir / "segmented.glb"
    projection_dir = result_dir / "projection"
    sentinel = projection_dir / "label_v00.png"
    event_base = {"group": group, "uid": uid, "arm": arm}

    if not glb_path.is_file():
        return {
            **event_base,
            "status": "skipped_missing",
            "seconds": time.perf_counter() - started,
            "output": str(projection_dir),
            "peak_gpu_mb": None,
            "error": None,
        }
    if sentinel.is_file() and not args.allow_overwrite:
        return {
            **event_base,
            "status": "skipped_cached",
            "seconds": time.perf_counter() - started,
            "output": str(projection_dir),
            "peak_gpu_mb": None,
            "error": None,
        }
    annotation_path = Path(args.root) / "annotations" / arm / group / f"{uid}.json"
    mapping, mapping_info = load_label_mapping(annotation_path, palette)
    trimesh_scene, bounds = load_textured_scene(glb_path)
    gpu_index = configure_egl_device(args.device)
    render_scene = prepare_pyrender_scene(trimesh_scene)
    memory_samples = [query_gpu_memory_mb(gpu_index)]
    render_started = time.perf_counter()
    labels_by_view: dict[str, np.ndarray] = {}
    camera_records: list[dict[str, Any]] = []
    view_stats: dict[str, dict[str, Any]] = {}
    with object_renderer(args.size) as renderer:
        for view_index, azimuth in enumerate(view_azimuths(args.views)):
            view_id = f"v{view_index:02d}"
            camera = camera_parameters(bounds, azimuth, 20.0, 42.0, args.size)
            rgba, depth = render_flat(renderer, render_scene, camera)
            foreground = foreground_from_render(rgba, depth)
            palette_indices, quantization = quantize_palette(
                rgba[:, :, :3], foreground, palette, args.color_threshold
            )
            labels, label_mapping = palette_indices_to_labels(palette_indices, mapping)
            labels_by_view[view_id] = labels
            camera_records.append(_camera_json_record(view_id, camera))
            view_stats[view_id] = {**quantization, **label_mapping}
        # renderer/纹理仍驻留时做一次 nvidia-smi 快照；不轮询、不自动选卡。
        memory_samples.append(query_gpu_memory_mb(gpu_index))
    measured = [value for value in memory_samples if value is not None]
    peak_gpu_mb = max(measured) if measured else None
    render_seconds = time.perf_counter() - render_started

    foreground_pixels = sum(int(stats["foreground_pixels"]) for stats in view_stats.values())
    hit_pixels = sum(int(stats["hit_pixels"]) for stats in view_stats.values())
    mapped_pixels = sum(int(stats["mapped_pixels"]) for stats in view_stats.values())
    reference_info: dict[str, Any] | None = None
    if args.with_reference:
        reference_path = Path(args.reference_root) / group / uid / "labels_uint16.png"
        if reference_path.is_file():
            reference_info = {
                "path": str(reference_path),
                **reference_consistency(labels_by_view["v00"], _read_reference(reference_path)),
            }
        else:
            reference_info = {"path": str(reference_path), "status": "missing"}

    cameras = {"views": camera_records}
    mapping_table = [
        {
            "palette_index": int(palette_index),
            "label_id": int(label_id),
            "color": palette[palette_index].astype(int).tolist(),
        }
        for palette_index, label_id in sorted(mapping.items())
    ]
    metadata = {
        "schema_version": "glb_label_projection@1",
        "arm": arm,
        "group": group,
        "uid": uid,
        "source_glb": {"path": str(glb_path), "sha256": sha256_file(glb_path)},
        "palette": {
            "path": str(args.palette),
            "sha256": sha256_file(args.palette),
            "index_0_reserved_for_background": True,
            "entries": int(len(palette)),
        },
        "annotation": {
            **mapping_info,
            "exists": annotation_path.is_file(),
        },
        "label_id_mapping": mapping_table,
        "quantization": {
            "method": "deterministic_nearest_palette_rgb",
            "tie_break": "smallest_palette_index",
            "distance": "euclidean_rgb",
            "threshold": float(args.color_threshold),
            "unknown_output_label": 0,
            "per_view": view_stats,
            "aggregate": {
                "foreground_pixels": foreground_pixels,
                "hit_pixels": hit_pixels,
                "unknown_pixels": foreground_pixels - hit_pixels,
                "mapped_pixels": mapped_pixels,
                "hit_rate": float(hit_pixels / foreground_pixels) if foreground_pixels else 0.0,
                "mapping_rate": float(mapped_pixels / hit_pixels) if hit_pixels else 0.0,
            },
        },
        "render": {
            "backend": "egl",
            "flags": ["FLAT", "RGBA"],
            "texture_filter": "NEAREST",
            "texture_wrap": "preserved_from_glb",
            "base_color_factor": "forced_white_in_memory",
            "views": int(args.views),
            "size": [int(args.size), int(args.size)],
            "seconds": render_seconds,
        },
        "peak_gpu_mb": peak_gpu_mb,
        "peak_gpu_method": "nvidia-smi_whole_device_memory_snapshots_before_and_during_object",
        "peak_gpu_scope": "whole_device_observed_max_not_process_isolated",
        "device": str(args.device),
        "reference_consistency": reference_info,
    }
    write_projection_bundle(projection_dir, labels_by_view, cameras, metadata)
    return {
        **event_base,
        "status": "ok",
        "seconds": time.perf_counter() - started,
        "output": str(projection_dir),
        "peak_gpu_mb": peak_gpu_mb,
        "error": None,
    }


def append_event(path: str | Path, event: dict[str, Any]) -> None:
    """带进程锁追加一条 JSONL，并立即 flush/fsync。"""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.arm = _safe_component(args.arm, "arm")
    if args.views < 1:
        raise SystemExit("--views 必须是正整数")
    if args.size < 1:
        raise SystemExit("--size 必须是正整数")
    if not np.isfinite(args.color_threshold) or args.color_threshold < 0:
        raise SystemExit("--color-threshold 必须是有限非负数")
    configure_egl_device(args.device)
    records = load_manifest(args.manifest)
    palette = load_palette(args.palette)
    output_root = Path(args.root) / "results" / args.arm
    log_path = Path(args.root) / "logs" / "projection_events.jsonl"
    print("==== GLB 部件标签投影配置 ====")
    print(f"arm: {args.arm}")
    print(f"对象数: {len(records)}")
    print(f"视角数: {args.views}")
    print(f"图像尺寸: {args.size}×{args.size}")
    print(f"输出根目录: {output_root}")
    print(f"颜色阈值: {args.color_threshold}")

    counts: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        try:
            event = process_one(record, args, palette)
        except Exception as exc:  # 单对象失败不影响后续对象
            event = {
                "group": record.get("group"),
                "uid": record.get("uid"),
                "arm": args.arm,
                "status": "failed",
                "seconds": time.perf_counter() - started,
                "output": str(
                    Path(args.root)
                    / "results"
                    / args.arm
                    / str(record.get("group"))
                    / str(record.get("uid"))
                    / "projection"
                ),
                "peak_gpu_mb": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        append_event(log_path, event)
        counts[event["status"]] = counts.get(event["status"], 0) + 1
        print(
            f"[{index}/{len(records)}] {event['group']}/{event['uid']} "
            f"{event['status']} ({event['seconds']:.2f}s)"
        )
    print("==== 完成 ====")
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    print(f"事件日志: {log_path}")
    return 0 if counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

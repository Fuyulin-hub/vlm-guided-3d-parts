#!/usr/bin/env python3
"""C_v5: audited multi-point + negative-point + feedback guidance for SegViGen.

This runner is deliberately isolated from C_v4.  It preserves the Qwen3-VL and
SAM2 models, checkpoints, seed, and single-view protocol; the only behavioural
changes are (1) full multi-positive/negative-point prompting followed by at
most one feedback pass, and (2) point-aware, small/tight-mask-first fusion.

The module-level helpers are CPU-only so that the fusion contract can be tested
before any manually reserved GPU run.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path("/data2/codefile/fuyulin/experiments/objaverse_684")
SCHEMA_CONFIG = "c_v5_config@1"
SCHEMA_ANNOTATION = "c_v5_annotation@1"
SCHEMA_RESULT = "c_v5_result@1"
FUSION_STRATEGY = "point_connected_small_bbox_first@1"
PROMPT_POLICY = "multi_point_negative_feedback"


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible metadata deterministically."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_frozen_config(
    *,
    run_id: str,
    manifest_path: str,
    manifest_sha256: str,
    explicit_uids: list[str],
    output_suffix: str,
) -> dict[str, Any]:
    """Return the versioned, auditable C_v5 contract.

    This is intentionally not a general ablation config: changing either prompt
    policy or fusion policy requires a new run id and a new schema/config hash.
    """
    config: dict[str, Any] = {
        "schema_version": SCHEMA_CONFIG,
        "run_id": run_id,
        "method_name": "Qwen3-VL-guided SAM2 feedback with point-aware fusion",
        "output_suffix": output_suffix,
        "manifest": {
            "path": manifest_path,
            "sha256": manifest_sha256,
            "explicit_uids": explicit_uids,
        },
        "prompting": {
            "policy": PROMPT_POLICY,
            "uses_bbox": True,
            "uses_all_positive_points": True,
            "uses_all_negative_points": True,
            "max_feedback_rounds": 1,
            "on_feedback_parse_failure": "mark_partial_never_silent_success",
        },
        "sam2": {
            "multimask_output": False,
            "mask_selection": "single_mask_to_isolate_prompt_and_fusion_changes",
        },
        "fusion": {
            "strategy": FUSION_STRATEGY,
            "candidate_cleanup": "foreground_clip+point_component+3x3_close+small_hole_fill",
            "priority": "bbox_area_then_cleaned_area_then_part_id_ascending",
            "candidate_min_pixels": "max(user_min,96,0.00012*foreground_pixels)",
            "overlap_min_pixels": "max(user_min,96,0.00010*foreground_pixels)",
            "residual_policy": "nearest_accepted_label_only",
            "whole_object_fallback": "forbidden",
        },
        "reproducibility": {"seed": 42},
        "artifacts": {
            "annotation_schema": SCHEMA_ANNOTATION,
            "result_schema": SCHEMA_RESULT,
            "save_raw_and_cleaned_masks": True,
            "save_prompt_arrays": True,
            "save_feedback_raw_response": True,
        },
    }
    config["config_sha256"] = canonical_sha256(config)
    return config


def extract_v4_prompt_hashes(v4_path: Path) -> dict[str, str]:
    """Read prompt constants without importing models or allocating a GPU."""
    tree = ast.parse(v4_path.read_text(encoding="utf-8"), filename=str(v4_path))
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"PROMPT_V3", "PROMPT_FEEDBACK"}:
                values[target.id] = node.value.value
    missing = {"PROMPT_V3", "PROMPT_FEEDBACK"} - values.keys()
    if missing:
        raise ValueError(f"Could not extract prompt constants from {v4_path}: missing {sorted(missing)}")
    return {
        "prompt_v3_sha256": hashlib.sha256(values["PROMPT_V3"].encode("utf-8")).hexdigest(),
        "prompt_feedback_sha256": hashlib.sha256(values["PROMPT_FEEDBACK"].encode("utf-8")).hexdigest(),
    }


def component_at_point(mask: np.ndarray, point_xy: list[float] | np.ndarray | None) -> np.ndarray:
    """Keep the 8-connected component supported by a positive prompt.

    If the point lies just outside a SAM2 mask, use the closest mask component;
    this makes the decision explicit rather than accepting arbitrary islands.
    Substantial satellite geometry is retained, while tiny detached speckles are
    removed.
    """
    region = np.asarray(mask, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), 8)
    if count <= 1:
        return region

    chosen = 0
    if point_xy is not None:
        x = int(np.clip(round(float(point_xy[0])), 0, region.shape[1] - 1))
        y = int(np.clip(round(float(point_xy[1])), 0, region.shape[0] - 1))
        chosen = int(labels[y, x])
        if chosen == 0:
            ys, xs = np.nonzero(region)
            if len(xs):
                nearest = int(np.argmin((xs - x) ** 2 + (ys - y) ** 2))
                chosen = int(labels[ys[nearest], xs[nearest]])

    if chosen == 0:
        chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    primary_area = int(stats[chosen, cv2.CC_STAT_AREA])
    keep = labels == chosen
    for label_id in range(1, count):
        if label_id == chosen:
            continue
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= max(96, int(primary_area * 0.18)):
            keep |= labels == label_id
    return keep


def fill_small_holes(mask: np.ndarray, max_hole: int) -> np.ndarray:
    inverse = (~np.asarray(mask, dtype=bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    result = np.asarray(mask, dtype=bool).copy()
    height, width = result.shape
    for label_id in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[label_id])
        touches_border = x == 0 or y == 0 or x + box_width >= width or y + box_height >= height
        if not touches_border and area <= max_hole:
            result[labels == label_id] = True
    return result


def clean_candidate(mask: np.ndarray, foreground: np.ndarray, point_xy: list[float] | np.ndarray | None) -> np.ndarray:
    """Clip a SAM2 candidate to foreground and retain its prompt-supported part."""
    region = np.asarray(mask, dtype=bool) & np.asarray(foreground, dtype=bool)
    if not region.any():
        return region
    region = component_at_point(region, point_xy)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    region &= foreground
    region = component_at_point(region, point_xy)
    region = fill_small_holes(region, max_hole=max(48, int(np.asarray(foreground, dtype=bool).sum() * 0.00015)))
    return region & foreground


def nearest_fill(labels: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Assign only still-unassigned foreground to an already accepted label."""
    missing = np.asarray(foreground, dtype=bool) & (labels == 0)
    if not missing.any() or not np.any(labels > 0):
        return labels
    indices = ndimage.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    nearest = labels[tuple(indices)]
    result = labels.copy()
    result[missing] = nearest[missing]
    return result


def _bbox_area(candidate: dict[str, Any], fallback_area: int) -> float:
    bbox = candidate.get("bbox_xyxy", candidate.get("bbox"))
    if bbox is None or len(bbox) != 4:
        return float(fallback_area)
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return max(1.0, (x2 - x1) * (y2 - y1))


def fuse_candidates_point_aware(
    candidates: list[dict[str, Any]],
    foreground: np.ndarray,
    *,
    min_mask_pixels: int,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fuse cleaned candidates so prompt-supported small parts win overlaps.

    The returned ``accepted`` objects are the source candidate dictionaries
    enriched with audit fields.  No whole-object fallback is allowed: if no
    candidate survives, callers must report an explicit failure.
    """
    fg = np.asarray(foreground, dtype=bool)
    candidate_min = max(int(min_mask_pixels), 96, int(fg.sum() * 0.00012))
    overlap_min = max(int(min_mask_pixels), 96, int(fg.sum() * 0.00010))
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for source in candidates:
        part = source.get("part", source)
        initial = np.asarray(source["region"], dtype=bool)
        points = part.get("positive_points", source.get("positive_points", []))
        point = points[0] if points else None
        cleaned = clean_candidate(initial, fg, point)
        part["mask_area_initial"] = int(initial.sum())
        part["mask_area_cleaned"] = int(cleaned.sum())
        part["point_component_anchor_xy"] = point
        if int(cleaned.sum()) < candidate_min:
            rejected.append({
                "part_id": part.get("part_id"),
                "reason": "too_small_after_point_connected_cleanup",
                "mask_area_cleaned": int(cleaned.sum()),
                "threshold": candidate_min,
            })
            continue
        prepared.append({
            "part": part,
            "region": cleaned,
            "bbox_area": _bbox_area(part, int(cleaned.sum())),
            "cleaned_area": int(cleaned.sum()),
        })
        # Kept only until the caller saves an auditable PNG, then removed before
        # JSON serialization of the annotation.
        part["_cleaned_region"] = cleaned

    prepared.sort(key=lambda item: (item["bbox_area"], item["cleaned_area"], str(item["part"].get("part_id", ""))))
    labels = np.zeros(fg.shape, dtype=np.uint16)
    accepted: list[dict[str, Any]] = []
    for priority_index, candidate in enumerate(prepared, start=1):
        part = candidate["part"]
        available = candidate["region"] & (labels == 0)
        if int(available.sum()) < overlap_min:
            rejected.append({
                "part_id": part.get("part_id"),
                "reason": "too_small_after_small_first_overlap_arbitration",
                "available_pixels": int(available.sum()),
                "threshold": overlap_min,
            })
            continue
        label_id = len(accepted) + 1
        labels[available] = label_id
        part.update(
            label_id=label_id,
            mask_area_final=int(available.sum()),
            fusion_priority={
                "rank": priority_index,
                "bbox_area": candidate["bbox_area"],
                "cleaned_mask_area": candidate["cleaned_area"],
                "tie_break_part_id": part.get("part_id"),
            },
        )
        accepted.append(part)

    missing_before = int((fg & (labels == 0)).sum())
    labels = nearest_fill(labels, fg)
    missing_after = int((fg & (labels == 0)).sum())
    audit = {
        "strategy": FUSION_STRATEGY,
        "candidate_min_pixels": candidate_min,
        "overlap_min_pixels": overlap_min,
        "prepared_count": len(prepared),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "unassigned_foreground_pixels_before_fill": missing_before,
        "unassigned_foreground_pixels_after_fill": missing_after,
    }
    return labels, accepted, rejected, audit


def build_annotation_record(
    *,
    run_id: str,
    config_sha256: str,
    group: str,
    uid: str,
    image_size: list[int],
    parts: list[dict[str, Any]],
    invalid_parts: list[dict[str, Any]],
    fusion: dict[str, Any],
    final_map: dict[str, Any],
    status: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_ANNOTATION,
        "run_id": run_id,
        "config_sha256": config_sha256,
        "group": group,
        "uid": uid,
        "status": status,
        "image_size": image_size,
        "coordinate_system": "original pixel xyxy; source normalized 0-1000",
        "prompt_policy": PROMPT_POLICY,
        "parts": parts,
        "invalid_parts": invalid_parts,
        "fusion": fusion,
        "final_map": final_map,
        **extra,
    }


def build_result_event(annotation: dict[str, Any], *, timing_seconds: dict[str, float], error: str | None = None) -> dict[str, Any]:
    parts = annotation.get("parts", [])
    feedback = [part.get("feedback", {}).get("status") for part in parts if part.get("feedback")]
    feedback_status = "skipped" if not feedback else ("success" if all(value == "good" or value in {"corrected", "kept_initial"} for value in feedback) else "partial")
    status = annotation.get("status", "failed")
    qwen_status = annotation.get("qwen_plan", {}).get("parse_status", "not_run")
    return {
        "schema_version": SCHEMA_RESULT,
        "run_id": annotation["run_id"],
        "config_sha256": annotation["config_sha256"],
        "group": annotation["group"],
        "uid": annotation["uid"],
        "status": status,
        "stage_status": {
            "qwen_plan": "success" if qwen_status in {"ok", "retry_ok"} else qwen_status,
            "sam2": "success" if parts else "failed",
            "fusion": "success" if annotation.get("fusion", {}).get("accepted_count", 0) else "failed",
            "feedback": feedback_status,
            "segvigen": "not_run",
        },
        "counts": {
            "qwen_parts_proposed": annotation.get("qwen_plan", {}).get("proposed_parts", 0),
            "sam_masks_accepted": annotation.get("fusion", {}).get("prepared_count", 0),
            "final_parts": len(parts),
            "invalid_parts": len(annotation.get("invalid_parts", [])),
            "feedback_queries": len(feedback),
            "feedback_applied": sum(bool(part.get("feedback", {}).get("applied")) for part in parts),
        },
        "timing_seconds": timing_seconds,
        "output": annotation.get("final_map", {}),
        "artifact_hashes": {
            "rgba": annotation.get("input", {}).get("rgba_sha256"),
            "qwen_raw": annotation.get("qwen_plan", {}).get("raw_response_sha256"),
            "guidance": annotation.get("final_map", {}).get("sha256"),
            "labels_uint16": annotation.get("final_map", {}).get("labels_sha256"),
        },
        "error": error,
    }


def load_v4_helpers(script_dir: Path):
    """Load V4 only for an actual model run; CPU tests never import it."""
    path = script_dir / "run_qwen_sam2_guidance_v4.py"
    spec = importlib.util.spec_from_file_location("c_v4_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import C_v4 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model", default="/data2/codefile/fuyulin/1trellis2/qwen3vl/Qwen3-VL-4B-Instruct")
    parser.add_argument("--sam-repo", default="/data2/codefile/fuyulin/fyl/segvigen/segment-anything-2")
    parser.add_argument("--sam-checkpoint", default="/data2/codefile/fuyulin/fyl/segvigen/data_toolkit/assets/full_seg_w_2d_map/sam2.1_hiera_base_plus.pt")
    parser.add_argument("--sam-config", default="configs/sam2.1_hiera_b+.yaml")
    parser.add_argument("--palette", default="/data2/codefile/fuyulin/1trellis2/segvigen_inputs/colors_pure.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--uid", action="append")
    parser.add_argument("--group-filter", default="")
    parser.add_argument("--smoke-per-group", type=int, default=0)
    parser.add_argument("--output-suffix", default="C_v5_feedback")
    parser.add_argument("--run-id", default="C_v5_feedback_r01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--min-mask-pixels", type=int, default=10)
    parser.add_argument("--allow-overwrite", action="store_true", help="Explicitly allow replacement of an existing C_v5 artifact")
    parser.add_argument("--dry-run", action="store_true", help="Freeze config and validate selection without loading GPU models")
    return parser.parse_args()


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    per_group: dict[str, int] = {}
    for row in rows:
        group = row["group"]
        if args.group_filter and group != args.group_filter:
            continue
        if args.uid and row["uid"] not in args.uid:
            continue
        if args.smoke_per_group and per_group.get(group, 0) >= args.smoke_per_group:
            continue
        per_group[group] = per_group.get(group, 0) + 1
        selected.append(row)
    return selected


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8)).save(path)


def _feedback_trace(
    *,
    raw_path: Path,
    overlay_path: Path,
    result: dict[str, Any],
    applied: bool,
    initial_mask_path: Path,
    corrected_mask_path: Path | None,
) -> dict[str, Any]:
    return {
        "round": 1,
        "overlay_path": str(overlay_path),
        "overlay_sha256": sha256_file(overlay_path),
        "raw_response_path": str(raw_path),
        "raw_response_sha256": sha256_file(raw_path),
        "parse_status": result.get("status", "parse_error"),
        "decision": result.get("status", "parse_error"),
        "applied": applied,
        "pre_mask_path": str(initial_mask_path),
        "post_mask_path": str(corrected_mask_path) if corrected_mask_path else str(initial_mask_path),
    }


def main() -> None:
    args = parse_args()
    root = args.experiment_root
    manifest = args.manifest or root / "manifest_frozen.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    selected = select_rows(rows, args)
    if not selected:
        raise ValueError("No samples selected; check --uid/--group-filter/manifest")

    v4_path = Path(__file__).resolve().parent / "run_qwen_sam2_guidance_v4.py"
    config = make_frozen_config(
        run_id=args.run_id,
        manifest_path=str(manifest),
        manifest_sha256=sha256_file(manifest),
        explicit_uids=[row["uid"] for row in selected],
        output_suffix=args.output_suffix,
    )
    config["runtime"] = {
        "model": args.model,
        "sam_repo": args.sam_repo,
        "sam_checkpoint": args.sam_checkpoint,
        "sam_config": args.sam_config,
        "palette": args.palette,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "max_retries": args.max_retries,
        "min_mask_pixels": args.min_mask_pixels,
        "retry_generation_policy": [
            {"do_sample": False, "temperature": 1.0, "top_p": 1.0},
            {"do_sample": True, "temperature": 0.7, "top_p": 0.9},
            {"do_sample": True, "temperature": 1.0, "top_p": 0.95},
        ][: max(1, args.max_retries)],
    }
    config["input_preprocessing"] = {
        "alpha_foreground_threshold": 10,
        "coordinate_system": "norm_1000_to_original_pixel_xyxy",
        "resize_policy": "original_resolution",
    }
    config["provenance"] = {
        "driver_path": str(Path(__file__).resolve()),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "v4_helper_path": str(v4_path),
        "v4_helper_sha256": sha256_file(v4_path),
        **extract_v4_prompt_hashes(v4_path),
    }
    config["reproducibility"]["seed"] = args.seed
    config.pop("config_sha256")
    config["config_sha256"] = canonical_sha256(config)
    log_dir = root / "logs" / args.output_suffix
    config_path = log_dir / f"{args.run_id}_config.json"
    write_json(config_path, config)
    print(f"Frozen C_v5 config: {config_path}")
    print(f"Selected {len(selected)} samples; policy={PROMPT_POLICY}; fusion={FUSION_STRATEGY}")
    if args.dry_run:
        print("Dry run complete: no model loaded and no GPU used.")
        return

    # Imported late so all config/schema tests remain CPU-only.
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    v4 = load_v4_helpers(Path(__file__).resolve().parent)
    palette = np.asarray([item[:3] for item in json.loads(Path(args.palette).read_text())], dtype=np.uint8)
    if len(palette) < 2 or len(np.unique(palette[1:], axis=0)) != len(palette) - 1:
        raise ValueError("Foreground palette must contain unique nonzero colors")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading Qwen3-VL...")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).to(args.device).eval()
    print("Loading SAM2...")
    sys.path.insert(0, args.sam_repo)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(args.sam_config, args.sam_checkpoint, device=args.device))
    retry_configs = [(False, 1.0, 1.0), (True, 0.7, 0.9), (True, 1.0, 0.95)][: max(1, args.max_retries)]
    log_path = log_dir / f"{args.run_id}_results.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(selected, start=1):
        group, uid = row["group"], row["uid"]
        started = time.perf_counter()
        ann_path = root / "annotations" / args.output_suffix / group / f"{uid}.json"
        guide_path = root / "guidance" / args.output_suffix / group / f"{uid}.png"
        if not args.allow_overwrite and (ann_path.exists() or guide_path.exists()):
            event = {
                "schema_version": SCHEMA_RESULT,
                "run_id": args.run_id,
                "config_sha256": config["config_sha256"],
                "group": group,
                "uid": uid,
                "status": "skipped_existing",
                "stage_status": {"qwen_plan": "not_run", "sam2": "not_run", "fusion": "not_run", "feedback": "not_run", "segvigen": "not_run"},
                "counts": {},
                "timing_seconds": {"total": 0.0},
                "output": {"guidance_map": str(guide_path), "annotation": str(ann_path)},
                "artifact_hashes": {},
                "error": "existing_artifact_not_overwritten",
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(selected)}] skipped_existing {group}/{uid}", flush=True)
            continue
        mask_dir = root / "masks" / args.output_suffix / group / uid
        mask_dir.mkdir(parents=True, exist_ok=True)
        timing = {"qwen": 0.0, "sam2": 0.0, "feedback": 0.0, "fusion": 0.0, "total": 0.0}
        invalid: list[dict[str, Any]] = []

        try:
            rgba_path = Path(row["input_rgba"])
            rgba = np.asarray(Image.open(rgba_path).convert("RGBA"))
            rgb, foreground = rgba[:, :, :3], rgba[:, :, 3] > 10
            height, width = foreground.shape
            if not foreground.any():
                raise ValueError("Input RGBA has no alpha foreground")

            raw = ""
            plan_data: dict[str, Any] = {}
            parts: list[dict[str, Any]] = []
            plan_parse_status = "failed"
            raw_path = ann_path.with_suffix(".qwen.raw.txt")
            for attempt, (do_sample, temperature, top_p) in enumerate(retry_configs, start=1):
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": Image.fromarray(rgb)},
                    {"type": "text", "text": v4.PROMPT_V3},
                ]}]
                stamp = time.perf_counter()
                raw = v4.generate_qwen(model, processor, messages, args.max_new_tokens, do_sample, temperature, top_p, args.device)
                timing["qwen"] += time.perf_counter() - stamp
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(raw, encoding="utf-8")
                try:
                    plan_data, parts, plan_invalid = v4.parse_parts_v3(raw, width, height)
                    invalid.extend(plan_invalid)
                    if parts:
                        plan_parse_status = "ok" if attempt == 1 else "retry_ok"
                        break
                except Exception as exc:
                    invalid.append({"stage": "qwen_plan", "attempt": attempt, "reason": repr(exc)})
            if not parts:
                raise ValueError("Qwen produced no valid parts; C_v5 forbids whole-object fallback")

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(rgb)
                candidates: list[dict[str, Any]] = []
                for part in parts:
                    snapped_points = []
                    for point in part.get("positive_points", []):
                        snapped, found = v4.find_nearby_fg(point, foreground)
                        if found:
                            snapped_points.append(snapped)
                    if not snapped_points:
                        invalid.append({"part_id": part.get("part_id"), "stage": "sam2", "reason": "no_positive_point_on_foreground"})
                        continue
                    part["positive_points"] = snapped_points
                    stamp = time.perf_counter()
                    region, score, used = v4.run_sam2_for_part(predictor, rgb, foreground, part, "multi_point", args.min_mask_pixels)
                    timing["sam2"] += time.perf_counter() - stamp
                    if region is None:
                        invalid.append({"part_id": part.get("part_id"), "stage": "sam2", "reason": "empty_or_tiny_mask", "score": score})
                        continue
                    initial_path = mask_dir / f"{part['part_id']}_initial.png"
                    _save_mask(initial_path, region)
                    positive = used.get("positive_points", [])
                    negative = used.get("negative_points", [])
                    part["sam_trace"] = [{
                        "round": 0,
                        "origin": "qwen_plan",
                        "bbox_xyxy_px": used.get("bbox"),
                        "points_xy_px": positive + negative,
                        "point_labels": [1] * len(positive) + [0] * len(negative),
                        "multimask_output": False,
                        "selected_mask_index": 0,
                        "predicted_iou": score,
                        "mask_path": str(initial_path),
                        "mask_sha256": sha256_file(initial_path),
                    }]
                    part["raw_mask_path"] = str(initial_path)
                    part["raw_mask_sha256"] = sha256_file(initial_path)
                    candidates.append({"part": part, "region": region, "initial_path": initial_path})

                improved: list[dict[str, Any]] = []
                for candidate in candidates:
                    part, initial_mask, initial_path = candidate["part"], candidate["region"], candidate["initial_path"]
                    overlay = v4.create_mask_overlay(rgb, initial_mask, color=[255, 80, 80], alpha=0.35)
                    overlay_path = mask_dir / f"{part['part_id']}_feedback_overlay.png"
                    Image.fromarray(overlay).save(overlay_path)
                    feedback_prompt = v4.PROMPT_FEEDBACK.format(label_zh=part["label_zh"], label_en=part["label_en"])
                    messages = [{"role": "user", "content": [
                        {"type": "image", "image": Image.fromarray(rgb)},
                        {"type": "image", "image": Image.fromarray(overlay)},
                        {"type": "text", "text": feedback_prompt},
                    ]}]
                    stamp = time.perf_counter()
                    feedback_raw = v4.generate_qwen(model, processor, messages, 1024, False, 1.0, 1.0, args.device)
                    timing["feedback"] += time.perf_counter() - stamp
                    feedback_raw_path = ann_path.with_suffix(f".feedback_{part['part_id']}.raw.txt")
                    feedback_raw_path.write_text(feedback_raw, encoding="utf-8")
                    feedback = v4.parse_feedback(feedback_raw, width, height)
                    corrected_mask_path: Path | None = None
                    applied = False
                    selected_mask = initial_mask
                    if feedback.get("status") not in {"good", "parse_error"}:
                        corrected = dict(part)
                        if feedback.get("correction_box"):
                            corrected["bbox"] = feedback["correction_box"]
                        corrected["positive_points"] = list(part.get("positive_points", [])) + list(feedback.get("correction_positive_points") or [])
                        corrected["negative_points"] = list(part.get("negative_points", [])) + list(feedback.get("correction_negative_points") or [])
                        stamp = time.perf_counter()
                        corrected_mask, corrected_score, corrected_used = v4.run_sam2_for_part(
                            predictor, rgb, foreground, corrected, "multi_point", args.min_mask_pixels
                        )
                        timing["sam2"] += time.perf_counter() - stamp
                        if corrected_mask is not None:
                            selected_mask = corrected_mask
                            corrected_mask_path = mask_dir / f"{part['part_id']}_after_feedback.png"
                            _save_mask(corrected_mask_path, corrected_mask)
                            positive = corrected_used.get("positive_points", [])
                            negative = corrected_used.get("negative_points", [])
                            part["sam_trace"].append({
                                "round": 1,
                                "origin": "feedback",
                                "bbox_xyxy_px": corrected_used.get("bbox"),
                                "points_xy_px": positive + negative,
                                "point_labels": [1] * len(positive) + [0] * len(negative),
                                "multimask_output": False,
                                "selected_mask_index": 0,
                                "predicted_iou": corrected_score,
                                "mask_path": str(corrected_mask_path),
                                "mask_sha256": sha256_file(corrected_mask_path),
                            })
                            applied = True
                            feedback["status"] = "corrected"
                        else:
                            feedback["status"] = "correction_failed_kept_initial"
                    part["feedback"] = {**feedback, "applied": applied}
                    part["feedback_trace"] = _feedback_trace(
                        raw_path=feedback_raw_path,
                        overlay_path=overlay_path,
                        result=feedback,
                        applied=applied,
                        initial_mask_path=initial_path,
                        corrected_mask_path=corrected_mask_path,
                    )
                    improved.append({"part": part, "region": selected_mask})

            stamp = time.perf_counter()
            labels, accepted, rejected, fusion = fuse_candidates_point_aware(improved, foreground, min_mask_pixels=args.min_mask_pixels)
            timing["fusion"] = time.perf_counter() - stamp
            invalid.extend(rejected)
            if not accepted:
                raise ValueError("No C_v5 candidate survived point-aware fusion; no whole-object fallback written")

            for part in accepted:
                part["palette_index"] = (part["label_id"] - 1) % (len(palette) - 1) + 1
                part["color"] = palette[part["palette_index"]].tolist()
                cleaned_mask_path = mask_dir / f"{part['part_id']}_cleaned.png"
                _save_mask(cleaned_mask_path, part.pop("_cleaned_region"))
                part["cleaned_mask_path"] = str(cleaned_mask_path)
                part["cleaned_mask_sha256"] = sha256_file(cleaned_mask_path)
                final_mask_path = mask_dir / f"{part['part_id']}.png"
                _save_mask(final_mask_path, labels == part["label_id"])
                part["final_mask_path"] = str(final_mask_path)
                part["final_mask_sha256"] = sha256_file(final_mask_path)

            label_path = mask_dir / "labels_uint16.png"
            Image.fromarray(labels).save(label_path)
            color_ids = np.where(labels > 0, (labels.astype(np.int32) - 1) % (len(palette) - 1) + 1, 0)
            guide = palette[color_ids]
            guide[~foreground] = 0
            guide_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(guide).save(guide_path)
            torch.cuda.synchronize()
            timing["total"] = time.perf_counter() - started
            had_feedback_issue = any(part.get("feedback", {}).get("status") in {"parse_error", "correction_failed_kept_initial"} for part in accepted)
            annotation = build_annotation_record(
                run_id=args.run_id,
                config_sha256=config["config_sha256"],
                group=group,
                uid=uid,
                image_size=[width, height],
                parts=accepted,
                invalid_parts=invalid,
                fusion=fusion,
                final_map={
                    "path": str(guide_path),
                    "sha256": sha256_file(guide_path),
                    "labels_path": str(label_path),
                    "labels_sha256": sha256_file(label_path),
                    "painted_regions": len(accepted),
                    "foreground_pixels": int(foreground.sum()),
                },
                status="partial" if had_feedback_issue else "success",
                input={"rgba_path": str(rgba_path), "rgba_sha256": sha256_file(rgba_path)},
                qwen_plan={
                    "raw_response_path": str(raw_path),
                    "raw_response_sha256": sha256_file(raw_path),
                    "parse_status": plan_parse_status,
                    "proposed_parts": len(plan_data.get("parts", [])),
                    "valid_parts": len(parts),
                },
            )
            write_json(ann_path, annotation)
            event = build_result_event(annotation, timing_seconds=timing)
        except Exception as exc:
            timing["total"] = time.perf_counter() - started
            annotation = build_annotation_record(
                run_id=args.run_id,
                config_sha256=config["config_sha256"],
                group=group,
                uid=uid,
                image_size=[],
                parts=[],
                invalid_parts=invalid,
                fusion={"strategy": FUSION_STRATEGY, "accepted_count": 0, "rejected_count": len(invalid)},
                final_map={},
                status="failed",
                error=repr(exc),
            )
            write_json(ann_path, annotation)
            event = build_result_event(annotation, timing_seconds=timing, error=repr(exc))
            traceback.print_exc()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(f"[{index}/{len(selected)}] {event['status']} {group}/{uid} parts={event['counts']['final_parts']} {timing['total']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

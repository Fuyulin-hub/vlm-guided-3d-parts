"""Label-ID metrics shared by the three paper-table evaluators.

Scores are fractions in [0, 1]. Label zero is background by default.
This is a project evaluator, not a released SegViGen evaluation script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


METRIC_KEYS = ("matched_miou", "gt_best_iou", "boundary_f1", "part_f1")


def _integer_labels(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("A label map must have two-dimensional shape (height, width)")
    if not np.issubdtype(array.dtype, np.integer) and array.dtype != bool:
        raise ValueError("Label IDs must use an integer dtype, not continuous values")
    return array.astype(np.int64, copy=False)


def read_label_map(path: str | Path, palette: dict | None = None) -> np.ndarray:
    """Read integer PNG/NPY labels; RGB requires an exact color-to-ID mapping.

    Disconnected pixels carrying the same ID remain one part. No resizing,
    color quantization, component splitting, or label repair is performed.
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)
    if array.ndim == 2:
        return _integer_labels(array)
    if array.ndim != 3 or array.shape[2] not in (3, 4) or not palette:
        raise ValueError(f"{path}: RGB/RGBA labels require an explicit exact palette")
    rgb = array[:, :, :3]
    labels = np.full(rgb.shape[:2], -1, dtype=np.int64)
    used_colors = set()
    for label, color in palette.items():
        color = tuple(color)
        if len(color) != 3 or color in used_colors or int(label) < 0:
            raise ValueError("Each palette entry needs a unique RGB color and nonnegative ID")
        used_colors.add(color)
        labels[np.all(rgb == color, axis=-1)] = int(label)
    if np.any(labels < 0):
        raise ValueError(f"{path}: {int((labels < 0).sum())} pixels do not match the exact palette")
    return labels


def threshold_match_count(iou: np.ndarray, threshold: float) -> int:
    """Maximize the number of threshold-passing one-to-one part matches."""
    if iou.size == 0:
        return 0
    if not 0 < threshold <= 1:
        raise ValueError("Part IoU threshold must lie in (0, 1]")
    cardinality_weight = min(iou.shape) + 1
    weights = (iou >= threshold) * cardinality_weight + iou
    pred, ref = linear_sum_assignment(-weights)
    return int(np.count_nonzero(iou[pred, ref] >= threshold))


def _boundary_f1(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray, tolerance: float) -> float:
    structure = np.ones((3, 3), dtype=bool)
    supported = ndimage.binary_erosion(valid, structure=structure, border_value=1)
    pred_boundary = pred & ~ndimage.binary_erosion(pred, structure=structure, border_value=0) & supported
    ref_boundary = ref & ~ndimage.binary_erosion(ref, structure=structure, border_value=0) & supported
    n_pred, n_ref = int(pred_boundary.sum()), int(ref_boundary.sum())
    if not n_pred or not n_ref:
        return 0.0
    precision = float((ndimage.distance_transform_edt(~ref_boundary)[pred_boundary] <= tolerance).mean())
    recall = float((ndimage.distance_transform_edt(~pred_boundary)[ref_boundary] <= tolerance).mean())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def label_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    background: int = 0,
    ignore_labels: Iterable[int] = (-1, 65535),
    boundary_tolerance: float = 2.0,
    part_threshold: float = 0.5,
) -> dict:
    """Evaluate anonymous parts, including zero contributions for missed GT.

    matched_miou: sum of one-to-one IoU matches / number of reference parts.
    gt_best_iou: mean, over GT parts, of their best prediction overlap. Unlike
    the former, this allows the same prediction to match multiple GT parts.
    Boundary F1 uses the IoU-maximizing matching and is GT-part averaged.
    Extra predictions are counted by Part F1, not its IoU normalization.
    """
    pred, ref = _integer_labels(pred), _integer_labels(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"Prediction/reference shape mismatch: {pred.shape} versus {ref.shape}")
    if boundary_tolerance < 0 or not np.isfinite(boundary_tolerance):
        raise ValueError("Boundary tolerance must be finite and nonnegative")
    if not 0 < part_threshold <= 1:
        raise ValueError("Part IoU threshold must lie in (0, 1]")
    ignored = tuple(ignore_labels)
    valid = ~np.isin(ref, ignored)
    pred = np.where(np.isin(pred, ignored), background, pred)
    if np.any(pred[valid] < 0) or np.any(ref[valid] < 0):
        raise ValueError("Unignored part IDs must be nonnegative")
    pred_values, pred_inverse = np.unique(pred[valid], return_inverse=True)
    ref_values, ref_inverse = np.unique(ref[valid], return_inverse=True)
    pred_keep = pred_values != background
    ref_keep = ref_values != background
    pred_ids, ref_ids = pred_values[pred_keep], ref_values[ref_keep]
    n_pred, n_ref = len(pred_ids), len(ref_ids)
    if not n_ref:
        raise ValueError("The reference contains no annotated foreground parts")
    if n_pred:
        counts = np.bincount(
            pred_inverse * len(ref_values) + ref_inverse,
            minlength=len(pred_values) * len(ref_values),
        ).reshape(len(pred_values), len(ref_values))
        intersection = counts[np.ix_(pred_keep, ref_keep)].astype(float)
        union = counts.sum(axis=1)[pred_keep, None] + counts.sum(axis=0)[None, ref_keep] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        pred_index, ref_index = linear_sum_assignment(-iou)
        matched_miou = float(iou[pred_index, ref_index].sum() / n_ref)
        gt_best_iou = float(iou.max(axis=0).mean())
        boundary_sum = sum(
            _boundary_f1(pred == pred_ids[p], ref == ref_ids[r], valid, boundary_tolerance)
            for p, r in zip(pred_index, ref_index)
            if iou[p, r] > 0
        )
        tp = threshold_match_count(iou, part_threshold)
    else:
        matched_miou = gt_best_iou = boundary_sum = 0.0
        tp = 0
    return {
        "matched_miou": matched_miou,
        "gt_best_iou": gt_best_iou,
        "boundary_f1": float(boundary_sum / n_ref),
        "part_f1": float(2 * tp / (n_pred + n_ref)),
        "predicted_parts": n_pred,
        "reference_parts": n_ref,
        "true_positive_parts": tp,
        "false_positive_parts": n_pred - tp,
        "false_negative_parts": n_ref - tp,
        "evaluated_pixels": int(valid.sum()),
    }


def summarize(records: list[dict], groups: list[str], keys: Iterable[str]) -> dict:
    """Report group macro and object means, preserving missing measurements."""
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("Evaluation groups must be nonempty and unique")
    if any(record["group"] not in groups for record in records):
        raise ValueError("A sample has a group absent from the configured evaluation groups")
    keys = tuple(keys)

    def means(rows):
        result = {"n_expected": len(rows)}
        for key in keys:
            values = [row.get(key) for row in rows]
            measured = [float(value) for value in values if value is not None]
            if any(not np.isfinite(value) for value in measured):
                raise ValueError(f"Non-finite measurement in {key}")
            result[key] = float(np.mean(measured)) if rows and len(measured) == len(rows) else None
            result[f"n_{key}"] = len(measured)
        return result

    by_group = {group: means([row for row in records if row["group"] == group]) for group in groups}
    macro = {}
    for key in keys:
        values = [by_group[group][key] for group in groups]
        macro[key] = float(np.mean(values)) if all(value is not None for value in values) else None
    return {"by_group": by_group, "macro": macro, "object_mean": means(records)}

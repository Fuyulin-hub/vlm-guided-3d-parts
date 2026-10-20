"""Reproducible 2D guide-mask agreement utilities.

The full command-line evaluator is added incrementally alongside its tests.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import ndimage


def quantize_exact_palette(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Convert a discrete RGB palette image to labels without lossy repair.

    Black is background. Any other color must match a supplied palette entry
    exactly; continuous/shaded renders are rejected so they cannot silently
    become a fabricated part-ID map.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    if palette.ndim != 2 or palette.shape[-1] != 3:
        raise ValueError("palette must have shape (n_colors, 3)")

    foreground = np.any(rgb != 0, axis=-1)
    matches = np.all(rgb[:, :, None, :] == palette[None, None, :, :], axis=-1)
    valid = np.any(matches, axis=-1)
    invalid_count = int((foreground & ~valid).sum())
    if invalid_count:
        raise ValueError(
            f"{invalid_count} non-palette foreground pixel(s); "
            "use validated discrete part-ID maps rather than a shaded render"
        )

    labels = matches.argmax(axis=-1).astype(np.int32) + 1
    labels[~foreground] = 0
    return labels


def label_components(palette_labels: np.ndarray, min_area: int) -> np.ndarray:
    """Give each retained connected palette region a spatially unique part ID."""
    if palette_labels.ndim != 2:
        raise ValueError("palette_labels must be a 2D array")

    output = np.zeros_like(palette_labels, dtype=np.int32)
    next_id = 1
    structure = np.ones((3, 3), dtype=np.uint8)
    for color_id in np.unique(palette_labels):
        if color_id == 0:
            continue
        connected, count = ndimage.label(palette_labels == color_id, structure=structure)
        if count == 0:
            continue
        areas = np.bincount(connected.ravel())
        for component_id in range(1, count + 1):
            if areas[component_id] < min_area:
                continue
            output[connected == component_id] = next_id
            next_id += 1
    return output


def evaluate_label_maps(pred_labels: np.ndarray, ref_labels: np.ndarray) -> dict[str, float | int]:
    """Hungarian-match non-background part labels and return agreement metrics."""
    if pred_labels.shape != ref_labels.shape:
        raise ValueError("prediction and reference label maps must have identical shapes")

    pred_ids = np.unique(pred_labels)
    pred_ids = pred_ids[pred_ids != 0]
    ref_ids = np.unique(ref_labels)
    ref_ids = ref_ids[ref_ids != 0]

    pred_masks = pred_labels[None, :, :] == pred_ids[:, None, None]
    ref_masks = ref_labels[None, :, :] == ref_ids[:, None, None]

    if len(pred_ids) == 0 or len(ref_ids) == 0:
        return {
            "mean_iou": 0.0,
            "mean_dice": 0.0,
            "matched_parts": 0,
            "total_ref_parts": int(len(ref_ids)),
            "total_pred_parts": int(len(pred_ids)),
            "missing_parts": int(len(ref_ids)),
            "extra_parts": int(len(pred_ids)),
            "missing_rate": 1.0 if len(ref_ids) else 0.0,
            "extra_rate": 1.0 if len(pred_ids) else 0.0,
        }

    intersection = (pred_masks[:, None] & ref_masks[None, :]).sum(axis=(2, 3))
    union = (pred_masks[:, None] | ref_masks[None, :]).sum(axis=(2, 3))
    iou = intersection / np.maximum(union, 1)
    pred_indices, ref_indices = linear_sum_assignment(-iou)
    matched_iou = iou[pred_indices, ref_indices]
    matched = int((matched_iou > 0).sum())

    dice = 2 * matched_iou / (1 + matched_iou)
    return {
        "mean_iou": float(matched_iou.mean()),
        "mean_dice": float(dice.mean()),
        "matched_parts": matched,
        "total_ref_parts": int(len(ref_ids)),
        "total_pred_parts": int(len(pred_ids)),
        "missing_parts": int(len(ref_ids) - matched),
        "extra_parts": int(len(pred_ids) - matched),
        "missing_rate": float((len(ref_ids) - matched) / len(ref_ids)),
        "extra_rate": float((len(pred_ids) - matched) / len(pred_ids)),
    }

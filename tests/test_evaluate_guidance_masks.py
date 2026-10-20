"""Tests for the reproducible 2D guide-mask agreement evaluator."""

from pathlib import Path
import importlib.util

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_guidance_masks.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evaluate_guidance_masks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_hungarian_matching_ignores_palette_order():
    """The score should match spatial parts even when color IDs are permuted."""
    evaluator = load_module()
    pred = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [2, 2, 0, 0],
            [2, 2, 0, 0],
        ],
        dtype=np.int16,
    )
    ref = np.array(
        [
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.int16,
    )

    result = evaluator.evaluate_label_maps(pred, ref)

    assert result["total_pred_parts"] == 2
    assert result["total_ref_parts"] == 2
    assert result["matched_parts"] == 2
    assert result["mean_iou"] == 1.0
    assert result["mean_dice"] == 1.0
    assert result["missing_rate"] == 0.0
    assert result["extra_rate"] == 0.0


def test_disconnected_regions_of_one_color_are_separate_parts():
    """A repeated palette color must not collapse spatially separate components."""
    evaluator = load_module()
    palette_labels = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1],
        ],
        dtype=np.int16,
    )

    components = evaluator.label_components(palette_labels, min_area=2)

    assert set(np.unique(components)) == {0, 1, 2}


def test_exact_palette_quantization_rejects_shaded_reference_pixels():
    """A continuous render must not silently become a fabricated part-ID map."""
    evaluator = load_module()
    palette = np.array([[229, 22, 22], [22, 229, 22]], dtype=np.uint8)
    exact = np.array([[[0, 0, 0], [229, 22, 22], [22, 229, 22]]], dtype=np.uint8)

    labels = evaluator.quantize_exact_palette(exact, palette)
    assert labels.tolist() == [[0, 1, 2]]

    shaded = exact.copy()
    shaded[0, 1] = [228, 23, 22]
    with pytest.raises(ValueError, match="non-palette"):
        evaluator.quantize_exact_palette(shaded, palette)

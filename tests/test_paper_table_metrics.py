"""Engineering checks for the paper-table evaluators, not experimental results."""

from pathlib import Path
import importlib.util

import numpy as np
import pytest
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "paper_table_metrics.py"


def evaluator():
    assert SCRIPT.is_file(), "The shared paper-table metric implementation is missing"
    spec = importlib.util.spec_from_file_location("paper_table_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_label_permutation_does_not_change_partition_scores():
    m = evaluator()
    ref = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [2, 2, 0, 0], [2, 2, 0, 0]])
    pred = np.where(ref == 1, 27, np.where(ref == 2, 91, 0))
    score = m.label_metrics(pred, ref, boundary_tolerance=0)
    for key in ("matched_miou", "gt_best_iou", "boundary_f1", "part_f1"):
        assert score[key] == pytest.approx(1)


def test_missing_reference_part_is_zero_in_gt_normalized_miou():
    m = evaluator()
    ref = np.array([[1, 1, 0, 2, 2], [1, 1, 0, 2, 2]])
    pred = np.where(ref == 1, 9, 0)
    score = m.label_metrics(pred, ref)
    assert score["matched_miou"] == pytest.approx(0.5)
    assert score["part_f1"] == pytest.approx(2 / 3)
    assert score["false_negative_parts"] == 1


def test_oversegmentation_is_counted_as_extra_parts():
    m = evaluator()
    ref = np.ones((3, 4), dtype=np.int16)
    pred = np.tile([7, 7, 8, 8], (3, 1))
    score = m.label_metrics(pred, ref)
    assert score["matched_miou"] == pytest.approx(0.5)
    assert score["part_f1"] == pytest.approx(2 / 3)
    assert score["false_positive_parts"] == 1


def test_best_overlap_protocol_is_distinct_from_one_to_one_matching():
    m = evaluator()
    ref = np.array([[1, 1, 2, 2], [1, 1, 2, 2]])
    pred = np.ones_like(ref)
    score = m.label_metrics(pred, ref)
    assert score["matched_miou"] == pytest.approx(0.25)
    assert score["gt_best_iou"] == pytest.approx(0.5)


def test_part_f1_matching_maximizes_threshold_passing_pairs():
    m = evaluator()
    iou = np.array([[0.99, 0.5], [0.5, 0.49]])
    assert m.threshold_match_count(iou, 0.5) == 2


def test_unlabeled_prediction_is_a_valid_zero_score():
    m = evaluator()
    ref = np.array([[1, 1], [2, 2]])
    score = m.label_metrics(np.zeros_like(ref), ref)
    assert score["matched_miou"] == 0
    assert score["boundary_f1"] == 0
    assert score["part_f1"] == 0


def test_ignored_reference_pixels_do_not_change_region_scores():
    m = evaluator()
    ref = np.array([[1, 1, 65535], [2, 2, 65535]])
    pred = np.array([[7, 7, 1234], [8, 8, 4321]])
    score = m.label_metrics(pred, ref)
    assert score["matched_miou"] == 1
    assert score["predicted_parts"] == 2


def test_rgb_requires_an_explicit_exact_palette(tmp_path):
    m = evaluator()
    rgb = np.array([[[0, 0, 0], [255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    path = tmp_path / "labels.png"
    Image.fromarray(rgb).save(path)
    with pytest.raises(ValueError, match="palette"):
        m.read_label_map(path)
    labels = m.read_label_map(path, palette={"0": [0, 0, 0], "7": [255, 0, 0], "9": [0, 255, 0]})
    assert labels.tolist() == [[0, 7, 9]]
    with pytest.raises(ValueError, match="palette"):
        m.read_label_map(path, palette={"0": [0, 0, 0], "7": [254, 0, 0], "9": [0, 255, 0]})


def test_integer_labels_preserve_disconnected_instances_with_shared_id(tmp_path):
    m = evaluator()
    labels = np.array([[3, 0, 3], [3, 0, 3]], dtype=np.uint16)
    path = tmp_path / "labels.png"
    Image.fromarray(labels).save(path)
    assert np.array_equal(m.read_label_map(path), labels)


def test_shapes_and_invalid_numeric_labels_are_rejected():
    m = evaluator()
    with pytest.raises(ValueError, match="shape"):
        m.label_metrics(np.zeros((2, 2), int), np.ones((3, 3), int))
    with pytest.raises(ValueError, match="integer"):
        m.label_metrics(np.array([[1.5, 2.0]]), np.array([[1, 2]]))


def test_group_macro_is_not_the_pooled_object_average():
    m = evaluator()
    rows = [{"group": "assemblies", "score": 1.0}] * 3 + [{"group": "solid_volume", "score": 0.0}]
    result = m.summarize(rows, ["assemblies", "solid_volume"], ["score"])
    assert result["macro"]["score"] == 0.5
    assert result["object_mean"]["score"] == 0.75


def test_missing_group_or_measurement_keeps_macro_unavailable():
    m = evaluator()
    result = m.summarize([{"group": "assemblies", "score": 1.0}], ["assemblies", "solid_volume"], ["score"])
    assert result["macro"]["score"] is None
    result = m.summarize([{"group": "assemblies", "score": 1.0}, {"group": "assemblies", "score": None}], ["assemblies"], ["score"])
    assert result["macro"]["score"] is None

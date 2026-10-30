"""Direct 2D evaluation must preserve IDs, missing coverage, and zero scores."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_c_d_guidance.py"


def evaluator():
    spec = importlib.util.spec_from_file_location("evaluate_c_d_guidance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_decode_preserves_disconnected_same_color_and_transparency():
    module = evaluator()
    rgba = np.array([[[229, 81, 22, 255], [12, 34, 56, 0],
                      [229, 81, 22, 255], [22, 140, 229, 255]]], np.uint8)
    labels = module.decode_colors(rgba, [(229, 81, 22), (22, 140, 229)])
    assert labels.tolist() == [[1, 0, 1, 2]]
    rgba[0, 0, 0] = 228
    with pytest.raises(ValueError, match="off-palette"):
        module.decode_colors(rgba, [(229, 81, 22), (22, 140, 229)])


@pytest.mark.parametrize("group", ["assemblies", "solid_volume"])
def test_missing_files_are_reported_and_empty_predictions_receive_zero(tmp_path, group):
    module = evaluator()
    predictions, references = tmp_path / "C", tmp_path / "D"
    predictions.mkdir()
    entries = []
    for uid in ("good", "empty", "missing"):
        label_path = references / uid / "labels_uint16.png"
        label_path.parent.mkdir(parents=True)
        Image.fromarray(np.array([[0, 9, 9, 0], [0, 9, 9, 0]], np.uint16)).save(label_path)
        entries.append({"uid": uid, "labels_file": f"{uid}/labels_uint16.png",
                        "labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest()})
    rgb = np.zeros((2, 4, 3), np.uint8)
    rgb[:, 1:3] = (229, 81, 22)
    Image.fromarray(rgb).save(predictions / "good.png")
    Image.fromarray(np.zeros_like(rgb)).save(predictions / "empty.png")
    manifest, palette = tmp_path / "manifest.json", tmp_path / "palette.json"
    manifest.write_text(json.dumps({"group": group, "per_uid": entries}))
    palette.write_text(json.dumps([[229, 22, 22, 255], [229, 81, 22, 255]]))
    summary = module.evaluate(predictions, references, manifest, palette, tmp_path / "out", progress=False)
    assert summary["n_reference_expected"] == 3
    assert summary["n_evaluated"] == 2
    assert summary["missing_prediction_uids"] == ["missing"]
    assert summary["metrics"]["matched_miou"]["mean"] == pytest.approx(0.5)
    assert summary["metrics"]["part_f1"]["mean"] == pytest.approx(0.5)
    assert summary["complete_reference_cohort"] is False
    good_record = json.loads((tmp_path / "out/per_object/good/result.json").read_text())
    assert good_record["group"] == group
    assert (tmp_path / "out/decoded_C_labels" / group / "good/labels_uint16.png").is_file()
    Image.fromarray(rgb).save(predictions / "missing.png")
    full = module.evaluate(predictions, references, manifest, palette, tmp_path / "out", progress=False)
    assert full["complete_reference_cohort"] is True
    assert "已覆盖" in (tmp_path / "out/report_zh.md").read_text(encoding="utf-8")
    # A changed D reference cannot silently reuse a previous successful score.
    Image.fromarray(np.array([[0, 9, 0, 0], [0, 9, 0, 0]], np.uint16)).save(references / "good/labels_uint16.png")
    revised = module.evaluate(predictions, references, manifest, palette, tmp_path / "out", progress=False)
    assert revised["n_evaluated"] == 2
    assert revised["n_errors"] == 1
    assert "Reference hash mismatch" in revised["errors"][0]["error"]
    assert json.loads((tmp_path / "out/per_object/good/result.json").read_text())["status"] == "evaluation_error"


def test_grid_mismatch_is_rejected_without_resizing():
    module = evaluator()
    metric = module.load_metrics()
    with pytest.raises(ValueError, match="shape mismatch"):
        metric.label_metrics(np.ones((2, 2), int), np.ones((3, 3), int))


def test_b_input_uses_explicit_rotation_and_supplied_metric_core(tmp_path):
    module = evaluator()
    predictions, references = tmp_path / "B", tmp_path / "D"
    predictions.mkdir()
    references.mkdir()
    ref = np.array([[0, 0, 0], [9, 9, 0], [0, 0, 0]], np.uint16)
    Image.fromarray(ref).save(references / "labels.png")
    rgba = np.zeros((3, 3, 4), np.uint8)
    rgba[ref > 0] = [229, 81, 22, 255]
    Image.fromarray(np.rot90(rgba, k=-1)).save(predictions / "sample.png")
    manifest, palette = tmp_path / "manifest.json", tmp_path / "palette.json"
    manifest.write_text(json.dumps({"group": "thin_fine_grained", "per_uid": [{
        "uid": "sample", "labels_file": "labels.png",
        "labels_sha256": hashlib.sha256((references / "labels.png").read_bytes()).hexdigest()}]}))
    palette.write_text(json.dumps([[229, 22, 22, 255], [229, 81, 22, 255]]))
    core = SCRIPT.with_name("paper_table_metrics.py")
    result = module.evaluate(predictions, references, manifest, palette, tmp_path / "out",
                            progress=False, method="B_sam2", rotation_k=1, metrics_path=core)
    assert result["metrics"]["matched_miou"]["mean"] == 1
    assert result["method"] == "B_sam2"
    row = json.loads((tmp_path / "out/per_object/sample/result.json").read_text())
    assert row["rotation_ccw"] == 90
    assert row["method"] == "B_sam2"
    assert (tmp_path / "out/decoded_labels/thin_fine_grained/sample/labels_uint16.png").is_file()
    assert result["input_paths"]["metrics_core"] == str(core.resolve())

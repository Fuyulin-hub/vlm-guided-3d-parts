"""Real-schema adapter checks on synthetic data, not paper measurements."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("objaverse_table_runner", SCRIPTS / "paper_table_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def sources(tmp_path):
    sample = {"group": "assemblies", "uid": "test_uid"}
    rgb = np.array([[[0, 0, 0], [229, 81, 22], [0, 200, 80]]], dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "test_uid.png")
    annotation = {"parts": [{"label_id": 7, "color": [229, 81, 22]},
                             {"label_id": 2, "color": [0, 200, 80]}],
                  "invalid_parts": [{"label_id": 9, "color": [229, 81, 22]}]}
    (tmp_path / "test_uid.json").write_text(json.dumps(annotation), encoding="utf-8")
    return tmp_path, sample, annotation


def test_annotation_parts_decode_exact_rgb_with_sample_paths(sources):
    root, sample, _ = sources
    labels = runner.source_labels(root, {"labels": "{uid}.png", "palette_annotation": "{uid}.json"}, sample)
    assert labels.tolist() == [[0, 7, 2]]


def test_reused_annotation_color_is_not_guessed_into_distinct_parts(sources):
    root, sample, annotation = sources
    annotation["parts"][1]["color"] = [229, 81, 22]
    (root / "test_uid.json").write_text(json.dumps(annotation), encoding="utf-8")
    with pytest.raises(ValueError, match="^Annotation palette collision"):
        runner.source_labels(root, {"labels": "{uid}.png", "palette_annotation": "{uid}.json"}, sample)


def test_b_palette_indices_are_explicitly_not_named_part_ids(sources):
    root, sample, _ = sources
    (root / "colors.json").write_text(json.dumps([[0, 0, 0], [229, 81, 22], [0, 200, 80]]), encoding="utf-8")
    source = {"labels": "{uid}.png", "palette": "colors.json", "palette_format": "index_array",
              "label_semantics": "palette_index"}
    np.save(root / "reference.npy", np.array([[0, 1, 2]], dtype=np.uint16))
    config = {"reference": {"guidance": {"labels": "reference.npy"}}}
    records, _ = runner.evaluate_maps(config, root, [sample], ["assemblies"],
                                    {"key": "B", "guidance": source}, "guidance", "unavailable")
    assert records[0]["matched_miou"] == 1.0
    assert records[0]["label_semantics"] == "palette_index"


def test_array_palette_requires_explicit_palette_index_semantics(sources):
    root, sample, _ = sources
    (root / "colors.json").write_text(json.dumps([[0, 0, 0], [229, 81, 22], [0, 200, 80]]), encoding="utf-8")
    with pytest.raises(ValueError, match="label_semantics"):
        runner.source_labels(root, {"labels": "{uid}.png", "palette": "colors.json",
                                    "palette_format": "index_array"}, sample)


def test_abc_filters_exclude_shared_reconstruction_other_arms_and_control_events(tmp_path):
    sample = {"group": "assemblies", "uid": "one"}
    common = {**sample, "status": "success", "stage": "segvigen", "arm": "B_sam2",
              "seconds": 12.0, "peak_allocated_bytes": 2_000_000_000}
    events = [{**common, "stage": "trellis", "arm": "shared", "seconds": 611.0},
              {**common, "arm": "A_raw"}, {**common, "status": "waiting", "seconds": 0.0001},
              {**common, "status": "cached", "seconds": 0.0}, common]
    (tmp_path / "abc_events.jsonl").write_text("\n".join(json.dumps(row) for row in events), encoding="utf-8")
    method = {"key": "B", "run_components": [{"path": "abc_events.jsonl", "phase": "segmentation",
              "filters": {"arm": "B_sam2", "stage": "segvigen"},
              "time_key": "seconds", "memory_key": "peak_allocated_bytes", "memory_unit": "bytes"}]}
    rows, summary = runner.run_measurements({}, tmp_path, [sample], method)
    assert summary["total_seconds"] == 12.0
    assert summary["peak_gpu_memory_gb"] == 2.0
    assert rows[0]["status"] == "success"


def test_d_waiting_with_stale_peak_memory_does_not_count_as_measured(tmp_path):
    sample = {"group": "assemblies", "uid": "one"}
    event = {**sample, "arm": "D_human_fixed", "stage": "segvigen", "status": "waiting",
             "seconds": 0.00012, "peak_allocated_bytes": 10_572_347_392, "reason": "guidance missing"}
    (tmp_path / "d_events.jsonl").write_text(json.dumps(event), encoding="utf-8")
    method = {"key": "D", "run_components": [{"path": "d_events.jsonl", "phase": "segmentation",
              "time_key": "seconds", "memory_key": "peak_allocated_bytes", "memory_unit": "bytes"}]}
    _, summary = runner.run_measurements({}, tmp_path, [sample], method)
    assert summary["total_seconds"] is None
    assert summary["peak_gpu_memory_gb"] is None
    assert summary["run_failure"] is None


def test_missing_d_annotation_is_retained_and_not_zeroed_as_prediction_failure(tmp_path):
    sample = {"group": "assemblies", "uid": "one"}
    config = {"reference": {"guidance": {"labels": "absent_D.npy"}}}
    method = {"key": "B", "guidance": {"labels": "absent_B.npy"}}
    rows, summary = runner.evaluate_maps(config, tmp_path, [sample], ["assemblies"], method, "guidance", "zero")
    assert rows[0]["status"] == "missing_reference"
    assert rows[0]["matched_miou"] is None
    assert summary["macro"]["matched_miou"] is None
    assert summary["by_group"]["assemblies"]["n_expected"] == 1


def test_frozen_group_counts_are_checked_instead_of_accepting_a_pilot(tmp_path):
    (tmp_path / "manifest.csv").write_text("group,uid\nassemblies,one\n", encoding="utf-8")
    config = {"root": str(tmp_path), "manifest": "manifest.csv", "methods": [],
              "expected_group_counts": {"assemblies": 220, "solid_volume": 232, "thin_fine_grained": 232}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="counts"):
        runner.read_config(tmp_path / "config.json")


def test_projection_camera_accepts_the_existing_c2w_and_k_schema(tmp_path):
    camera = {"c2w": np.eye(4).tolist(), "K": np.eye(3).tolist(), "image_size": [1024, 1024]}
    (tmp_path / "camera.json").write_text(json.dumps(camera), encoding="utf-8")
    loaded = runner.camera_parameters(tmp_path / "camera.json")
    assert loaded["c2w"] == camera["c2w"]
    assert loaded["K"] == camera["K"]
    assert loaded["image_size"] == [1024, 1024]


def test_segmentation_only_measurements_do_not_become_end_to_end_cost(tmp_path):
    sample = {"group": "assemblies", "uid": "one"}
    event = {**sample, "status": "success", "seconds": 12.0, "peak_allocated_bytes": 2_000_000_000}
    (tmp_path / "events.jsonl").write_text(json.dumps(event), encoding="utf-8")
    method = {"key": "B", "measurement_scope": "segmentation_only", "run_components": [
        {"path": "events.jsonl", "phase": "segmentation", "time_key": "seconds",
         "memory_key": "peak_allocated_bytes", "memory_unit": "bytes"}]}
    rows, summary = runner.run_measurements({}, tmp_path, [sample], method)
    assert summary["total_seconds"] is None
    assert summary["peak_gpu_memory_gb"] is None
    assert rows[0]["measured_component_seconds"] == 12.0

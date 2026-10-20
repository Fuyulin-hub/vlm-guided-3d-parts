"""End-to-end CLI checks on generated fixtures, not paper measurements."""

from pathlib import Path
import csv
import json
import subprocess
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "reference").mkdir()
    (root / "ours").mkdir()
    (root / "logs").mkdir()
    (root / "manifest.csv").write_text("group,uid\nassemblies,u1\nsolid_volume,u2\n", encoding="utf-8")
    camera = {"transform_matrix": np.eye(4).tolist(), "camera_angle_x": 0.7}
    labels = np.array([[0, 1, 1, 0], [0, 1, 1, 0], [2, 2, 0, 0], [2, 2, 0, 0]], dtype=np.uint16)
    for uid in ("u1", "u2"):
        np.save(root / "reference" / f"{uid}.npy", labels)
        np.save(root / "ours" / f"{uid}.npy", np.where(labels == 1, 17, np.where(labels == 2, 23, 0)))
        (root / "reference" / f"{uid}.json").write_text(json.dumps(camera), encoding="utf-8")
        (root / "ours" / f"{uid}.json").write_text(json.dumps(camera), encoding="utf-8")
    ref = {"labels": "reference/{uid}.npy", "camera": "reference/{uid}.json"}
    pred = {"labels": "ours/{uid}.npy", "camera": "ours/{uid}.json"}
    config = {
        "root": str(root), "manifest": "manifest.csv", "groups": ["assemblies", "solid_volume"],
        "reference": {"guidance": ref.copy(), "projection": ref.copy()},
        "methods": [{"key": "ours", "name": "Ours", "tables": [1, 2, 3], "guidance": pred.copy(), "projection": pred.copy()}],
        "metrics": {"boundary_tolerance_px": 0, "part_iou_threshold": 0.5},
    }
    config_path = root / "evaluation.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return root, config, config_path


def run_table(table, experiment):
    root, config, config_path = experiment
    script = SCRIPTS / f"eval_table{table}.py"
    assert script.is_file(), f"Missing eval_table{table}.py"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), "--config", str(config_path), "--out", str(root / "evaluation")], capture_output=True, text=True, encoding="utf-8")
    return result, root / "evaluation"


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_table1_exports_scores_and_does_not_invent_human_ratings(experiment):
    result, output = run_table(1, experiment)
    assert result.returncode == 0, result.stderr
    row = read_table(output / "table1.csv")[0]
    assert row["Proj. mIoU (%)"] == "100.00"
    assert row["Part Plausibility (1-5)"] == "TBD"
    assert row["Boundary Quality (1-5)"] == "TBD"
    detail = json.loads((output / "table1.json").read_text(encoding="utf-8"))
    assert detail["methods"][0]["projection"]["macro"]["gt_best_iou"] == 1


def test_table1_human_scores_are_averaged_by_object_before_groups(experiment):
    root, config, _ = experiment
    (root / "ratings.csv").write_text("method,group,uid,rater_id,plausibility,boundary_quality\nours,assemblies,u1,r1,5,5\nours,assemblies,u1,r2,1,1\nours,solid_volume,u2,r1,1,1\n", encoding="utf-8")
    config["human_ratings"] = "ratings.csv"
    result, output = run_table(1, experiment)
    assert result.returncode == 0, result.stderr
    assert read_table(output / "table1.csv")[0]["Part Plausibility (1-5)"] == "2.00"


def test_table2_missing_prediction_is_not_silently_dropped(experiment):
    root, _, _ = experiment
    (root / "ours" / "u2.npy").unlink()
    result, output = run_table(2, experiment)
    assert result.returncode == 0, result.stderr
    row = read_table(output / "table2.csv")[0]
    assert row["2D Matched mIoU (%)"] == "TBD"
    samples = read_table(output / "table2_per_sample.csv")
    assert any(row["status"] == "missing_prediction" and row["uid"] == "u2" for row in samples)


def test_camera_mismatch_makes_evaluation_exit_with_input_error(experiment):
    root, _, _ = experiment
    camera_path = root / "ours" / "u2.json"
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    camera["transform_matrix"][0][3] = 5
    camera_path.write_text(json.dumps(camera), encoding="utf-8")
    result, output = run_table(1, experiment)
    assert result.returncode == 2
    assert read_table(output / "table1.csv")[0]["Proj. mIoU (%)"] == "TBD"
    assert "camera" in (output / "table1.json").read_text(encoding="utf-8").lower()


def logs(experiment, *, segmentation_not_run=False):
    root, config, _ = experiment
    guidance, segmentation = [], []
    for group, uid in (("assemblies", "u1"), ("solid_volume", "u2")):
        guidance.append({"group": group, "uid": uid, "status": "partial", "timing_seconds": {"total": 8.0}, "peak_gpu_memory_gb": 4.0, "stage_status": {"segvigen": "not_run"}})
        segmentation.append({"group": group, "uid": uid, "status": "success", "timing_seconds": {"total": 12.0}, "peak_gpu_memory_gb": 6.0, "stage_status": {"segvigen": "not_run" if segmentation_not_run else "success"}, "collapsed": uid == "u2"})
    guidance.append({"group": "assemblies", "uid": "u1", "status": "skipped_existing", "timing_seconds": {"total": 0.0}})
    for name, records in (("guidance", guidance), ("segmentation", segmentation)):
        (root / "logs" / f"{name}.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    config["methods"][0]["run_components"] = [
        {"path": "logs/guidance.jsonl", "phase": "guidance"},
        {"path": "logs/segmentation.jsonl", "phase": "segmentation", "collapse_key": "collapsed"},
    ]


def test_table3_sums_stage_times_and_does_not_use_skipped_zero(experiment):
    logs(experiment)
    result, output = run_table(3, experiment)
    assert result.returncode == 0, result.stderr
    row = read_table(output / "table3.csv")[0]
    assert row["Total Time (s/object)"] == "20.00"
    assert row["Peak GPU Memory (GB)"] == "6.00"
    assert row["Run Failure Rate (%)"] == "0.00"
    assert row["Collapse Rate (%)"] == "50.00"


def test_guidance_success_does_not_imply_completed_3d_inference(experiment):
    logs(experiment, segmentation_not_run=True)
    result, output = run_table(3, experiment)
    assert result.returncode == 0, result.stderr
    row = read_table(output / "table3.csv")[0]
    assert row["Run Failure Rate (%)"] == "TBD"
    assert row["Total Time (s/object)"] == "TBD"
    assert row["Collapse Rate (%)"] == "TBD"


def test_table1_preserves_the_part_condition_column(experiment):
    experiment[1]["methods"][0]["part_condition"] = "Qwen3-VL spatial prompts"
    result, output = run_table(1, experiment)
    assert result.returncode == 0, result.stderr
    assert read_table(output / "table1.csv")[0]["Part Condition"] == "Qwen3-VL spatial prompts"


def test_nondefault_background_is_used_for_confirmed_missing_failures(experiment):
    root, config, config_path = experiment
    for path in (root / "reference").glob("*.npy"):
        labels = np.load(path)
        np.save(path, np.where(labels == 0, 7, labels))
    for path in (root / "ours").glob("*.npy"):
        path.unlink()
    config["metrics"]["background"] = 7
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = subprocess.run([sys.executable, str(SCRIPTS / "eval_table2.py"), "--config", str(config_path),
                             "--out", str(root / "evaluation"), "--missing-policy", "zero"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert read_table(root / "evaluation" / "table2.csv")[0]["2D Matched mIoU (%)"] == "0.00"


def test_partial_segmentation_is_not_assumed_to_be_a_valid_glb(experiment):
    logs(experiment)
    path = experiment[0] / "logs" / "segmentation.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["status"] = "partial"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result, output = run_table(3, experiment)
    assert result.returncode == 0, result.stderr
    assert read_table(output / "table3.csv")[0]["Run Failure Rate (%)"] == "TBD"


def test_table2_does_not_mislabel_a_different_part_threshold(experiment):
    experiment[1]["metrics"]["part_iou_threshold"] = 0.75
    result, _ = run_table(2, experiment)
    assert result.returncode == 2
    assert "0.5" in result.stderr

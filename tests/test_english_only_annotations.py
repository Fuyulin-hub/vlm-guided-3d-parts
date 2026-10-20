"""Check that name-only exports preserve segmentation identity and inputs."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "english_only_annotations.py"
spec = importlib.util.spec_from_file_location("english_only_annotations", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_names_are_removed_recursively_without_changing_labels_or_inputs():
    source = {"uid": "sample", "parts": [{"label_id": 7, "part_id": "part_03",
              "label_zh": "车轮", "label_en": "Wheel", "color": [1, 2, 3],
              "feedback": {"name_zh": "车轮", "status": "good"}}],
              "invalid_parts": [{"label_zh": "车身", "label_en": "Body"}],
              "final_map": {"labels_sha256": "a" * 64}}
    exported = module.english_only(source)
    assert exported["parts"][0]["label"] == "Wheel"
    assert "label_en" not in exported["parts"][0]
    assert exported["parts"][0]["label_id"] == 7
    assert exported["parts"][0]["color"] == [1, 2, 3]
    assert exported["parts"][0]["feedback"] == {"status": "good"}
    assert exported["invalid_parts"] == [{"label": "Body"}]
    assert exported["final_map"] == source["final_map"]
    assert source["parts"][0]["label_zh"] == "车轮"


def test_cli_exports_existing_annotations_and_rejects_overwriting(tmp_path):
    source = tmp_path / "input.json"
    target = tmp_path / "english.json"
    source.write_text(json.dumps({"parts": [{"label_en": "Seat", "label_zh": "座位"}]}), encoding="utf-8")
    command = [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(target)]
    assert subprocess.run(command, capture_output=True).returncode == 0
    assert json.loads(target.read_text(encoding="utf-8")) == {"parts": [{"label": "Seat"}]}
    previous = target.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert target.read_bytes() == previous


def test_canonical_label_takes_precedence_and_export_is_idempotent():
    source = {"parts": [{"label": "Wheel", "label_en": "Old wheel", "label_zh": "车轮"}]}
    exported = module.english_only(source)
    assert exported == {"parts": [{"label": "Wheel"}]}
    assert module.english_only(exported) == exported

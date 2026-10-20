"""CPU-only contract tests for the isolated C_v5 guidance runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen_sam2_guidance_v5_feedback.py"


def load_module():
    spec = importlib.util.spec_from_file_location("c_v5_feedback", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c_v5_contract_is_fixed_to_multi_point_negative_feedback():
    mod = load_module()

    config = mod.make_frozen_config(
        run_id="unit-test",
        manifest_path="manifest.csv",
        manifest_sha256="a" * 64,
        explicit_uids=["uid-1"],
        output_suffix="C_v5_feedback_unit",
    )

    assert config["schema_version"] == "c_v5_config@1"
    assert config["prompting"]["policy"] == "multi_point_negative_feedback"
    assert config["prompting"]["max_feedback_rounds"] == 1
    assert config["sam2"]["multimask_output"] is False
    assert config["fusion"]["priority"] == "bbox_area_then_cleaned_area_then_part_id_ascending"
    assert len(config["config_sha256"]) == 64


def test_point_connected_cleanup_removes_detached_noise_but_keeps_prompt_component():
    mod = load_module()
    foreground = np.ones((20, 20), dtype=bool)
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:10, 2:10] = True
    mask[16:18, 16:18] = True  # disconnected SAM speckle

    cleaned = mod.clean_candidate(mask, foreground, point_xy=[5, 5])

    assert cleaned[5, 5]
    assert cleaned[16:18, 16:18].sum() == 0


def test_small_prompt_supported_part_is_not_swallowed_by_large_mask():
    mod = load_module()
    foreground = np.ones((60, 60), dtype=bool)
    large = np.zeros((60, 60), dtype=bool)
    large[2:58, 2:58] = True
    small = np.zeros((60, 60), dtype=bool)
    small[20:32, 20:32] = True

    labels, accepted, rejected, audit = mod.fuse_candidates_point_aware(
        [
            {"part_id": "body", "region": large, "bbox_xyxy": [2, 2, 58, 58], "positive_points": [[4, 4]]},
            {"part_id": "knob", "region": small, "bbox_xyxy": [20, 20, 32, 32], "positive_points": [[25, 25]]},
        ],
        foreground,
        min_mask_pixels=4,
    )

    knob = next(part for part in accepted if part["part_id"] == "knob")
    assert (labels == knob["label_id"]).sum() == 144
    assert labels[25, 25] == knob["label_id"]
    assert knob["mask_area_cleaned"] == 144
    assert not rejected
    assert audit["accepted_count"] == 2


def test_annotation_and_jsonl_event_keep_auditable_prompt_and_feedback_fields():
    mod = load_module()

    annotation = mod.build_annotation_record(
        run_id="unit-test",
        config_sha256="b" * 64,
        group="assemblies",
        uid="uid-1",
        image_size=[64, 64],
        parts=[
            {
                "part_id": "part_01",
                "label_zh": "座垫",
                "label_en": "seat",
                "sam_trace": [{"points_xy_px": [[20, 20], [21, 20], [4, 4]], "point_labels": [1, 1, 0]}],
                "feedback": {"status": "good", "applied": False},
            }
        ],
        invalid_parts=[],
        fusion={"strategy": mod.FUSION_STRATEGY, "accepted_count": 1, "rejected_count": 0},
        final_map={"sha256": "c" * 64, "painted_regions": 1},
        status="success",
    )
    event = mod.build_result_event(annotation, timing_seconds={"qwen": 0.1, "sam2": 0.2})

    assert annotation["schema_version"] == "c_v5_annotation@2"
    assert annotation["annotation_language"] == "en"
    assert "label_zh" not in annotation["parts"][0]
    assert "label_en" not in annotation["parts"][0]
    assert annotation["parts"][0]["label"] == "seat"
    assert annotation["parts"][0]["sam_trace"][0]["point_labels"] == [1, 1, 0]
    assert annotation["parts"][0]["feedback"]["status"] == "good"
    assert event["schema_version"] == "c_v5_result@1"
    assert event["stage_status"]["feedback"] == "success"
    assert event["counts"]["final_parts"] == 1

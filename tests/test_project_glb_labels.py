"""project_glb_labels.py 的确定性工程测试，不产生论文实验结果。"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "project_glb_labels.py"


def projector():
    assert SCRIPT.is_file(), "缺少 project_glb_labels.py"
    spec = importlib.util.spec_from_file_location("project_glb_labels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("TRUE", True), ("1", True), ("false", False), ("0", False), ("no", False)],
)
def test_parse_bool_accepts_explicit_cli_values(value, expected):
    assert projector().parse_bool(value) is expected


def test_parse_bool_rejects_ambiguous_text():
    with pytest.raises(ValueError, match="布尔"):
        projector().parse_bool("sometimes")


def test_view_azimuths_start_at_45_and_cover_circle_uniformly():
    m = projector()
    assert m.view_azimuths(1) == [45.0]
    assert m.view_azimuths(4) == [45.0, 135.0, 225.0, 315.0]
    with pytest.raises(ValueError, match="views"):
        m.view_azimuths(0)


def test_camera_matches_project_convention():
    m = projector()
    bounds = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    camera = m.camera_parameters(bounds, 45.0, 20.0, 42.0, 1024)
    c2w = np.asarray(camera["c2w"])
    rotation = c2w[:3, :3]
    assert np.allclose(camera["target"], [0.0, 0.0, 0.0])
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    target_h = np.r_[camera["target"], 1.0]
    target_camera = np.linalg.inv(c2w) @ target_h
    assert target_camera[:2] == pytest.approx([0.0, 0.0], abs=1e-7)
    assert target_camera[2] == pytest.approx(-camera["distance"])
    focal = 1024.0 / (2.0 * math.tan(math.radians(42.0) / 2.0))
    assert np.allclose(
        np.asarray(camera["K"]),
        [[focal, 0.0, 512.0], [0.0, focal, 512.0], [0.0, 0.0, 1.0]],
    )
    radius = math.sqrt(12.0) / 2.0
    assert camera["znear"] == pytest.approx(camera["distance"] - radius * 2.2)
    assert camera["zfar"] == pytest.approx(camera["distance"] + radius * 2.2)


def test_camera_rejects_empty_or_degenerate_bounds():
    m = projector()
    with pytest.raises(ValueError, match="包围盒"):
        m.camera_parameters(np.zeros((2, 3)), 45.0, 20.0, 42.0, 1024)
    bad = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]])
    with pytest.raises(ValueError, match="包围盒"):
        m.camera_parameters(bad, 45.0, 20.0, 42.0, 1024)


def test_camera_accepts_planar_nonzero_scene_bounds():
    camera = projector().camera_parameters(
        np.array([[-1.0, -2.0, 0.0], [1.0, 2.0, 0.0]]),
        45.0,
        20.0,
        42.0,
        1024,
    )
    assert camera["radius"] > 0
    assert 0 < camera["znear"] < camera["zfar"]


def test_palette_loader_accepts_rgba_and_drops_alpha(tmp_path):
    path = tmp_path / "palette.json"
    path.write_text(json.dumps([[9, 9, 9, 255], [10, 20, 30, 255], [40, 50, 60, 255]]))
    palette = projector().load_palette(path)
    assert palette.dtype == np.uint8
    assert palette.tolist() == [[9, 9, 9], [10, 20, 30], [40, 50, 60]]


def test_quantization_is_thresholded_deterministic_and_background_aware():
    m = projector()
    palette = np.array([[99, 99, 99], [10, 0, 0], [14, 0, 0]], dtype=np.uint8)
    rgb = np.array([[[12, 0, 0], [14, 0, 0], [40, 0, 0], [10, 0, 0]]], dtype=np.uint8)
    foreground = np.array([[True, True, True, False]])
    indices, stats = m.quantize_palette(rgb, foreground, palette, threshold=5.0)
    assert indices.dtype == np.uint16
    assert indices.tolist() == [[1, 2, 0, 0]]  # 12 等距时固定选择较小 index=1
    assert stats == {"foreground_pixels": 3, "hit_pixels": 2, "unknown_pixels": 1, "hit_rate": 2 / 3}


def test_quantization_rejects_nonempty_foreground_with_zero_palette_hits():
    m = projector()
    with pytest.raises(RuntimeError, match="未命中"):
        m.quantize_palette(
            np.array([[[250, 250, 250]]], dtype=np.uint8),
            np.array([[True]]),
            np.array([[0, 0, 0], [10, 20, 30]], dtype=np.uint8),
            threshold=1.0,
        )


def test_annotation_mapping_is_validated_and_missing_annotation_has_identity_fallback(tmp_path):
    m = projector()
    palette = np.array([[0, 0, 0], [10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    annotation = tmp_path / "ann.json"
    annotation.write_text(
        json.dumps(
            {
                "parts": [
                    {"label_id": 1, "palette_index": 1, "color": [10, 20, 30, 255]},
                    {"label_id": 2, "palette_index": 2, "color": [40, 50, 60, 255]},
                ]
            }
        )
    )
    mapping, info = m.load_label_mapping(annotation, palette)
    assert mapping == {1: 1, 2: 2}
    assert info["source"] == "annotation"
    fallback, fallback_info = m.load_label_mapping(tmp_path / "missing.json", palette)
    assert fallback == {1: 1, 2: 2}
    assert fallback_info["source"] == "palette_identity_fallback"


def test_annotation_mapping_merges_irreversible_color_collision_deterministically(tmp_path):
    m = projector()
    palette = np.array([[0, 0, 0], [10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    annotation = tmp_path / "collision.json"
    annotation.write_text(
        json.dumps(
            {
                "parts": [
                    {"label_id": 1, "palette_index": 1, "color": [10, 20, 30]},
                    {"label_id": 3, "palette_index": 1, "color": [10, 20, 30]},
                ]
            }
        )
    )
    mapping, info = m.load_label_mapping(annotation, palette)
    assert mapping == {1: 1}
    assert info["palette_collisions"] == [
        {"palette_index": 1, "label_ids": [1, 3], "selected_label_id": 1}
    ]


def test_palette_indices_map_to_uint16_labels_and_count_unmapped():
    m = projector()
    indices = np.array([[0, 1, 2, 3]], dtype=np.uint16)
    labels, stats = m.palette_indices_to_labels(indices, {1: 7, 2: 8})
    assert labels.dtype == np.uint16
    assert labels.tolist() == [[0, 7, 8, 0]]
    assert stats == {"palette_hit_pixels": 3, "mapped_pixels": 2, "unmapped_pixels": 1, "mapping_rate": 2 / 3}


def test_foreground_uses_depth_but_still_rejects_globally_empty_alpha():
    m = projector()
    rgba = np.array(
        [[[255, 255, 255, 0], [10, 20, 30, 255], [40, 50, 60, 0]]],
        dtype=np.uint8,
    )
    depth = np.array([[0.0, 1.0, 2.0]], dtype=np.float32)
    assert m.foreground_from_render(rgba, depth).tolist() == [[False, True, True]]
    with pytest.raises(RuntimeError, match="全空"):
        m.foreground_from_render(np.zeros((2, 2, 4), np.uint8), np.ones((2, 2), np.float32))


def test_manifest_validates_group_uniqueness_and_path_components(tmp_path):
    m = projector()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "group,uid,input_rgba,width,height,mode,format\n"
        "assemblies,a1,/tmp/a.png,1024,1024,RGBA,PNG\n"
        "solid_volume,b2,/tmp/b.png,1024,1024,RGBA,PNG\n",
        encoding="utf-8",
    )
    records = m.load_manifest(manifest)
    assert [(row["group"], row["uid"]) for row in records] == [("assemblies", "a1"), ("solid_volume", "b2")]
    manifest.write_text(
        "group,uid,input_rgba,width,height,mode,format\n"
        "assemblies,a1,/tmp/a.png,1,1,RGBA,PNG\n"
        "assemblies,a1,/tmp/a.png,1,1,RGBA,PNG\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        m.load_manifest(manifest)
    manifest.write_text(
        "group,uid,input_rgba,width,height,mode,format\n"
        "assemblies,../escape,/tmp/a.png,1,1,RGBA,PNG\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="安全"):
        m.load_manifest(manifest)


def test_reference_consistency_is_label_permutation_invariant():
    m = projector()
    reference = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.uint16)
    prediction = np.array([[0, 7, 7], [9, 9, 0]], dtype=np.uint16)
    metrics = m.reference_consistency(prediction, reference)
    assert metrics["status"] == "ok"
    assert metrics["foreground_iou"] == pytest.approx(1.0)
    assert metrics["matched_miou"] == pytest.approx(1.0)
    assert metrics["prediction_parts"] == 2
    assert metrics["reference_parts"] == 2


def test_reference_shape_mismatch_is_reported_without_resizing():
    metrics = projector().reference_consistency(
        np.ones((2, 2), dtype=np.uint16),
        np.ones((3, 3), dtype=np.uint16),
    )
    assert metrics["status"] == "incompatible_shape"
    assert metrics["prediction_shape"] == [2, 2]
    assert metrics["reference_shape"] == [3, 3]


def test_cli_parses_allow_overwrite_false_as_false(tmp_path):
    m = projector()
    args = m.build_parser().parse_args(
        [
            "--arm",
            "B_sam2",
            "--manifest",
            str(tmp_path / "manifest.csv"),
            "--root",
            str(tmp_path),
            "--allow-overwrite",
            "false",
        ]
    )
    assert args.allow_overwrite is False
    assert args.views == 1
    assert args.size == 1024
    assert args.color_threshold == 64.0


def test_atomic_bundle_commits_v00_as_final_cache_sentinel(tmp_path, monkeypatch):
    m = projector()
    labels = {
        "v00": np.array([[1, 0], [0, 1]], dtype=np.uint16),
        "v01": np.array([[2, 0], [0, 2]], dtype=np.uint16),
    }
    order = []
    real_replace = m.os.replace

    def recording_replace(source, destination):
        order.append(Path(destination).name)
        real_replace(source, destination)

    monkeypatch.setattr(m.os, "replace", recording_replace)
    m.write_projection_bundle(
        tmp_path,
        labels,
        {"views": []},
        {"source_glb": {"path": "/tmp/example.glb"}},
    )
    assert order[-1] == "label_v00.png"
    assert (tmp_path / "label_v00.png").is_file()
    assert (tmp_path / "label_v01.png").is_file()
    assert (tmp_path / "cameras.json").is_file()
    assert (tmp_path / "projection_meta.json").is_file()


def test_failed_overwrite_removes_old_v00_cache_sentinel(tmp_path, monkeypatch):
    m = projector()
    old_sentinel = tmp_path / "label_v00.png"
    old_sentinel.write_bytes(b"old-cache-sentinel")
    labels = {
        "v00": np.array([[1]], dtype=np.uint16),
        "v01": np.array([[2]], dtype=np.uint16),
    }
    real_replace = m.os.replace

    def fail_during_commit(source, destination):
        if Path(destination).name == "projection_meta.json":
            raise OSError("simulated commit failure")
        real_replace(source, destination)

    monkeypatch.setattr(m.os, "replace", fail_during_commit)
    with pytest.raises(OSError, match="simulated"):
        m.write_projection_bundle(tmp_path, labels, {"views": []}, {"schema": 1})
    assert not old_sentinel.exists(), "提交失败后不能保留会触发 skipped_cached 的旧 v00"


def test_overwrite_removes_stale_label_views(tmp_path):
    m = projector()
    for view_id in range(4):
        (tmp_path / f"label_v{view_id:02d}.png").write_bytes(b"old")
    m.write_projection_bundle(
        tmp_path,
        {"v00": np.array([[1]], dtype=np.uint16)},
        {"views": [{"view_id": "v00"}]},
        {"schema": 1},
    )
    assert sorted(path.name for path in tmp_path.glob("label_v*.png")) == ["label_v00.png"]


def test_process_one_reports_missing_and_cached_without_renderer(tmp_path):
    m = projector()
    args = SimpleNamespace(
        root=tmp_path,
        arm="B_sam2",
        allow_overwrite=False,
        views=1,
        size=64,
        color_threshold=24.0,
        with_reference=False,
        reference_root=tmp_path / "reference",
        palette=tmp_path / "palette.json",
        device="cuda:0",
    )
    record = {"group": "assemblies", "uid": "sample"}
    palette = np.array([[0, 0, 0], [10, 20, 30]], dtype=np.uint8)
    missing = m.process_one(record, args, palette)
    assert missing["status"] == "skipped_missing"
    glb = tmp_path / "results" / "B_sam2" / "assemblies" / "sample" / "segmented.glb"
    glb.parent.mkdir(parents=True)
    glb.write_bytes(b"placeholder")
    sentinel = glb.parent / "projection" / "label_v00.png"
    sentinel.parent.mkdir()
    sentinel.write_bytes(b"cached")
    cached = m.process_one(record, args, palette)
    assert cached["status"] == "skipped_cached"


def test_object_renderer_is_deleted_even_when_rendering_raises(monkeypatch):
    m = projector()

    class FakeRenderer:
        deleted = False

        def delete(self):
            self.deleted = True

    fake = FakeRenderer()
    monkeypatch.setattr(m, "create_renderer", lambda size: fake)
    with pytest.raises(RuntimeError, match="render failed"):
        with m.object_renderer(64):
            raise RuntimeError("render failed")
    assert fake.deleted is True


def test_script_contains_no_background_nvidia_smi_polling():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "threading.Thread" not in source
    assert "_sample_loop" not in source


def test_egl_flat_render_preserves_two_texture_palette_colors(tmp_path):
    m = projector()
    # 用真实 TextureVisuals 合成一个立方体；UV 按 x/y 归一化，纹理左右各一种纯色。
    import trimesh
    from PIL import Image

    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    vertices = np.asarray(mesh.vertices)
    uv = (vertices[:, :2] - vertices[:, :2].min(axis=0)) / np.ptp(vertices[:, :2], axis=0)
    texture = np.zeros((8, 8, 4), dtype=np.uint8)
    texture[:, :4] = [10, 20, 30, 255]
    texture[:, 4:] = [40, 50, 60, 255]
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[255, 255, 255, 255],
        baseColorTexture=Image.fromarray(texture, mode="RGBA"),
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    scene = trimesh.Scene(mesh)
    glb_path = tmp_path / "textured.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))

    loaded, bounds = m.load_textured_scene(glb_path)
    render_scene = m.prepare_pyrender_scene(loaded)
    camera = m.camera_parameters(bounds, 45.0, 20.0, 42.0, 64)
    renderer = m.create_renderer(64)
    try:
        rgba, depth = m.render_flat(renderer, render_scene, camera)
    finally:
        renderer.delete()
    foreground = m.foreground_from_render(rgba, depth)
    palette = np.array([[0, 0, 0], [10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    indices, stats = m.quantize_palette(rgba[:, :, :3], foreground, palette, threshold=8.0)
    assert stats["foreground_pixels"] > 100
    assert stats["hit_rate"] > 0.90
    assert set(np.unique(indices[foreground])) >= {1, 2}

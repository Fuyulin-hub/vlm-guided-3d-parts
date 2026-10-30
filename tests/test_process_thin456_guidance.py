"""Behavioral checks for source-preserving thin456 normalization."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def save_image(path, color=(90, 90, 90, 255), size=(16, 16)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", size, color)
    if path.suffix == ".jpg":
        image = image.convert("RGB")
    image.save(path)


def test_inventory_matches_hum_whitespace_and_retains_missing_pair(tmp_path):
    # Removing whitespace tolerance or excluding unpaired raw files breaks this.
    from process_thin456_guidance import inventory_sources
    paired, missing = "a" * 32, "b" * 32
    save_image(tmp_path / "thin6" / f"{paired}.png")
    save_image(tmp_path / "thin6" / f"{paired}_hum .jpg")
    save_image(tmp_path / "thin5" / f"{missing}.png")
    rows = inventory_sources(tmp_path)
    assert [row["uid"] for row in rows] == [paired, missing]
    assert rows[0]["human_file"].endswith("_hum .jpg")
    assert rows[1]["human_file"] is None
    assert rows[1]["status"] == "pending_missing_human_annotation"


def test_duplicate_differing_humans_require_explicit_correspondence_selection(tmp_path):
    # A global extension preference would silently accept an unreviewed duplicate.
    from process_thin456_guidance import inventory_sources
    uid = "c" * 32
    for group in ["thin5", "thin6"]:
        save_image(tmp_path / group / f"{uid}.png")
    save_image(tmp_path / "thin5" / f"{uid}_hum.jpg", (255, 0, 0, 255))
    save_image(tmp_path / "thin6" / f"{uid}_hum.png", (0, 0, 255, 255))
    unresolved = inventory_sources(tmp_path)[0]
    assert unresolved["status"] == "pending_conflicting_human_annotations"
    assert unresolved["human_file"] is None
    preference = {uid: {"path": f"thin6/{uid}_hum.png", "reason": "visual_raw_correspondence", "rejected_status": "source_annotation_mismatch"}}
    selected = inventory_sources(tmp_path, preference)[0]
    assert selected["human_file"] == f"thin6/{uid}_hum.png"
    assert selected["duplicate_resolution"]["reason"] == "visual_raw_correspondence"
    assert selected["duplicate_resolution"]["alternatives"][0]["status"] == "source_annotation_mismatch"
    assert len(selected["raw_sources"]) == 2


def test_diagnostic_raw_mask_removes_opaque_near_white_without_touching_dark_pixels():
    # Treating opaque alpha as silhouette would count all four pixels as foreground.
    from process_thin456_guidance import diagnostic_raw_mask
    rgba = np.array([[[255,255,255,255], [250,249,248,255], [220,210,205,255], [35,35,35,255]]], dtype=np.uint8)
    mask, info = diagnostic_raw_mask(rgba)
    assert mask.tolist() == [[False, False, True, True]]
    assert info["method"] == "alpha_128_excluding_near_white_neutral_245_range_15"
    assert info["near_white_removed_pixels"] == 2


def test_raw_white_foreground_is_preserved_when_alpha_defines_real_background():
    # Applying the opaque-white fallback to a transparent raw erases white objects.
    from process_thin456_guidance import diagnostic_raw_mask
    rgba = np.array([[[255,255,255,255],[25,25,25,255],[255,255,255,0]]], dtype=np.uint8)
    mask, info = diagnostic_raw_mask(rgba)
    assert mask.tolist() == [[True,True,False]]
    assert info["method"] == "alpha_128_with_meaningful_transparency"
    assert info["near_white_removed_pixels"] == 0


def test_candidate_encoding_preserves_silhouette_and_palette_alpha_ids():
    # Clipping to raw or collapsing integer IDs would lose an annotated region.
    from process_thin456_guidance import encode_candidate
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    rgba[2:12, 2:12] = [240,80,20,255]
    rgba[2:12, 18:28] = [20,80,240,255]
    source = {7: (240,80,20), 12: (20,80,240)}
    target = [(229,81,22), (22,81,229)]
    labels, guide, mapping, info = encode_candidate(rgba, source, target)
    assert labels.dtype == np.uint16
    assert set(np.unique(labels)) == {0,7,12}
    assert np.array_equal(labels > 0, rgba[...,3] > 0)
    assert np.array_equal(guide[...,3] > 0, labels > 0)
    assert set(np.unique(guide[...,3])) == {0,255}
    assert np.all(guide[labels == 0] == 0)
    assert info["status"] == "automatic_color_normalization_requires_review"
    for label_id, index in mapping.items():
        assert np.all(guide[labels == label_id,:3] == target[index-1])


def test_source_sheets_include_missing_card_and_do_not_modify_sources(tmp_path):
    # Omitting pending records would produce fewer than two visible inventory cards.
    from process_thin456_guidance import inventory_sources, make_source_sheets
    source = tmp_path / "sources"
    for uid in ["a" * 32, "b" * 32]:
        save_image(source / "thin4" / f"{uid}.png")
    save_image(source / "thin4" / f"{'a' * 32}_hum.jpg")
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    rows = inventory_sources(source)
    sheets = make_source_sheets(source, rows, tmp_path / "review")
    assert len(sheets) == 1
    assert Image.open(sheets[0]).size == (1200, 1500)
    assert all(path.read_bytes() == content for path, content in before.items())


def test_process_withholds_size_mismatch_and_missing_annotations(tmp_path):
    # Auto-resizing the mismatch or synthesizing the missing human would violate scope.
    from process_thin456_guidance import process_directory
    source = tmp_path / "sources"
    uid, missing = "a" * 32, "b" * 32
    save_image(source / "thin4" / f"{uid}.png", size=(20,20))
    save_image(source / "thin4" / f"{uid}_hum.png", size=(10,10))
    save_image(source / "thin4" / f"{missing}.png", size=(20,20))
    receipt = process_directory(source, tmp_path / "output", {1:(90,90,90)}, [(229,81,22)], expected_size=(20,20))
    assert receipt["inventory_count"] == 2
    assert receipt["candidate_count"] == 0
    assert {r["status"] for r in receipt["per_uid"]} == {"pending_geometry_mismatch", "pending_missing_human_annotation"}
    assert not list((tmp_path / "output").rglob("labels_uint16.png"))


def test_process_roundtrip_serializes_provenance_labels_mapping_and_review(tmp_path):
    # A lossy label save, broken mapping, or absent review links fails this roundtrip.
    from process_thin456_guidance import process_directory
    uid = "d" * 32
    source, output = tmp_path / "source", tmp_path / "output"
    raw_path = source / "thin4" / f"{uid}.png"
    save_image(raw_path, size=(20,20))
    human_path = source / "thin4" / f"{uid}_hum.png"
    rgba = np.zeros((20,20,4), dtype=np.uint8)
    rgba[2:10,2:10] = (240,80,20,255)
    rgba[10:18,10:18] = (20,80,240,255)
    Image.fromarray(rgba).save(human_path)
    original = raw_path.read_bytes()
    receipt = process_directory(source, output, {7:(240,80,20),12:(20,80,240)},
                                [(229,81,22),(22,81,229)], expected_size=(20,20))
    assert receipt["candidate_count"] == 1
    row = receipt["per_uid"][0]
    labels = np.asarray(Image.open(output/row["labels_file"]))
    guide = np.asarray(Image.open(output/row["guide_file"]))
    metadata = json.loads((output/row["mapping_file"]).read_text(encoding="utf-8"))
    assert labels.dtype == np.uint16
    assert set(np.unique(labels)) == {0,7,12}
    assert np.array_equal(guide[...,3] > 0, labels > 0)
    assert metadata["label_to_bc_palette_index"] == row["label_to_bc_palette_index"]
    assert metadata["color_to_ids"]
    assert (output/row["raw_file"]).read_bytes() == original
    assert metadata["status"] == "automatic_color_normalization_requires_review"
    review = (output/"review"/"index.html").read_text(encoding="utf-8")
    assert uid in review and row["mapping_file"] in review

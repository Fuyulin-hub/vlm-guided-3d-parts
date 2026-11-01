"""Reviewed source colors must survive encoding without creating blend IDs."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
ROOT=Path(__file__).resolve().parents[1]


def palettes():
    source={int(x["label_id"]):tuple(x["rgb"]) for x in json.loads((ROOT/"configs/palettes/part_palette_30_high_contrast_20261021.json").read_text())["colors"]}
    target=[tuple(x[:3]) for x in json.loads((ROOT/"configs/palettes/colors_pure_bc_20261022.json").read_text())[1:]]
    return source,target


def test_jpeg_blends_cannot_add_ids_and_one_pixel_blue_line_survives():
    # Hue peak detection would add cyan/teal IDs; eroding all seeds could erase blue.
    from process_thin456_guidance_v2 import encode_reviewed_colors
    from repair_d_guidance_images import foreground_mask
    source,target=palettes()
    rgba=np.zeros((18,24,4),dtype=np.uint8)
    rgba[2:16,2:22]=[*source[1],255]
    rgba[2:16,10]=[32,159,122,255]  # JPEG-like intermediate RGB
    rgba[2:16,11]=[*source[3],255]
    rgba[2:16,20]=[*source[2],255]  # legitimate narrow blue
    labels,guide,mapping,info=encode_reviewed_colors(rgba,source,target,[1,2,3])
    assert set(np.unique(labels))=={0,1,2,3}
    assert np.all(labels[2:16,20]==2)
    assert np.array_equal(labels>0,foreground_mask(rgba))
    assert np.array_equal(guide[...,3]>0,labels>0)
    assert info["active_source_palette_indices"]==[1,2,3]
    assert info["classification_distance_rgb"]["maximum"]>0
    assert info["ground_truth_ready"] is False


def test_distinct_cyan_dark_teal_yellow_gold_pink_darkpink_remain_distinct():
    # Hue-only grouping merges these brightness-distinct, separately painted colors.
    from process_thin456_guidance_v2 import encode_reviewed_colors
    source,target=palettes()
    ids=[4,10,5,12,9,7]
    rgba=np.array([[[*source[index],255] for index in ids]],dtype=np.uint8)
    labels,guide,mapping,info=encode_reviewed_colors(rgba,source,target,ids)
    assert labels.tolist()==[ids]
    assert set(mapping)==set(ids)
    assert len(set(mapping.values()))==len(ids)
    assert info["classification_distance_rgb"]["maximum"]==0


def test_reviewed_black_extension_uses_explicit_id31_only():
    from process_thin456_guidance_v2 import encode_reviewed_colors
    source,target=palettes()
    rgba=np.array([[[35,35,35,255],[*source[1],255]]],dtype=np.uint8)
    labels,guide,mapping,info=encode_reviewed_colors(rgba,source,target,[1,31])
    assert labels.tolist()==[[31,1]]
    assert info["source_rgb_by_candidate_id"]["31"]==[35,35,35]
    assert "neutral_region_semantics_unconfirmed" in info["encoding_quality_flags"]


def test_v2_preserves_v1_and_unresolved_outputs_and_keeps_missing_pending(tmp_path):
    from process_thin456_guidance import process_directory
    from process_thin456_guidance_v2 import process_from_v1
    source,target=palettes()
    inputs=tmp_path/"inputs"/"thin4"
    inputs.mkdir(parents=True)
    reviewed,unresolved,missing="a"*32,"b"*32,"c"*32
    raw=np.zeros((20,20,4),dtype=np.uint8)
    raw[3:17,3:17]=[80,80,80,255]
    human=raw.copy()
    human[3:17,3:10]=[*source[1],255]
    human[3:17,10:17]=[*source[2],255]
    for uid in [reviewed,unresolved,missing]:
        Image.fromarray(raw).save(inputs/f"{uid}.png")
        if uid!=missing:
            Image.fromarray(human).save(inputs/f"{uid}_hum.png")
    v1=tmp_path/"v1"
    process_directory(inputs.parent,v1,source,target,expected_size=(20,20))
    original={p.relative_to(v1):p.read_bytes() for p in v1.rglob("*") if p.is_file()}
    receipt=process_from_v1(v1,tmp_path/"v2",source,target,{reviewed:[1,2]}, {unresolved},expected_size=(20,20))
    assert (receipt["reviewed_color_encoding_count"],receipt["unresolved_candidate_count"],receipt["pending_count"])==(1,1,1)
    assert all((v1/p).read_bytes()==content for p,content in original.items())
    rows={row["uid"]:row for row in receipt["per_uid"]}
    assert all(row["ground_truth_ready"] is False for row in rows.values())
    assert rows[unresolved]["evaluable"] is False
    assert "color_schema_unresolved" in rows[unresolved]["quality_flags"]
    for key in ["labels_file","guide_file"]:
        assert (tmp_path/"v2"/rows[unresolved][key]).read_bytes()==(v1/rows[unresolved][key]).read_bytes()
    assert rows[missing]["status"]=="pending_missing_human_annotation"
    assert "guide_file" not in rows[missing]
    review=(tmp_path/"v2"/"review"/"index.html").read_text(encoding="utf-8")
    assert "1 张已核对来源颜色编码" in review
    assert "1 张颜色定义未解决候选" in review
    assert "1 张待补" in review

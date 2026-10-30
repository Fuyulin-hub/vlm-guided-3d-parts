"""Normalize supplied thin456 colored annotations; never manufacture missing labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_d_repair_review import thumb
from repair_d_guidance_images import infer_regions
from recolor_d_guidance_to_bc_palette import recolor_labels

ROOT = Path(__file__).resolve().parents[1]
STATUS = "automatic_color_normalization_requires_review"
NAME_RE = re.compile(r"^([0-9a-f]{32})(?:_(hum|raw)\s*)?\.(png|jpe?g)$", re.I)
PREFERENCES = {
    "bdcd141c36b4465fa4fb18620af3b12b": {
        "path": "thin6/bdcd141c36b4465fa4fb18620af3b12b_hum.png",
        "reason": "visually_verified_raw_correspondence_png_screwdriver_jpg_tapered_rod",
        "rejected_status": "source_annotation_mismatch",
        "note": "Rejected JPG may be misnamed and resemble b2b57619fe3244d4aafbf1e8639b418f; no reassignment authorized. Missing annotation remains pending.",
    }
}
VISUAL_FLAGS = {
    "b5a4ae23bed040e8af561a4f1293f94d": ["visual_source_enlarged_or_reposed_vs_raw"],
    "c2cf3c69a2b74f97941a9072334e08c5": ["visual_source_scale_or_framing_differs_from_raw"],
    "b2595bc8f28a4c8a87db7e48c04f0d79": ["visual_raw_white_plane_background_role_unconfirmed"],
    "ad2ce94a931a44c1b6ceb497351568db": ["incomplete_color_annotation_semantics_review", "gray_framework_ids_not_verified_parts"],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory_sources(source_root, preferences=None):
    """Keep all input variants, resolve only identical bytes or explicit reviewed choices."""
    source_root = Path(source_root)
    preferences = preferences or {}
    buckets = {}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        match = NAME_RE.fullmatch(path.name)
        if not match:
            raise ValueError(f"unrecognized source filename: {path}")
        uid, role, extension = match.groups()
        kind = "human" if role and role.lower() == "hum" else "raw"
        with Image.open(path) as image:
            entry = {"file": path.relative_to(source_root).as_posix(), "sha256": sha(path),
                     "size": list(image.size), "mode": image.mode, "bytes": path.stat().st_size}
        buckets.setdefault(uid.lower(), {"raw": [], "human": []})[kind].append(entry)
    rows = []
    for uid, variants in sorted(buckets.items()):
        raws, humans = variants["raw"], variants["human"]
        row = {"uid": uid, "group": "thin_fine_grained", "raw_sources": raws,
               "human_sources": humans, "raw_file": None, "human_file": None,
               "status": "ready_for_encoding"}
        if not raws:
            row["status"] = "pending_missing_raw"
        elif len({item["sha256"] for item in raws}) > 1:
            row["status"] = "pending_conflicting_raw_images"
        else:
            row["raw_file"] = raws[0]["file"]
            row["raw_sha256"] = raws[0]["sha256"]
            row["source_group"] = Path(raws[0]["file"]).parts[0]
        if not humans:
            row["status"] = "pending_missing_human_annotation"
        elif len({item["sha256"] for item in humans}) == 1:
            chosen = humans[0]
            row["human_file"] = chosen["file"]
            row["human_sha256"] = chosen["sha256"]
        elif uid in preferences:
            preference = preferences[uid]
            chosen = next((item for item in humans if item["file"] == preference["path"]), None)
            if chosen is None:
                raise ValueError(f"reviewed source selection not found: {uid}")
            row["human_file"], row["human_sha256"] = chosen["file"], chosen["sha256"]
            row["duplicate_resolution"] = {
                "selected": chosen["file"], "reason": preference["reason"],
                "note": preference.get("note", ""),
                "alternatives": [dict(item, status=preference.get("rejected_status", "unselected_variant"))
                                 for item in humans if item is not chosen],
            }
        else:
            row["status"] = "pending_conflicting_human_annotations"
        rows.append(row)
    return rows


def make_source_sheets(source_root, rows, review_dir):
    review_dir = Path(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    paths = []
    for page, start in enumerate(range(0, len(rows), 10), 1):
        sheet = Image.new("RGB", (1200, 1500), "white")
        draw = ImageDraw.Draw(sheet)
        for slot, row in enumerate(rows[start:start+10]):
            x, y = (slot % 2) * 600, (slot // 2) * 300
            draw.text((x+10, y+5), f"{start+slot+1:02d} {row['uid'][:16]} / {row.get('source_group', '?')}", fill="black", font=font)
            for j, (key, caption) in enumerate([("raw_file", "RAW"), ("human_file", "D SOURCE")]):
                draw.text((x+10+j*300, y+28), caption, fill="black", font=font)
                if row.get(key):
                    with Image.open(Path(source_root)/row[key]) as image:
                        sheet.paste(thumb(image, (280,240)), (x+10+j*300, y+52))
                else:
                    draw.text((x+310, y+140), "MISSING / PENDING", fill="red", font=font)
        path = review_dir / f"source_{page:02d}.jpg"
        sheet.save(path, quality=95)
        paths.append(path)
    return paths


def diagnostic_raw_mask(rgba):
    """Trust meaningful raw alpha; approximate white only for fully opaque inputs."""
    rgb = rgba[..., :3].astype(np.int16)
    visible = rgba[..., 3] >= 128
    if np.any(~visible):
        near_white = np.zeros_like(visible)
        method = "alpha_128_with_meaningful_transparency"
    else:
        near_white = (rgb.min(axis=2) >= 245) & ((rgb.max(axis=2)-rgb.min(axis=2)) <= 15)
        method = "alpha_128_excluding_near_white_neutral_245_range_15"
    mask = visible & ~near_white
    return mask, {"method": method,
                  "alpha_opaque_pixels": int(np.sum(rgba[...,3] == 255)),
                  "near_white_removed_pixels": int(np.sum(visible & near_white)),
                  "foreground_pixels": int(mask.sum()),
                  "purpose": "diagnostic_only_not_used_to_clip_or_transform_annotation"}


def encode_candidate(rgba, source_palette, target_colors):
    labels, colors, info = infer_regions(rgba, source_palette)
    guide, mapping = recolor_labels(labels, colors, target_colors)
    if not np.array_equal(guide[...,3] == 255, labels > 0):
        raise AssertionError("label/alpha mismatch")
    for label_id, index in mapping.items():
        if not np.all(guide[labels == label_id, :3] == target_colors[index-1]):
            raise AssertionError("label/palette mismatch")
    info.update(status=STATUS, source_rgb_by_candidate_id={str(k): list(v) for k,v in colors.items()},
                candidate_region_meaning="source_color_or_hue_group_not_verified_semantic_part",
                geometric_transform="none", raw_clipping=False)
    return labels, guide, mapping, info


def process_directory(source_root, output, source_palette, target_colors, preferences=None,
                      expected_size=(1024,1024), provenance=None):
    source_root, output = Path(source_root), Path(output)
    if (output/"normalization_receipt.json").exists():
        raise FileExistsError("normalization receipt already exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = inventory_sources(source_root, preferences)
    # Byte-preserving copies only; the default source folder is already inside output.
    target_source = output/"sources"
    if source_root.resolve() != target_source.resolve():
        shutil.copytree(source_root, target_source, dirs_exist_ok=False)
    make_source_sheets(target_source, rows, output/"review")
    receipt = {
        "status": STATUS, "source_archive": "thin456.zip", "provenance": provenance or {},
        "inventory_count": len(rows), "candidate_count": 0, "pending_count": 0,
        "bc_foreground_palette_indices": list(range(1,len(target_colors)+1)),
        "min_gap_px_for_color_reuse": 5, "geometry_policy": "no_rotate_mirror_resize_align_or_raw_clipping",
        "label_meaning": "source_color_or_hue_group_candidate_not_semantic_ground_truth",
        "raw_source_bytes_preserved": True, "per_uid": [],
    }
    for index, source_row in enumerate(rows,1):
        row = dict(source_row)
        for key in ["raw_file", "human_file"]:
            if row[key]:
                row[key] = "sources/" + row[key]
        row["quality_flags"] = list(VISUAL_FLAGS.get(row["uid"], []))
        if row.get("duplicate_resolution"):
            row["quality_flags"].append("duplicate_human_source_resolved_by_visual_correspondence")
        if row["status"] != "ready_for_encoding":
            row["quality_flags"].append(row["status"])
            receipt["pending_count"] += 1
            receipt["per_uid"].append(row)
            continue
        with Image.open(output/row["raw_file"]) as image:
            raw_rgba = np.asarray(image.convert("RGBA"))
            raw_size = image.size
        with Image.open(output/row["human_file"]) as image:
            human_rgba = np.asarray(image.convert("RGBA"))
            human_size = image.size
        row.update(raw_size=list(raw_size), human_size=list(human_size))
        if raw_size != human_size or raw_size != tuple(expected_size):
            row["status"] = "pending_geometry_mismatch"
            row["quality_flags"].append(row["status"])
            receipt["pending_count"] += 1
            receipt["per_uid"].append(row)
            continue
        labels, guide, mapping, info = encode_candidate(human_rgba, source_palette, target_colors)
        row.update(info)
        raw_mask, raw_info = diagnostic_raw_mask(raw_rgba)
        mask = labels > 0
        overlap, union = int(np.sum(mask & raw_mask)), int(np.sum(mask | raw_mask))
        row.update(raw_mask_diagnostic=raw_info, raw_silhouette_iou=overlap/max(1,union),
                   raw_foreground_coverage=overlap/max(1,int(raw_mask.sum())),
                   annotation_outside_raw_pixels=int(np.sum(mask & ~raw_mask)),
                   annotation_foreground_pixels=int(mask.sum()),
                   source_alpha_visible_pixels=int(np.sum(human_rgba[...,3] >= 128)),
                   label_count=len(mapping), guide_foreground_pixels=int(np.sum(guide[...,3] == 255)),
                   geometry_preserved=True)
        if info["method"] == "dominant_hue_regions":
            row["quality_flags"].append("inferred_hue_groups_not_verified_parts")
        if info.get("neutral_region"):
            row["quality_flags"].append("neutral_region_semantics_unconfirmed")
        if info["uncertain_fraction"] > 0.02:
            row["quality_flags"].append("color_ambiguity")
        if Path(row["human_file"]).suffix.lower() in {".jpg", ".jpeg"}:
            row["quality_flags"].append("jpeg_source")
        if row["raw_silhouette_iou"] < 0.85:
            row["quality_flags"].append("silhouette_differs_from_raw")
        row["label_to_bc_palette_index"] = {str(k): v for k,v in sorted(mapping.items())}
        color_to_ids = {}
        for label_id, palette_index in sorted(mapping.items()):
            color_to_ids.setdefault(str(palette_index), []).append(label_id)
        row["color_to_ids"] = color_to_ids
        row["color_reuse"] = any(len(ids)>1 for ids in color_to_ids.values())
        row["foreground_pixels_by_label"] = {str(k): int(np.sum(labels==k)) for k in sorted(mapping)}
        row["target_rgb_by_palette_index"] = {str(k): list(target_colors[k-1]) for k in sorted(set(mapping.values()))}
        row["guide_file"] = f"guidance/D_human_fixed_bc20/thin_fine_grained/{row['uid']}.png"
        row["labels_file"] = f"labels/thin_fine_grained/{row['uid']}/labels_uint16.png"
        row["mapping_file"] = f"labels/thin_fine_grained/{row['uid']}/mapping.json"
        for key in ["guide_file", "labels_file"]:
            path = output/row[key]
            if path.exists():
                raise FileExistsError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(labels.astype(np.uint16)).save(output/row["labels_file"])
        Image.fromarray(guide).save(output/row["guide_file"])
        row["guide_sha256"], row["labels_sha256"] = sha(output/row["guide_file"]), sha(output/row["labels_file"])
        row["provenance"] = receipt["provenance"]
        (output/row["mapping_file"]).write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        receipt["candidate_count"] += 1
        receipt["per_uid"].append(row)
        print(f"{index}/{len(rows)} {row['uid'][:8]} candidateIDs={len(mapping)} IoU={row['raw_silhouette_iou']:.3f}",flush=True)
    receipt["color_reuse_count"] = sum(bool(r.get("color_reuse")) for r in receipt["per_uid"])
    (output/"source_manifest.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output/"normalization_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (output/"review_list.csv").open("w",encoding="utf-8-sig",newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["uid","source_group","status","candidate_id_count","raw_silhouette_iou","quality_flags"])
        for row in receipt["per_uid"]:
            writer.writerow([row["uid"],row.get("source_group"),row["status"],row.get("label_count"),row.get("raw_silhouette_iou"),"|".join(row["quality_flags"])])
    make_review(output, receipt)
    return receipt


def make_review(output, receipt):
    output = Path(output)
    review = output/"review"
    review.mkdir(exist_ok=True)
    rows = receipt["per_uid"]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",16)
    for page,start in enumerate(range(0,len(rows),10),1):
        sheet = Image.new("RGB", (1500,1600), "white")
        draw = ImageDraw.Draw(sheet)
        for slot,row in enumerate(rows[start:start+10]):
            x,y=(slot%2)*750,(slot//2)*320
            summary = "PENDING" if "label_count" not in row else f"IDs={row['label_count']} IoU={row['raw_silhouette_iou']:.3f}"
            draw.text((x+8,y+5),f"{start+slot+1:02d} {row['uid'][:16]}  {summary}",fill="black",font=font)
            for j,(key,caption) in enumerate([("raw_file","RAW"),("human_file","D SOURCE"),("guide_file","BC20 CANDIDATE")]):
                draw.text((x+8+j*250,y+28),caption,fill="black",font=font)
                if row.get(key):
                    with Image.open(output/row[key]) as image:
                        sheet.paste(thumb(image,(240,240)),(x+5+j*250,y+52))
                else:
                    draw.text((x+10+j*250,y+150),"MISSING / PENDING",fill="red",font=font)
            flags=[flag for flag in row["quality_flags"] if flag not in {"jpeg_source","inferred_hue_groups_not_verified_parts"}]
            draw.text((x+8,y+297),", ".join(flags)[:80],fill="#9f2b1f",font=font)
        sheet.save(review/f"comparison_{page:02d}.jpg",quality=95)
    cards=[]
    priority=[]
    for number,row in enumerate(rows,1):
        figures=[]
        for key,caption in [("raw_file","RAW 输入"),("human_file","来源人工彩色图"),("guide_file","B/C 20 色候选")]:
            if row.get(key):
                href="../"+quote(row[key],safe="/")
                figures.append(f'<figure><a href="{href}"><img loading="lazy" src="{href}"></a><figcaption>{caption}</figcaption></figure>')
            else:
                figures.append(f'<figure class="pending"><p>缺少人工标注，保留待补</p><figcaption>{caption}</figcaption></figure>')
        count=row.get("label_count","未生成")
        score=f"{row['raw_silhouette_iou']:.3f}" if "raw_silhouette_iou" in row else "无"
        links="".join(f'<a href="../{quote(row[key],safe="/")}">{caption}</a> ' for key,caption in [("labels_file","uint16 标签"),("mapping_file","映射与来源记录")] if row.get(key))
        extra=""
        if row.get("duplicate_resolution"):
            extra='<p class="alert">重复来源：已按与 RAW 的形状对应关系选用 thin6 PNG；thin5 JPG 为来源标注不匹配，保留原文件，未关联给缺失 UID。</p>'
            extra += "".join(f'<a href="../sources/{quote(alt["file"],safe="/")}">查看未选用的 JPG 来源</a>' for alt in row["duplicate_resolution"]["alternatives"])
        card=f'<article id="{row["uid"]}"><h2>{number:02d} · {row["uid"]}</h2><p>状态：{row["status"]} · 候选颜色组 ID：{count} · RAW 轮廓 IoU：{score}</p><p class="flags">{html.escape(" · ".join(row["quality_flags"]))}</p><div class="images">{"".join(figures)}</div>{links}{extra}</article>'
        cards.append(card)
        if any(flag not in {"jpeg_source","inferred_hue_groups_not_verified_parts"} for flag in row["quality_flags"]):
            priority.append(card)
    head='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>thin456 人工引导图核对</title><style>body{font:16px system-ui;background:#f4f5f7;color:#18212b;margin:28px}h1{font-size:27px}h2{font-size:18px;overflow-wrap:anywhere}article{background:white;padding:20px;margin:22px 0;border-radius:12px}.images{display:flex;gap:16px}figure{margin:0;width:33%}img{width:100%;background:repeating-conic-gradient(#e4e4e4 0% 25%,#fafafa 0% 50%) 0/24px 24px}figcaption{padding:8px}p{line-height:1.6}.flags,.alert{color:#923a19;font-size:14px}.pending{background:#faf3e9;min-height:200px;display:grid;place-items:center}a{color:#175da2;margin-right:14px}@media(max-width:700px){.images{flex-direction:column}figure{width:100%}}</style>'''
    intro=f'<h1>thin456：{receipt["inventory_count"]} 个来源 / {receipt["candidate_count"]} 张颜色规范化候选 / {receipt["pending_count"]} 张待补</h1><p>所有结果均需人工复核，不作为新真值。颜色组和灰色区域的 ID 不代表已验证的真实部件；色彩阴影、压缩和细碎边缘可能产生候选分组误差。标签保存独立整数 ID，允许相隔足够远的不同 ID 在必要时复用显示颜色。</p><p>保持来源图的 1024×1024 坐标；未旋转、镜像、缩放、对齐或按 RAW 裁剪。RAW 有透明背景时使用 alpha≥128 保留包括白色实体在内的前景；仅对完全不透明的 RAW 排除 RGB≥245、通道差≤15 的近白背景作为诊断回退。IoU 不代表分割准确率。</p><nav><a href="index.html">全部核对</a><a href="priority.html">优先核对</a><a href="../normalization_receipt.json">完整处理记录</a><a href="../source_pairing_audit.json">来源配对审计</a><a href="../README.md">使用说明</a></nav>'
    sheets='<p>'+''.join(f'<a href="comparison_{p:02d}.jpg">三图总览 {p}</a>' for p in range(1,(len(rows)+9)//10+1))+'</p>'
    (review/"index.html").write_text(head+intro+sheets+"".join(cards)+"</html>",encoding="utf-8")
    (review/"priority.html").write_text(head+intro+f'<h2>优先核对 {len(priority)} 项</h2>'+"".join(priority)+"</html>",encoding="utf-8")
    (output/"index.html").write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=review/index.html"><a href="review/index.html">打开 thin456 核对页</a>',encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"downloads/D_thin456_bc20_20261023/sources")
    parser.add_argument("--output",type=Path,default=ROOT/"downloads/D_thin456_bc20_20261023")
    parser.add_argument("--source-only",action="store_true")
    args=parser.parse_args()
    if args.source_only:
        make_source_sheets(args.source,inventory_sources(args.source,PREFERENCES),args.output/"review")
        return
    source_palette_path=ROOT/"downloads/part_palette_30_high_contrast_20261021.json"
    bc_palette_path=ROOT/"downloads/colors_pure_bc_20261022.json"
    source_palette={int(entry["label_id"]):tuple(entry["rgb"]) for entry in json.loads(source_palette_path.read_text(encoding="utf-8"))["colors"]}
    target_colors=[tuple(entry[:3]) for entry in json.loads(bc_palette_path.read_text(encoding="utf-8"))[1:]]
    audit_path=args.output/"source_pairing_audit.json"
    audit=json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    provenance={"source_archive_sha256":audit.get("archive_sha256"),
                "source_palette_sha256":sha(source_palette_path),"bc_palette_sha256":sha(bc_palette_path),
                "processing_script_sha256":sha(Path(__file__)),
                "inference_script_sha256":sha(ROOT/"scripts/repair_d_guidance_images.py"),
                "recolor_script_sha256":sha(ROOT/"scripts/recolor_d_guidance_to_bc_palette.py")}
    receipt=process_directory(args.source,args.output,source_palette,target_colors,PREFERENCES,provenance=provenance)
    shutil.copyfile(source_palette_path,args.output/"source_palette_30.json")
    shutil.copyfile(bc_palette_path,args.output/"colors_pure_bc.json")
    print(json.dumps({key:value for key,value in receipt.items() if key!="per_uid"},ensure_ascii=False))


if __name__=="__main__":
    main()

"""Encode analyst-reviewed source colors without adding hue-fragment IDs."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from process_thin456_guidance import ROOT, STATUS, make_review, make_source_sheets, sha
from repair_d_guidance_images import foreground_mask, nearest_rgb
from recolor_d_guidance_to_bc_palette import recolor_labels

REVIEWED_PREFIX_COLORS = {
    "a225264a": [1,2,3],
    "a31191e5": [1,2,3,4,5,6],
    "a753edea": [1,2,3,4,5,6,7,8,9],
    "a80f3639": [1,2],
    "a8677d9c": [1,2,3],
    "a9861b75": [1,2,3,4],
    "aa1ccbb0": [1,2],
    "ab858477": [1,2],
    "adf93b14": [1,2,4,5],
    "b0015e78": [1,2,3,4,5,6,8,9,10,12,13,14],
    "b0a2d8c6": [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    "b2595bc8": [1,2,3],
    "b27295e0": [1,2,3,5],
    "b2ccc7bc": [1,2,3,4,5,6,7],
    "b55856fd": [1,2,3,4,5,6,7,8,9],
    "b5a4ae23": [1,2,3,4,5,6,7,8,9,10,11,12,13],
    "b72b8828": [1,2,3,4,5],
    "b737399e": [1,2,3,4,5,6,7,8,9],
    "b9ff2379": [1,2],
    "bbd7420c": [1,2,3,4],
    "bc0ce5dc": [1,2,3,4,5],
    "bdcd141c": [1,2,3],
    "bde4b690": [1,2,3,4,5,6,7,8,9,10,31],
    "bf1367c6": [1,2,3,4,5,6,7,8,9],
    "bf689774": [1,2],
    "c2cf3c69": [1,4,5],
    "c3dbcefe": [1,2,3,4,5,6,7],
}
UNRESOLVED_PREFIXES = {"ad2ce94a", "afbf2e77"}


def encode_reviewed_colors(rgba, source_palette, target_colors, active_ids):
    """Nearest RGB only among visually approved colors; retain original foreground."""
    if not active_ids or len(set(active_ids)) != len(active_ids):
        raise ValueError("reviewed source IDs must be nonempty and unique")
    colors = dict(source_palette)
    colors[31] = (35,35,35)  # Explicit analyst-reviewed painted-black extension.
    if any(label_id not in colors for label_id in active_ids):
        raise ValueError("reviewed source ID has no canonical RGB center")
    mask = foreground_mask(rgba)
    if not mask.any():
        raise ValueError("empty foreground after unchanged source cleanup")
    ids = sorted(active_ids)
    nearest, distances = nearest_rgb(rgba[...,:3][mask], [colors[k] for k in ids])
    labels = np.zeros(mask.shape,dtype=np.uint16)
    labels[mask] = np.asarray(ids,dtype=np.uint16)[nearest]
    present = sorted(int(k) for k in np.unique(labels) if k)
    selected_colors = {k: colors[k] for k in present}
    guide, mapping = recolor_labels(labels,selected_colors,target_colors)
    info = {
        "status": STATUS, "method": "reviewed_source_palette_nearest_rgb",
        "encoding_review_category": "reviewed_source_color_encoding_candidate",
        "ground_truth_ready": False, "evaluable": False,
        "evaluable_reason": "color_encoding_review_does_not_verify_semantic_parts_or_raw_geometry",
        "active_source_palette_indices": ids,
        "unused_reviewed_source_ids": sorted(set(ids)-set(present)),
        "source_rgb_by_candidate_id": {str(k):list(v) for k,v in selected_colors.items()},
        "source_exact_palette_fraction": float(np.mean(distances==0)),
        "source_foreground_pixels": int(mask.sum()), "alpha_threshold": 128,
        "removed_bright_neutral_pixels": int(np.sum((rgba[...,3]>=128)&~mask)),
        "uncertain_fraction": float(np.mean(distances>50)),
        "classification_distance_rgb": {
            "metric": "euclidean_rgb_uint8_to_nearest_allowed_source_center",
            "mean": float(np.mean(distances)), "median": float(np.median(distances)),
            "p95": float(np.percentile(distances,95)), "maximum": float(np.max(distances)),
            "uncertainty_distance_threshold": 50,
            "pixels_above_threshold": int(np.sum(distances>50)),
            "fraction_above_threshold": float(np.mean(distances>50)),
            "interpretation": "encoding_distance_diagnostic_not_semantic_confidence",
        },
        "candidate_region_meaning": "visually_reviewed_source_color_id_not_verified_semantic_part",
        "palette_selection_basis": "analyst_inspected_source_images_and_supplied_per_uid_active_source_indices",
        "geometric_transform": "none", "raw_clipping": False,
        "encoding_quality_flags": ["source_colors_reviewed_semantics_unverified"],
    }
    if 31 in ids:
        info["source_palette_extension"] = {"31": [35,35,35]}
        info["encoding_quality_flags"].append("neutral_region_semantics_unconfirmed")
    return labels,guide,mapping,info


def process_from_v1(v1, output, source_palette, target_colors, reviewed_colors,
                    unresolved_uids, expected_size=(1024,1024), provenance=None):
    """Create an independent revision; leave every v1 byte in place."""
    v1,output=Path(v1),Path(output)
    if output.exists():
        raise FileExistsError(output)
    old_receipt_path=v1/"normalization_receipt.json"
    old_receipt=json.loads(old_receipt_path.read_text(encoding="utf-8"))
    active_uids={row["uid"] for row in old_receipt["per_uid"] if row.get("guide_file")}
    if active_uids != set(reviewed_colors)|set(unresolved_uids) or set(reviewed_colors)&set(unresolved_uids):
        raise ValueError("every existing candidate must be exactly reviewed or unresolved")
    output.mkdir(parents=True)
    shutil.copytree(v1/"sources",output/"sources")
    for name in ["source_manifest.json","source_pairing_audit.json","source_palette_30.json","colors_pure_bc.json"]:
        if (v1/name).exists():
            shutil.copyfile(v1/name,output/name)
    manifest=json.loads((v1/"source_manifest.json").read_text(encoding="utf-8"))
    make_source_sheets(output/"sources",manifest,output/"review")
    receipt={key:copy.deepcopy(value) for key,value in old_receipt.items() if key!="per_uid"}
    receipt.update(revision="v2_reviewed_source_color_encoding", ground_truth_ready=False,
                   reviewed_color_encoding_count=0,unresolved_candidate_count=0,
                   provenance=dict(provenance or {},v1_receipt_sha256=sha(old_receipt_path)),
                   per_uid=[])
    choices=[]
    for number,old in enumerate(old_receipt["per_uid"],1):
        row=copy.deepcopy(old)
        uid=row["uid"]
        row["ground_truth_ready"]=False
        row["evaluable"]=False
        row["provenance"]=receipt["provenance"]
        if not row.get("guide_file"):
            row["encoding_review_category"]="pending_missing_human_annotation"
            receipt["per_uid"].append(row)
            continue
        for key in ["guide_file","labels_file","mapping_file"]:
            (output/row[key]).parent.mkdir(parents=True,exist_ok=True)
        with Image.open(v1/row["labels_file"]) as image:
            old_labels=np.asarray(image)
        if uid in unresolved_uids:
            row["encoding_review_category"]="unresolved_source_color_candidate"
            row["evaluable_reason"]="color_schema_unresolved"
            row["quality_flags"].append("color_schema_unresolved")
            row["v1_candidate_preserved_byte_identically"]=True
            for key in ["guide_file","labels_file"]:
                shutil.copyfile(v1/row[key],output/row[key])
            receipt["unresolved_candidate_count"]+=1
        else:
            with Image.open(output/row["human_file"]) as image:
                if image.size!=tuple(expected_size):
                    raise ValueError(f"source size differs from approved geometry: {uid}")
                rgba=np.asarray(image.convert("RGBA"))
            labels,guide,mapping,info=encode_reviewed_colors(rgba,source_palette,target_colors,reviewed_colors[uid])
            if labels.shape!=old_labels.shape or not np.array_equal(labels>0,old_labels>0):
                raise AssertionError(f"v1 foreground geometry changed: {uid}")
            for key in ["hue_peak_degrees","neutral_region","stable_interior_color_seeds"]:
                row.pop(key,None)
            row.update(info)
            row["quality_flags"]=[flag for flag in row["quality_flags"] if flag not in {
                "inferred_hue_groups_not_verified_parts","color_ambiguity","neutral_region_semantics_unconfirmed"}]
            row["quality_flags"].extend(info["encoding_quality_flags"])
            if info["uncertain_fraction"]>0.02:
                row["quality_flags"].append("color_classification_distance_requires_review")
            row["label_count"]=len(mapping)
            row["label_to_bc_palette_index"]={str(k):v for k,v in sorted(mapping.items())}
            row["color_to_ids"]={}
            for label_id,palette_index in sorted(mapping.items()):
                row["color_to_ids"].setdefault(str(palette_index),[]).append(label_id)
            row["color_reuse"]=any(len(ids)>1 for ids in row["color_to_ids"].values())
            row["foreground_pixels_by_label"]={str(k):int(np.sum(labels==k)) for k in sorted(mapping)}
            row["target_rgb_by_palette_index"]={str(k):list(target_colors[k-1]) for k in sorted(set(mapping.values()))}
            row["v1_foreground_preserved_exactly"]=True
            row["v1_candidate_label_count"]=old["label_count"]
            row["guide_foreground_pixels"]=int(np.sum(guide[...,3]==255))
            Image.fromarray(labels).save(output/row["labels_file"])
            Image.fromarray(guide).save(output/row["guide_file"])
            receipt["reviewed_color_encoding_count"]+=1
            choices.append({"uid":uid,"active_source_palette_indices":reviewed_colors[uid],
                            "basis":info["palette_selection_basis"],"source_palette_extension":info.get("source_palette_extension",{})})
        row["guide_sha256"],row["labels_sha256"]=sha(output/row["guide_file"]),sha(output/row["labels_file"])
        (output/row["mapping_file"]).write_text(json.dumps(row,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        receipt["per_uid"].append(row)
        print(f"{number}/{len(old_receipt['per_uid'])} {uid[:8]} IDs={row['label_count']} {row['encoding_review_category']}",flush=True)
    receipt["color_reuse_count"]=sum(bool(row.get("color_reuse")) for row in receipt["per_uid"])
    (output/"normalization_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output/"reviewed_color_choices.json").write_text(json.dumps(choices,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (output/"review_list.csv").open("w",encoding="utf-8-sig",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["uid","encoding_review_category","source_color_ids","candidate_id_count","ground_truth_ready","evaluable","raw_silhouette_iou","quality_flags"])
        for row in receipt["per_uid"]:
            writer.writerow([row["uid"],row["encoding_review_category"],row.get("active_source_palette_indices",[]),row.get("label_count"),False,False,row.get("raw_silhouette_iou"),"|".join(row["quality_flags"])])
    make_review(output,receipt)
    summary=(f"{receipt['reviewed_color_encoding_count']} 张已核对来源颜色编码 / "
             f"{receipt['unresolved_candidate_count']} 张颜色定义未解决候选 / {receipt['pending_count']} 张待补")
    explanation=(f'<aside><h2>v2：{summary}</h2><p>逐图核对来源颜色后，27 张仅在各自允许的来源色板中心间按 RGB 最近邻编码，ID 使用来源色板编号；不自动增加混色色相或连通分量。bde4b690 的明确黑色涂色另用来源 ID 31，RGB=(35,35,35)。ad2ce94a、afbf2e77 保留 v1 候选且 evaluable=false。所有图 ground_truth_ready=false，来源颜色核对不等于真实部件或几何验收。</p><a href="../reviewed_color_choices.json">逐图来源颜色选择</a></aside>')
    for name in ["index.html","priority.html"]:
        path=output/"review"/name
        document=path.read_text(encoding="utf-8")
        document=document.replace("<nav>",explanation+"<nav>",1)
        path.write_text(document,encoding="utf-8")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1",type=Path,default=ROOT/"downloads/D_thin456_bc20_20261023")
    parser.add_argument("--output",type=Path,default=ROOT/"downloads/D_thin456_bc20_20261023_v2")
    args=parser.parse_args()
    old=json.loads((args.v1/"normalization_receipt.json").read_text(encoding="utf-8"))
    reviewed={row["uid"]:REVIEWED_PREFIX_COLORS[row["uid"][:8]] for row in old["per_uid"] if row["uid"][:8] in REVIEWED_PREFIX_COLORS}
    unresolved={row["uid"] for row in old["per_uid"] if row["uid"][:8] in UNRESOLVED_PREFIXES}
    if len(reviewed)!=27 or len(unresolved)!=2:
        raise ValueError("reviewed source inventory does not match approved 27 + 2 choices")
    source_path=ROOT/"downloads/part_palette_30_high_contrast_20261021.json"
    bc_path=ROOT/"downloads/colors_pure_bc_20261022.json"
    source={int(x["label_id"]):tuple(x["rgb"]) for x in json.loads(source_path.read_text())["colors"]}
    target=[tuple(x[:3]) for x in json.loads(bc_path.read_text())[1:]]
    provenance=dict(old["provenance"],processing_script_sha256=sha(Path(__file__)),
                    processing_script="scripts/process_thin456_guidance_v2.py",
                    v1_processing_script_sha256=old["provenance"]["processing_script_sha256"],
                    source_palette_sha256=sha(source_path),bc_palette_sha256=sha(bc_path))
    receipt=process_from_v1(args.v1,args.output,source,target,reviewed,unresolved,provenance=provenance)
    print(json.dumps({k:v for k,v in receipt.items() if k!="per_uid"},ensure_ascii=False))


if __name__=="__main__":
    main()

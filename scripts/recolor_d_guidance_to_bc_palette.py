"""Recolor D-arm integer labels with B/C colors without changing part IDs.

The B/C JSON's index 0 is reserved by its producers, leaving indices 1..N-1
for foreground. If parts outnumber those colors, only spatially separated
parts may share a color. Such RGB guides are not invertible label maps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageFilter


UID_RE = re.compile(r"[0-9a-f]{32}", re.IGNORECASE)


def recolor_labels(
    labels: np.ndarray,
    source_rgb_by_id: dict[int, tuple[int, int, int]],
    target_rgb: list[tuple[int, int, int]],
    min_gap_px: int = 5,
) -> tuple[np.ndarray, dict[int, int]]:
    """Return RGBA guide and old-label → B/C palette-index mapping."""
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a two-dimensional integer map")
    if min_gap_px < 0:
        raise ValueError("min_gap_px must be nonnegative")
    if not target_rgb or len(set(target_rgb)) != len(target_rgb):
        raise ValueError("foreground target colors must be nonempty and unique")
    if any(len(rgb) != 3 or any(not 0 <= int(channel) <= 255 for channel in rgb) for rgb in target_rgb):
        raise ValueError("target colors must be RGB uint8 triples")
    ids, counts = np.unique(labels, return_counts=True)
    parts = {int(label_id): int(count) for label_id, count in zip(ids, counts) if label_id != 0}
    if not parts or any(label_id not in source_rgb_by_id for label_id in parts):
        raise ValueError("non-background labels need a source palette color")
    ordered = sorted(parts, key=lambda label_id: (-parts[label_id], label_id))
    mapping: dict[int, int] = {}
    color_groups: dict[int, set[int]] = {index: set() for index in range(1, len(target_rgb) + 1)}

    for label_id in ordered:
        if len(mapping) < len(target_rgb):
            candidates = [index for index, group in color_groups.items() if not group]
        else:
            mask = ((labels == label_id) * 255).astype(np.uint8)
            near = np.asarray(
                Image.fromarray(mask).filter(ImageFilter.MaxFilter(2 * min_gap_px + 1))
            ) > 0
            near_ids = {int(x) for x in np.unique(labels[near]) if x != 0 and x != label_id}
            candidates = [
                index for index, group in color_groups.items() if not group.intersection(near_ids)
            ]
            if not candidates:
                raise ValueError(f"label {label_id}: no spatially separated color available")

        source_color = source_rgb_by_id[label_id]
        index = min(
            candidates,
            key=lambda candidate: (
                sum(
                    (int(source_color[channel]) - int(target_rgb[candidate - 1][channel])) ** 2
                    for channel in range(3)
                ),
                candidate,
            ),
        )
        mapping[label_id] = index
        color_groups[index].add(label_id)

    lookup = np.zeros((int(labels.max()) + 1, 4), dtype=np.uint8)
    for label_id, index in mapping.items():
        lookup[label_id, :3] = target_rgb[index - 1]
        lookup[label_id, 3] = 255
    return lookup[labels], mapping


def recolor_directory(
    source_root: Path,
    source_palette_path: Path,
    bc_palette_path: Path,
    output_root: Path,
    group: str = "assemblies",
) -> dict:
    source_root = Path(source_root)
    source_palette_path = Path(source_palette_path)
    bc_palette_path = Path(bc_palette_path)
    output_root = Path(output_root)
    if group not in {"assemblies", "solid_volume", "thin_fine_grained"}:
        raise ValueError(f"unsupported group: {group}")
    source_document = json.loads(source_palette_path.read_text(encoding="utf-8"))
    source_colors = {
        int(entry["label_id"]): tuple(entry["rgb"]) for entry in source_document["colors"]
    }
    bc_entries = json.loads(bc_palette_path.read_text(encoding="utf-8"))
    if len(bc_entries) < 2:
        raise ValueError("B/C palette needs reserved index 0 plus foreground colors")
    target_colors = [tuple(entry[:3]) for entry in bc_entries[1:]]
    if any(len(entry) not in (3, 4) for entry in bc_entries):
        raise ValueError("B/C palette entries must be RGB or RGBA rows")
    label_files = sorted((source_root / "references" / group).glob("*/labels_uint16.png"))
    if not label_files:
        raise ValueError(f"no {group} labels_uint16.png files found")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "recolored_guidance_not_new_ground_truth",
        "source_label_root": str(source_root.resolve()),
        "source_palette_sha256": hashlib.sha256(source_palette_path.read_bytes()).hexdigest(),
        "bc_palette_sha256": hashlib.sha256(bc_palette_path.read_bytes()).hexdigest(),
        "group": group,
        "bc_palette_entries": len(bc_entries),
        "usable_foreground_colors": len(target_colors),
        "min_gap_px_for_color_reuse": 5,
        "recolored_count": 0,
        "non_injective_count": 0,
        "per_uid": {},
    }
    for label_path in label_files:
        uid = label_path.parent.name.lower()
        if not UID_RE.fullmatch(uid):
            raise ValueError(f"invalid UID: {uid}")
        with Image.open(label_path) as image:
            labels = np.asarray(image)
        rgba, mapping = recolor_labels(labels, source_colors, target_colors)
        guide_path = output_root / "guidance" / "D_human_fixed_bc20" / group / f"{uid}.png"
        guide_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba).save(guide_path)
        reference_path = output_root / "references" / group / uid / "labels_uint16.png"
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(label_path, reference_path)
        color_to_ids: dict[int, list[int]] = {}
        for label_id, index in mapping.items():
            color_to_ids.setdefault(index, []).append(label_id)
        reused = {str(index): ids for index, ids in color_to_ids.items() if len(ids) > 1}
        receipt["per_uid"][uid] = {
            "source_labels_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            "copied_labels_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "guide_sha256": hashlib.sha256(guide_path.read_bytes()).hexdigest(),
            "part_count": len(mapping),
            "color_reuse": bool(reused),
            "reused_colors_to_label_ids": reused,
            "label_to_bc_palette_index": {str(label_id): index for label_id, index in sorted(mapping.items())},
        }
        receipt["recolored_count"] += 1
        receipt["non_injective_count"] += bool(reused)
    (output_root / "recolor_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-label-root", required=True, type=Path)
    parser.add_argument("--source-palette", required=True, type=Path)
    parser.add_argument("--bc-palette", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group", default="assemblies", choices=["assemblies", "solid_volume", "thin_fine_grained"])
    args = parser.parse_args()
    receipt = recolor_directory(
        args.source_label_root, args.source_palette, args.bc_palette, args.output, group=args.group
    )
    print(
        f"recolored={receipt['recolored_count']} "
        f"color_reuse={receipt['non_injective_count']}"
    )


if __name__ == "__main__":
    main()

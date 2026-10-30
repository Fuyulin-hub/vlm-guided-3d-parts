"""Losslessly decode discrete D-arm assemblies guidance into uint16 part IDs.

This does not infer semantic part names or repair a human segmentation. The
provided palette's IDs are used verbatim; fully transparent pixels become 0.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

import numpy as np
from PIL import Image


UID_RE = re.compile(r"[0-9a-f]{32}\.png", re.IGNORECASE)
PAIRED_NAME_RE = re.compile(r"(?P<uid>[0-9a-f]{32})(?P<human>_hum)?\.(?P<ext>png|jpg|jpeg)", re.IGNORECASE)


def rgba_to_labels(rgba: np.ndarray, palette: dict[tuple[int, int, int], int]) -> np.ndarray:
    """Map exact opaque palette RGBs to IDs; reject ambiguous pixel values."""
    if rgba.ndim != 3 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
        raise ValueError("source must be an 8-bit RGBA image")
    alpha = rgba[..., 3]
    if np.any((alpha != 0) & (alpha != 255)):
        raise ValueError("partial alpha is not a discrete label")
    opaque = alpha == 255
    rgb = rgba[..., :3].astype(np.uint32)
    packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    color_ids = sorted(
        ((r << 16) | (g << 8) | b, label_id)
        for (r, g, b), label_id in palette.items()
    )
    if not color_ids:
        raise ValueError("palette is empty")
    keys = np.asarray([key for key, _ in color_ids], dtype=np.uint32)
    ids = np.asarray([label_id for _, label_id in color_ids], dtype=np.uint16)
    opaque_colors = packed[opaque]
    index = np.searchsorted(keys, opaque_colors)
    valid = (index < len(keys)) & (keys[np.minimum(index, len(keys) - 1)] == opaque_colors)
    if not np.all(valid):
        raise ValueError(f"{int((~valid).sum())} off-palette opaque pixel(s)")
    labels = np.zeros(alpha.shape, dtype=np.uint16)
    labels[opaque] = ids[index]
    return labels


def validate_archive_entries(archive_path: Path) -> dict[str, str]:
    """Return UID-to-archive-entry mapping without trusting ZIP paths."""
    entries: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive path: {info.filename}")
            if not UID_RE.fullmatch(path.name):
                raise ValueError(f"unexpected archive file: {info.filename}")
            uid = path.stem.lower()
            if uid in entries:
                raise ValueError(f"duplicate UID in archive: {uid}")
            entries[uid] = info.filename
    if not entries:
        raise ValueError("archive contains no UID PNGs")
    return entries


def validate_paired_archives(archive_paths: list[Path]) -> dict[str, tuple[Path, str, str]]:
    """Locate complete raw/human pairs, rejecting ambiguous or unsafe ZIP entries."""
    pairs: dict[str, tuple[Path, str, str]] = {}
    for archive_path in map(Path, archive_paths):
        within: dict[str, dict[str, str]] = {}
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                path = PurePosixPath(info.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"unsafe archive path: {info.filename}")
                match = PAIRED_NAME_RE.fullmatch(path.name)
                if not match or (not match.group("human") and match.group("ext").lower() != "png"):
                    raise ValueError(f"unexpected archive file: {info.filename}")
                uid = match.group("uid").lower()
                role = "human" if match.group("human") else "raw"
                if uid in pairs or role in within.setdefault(uid, {}):
                    raise ValueError(f"duplicate UID or paired role: {uid}")
                within[uid][role] = info.filename
        for uid, roles in within.items():
            if set(roles) != {"raw", "human"}:
                raise ValueError(f"incomplete raw/human pair: {uid}")
            pairs[uid] = archive_path, roles["raw"], roles["human"]
    if not pairs:
        raise ValueError("archives contain no UID pairs")
    return pairs


def load_palette(palette_path: Path) -> dict[tuple[int, int, int], int]:
    document = json.loads(palette_path.read_text(encoding="utf-8"))
    palette: dict[tuple[int, int, int], int] = {}
    seen_ids: set[int] = set()
    for entry in document["colors"]:
        color = tuple(entry["rgb"])
        label_id = int(entry["label_id"])
        if len(color) != 3 or any(not isinstance(x, int) or not 0 <= x <= 255 for x in color):
            raise ValueError("palette RGB must contain three uint8 values")
        if not 1 <= label_id < 65535 or color in palette or label_id in seen_ids:
            raise ValueError("palette colors and non-background IDs must be unique")
        palette[color] = label_id
        seen_ids.add(label_id)
    if not palette:
        raise ValueError("palette is empty")
    return palette


def convert_archive(
    archive_path: Path, palette_path: Path, output_root: Path, group: str = "assemblies"
) -> dict:
    archive_path = Path(archive_path)
    palette_path = Path(palette_path)
    output_root = Path(output_root)
    if group not in {"assemblies", "solid_volume", "thin_fine_grained"}:
        raise ValueError(f"unsupported group: {group}")
    entries = validate_archive_entries(archive_path)
    palette = load_palette(palette_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "exact_palette_decoding_only_not_human_verified",
        "source_archive": str(archive_path.resolve()),
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "palette_file": str(palette_path.resolve()),
        "palette_sha256": hashlib.sha256(palette_path.read_bytes()).hexdigest(),
        "group": group,
        "converted_count": 0,
        "per_uid": {},
    }
    with zipfile.ZipFile(archive_path) as archive:
        for uid, entry_name in sorted(entries.items()):
            source_bytes = archive.read(entry_name)
            with Image.open(io.BytesIO(source_bytes)) as source_image:
                if source_image.mode != "RGBA":
                    raise ValueError(f"{uid}: expected RGBA, got {source_image.mode}")
                rgba = np.asarray(source_image)
            try:
                labels = rgba_to_labels(rgba, palette)
            except ValueError as exc:
                raise ValueError(f"{uid}: {exc}") from exc
            label_ids, counts = np.unique(labels, return_counts=True)
            color_counts = {int(label_id): int(count) for label_id, count in zip(label_ids, counts)}
            uid_dir = output_root / "references" / group / uid
            uid_dir.mkdir(parents=True, exist_ok=False)
            label_path = uid_dir / "labels_uint16.png"
            Image.fromarray(labels).save(label_path, format="PNG")
            (uid_dir / "preview_color.png").write_bytes(source_bytes)
            receipt["per_uid"][uid] = {
                "source_png_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "labels_png_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                "width": int(labels.shape[1]),
                "height": int(labels.shape[0]),
                "label_ids": [int(x) for x in label_ids if x != 0],
                "foreground_pixels": int(labels.size - color_counts.get(0, 0)),
                "pixels_per_label": {str(k): v for k, v in color_counts.items() if k != 0},
            }
            receipt["converted_count"] += 1
    (output_root / "conversion_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def convert_paired_archives(
    archive_paths: list[Path],
    palette_path: Path,
    output_root: Path,
    group: str,
    required_size: tuple[int, int] = (1024, 1024),
) -> dict:
    """Decode only losslessly valid `_hum` images; record every rejected UID."""
    if group not in {"assemblies", "solid_volume", "thin_fine_grained"}:
        raise ValueError(f"unsupported group: {group}")
    archive_paths = [Path(path) for path in archive_paths]
    palette_path = Path(palette_path)
    output_root = Path(output_root)
    pairs = validate_paired_archives(archive_paths)
    palette = load_palette(palette_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "exact_palette_decoding_only_not_human_verified",
        "group": group,
        "required_size": list(required_size),
        "source_archives": [
            {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in archive_paths
        ],
        "palette_sha256": hashlib.sha256(palette_path.read_bytes()).hexdigest(),
        "paired_uid_count": len(pairs),
        "converted_count": 0,
        "rejected_count": 0,
        "per_uid": {},
        "rejected": {},
    }
    for uid, (archive_path, raw_entry, human_entry) in sorted(pairs.items()):
        with zipfile.ZipFile(archive_path) as archive:
            human_bytes = archive.read(human_entry)
        source = {
            "archive": str(archive_path.resolve()),
            "raw_entry": raw_entry,
            "human_entry": human_entry,
            "human_sha256": hashlib.sha256(human_bytes).hexdigest(),
        }
        try:
            with Image.open(io.BytesIO(human_bytes)) as image:
                if image.mode != "RGBA":
                    raise ValueError(f"expected RGBA, got {image.mode}")
                if image.size != required_size:
                    raise ValueError(f"expected size {required_size}, got {image.size}")
                rgba = np.asarray(image)
            labels = rgba_to_labels(rgba, palette)
            if not np.any(labels):
                raise ValueError("empty foreground")
        except (OSError, ValueError) as exc:
            receipt["rejected"][uid] = {**source, "reason": str(exc)}
            receipt["rejected_count"] += 1
            continue
        uid_dir = output_root / "references" / group / uid
        uid_dir.mkdir(parents=True, exist_ok=False)
        label_path = uid_dir / "labels_uint16.png"
        Image.fromarray(labels).save(label_path, format="PNG")
        (uid_dir / "preview_color.png").write_bytes(human_bytes)
        label_ids, counts = np.unique(labels, return_counts=True)
        receipt["per_uid"][uid] = {
            **source,
            "labels_png_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
            "label_ids": [int(label_id) for label_id in label_ids if label_id != 0],
            "foreground_pixels": int(np.count_nonzero(labels)),
            "pixels_per_label": {
                str(int(label_id)): int(count)
                for label_id, count in zip(label_ids, counts) if label_id != 0
            },
        }
        receipt["converted_count"] += 1
    (output_root / "conversion_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, nargs="+")
    parser.add_argument("--palette", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--group", default="assemblies", choices=["assemblies", "solid_volume", "thin_fine_grained"])
    parser.add_argument("--paired-human", action="store_true", help="use UID_hum images from raw/human pairs and skip non-lossless cases")
    args = parser.parse_args()
    if args.paired_human:
        receipt = convert_paired_archives(args.archive, args.palette, args.output, group=args.group)
        print(f"Converted {receipt['converted_count']}; rejected {receipt['rejected_count']} {args.group} guidance images")
    else:
        if len(args.archive) != 1:
            parser.error("one archive is required without --paired-human")
        receipt = convert_archive(args.archive[0], args.palette, args.output, group=args.group)
        print(f"Converted {receipt['converted_count']} {args.group} guidance images")


if __name__ == "__main__":
    main()

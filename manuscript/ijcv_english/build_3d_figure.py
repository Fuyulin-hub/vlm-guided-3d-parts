"""Compose the approved A/B/C/D GLB renders after exact background removal.

Only pixels exactly equal to the renderer's declared flat background are made
transparent. This conservative rule preserves pale object surfaces and all
anti-aliased boundary pixels. The four arms in each row retain one shared crop.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS = Path(
    r"C:\Users\ASUS\Documents\Codex\2026-09-03\ai\downloads"
    r"\ABCD500_usercode_evaluation_20261025\abcd_glb_local_render_visual5_20261029"
)
MANIFEST = ASSETS / "render_manifest_final.json"
STEM = "fig4_3d_comparison_no_background"
PROCESSED = ROOT / "processed_panels"
ARMS = ["A_raw", "B_sam2", "C_v5_full", "D_human_fixed_repaired"]
HEADERS = ["A: Raw-guided", "B: SAM2-guided", "C: Ours", "D: Human-guided"]
BACKGROUND = np.array([247, 248, 250], dtype=np.uint8)
WIDTH_MM, HEIGHT_MM = 180.0, 204.0

sys.path.insert(0, str(Path(r"C:\Users\ASUS\.codex\skills\nature-figure\scripts")))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_exact_background(image: Image.Image) -> tuple[Image.Image, int]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    exact = np.all(rgb == BACKGROUND, axis=2)
    rgba = np.empty((*rgb.shape[:2], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(exact, 0, 255).astype(np.uint8)
    # Transparent pixels are normalized to white for clean downstream compositing.
    rgba[exact, :3] = 255
    return Image.fromarray(rgba, mode="RGBA"), int(exact.sum())


def pad_to_panel_aspect(image: Image.Image, target_aspect: float = 4.0 / 3.0) -> Image.Image:
    """Pad with transparency so imshow can fill a 4:3 panel without distortion."""
    width, height = image.size
    current = width / height
    if current < target_aspect:
        canvas_size = (int(np.ceil(height * target_aspect)), height)
    else:
        canvas_size = (width, int(np.ceil(width / target_aspect)))
    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 0))
    canvas.alpha_composite(image, ((canvas_size[0] - width) // 2, (canvas_size[1] - height) // 2))
    return canvas


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    objects = manifest["objects"]
    if len(objects) != 6:
        raise ValueError("The approved figure requires six objects.")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor="white")
    axes = []
    panel_ids = []
    records = []
    left_mm, column_mm, row_mm, image_height_mm = 10.0, 42.0, 32.0, 30.0

    for j, header in enumerate(HEADERS):
        fig.text((left_mm + (j + 0.5) * column_mm) / WIDTH_MM,
                 (HEIGHT_MM - 5.0) / HEIGHT_MM, header,
                 ha="center", va="center", fontsize=9, fontweight="bold")

    for i, obj in enumerate(objects):
        images: list[Image.Image] = []
        paths: list[Path] = []
        removed_counts: list[int] = []
        for arm in ARMS:
            output = next(entry for entry in obj["outputs"] if entry["arm"] == arm)
            path = ASSETS / output["png"]
            if file_sha256(path) != output["png_sha256"]:
                raise ValueError(f"Source render changed: {path}")
            processed, removed = remove_exact_background(Image.open(path))
            images.append(processed)
            paths.append(path)
            removed_counts.append(removed)

        boxes = [image.getchannel("A").getbbox() for image in images]
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            raise ValueError(f"No visible geometry: {obj['uid']}")
        x0 = max(0, min(box[0] for box in boxes) - 24)
        y0 = max(0, min(box[1] for box in boxes) - 24)
        x1 = min(images[0].width, max(box[2] for box in boxes) + 24)
        y1 = min(images[0].height, max(box[3] for box in boxes) + 24)
        crop = (x0, y0, x1, y1)

        bottom_mm = HEIGHT_MM - 9.0 - (i + 1) * row_mm
        panel_records = []
        for j, (arm, image, path, removed) in enumerate(zip(ARMS, images, paths, removed_counts)):
            cropped = image.crop(crop)
            display_image = pad_to_panel_aspect(cropped)
            panel_dir = PROCESSED / obj["uid"]
            panel_dir.mkdir(parents=True, exist_ok=True)
            panel_path = panel_dir / f"{arm}.png"
            display_image.save(panel_path)

            x_mm = left_mm + j * column_mm + 1.0
            y_mm = bottom_mm + (row_mm - image_height_mm) / 2.0
            ax = fig.add_axes([
                x_mm / WIDTH_MM,
                y_mm / HEIGHT_MM,
                (column_mm - 2.0) / WIDTH_MM,
                image_height_mm / HEIGHT_MM,
            ])
            # The local Matplotlib style uses a light-gray axes face by default.
            # Force paper white so transparent render backgrounds disappear in
            # both raster previews and the embedded PDF.
            ax.set_facecolor("white")
            ax.imshow(np.asarray(display_image), interpolation="none", aspect="auto")
            ax.set_axis_off()
            axes.append(ax)
            panel_ids.append(f"{chr(97 + i)}{j + 1}")
            panel_records.append({
                "arm": arm,
                "source": str(path),
                "source_sha256": file_sha256(path),
                "processed": str(panel_path),
                "transparent_pixels_full_frame": removed,
            })

        fig.text(4.5 / WIDTH_MM, (bottom_mm + row_mm / 2.0) / HEIGHT_MM,
                 f"({chr(97 + i)})", ha="center", va="center", fontsize=8)
        if i < len(objects) - 1:
            y = bottom_mm / HEIGHT_MM
            fig.add_artist(plt.Line2D([left_mm / WIDTH_MM, (WIDTH_MM - 2.0) / WIDTH_MM],
                                      [y, y], transform=fig.transFigure,
                                      color="#D6DBE1", linewidth=0.35))
        records.append({
            "panel": chr(97 + i),
            "uid": obj["uid"],
            "group": obj["group"],
            "shared_crop_xyxy": list(crop),
            "pose": obj["pose"],
            "outputs": panel_records,
        })

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=ROOT / f"{STEM}.alignment.json",
        overlay_svg=ROOT / f"{STEM}.alignment.svg",
        axes=axes,
        panel_ids=panel_ids,
        row_groups=[panel_ids[row * 4:(row + 1) * 4] for row in range(6)],
        column_groups=[[panel_ids[row * 4 + col] for row in range(6)] for col in range(4)],
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(ROOT / f"{STEM}.pdf")
    fig.savefig(ROOT / f"{STEM}.svg")
    fig.savefig(ROOT / f"{STEM}.png", dpi=600)
    plt.close(fig)

    provenance = {
        "backend": "Python / Matplotlib + Pillow exact-color transparency",
        "source_manifest": str(MANIFEST),
        "source_manifest_sha256": file_sha256(MANIFEST),
        "background_rgb_removed": BACKGROUND.tolist(),
        "removal_rule": "Only exact RGB matches are transparent; no threshold, inpainting, recoloring, or generative edit.",
        "panels": 24,
        "size_mm": [WIDTH_MM, HEIGHT_MM],
        "records": records,
    }
    (ROOT / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {STEM}: 24 panels with exact flat-background pixels removed")


if __name__ == "__main__":
    main()

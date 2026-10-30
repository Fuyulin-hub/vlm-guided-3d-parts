"""Extract paired guidance assets and build visual review sheets without changing sources."""
from pathlib import Path
import io
import json
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from convert_assemblies_guidance_labels import validate_paired_archives

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\weixin\xwechat_files\wxid_fe1hofte5xox22_2446\msg\file\2026-10")
OUT = ROOT / "downloads" / "D_thin_solid_repair_20261023"

def thumb(image, size=(160, 160)):
    image = image.convert("RGBA")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    y, x = np.indices((size[1], size[0]))
    grid = np.where(((x // 12 + y // 12) % 2)[..., None], 220, 245)
    bg = Image.fromarray(np.repeat(grid, 3, axis=2).astype(np.uint8)).convert("RGBA")
    bg.alpha_composite(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return bg.convert("RGB")

def main():
    (OUT / "review").mkdir(parents=True, exist_ok=True)
    rows = []
    for group, names in [("thin_fine_grained", ["总thin.zip"]), ("solid_volume", ["solid6.zip", "总solid.zip"])]:
        pairs = validate_paired_archives([SOURCE / name for name in names])
        group_rows = []
        for uid, (archive_path, raw_entry, human_entry) in sorted(pairs.items()):
            dest = OUT / "sources" / group
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as archive:
                raw_bytes, human_bytes = archive.read(raw_entry), archive.read(human_entry)
            raw_path = dest / f"{uid}_raw.png"
            human_path = dest / f"{uid}_hum{Path(human_entry).suffix.lower()}"
            raw_path.write_bytes(raw_bytes)
            human_path.write_bytes(human_bytes)
            with Image.open(io.BytesIO(human_bytes)) as image:
                rgba = np.asarray(image.convert("RGBA"))
                entry = {"uid": uid, "group": group, "archive": str(archive_path), "raw_entry": raw_entry,
                         "human_entry": human_entry, "raw_path": str(raw_path.relative_to(OUT)),
                         "human_path": str(human_path.relative_to(OUT)), "size": list(image.size),
                         "mode": image.mode, "partial_alpha_pixels": int(np.sum((rgba[...,3]>0)&(rgba[...,3]<255)))}
            rows.append(entry)
            group_rows.append(entry)
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
        for page, start in enumerate(range(0, len(group_rows), 15), 1):
            sheet = Image.new("RGB", (1020, 1000), "white")
            draw = ImageDraw.Draw(sheet)
            for slot, row in enumerate(group_rows[start:start+15]):
                x, y = (slot % 3)*340, (slot // 3)*200
                draw.text((x+5,y+3), row["uid"][:12]+" / "+str(row["size"]), fill="black", font=font)
                for offset, key, name in [(5,"raw_path","raw"),(175,"human_path","D source")]:
                    with Image.open(OUT / row[key]) as image:
                        sheet.paste(thumb(image), (x+offset,y+37))
                    draw.text((x+offset,y+20), name, fill="black", font=font)
            sheet.save(OUT / "review" / f"source_{group}_{page:02}.jpg", quality=94)
    (OUT / "source_manifest.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Extracted {len(rows)} pairs into {OUT}")

if __name__ == "__main__":
    main()

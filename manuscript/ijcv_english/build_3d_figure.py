"""Compose the six approved GLB-render comparisons for the manuscript."""
from pathlib import Path
from io import BytesIO
import hashlib
import json

from PIL import Image, ImageChops
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parents[1] / 'downloads/ABCD500_usercode_evaluation_20261025/abcd_glb_local_render_visual5_20261029'
MANIFEST = ASSETS / 'render_manifest_final.json'
BUILD = ROOT / 'build/qualitative'
OUT = ROOT / 'fig4_3d_comparison.pdf'
ARMS = ['A_raw', 'B_sam2', 'C_v5_full', 'D_human_fixed_repaired']
HEADERS = ['A: Raw-guided', 'B: SAM2-guided', 'C: Ours', 'D: Human-guided']


def main():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    objects = manifest['objects']
    if len(objects) != 6:
        raise ValueError('The approved figure requires six objects.')
    BUILD.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont('FigureTimes', 'C:/Windows/Fonts/times.ttf'))
    pdfmetrics.registerFont(TTFont('FigureTimesBold', 'C:/Windows/Fonts/timesbd.ttf'))
    width, height = 180 * mm, 204 * mm
    left, column, row, image_height = 10 * mm, 42 * mm, 32 * mm, 30 * mm
    pdf = canvas.Canvas(str(OUT), pagesize=(width, height), pageCompression=1)
    pdf.setTitle('SegViGen outputs under six approved A/B/C/D conditions')
    pdf.setAuthor('')
    pdf.setFont('FigureTimesBold', 9)
    for j, header in enumerate(HEADERS):
        pdf.drawCentredString(left + (j + 0.5) * column, height - 5 * mm, header)
    records = []
    background = (247, 248, 250)
    for i, obj in enumerate(objects):
        images = []
        paths = []
        for arm in ARMS:
            output = next(x for x in obj['outputs'] if x['arm'] == arm)
            path = ASSETS / output['png']
            if hashlib.sha256(path.read_bytes()).hexdigest() != output['png_sha256']:
                raise ValueError(f'Source render changed: {path}')
            images.append(Image.open(path).convert('RGB'))
            paths.append(path)
        # One shared crop per object keeps corresponding views and scales aligned.
        boxes = []
        for image in images:
            diff = ImageChops.difference(image, Image.new('RGB', image.size, background))
            channels = diff.split()
            support = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
            bounds = support.point(lambda value: 255 if value > 18 else 0).getbbox()
            if bounds:
                boxes.append(bounds)
        if not boxes:
            raise ValueError(f'No visible geometry: {obj["uid"]}')
        x0, y0 = max(0, min(b[0] for b in boxes)-24), max(0, min(b[1] for b in boxes)-24)
        x1 = min(images[0].width, max(b[2] for b in boxes)+24)
        y1 = min(images[0].height, max(b[3] for b in boxes)+24)
        crop = (x0, y0, x1, y1)
        bottom = height - 9 * mm - (i+1) * row
        for j, image in enumerate(images):
            image = image.crop(crop)
            payload = BytesIO()
            image.save(payload, format='PNG')
            payload.seek(0)
            scale = min((column-2*mm)/image.width, image_height/image.height)
            w, h = image.width*scale, image.height*scale
            x = left + j*column + (column-w)/2
            y = bottom + (row-h)/2
            pdf.drawImage(ImageReader(payload), x, y, width=w, height=h, mask='auto')
        pdf.setFont('FigureTimes', 8)
        pdf.drawCentredString(4.5*mm, bottom+row/2, f'({chr(97+i)})')
        if i < len(objects)-1:
            pdf.setStrokeColorRGB(0.84, 0.86, 0.89)
            pdf.setLineWidth(0.35)
            pdf.line(left, bottom, width-2*mm, bottom)
        records.append({'panel':chr(97+i), 'uid':obj['uid'], 'group':obj['group'],
                        'shared_crop_xyxy':list(crop), 'sources':[str(p) for p in paths],
                        'pose':obj['pose']})
    pdf.showPage()
    pdf.save()
    (BUILD/'sources.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Created {OUT}: {len(objects)} rows, {len(objects)*len(ARMS)} original GLB renders')


if __name__ == '__main__':
    main()

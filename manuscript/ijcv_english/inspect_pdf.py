"""Create a contact sheet and check the current manuscript PDF."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

root = Path(__file__).resolve().parent
reader = PdfReader(str(root/'main.pdf'))
folder = root/'build/current_pdf'
pages = sorted(folder.glob('page-*.png'))
if len(pages) != len(reader.pages):
    raise ValueError('Rendered pages do not match the current PDF.')
tile_w, tile_h = 340, 505
sheet = Image.new('RGB', (tile_w*4, tile_h*((len(pages)+3)//4)), 'white')
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
for i, page in enumerate(pages):
    image = Image.open(page).convert('RGB')
    image.thumbnail((tile_w-12, tile_h-30), Image.Resampling.LANCZOS)
    x, y = (i%4)*tile_w, (i//4)*tile_h
    sheet.paste(image, (x+(tile_w-image.width)//2, y+25))
    draw.text((x+12, y+3), f'Page {i+1}', fill='black', font=font)
sheet.save(folder/'contact_sheet.png')
texts = [page.extract_text() or '' for page in reader.pages]
figure_page = next(i for i,text in enumerate(texts) if 'A: Raw-guided' in text)
checks = {
    'pages': len(reader.pages),
    'updated_baseline_numbers_present': all(value in '\n'.join(texts) for value in
       ['38.19','17.92','35.13','24.35','10.88','31.66','32.71','13.94','16.75']),
    'figure4_page': figure_page+1,
    'figure4_image_count': len(reader.pages[figure_page].images),
    'figure4_before_references': figure_page < next(i for i,text in enumerate(texts) if 'References' in text),
    'springer_class_preserved': (root/'sn-jnl.cls').exists(),
}
if checks['figure4_image_count'] != 24 or not checks['updated_baseline_numbers_present']:
    raise ValueError(checks)
(folder/'checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(json.dumps(checks))

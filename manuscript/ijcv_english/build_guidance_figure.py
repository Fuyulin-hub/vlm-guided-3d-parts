"""Compose author-supplied PPT image assets without repainting their labels."""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import base64
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r'D:/weixin/xwechat_files/wxid_fe1hofte5xox22_2446/msg/file/2026-10/3D Asset Segmentation(4).pptx')
ROWS = [
    ['image2.png', 'image3.png', 'image6.png', 'image8.png'],
    ['image1.png', 'image4.png', 'image5.png', 'image7.png'],
    ['image17.png', 'image18.png', 'image19.png', 'image18.png'],
    ['image23.png', 'image20.png', 'image21.png', 'image22.png'],
    ['image10.png', 'image11.png', 'image13.png', 'image16.png'],
    ['image9.png', 'image12.png', 'image14.png', 'image15.png'],
]
WIDTH, HEIGHT = 2000, 1580
LEFT, TOP, COL, GAP, ROW, GROUP_GAP = 110, 80, 450, 16, 236, 24
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">', '<rect width="100%" height="100%" fill="white"/>']
labels = ['A · Input', 'B · Automatic SAM2', 'C · Ours', 'D · Human Reference']
for j, label in enumerate(labels):
    center = LEFT + j*(COL+GAP) + COL/2
    svg.append(f'<text x="{center}" y="43" text-anchor="middle" font-family="Times New Roman" font-size="34" font-weight="bold" fill="#20252b">{label}</text>')
svg.append('<path d="M96 60 H1988" fill="none" stroke="#cfd5db" stroke-width="2"/>')
with ZipFile(SOURCE) as deck:
    for i, row in enumerate(ROWS):
        y = TOP + i*ROW + (i//2)*GROUP_GAP
        for j, filename in enumerate(row):
            blob = deck.read('ppt/media/' + filename)
            im = Image.open(BytesIO(blob)).convert('RGBA')
            bbox = im.getchannel('A').getbbox() or (0,0,*im.size)
            x0,y0,x1,y1 = bbox
            x0=max(0,x0-8);y0=max(0,y0-8);x1=min(im.width,x1+8);y1=min(im.height,y1+8)
            payload = base64.b64encode(blob).decode('ascii')
            x = LEFT + j*(COL+GAP)
            svg.append(f'<svg x="{x+12}" y="{y+8}" width="{COL-24}" height="{ROW-18}" viewBox="{x0} {y0} {x1-x0} {y1-y0}" preserveAspectRatio="xMidYMid meet"><image x="0" y="0" width="{im.width}" height="{im.height}" xlink:href="data:image/png;base64,{payload}"/></svg>')
    for g, label in enumerate(['Assemblies','Solid Volume','Thin / Fine-Grained']):
        group_top = TOP + g*(2*ROW+GROUP_GAP)
        center = group_top + ROW
        svg.append(f'<text x="40" y="{center}" transform="rotate(-90 40 {center})" text-anchor="middle" font-family="Times New Roman" font-size="33" font-weight="bold" fill="#4e5b67">{label}</text>')
        if g<2:
            line_y=group_top+2*ROW+GROUP_GAP/2
            svg.append(f'<path d="M96 {line_y} H1988" fill="none" stroke="#dce1e6" stroke-width="2"/>')
svg.append('</svg>')
(ROOT/'fig3_guidance.svg').write_text('\n'.join(svg),encoding='utf-8')
print('Created fig3_guidance.svg from all 24 original placements; B/D cone duplicate retained as confirmed.')

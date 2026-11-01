"""Freeze the selected five RAW inputs, D references and Automatic SAM2 outputs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/data2/codefile/fuyulin/experiments/objaverse_684')
DROOT = BASE / 'guidance/D_human_fixed_repaired_20261022'
selected = json.loads((ROOT / 'selected_objects.json').read_text())
uids = {r['uid'] for r in selected}
assert len(uids) == 5
rows = list(csv.DictReader((BASE / 'manifest_frozen.csv').open()))
rows = [r for r in rows if r['group'] == 'assemblies' and r['uid'] in uids]
assert {r['uid'] for r in rows} == uids
rows.sort(key=lambda r: r['uid'])
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
expected = {r['uid']: r for r in selected}
for row in rows:
    uid = row['uid']
    row['raw_sha256'] = sha(Path(row['input_rgba']))
    label = DROOT / 'references/assemblies' / uid / 'labels_uint16.png'
    assert sha(label) == expected[uid]['labels_sha256']
    dest = ROOT / 'references/assemblies' / uid
    dest.mkdir(parents=True, exist_ok=True)
    for p in label.parent.glob('*.json'):
        shutil.copyfile(p, dest / p.name)
    shutil.copyfile(label, dest / label.name)
    bpath = BASE / 'runs/b_sam2_rot90_606_20261022/guidance/B_sam2/assemblies' / (uid + '.png')
    target = ROOT / 'guidance/sam2/assemblies' / bpath.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bpath, target)
references = []
for row in selected:
    references.append({**row, 'labels_file': 'references/assemblies/' + row['uid'] + '/labels_uint16.png'})
(ROOT / 'references_manifest.json').write_text(json.dumps({'group': 'assemblies', 'per_uid': references}, indent=2))
palette = json.loads((ROOT / 'palette.json').read_text())
(ROOT / 'B_decoder.json').write_text(json.dumps([[0, 0, 0, 255]] + palette))
with (ROOT / 'manifest.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
source_hashes = {p.name: sha(p) for p in (ROOT / 'source').glob('*.py')}
(ROOT / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=2))
print(json.dumps({'frozen_objects': [r['uid'] for r in rows], 'D_references': 5,
                  'B_guides_reused': 5, 'new_inference_objects_per_variant': 5,
                  'source_hashes': source_hashes}))

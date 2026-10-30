"""Freeze the updated server D cohort and its existing B guides, without inference."""
from pathlib import Path
import csv
import datetime
import hashlib
import io
import json
import tarfile

BASE = Path('/data2/codefile/fuyulin/experiments/objaverse_684')
DROOT = BASE / 'guidance/D_human_fixed_repaired_20261022'
OUTPUT = BASE / 'evaluations/guidance_category_refresh_20261025'
BROOT = (BASE / 'runs/b_sam2_rot90_606_20261022/guidance/B_sam2').resolve()
EXTRA = BASE / 'runs/extra11_abc_segvigen_20261025'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declared_label_hash(row):
    value = row.get('sha256')
    return row.get('labels_sha256') or (value.get('labels') if isinstance(value, dict) else None)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    exclusions = json.loads((DROOT / 'exclusions_solid_20261025.json').read_text())['excluded_uids']
    extra = json.loads((DROOT / 'manifest_assemblies_extra10_20261025.json').read_text())['cases']
    extra_map = {r['uid']: r for r in extra}
    b_transforms = {r['uid']: r for r in json.loads((EXTRA / 'b_guidance_transform_record.json').read_text())}
    solid = json.loads((DROOT / 'manifest_solid_repaired_20261024.json').read_text())['per_uid']
    solid_map = {r['uid']: r for r in solid}
    old_assembly = {r['uid']: r for r in csv.DictReader((DROOT / 'manifest_assemblies_repaired.csv').open())}
    inventory = {'snapshot_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'D_root': str(DROOT), 'B_frame': 'renders_rot90', 'excluded_solid_uids': exclusions,
                 'counts': {}, 'cases': [], 'source_files': [], 'archive_files': {}}
    archive = OUTPUT / 'server_scoring_snapshot_20261025.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        def add(source, relative):
            source = source.resolve()
            if not source.is_file():
                raise ValueError('Not a regular source file: ' + str(source))
            digest = sha(source)
            tar.add(str(source), arcname=relative, recursive=False)
            inventory['source_files'].append({'path': str(source), 'sha256': digest})
            inventory['archive_files'][relative] = digest
            return digest
        for group in ['assemblies', 'solid_volume']:
            guides = {p.stem: p for p in (DROOT / group).glob('*.png')}
            refs = {p.parent.name: p for p in (DROOT / 'references' / group).glob('*/labels_uint16.png')}
            if guides.keys() != refs.keys():
                raise ValueError('Guidance and reference UID sets differ')
            if group == 'solid_volume' and set(refs) & set(exclusions):
                raise ValueError('Excluded UID still active')
            inventory['counts'][group] = len(refs)
            for uid in sorted(refs):
                row = solid_map[uid] if group == 'solid_volume' else extra_map.get(uid, old_assembly[uid])
                bpath = BROOT / group / (uid + '.png')
                origin = 'existing_B_rot90k1'
                if group == 'assemblies' and uid in extra_map:
                    bpath = EXTRA / 'guidance/B_sam2/assemblies' / (uid + '.png')
                    origin = b_transforms[uid]['origin']
                    if sha(bpath) != b_transforms[uid]['sha256']:
                        raise ValueError('B transform receipt hash mismatch: ' + uid)
                if not bpath.exists():
                    raise ValueError('Missing B guide: ' + uid)
                record = {'uid': uid, 'group': group, 'source_D_metadata': row,
                          'D_labels_source': str(refs[uid]), 'B_source': str(bpath.resolve()),
                          'B_source_origin': origin, 'B_frame': 'renders_rot90',
                          'labels_file': 'D/references/' + group + '/' + uid + '/labels_uint16.png'}
                for p in sorted(refs[uid].parent.rglob('*')):
                    if p.is_file():
                        add(p, 'D/' + p.relative_to(DROOT).as_posix())
                record['labels_sha256'] = sha(refs[uid])
                declared = declared_label_hash(row)
                if declared is not None and record['labels_sha256'] != declared:
                    raise ValueError('D manifest hash mismatch: ' + uid)
                record['guide_sha256'] = add(guides[uid], 'D/' + group + '/' + uid + '.png')
                record['B_sha256'] = add(bpath, 'B_rot90/' + group + '/' + uid + '.png')
                inventory['cases'].append(record)
        for name in ['manifest_assemblies_repaired.csv', 'manifest_assemblies_extra10_20261025.json',
                     'manifest_solid_repaired_20261024.json', 'manifest_solid_repaired_20261024.csv',
                     'summary_solid_20261024.json', 'exclusions_solid_20261025.json']:
            add(DROOT / name, 'D/' + name)
        quarantine = BASE / 'guidance/D_human_fixed_excluded_user_20261025/operation_receipt.json'
        add(quarantine, 'quarantine_operation_receipt.json')
        data = json.dumps(inventory, ensure_ascii=False, indent=2).encode('utf-8')
        info = tarfile.TarInfo('inventory.json')
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    if any(sha(Path(r['path'])) != r['sha256'] for r in inventory['source_files']):
        raise ValueError('A source changed while creating the snapshot')
    receipt = {'archive': str(archive), 'archive_sha256': sha(archive), 'counts': inventory['counts'],
               'objects': len(inventory['cases']), 'files': len(inventory['archive_files']),
               'excluded_uids_absent': True, 'source_files_unchanged': True}
    (OUTPUT / 'snapshot_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

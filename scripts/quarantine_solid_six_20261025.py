"""Recoverably exclude the six user-named solid references, preserving other data."""

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import shutil

UIDS = ['222cbb6fec764c319be88ca67b7a4e70', '2903d7855bfb4164ad6749c1c8c5fede',
        '463ead51d6254a12bfd623f840e5ca21', '7a179f2b4837457eafa66c2f7094d280',
        'd35b5a6c707345c7a66c81c40087c43d', 'd215bdb9b8204f899a78fb1dc4b30f72']
DEFAULT_ROOT = '/data2/codefile/fuyulin/experiments/objaverse_684/guidance/D_human_fixed_repaired_20261022'
DEFAULT_QUARANTINE = '/data2/codefile/fuyulin/experiments/objaverse_684/guidance/D_human_fixed_excluded_user_20261025'
META = ['manifest_solid_repaired_20261024.json', 'manifest_solid_repaired_20261024.csv',
        'summary_solid_20261024.json', 'SHA256SUMS_solid_20261024.json', 'README_solid_20261024.md']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.solid_exclusions.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def snapshot(root, group):
    files = {}
    for directory in [root / group, root / 'references' / group]:
        for p in directory.rglob('*'):
            if p.is_symlink():
                raise ValueError('Symlink in scoped asset tree: ' + str(p))
            if p.is_file():
                p.resolve().relative_to(root.resolve())
                files[p.relative_to(root).as_posix()] = sha(p)
    return files


def counts(root):
    return {'solid_guidance': len(list((root / 'solid_volume').glob('*.png'))),
            'solid_labels': len(list((root / 'references/solid_volume').glob('*/labels_uint16.png')))}


def make_plan(root, quarantine, uids=UIDS):
    root, quarantine = Path(root).resolve(), Path(quarantine).resolve()
    if root == quarantine or root in quarantine.parents:
        raise ValueError('Quarantine must be outside the active reference root')
    if len(set(uids)) != len(uids) or any(not re.fullmatch('[0-9a-f]{32}', u) for u in uids):
        raise ValueError('Expected unique complete hexadecimal UIDs')
    manifest = json.loads((root / META[0]).read_text(encoding='utf-8'))
    if not set(uids) <= {r['uid'] for r in manifest['per_uid']}:
        raise ValueError('An exclusion UID is missing from the active manifest')
    paths = []
    for uid in uids:
        for relative in [Path('solid_volume') / (uid + '.png'), Path('references/solid_volume') / uid]:
            source, destination = root / relative, quarantine / relative
            if not source.exists():
                raise ValueError('Missing source ' + str(source))
            source.resolve().relative_to(root)
            destination.resolve().relative_to(quarantine)
            if source.is_symlink() or destination.exists():
                raise ValueError('Unsafe source or existing destination: ' + str(destination))
            paths.append(relative.as_posix())
    all_solid = snapshot(root, 'solid_volume')
    removed = {k: v for k, v in all_solid.items()
               if any(k == p or k.startswith(p + '/') for p in paths)}
    return {'uids': list(uids), 'root': str(root), 'quarantine': str(quarantine), 'move_paths': paths,
            'before_counts': counts(root), 'solid_assets_before': all_solid,
            'removed_asset_hashes': removed, 'assemblies_before': snapshot(root, 'assemblies'),
            'metadata_hashes': {name: sha(root / name) for name in META if (root / name).exists()}}


def apply_plan(root, quarantine, plan):
    root, quarantine = Path(root).resolve(), Path(quarantine).resolve()
    if str(root) != plan['root'] or str(quarantine) != plan['quarantine']:
        raise ValueError('Plan roots changed')
    if snapshot(root, 'solid_volume') != plan['solid_assets_before']:
        raise ValueError('Solid input changed after planning')
    if snapshot(root, 'assemblies') != plan['assemblies_before']:
        raise ValueError('Assemblies input changed after planning')
    if any(sha(root / name) != digest for name, digest in plan['metadata_hashes'].items()):
        raise ValueError('Metadata changed after planning')
    if any((quarantine / p).exists() for p in plan['move_paths']):
        raise ValueError('Quarantine destination changed after planning')
    uids = set(plan['uids'])
    backup = quarantine / 'metadata_before'
    backup.mkdir(parents=True, exist_ok=True)
    for name in plan['metadata_hashes']:
        shutil.copy2(root / name, backup / name)
    receipt = {**plan, 'status': 'planned', 'reason': 'User requested removal from solid and all scoring',
               'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    save(quarantine / 'operation_receipt.json', receipt)
    moved = []
    exclusion_file = root / 'exclusions_solid_20261025.json'
    if exclusion_file.exists():
        raise ValueError('Existing active exclusion file must be reviewed before reuse')
    try:
        for relative in plan['move_paths']:
            source, destination = root / relative, quarantine / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moved.append(relative)
        for relative, digest in plan['removed_asset_hashes'].items():
            if sha(quarantine / relative) != digest:
                raise ValueError('Moved asset hash mismatch: ' + relative)
        manifest_path = root / META[0]
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['per_uid'] = [r for r in manifest['per_uid'] if r['uid'] not in uids]
        manifest['total'] = len(manifest['per_uid'])
        manifest['excluded_uids'] = sorted(set(manifest.get('excluded_uids', [])) | uids)
        manifest['exclusion_receipt'] = str(quarantine / 'operation_receipt.json')
        save(manifest_path, manifest)
        csv_path = root / META[1]
        if csv_path.exists():
            reader = csv.DictReader(io.StringIO(csv_path.read_text(encoding='utf-8-sig')))
            fields = reader.fieldnames
            rows = [r for r in reader if r['uid'] not in uids]
            handle = io.StringIO(newline='')
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            temporary = csv_path.with_suffix('.csv.solid_exclusions.tmp')
            temporary.write_text(handle.getvalue(), encoding='utf-8-sig')
            temporary.replace(csv_path)
        summary_path = root / META[2]
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        for key in ['count', 'guide_count', 'label_count', 'mapping_count']:
            summary[key] = manifest['total']
        summary['origin_counts'] = dict(Counter(r.get('origin', 'unspecified') for r in manifest['per_uid']))
        summary['excluded_uids'] = sorted(uids)
        summary['exclusion_receipt'] = str(quarantine / 'operation_receipt.json')
        save(summary_path, summary)
        readme = root / META[4]
        if readme.exists():
            text = readme.read_text(encoding='utf-8')
            readme.write_text('2026-10-25 更新：用户指定的 6 个 UID 已隔离，当前活动 solid 为 '
                              + str(manifest['total']) + ' 套；详 exclusions_solid_20261025.json。\n\n' + text,
                              encoding='utf-8')
        save(exclusion_file, {'group': 'solid_volume', 'excluded_uids': sorted(uids),
                             'reason': receipt['reason'], 'quarantine_root': str(quarantine),
                             'current_count': manifest['total'], 'applies_to_all_scoring_methods': True})
        hashes_path = root / META[3]
        if hashes_path.exists():
            hashes = json.loads(hashes_path.read_text(encoding='utf-8'))
            hashes = {k: v for k, v in hashes.items() if k not in plan['removed_asset_hashes']}
            for name in META:
                if name != META[3] and (root / name).exists():
                    hashes[name] = sha(root / name)
            hashes[exclusion_file.name] = sha(exclusion_file)
            for relative, digest in hashes.items():
                if sha(root / relative) != digest:
                    raise ValueError('Active hash table mismatch: ' + relative)
            save(hashes_path, hashes)
        remaining = {k: v for k, v in plan['solid_assets_before'].items()
                     if k not in plan['removed_asset_hashes']}
        if snapshot(root, 'solid_volume') != remaining:
            raise ValueError('Unexpected change to a retained solid asset')
        if snapshot(root, 'assemblies') != plan['assemblies_before']:
            raise ValueError('Unexpected change to assemblies')
        receipt.update(status='completed', after_counts=counts(root), assemblies_unchanged=True,
                       retained_solid_assets_unchanged=True, moved_asset_count=len(plan['removed_asset_hashes']),
                       current_manifest_sha256=sha(manifest_path), all_removed_assets_recoverable=True)
        save(quarantine / 'operation_receipt.json', receipt)
        return receipt
    except Exception as exc:
        for relative in reversed(moved):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            (quarantine / relative).rename(destination)
        for name in plan['metadata_hashes']:
            shutil.copy2(backup / name, root / name)
        if exclusion_file.exists():
            exclusion_file.unlink()
        receipt.update(status='failed_rolled_back', error=repr(exc))
        save(quarantine / 'operation_receipt.json', receipt)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--quarantine', type=Path, default=Path(DEFAULT_QUARANTINE))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    import fcntl
    with (args.root / '.solid_user_exclusions.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = make_plan(args.root, args.quarantine)
        result = apply_plan(args.root, args.quarantine, plan) if args.apply else plan
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ['solid_assets_before', 'assemblies_before', 'removed_asset_hashes']},
                         ensure_ascii=False))

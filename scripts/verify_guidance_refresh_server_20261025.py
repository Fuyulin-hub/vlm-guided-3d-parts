"""Read-only final audit of active and quarantined server assets."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path('/data2/codefile/fuyulin/experiments/objaverse_684')
DROOT = BASE / 'guidance/D_human_fixed_repaired_20261022'
QUARANTINE = BASE / 'guidance/D_human_fixed_excluded_user_20261025'
OUTPUT = BASE / 'evaluations/guidance_category_refresh_20261025'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = read(QUARANTINE / 'operation_receipt.json')
    assert receipt['status'] == 'completed'
    excluded = set(receipt['uids'])
    assert len(excluded) == 6
    for relative, digest in receipt['removed_asset_hashes'].items():
        assert not (DROOT / relative).exists(), relative
        assert sha(QUARANTINE / relative) == digest, relative
    counts = {}
    for group, expected in [('assemblies', 214), ('solid_volume', 181)]:
        guides = {p.stem for p in (DROOT / group).glob('*.png')}
        labels = {p.parent.name for p in (DROOT / 'references' / group).glob('*/labels_uint16.png')}
        assert guides == labels and len(labels) == expected
        assert not excluded.intersection(labels)
        counts[group] = len(labels)
    manifest = read(DROOT / 'manifest_solid_repaired_20261024.json')
    assert len(manifest['per_uid']) == 181
    assert not excluded.intersection(r['uid'] for r in manifest['per_uid'])
    with (DROOT / 'manifest_solid_repaired_20261024.csv').open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 181 and not excluded.intersection(r['uid'] for r in rows)
    assert set(read(DROOT / 'exclusions_solid_20261025.json')['excluded_uids']) == excluded
    for relative, digest in receipt['assemblies_before'].items():
        assert sha(DROOT / relative) == digest
    with tarfile.open(OUTPUT / 'server_scoring_snapshot_20261025.tar.gz', 'r:gz') as archive:
        inventory = json.load(archive.extractfile('inventory.json'))
    for entry in inventory['source_files']:
        assert sha(Path(entry['path'])) == entry['sha256'], entry['path']
    result = {'active_counts': counts, 'excluded_count': 6,
              'recoverable_files_hash_verified': len(receipt['removed_asset_hashes']),
              'all_preexisting_assemblies_unchanged': True,
              'solid_json_and_csv_manifests_match_181': True,
              'all_snapshot_server_sources_unchanged': True,
              'snapshot_source_files_verified': len(inventory['source_files']),
              'quarantine': str(QUARANTINE)}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

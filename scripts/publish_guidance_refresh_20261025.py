"""Publish completed outputs into a new server results subdirectory, without overwrites."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/data2/codefile/fuyulin/experiments/objaverse_684/evaluations/guidance_category_refresh_20261025')
EXPECTED = 'c71ae0fe1822ed9e0af7e575de2910a36edf66e033639061cacf1b8aaa2f8df4'


def main():
    bundle = ROOT / 'scoring_results_20261025.tar.gz'
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == EXPECTED
    destination = ROOT / 'results'
    with tarfile.open(bundle, 'r:gz') as archive:
        manifest = json.load(archive.extractfile('result_bundle_manifest.json'))
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            target.relative_to(destination.resolve())
            assert member.isfile(), member.name
            data = archive.extractfile(member).read()
            if member.name in manifest['files']:
                assert hashlib.sha256(data).hexdigest() == manifest['files'][member.name]
            if target.exists():
                assert target.read_bytes() == data, 'Refusing to overwrite different output: ' + member.name
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == digest
    summary = json.loads((destination / 'summary.json').read_text())
    print(json.dumps({'server_results': str(destination), 'verified_files': len(manifest['files']),
                      'paired_counts': {g: r['n_paired'] for g, r in summary['groups'].items()},
                      'complete': summary['complete_on_available_pairs'], 'excluded_count': len(summary['excluded_solid_uids'])}))


if __name__ == '__main__':
    main()

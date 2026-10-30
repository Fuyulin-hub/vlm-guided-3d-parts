"""Check exported native numeric cells and package the completed scoring outputs."""
import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/guidance_category_refresh_20261025'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def main():
    summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    workbook = OUT / 'B_C_三类二维引导图评估.xlsx'
    groups = ['assemblies', 'solid_volume', 'thin_fine_grained']
    metrics = ['matched_miou', 'boundary_f1', 'part_f1']
    methods = ['B_sam2', 'C_v5_full', 'delta_C_minus_B']
    checked = 0
    with zipfile.ZipFile(workbook) as archive:
        cells = []
        for name in ['sheet1', 'sheet2', 'sheet3']:
            tree = ET.fromstring(archive.read('xl/worksheets/' + name + '.xml'))
            entries = {c.attrib['r']: c for c in tree.findall('.//m:sheetData/m:row/m:c', NS)}
            assert not [c for c in entries.values() if c.attrib.get('t') == 'e']
            cells.append(entries)

        def check(sheet, address, target):
            nonlocal checked
            value = cells[sheet][address].find('m:v', NS)
            assert value is not None and abs(float(value.text) - target) < 1e-9, address
            checked += 1

        for gi, group in enumerate(groups):
            result = summary['groups'][group]
            coverage = [result['n_reference'], result['independent_method_coverage']['B_sam2'],
                        result['independent_method_coverage']['C_v5_full'], result['n_paired'],
                        len(result['missing_by_method']['B_sam2']), len(result['missing_by_method']['C_v5_full']),
                        result['evaluation_errors']]
            for ci, number in enumerate(coverage):
                check(2, 'CDEFGHI'[ci] + str(6 + gi), number)
            for mi, method in enumerate(methods):
                stats = result['methods'][method] if mi < 2 else result[method]
                row_main = 6 + gi * 2 + mi if mi < 2 else 16 + gi
                check(0, 'C' + str(row_main), result['n_paired'])
                for ki, key in enumerate(metrics):
                    stat = stats[key]
                    row = 6 + gi * 9 + mi * 3 + ki
                    for ci, value in enumerate([stat['mean'], *stat['ci95']]):
                        check(1, 'GHI'[ci] + str(row), value * 100)
                    check(0, 'EFG'[ki] + str(row_main), stat['mean'] * 100)
    verification = {'native_numeric_cells_verified': checked, 'native_error_cells': 0,
                    'three_sheet_visual_review_passed': True,
                    'workbook_sha256': hashlib.sha256(workbook.read_bytes()).hexdigest()}
    (OUT / 'export/native_export_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    relative = ['summary.json', 'report_zh.md', 'category_comparison.csv', workbook.name,
                'local_verification.json', 'server_verification.json', 'frozen_inputs.json',
                'export/export_verification.json', 'export/native_export_verification.json',
                'inputs/refresh_preparation.json', 'inputs/B_source_audit.json',
                'inputs/B_decoder_all_21_declared_colors.json', 'inputs/evaluation_code_clean.zip',
                'inputs/snapshot_receipt.json']
    for group in groups:
        relative.append('inputs/' + group + '_references.json')
        for method in methods[:2]:
            relative.extend(method + '/' + group + '/' + name for name in
                            ['summary.json', 'per_object_metrics.json', 'per_object_metrics.csv'])
    files = [(OUT / name, name) for name in relative]
    files.extend((path, 'code/' + path.name) for path in [
        ROOT / 'scripts/rerun_guidance_refresh_20261025.py',
        ROOT / 'scripts/run_guidance_category_comparison.py',
        ROOT / 'scripts/evaluate_c_d_guidance.py',
        ROOT / 'scripts/verify_guidance_refresh_local_20261025.py',
        ROOT / 'scripts/verify_guidance_refresh_server_20261025.py',
        ROOT / 'scripts/package_guidance_refresh_20261025.py'])
    files.extend((path, 'code/evaluation_code_clean/' + path.name) for path in
                 (OUT / 'inputs/evaluation_code_clean').glob('*.py'))
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for path, name in files}
    manifest_path = OUT / 'result_bundle_manifest.json'
    manifest_path.write_text(json.dumps({'files': hashes, 'C_source': summary['C_root'],
        'scope': summary['scope'], 'note': 'Reference labels and B images remain in the server input snapshot; C retains the explicitly selected local version.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    bundle = OUT / 'scoring_results_20261025.tar.gz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for path, name in files:
            archive.add(path, arcname=name, recursive=False)
        archive.add(manifest_path, arcname=manifest_path.name, recursive=False)
    with tarfile.open(bundle, 'r:gz') as archive:
        for member in archive.getmembers():
            if member.name in hashes:
                assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == hashes[member.name]
    print(json.dumps({**verification, 'bundle': str(bundle), 'bundle_files': len(files) + 1,
                      'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()

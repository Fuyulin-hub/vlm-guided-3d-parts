"""Re-score the user-updated D cohort with the previously fixed B/C metric protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np
from PIL import Image


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load(Path(__file__).with_name('run_guidance_category_comparison.py'), 'refresh_base')
direct = base.direct


def server_b_to_legacy_frame(pixels):
    """Only adapts storage: the evaluator's CCW90 restores the exact server pixels."""
    return np.rot90(pixels, k=-1).copy()


def prepare(workspace, previous, output, croot):
    inputs = output / 'inputs'
    snapshot_archive = inputs / 'server_scoring_snapshot_20261025.tar.gz'
    receipt = base.read(inputs / 'snapshot_receipt.json')
    if direct.digest(snapshot_archive) != receipt['archive_sha256']:
        raise ValueError('Server snapshot archive hash mismatch')
    snapshot = inputs / 'server_snapshot'
    snapshot.mkdir(parents=True, exist_ok=True)
    with tarfile.open(snapshot_archive, 'r:gz') as archive:
        for member in archive.getmembers():
            (snapshot / member.name).resolve().relative_to(snapshot.resolve())
            if not member.isfile():
                raise ValueError('Unexpected non-file archive entry')
        archive.extractall(snapshot, filter='data')
    inventory = base.read(snapshot / 'inventory.json')
    for relative, digest in inventory['archive_files'].items():
        if direct.digest(snapshot / relative) != digest:
            raise ValueError('Snapshot member hash mismatch: ' + relative)
    excluded = set(inventory['excluded_solid_uids'])
    entries = {g: [] for g in base.GROUPS}
    b_audit = []
    for case in inventory['cases']:
        group, uid = case['group'], case['uid']
        if uid in excluded:
            raise ValueError('Excluded UID present in the refreshed reference cohort')
        metadata = case['source_D_metadata']
        reference = snapshot / case['labels_file']
        entry = {'uid': uid, 'group': group, 'labels_file': str(reference.resolve()),
                 'labels_sha256': case['labels_sha256'],
                 'origin': metadata.get('origin', 'new_ai_assisted_assemblies_20261025'
                           if 'source_guidance_path' in metadata else 'existing_server_assemblies_204'),
                 'status': metadata.get('status'), 'ground_truth_ready': metadata.get('ground_truth_ready', False),
                 'server_reference_path': case['D_labels_source'], 'source_metadata': metadata}
        entries[group].append(entry)
        bpath = snapshot / 'B_rot90' / group / (uid + '.png')
        with Image.open(bpath) as image:
            pixels = np.asarray(image.convert('RGBA'))
        original = server_b_to_legacy_frame(pixels)
        if not np.array_equal(np.rot90(original, k=1), pixels):
            raise ValueError('Server B preparation is not exactly reversible')
        old = previous / 'inputs/B_original' / group / (uid + '.png')
        matches = None
        if old.exists():
            with Image.open(old) as image:
                old_pixels = np.asarray(image.convert('RGBA'))
            matches = bool(np.array_equal(np.rot90(old_pixels, k=1), pixels))
            if not matches:
                raise ValueError('An existing B guide changed since the previous run: ' + uid)
        destination = inputs / 'B_original' / group / (uid + '.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(original).save(destination)
        b_audit.append({'group': group, 'uid': uid, 'server_path': case['B_source'],
                        'server_sha256': case['B_sha256'], 'origin': case['B_source_origin'],
                        'server_frame': 'renders_rot90', 'preparation_rotation_ccw': -90,
                        'evaluation_rotation_ccw': 90, 'net_rotation_from_server': 0,
                        'pixel_equal_to_previous_B_rot90': matches})
    thin_manifest = base.read(previous / 'inputs/thin_fine_grained_references.json')
    for entry in thin_manifest['per_uid']:
        if direct.digest(entry['labels_file']) != entry['labels_sha256']:
            raise ValueError('Frozen thin reference changed: ' + entry['uid'])
        entries['thin_fine_grained'].append(entry)
        source = previous / 'inputs/B_original/thin_fine_grained' / (entry['uid'] + '.png')
        destination = inputs / 'B_original/thin_fine_grained' / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for group in base.GROUPS:
        uids = {r['uid'] for r in entries[group]}
        if len(uids) != len(entries[group]) or uids & excluded:
            raise ValueError('Duplicate or excluded UID in the scoring cohort')
        direct.save_json(inputs / (group + '_references.json'),
                         {'group': group, 'ground_truth_ready': False,
                          'per_uid': sorted(entries[group], key=lambda r: r['uid'])})
    previous_c_checked = 0
    for group in base.GROUPS:
        old_rows = base.read(previous / 'C_v5_full' / group / 'per_object_metrics.json')
        expected = {r['uid'] for r in entries[group]}
        for row in old_rows:
            if row['uid'] in expected:
                if direct.digest(croot / group / (row['uid'] + '.png')) != row['prediction_sha256']:
                    raise ValueError('Previously fixed local C image changed: ' + row['uid'])
                previous_c_checked += 1
    core_dir = inputs / 'evaluation_code_clean'
    core_dir.mkdir(parents=True, exist_ok=True)
    for p in (previous / 'inputs/evaluation_code_clean').glob('*.py'):
        shutil.copyfile(p, core_dir / p.name)
    shutil.copyfile(previous / 'inputs/evaluation_code_clean.zip', inputs / 'evaluation_code_clean.zip')
    shutil.copyfile(Path(direct.__file__), inputs / 'direct_adapter_used.py')
    palette = workspace / 'downloads/D_human_fixed_bc20_294_20261023/colors_pure_bc.json'
    b_palette = inputs / 'B_decoder_all_21_declared_colors.json'
    direct.save_json(b_palette, [[0, 0, 0, 255]] + base.read(palette))
    frozen = []
    for group in base.GROUPS:
        for row in entries[group]:
            frozen.append({'path': row['labels_file'], 'sha256': row['labels_sha256']})
            bpath = inputs / 'B_original' / group / (row['uid'] + '.png')
            frozen.append({'path': str(bpath.resolve()), 'sha256': direct.digest(bpath)})
            cpath = croot / group / (row['uid'] + '.png')
            if cpath.exists():
                frozen.append({'path': str(cpath.resolve()), 'sha256': direct.digest(cpath)})
    direct.save_json(output / 'frozen_inputs.json', frozen)
    direct.save_json(inputs / 'B_source_audit.json', b_audit)
    direct.save_json(inputs / 'refresh_preparation.json', {
        'reference_counts': {g: len(rows) for g, rows in entries.items()},
        'excluded_uids': sorted(excluded), 'previous_C_files_hash_checked': previous_c_checked,
        'old_B_guides_pixel_checked': sum(r['pixel_equal_to_previous_B_rot90'] is True for r in b_audit),
        'new_B_guides': [r['uid'] for r in b_audit if r['pixel_equal_to_previous_B_rot90'] is None],
        'thin_cohort_unchanged_from_previous': True, 'new_inference_run': False})
    return palette, b_palette, core_dir / 'paper_table_metrics.py'


def run(workspace, previous, output, croot):
    palette, b_palette, core = prepare(workspace, previous, output, croot)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {}
        for group in base.GROUPS:
            for method in base.METHODS:
                prediction_root = output / 'inputs/B_original' if method == 'B_sam2' else croot
                future = pool.submit(direct.evaluate, prediction_root / group, workspace,
                    output / 'inputs' / (group + '_references.json'),
                    b_palette if method == 'B_sam2' else palette, output / method / group,
                    method=method, rotation_k=1 if method == 'B_sam2' else 0, metrics_path=core)
                futures[future] = (method, group)
        for future in as_completed(futures):
            result = future.result()
            print(json.dumps({'finished': futures[future], 'n': result['n_evaluated'],
                              'errors': result['n_errors']}), flush=True)
    summary = base.finalize(output, output / 'inputs/evaluation_code_clean.zip',
                            output / 'inputs/server_scoring_snapshot_20261025.tar.gz', croot)
    summary['evaluation_date'] = '2026-10-25'
    summary['excluded_solid_uids'] = base.read(output / 'inputs/refresh_preparation.json')['excluded_uids']
    summary['B_source_archives'] = [
        {'scope': ['assemblies', 'solid_volume'], 'path': summary.pop('B_zip_path'),
         'sha256': summary.pop('B_zip_sha256'), 'source_frame': 'renders_rot90',
         'preparation': 'CW90 storage followed by frozen evaluator CCW90; exact net identity'},
        {'scope': ['thin_fine_grained'], 'path': str(previous.resolve()),
         'note': 'Same frozen 84 references and original B inputs as previous run; freshly scored'}]
    summary['previous_run'] = str(previous.resolve())
    summary['server_snapshot_counts'] = base.read(output / 'inputs/snapshot_receipt.json')['counts']
    summary['thin_reference_scope'] = 'Previous frozen 84 references, freshly rescored; no unuploaded later candidates added'
    for group, result in summary['groups'].items():
        if set(result['paired_uids']) & set(summary['excluded_solid_uids']):
            raise ValueError('An excluded UID was scored')
    direct.save_json(output / 'summary.json', summary)
    base.report(output, summary)
    notes = ('\n## 2026-10-25 样本更新\n\n'
             '- 服务器 D 的 assemblies 从 204 增为 214；新增 10 个参考的既有 B 图已补齐，未运行新推理。\n'
             '- solid 从 187 减为 181；用户指定的 6 个 UID 已隔离，所有方法评分均排除。\n'
             '- C 仍使用上轮确认的本地 D:/c_v5_full_guidance/C_v5_full，未切换到服务器其他 C 版本。\n'
             '- thin 仍使用上轮冻结的 84 套参考并重新计算，没有混入后来未上传的新候选。\n'
             '- 服务器 B 已在 renders_rot90 坐标系：先顺时针 90° 存储，再由旧评估器逆时针 90° 读入，净变换为恒等，逐像素核对。\n'
             '- 被排除的 6 个 UID 的 GLB/VXZ、A/B/C 原始资产及历史运行清单未删除。\n')
    report = output / 'report_zh.md'
    report.write_text(report.read_text(encoding='utf-8') + notes, encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--c-root', type=Path, default=Path('D:/c_v5_full_guidance/C_v5_full'))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    previous = (args.previous or workspace / 'outputs/guidance_category_comparison').resolve()
    output = (args.output or workspace / 'outputs/guidance_category_refresh_20261025').resolve()
    result = run(workspace, previous, output, args.c_root.resolve())
    raise SystemExit(0 if result['complete_on_available_pairs'] else 1)

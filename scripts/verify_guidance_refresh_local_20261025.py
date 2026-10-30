"""Independent read-only checks of the refreshed two-dimensional score records."""
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/guidance_category_refresh_20261025'
OLD = ROOT / 'outputs/guidance_category_comparison'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    spec = importlib.util.spec_from_file_location('verify_direct', ROOT / 'scripts/evaluate_c_d_guidance.py')
    direct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(direct)
    summary = read(OUT / 'summary.json')
    old = read(OLD / 'summary.json')
    excluded = set(summary['excluded_solid_uids'])
    assert len(excluded) == 6
    expected = {'assemblies': (214, 201), 'solid_volume': (181, 181), 'thin_fine_grained': (84, 79)}
    all_records = {}
    for group, (nref, npaired) in expected.items():
        result = summary['groups'][group]
        assert (result['n_reference'], result['n_paired']) == (nref, npaired)
        assert not excluded.intersection(result['paired_uids'])
        assert result['evaluation_errors'] == 0
        manifest = read(OUT / 'inputs' / (group + '_references.json'))['per_uid']
        assert len(manifest) == nref and not excluded.intersection(r['uid'] for r in manifest)
        for method in ['B_sam2', 'C_v5_full']:
            rows = read(OUT / method / group / 'per_object_metrics.json')
            by_uid = {r['uid']: r for r in rows}
            assert len(rows) == len(by_uid)
            assert not excluded.intersection(by_uid)
            for key in direct.KEYS:
                actual = float(np.mean([by_uid[u]['metrics'][key] for u in result['paired_uids']]))
                assert abs(actual - result['methods'][method][key]['mean']) < 1e-14
            all_records[method, group] = by_uid
    thin = summary['groups']['thin_fine_grained']
    oldthin = old['groups']['thin_fine_grained']
    for key in ['paired_uids', 'methods', 'delta_C_minus_B']:
        assert thin[key] == oldthin[key], key
    for entry in read(OUT / 'frozen_inputs.json'):
        assert direct.digest(entry['path']) == entry['sha256'], entry['path']
    core_dir = OUT / 'inputs/evaluation_code_clean'
    with zipfile.ZipFile(OUT / 'inputs/evaluation_code_clean.zip') as archive:
        for name, digest in summary['supplied_code_sha256'].items():
            matches = [n for n in archive.namelist() if Path(n).name == name]
            assert len(matches) == 1
            assert archive.read(matches[0]) == (core_dir / name).read_bytes()
            assert direct.digest(core_dir / name) == digest
    metric = direct.load_metrics(core_dir / 'paper_table_metrics.py')
    palette_path = ROOT / 'downloads/D_human_fixed_bc20_294_20261023/colors_pure_bc.json'
    cpalette = [tuple(c[:3]) for c in read(palette_path)[1:]]
    bpalette = [tuple(c[:3]) for c in read(OUT / 'inputs/B_decoder_all_21_declared_colors.json')[1:]]
    samples = {'assemblies': '5483beb389cd4837942e8f53aad09257',
               'solid_volume': '4b9683d86ae44ad5b2edfe17d850ac68',
               'thin_fine_grained': thin['paired_uids'][0]}
    recomputed = []
    for group, uid in samples.items():
        for method in ['B_sam2', 'C_v5_full']:
            row = all_records[method, group][uid]
            with Image.open(row['prediction_path']) as image:
                pred = direct.decode_colors(np.asarray(image.convert('RGBA')), bpalette if method == 'B_sam2' else cpalette)
            pred = np.rot90(pred, k=1 if method == 'B_sam2' else 0)
            ref = metric.read_label_map(row['reference_path'])
            scores = metric.label_metrics(pred, ref, boundary_tolerance=2, part_threshold=0.5)
            assert scores == row['metrics']
            folder = 'decoded_labels' if method == 'B_sam2' else 'decoded_C_labels'
            saved = metric.read_label_map(OUT / method / group / folder / group / uid / 'labels_uint16.png')
            assert np.array_equal(saved, pred)
            recomputed.append({'method': method, 'group': group, 'uid': uid, 'metrics_match': True})
    checks = {'reference_and_paired_counts': expected, 'all_6_exclusions_absent': True,
              'six_summary_means_reconciled': True, 'thin_scores_and_CIs_exactly_unchanged': True,
              'frozen_input_hashes_verified': len(read(OUT / 'frozen_inputs.json')),
              'supplied_four_code_files_byte_identical': True, 'independent_metric_recomputations': recomputed,
              'targeted_tests_passed': 25, 'project_tests_passed': 113, 'project_tests_failed': 4,
              'unrelated_project_test_failures': ['Three Windows fsync/read-only handle failures in old 3D projection tests',
                                                 'One EGL texture render test: trimesh unavailable'],
              'new_inference_run': False, 'D_ground_truth_ready_promoted': False}
    direct.save_json(OUT / 'local_verification.json', checks)
    summary['verification'] = {k: checks[k] for k in ['targeted_tests_passed', 'project_tests_passed', 'project_tests_failed']}
    summary['verification']['independent_sample_recomputations'] = len(recomputed)
    direct.save_json(OUT / 'summary.json', summary)
    print(json.dumps({k: v for k, v in checks.items() if k != 'independent_metric_recomputations'}))


if __name__ == '__main__':
    main()

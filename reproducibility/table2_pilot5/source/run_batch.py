"""Sequential GPU generation, then CPU-only scoring for the approved Table 2 pilot."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    variants = ['ours', 'box_point', 'single_positive', 'no_negative', 'no_feedback', 'no_point_fusion']
    state = {'status': 'running', 'pid': os.getpid(), 'physical_gpu': args.gpu,
             'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'objects_per_variant': 5, 'guides_expected': 35, 'new_guides_expected': 30,
             'segvigen_run': False, 'completed_variants': [], 'stage': 'generation'}
    save(root / 'state.json', state)
    for variant in variants:
        state['current_variant'] = variant
        save(root / 'state.json', state)
        with (root / ('worker_' + variant + '.log')).open('a') as log:
            result = subprocess.run([sys.executable, str(root / 'source/run_variant.py'),
                '--root', str(root), '--variant', variant], stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            state.update(status='failed', failed_variant=variant, returncode=result.returncode)
            save(root / 'state.json', state)
            return result.returncode
        state['completed_variants'].append(variant)
        save(root / 'state.json', state)
    state.update(stage='2d_scoring', current_variant=None)
    save(root / 'state.json', state)
    sys.path.insert(0, str(root / 'source'))
    from evaluate_c_d_guidance import evaluate
    names = [('sam2', 'Automatic SAM2'), ('box_point', 'Qwen-SAM2 (box + point)'),
             ('single_positive', 'Ours w/ single positive point'), ('no_negative', 'Ours w/o negative points'),
             ('no_feedback', 'Ours w/o feedback refinement'), ('no_point_fusion', 'Ours w/o point-aware fusion'),
             ('ours', 'Ours')]
    methods = []
    for key, name in names:
        palette = root / ('B_decoder.json' if key == 'sam2' else 'palette.json')
        output = root / 'evaluation' / key
        result = evaluate(root / 'guidance' / key / 'assemblies', root,
            root / 'references_manifest.json', palette, output, method=key, rotation_k=0,
            metrics_path=root / 'source/paper_table_metrics.py')
        if result['n_errors'] or result['n_evaluated'] != 5:
            raise ValueError('A pilot variant cannot be reported on fewer than the five frozen objects')
        methods.append({'key': key, 'name': name, 'tables': [2], 'guidance': {
            'labels': str(output / 'decoded_labels/{group}/{uid}/labels_uint16.png'),
            'label_semantics': 'palette_index'}})
    config = {'root': str(root), 'manifest': 'manifest.csv', 'groups': ['assemblies'],
        'expected_group_counts': {'assemblies': 5}, 'purpose': 'five_object_pilot_not_full_dataset',
        'metrics': {'background': 0, 'ignore_labels': [-1, 65535], 'boundary_tolerance_px': 2, 'part_iou_threshold': 0.5},
        'reference': {'guidance': {'labels': 'references/{group}/{uid}/labels_uint16.png'}}, 'methods': methods}
    save(root / 'table2_config.json', config)
    result = subprocess.run([sys.executable, str(root / 'source/eval_table2.py'),
        '--config', str(root / 'table2_config.json'), '--out', str(root / 'evaluation/table2')], env=env)
    state.update(status='complete' if result.returncode == 0 else 'failed', stage='complete',
                 completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), returncode=result.returncode)
    save(root / 'state.json', state)
    return result.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        root = Path(sys.argv[sys.argv.index('--root') + 1])
        state_path = root / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state.update(status='failed', error=repr(exc))
        save(state_path, state)
        raise

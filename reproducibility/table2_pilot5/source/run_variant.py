"""Run one approved ablation using byte-frozen original V4/V5 drivers."""
import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

from table2_ablation_controls import FrozenPlanner, policy, transform_part, without_feedback_source, require_fresh_variant


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--variant', required=True)
    args = parser.parse_args()
    root, variant = args.root.resolve(), args.variant
    controls = policy(variant)
    source = Path(__file__).resolve().parent
    frozen_sources = json.loads((root / 'source_hashes.json').read_text())
    for name, digest in frozen_sources.items():
        if sha(source / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    rows = list(csv.DictReader((root / 'manifest.csv').open()))
    if len(rows) != 5 or len({r['uid'] for r in rows}) != 5:
        raise ValueError('Pilot requires exactly five unique objects')
    require_fresh_variant(root, variant, rows)
    for row in rows:
        if sha(row['input_rgba']) != row['raw_sha256']:
            raise ValueError('RAW input changed: ' + row['uid'])
    plans = []
    if variant != 'ours':
        for row in rows:
            annotation = root / 'annotations/ours' / row['group'] / (row['uid'] + '.json')
            record = json.loads(annotation.read_text())
            if record.get('status') not in ('success', 'partial'):
                raise ValueError('Full method must produce the frozen plan first')
            plans.append({'uid': row['uid'], 'path': record['qwen_plan']['raw_response_path'],
                          'sha256': record['qwen_plan']['raw_response_sha256']})
    common = ['--experiment-root', str(root), '--manifest', str(root / 'manifest.csv'),
        '--palette', str(root / 'palette.json'), '--device', 'cuda:0', '--seed', '42',
        '--group-filter', 'assemblies', '--output-suffix', variant]
    planner_instances = []
    if variant == 'box_point':
        driver = load(source / 'run_qwen_sam2_guidance_v4.py', 'pilot_v4')
        planner = FrozenPlanner(driver.generate_qwen, driver.PROMPT_V3, plans)
        driver.generate_qwen = planner
        planner_instances.append(planner)
        sys.argv = [str(source / 'run_qwen_sam2_guidance_v4.py'), *common, '--mode', 'box_point']
        driver.main()
    else:
        driver = load(source / 'run_qwen_sam2_guidance_v5_feedback.py', 'pilot_v5')
        original_config = driver.make_frozen_config
        def frozen_config(**kwargs):
            value = original_config(**kwargs)
            value['schema_version'] = 'table2_ablation_config@1'
            value['ablation'] = {'variant': variant, 'controls': controls,
                'frozen_planning': variant != 'ours', 'full_plan_records': plans,
                'wrapper_sha256': sha(__file__), 'controls_sha256': sha(source / 'table2_ablation_controls.py'),
                'pilot_objects': 5, 'segvigen_run': False}
            value['prompting']['uses_all_positive_points'] = controls['max_positive_points'] is None
            value['prompting']['uses_all_negative_points'] = controls['negative_points']
            value['prompting']['max_feedback_rounds'] = 1 if controls['feedback'] else 0
            if not controls['point_filter']:
                value['fusion']['candidate_cleanup'] = 'foreground_clip+3x3_close+small_hole_fill'
            return value
        driver.make_frozen_config = frozen_config
        original_helpers = driver.load_v4_helpers
        def helpers(path):
            helper = original_helpers(path)
            if variant != 'ours':
                planner = FrozenPlanner(helper.generate_qwen, helper.PROMPT_V3, plans)
                helper.generate_qwen = planner
                planner_instances.append(planner)
            if variant in ('single_positive', 'no_negative'):
                original_sam = helper.run_sam2_for_part
                def sam(predictor, rgb, foreground, part, mode, minimum):
                    return original_sam(predictor, rgb, foreground, transform_part(part, variant), mode, minimum)
                helper.run_sam2_for_part = sam
            return helper
        driver.load_v4_helpers = helpers
        if variant != 'ours':
            driver.PROMPT_POLICY = variant + '_frozen_plan'
        if not controls['point_filter']:
            driver.component_at_point = lambda mask, point: mask.copy()
            driver.FUSION_STRATEGY = 'pointfree_small_bbox_first@1'
        transformed_sha = None
        if not controls['feedback']:
            rewritten = without_feedback_source(inspect.getsource(driver.main))
            transformed_sha = hashlib.sha256(rewritten.encode()).hexdigest()
            exec(compile(rewritten, str(source / 'run_qwen_sam2_guidance_v5_feedback.py'), 'exec'), driver.__dict__)
        sys.argv = [str(source / 'run_qwen_sam2_guidance_v5_feedback.py'), *common,
                    '--run-id', 'table2_pilot5_' + variant + '_20261025']
        driver.main()
    for planner in planner_instances:
        planner.assert_complete()
    statuses = {}
    for row in rows:
        annotation = json.loads((root / 'annotations' / variant / row['group'] / (row['uid'] + '.json')).read_text())
        statuses[row['uid']] = annotation.get('status')
        if annotation.get('status') == 'failed' or not (root / 'guidance' / variant / row['group'] / (row['uid'] + '.png')).is_file():
            raise ValueError('Variant failed for ' + row['uid'])
        if variant == 'box_point' and 'full foreground fallback' in json.dumps(annotation):
            raise ValueError('Box-point baseline used a whole-object fallback; not a valid measured result')
    receipt = {'variant': variant, 'controls': controls, 'statuses': statuses,
        'planning_reused': variant != 'ours', 'planning_calls_reused': sum(p.used for p in planner_instances),
        'source_hashes': frozen_sources, 'segvigen_run': False}
    (root / ('receipt_' + variant + '.json')).write_text(json.dumps(receipt, indent=2))
    print(json.dumps({'variant_complete': variant, 'objects': len(statuses), 'statuses': statuses}), flush=True)


if __name__ == '__main__':
    main()

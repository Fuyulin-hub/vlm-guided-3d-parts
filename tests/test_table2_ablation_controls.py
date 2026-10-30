import hashlib
import importlib.util
from pathlib import Path

import pytest


def controls():
    path = Path(__file__).resolve().parents[1] / 'scripts/table2_ablation_controls.py'
    assert path.is_file(), 'The approved ablation controls have not been implemented'
    spec = importlib.util.spec_from_file_location('table2_controls_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_positive_keeps_negative_and_box_without_mutating_plan():
    module = controls()
    part = {'bbox': [0, 0, 10, 10], 'positive_points': [[2, 2], [4, 4]], 'negative_points': [[8, 8]], 'label': 'leg'}
    changed = module.transform_part(part, 'single_positive')
    assert changed['positive_points'] == [[2, 2]]
    assert changed['negative_points'] == [[8, 8]] and changed['bbox'] == part['bbox']
    assert part['positive_points'] == [[2, 2], [4, 4]]


def test_no_negative_removes_feedback_added_negatives_but_not_positives():
    module = controls()
    corrected = {'positive_points': [[2, 2], [5, 5]], 'negative_points': [[8, 8], [9, 9]]}
    changed = module.transform_part(corrected, 'no_negative')
    assert changed['negative_points'] == []
    assert changed['positive_points'] == corrected['positive_points']


def test_no_feedback_skips_real_review_loop_and_retains_initial_masks():
    module = controls()
    source = '''def main():
    candidates = [{"part": "leg", "region": [1, 0]}]
    improved: list[dict] = []
    for candidate in candidates:
        reviews.append(candidate)
        improved.append({"part": candidate["part"], "region": [0, 1]})
    return improved
'''
    namespace = {'reviews': []}
    exec(module.without_feedback_source(source), namespace)
    assert namespace['main']() == [{'part': 'leg', 'region': [1, 0]}]
    assert namespace['reviews'] == []


def test_no_feedback_rejects_unrecognized_driver_instead_of_silently_running_review():
    with pytest.raises(ValueError, match='feedback loop'):
        controls().without_feedback_source('def main():\n    return []\n')


def test_each_ours_ablation_changes_only_one_control():
    module = controls()
    full = module.policy('ours')
    expected = {'single_positive': 'max_positive_points', 'no_negative': 'negative_points',
                'no_feedback': 'feedback', 'no_point_fusion': 'point_filter'}
    for variant, key in expected.items():
        assert {k for k, v in module.policy(variant).items() if full[k] != v} == {key}


def test_frozen_planning_reuses_only_plan_and_does_not_fake_feedback(tmp_path):
    module = controls()
    plan = tmp_path / 'plan.txt'
    plan.write_text('{"parts": []}', encoding='utf-8')
    calls = []
    def real_generator(*args, **kwargs):
        calls.append(args[2])
        return 'real-feedback'
    frozen = module.FrozenPlanner(real_generator, 'PLAN', [{'uid': 'one', 'path': str(plan),
        'sha256': hashlib.sha256(plan.read_bytes()).hexdigest()}])
    assert frozen(None, None, [{'content': [{'text': 'PLAN'}]}]) == '{"parts": []}'
    assert calls == []
    assert frozen(None, None, [{'content': [{'text': 'FEEDBACK'}]}]) == 'real-feedback'
    assert len(calls) == 1
    frozen.assert_complete()


def test_changed_frozen_plan_is_rejected(tmp_path):
    module = controls()
    plan = tmp_path / 'plan.txt'
    plan.write_text('changed', encoding='utf-8')
    frozen = module.FrozenPlanner(lambda *args: 'unused', 'PLAN', [{'uid': 'one', 'path': str(plan), 'sha256': 'bad'}])
    with pytest.raises(ValueError, match='hash'):
        frozen(None, None, [{'content': [{'text': 'PLAN'}]}])


def test_any_existing_output_is_rejected_before_plan_reuse(tmp_path):
    module = controls()
    existing = tmp_path / 'guidance/no_negative/assemblies/two.png'
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b'prior partial result')
    with pytest.raises(FileExistsError, match='new run directory'):
        module.require_fresh_variant(tmp_path, 'no_negative', [
            {'uid': 'one', 'group': 'assemblies'}, {'uid': 'two', 'group': 'assemblies'}])


def test_fresh_variant_does_not_create_or_modify_outputs(tmp_path):
    controls().require_fresh_variant(tmp_path, 'ours', [{'uid': 'one', 'group': 'assemblies'}])
    assert list(tmp_path.iterdir()) == []

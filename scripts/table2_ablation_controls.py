"""Small, auditable controls for the approved five-object 2D ablation pilot."""
import ast
import copy
import hashlib
from pathlib import Path


def policy(variant):
    result = {'max_positive_points': None, 'negative_points': True, 'feedback': True, 'point_filter': True}
    changes = {'ours': {}, 'single_positive': {'max_positive_points': 1},
               'no_negative': {'negative_points': False}, 'no_feedback': {'feedback': False},
               'no_point_fusion': {'point_filter': False},
               'box_point': {'max_positive_points': 1, 'negative_points': False, 'feedback': False, 'point_filter': False}}
    if variant not in changes:
        raise ValueError('Unknown ablation variant: ' + variant)
    result.update(changes[variant])
    return result


def transform_part(part, variant):
    result = copy.deepcopy(part)
    controls = policy(variant)
    if controls['max_positive_points'] is not None:
        result['positive_points'] = list(result.get('positive_points', []))[:controls['max_positive_points']]
    if not controls['negative_points']:
        result['negative_points'] = []
    return result


def require_fresh_variant(root, variant, rows):
    root = Path(root)
    for row in rows:
        group, uid = row['group'], row['uid']
        paths = [root / 'guidance' / variant / group / (uid + '.png'),
                 root / 'annotations' / variant / group / (uid + '.json'),
                 root / 'masks' / variant / group / uid]
        if any(path.exists() for path in paths):
            raise FileExistsError('Variant has existing outputs; use a new run directory to preserve object-plan identity: ' + uid)


def without_feedback_source(source):
    tree = ast.parse(source)
    assignments, loops = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == 'improved':
            node.value = ast.parse('[{"part": c["part"], "region": c["region"]} for c in candidates]', mode='eval').body
            assignments += 1
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'candidate' and isinstance(node.iter, ast.Name) and node.iter.id == 'candidates':
            node.iter = ast.List(elts=[], ctx=ast.Load())
            loops += 1
    if (assignments, loops) != (1, 1):
        raise ValueError('Unrecognized feedback loop in the frozen V5 driver')
    return ast.unparse(ast.fix_missing_locations(tree))


class FrozenPlanner:
    def __init__(self, original, prompt, plans):
        self.original = original
        self.prompt = prompt
        self.plans = plans
        self.used = 0

    def __call__(self, model, processor, messages, *args, **kwargs):
        is_plan = any(item.get('text') == self.prompt for message in messages for item in message.get('content', []))
        if not is_plan:
            return self.original(model, processor, messages, *args, **kwargs)
        if self.used >= len(self.plans):
            raise ValueError('More planning calls than frozen objects')
        record = self.plans[self.used]
        data = Path(record['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Frozen Qwen plan hash mismatch: ' + record['uid'])
        self.used += 1
        return data.decode('utf-8')

    def assert_complete(self):
        if self.used != len(self.plans):
            raise ValueError('Not every frozen Qwen plan was consumed exactly once')

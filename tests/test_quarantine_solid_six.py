import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / 'scripts/quarantine_solid_six_20261025.py'


def module():
    assert SCRIPT.exists(), 'Recoverable solid exclusion helper is not implemented'
    spec = importlib.util.spec_from_file_location('solid_exclusion', SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def fixture(root, uids):
    rows = []
    for uid in uids:
        guide = root / 'solid_volume' / (uid + '.png')
        label = root / 'references/solid_volume' / uid / 'labels_uint16.png'
        guide.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        guide.write_bytes(('guide-' + uid).encode())
        label.write_bytes(('label-' + uid).encode())
        (label.parent / 'label_mapping.json').write_text('{}')
        rows.append({'uid': uid, 'group': 'solid_volume', 'origin': 'fixture',
                     'guidance_file': str(guide.relative_to(root)).replace('\\', '/'),
                     'labels_file': str(label.relative_to(root)).replace('\\', '/'),
                     'guide_sha256': hashlib.sha256(guide.read_bytes()).hexdigest(),
                     'labels_sha256': hashlib.sha256(label.read_bytes()).hexdigest()})
    (root / 'manifest_solid_repaired_20261024.json').write_text(json.dumps({'total': len(rows), 'group': 'solid_volume', 'per_uid': rows}))
    (root / 'summary_solid_20261024.json').write_text(json.dumps({'count': len(rows), 'guide_count': len(rows), 'label_count': len(rows), 'mapping_count': len(rows)}))
    assembly = root / 'assemblies/untouched.png'
    assembly.parent.mkdir()
    assembly.write_bytes(b'preserve-me')


def test_exclusions_are_recoverable_and_update_active_manifest(tmp_path):
    m = module()
    root, quarantine = tmp_path / 'active', tmp_path / 'quarantine'
    uids = ['a' * 32, 'b' * 32]
    fixture(root, uids + ['c' * 32])
    plan = m.make_plan(root, quarantine, uids)
    assert (root / 'solid_volume' / (uids[0] + '.png')).exists()
    result = m.apply_plan(root, quarantine, plan)
    assert result['after_counts']['solid_guidance'] == 1
    assert result['after_counts']['solid_labels'] == 1
    assert result['assemblies_unchanged'] is True
    manifest = json.loads((root / 'manifest_solid_repaired_20261024.json').read_text())
    assert manifest['total'] == 1
    assert [r['uid'] for r in manifest['per_uid']] == ['c' * 32]
    for uid in uids:
        assert not (root / 'solid_volume' / (uid + '.png')).exists()
        assert (quarantine / 'solid_volume' / (uid + '.png')).exists()
        assert (quarantine / 'references/solid_volume' / uid / 'labels_uint16.png').exists()


def test_plan_rejects_destination_collision_without_moving_sources(tmp_path):
    m = module()
    root, quarantine = tmp_path / 'active', tmp_path / 'quarantine'
    uid = 'a' * 32
    fixture(root, [uid])
    target = quarantine / 'solid_volume' / (uid + '.png')
    target.parent.mkdir(parents=True)
    target.write_bytes(b'not-the-same')
    with pytest.raises(ValueError, match='destination'):
        m.make_plan(root, quarantine, [uid])
    assert (root / 'solid_volume' / (uid + '.png')).exists()


def test_changed_plan_input_is_rejected_before_any_move(tmp_path):
    m = module()
    root, quarantine = tmp_path / 'active', tmp_path / 'quarantine'
    uid = 'a' * 32
    fixture(root, [uid])
    plan = m.make_plan(root, quarantine, [uid])
    p = root / 'solid_volume' / (uid + '.png')
    p.write_bytes(b'changed-after-plan')
    with pytest.raises(ValueError, match='changed'):
        m.apply_plan(root, quarantine, plan)
    assert p.exists()

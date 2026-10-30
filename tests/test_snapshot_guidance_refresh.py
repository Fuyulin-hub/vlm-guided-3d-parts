import importlib.util
from pathlib import Path


def test_csv_guidance_hash_is_not_misread_as_a_label_hash():
    p = Path(__file__).parents[1] / 'scripts/snapshot_guidance_refresh_20261025.py'
    s = importlib.util.spec_from_file_location('snapshot_refresh', p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    assert m.declared_label_hash({'sha256': 'guidance-only-hash'}) is None
    assert m.declared_label_hash({'sha256': {'labels': 'label-hash'}}) == 'label-hash'
    assert m.declared_label_hash({'labels_sha256': 'label-hash'}) == 'label-hash'

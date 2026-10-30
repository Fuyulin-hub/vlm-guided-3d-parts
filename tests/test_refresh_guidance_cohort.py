import importlib.util
from pathlib import Path

import numpy as np


def test_server_b_preparation_is_exactly_reversed_by_the_frozen_evaluator():
    path = Path(__file__).parents[1] / 'scripts/rerun_guidance_refresh_20261025.py'
    assert path.is_file(), 'Updated-cohort scoring driver is not implemented'
    spec = importlib.util.spec_from_file_location('refresh_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pixels = np.arange(3 * 5 * 4, dtype=np.uint8).reshape(3, 5, 4)
    prepared = module.server_b_to_legacy_frame(pixels)
    assert np.array_equal(np.rot90(prepared, k=1), pixels)

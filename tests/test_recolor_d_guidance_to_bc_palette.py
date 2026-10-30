"""B/C-palette recoloring must preserve D-arm spatial part labels."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "recolor_d_guidance_to_bc_palette.py"


def load_module():
    spec = importlib.util.spec_from_file_location("recolor_d_guidance_to_bc_palette", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class RecolorDGuidanceTests(unittest.TestCase):
    def test_two_parts_use_exact_distinct_bc_colors_and_transparent_background(self):
        module = load_module()
        labels = np.array([[0, 7, 8]], dtype=np.uint16)
        rgba, mapping = module.recolor_labels(
            labels,
            {7: (229, 81, 22), 8: (22, 81, 229)},
            [(229, 81, 22), (22, 81, 229)],
        )
        self.assertEqual(rgba.tolist(), [[[0, 0, 0, 0], [229, 81, 22, 255], [22, 81, 229, 255]]])
        self.assertNotEqual(mapping[7], mapping[8])
        self.assertEqual(labels.tolist(), [[0, 7, 8]])

    def test_overflow_reuses_color_only_for_spatially_separated_parts(self):
        module = load_module()
        labels = np.array([[1, 0, 0, 0, 0, 0, 3, 2]], dtype=np.uint16)
        rgba, mapping = module.recolor_labels(
            labels,
            {1: (229, 81, 22), 2: (22, 81, 229), 3: (229, 81, 22)},
            [(229, 81, 22), (22, 81, 229)],
            min_gap_px=5,
        )
        self.assertEqual(mapping[1], mapping[3])
        self.assertNotEqual(mapping[2], mapping[3])
        self.assertEqual(rgba[0, 0].tolist(), rgba[0, 6].tolist())

    def test_overflow_rejects_reuse_for_touching_parts(self):
        module = load_module()
        labels = np.array([[1, 2, 3]], dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "no spatially separated color"):
            module.recolor_labels(
                labels,
                {1: (229, 81, 22), 2: (22, 81, 229), 3: (229, 81, 22)},
                [(229, 81, 22), (22, 81, 229)],
                min_gap_px=5,
            )

    def test_batch_outputs_recolored_guide_and_unchanged_integer_labels(self):
        module = load_module()
        with TemporaryDirectory(dir=SCRIPT.parents[1]) as temporary:
            root = Path(temporary)
            uid = "0" * 32
            src = root / "src" / "references" / "assemblies" / uid
            src.mkdir(parents=True)
            Image.fromarray(np.array([[0, 1]], dtype=np.uint16)).save(src / "labels_uint16.png")
            old_palette = root / "old.json"
            old_palette.write_text(json.dumps({"colors": [{"label_id": 1, "rgb": [229, 81, 22]}]}), encoding="utf-8")
            bc_palette = root / "bc.json"
            bc_palette.write_text(
                json.dumps([[1, 1, 1, 255], [229, 81, 22, 255]]),
                encoding="utf-8",
            )
            result = module.recolor_directory(root / "src", old_palette, bc_palette, root / "out")
            guide = root / "out" / "guidance" / "D_human_fixed_bc20" / "assemblies" / f"{uid}.png"
            with Image.open(guide) as image:
                self.assertEqual(np.asarray(image).tolist(), [[[0, 0, 0, 0], [229, 81, 22, 255]]])
            copied = root / "out" / "references" / "assemblies" / uid / "labels_uint16.png"
            self.assertEqual(copied.read_bytes(), (src / "labels_uint16.png").read_bytes())
            self.assertEqual(result["recolored_count"], 1)

    def test_batch_uses_requested_group_directory(self):
        module = load_module()
        with TemporaryDirectory(dir=SCRIPT.parents[1]) as temporary:
            root = Path(temporary)
            uid = "2" * 32
            src = root / "src" / "references" / "solid_volume" / uid
            src.mkdir(parents=True)
            Image.fromarray(np.array([[0, 1]], dtype=np.uint16)).save(src / "labels_uint16.png")
            old_palette = root / "old.json"
            old_palette.write_text(
                json.dumps({"colors": [{"label_id": 1, "rgb": [229, 81, 22]}]}),
                encoding="utf-8",
            )
            bc_palette = root / "bc.json"
            bc_palette.write_text(json.dumps([[1, 1, 1], [229, 81, 22]]), encoding="utf-8")

            receipt = module.recolor_directory(
                root / "src", old_palette, bc_palette, root / "out", group="solid_volume"
            )

            guide = root / "out" / "guidance" / "D_human_fixed_bc20" / "solid_volume" / f"{uid}.png"
            label_copy = root / "out" / "references" / "solid_volume" / uid / "labels_uint16.png"
            self.assertTrue(guide.exists())
            self.assertEqual(label_copy.read_bytes(), (src / "labels_uint16.png").read_bytes())
            self.assertEqual(receipt["group"], "solid_volume")


if __name__ == "__main__":
    unittest.main()

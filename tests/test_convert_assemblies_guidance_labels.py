"""Exact conversion of discrete D-arm guidance colors into integer labels."""

from pathlib import Path
import importlib.util
import io
import json
import unittest
import zipfile

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_assemblies_guidance_labels.py"


def load_module():
    spec = importlib.util.spec_from_file_location("convert_assemblies_guidance_labels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ConvertAssembliesGuidanceLabelsTests(unittest.TestCase):
    def test_exact_rgba_mapping_keeps_palette_ids_and_transparent_background(self):
        converter = load_module()
        rgba = np.array(
            [[[7, 9, 11, 0], [229, 81, 22, 255], [22, 81, 229, 255]]],
            dtype=np.uint8,
        )
        labels = converter.rgba_to_labels(rgba, {(229, 81, 22): 1, (22, 81, 229): 2})
        self.assertEqual(labels.dtype, np.uint16)
        self.assertEqual(labels.tolist(), [[0, 1, 2]])

    def test_unknown_opaque_color_is_rejected_not_guessed(self):
        converter = load_module()
        rgba = np.array([[[229, 80, 22, 255]]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "off-palette"):
            converter.rgba_to_labels(rgba, {(229, 81, 22): 1})

    def test_partial_alpha_is_rejected(self):
        converter = load_module()
        rgba = np.array([[[229, 81, 22, 128]]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "alpha"):
            converter.rgba_to_labels(rgba, {(229, 81, 22): 1})

    def test_archive_conversion_saves_lossless_uint16_png_and_receipt(self):
        converter = load_module()
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(dir=SCRIPT.parents[1]) as temporary:
            tmp_path = Path(temporary)
            uid = "0" * 32
            palette_path = tmp_path / "palette.json"
            palette_path.write_text(
                json.dumps({"colors": [{"label_id": 7, "rgb": [229, 81, 22]}]}),
                encoding="utf-8",
            )
            rgba = np.array(
                [[[0, 0, 0, 0], [229, 81, 22, 255]]], dtype=np.uint8
            )
            image_stream = io.BytesIO()
            Image.fromarray(rgba, "RGBA").save(image_stream, format="PNG")
            archive_path = tmp_path / "guidance.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"assemblies/{uid}.png", image_stream.getvalue())

            converter.convert_archive(archive_path, palette_path, tmp_path / "out")

            label_path = tmp_path / "out" / "references" / "assemblies" / uid / "labels_uint16.png"
            with Image.open(label_path) as label_image:
                self.assertIn(label_image.mode, {"I;16", "I"})
                self.assertEqual(np.asarray(label_image).tolist(), [[0, 7]])
            preview_path = label_path.with_name("preview_color.png")
            with Image.open(preview_path) as preview_image:
                self.assertEqual(preview_image.mode, "RGBA")
                self.assertEqual(np.asarray(preview_image).tolist(), rgba.tolist())
            receipt = json.loads((tmp_path / "out" / "conversion_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["converted_count"], 1)
            self.assertEqual(receipt["per_uid"][uid]["label_ids"], [7])

    def test_duplicate_uid_in_archive_is_rejected(self):
        converter = load_module()
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(dir=SCRIPT.parents[1]) as temporary:
            uid = "a" * 32
            archive_path = Path(temporary) / "duplicate.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"first/{uid}.png", b"first")
                archive.writestr(f"second/{uid}.png", b"second")
            with self.assertRaisesRegex(ValueError, "duplicate UID"):
                converter.validate_archive_entries(archive_path)

    def test_archive_conversion_uses_requested_group_directory(self):
        converter = load_module()
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(dir=SCRIPT.parents[1]) as temporary:
            root = Path(temporary)
            uid = "1" * 32
            palette = root / "palette.json"
            palette.write_text(
                json.dumps({"colors": [{"label_id": 2, "rgb": [229, 81, 22]}]}),
                encoding="utf-8",
            )
            image_bytes = io.BytesIO()
            Image.fromarray(np.array([[[229, 81, 22, 255]]], dtype=np.uint8)).save(
                image_bytes, format="PNG"
            )
            archive_path = root / "thin.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"thin_fine_grained/{uid}.png", image_bytes.getvalue())

            receipt = converter.convert_archive(
                archive_path, palette, root / "out", group="thin_fine_grained"
            )

            label_file = root / "out" / "references" / "thin_fine_grained" / uid / "labels_uint16.png"
            self.assertTrue(label_file.exists())
            self.assertEqual(receipt["group"], "thin_fine_grained")

    def test_paired_archive_decodes_only_exact_human_png_and_reports_rejections(self):
        converter = load_module()
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(dir=SCRIPT.parents[1]) as temporary:
            root = Path(temporary)
            good_uid, bad_uid = "a" * 32, "b" * 32
            palette_path = root / "palette.json"
            palette_path.write_text(
                json.dumps({"colors": [{"label_id": 3, "rgb": [229, 81, 22]}]}),
                encoding="utf-8",
            )

            def png(alpha):
                stream = io.BytesIO()
                pixels = np.zeros((2, 2, 4), dtype=np.uint8)
                pixels[0, 0] = [229, 81, 22, alpha]
                Image.fromarray(pixels, "RGBA").save(stream, format="PNG")
                return stream.getvalue()

            archive_path = root / "pairs.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for uid, alpha in [(good_uid, 255), (bad_uid, 128)]:
                    archive.writestr(f"solid/{uid}.png", b"raw photo unused")
                    archive.writestr(f"solid/{uid}_hum.png", png(alpha))

            receipt = converter.convert_paired_archives(
                [archive_path], palette_path, root / "out", group="solid_volume", required_size=(2, 2)
            )

            label_file = root / "out" / "references" / "solid_volume" / good_uid / "labels_uint16.png"
            with Image.open(label_file) as image:
                self.assertEqual(np.asarray(image).tolist(), [[3, 0], [0, 0]])
            self.assertFalse((root / "out" / "references" / "solid_volume" / bad_uid).exists())
            self.assertEqual(receipt["converted_count"], 1)
            self.assertEqual(receipt["rejected_count"], 1)
            self.assertIn("partial alpha", receipt["rejected"][bad_uid]["reason"])

    def test_paired_archives_reject_uid_overlap_before_writing_output(self):
        converter = load_module()
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(dir=SCRIPT.parents[1]) as temporary:
            root = Path(temporary)
            uid = "c" * 32
            archives = [root / "first.zip", root / "second.zip"]
            for archive_path in archives:
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(f"solid/{uid}.png", b"raw")
                    archive.writestr(f"solid/{uid}_hum.png", b"human")
            with self.assertRaisesRegex(ValueError, "duplicate UID"):
                converter.validate_paired_archives(archives)

"""Pair anonymous 2D guidance results by category and UID, never by score."""

import importlib.util
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts/run_guidance_category_comparison.py"


def runner():
    assert SCRIPT.is_file(), "Category comparison runner is not implemented"
    spec = importlib.util.spec_from_file_location("category_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(uid, value):
    return {"uid": uid, "metrics": dict.fromkeys(
        ("matched_miou", "boundary_f1", "part_f1"), value)}


def test_pairing_uses_same_uids_and_paired_deltas():
    module = runner()
    result = module.paired_summary({"B_sam2": [row("a", 0.2), row("b", 0.9)],
                                    "C_v5_full": [row("a", 0.6), row("c", 0.1)]},
                                   {"a", "b", "c"})
    assert result["paired_uids"] == ["a"]
    assert result["methods"]["B_sam2"]["matched_miou"]["mean"] == 0.2
    assert result["methods"]["C_v5_full"]["matched_miou"]["mean"] == 0.6
    assert result["delta_C_minus_B"]["matched_miou"]["mean"] == pytest.approx(0.4)
    assert result["missing_by_method"] == {"B_sam2": ["c"], "C_v5_full": ["b"]}


def test_pairing_rejects_duplicate_records_and_handles_no_pairs():
    module = runner()
    with pytest.raises(ValueError, match="Duplicate"):
        module.paired_summary({"B_sam2": [row("a", 0), row("a", 1)], "C_v5_full": []}, {"a"})
    result = module.paired_summary({"B_sam2": [row("a", 0.2)], "C_v5_full": []}, {"a"})
    assert result["n_paired"] == 0
    assert result["methods"] == {"B_sam2": {}, "C_v5_full": {}}
    assert result["delta_C_minus_B"] == {}


def test_finalizer_validates_fresh_sources_before_aggregation(tmp_path):
    module = runner()
    out, croot = tmp_path / "out", tmp_path / "C"
    core = out / "inputs/evaluation_code_clean/paper_table_metrics.py"
    core.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT.with_name("paper_table_metrics.py"), core)
    palette = tmp_path / "palette.json"
    palette.write_text(json.dumps([[229, 22, 22, 255], [229, 81, 22, 255]]))
    frozen = []
    for group in module.GROUPS:
        refpath = tmp_path / "D" / group / "labels.png"
        refpath.parent.mkdir(parents=True)
        ref = np.array([[0, 0, 0], [0, 9, 9], [0, 0, 0]], np.uint16)
        Image.fromarray(ref).save(refpath)
        entry = {"uid": "same", "labels_file": str(refpath),
                 "labels_sha256": module.direct.digest(refpath), "origin": "test",
                 "reference_subset": "existing_63"}
        manifest = out / "inputs" / f"{group}_references.json"
        module.direct.save_json(manifest, {"group": group, "per_uid": [entry]})
        frozen.append({"path": str(refpath), "sha256": module.direct.digest(refpath)})
        rgba = np.zeros((3, 3, 4), np.uint8)
        rgba[ref > 0] = [229, 81, 22, 255]
        for method in module.METHODS:
            rotation = 1 if method == "B_sam2" else 0
            predroot = (out / "inputs/B_original" if rotation else croot) / group
            predroot.mkdir(parents=True)
            path = predroot / "same.png"
            Image.fromarray(np.rot90(rgba, k=-rotation)).save(path)
            frozen.append({"path": str(path), "sha256": module.direct.digest(path)})
            module.direct.evaluate(predroot, tmp_path, manifest, palette, out / method / group,
                                   progress=False, method=method, rotation_k=rotation, metrics_path=core)
    module.direct.save_json(out / "frozen_inputs.json", frozen)
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as handle:
        handle.write(core, "paper_table_metrics.py")
    result = module.finalize(out, archive, archive, croot)
    assert result["source_files_unchanged"] is True
    assert result["complete_on_available_pairs"] is True
    assert {g: x["n_paired"] for g, x in result["groups"].items()} == dict.fromkeys(module.GROUPS, 1)
    assert result["groups"]["assemblies"]["methods"]["C_v5_full"]["matched_miou"]["mean"] == 1
    Image.fromarray(np.zeros((3, 3, 4), np.uint8)).save(croot / "assemblies/same.png")
    with pytest.raises(ValueError, match="Result source changed"):
        module.finalize(out, archive, archive, croot)

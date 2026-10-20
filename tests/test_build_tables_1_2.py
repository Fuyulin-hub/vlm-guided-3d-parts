"""Regression checks for paper-table evidence discipline."""

from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_tables_1_2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_tables_1_2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_c_v4_never_uses_c_v3_metrics_as_a_proxy():
    """A final-method row must remain TBD until direct C_v4 evidence exists."""
    tables = load_module()
    samples = {
        "C_v3_box_point": {
            ("assemblies", "sample_1"): {
                "mean_iou": 0.5,
                "mean_boundary_f1": 0.5,
                "missing_rate": 0.1,
                "matched_parts": 1,
                "total_ref_parts": 1,
                "total_pred_parts": 1,
            }
        }
    }

    result = tables.compute_group_metrics_2d(
        "C_v4_exhaustive_box_point",
        "assemblies",
        samples,
        {("assemblies", "sample_1")},
        {"assemblies": ["sample_1"]},
    )

    assert result["mask_iou"] == tables.TBD
    assert result["n_samples"] == 0


def test_macro_average_requires_all_three_groups_to_be_measured():
    """An assemblies-only number must not be printed as a three-group macro average."""
    tables = load_module()
    arm_data = {
        "assemblies": {"n_samples": 203, "mask_iou": 0.5, "boundary_f1": 0.4,
                       "part_recall": 0.6, "valid_part_rate": 0.5},
        "solid_volume": {"n_samples": 0, "mask_iou": tables.TBD, "boundary_f1": tables.TBD,
                         "part_recall": tables.TBD, "valid_part_rate": tables.TBD},
        "thin_fine_grained": {"n_samples": 0, "mask_iou": tables.TBD, "boundary_f1": tables.TBD,
                               "part_recall": tables.TBD, "valid_part_rate": tables.TBD},
    }

    macro = tables.compute_macro_avg_2d(
        arm_data, ["assemblies", "solid_volume", "thin_fine_grained"]
    )

    assert macro["mask_iou"] == tables.TBD
    assert macro["n_samples"] == 0


def test_table2_export_never_mentions_a_c_v3_proxy(tmp_path):
    """The paper-facing export must name missing C_v4 evidence rather than a proxy."""
    tables = load_module()
    tables.write_table2_md({arm: {} for arm in tables.ARMS}, str(tmp_path))
    text = (tmp_path / "table2_main_abcd.md").read_text(encoding="utf-8")

    assert "C_v3" not in text
    assert "proxy" not in text.lower()
    assert "interim" in text.lower()

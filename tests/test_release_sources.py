"""Release integrity and a configuration-only CLI smoke test."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "reproducibility" / "table2_pilot5"


def test_frozen_pilot_sources_match_original_manifest():
    expected = json.loads((PILOT / "source_hashes.json").read_text(encoding="utf-8"))
    assert {p.name for p in (PILOT / "source").glob("*.py")} == set(expected)
    for name, digest in expected.items():
        assert hashlib.sha256((PILOT / "source" / name).read_bytes()).hexdigest() == digest, name


def test_active_v4_helper_is_the_verified_original():
    filename = "run_qwen_sam2_guidance_v4.py"
    assert (ROOT / "scripts" / filename).read_bytes() == (PILOT / "source" / filename).read_bytes()


def test_guidance_cli_freezes_config_without_gpu_assets(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("group,uid,input_rgba\nassemblies,synthetic-test,unused.png\n", encoding="utf-8")
    run_root = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_qwen_sam2_guidance_v5_feedback.py"),
         "--experiment-root", str(run_root), "--manifest", str(manifest),
         "--run-id", "smoke", "--output-suffix", "C_check", "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads((run_root / "logs/C_check/smoke_config.json").read_text(encoding="utf-8"))
    assert config["manifest"]["explicit_uids"] == ["synthetic-test"]
    expected = json.loads((PILOT / "source_hashes.json").read_text(encoding="utf-8"))
    assert config["provenance"]["v4_helper_sha256"] == expected["run_qwen_sam2_guidance_v4.py"]
    assert not (run_root / "masks").exists()
    assert not (run_root / "guidance").exists()

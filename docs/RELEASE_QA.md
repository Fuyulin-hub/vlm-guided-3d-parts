# Source packaging checks — 2026-11-01

Verified on the local Windows host with Python 3.12 and Node.js. No experiment server was contacted, no model inference was run, and no scientific scores were recomputed during packaging.

| Check | Result |
|---|---|
| Python CPU suite | 128 passed, 1 EGL test deselected |
| Viewer unit suite | 55 passed, 0 failed |
| Original pilot source manifest | All 11 files match their original SHA-256 |
| Restored V4 helper | Byte-identical to the frozen pilot helper |
| Guidance configuration CLI | Passes without image/checkpoint/GPU assets; source hashes recorded |
| Windows atomic PNG writes | Three existing regression tests pass after writable-descriptor fsync fix |
| Clean Git export | Tests run against exported tracked files; annotation palettes included under configs/palettes |

Commands:

```bash
python -m pytest tests -k "not egl_flat_render" -q
node --test work/c-viewer/provenance.test.mjs work/c-viewer/semantic-core.test.mjs work/c-viewer/semantic-three.test.mjs
```

The local test environment used existing NumPy/SciPy/Pillow/OpenCV packages plus an isolated pytest installation. This is a source check, not an inference-environment lockfile. The GitHub CPU workflow defines Linux/Windows checks but has not been claimed to pass before its remote results are available.

Not rerun here: actual EGL rendering, browser integration, GPU model inference, server camera acceptance, full-cohort evaluation, and manuscript compilation. The manuscript retains its separately dated integration/compilation QA; packaging checks that its referenced figures are present.

The Git export was inspected for high-confidence credential patterns and unintended experiment assets; no matches were found. All nine referenced manuscript figures are present and the eleven frozen source hashes match after export. Dataset files, weights, run logs, credentials, local archives, and duplicate submission packages are excluded. Existing scientific metadata gaps remain documented in the manuscript.

# VLM-Guided 3D Parts

Research workspace for visual-language-guided 2D spatial prompting, 3D part segmentation, and English part annotations for generated 3D assets.

This repository contains the guidance and evaluation code, a local GLB viewer, tests, and the current IJCV English manuscript. The earlier Chinese draft is retained. Unmeasured table entries remain marked TBD.

## Contents

| Directory | Purpose |
| --- | --- |
| `scripts/` | Guidance prototype, GLB projection, label metrics, table evaluators, and annotation conversion |
| `tests/` | CPU checks for metrics, file adapters, projection, and guidance postprocessing |
| `docs/evaluation/` | Evaluation protocol and configuration templates |
| `work/c-viewer/` | Local Three.js viewer and semantic color correspondence tools |
| `manuscript/ijcv_english/` | Current Springer Nature LaTeX manuscript, PDF, bibliography, and figure inputs |
| `manuscript/` | Earlier Chinese draft and author-supplied template files |

Datasets, model weights, generated GLB/VXZ files, raw run logs, credentials, local caches, and downloaded experiment assets are excluded from Git. Paths in experiment-specific configurations need to be adapted to the machine running the experiments.

## Annotation format

English part names use `label`; numeric region identities use `label_id`:

```json
{
  "parts": [
    {
      "part_id": "part_05",
      "label_id": 4,
      "label": "Front wheel",
      "color": [199, 229, 22]
    }
  ]
}
```

The viewer accepts legacy `label_en` inputs, normalizes them to `label`, and removes Chinese-name fields from exported part records. Name conversion preserves label IDs and does not rerun segmentation.

```bash
python scripts/english_only_annotations.py \
  --input /path/to/original/annotations \
  --output /path/to/english/annotations
```

Use a separate output directory. If a provenance record declares the annotation file checksum, retain the transformation history and update that checksum for the exported JSON.

## Evaluate existing results

```bash
python -m pip install -r scripts/requirements-evaluation.txt
python scripts/eval_table2.py \
  --config docs/evaluation/config.example.json \
  --root /path/to/experiments \
  --out /path/to/evaluation-output
```

Configure the manifest and label sources first. The evaluators read existing outputs and do not launch model inference. See `docs/evaluation/README.md` for metrics, missing-data handling, camera requirements, and the other table entry points.

## Guidance prototype dependencies

`scripts/run_qwen_sam2_guidance_v5_feedback.py` contains the V5 guidance and fusion driver. Full inference additionally requires the original `run_qwen_sam2_guidance_v4.py` helper, Qwen3-VL and SAM2 installations and checkpoints, OpenCV, and a configured CUDA environment. The helper and pretrained weights are not included in this snapshot. The annotation output format was updated after the currently running experiment; preserve each run's original source and configuration hashes when interpreting results.

## Local viewer

```bash
cd work/c-viewer
node serve.cjs
```

Open `http://127.0.0.1:4173/` and load a GLB with its corresponding annotations and provenance record. Bundled Three.js sources include their MIT license. The real firetruck asset is excluded from Git, so the optional sample button requires restoring the local asset bundle first. Synthetic tests remain available.

The viewer builds conservative color correspondences. Its color-match coverage is not a segmentation accuracy score; unresolved regions remain unassigned. See the viewer README for limitations and export details.

## Tests

```bash
python -m pip install pytest
python -m pytest tests/test_paper_table_metrics.py tests/test_english_only_annotations.py
```

Additional guidance checks require OpenCV. Viewer unit tests use Node.js; browser integration checks additionally require Playwright and a browser. Their current local browser paths need to be adjusted on other machines.

Validation on the Windows packaging host (2026-10-20): 55 viewer unit tests passed; the full Python suite had 70 passes and 4 failures. Three projection tests failed at `os.fsync` on a read-only temporary file descriptor; the EGL rendering test lacked `trimesh`. Projection additionally requires `trimesh`, `pyrender`/OpenGL, and the intended Linux/EGL environment (the runner uses `fcntl`). These limitations are retained in this source snapshot, not reported as a fully passing suite. GPU inference and browser integration were not rerun for this packaging check.

## Manuscript

The current English source is [manuscript/ijcv_english/main.tex](manuscript/ijcv_english/main.tex), with the compiled [PDF](manuscript/ijcv_english/main.pdf). Select XeLaTeX when compiling it. The project includes the Springer Nature class and bibliography style plus all four manuscript figure PDFs.

The latest revision updates the three supplied external-baseline summaries and includes six A/B/C/D comparisons rendered from 24 real GLB outputs in Figure 4. The chair and rod use the display orientations approved by the author. The manuscript states the evaluation coverage and aggregation for each comparison. Missing measurements and author declarations remain explicit. See [the English manuscript README](manuscript/ijcv_english/README.md) for build details and [QA notes](manuscript/ijcv_english/QA.md) for the current checks.

The earlier Chinese working draft remains at `manuscript/main.tex`; its instructions are in `manuscript/README.md`.

## Version scope

This initial repository captures the local working source. It is not a verified snapshot of every script currently running on the server. Preserve server run IDs, actual source hashes, model configurations, and input identities separately before assigning any commit to a completed experiment.

# VLM-Guided 3D Parts

Research code for language-guided 2D part masks, their use as guidance for a frozen 3D segmentation backend, and evaluation of the resulting projections.

Qwen3-VL proposes part names, boxes, and points; SAM2 predicts masks; a part-level audit permits at most one retry. Point-aware assembly combines selected masks into integer labels and a colored guidance image. The downstream backend used in the experiments is SegViGen.

> **Public source release.** This repository contains code, small configuration assets, synthetic tests, and a frozen five-object ablation snapshot. It intentionally excludes data, model weights, rendered outputs, per-object predictions, server manifests, credentials, and private experiment logs. Read the [code-release scope](docs/CODE_RELEASE.md) before attempting a reproduction.

## Repository map

| Location | Contents |
|---|---|
| [scripts/](scripts/README.md) | Guidance, fusion, annotation conversion, projection, and metrics |
| [docs/GUIDANCE.md](docs/GUIDANCE.md) | Input format, configuration dry run, and GPU command |
| [docs/evaluation/](docs/evaluation/README.md) | Metrics, camera requirements, and aggregation |
| [reproducibility/table2_pilot5/](reproducibility/table2_pilot5/README.md) | Original five-object ablation source and SHA-256 hashes |
| [tests/](tests/) | Synthetic CPU checks and a separate EGL rendering test |
| [work/c-viewer/](work/c-viewer/) | Local Three.js viewer and semantic color correspondence |
| [manuscript/](manuscript/) | Current manuscript source (IJCV template) with figures and a compiled PDF |

This is a source release, not a complete archive of server runs. The dataset, weights, per-object predictions, evaluation manifests, and frozen camera snapshots are not distributed here. Historical paths must be replaced explicitly. See [reproducibility scope](docs/REPRODUCIBILITY.md) before attributing a paper result to this checkout.

## Quick start: CPU checks

Use Python 3.10 or later in a separate environment:

~~~bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-test.txt
python -m pytest tests -k "not egl_flat_render" -q
~~~

These checks do not load models or produce experimental scores. The excluded EGL test requires a working Linux/OpenGL environment. GPU inference and full-dataset reproduction are separate validation tasks.

## Generate 2D guidance

The maintained entry point is `scripts/run_qwen_sam2_guidance_v5_feedback.py`. Its original V4 helper is included.

~~~bash
python -m pip install -r scripts/requirements-guidance-cpu.txt
python scripts/run_qwen_sam2_guidance_v5_feedback.py \
  --experiment-root /path/to/new/run \
  --manifest /path/to/manifest.csv \
  --run-id config-check --output-suffix C_check --dry-run
~~~

The manifest contains `group`, `uid`, and `input_rgba`. A dry run checks selection and freezes source/configuration hashes; it does not validate every image or checkpoint. Full inference needs Qwen3-VL, SAM2, local checkpoints, a palette, and a compatible CUDA environment. See [GUIDANCE.md](docs/GUIDANCE.md). This driver does not launch SegViGen.

## Score existing label maps

~~~bash
python -m pip install -r scripts/requirements-evaluation.txt
python scripts/eval_table2.py \
  --config /path/to/evaluation.json \
  --root /path/to/experiment \
  --out /path/to/new/evaluation-output
~~~

Copy and adapt [the configuration template](docs/evaluation/config.example.json) first. Set `boundary_tolerance_px` explicitly: current primary manuscript columns use **5 px**, while archived and sensitivity analyses also contain **2 px**. The template retains its historical 2 px default.

The metrics module reports matched mIoU, Boundary F1, and Part F1@0.5, with GT-best IoU as a separate diagnostic. The older table CLI exports group-macro summaries and also stores object means in JSON; its filenames do not correspond to the current manuscript table numbering. See [evaluation instructions](docs/evaluation/README.md).

## Local viewer

With Node.js installed:

~~~bash
cd work/c-viewer
npm test
npm start
~~~

Open `http://127.0.0.1:4173/` and load your GLB, annotations, and provenance record. Unit tests use synthetic assets. `npm run test:browser` runs a separate integration suite whose Playwright/browser paths need configuration.

English part names use `label`; region identities use `label_id`. Colors encode local partitions, not a cross-object semantic vocabulary. Viewer color coverage is not segmentation accuracy.

## Evaluation boundaries

When the scoring reference is also supplied as D-arm guidance, D measures reference-conditioned consistency, not performance against an input-independent ground truth. Projected scores evaluate the observed view, not full-surface 3D segmentation accuracy. Sample counts and aggregation must be read from the evaluation manifest for the specific run.

The manuscript directory holds the current source and is not required to run the source release. Earlier drafts, submission archives, and superseded journal templates were removed on 2026-11-09; they remain in the repository history.

[RELEASE_QA.md](docs/RELEASE_QA.md) records checks run for this revision. [THIRD_PARTY.md](docs/THIRD_PARTY.md) describes upstream components.

## License

This repository is released under the Apache License 2.0 (see [LICENSE](LICENSE)). Third-party components keep their own licenses, listed in [THIRD_PARTY.md](docs/THIRD_PARTY.md). The manuscript text and figures are distributed for review and citation; if you reuse them, cite the paper.

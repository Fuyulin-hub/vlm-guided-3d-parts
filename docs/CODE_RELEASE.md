# Public code-release scope

This repository is a source release for the VLM-guided 3D part-segmentation pipeline. It is intended to make the implementation and its CPU-level validation inspectable without redistributing restricted assets.

## Included

- The maintained Qwen3-VL and SAM2 guidance driver, its one-pass audit logic, and point-aware assembly.
- Projection and integer-label evaluation utilities.
- Palette configurations, synthetic regression tests, and a hash-verified five-object ablation source snapshot.
- A local Three.js viewer with provenance and semantic-color checks.
- Documentation describing required manifests, camera constraints, and result-aggregation boundaries.

## Not included

- Objaverse assets, rendered source images, human-reference annotations, or the approved object rosters.
- Qwen3-VL, SAM2, SegViGen, TRELLIS, or any other upstream model weights.
- Experiment-server paths, credentials, checkpoints, run logs, frozen camera snapshots, prediction GLBs, or per-object scores.
- Any claim that a fresh checkout alone reproduces a reported full-cohort result.

## Validate a checkout

Create an isolated Python environment, install the CPU test requirements, and run:

```bash
python -m pytest tests -k "not egl_flat_render" -q
```

With Node.js installed, the viewer unit suite is:

```bash
cd work/c-viewer
npm test
```

The EGL rendering test, browser integration test, GPU inference, model installation, and full-cohort scoring require external environments and assets. They are intentionally not invoked by the default checks.

## Configuration discipline

Start new runs in a new experiment root and explicitly provide manifests, output locations, model paths, and palette paths. Do not infer an approved cohort from available files or replace a missing artifact with a same-UID artifact from another run. Historical operation scripts remain in `scripts/` for traceability and can contain dated absolute paths; they are not a one-command reproduction recipe.

No redistribution license for project data or third-party components is asserted by this release. Consult the upstream licenses listed in [THIRD_PARTY.md](THIRD_PARTY.md) before obtaining or using any external dependency.

# Evaluation of existing predictions

This is the project's evaluator, not an official SegViGen benchmark implementation. It reads existing labels, cameras, and optional event logs. It does not launch models or fit cameras.

## Configure and run

~~~bash
python -m pip install -r scripts/requirements-evaluation.txt
python scripts/eval_table2.py \
  --config /path/to/evaluation.json \
  --root /path/to/experiment \
  --out /path/to/new/report-directory
~~~

Adapt `config.example.json` first. The manifest needs unique group/UID pairs and the configured group names. This example and `config.objaverse684.json` are historical templates, not the publication roster. A requested output directory may have its reports overwritten; use a new directory.

| Legacy CLI | Purpose |
|---|---|
| `eval_table1.py` | Projection quality and optional human ratings |
| `eval_table2.py` | 2D guidance quality and projected mIoU for variants |
| `eval_table3.py` | Runtime/reliability from available events |

These numbers are not current manuscript table numbers. Each command writes CSV, per-sample CSV, and JSON. Exit code 0 means report generation completed, not that all objects passed. Check input errors, per-sample status, and coverage.

## Metrics and aggregation

- **Matched mIoU:** maximum-sum one-to-one IoU matching divided by the reference-part count; unmatched reference parts contribute zero.
- **Boundary F1:** boundary agreement of those matches, averaged over reference parts. Set `metrics.boundary_tolerance_px` explicitly. Primary manuscript columns use 5 px; archived/sensitivity columns also use 2 px. The example retains 2 px.
- **Part F1@0.5:** maximum count of one-to-one matches with IoU >= 0.5, expressed as `2TP/(N_pred+N_ref)`.
- **GT-best IoU:** a separate diagnostic that allows reuse of predictions.

JSON scores are fractions; formatted quality tables use percentages. `paper_table_metrics.summarize` returns `by_group`, `macro` (equal groups), and `object_mean` (equal objects). Legacy CSV output uses macro. Choose and label the correct JSON aggregation for each manuscript row.

Incomplete measurements leave affected means unavailable by default. Freeze a common eligible UID set for paired comparisons and report exclusions. Do not silently assign zeros or change a cohort. The optional zero-missing policy is not a way to fill unavailable manuscript results.

## Labels and reference identity

Prefer integer PNG/NPY maps. Default background is 0; reference ignore IDs are -1 and 65535. RGB requires an exact unambiguous palette. A 16-bit label PNG may appear black without being empty. Disconnected areas with one ID remain one part; colors are local codes, not universal semantic categories.

The metrics module does not resize, rotate, repair, or split masks. Record source SHA-256, frame, palette, resolution, metrics SHA, and tolerance. Recorded aligned-frame and fixed-protocol-frame analyses are different protocols; never pick an orientation by segmentation score.

D is a designated reference with its own verification status. If D is also model guidance, D-arm quality is reference-conditioned consistency, not independent blind-test accuracy.

## Projection requirements

Score integer projections from actual 3D predictions, not 2D input guidance. Verify UID, geometry, transforms, RAW, frozen camera, and hashes. Same filenames do not establish identity.

The generic projector has its own view setup. Running it alone does not reproduce the server acceptance pipeline. Rendering needs Linux, trimesh, pyrender/OpenGL/EGL, and appropriate mesh/camera adapters; the runner also uses fcntl. Full frozen server acceptance snapshots are not all included.

Recorded controlled comparisons used camera foreground IoU >= 0.50, per-arm exact foreground/face-count checks, and pairwise foreground IoU >= 0.99 (six pairs for four arms; fifteen for six methods). Use the same eligible objects and frozen camera across methods. Asset prechecks do not establish runtime gates. Do not crop or patch predictions to force a pass.

These metrics describe the reference view, not the entire 3D surface. Independent methods may need documented geometric registration.

## Historical notes

[legacy_objaverse684.md](legacy_objaverse684.md) and [QA.md](QA.md) preserve the original adapter notes and dated checks. Their cohort/state descriptions are historical. Current packaging checks are in [RELEASE_QA.md](../RELEASE_QA.md).

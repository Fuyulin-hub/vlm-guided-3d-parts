# Historical annotation palettes

These two JSON files are configuration assets, copied unchanged from the local annotation-processing inputs so tests do not depend on an untracked downloads directory.

- `part_palette_30_high_contrast_20261021.json`: source colors used in reviewed annotations.
- `colors_pure_bc_20261022.json`: the historical B/C palette consumed by annotation-repair tools. Those tools intentionally use its entries from index 1 onward; this release does not change their indexing.

Colors do not establish semantic categories across objects. These files are not a per-UID reference mapping or a substitute for a run's actual palette hash. Use the recorded palette and integer label maps for each experiment.

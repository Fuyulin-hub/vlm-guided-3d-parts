# Reproducibility scope

This release organizes local sources. It does not certify current experiment-server state or release the dataset.

## Source identity

| Source | SHA-256 |
|---|---|
| Original V4 helper | 13d5229e85752c7dee13f663e826d7bee0415e63258042ccdf1915e90069f42c |
| Original pilot V5 | de85b40248a17e630508f9d7328de8ae33d45741a97e828dea2559a4758afba0 |
| Maintained V5 at packaging | f7ec454d6938ee5121b2d100894cef8952ce38aa017e1e41b1b1bc145b2327fa |
| Metrics module | 2e90fdb32004384059ca02234a8dc8a21783cb8639527c687798dae6f97adc1e |
| Ablation controls | ba39a2dae553038d480dd14c0cbccd6333cdbf591f119178b71523a9e577012c |

The helper was recovered from matching local pilot source copies and checked against the original hash manifest. The full eleven-file pilot source is under reproducibility/table2_pilot5. Preserve hashes emitted by future runs rather than assuming they match this table.

The generic projector's temporary PNG fsync descriptor was fixed for Windows. No metric or mask logic changed. This revised source must not be assigned a historical server projector hash.

## Available checks

Synthetic CPU regression tests; configuration-only guidance smoke check; frozen pilot source verification; viewer unit tests; compilation from supplied manuscript figures and bibliography.

## External requirements

- The exact approved roster and each table's eligible subset; final 500 and historical 500/527/545 are not interchangeable.
- RAW, geometry, predicted labels, human references, and reference-readiness records.
- Models, checkpoints, upstream licenses, and actual GPU/software environment.
- Complete server camera/acceptance snapshots, full-cohort batch drivers, and prediction commits.
- Per-object records for recomputing aggregates and checking object-weighted versus group-weighted means.

Do not infer a cohort from successful outputs, choose cameras by D part scores, or replace missing files with same-UID artifacts from a different run. The manuscript remains a working submission document with explicitly recorded limitations. Code publication does not release the dataset or grant rights over third-party assets.

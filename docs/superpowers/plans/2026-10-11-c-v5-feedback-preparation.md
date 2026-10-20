# C_v5 feedback preparation plan

> Scope frozen on 2026-10-11. This preparation changes no existing C_v4 outputs and starts no GPU workload.

## Goal

Create an isolated `C_v5_feedback` guidance runner that changes exactly two confirmed causes of C_v4 failure:

1. Every prompted SAM2 inference uses the full positive-point set, negative-point set, bounding box, and one audited feedback pass; it no longer uses the production `box_point` policy.
2. Candidate masks are cleaned around their positive prompt and fused small/tight masks first, so a broad body mask cannot silently consume a supported small part.

## Non-goals

- Do not overwrite `C_v4_exhaustive_box_point` or any previous output.
- Do not add a B-candidate-pool method, alter SAM2 checkpoints, enable multimask selection, change SegViGen, or start a GPU job.
- Do not call the human-refined D reference ground truth.

## Frozen experimental contract

- Run name: `C_v5_feedback` (pilot suffix: `C_v5_feedback_pilot5`).
- Inputs: frozen Objaverse-684 manifest, shared TRELLIS GLB/VXZ, source RGBA image and alpha foreground.
- Prompt contract: `multi_point + negative_point + feedback`, one feedback round maximum, `multimask_output=False` to avoid adding a third confound.
- Determinism: seed 42; checkpoint, Qwen model revision, prompt-template hashes, code revision, input-manifest SHA256, and candidate-fusion thresholds saved in the config.
- Outputs: isolated guidance, masks, annotations, config JSON, and JSONL event log under the run suffix.

## Implementation steps

1. Add pure CPU helpers and failing synthetic-mask tests for point-connected cleaning and small-first fusion.
2. Add a new, isolated V5 runner based on V4 imports; do not edit V4.
3. Enforce the prompt contract in V5 and write the precise positive/negative-point arrays used by SAM2.
4. Clean each raw SAM2 mask within alpha foreground, retain the point-connected component, and record the cleanup decision.
5. Fuse candidates by `(bbox_area, cleaned_mask_area, part_id)` ascending; write only unclaimed pixels, with an explicit rejection reason when insufficient pixels remain.
6. Fill only residual foreground from accepted labels; report zero-accepted-candidate samples as failed rather than whole-object fallback.
7. Save versioned configuration, per-object annotation, and event logs. Verify schema in CPU tests.
8. Run CPU-only tests remotely in the TRELLIS Python environment. GPU pilot execution remains a separate, manually authorized action after reservation and 20-minute availability requirements are met.

## Acceptance checks

- A synthetic small prompt-supported mask overlapping a large body remains present in the final label map.
- A detached satellite/noise island not supported by the positive point is removed.
- Config, annotation, and JSONL records have a stable `schema_version`, `config_sha256`, prompt arrays, feedback state, candidate decisions, and final-map hash.
- Existing V4 files and outputs remain untouched.

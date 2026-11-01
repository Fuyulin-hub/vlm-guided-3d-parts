# Guidance generation

Use a UTF-8 manifest with one row per object:

~~~csv
group,uid,input_rgba
assemblies,YOUR_UID,/absolute/path/to/source_rgba.png
~~~

The image must match the UID and source geometry version. Foreground is alpha > 10. Prompts use original image resolution. Supply the palette as JSON; integer labels remain the preferred evaluation representation.

## Configuration check

~~~bash
python -m pip install -r scripts/requirements-guidance-cpu.txt
python scripts/run_qwen_sam2_guidance_v5_feedback.py \
  --experiment-root /absolute/path/to/new/run \
  --manifest /absolute/path/to/manifest.csv \
  --run-id check01 --output-suffix C_check --dry-run
~~~

This writes `logs/C_check/check01_config.json` with manifest, driver, helper, and prompt hashes. It parses prompt constants without importing GPU models. It does not inspect images or verify checkpoint files.

## GPU inference

Install the upstream Qwen3-VL and SAM2 environments and obtain checkpoints under their applicable licenses. Use a compatible PyTorch/CUDA stack and a Transformers version exposing `Qwen3VLForConditionalGeneration`. CPU dependency files do not reconstruct a historical server environment.

~~~bash
python scripts/run_qwen_sam2_guidance_v5_feedback.py \
  --experiment-root /absolute/path/to/new/run \
  --manifest /absolute/path/to/manifest.csv \
  --model /absolute/path/to/Qwen3-VL-4B-Instruct \
  --sam-repo /absolute/path/to/sam2 \
  --sam-checkpoint /absolute/path/to/sam2.1_hiera_base_plus.pt \
  --sam-config configs/sam2.1_hiera_b+.yaml \
  --palette /absolute/path/to/colors_pure.json \
  --device cuda:0 --seed 42 \
  --run-id guidance01 --output-suffix C_v5_full
~~~

Use a fresh output root and preserve the actual environment, checkpoint/source hashes, and input identities. Existing annotations/guidance are skipped by default; inspect provenance before assigning them to a new run. `--allow-overwrite` is an explicit override, not part of this command. See `--help` for UID and group selection.

## Outputs

| Path under the run root | Purpose |
|---|---|
| `guidance/<suffix>/<group>/<uid>.png` | Colored guidance |
| `annotations/<suffix>/<group>/<uid>.json` | Parts, prompts, feedback, paths, hashes |
| `masks/<suffix>/<group>/<uid>/` | Initial/retry masks and integer labels |
| `logs/<suffix>/` | Configurations and run events |

Each planned part has an initial SAM2 prediction and at most one audit-triggered retry. Fusion uses the original box and positive anchor, resolves overlaps, and fills residual foreground from accepted labels. Inspect failure/partial statuses; file existence alone is not acceptance.

The maintained V5 driver exports English part names as `label`. The archived pilot preserves its original schema and SHA. Do not swap these in a frozen run. Neither driver performs downstream SegViGen generation, camera acceptance, or final paired projection evaluation.

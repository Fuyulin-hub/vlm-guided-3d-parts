# Script map

| Entry point | Role |
|---|---|
| `run_qwen_sam2_guidance_v5_feedback.py` | Qwen3-VL planning, SAM2 masks, one-pass audit, point-aware assembly |
| `run_qwen_sam2_guidance_v4.py` | Original V5 helper: prompts and model adapters |
| `table2_ablation_controls.py` | Controlled-ablation helpers, not a batch launcher |
| `paper_table_metrics.py` | Integer-label metrics and group/object summaries |
| `paper_table_runner.py`, `eval_table{1,2,3}.py` | Evaluation of existing labels/logs |
| `project_glb_labels.py` | Generic GLB-to-label renderer; Linux/EGL for rendering |
| `english_only_annotations.py` | Name conversion without changing numeric label IDs |

The generic projector is not a replacement for a run's frozen camera acceptance pipeline. The CLI table numbers describe an earlier layout, not the current eight-table manuscript. See [evaluation instructions](../docs/evaluation/README.md).

Other scripts record annotation repair, recoloring, cohort preparation, publishing, and dated experiment operations. They may contain original absolute paths and copy, move, or write artifacts. They are retained for traceability, not an installation or run-all sequence.

The [original five-object ablation wrapper](../reproducibility/table2_pilot5/README.md) retains its fixed-five-object guard and source-hash manifest. It is not a full-cohort batch driver.

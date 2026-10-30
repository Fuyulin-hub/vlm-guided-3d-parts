# Camera Rescue19 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Inspect and perform bounded RAW/source camera adjustments for exactly19 original IoU50%-65% objects.

**Architecture:** Preserve original measurements and render in a separate camera_rescue19 directory. Freeze per-object visual hypotheses before search, reuse the existing source renderer and full-canvas sampling, and verify the resulting fixed camera independently before reporting passes.

**Tech Stack:** Python, NumPy, Pillow, SciPy, trimesh, pyrender/EGL; SSH155.

**Spec:** downloads/ABCD500_usercode_evaluation_20261025/camera_rescue19/protocol.json and diagnosis.json, implementing the user's explicit request in this chat.

## Global Constraints

- Exact19 in original band [0.50,0.65), exclude fishing rod43d7c0540bb3468f837b23543882434e.
- RAW alpha>127 and shared source geometry only; FOV42,1024 canvas, unchanged nearest sampling, threshold0.70.
- Per-object hard300s; stop new views235s and fitting255s to allow final evidence writing.
- No mirroring, nonuniform stretch, reference/predicted part fitting, model inference, old-result replacement or automatic retries.
- Keep original456 evaluation running and separate. Later scoring, if authorized, shares one fixed camera/plan per object across ABCD.

## Review Focus

- Wrong-view or front/back ambiguity must not be called a harmless translation error.
- Transparency/source geometry mismatch may remain irreducible and must be reported as suspected, not silently edited away.
- Fine grids must remain inside the original frozen angular bounds, including across azimuth wrap and poles.
- Timeout/error checkpoints must never be promoted without complete return and exact replay verification.
- All original input/code/camera identities must match the frozen selection at start and end.

## Tasks

- [x] Collect only RAW/source evidence and114 checksummed files; inspect all4 initial contact sheets and freeze19 hypotheses.
- [x] Add meaningful selection/bounds/promotion tests; observe3 RED failures before targeted.py; implement and observe3 GREEN tests.
- [x] Implement source/run_targeted.py using prior audited runner interfaces, frozen per-object modes, bounded workers and immutable commits.
- [x] Run targeted and prior camera helper tests; independently review the new runner against scope and deadline rules; fix concrete findings before launch.
- [x] Upload only new config/code/selection to the separate155 root; verify whole-package SHA and exact remote code hashes. Start once on EGL0 with2workers and record actual PID.
- [x] Wait for real completion without restart, independently re-render each available frozen source camera, replay transforms and verify every original pin.
- [x] Download committed evidence, verify receipt, build and visually inspect local before/after review; report19-case coverage, pass/defer counts, exact UIDs and reasons. Do not score or merge automatically.

Execution is authorized by the user's current action request and retained five-minute rule; no additional approval is needed for this isolated camera-only run.

# Camera method pilot3 implementation plan

> Execute the already approved bounded three-object probe in this chat. Keep all code and output isolated; this is a diagnostic prototype, not a replacement evaluation pipeline.

Goal: test whether a full rigid camera initialized by sparse RAW/source correspondences improves one bicycle, one transparent cabinet and one thin truss.

Constraints: RAW and shared input geometry only; no D/prediction pixel reads; fixed FOV42, 1024 canvas, zero distortion; keep all vertices/materials and relative component positions unchanged. Each child has an external 300-second limit including imports, loading, fitting, fresh GLB replay and saving. Original alpha>127 and 0.70 gate remain the formal diagnostic acceptance rule. Other alpha support is diagnostic only. No scoring or merging.

- [x] Freeze the exact three original low-IoU objects, input/previous-camera/code identities and annotated point hypotheses. Audit original RAW camera/rot90/normalization provenance; do not mistake original-asset camera for reconstructed-TRELLIS pose.
- [x] Test camera/backprojection/PnP geometry, degenerate correspondences, validation gates and original-alpha accounting with meaningful synthetic fixtures (RED then GREEN). Implement minimal full-SE(3) PnP + bounded edge refinement. Keep incumbent separately when candidate worsens.
- [x] Review method/code independently, run one bounded child per object, reload original GLB for physical replay, verify pinned inputs and reports, download outputs and make a local before/after report. Decide whether this method warrants expansion; do not launch the other sixteen without evidence.

Outcome:0/3 passed70%, therefore no batch expansion. User subsequently requested60% for the currently displayed three pilots: separate local/server reassessment also0/3, best existing strict IoUs58.576%,38.7259%,31.1723%. Original frozen70% measurements unchanged. Independent audit and report are in the task directory.

Review focus: OpenCV/OpenGL conventions; rotated nearest-neighbour point provenance; point ambiguity/holdouts; transparent silhouette incompatibility; timeout/replay failures must never pass.

# AnchorPart project page

Public page: https://fuyulin-hub.github.io/vlm-guided-3d-parts/

This `gh-pages` branch contains the static project page. The implementation repository remains on `main`.

The browser viewer displays four precomputed AnchorPart/SegviGen outputs: Pineapple, Tractor, Faucet, and Shovel. It does not run model inference. `assets/models.json` records each object UID, the original output checksum, and the source or display orientation. Geometry and segmentation textures are unchanged; the viewer applies only rotation, centering, and uniform scale for display. Examples are selected qualitative illustrations, not a random evaluation set. Region colors are local to each object.

Click a mesh region or its list entry to inspect it. The selected region is highlighted while other regions fade; the inspector shows its descriptive name, region ID, decoded palette color, triangle count, and share of total mesh surface area. It can also be isolated or restored to its context. Drag gestures rotate without selecting; Escape clears selection.

`assets/*.face-ids.bin` contain little-endian uint16 IDs in original mesh face order. Pineapple retains its archived evaluation partition unchanged. For the three replacement objects, the same pinned evaluation decoder was rerun on the unchanged manuscript GLBs to reproduce surface-region IDs; no model inference or scores were changed. Every triangle was matched against the corresponding per-part GLB before export. `assets/*.parts.json` record binding and mesh checksums and region measurements. The browser checks the binding checksum and face counts. All 39 regions have display names assigned after inspecting their isolated geometry and position in the complete model. Mixed regions use compound names; small surface fragments are labeled as fragments. A/B distinguishes regions with similar descriptions within an object. These descriptive names are not semantic ground truth or model-generated semantic predictions; region IDs describe the decoded surface partition rather than VLM proposal IDs. Highlighting changes display opacity only, not geometry, textures, or labels.

The separate prototype video is a silent, browser-compatible conversion of the author's earlier local-interface recording. It illustrates interface interactions and is not quantitative evaluation evidence.

Three.js and its local add-ons are distributed under the license in `vendor/three/LICENSE`. Scientific figures are rendered from the manuscript's existing figures.

Serve this directory over HTTP for local preview (for example, `python -m http.server 8000`). GitHub Pages publishes this branch from its root directory; `.nojekyll` keeps the files unchanged.

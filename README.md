# AnchorPart project page

Public page: https://fuyulin-hub.github.io/vlm-guided-3d-parts/

This `gh-pages` branch contains the static project page. The implementation repository remains on `main`.

The browser viewer displays four precomputed AnchorPart/SegviGen outputs: Pineapple, Tractor, Faucet, and Screwdriver. The viewer does not run model inference. `assets/models.json` records each object UID, the original output checksum, and the source or display orientation. Geometry and segmentation textures are unchanged; the viewer applies only rotation, centering, and uniform scale for display. Examples are selected qualitative illustrations, not a random evaluation set. Region colors are local to each object.

Click a mesh region or its list entry to inspect it. The selected region is highlighted while other regions fade; the inspector shows its associated Qwen name when available, region ID, decoded palette color, triangle count, and share of total mesh surface area. It can also be isolated or restored to its context. Drag gestures rotate without selecting; Escape clears selection. Expand "Original Qwen labels" to inspect the archived guidance map, all retained names, and the downloadable label records.

`assets/*.face-ids.bin` contain little-endian uint16 IDs in original mesh face order. Pineapple retains its archived evaluation partition unchanged. For the three replacement objects, the same pinned evaluation decoder was rerun on the unchanged manuscript GLBs to reproduce surface-region IDs; no model inference or scores were changed. Every triangle was matched against the corresponding per-part GLB before export. `assets/*.parts.json` record binding and mesh checksums and region measurements. The browser checks the binding checksum and face counts. Highlighting changes display opacity only, not geometry, textures, or labels.

Names in `assets/*.qwen.json` are copied verbatim from the retained parts of the original Qwen3-VL output, with raw-response, annotation, and guidance checksums. There are 27 retained 2D labels across these examples. Qwen part IDs, assembled 2D label IDs, and decoded 3D region IDs are distinct. No numeric ID equality or appearance-based renaming is used to connect them.

For display, the unchanged 3D partition is rendered through each archived source-view camera and transform, then intersected with the original 2D guidance labels. A region receives a Qwen name only with at least 32 visible pixels, at least 65% supporting pixels, and at least a 20-percentage-point margin over the second label. These fixed display-association rules name 14 of 35 regions; the other 21 remain explicitly unmatched. Source-view overlap does not establish the semantics of hidden faces, and a Qwen name itself can be incorrect. These associations are not semantic ground truth or new evaluation scores. The CPU projection agrees with matching archived integer renders at over 99.99% of union-foreground pixels for Pineapple, Tractor, and Faucet; Screwdriver uses the same checked renderer on its unchanged displayed mesh. All original Qwen labels remain accessible, including labels without a reliable 3D association.

The separate prototype video is a silent, browser-compatible conversion of the author's earlier local-interface recording. It illustrates interface interactions and is not quantitative evaluation evidence.

Three.js and its local add-ons are distributed under the license in `vendor/three/LICENSE`. Scientific figures are rendered from the manuscript's existing figures.

Serve this directory over HTTP for local preview (for example, `python -m http.server 8000`). GitHub Pages publishes this branch from its root directory; `.nojekyll` keeps the files unchanged.

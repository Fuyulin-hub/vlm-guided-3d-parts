# AnchorPart project page

Public page: https://fuyulin-hub.github.io/vlm-guided-3d-parts/

This `gh-pages` branch contains the static project page. The implementation repository remains on `main`.

The browser viewer displays four precomputed AnchorPart/SegviGen outputs. It does not run model inference. `assets/models.json` records each object UID, the original output checksum, and its recorded source-view orientation. Geometry and segmentation textures are unchanged; the viewer applies only rotation, centering, and uniform scale for display. Examples are selected qualitative illustrations, not a random evaluation set. Region colors are local to each object.

The separate prototype video is a silent, browser-compatible conversion of the author's earlier local-interface recording. It illustrates interface interactions and is not quantitative evaluation evidence.

Three.js and its local add-ons are distributed under the license in `vendor/three/LICENSE`. Scientific figures are rendered from the manuscript's existing figures.

Serve this directory over HTTP for local preview (for example, `python -m http.server 8000`). GitHub Pages publishes this branch from its root directory; `.nojekyll` keeps the files unchanged.

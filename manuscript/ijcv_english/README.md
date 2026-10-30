# IJCV English Manuscript Draft

This English manuscript uses the author's Springer Nature template. The current PDF is `main.pdf`. Unmeasured results are marked TBD, and the remaining statistical figure slots are retained.

## Current Revision, 2026-10-30

The supplied PDF named `IJCV_论文_482版_20261029.pdf` matched the prior local PDF in extracted text. Its body reports 492 camera-aligned objects; that count is retained. Table 1 now includes the author's updated object-weighted 500-object summaries: P3-SAM 38.19 / 17.92 / 35.13, GeoSAM2 24.35 / 10.88 / 31.66, and PartField 32.71 / 13.94 / 16.75 for projection mIoU / Boundary F1 / Part F1 at 0.5. Other rows retain their earlier results and coverage. Their aggregation differences are stated in the caption.

Figure 4 contains the six approved objects and all 24 local GLB renders, with the chair upright and `07c5d34f8abc439d806be0eb5e0175d0` rotated counterclockwise by 90 degrees across all conditions. The D column is a human-guided SegViGen output. The source list and row-wise crop rectangles are in `build/qualitative/sources.json`; `build_3d_figure.py` regenerates the figure from the existing local render cache.

The humanizer revision preserves all 12 equation environments and all 33 citation commands. `main.pdf` has 16 pages, with Table 1 on page 9 and Figure 4 on page 11. `source.zip` contains the files needed to compile the current manuscript. Earlier dated exports outside this project are retained as snapshots.

## Edit And Build

`main.tex` is the single manuscript source. It contains the abstract, all sections, all three tables, figure captions, and declarations. `references.bib` is the sole bibliography. No section files are included through `input`.

The original Chinese draft in the parent directory is preserved. This English project has its own purpose and entry point.

On Overleaf select `main.tex` and XeLaTeX. The flattened source archive contains all required class, bibliography-style, and figure files at its root. Local compilation uses XeLaTeX, BibTeX, and two further XeLaTeX passes. `build_figures.cjs` and `build_3d_figure.py` are local figure helpers; their pre-exported figure PDFs are sufficient for compiling the manuscript.

## Formatting Basis

IJCV author instructions checked on 2026-10-26 at https://link.springer.com/journal/11263/submission-guidelines recommend the Springer Nature class with `iicol`, a 150–250-word abstract, 4–6 keywords, and author–year references. The project uses `sn-jnl` with `sn-apa` and its supplied `sn-apacite.bst`. The class is copied unchanged from the author-supplied December 2024 template. It does not impose an invented ten-page journal limit. Manuscript type and any special-issue rules still require final confirmation.

The main text keeps the template's typography. The imported figures use Times New Roman. Table captions and all table explanations are above their tables. All baseline rows have citations. No measured result is fabricated or highlighted as a best score.

## Figures And Colors

1. Figure 1 uses the selected yellow-box air-conditioner comparison. It contains real source-view A/B/C/D images, with matched display colors and unchanged region geometry. It is one illustrative 2D case, not an aggregate 3D result.
2. Figure 2 is the author's latest `fig2_compact_backbone (1)-Figure 2 Main Pipeline.drawio (1).svg`. The source is retained as `fig2_pipeline.drawio.svg`, and the imported PDF follows it. Only exterior white canvas margins are trimmed in the PDF export. The source diagram is not redesigned. Its illustrative 3D truck assets are not new measured SegviGen outputs.
3. Figure 3 uses the original embedded images from the author-supplied `3D Asset Segmentation(4).pptx` as 2D guidance comparisons, not final 3D results. All 24 placements are retained. The shared B/D traffic-cone image is intentional, as confirmed by the author. The B orientations of the hammer and axe still need alignment. The microwave UID in the presentation does not match the current 500-object manifest, so these are described as supplied illustrative examples rather than all being asserted to belong to that test subset.
4. Figure 4 shows 24 real final-output GLB renders in six A/B/C/D rows. Figure 5–7 remain placeholders for statistical group comparisons, ablations, and computational analysis.

Future statistical plots use A #5878A6, B #A7C7E7, C #F4A7AE, D #A6D0C6, with white background and supplementary #FBD9C9, #CFEAF6, and #DADDE3. This plotting convention does not change raw integer-label evaluation data.

## Evidence To Complete

- The 500-object manifest has 154 assemblies, 181 solid-volume, and 165 thin/fine-grained objects. It is a success-conditioned aligned subset. Reliability denominators must come from the original attempt manifest, not this subset alone.
- The D source is the project-designated human-repaired reference. Some mapping files explicitly remain unverified. Finish its reference audit before treating it as accepted annotation.
- The original unsegmented GLB and D-guided colored GLB are not full 3D part ground truth. Discrete final-output projections require verified camera and label rendering.
- The 3D frozen configuration supports seed 42, voxel resolution 512, twenty sampling steps, and the recorded guided checkpoint. The Qwen/SAM2 identifiers in the five-object pilot are known, but the full-run frozen configuration still needs confirmation.
- The tables retain the supplied measured results and explicitly mark unavailable entries. New Part F1 values for Find3D, SegViGen, and Ours were not supplied and remain TBD. Timing, memory, and collapse statistics still require measurement.
- Finalize the held-out split, reference protocol, baseline adapters, collapsed-output criterion, original attempt counts, complete numerical findings, author identities, funding, license and release statements, GitHub URL, and generative-tool disclosure.

## Source Traceability

The method equations were checked against `scripts/run_qwen_sam2_guidance_v5_feedback.py` and the frozen pilot source. The original-box priority, original positive-point anchor, one retry, fallback on parse/invalid retry, twenty foreground palette entries, and lack of a second audit are represented explicitly. Schema @2 changes the annotation language fields, not the segmentation algorithm.

The current manifest and data-state statements come from `downloads/finaldata_500_20261025/` and `outputs/paper_figures/fig1_motivation_candidates/cache/finaldata/`. The final 3D configuration is preserved in `downloads/ABCD500_usercode_evaluation_20261025/acceptance12/gold_snapshot.json`. These local materials were read without launching new server GPU work.

Published metadata for the main methods was checked against CVF, ICLR, ACM, and arXiv primary records. P3-SAM remains cited by its verified arXiv version because formal-venue metadata was inconsistent. Internal citation keys retain their earlier names even when the verified publication year changed.

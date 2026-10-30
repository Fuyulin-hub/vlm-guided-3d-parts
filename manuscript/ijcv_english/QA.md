# Draft Validation

## Current Revision, 2026-10-30

The previous supplied PDF and prior local PDF had identical extracted text. Although the supplied filename says 482, the manuscript body uses 492 camera-aligned objects; the revision preserves that recorded count. The three updated external baselines use object-weighted means over 500 objects, as stated in the supplied screenshot. Table 1 includes all nine new values and an explicit Part F1 column. Find3D retains N=468 and SegViGen/Ours retain N=492, with the aggregation and coverage differences stated in the caption.

The requested humanizer pass removed repetitive framing and revised headings to sentence case. All 12 equation environments and all 33 citation commands were preserved. The prose lint's four dash flags refer to the pre-existing missing-value markers in Table 2, not authored prose dashes. These table markers were retained. Sentence-rhythm and semicolon advisories were reviewed in context.

Figure 4 replaces the empty slot with all 24 original renders from the six approved GLB object comparisons. Every condition uses the approved row-wise orientation. One common crop rectangle per object preserves cross-condition framing. The PDF contains exactly 24 embedded image panels on page 11, before the references. Figure labels use embedded Times New Roman fonts. Render provenance and crop coordinates are retained in `build/qualitative/sources.json`.

XeLaTeX, BibTeX, and the subsequent reference passes completed successfully. The resulting `main.pdf` has 16 pages. The updated Table 1 and the new Figure 4 were visually inspected at full-page resolution, and a contact sheet of all 16 pages was checked. There are no undefined references, undefined citations, overfull boxes, or oversized-float errors. Table 1 is on page 9 and Figure 4 on page 11. The visual comparison is a selected qualitative panel; its display cameras are separate from the scoring cameras.

Remaining TBD measurements and author declarations are preserved. Figure 5–7 remain the earlier statistical placeholders. Earlier exported snapshots and the supplied desktop PDF were not overwritten.

## Current Editor Revision, 2026-10-27

The existing `main.tex` was edited in place. Figure 3 now contains the 24 supplied 2D-guidance placements, with original image pixels and label colors retained. The shared B/D traffic-cone image is intentional and was confirmed by the author. Figure 4 remains reserved for final 3D segmentation outputs. Statistical figure slots are now Figures 5–7. The desktop compiler returned success for the saved main file. The current editor remains the source of the updated PDF preview. No separate manuscript PDF or replacement document was compiled for this revision, and the earlier exported PDF and ZIP remain the initial snapshots.

The B-column hammer and axe orientations still require alignment. The microwave description UID is absent from the current 500-object manifest, so the supplied panel is not described as wholly belonging to that evaluation subset. No palette recoloring or pose correction was inferred from appearance alone.

## Initial Export Validation, 2026-10-26

Target is an IJCV English research-article draft. The supplied Springer Nature class and APA-based author–year bibliography style are used with the recommended two-column option. No unified ten-page IJCV limit was established by the author guidelines, so the draft is not padded or compressed to ten pages.

The initial exported draft has 12 pages including the declarations and references. XeLaTeX, BibTeX, and the required subsequent XeLaTeX passes completed successfully. The desktop editor's own compiler also returned success for that saved project. No undefined citations, unresolved references, or overfull boxes remained in that export. Eighteen cited keys resolve. The abstract is approximately 175 words and there are five keywords. Related Work occupies less than one page of text.

The main manuscript is one `main.tex` with no `input` section dependencies. There are three booktabs tables with 84 unmeasured metric cells. Every external method row is cited, all captions and table explanations are above their tables, and four figure slots are intentionally left without fabricated data. The source has no prose em dashes and no prose semicolons. A supporting prose lint flagged one similar-sentence-length run, which was reviewed without mechanical rewriting.

The PDF contact sheet and the main-pipeline and comparison-table pages were visually checked. Only external white margins were trimmed from the exported Figure 2 PDF. The original editable SVG was retained unchanged. Figures 1 and 2 have stable numbering and the remaining slots are Figures 3–6. No clipped text, overlapping cells, or accidental blank final page remains. Intentional unfilled figure slots are retained at the author's request.

Scientifically necessary pending items remain visible in the draft. They include the full-run 2D configuration, reference audit, held-out split, baseline adapters, projection validation, original attempt counts, all three result tables, human scores, timing and memory measurements, author information, repository, and release declarations. The aligned subset does not establish a zero failure rate, and the current reference is not represented as full 3D ground truth.

Final submission still needs author review, genuine complete results, final figure readability at publication scale, precise APA 7 bibliography checks against the publisher's production requirements, and the completed declarations. No Grammarly service was used and no claim of passing Grammarly is made. Generative-tool disclosure remains for author completion.

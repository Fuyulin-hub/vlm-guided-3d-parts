# IJCV manuscript: single main-text edition

Updated 2026-11-01. Edit `main.tex`; the corresponding compiled manuscript is `main.pdf`.
All scientific content previously in the appendix and standalone supplement is now in the main text.
The manuscript contains 8 consecutively numbered tables and 9 figures. It has no appendix or Supplementary Table S1 references.

## Content map

| Material | Current location |
|---|---|
| Archived pre/post feedback analysis (former Appendix A.1) | Section 4.8; Table 5 |
| C guidance by structural group | Section 4.9.1; Table 6 |
| Available-set projection breakdown (former Appendix A.2) | Section 4.9.2; Table 7 |
| Archived part count and part size (former Appendix A.3 and Supplementary Table S1) | Section 4.9.3; Table 8 |
| Six-method qualitative comparison | Figure 5 |
| A/B/C/D qualitative comparison, method labels along the bottom | Figure 8 |

The primary projection cohort remains 479 objects. The older 500-object cohort, the final selected 500-object roster, the 468-object Find3D summary, and the 471-object BF1@2px column are explicitly distinguished. No scores, labels, predictions, or model outputs were recomputed for this edit. Figure files were not altered.

## Compile

Select `main.tex` as the main document and XeLaTeX as compiler. All required figure files, `references.bib`, `sn-jnl.cls`, and `sn-apacite.bst` are supplied. Run BibTeX to regenerate `main.bbl`; the local source ZIP also includes a fallback copy, but generated bibliography files are not tracked in Git.

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## Overleaf update

The user's existing project stores its paper files in `01_Manuscript_Source/`. The locally prepared `IJCV_MainText_Source_20261101.zip` can be uploaded into that folder; this generated archive is not tracked in Git. Alternatively upload the current TeX source, bibliography, class/style files, and all figure files referenced by `main.tex`. Select `01_Manuscript_Source/main.tex` as the main file. The new six-method PDF and the latest A/B/C/D figure are included in this repository.

Move the old `supplementary.tex` and any standalone `supplementary.pdf` out of the active submission files. Their scientific content is already in `main.tex`. Preserve them as historical backups if needed; they are not separate submission items.

This revision was compiled locally and in the Codex LaTeX editor. The Overleaf project was inspected but not updated: browser control stopped because the current URL could not be verified reliably on Windows. The clean ZIP provides the complete local revision for upload.

## Historical files

The previous supplement is archived under `archive/supplementary_merged_20261101/` in the local working project. Old source/PDF packages are historical snapshots, not the current submission. The clean ZIP excludes these archives, build intermediates, and separate supplementary files.

## Remaining scientific checks

This is a content-organization and compilation update, not a new data audit. Existing manuscript placeholders and provenance limitations remain visible: corresponding-author email, code URL, reference audit, attempted-population/development split, frozen model/method configurations, ablation cohort reconciliation, and the recorded projection-source verification issue.

# Manuscript source

Current source of the manuscript on decoupling mask refinement from assembly for
language-guided 3D part segmentation.

| File | Role |
|---|---|
| `main.tex` | Single-file source, Springer `sn-jnl` class, author-year references |
| `sn-jnl.cls` | Springer Nature journal class |
| `sn-mathphys-ay.bst` | Bibliography style (author-year) |
| `references.bib` | Bibliography database |
| `fig1_motivation.pdf` | Figure 1, motivation |
| `fig2_pipeline.pdf` | Figure 2, pipeline |
| `fig_audit_assembly_single_column.pdf` | Audit and assembly controls |
| `fig_sixmethod_3d_comparison_clean.pdf` | Six-method qualitative comparison |
| `fig3_guidance_clean.png` | Guidance comparison, source images |
| `fig4_3d_comparison_clean.pdf` | 3D comparison across conditions |
| `fig_summary_bars.pdf` | Summary of quantitative results |

`main.pdf` is produced by the build below and is not versioned, because `.gitignore`
excludes generated PDFs at this level.

## Build

`pdflatex` → `bibtex` → `pdflatex` → `pdflatex`:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The source builds with an unmodified TeX Live or MiKTeX installation. The preamble
defines `\burl` as `\url`, because the bibliography style emits URLs whose
underscores are otherwise interpreted in text mode.

## Notes

- Section numbering, table numbering, and the figure order above match `main.pdf`.
- The reference annotations discussed in Section 5.1 were model-drafted and
  manually corrected; see the Acknowledgements section for the annotator credit.
- The code URL in the abstract points to this repository.

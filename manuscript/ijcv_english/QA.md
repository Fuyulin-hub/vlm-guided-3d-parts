# Main-text integration QA — 2026-11-01

- Existing `main.tex` edited in place; all pre-merge source and PDF files were backed up.
- Former appendix feedback and projection analyses moved into Results/Experiments; appendix counters removed.
- The full Supplementary Table S1 and its unique methods, scope, and interpretation merged into Section 4.9.3 as Table 8.
- All original numeric table rows preserved; no model inference, projection, relabeling, or metric recomputation.
- All nine figure assets retain their pre-merge SHA256. Figure 3 retains `width=\columnwidth`.
- Current manuscript: 20 pages, eight consecutive tables, nine figures.
- No remaining Appendix/Supplementary references, undefined references/citations, multiply-defined labels, or overfull boxes in the local compile.
- Built-in Codex compile passed. Local XeLaTeX and xdvipdfmx exported `main.pdf` successfully.
- Visual inspection covers the full contact sheet and the newly integrated tables. Table 8 uses legible grouped metric headers and retains the archived BF1@2px protocol.
- Underfull-box notices remain from the template's ragged-bottom two-column layout; they are not compile failures.
- Online synchronization was not performed because browser control was stopped by URL-verification policy. No Overleaf source was modified.

## Scope preserved

1. Primary projection: 479 jointly evaluated objects. Find3D is a separate 468-object group-macro summary; four BF1@2px entries use 471 objects.
2. Direct 2D B/C coverage remains 440/479; unequal-set differences are descriptive.
3. Archived feedback and size analyses retain the earlier 500-object manifest, reference version, group-macro averaging, and provisional-reference limitations.
4. D uses the scoring reference as model guidance and is described as reference-conditioned consistency.
5. Selected qualitative examples remain explicitly favorable cases, without implying universal method superiority.

## Submission limitations retained

Compilation success does not resolve the existing scientific placeholders or metadata gaps. No unsupported readiness, configuration, source-identity, or paired-cohort claim was added by this merge.

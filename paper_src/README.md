# paper_src

Organised so that new measurements are *appended*, never edited into shape.

| file | holds |
|---|---|
| `main.tex` | document skeleton, abstract, \input order |
| `method.tex` | the formulas: I_RL silence, self-target, the routing rule |
| `figures/routing_flow.tex` | the method flowchart (TikZ; also exported to SVG) |
| `related.tex` | related work organised by the claim each line of work owns |
| `results.tex` | **append-only** results tables, each naming its run id |

Rules, which exist because breaking them has already cost us a result:

1. **`results.tex` is append-only.** A number that turns out to be wrong gets a correction
   added next to it, not a silent edit. Two entries already in the file are corrections.
2. **Nothing enters a table until it was measured end-to-end** on a held-out benchmark.
   Train reward is not a result; `results.tex` opens by showing it mis-orders checkpoints.
3. **Every table names the run** that produced it, so a row can be re-derived.
4. Report the noise floor next to any effect it qualifies.

## Build

    bash scripts/build_paper.sh      # 2 pdflatex passes, then greps '^!' and fails loudly
    bash scripts/build_figs.sh       # figures/*.tex -> figures/*.svg (needs dvisvgm)

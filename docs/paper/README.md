# Paper

A short engineering write-up of the pipeline: what problem it solves, how the
stages fit together, and what we have and haven't measured.

- `paper.pdf` — compiled, 7 pages
- `paper.tex` — source
- `fig/track_selection.pdf` — generated from real tracking runs
- Figure 1 is TikZ inside the `.tex`; Figure 2 uses `../review_gui.png`, the
  same screenshot the top-level README shows, so it can't drift from it

Rebuild after editing:

```bash
cd docs/paper && tectonic -X compile paper.tex   # or pdflatex paper.tex
```

Figures come from real runs against MBARI benthic footage. The track-selection
numbers in the paper are from 19 tracks across three 12-second clips.

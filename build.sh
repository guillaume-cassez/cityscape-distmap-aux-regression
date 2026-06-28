#!/usr/bin/env bash
# Build the paper PDFs (EN + FR). Format matches the companion BRATS report
# (house style for this series): XeLaTeX via pandoc, US letter, 1.7 cm margins,
# Liberation Serif, with header.tex mapping inline Unicode superscripts to LaTeX
# (the main font lacks U+207B). Requires: pandoc + TeX Live with xelatex.
set -e
cd "$(dirname "$0")"
OPTS=(--pdf-engine=xelatex -V papersize=letter -V geometry:margin=1.7cm
      -V mainfont="Liberation Serif" -H header.tex)
pandoc paper.md    -o paper.pdf    "${OPTS[@]}"
pandoc paper_fr.md -o paper_fr.pdf "${OPTS[@]}"
echo "built paper.pdf + paper_fr.pdf (letter, 1.7cm margins, Liberation Serif)"

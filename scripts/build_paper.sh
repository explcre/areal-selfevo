#!/usr/bin/env bash
# Build paper_src/main.tex. Two passes for refs, then fail loudly on a real LaTeX error.
# `grep -a` is mandatory: pdflatex logs are treated as binary and a plain grep silently
# finds nothing, which reads as "no errors".
set -u -o pipefail
cd "$(dirname "$0")/../paper_src" || exit 1
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null 2>&1 \
    || { echo "PASS $pass FAILED"; grep -a -A4 '^!' main.log | head -40; exit 1; }
done
if grep -aq '^!' main.log; then
  echo "LaTeX errors:"; grep -a -A4 '^!' main.log | head -40; exit 1
fi
pages=$(grep -aoE 'Output written on main\.pdf \([0-9]+ page' main.log | grep -oE '[0-9]+' | head -1)
echo "OK: main.pdf, ${pages:-?} page(s)"

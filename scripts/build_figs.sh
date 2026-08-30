#!/usr/bin/env bash
# Export each paper figure to SVG from the SAME TikZ source the PDF uses, so the figure in
# the paper and the figure on the web cannot drift apart.
#
# The figure files are written for the paper (wrapped in figure/caption/label), so we lift
# just the tikzpicture out rather than redefining LaTeX internals -- redefining `figure`
# fights whichever class is loaded and breaks differently under each one.
set -u -o pipefail
cd "$(dirname "$0")/../paper_src" || exit 1
command -v dvisvgm >/dev/null || { echo "dvisvgm not installed; skipping SVG export"; exit 0; }
mkdir -p figures/build
rc=0
for fig in figures/*.tex; do
  name=$(basename "$fig" .tex)
  body="figures/build/${name}_body.tex"
  sed -n '/\\begin{tikzpicture}/,/\\end{tikzpicture}/p' "$fig" > "$body"
  if [ ! -s "$body" ]; then echo "SKIP ${name}: no tikzpicture found"; continue; fi
  cat > "figures/build/${name}_standalone.tex" <<EOF
\\documentclass[dvisvgm,border=4pt]{standalone}
\\usepackage{amsmath,amssymb}
\\usepackage{tikz}
\\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds}
\\begin{document}
\\input{${name}_body}
\\end{document}
EOF
  if ( cd figures/build \
       && latex -interaction=nonstopmode -halt-on-error "${name}_standalone.tex" >/dev/null 2>&1 \
       && dvisvgm --no-fonts --exact-bbox -o "../${name}.svg" "${name}_standalone.dvi" >/dev/null 2>&1 )
  then
    echo "OK figures/${name}.svg ($(wc -c < "figures/${name}.svg") bytes)"
  else
    echo "FAILED ${name}"; grep -a -A4 '^!' "figures/build/${name}_standalone.log" 2>/dev/null | head -20
    rc=1
  fi
done
exit $rc

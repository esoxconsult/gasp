#!/usr/bin/env bash
# Build RNAAS manuscript PDF (requires MacTeX / TeX Live with revtex4-1).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# macOS: MacTeX installs here; GUI login adds PATH, Terminal often does not.
if [[ "$(uname -s)" == "Darwin" ]] && [[ -d "/Library/TeX/texbin" ]]; then
  export PATH="/Library/TeX/texbin:${PATH}"
fi

if [[ ! -d aastex631 ]]; then
  echo "ERROR: aastex631/ not found. Extract with:"
  echo "  tar -xzf aastexv631.tar.gz"
  echo "Download tarball from:"
  echo "  https://journals.aas.org/aastex-package-for-manuscript-preparation/"
  exit 1
fi

if [[ ! -f figures/05_ecas_validation.png ]] || [[ ! -f figures/03_astar_vs_s1.png ]]; then
  echo "ERROR: PNG figures missing under docs/figures/"
  echo "Copy from VPS, e.g.:"
  echo "  scp media:~/gasp/figures/{05_ecas_validation,03_astar_vs_s1}.png docs/figures/"
  exit 1
fi

export TEXINPUTS="${DIR}/aastex631//:${TEXINPUTS:-}"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "ERROR: pdflatex not found."
  echo "  Install MacTeX: https://www.tug.org/mactex/  (then restart Terminal)"
  echo "  Or add to ~/.zshrc: export PATH=\"/Library/TeX/texbin:\$PATH\""
  exit 1
fi

pdflatex -interaction=nonstopmode rnaas_paper.tex
pdflatex -interaction=nonstopmode rnaas_paper.tex
echo "OK: ${DIR}/rnaas_paper.pdf"

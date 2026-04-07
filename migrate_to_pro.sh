#!/usr/bin/env bash
# Move commercial GASP assets into sibling repo ../esoxspace
# Run from the open-source GASP repository root.

set -euo pipefail

PRO="../esoxspace"

mkdir -p "${PRO}/pipeline" "${PRO}/data/final"

move_if_exists() {
  local src=$1
  local dst=$2
  if [[ -f "$src" ]]; then
    mv "$src" "$dst"
    echo "OK: moved $src -> $dst"
  else
    echo "SKIP: not found: $src" >&2
  fi
}

move_if_exists "pipeline/09_ml_taxonomy_baseline.py" "${PRO}/pipeline/"
move_if_exists "pipeline/10_find_mining_targets.py" "${PRO}/pipeline/"
move_if_exists "pipeline/11_find_family_interlopers.py" "${PRO}/pipeline/"
move_if_exists "data/final/gasp_catalog_v1_with_ml.parquet" "${PRO}/data/final/"
move_if_exists "data/final/mining_targets_x_complex.csv" "${PRO}/data/final/"
move_if_exists "data/final/family_interlopers.csv" "${PRO}/data/final/"

echo
echo "Done. Next (private repo):"
echo "  cd ${PRO}"
echo "  git init"

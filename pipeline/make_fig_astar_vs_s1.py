#!/usr/bin/env python3
"""Generate Fig. 2 (a* vs s1) to A&A two-column standards.

Run from the repo root:  python pipeline/make_fig_astar_vs_s1.py
Output: figures/03_astar_vs_s1.png  (also copied to docs/figures/)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FINAL_DIR = Path("data/final")
INPUT_FILE = FINAL_DIR / "gasp_catalog_v1.parquet"
FIGURES_DIR = Path("figures")
DOCS_FIGURES_DIR = Path("docs/figures")

if not INPUT_FILE.is_file():
    print(f"ERROR: {INPUT_FILE} not found", file=sys.stderr)
    sys.exit(1)

df = pd.read_parquet(INPUT_FILE, engine="pyarrow")
subset = df[df["s1"].notna() & df["a_star"].notna()].copy()
print(f"Subset for a* vs s1: {len(subset):,} asteroids")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(18 / 2.54, 5))

has_albedo = subset["albedo"].notna()

sc = ax.scatter(
    subset.loc[has_albedo, "a_star"],
    subset.loc[has_albedo, "s1"].clip(-25, 15),
    c=subset.loc[has_albedo, "albedo"],
    cmap="RdYlGn", alpha=0.4, s=15,
    vmin=0, vmax=0.4,
)
ax.scatter(
    subset.loc[~has_albedo, "a_star"],
    subset.loc[~has_albedo, "s1"].clip(-25, 15),
    color="lightgray", alpha=0.2, s=8,
)

cb = plt.colorbar(sc, ax=ax)
cb.set_label(r"Albedo $p_V$ (NEOWISE)", fontsize=12)
cb.ax.tick_params(labelsize=12)

ax.axvline(-0.1, color="steelblue", ls="--", lw=1.5, alpha=0.7,
           label=r"C/ambiguous boundary ($a^*=-0.10$)")
ax.axvline(0.1, color="tomato", ls="--", lw=1.5, alpha=0.7,
           label=r"ambiguous/S boundary ($a^*=+0.10$)")
ax.axhline(0, color="black", ls=":", lw=1.5, alpha=0.5)

ax.set_xlabel(r"SDSS $a^*$ color index", fontsize=12)
ax.set_ylabel(r"$s_1$ — NUV slope 374–418 nm ($\mu$m$^{-1}$)", fontsize=12)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12)
ax.set_xlim(-0.5, 0.5)

plt.tight_layout()
out = FIGURES_DIR / "03_astar_vs_s1.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy(out, DOCS_FIGURES_DIR / out.name)
print(f"Copied to: {DOCS_FIGURES_DIR / out.name}")

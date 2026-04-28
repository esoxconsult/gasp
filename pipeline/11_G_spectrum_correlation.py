#!/usr/bin/env python3
"""
11_G_spectrum_correlation.py
GASP — G × Gaia reflectance spectrum for the GAPC×GASP crossmatch objects.

For objects that have both a GAPC phase-curve G value (from gapc_catalog_v8)
and a GASP Gaia reflectance spectrum (16 bands + 4 slopes), we compute:

  1. Spearman rho(G, refl_lambda) for each of the 16 wavelength bands
  2. Spearman rho(G, s1..s4) for the 4 spectral slopes
  3. Multiple-band regression: which combination of bands best predicts G?
     (RF feature importance, 5-fold CV)
  4. Partial correlation: r(G, refl_lambda | logD) — spectral signal beyond size

The goal is to find spectral features that encode space weathering / surface
age information beyond what G alone carries.

Input:  data/final/gasp_catalog_v2.parquet  (has G column from GAPC crossmatch)
Output:
  figures/11_G_spectrum_correlation.png
  data/final/11_G_spectrum_stats.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parents[1]
V2_PATH     = ROOT / "data" / "final" / "gasp_catalog_v2.parquet"
FIGURES_DIR = ROOT / "figures"
LOG_DIR     = ROOT / "data" / "final"

BANDS      = [374, 418, 462, 506, 550, 594, 638, 682,
              726, 770, 814, 858, 902, 946, 990, 1034]
REFL_COLS  = [f"refl_{b}" for b in BANDS]
SLOPE_COLS = ["s1", "s2", "s3", "s4"]
ALL_FEAT   = REFL_COLS + SLOPE_COLS  # 20 features


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)[0]


def main() -> None:
    print("\n" + "=" * 65)
    print("  GASP Step 11 — G × Gaia spectrum correlation")
    print("=" * 65)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not V2_PATH.exists():
        print(f"  ERROR: {V2_PATH} not found"); return

    df = pd.read_parquet(V2_PATH)
    print(f"\n  GASP v2: {len(df):,} rows, {len(df.columns)} cols")

    # ── Find G column (from GAPC crossmatch) ─────────────────────────────────
    g_col = None
    for candidate in ["G", "gapc_G", "phase_slope_G"]:
        if candidate in df.columns:
            g_col = candidate
            break

    if g_col is None:
        # GASP v2 was built from gasp_catalog_v1 which doesn't directly have GAPC G.
        # Check if any column contains phase slope info
        phase_cols = [c for c in df.columns if "phase" in c.lower() or
                      c.upper() in ("G", "G_HG")]
        print(f"  No G column found. Available phase-related: {phase_cols}")
        print("  The GAPC G values need to be merged into GASP v2 first.")
        print("  Merging from ~/gapc/data/final/gapc_catalog_v8.parquet ...")
        gapc_path = ROOT.parent / "gapc" / "data" / "final" / "gapc_catalog_v8.parquet"
        if not gapc_path.exists():
            print(f"  ERROR: {gapc_path} not found"); return
        gapc = pd.read_parquet(gapc_path)[["number_mp", "G", "D_km",
                                            "sigma_G", "n_obs"]].copy()
        gapc = gapc.rename(columns={"G": "gapc_G", "D_km": "gapc_D_km",
                                     "sigma_G": "gapc_sigma_G",
                                     "n_obs": "gapc_n_obs"})
        df = df.merge(gapc, on="number_mp", how="left")
        g_col = "gapc_G"
        print(f"  Merged: {df['gapc_G'].notna().sum():,} objects with gapc_G")

    # ── Crossmatch subset ─────────────────────────────────────────────────────
    feat_avail = [c for c in REFL_COLS if c in df.columns]
    if not feat_avail:
        print("  No refl_ columns found"); return

    cross = df[df[g_col].notna()].copy()
    cross = cross[cross[feat_avail].notna().all(axis=1)]
    print(f"\n  Crossmatch (G + full spectrum): {len(cross):,} objects")

    if len(cross) < 20:
        print("  Too few objects for analysis"); return

    G = cross[g_col].values

    # ── 1. rho(G, band) per wavelength ───────────────────────────────────────
    print(f"\n  1. Spearman rho(G, refl_lambda):")
    band_rho, band_p = [], []
    for col, b in zip(REFL_COLS, BANDS):
        if col not in cross.columns:
            band_rho.append(np.nan); band_p.append(np.nan)
            continue
        rho, p = spearmanr(G, cross[col].values)
        band_rho.append(rho); band_p.append(p)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else " ")
        print(f"    {b:4d} nm: rho={rho:+.4f}  p={p:.2e}  {sig}")

    # ── 2. rho(G, slope) ─────────────────────────────────────────────────────
    print(f"\n  2. Spearman rho(G, spectral slope):")
    slope_results = {}
    for col in SLOPE_COLS:
        if col not in cross.columns:
            continue
        sub = cross[cross[col].notna()]
        rho, p = spearmanr(sub[g_col], sub[col])
        print(f"    {col}: rho={rho:+.4f}  p={p:.2e}  n={len(sub):,}")
        slope_results[col] = dict(rho=rho, p=p, n=len(sub))

    # ── 3. Partial: r(G, refl | logD) ────────────────────────────────────────
    d_col = "gapc_D_km" if "gapc_D_km" in cross.columns else "diameter_km"
    band_rho_partial = []
    if d_col in cross.columns:
        ctrl = cross[cross[d_col].notna() & (cross[d_col] > 0)].copy()
        ctrl["log_D"] = np.log10(ctrl[d_col])
        print(f"\n  3. Partial r(G, refl | logD)  (n={len(ctrl):,}):")
        for col, b in zip(REFL_COLS, BANDS):
            if col not in ctrl.columns or ctrl[col].isna().all():
                band_rho_partial.append(np.nan)
                continue
            sub = ctrl[ctrl[col].notna()]
            r_part = partial_spearman(sub[g_col].values,
                                      sub[col].values,
                                      sub["log_D"].values)
            band_rho_partial.append(r_part)
            if abs(r_part) > 0.05:
                print(f"    {b:4d} nm: r_partial={r_part:+.4f}")
    else:
        band_rho_partial = [np.nan] * len(BANDS)
        print("  No diameter column — skipping partial correlation")

    # ── 4. RF feature importance ──────────────────────────────────────────────
    feat_cols = [c for c in ALL_FEAT if c in cross.columns]
    X = cross[feat_cols].fillna(cross[feat_cols].median()).values
    print(f"\n  4. RF feature importance for G (n={len(cross):,}, "
          f"{len(feat_cols)} features):")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    cv_r2 = cross_val_score(rf, X, G, cv=5, scoring="r2")
    print(f"    5-fold CV R²: {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
    rf.fit(X, G)
    importances = rf.feature_importances_
    feat_imp = sorted(zip(feat_cols, importances), key=lambda x: -x[1])
    print("    Top 8 features:")
    for feat, imp in feat_imp[:8]:
        print(f"      {feat:20s}: importance={imp:.4f}")

    # ── By taxonomy ───────────────────────────────────────────────────────────
    tax_col = ("taxonomy_final" if "taxonomy_final" in cross.columns
               else "taxonomy_ml")
    s1_col  = "s1"
    if s1_col in cross.columns and tax_col in cross.columns:
        print(f"\n  5. rho(G, s1) by taxonomy ({tax_col}):")
        for t in ["S", "C", "X", "V", "D"]:
            sub = cross[cross[tax_col].astype(str).str.startswith(t) &
                        cross[s1_col].notna()]
            if len(sub) < 15:
                continue
            rho, p = spearmanr(sub[g_col], sub[s1_col])
            print(f"    {t}: rho={rho:+.4f}  p={p:.2e}  n={len(sub):,}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"G × Gaia spectrum  (GASP v2, n={len(cross):,})", fontsize=12)

    # rho profile across bands
    ax = axes[0, 0]
    valid_b = [(b, r, p) for b, r, p in zip(BANDS, band_rho, band_p)
               if not (np.isnan(r) or np.isnan(p))]
    if valid_b:
        bs, rs, ps = zip(*valid_b)
        colors_b = ["#d62728" if p < 0.05 else "#aec7e8" for p in ps]
        ax.bar(range(len(bs)), rs, color=colors_b, alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels([str(b) for b in bs], rotation=45, fontsize=7)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel(r"Spearman $\rho(G, \mathrm{refl})$")
        ax.set_title("G vs reflectance by wavelength  (red = p<0.05)")
        ax.grid(alpha=0.2, axis="y")

    # Partial rho profile
    ax = axes[0, 1]
    valid_bp = [(b, r) for b, r in zip(BANDS, band_rho_partial) if not np.isnan(r)]
    if valid_bp:
        bs2, rs2 = zip(*valid_bp)
        cols2 = ["#d62728" if abs(r) > 0.08 else "#aec7e8" for r in rs2]
        ax.bar(range(len(bs2)), rs2, color=cols2, alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(bs2)))
        ax.set_xticklabels([str(b) for b in bs2], rotation=45, fontsize=7)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel(r"Partial $r(G, \mathrm{refl}\,|\,\log D)$")
        ax.set_title("Partial G×refl (size-controlled)")
        ax.grid(alpha=0.2, axis="y")

    # RF feature importance
    ax = axes[1, 0]
    top_n = min(10, len(feat_imp))
    top_names = [f[0].replace("refl_", "").replace("gasp_", "")
                 for f, _ in zip(feat_imp[:top_n], range(top_n))]
    top_imps  = [imp for _, imp in feat_imp[:top_n]]
    ax.barh(range(top_n), top_imps[::-1],
            color="steelblue", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("RF feature importance")
    ax.set_title(f"Top features for G prediction\nCV R²={cv_r2.mean():.3f}")
    ax.grid(alpha=0.2, axis="x")

    # G vs s1 scatter
    ax = axes[1, 1]
    if s1_col in cross.columns:
        smp = cross[cross[s1_col].notna()].sample(
            min(8000, len(cross)), random_state=42)
        ax.scatter(smp[s1_col], smp[g_col], s=2, alpha=0.2,
                   color="steelblue", rasterized=True)
        if slope_results.get(s1_col):
            rho_s1 = slope_results[s1_col]["rho"]
            p_s1   = slope_results[s1_col]["p"]
            ax.set_title(f"G vs s1  rho={rho_s1:+.3f}  p={p_s1:.2e}")
        ax.set_xlabel("s1 (slope 374–506 nm)")
        ax.set_ylabel(f"{g_col}")
        ax.grid(alpha=0.2)

    fig.tight_layout()
    out = FIGURES_DIR / "11_G_spectrum_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  → {out}")

    stats = {
        "n_crossmatch": int(len(cross)),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std":  float(cv_r2.std()),
        "band_rho":  {str(b): float(r) for b, r in zip(BANDS, band_rho)
                      if not np.isnan(r)},
        "band_p":    {str(b): float(p) for b, p in zip(BANDS, band_p)
                      if not np.isnan(p)},
        "slope_rho": {c: float(v["rho"]) for c, v in slope_results.items()},
        "top_features": [(n, float(i)) for n, i in feat_imp[:5]],
    }
    (LOG_DIR / "11_G_spectrum_stats.json").write_text(
        json.dumps(stats, indent=2))
    print(f"  Log → data/final/11_G_spectrum_stats.json\n")


if __name__ == "__main__":
    main()

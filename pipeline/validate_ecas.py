#!/usr/bin/env python3
"""GASP Phase 9: ECAS (Zellner et al. 1985) cross-validation vs GASP NUV reflectance."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
from scipy import stats

FINAL_DIR = Path("data/final")
RAW_DIR = Path("data/raw")
FIGURES_DIR = Path("figures")
INPUT_FILE = FINAL_DIR / "gasp_catalog_v1.parquet"
ECAS_FILE = RAW_DIR / "ecas_zellner1985.parquet"
LOG_FILE = FINAL_DIR / "ecas_validation_log.json"

ECAS_VIZIER = "VII/91/g8color"

BANDS = [374, 418, 462, 506, 550, 594, 638, 682, 726, 770, 814, 858, 902, 946, 990, 1034]
REFL_COLS = [f"refl_{b}" for b in BANDS]
BAND_UM = [b / 1000.0 for b in BANDS]


def _load_ecas() -> pd.DataFrame:
    if ECAS_FILE.is_file():
        return pd.read_parquet(ECAS_FILE, engine="pyarrow")

    v = Vizier(columns=["*"], row_limit=-1)
    last_err: Exception | None = None
    t = None
    try:
        result = v.get_catalogs(ECAS_VIZIER)
        if len(result) and len(result[0]) > 0:
            t = result[0].to_pandas()
    except Exception as e:
        last_err = e
    if t is None:
        raise RuntimeError(f"ECAS Vizier download failed: {last_err}")

    rename = {"ID": "number_mp"}
    if "b-v" in t.columns:
        rename["b-v"] = "b_v_ecas"
    t = t.rename(columns=rename)
    t["number_mp"] = pd.to_numeric(t["number_mp"], errors="coerce").astype("Int64")
    t = t[t["number_mp"].notna()].copy()
    t["number_mp"] = t["number_mp"].astype(np.int64)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t.to_parquet(ECAS_FILE, engine="pyarrow", compression="snappy")
    print(f"ECAS: {len(t)} asteroids loaded → {ECAS_FILE}")
    return t


def main() -> int:
    t0 = time.time()
    if not INPUT_FILE.is_file():
        print(f"ERROR: {INPUT_FILE} not found", file=sys.stderr)
        return 1

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_gasp = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    n_gasp = len(df_gasp)

    ecas = _load_ecas()
    n_ecas = len(ecas)

    df_match = df_gasp.merge(
        ecas[["number_mp", "b_v_ecas"]],
        on="number_mp",
        how="inner",
    )
    n_match = len(df_match)
    print(f"GASP asteroids:    {n_gasp:,}")
    print(f"ECAS asteroids:    {n_ecas:,}")
    print(f"Matched:           {n_match:,}")

    if n_match < 5:
        print("ERROR: too few matches for validation", file=sys.stderr)
        return 1

    ok = df_match["b_v_ecas"].notna() & df_match["refl_418"].notna()
    x = df_match.loc[ok, "b_v_ecas"].astype(float).values
    y = df_match.loc[ok, "refl_418"].astype(float).values
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        r, p = float("nan"), float("nan")
    else:
        r, p = stats.pearsonr(x, y)
    print(f"Pearson r = {r:.3f}, p = {p:.4f}")

    mean_m = df_match[REFL_COLS].mean()
    mean_f = df_gasp[REFL_COLS].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(df_match["b_v_ecas"], df_match["refl_418"], alpha=0.6, s=30, color="steelblue")
    axes[0].set_xlabel("ECAS b-v color (ground truth)")
    axes[0].set_ylabel("GASP refl_418 (NUV-corrected)")
    axes[0].set_title(f"ECAS vs GASP NUV\nr={r:.3f}, p={p:.4f}")

    axes[1].plot(BAND_UM, mean_f.values, "gray", lw=2, label=f"Full GASP (n={n_gasp:,})")
    axes[1].plot(
        BAND_UM,
        mean_m.values,
        "steelblue",
        lw=2,
        label=f"ECAS-matched (n={len(df_match):,})",
    )
    axes[1].axvspan(0.374, 0.506, alpha=0.1, color="purple", label="NUV corrected")
    axes[1].set_xlabel("Wavelength (µm)")
    axes[1].set_ylabel("Mean reflectance")
    axes[1].set_title("Mean spectrum: ECAS subset vs full GASP")
    axes[1].legend(fontsize=9)

    plt.suptitle("GASP v1 — ECAS Validation", fontsize=13)
    plt.tight_layout()
    out_png = FIGURES_DIR / "05_ecas_validation.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")

    mean418_m = float(mean_m["refl_418"])
    mean418_f = float(mean_f["refl_418"])
    conclusion = "consistent" if r > 0.3 else "check data"

    duration = time.time() - t0
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gasp_phase": "09_ecas_validation",
        "ecas_asteroids": int(n_ecas),
        "matched_count": int(n_match),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "mean_refl_418_matched": mean418_m,
        "mean_refl_418_full": mean418_f,
        "conclusion": conclusion,
        "duration_seconds": round(duration, 3),
    }
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print()
    print("=== GASP ECAS Validation complete ===")
    print(f"Matched: {n_match} asteroids")
    print(f"Pearson r (ECAS b-v vs GASP refl_418): {r:.3f}")
    print(f"Conclusion: {conclusion}")
    print(f"Figure: {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

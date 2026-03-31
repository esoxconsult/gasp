#!/usr/bin/env python3
"""
NUV correction for Gaia DR3 SSO reflectance spectra (GASP Phase 4).

Scientific background (Tinaut-Ruano et al. 2023):
The Gaia DR3 solar analogues used to compute asteroid reflectance
spectra have a systematically redder spectral slope below 0.55 µm
compared to the well-characterised solar analogue Hyades 64.
This introduces an artificial drop in reflectance (fake reddening)
at NUV wavelengths.

Correction: divide the raw reflectance by the correction factor
at each affected wavelength band:

    corrected = raw / factor

This is a DIVISION (not multiplication) because the factor represents
the ratio SA_mean/Hyades64 by which the raw spectrum is inflated.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

INTERIM_DIR = Path("data/interim")
INPUT_FILE = INTERIM_DIR / "gaia_sso_wide.parquet"
OUTPUT_FILE = INTERIM_DIR / "gaia_sso_nuv_corrected.parquet"
LOG_FILE = INTERIM_DIR / "nuv_correction_log.json"

# NUV correction factors from Table 1, Tinaut-Ruano et al. (2023)
# DOI: 10.1051/0004-6361/202245134
# Correction: refl_corrected = refl_raw / factor
# Only bands below 0.55 µm are affected.
# At 0.550 µm the factor is 1.00 (no correction needed).
NUV_CORRECTION_FACTORS = {
    374.0: 1.07,
    418.0: 1.05,
    462.0: 1.02,
    506.0: 1.01,
    # 550.0: 1.00  # reference band, no correction
}


def main() -> int:
    t0 = time.time()

    if OUTPUT_FILE.exists():
        print(
            f"WARNING: {OUTPUT_FILE} already exists — not overwriting. "
            "Remove or rename it before re-running.",
            file=sys.stderr,
        )
        return 0

    if not INPUT_FILE.is_file():
        print(f"ERROR: input not found: {INPUT_FILE}", file=sys.stderr)
        return 1

    df = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    print(f"Loaded: shape {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    sanity_bands = list(NUV_CORRECTION_FACTORS.keys()) + [550.0]
    before_means: dict[str, float] = {}
    for wl in sanity_bands:
        col = f"refl_{int(wl)}"
        if col in df.columns:
            before_means[col] = float(df[col].mean())

    for wl, factor in NUV_CORRECTION_FACTORS.items():
        col = f"refl_{int(wl)}"
        if col in df.columns:
            df[col] = df[col] / factor
            print(f"  Corrected {col}: divided by {factor}")
        else:
            print(f"  WARNING: column {col} not found, skipping")

    df["nuv_correction_applied"] = True
    df["nuv_correction_source"] = "Tinaut-Ruano et al. (2023) Table 1"
    df["nuv_correction_doi"] = "10.1051/0004-6361/202245134"

    print()
    print("Sanity check (means):")
    print(
        "Band     | Before mean | After mean | Factor | Change %\n"
        "---------|-------------|------------|--------|----------"
    )
    refl_550_before = before_means.get("refl_550")
    refl_550_after = float(df["refl_550"].mean()) if "refl_550" in df.columns else float("nan")

    for wl in [374.0, 418.0, 462.0, 506.0]:
        col = f"refl_{int(wl)}"
        factor = NUV_CORRECTION_FACTORS[wl]
        b = before_means.get(col, float("nan"))
        a = float(df[col].mean()) if col in df.columns else float("nan")
        chg = (100.0 * (a - b) / b) if b and not np.isnan(b) else float("nan")
        print(
            f"{col:8} | {b:11.6f} | {a:10.6f} | {factor:4.2f} | {chg:7.2f}%"
        )

    b550 = refl_550_before
    a550 = refl_550_after
    chg550 = (100.0 * (a550 - b550) / b550) if b550 and not np.isnan(b550) else 0.0
    print(
        f"{'refl_550':8} | {b550:11.6f} | {a550:10.6f} | 1.00 | {chg550:6.1f}% (unchanged)"
    )

    print()
    if refl_550_before is not None and np.isfinite(refl_550_before):
        diff = abs(refl_550_after - refl_550_before)
        print(
            f"refl_550 mean before: {refl_550_before:.8f}  after: {refl_550_after:.8f}  "
            f"(diff {diff:.2e})"
        )
        if diff > 1e-10:
            print(
                "WARNING: refl_550 mean changed more than expected (should be unchanged).",
                file=sys.stderr,
            )

    refl_cols = [c for c in df.columns if c.startswith("refl_")]
    if refl_cols:
        n_cells = int((df[refl_cols] < 0).to_numpy().sum())
        n_rows_any_neg = int((df[refl_cols] < 0).any(axis=1).sum())
    else:
        n_cells = 0
        n_rows_any_neg = 0

    print(f"Negative reflectance values (cells): {n_cells}")
    print(f"Rows with any negative reflectance: {n_rows_any_neg}")

    df.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy")
    size_mb = OUTPUT_FILE.stat().st_size / (1024**2)
    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Shape: {len(df)} rows × {len(df.columns)} columns")
    print(f"File size: {size_mb:.2f} MB")

    duration = time.time() - t0
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gasp_phase": "04_nuv_correction",
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "correction_source": "Tinaut-Ruano et al. (2023) Table 1",
        "correction_doi": "10.1051/0004-6361/202245134",
        "bands_corrected": [374, 418, 462, 506],
        "factors_applied": {int(k): v for k, v in NUV_CORRECTION_FACTORS.items()},
        "n_asteroids": int(len(df)),
        "refl_550_mean_before": refl_550_before,
        "refl_550_mean_after": refl_550_after,
        "negative_values_after_correction": n_cells,
        "duration_seconds": round(duration, 3),
    }
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print()
    print("=== GASP Phase 4 complete ===")
    print("NUV correction applied (Tinaut-Ruano et al. 2023, Table 1)")
    print(
        "Bands corrected: refl_374 (/1.07), refl_418 (/1.05),\n"
        "                 refl_462 (/1.02), refl_506 (/1.01)"
    )
    print(f"Output: {OUTPUT_FILE}")
    print(f"Asteroids: {len(df):,}")
    print("Ready for Phase 5 (SDSS MOC4 cross-match).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

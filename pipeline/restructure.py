#!/usr/bin/env python3
"""GASP Phase 3: long-format Gaia SSO spectra → wide format (one row per asteroid)."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")

INPUT_FILE = RAW_DIR / "gaia_sso_raw.parquet"
OUTPUT_FILE = INTERIM_DIR / "gaia_sso_wide.parquet"
LOG_FILE = INTERIM_DIR / "restructure_log.json"

# The 16 Gaia wavelength bands in µm (sorted) — values match archive (nm scale as in Gaia table)
WAVELENGTHS = [
    374.0,
    418.0,
    462.0,
    506.0,
    550.0,
    594.0,
    638.0,
    682.0,
    726.0,
    770.0,
    814.0,
    858.0,
    902.0,
    946.0,
    990.0,
    1034.0,
]


def main() -> int:
    t0 = time.time()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

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

    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Shape: {len(df)} rows × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    wvals = sorted(df["wavelength"].dropna().unique())
    print(f"Unique wavelength values found: {wvals}")
    print(
        f"Unique number_mp (numbered asteroids): "
        f"{df['number_mp'].dropna().nunique()}"
    )
    print(f"Rows with number_mp IS NULL: {df['number_mp'].isna().sum()}")
    print(f"Memory usage: {mem_mb:.2f} MB")

    index_cols = ["denomination", "number_mp", "num_of_spectra", "nb_samples"]

    df_refl = df.pivot_table(
        index=index_cols,
        columns="wavelength",
        values="reflectance_spectrum",
        aggfunc="mean",
    )
    df_err = df.pivot_table(
        index=index_cols,
        columns="wavelength",
        values="reflectance_spectrum_err",
        aggfunc="mean",
    )

    def refl_name(w: float) -> str:
        return f"refl_{int(round(w))}"

    def err_name(w: float) -> str:
        return f"err_{int(round(w))}"

    df_refl = df_refl.rename(columns={c: refl_name(float(c)) for c in df_refl.columns})
    df_err = df_err.rename(columns={c: err_name(float(c)) for c in df_err.columns})

    df_wide = df_refl.join(df_err, how="outer")
    df_wide = df_wide.reset_index()

    expected_refl = [f"refl_{int(w)}" for w in WAVELENGTHS]
    expected_err = [f"err_{int(w)}" for w in WAVELENGTHS]
    for c in expected_refl + expected_err:
        if c not in df_wide.columns:
            df_wide[c] = np.nan

    refl_cols = expected_refl

    df_wide["n_valid_bands"] = df_wide[refl_cols].notna().sum(axis=1)

    df_wide["quality"] = pd.cut(
        df_wide["n_valid_bands"],
        bins=[-1, 8, 12, 16],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )

    df_wide["norm_ok"] = df_wide["refl_550"].between(0.95, 1.05)

    n = len(df_wide)
    qh = int((df_wide["quality"] == "high").sum())
    qm = int((df_wide["quality"] == "medium").sum())
    ql = int((df_wide["quality"] == "low").sum())

    def pct(x: int) -> float:
        return 100.0 * x / n if n else 0.0

    print()
    print("Quality distribution:")
    print(f"  high   (13–16 bands): {qh} asteroids ({pct(qh):.1f}%)")
    print(f"  medium  (9–12 bands): {qm} asteroids ({pct(qm):.1f}%)")
    print(f"  low      (1–8 bands): {ql} asteroids ({pct(ql):.1f}%)")
    print()
    norm_n = int(df_wide["norm_ok"].sum())
    print(f"norm_ok (reflectance at 550nm ≈ 1.0): {norm_n} asteroids")
    print()
    num_named = int(df_wide["number_mp"].notna().sum())
    num_unnamed = int(df_wide["number_mp"].isna().sum())
    print(f"Numbered asteroids (number_mp not null): {num_named}")
    print(f"Unnamed asteroids (number_mp is null):  {num_unnamed}")

    table = pa.Table.from_pandas(df_wide, preserve_index=False)
    pq.write_table(table, OUTPUT_FILE, compression="snappy")

    size_mb = OUTPUT_FILE.stat().st_size / (1024**2)
    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Shape: {len(df_wide)} rows × {len(df_wide.columns)} columns")
    print(f"File size: {size_mb:.2f} MB")

    duration = time.time() - t0
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gasp_phase": "03_restructure",
        "input_rows": int(len(df)),
        "output_rows": int(len(df_wide)),
        "output_columns": list(df_wide.columns),
        "quality_high": qh,
        "quality_medium": qm,
        "quality_low": ql,
        "norm_ok_count": norm_n,
        "duration_seconds": round(duration, 3),
    }
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print()
    print("=== GASP Phase 3 complete ===")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Asteroids: {len(df_wide)}")
    print(f"Columns: {len(df_wide.columns)}")
    print("Ready for Phase 4 (NUV correction).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

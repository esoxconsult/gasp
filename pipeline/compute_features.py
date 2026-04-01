#!/usr/bin/env python3
"""GASP Phase 7: derived spectral slopes, NUV S/N proxy, SDSS C/S classification."""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FINAL_DIR = Path("data/final")
INPUT_FILE = FINAL_DIR / "gasp_catalog_v1.parquet"
BACKUP_FILE = FINAL_DIR / "gasp_catalog_v1_pre_features.parquet"
LOG_FILE = FINAL_DIR / "features_log.json"

REFL_BANDS = [
    374, 418, 462, 506, 550, 594, 638, 682,
    726, 770, 814, 858, 902, 946, 990, 1034,
]


def _polyfit_slope_per_row(
    df: pd.DataFrame, cols: list[str], wls_um: list[float]
) -> np.ndarray:
    """Least-squares slope (µm⁻¹) per row; NaN if any band missing."""
    x = np.asarray(wls_um, dtype=np.float64)
    y = df[cols].to_numpy(dtype=np.float64)
    n = y.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        row = y[i]
        if np.any(~np.isfinite(row)):
            continue
        coef = np.polyfit(x, row, 1)
        out[i] = float(coef[0])
    return out


def _print_slope_table(df: pd.DataFrame) -> None:
    print()
    print("Slope | Mean   | Std    | Min    | Max    | NaN count")
    print("------|--------|--------|--------|--------|----------")
    for col in ("s1", "s2", "s3", "s4"):
        s = df[col]
        nan_c = int(s.isna().sum())
        if s.notna().any():
            mean_v = float(s.mean())
            std_v = float(s.std())
            min_v = float(s.min())
            max_v = float(s.max())
        else:
            mean_v = std_v = min_v = max_v = float("nan")
        print(
            f"{col:5} | {mean_v:6.3f} | {std_v:6.3f} | {min_v:6.3f} | {max_v:6.3f} | {nan_c}"
        )
    print()


def _print_nuv_distribution(df: pd.DataFrame, n_total: int) -> None:
    print("NUV S/N score distribution:")
    for score, label in [
        (3, "Score 3 (reliable,  ≥10 spectra)"),
        (2, "Score 2 (acceptable, 5-9)"),
        (1, "Score 1 (marginal,   3-4)"),
        (0, "Score 0 (unreliable, <3)"),
    ]:
        n = int((df["nuv_snr_score"] == score).sum())
        pct = 100.0 * n / n_total if n_total else 0.0
        print(f"  {label}: {n} ({pct:.1f}%)")
    print()


def _print_sdss_complex_distribution(df: pd.DataFrame, n_total: int) -> None:
    print("SDSS color classification:")
    for cat, label in [
        ("C-complex", "C-complex  (a* < -0.10)"),
        ("ambiguous", "Ambiguous  (-0.10–0.10)"),
        ("S-complex", "S-complex  (a* >  0.10)"),
    ]:
        n = int((df["sdss_complex"] == cat).sum())
        pct = 100.0 * n / n_total if n_total else 0.0
        print(f"  {label}: {n} ({pct:.1f}%)")
    print()


def _taxonomy_sdss_crossval(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """C-types vs C-complex, S-types vs S-complex agreement (%)."""
    sub = df[df["taxonomy"].notna() & df["sdss_complex"].notna()].copy()
    if sub.empty:
        print("  Taxonomy vs SDSS: no rows with both taxonomy and sdss_complex.")
        return None, None
    tax = sub["taxonomy"].astype(str).str.strip()
    sdss = sub["sdss_complex"].astype(str)
    c_like = tax.str.match(r"^C", na=False) | tax.str.upper().eq("B")
    s_like = tax.str.match(r"^S", na=False)
    agree_c: float | None = None
    agree_s: float | None = None
    if c_like.sum() > 0:
        agree_c = float((sdss[c_like] == "C-complex").mean() * 100.0)
        print(f"  Taxonomy C* types vs sdss C-complex agreement: {agree_c:.1f}%")
    else:
        print("  Taxonomy C* types vs sdss C-complex agreement: n/a (no C-types)")
    if s_like.sum() > 0:
        agree_s = float((sdss[s_like] == "S-complex").mean() * 100.0)
        print(f"  Taxonomy S type  vs sdss S-complex agreement:  {agree_s:.1f}%")
    else:
        print("  Taxonomy S type  vs sdss S-complex agreement:  n/a (no S-types)")
    return agree_c, agree_s


def main() -> int:
    t0 = time.time()
    if not INPUT_FILE.is_file():
        print(f"ERROR: input not found: {INPUT_FILE}", file=sys.stderr)
        return 1

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT_FILE, BACKUP_FILE)
    print(f"Backup saved: {BACKUP_FILE}")

    df = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    n0 = len(df)
    print(f"Loaded: {INPUT_FILE} shape {df.shape}")

    refl_cols = [f"refl_{b}" for b in REFL_BANDS]
    missing = [c for c in refl_cols if c not in df.columns]
    if missing:
        print(f"ERROR: missing reflectance columns: {missing[:10]}...", file=sys.stderr)
        return 1
    print("All refl_* bands present (16 bands).")

    # --- s1: two-point slope 374–418 nm
    d_wl = 0.418 - 0.374
    ok1 = df["refl_374"].notna() & df["refl_418"].notna()
    s1 = ((df["refl_418"] - df["refl_374"]) / d_wl).where(ok1)
    df["s1"] = s1.astype("float64")

    # --- s2, s3, s4: polyfit
    s2_cols = ["refl_418", "refl_462", "refl_506", "refl_550"]
    s2_wl = [0.418, 0.462, 0.506, 0.550]
    df["s2"] = _polyfit_slope_per_row(df, s2_cols, s2_wl)

    s3_cols = ["refl_550", "refl_594", "refl_638", "refl_682"]
    s3_wl = [0.550, 0.594, 0.638, 0.682]
    df["s3"] = _polyfit_slope_per_row(df, s3_cols, s3_wl)

    s4_cols = [
        "refl_726", "refl_770", "refl_814", "refl_858",
        "refl_902", "refl_946", "refl_990", "refl_1034",
    ]
    s4_wl = [0.726, 0.770, 0.814, 0.858, 0.902, 0.946, 0.990, 1.034]
    df["s4"] = _polyfit_slope_per_row(df, s4_cols, s4_wl)

    _print_slope_table(df)

    # --- NUV S/N proxy
    if "num_of_spectra" not in df.columns:
        print("ERROR: num_of_spectra missing", file=sys.stderr)
        return 1
    n_spec = pd.to_numeric(df["num_of_spectra"], errors="coerce").fillna(0)
    df["nuv_snr_flag"] = n_spec >= 5
    score_cat = pd.cut(
        n_spec,
        bins=[-1, 2, 4, 9, 999],
        labels=[0, 1, 2, 3],
        include_lowest=True,
    )
    df["nuv_snr_score"] = score_cat.astype(np.int64)

    _print_nuv_distribution(df, n0)

    # --- SDSS C/S
    if "a_star" not in df.columns:
        print("ERROR: a_star missing", file=sys.stderr)
        return 1
    a_star = pd.to_numeric(df["a_star"], errors="coerce")
    df["sdss_complex"] = pd.cut(
        a_star,
        bins=[-99.0, -0.10, 0.10, 99.0],
        labels=["C-complex", "ambiguous", "S-complex"],
    ).astype("object")

    _print_sdss_complex_distribution(df, n0)

    print("Taxonomy vs SDSS cross-validation:")
    _taxonomy_sdss_crossval(df)

    new_cols = [
        "s1", "s2", "s3", "s4",
        "nuv_snr_flag", "nuv_snr_score", "sdss_complex",
    ]
    df.to_parquet(INPUT_FILE, engine="pyarrow", compression="snappy")
    size_mb = INPUT_FILE.stat().st_size / (1024**2)
    n_cols = len(df.columns)

    print()
    print(f"Updated: {INPUT_FILE}")
    print(
        "New columns added: s1, s2, s3, s4,\n"
        "                   nuv_snr_flag, nuv_snr_score,\n"
        "                   sdss_complex"
    )
    print(f"Total columns: {n_cols}")
    print(f"File size: {size_mb:.2f} MB")

    duration = time.time() - t0
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gasp_phase": "07_compute_features",
        "new_columns": new_cols,
        "s1_mean": float(df["s1"].mean()) if df["s1"].notna().any() else None,
        "s2_mean": float(df["s2"].mean()) if df["s2"].notna().any() else None,
        "s3_mean": float(df["s3"].mean()) if df["s3"].notna().any() else None,
        "s4_mean": float(df["s4"].mean()) if df["s4"].notna().any() else None,
        "nuv_snr_score_3_count": int((df["nuv_snr_score"] == 3).sum()),
        "sdss_c_complex_count": int((df["sdss_complex"] == "C-complex").sum()),
        "sdss_s_complex_count": int((df["sdss_complex"] == "S-complex").sum()),
        "duration_seconds": round(duration, 3),
        "total_columns": int(n_cols),
    }
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print()
    print("=== GASP Features complete ===")
    print("Added: s1, s2, s3, s4 (spectral slopes)")
    print("Added: nuv_snr_flag, nuv_snr_score (S/N reliability)")
    print("Added: sdss_complex (C/S classification)")
    print(f"Total columns: {n_cols}")
    print(f"Output: {INPUT_FILE}")
    print("Ready for notebook exploration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

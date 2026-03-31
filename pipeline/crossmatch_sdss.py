#!/usr/bin/env python3
"""GASP Phase 5: download SDSS MOC4 and cross-match with Gaia on number_mp."""

from __future__ import annotations

import gzip
import json
import sys
import time
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

INTERIM_DIR = Path("data/interim")
RAW_DIR = Path("data/raw")
INPUT_FILE = INTERIM_DIR / "gaia_sso_nuv_corrected.parquet"
SDSS_FILE = RAW_DIR / "sdss_moc4.parquet"
OUTPUT_FILE = INTERIM_DIR / "gaia_sdss_merged.parquet"
LOG_FILE = INTERIM_DIR / "crossmatch_log.json"

ADR4_URL = "https://faculty.washington.edu/ivezic/sdssmoc/ADR4.dat.gz"
CDS_README_URL = "https://cdsarc.cds.unistra.fr/ftp/II/275/ReadMe"


def _field(line: str, start_1: int, end_1_exclusive: int) -> str:
    """Extract fixed-width field; start/end are 1-based, end exclusive (VizieR-style table)."""
    return line[start_1 - 1 : end_1_exclusive - 1].strip()


def parse_adr4_line(line: str) -> dict[str, float | int | str] | None:
    """Parse one ADR4 (SDSS MOC4) line using fixed columns from Ivezic MOC4 format."""
    if len(line) < 252:
        return None
    try:
        num_str = _field(line, 245, 253)
        numeration = int(num_str) if num_str and num_str != "-" else 0
    except ValueError:
        numeration = 0

    def ffloat(s: str) -> float | np.nan:
        if s in {"", "-", "--"}:
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    # Fixed columns from Ivezic MOC4 format (ADR4.dat), 1-based [start, end)
    return {
        "number_mp": numeration,
        "u": ffloat(_field(line, 164, 169)),
        "e_u": ffloat(_field(line, 170, 175)),
        "g": ffloat(_field(line, 175, 180)),
        "e_g": ffloat(_field(line, 181, 186)),
        "r": ffloat(_field(line, 186, 191)),
        "e_r": ffloat(_field(line, 192, 197)),
        "i": ffloat(_field(line, 197, 202)),
        "e_i": ffloat(_field(line, 203, 207)),
        "z": ffloat(_field(line, 208, 213)),
        "e_z": ffloat(_field(line, 214, 219)),
        "a_star": ffloat(_field(line, 219, 224)),
        "e_a_star": ffloat(_field(line, 225, 230)),
        "V": ffloat(_field(line, 231, 236)),
        "B": ffloat(_field(line, 237, 242)),
        "moID": _field(line, 1, 8),
    }


def _open_adr4_text(path: Path):
    """Open ADR4 as gzip or plain text (magic 0x1f8b = gzip)."""
    sig = path.read_bytes()[:2]
    if sig == b"\x1f\x8b":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "rt", errors="replace")


def load_sdss_from_adr4_gz(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with _open_adr4_text(path) as f:
        for line in tqdm(f, desc="Parsing ADR4", unit="lines"):
            rec = parse_adr4_line(line)
            if rec is None:
                continue
            rows.append(rec)
    return pd.DataFrame(rows)


def download_adr4(dest_gz: Path) -> None:
    print(f"Downloading {ADR4_URL} …")
    # Avoid transparent gzip decode so we store true .gz bytes (~64 MB, not ~290 MB decompressed).
    r = requests.get(
        ADR4_URL,
        timeout=600,
        headers={"Accept-Encoding": "identity"},
    )
    r.raise_for_status()
    data = r.content
    dest_gz.parent.mkdir(parents=True, exist_ok=True)
    dest_gz.write_bytes(data)
    print(f"Downloaded {len(data) / (1024**2):.1f} MB")


def try_vizier_ii275() -> pd.DataFrame | None:
    """Attempt VizieR II/275 as requested; VizieR II/275 is IRAS, not SDSS MOC4."""
    from astroquery.vizier import Vizier
    import astropy.units as u  # noqa: F401

    print("Trying Vizier catalog II/275 …")
    Vizier.ROW_LIMIT = -1
    v = Vizier(columns=["*"])
    catalog_list = v.get_catalogs("II/275")
    if len(catalog_list) == 0:
        return None
    t = catalog_list[0]
    cols = list(t.colnames)
    if any("IRAS" in str(c).upper() for c in cols[:3]):
        print(
            "VizieR II/275 resolves to IRAS Faint Source Reject Catalog, "
            "not SDSS MOC4. Falling back to official SDSS MOC4 (ADR4.dat.gz)."
        )
        return None
    return t.to_pandas()


def ensure_sdss_table() -> tuple[pd.DataFrame, str]:
    """Return SDSS dataframe and provenance label."""
    if SDSS_FILE.is_file():
        print("SDSS MOC4 already downloaded, skipping.")
        df = pd.read_parquet(SDSS_FILE, engine="pyarrow")
        return df, "cached_parquet"

    df_v = None
    try:
        df_v = try_vizier_ii275()
    except Exception as e:
        print(f"Vizier II/275 failed: {e}")

    if df_v is not None and len(df_v) > 0 and "number_mp" in df_v.columns:
        print(f"Using Vizier II/275 table: {len(df_v)} rows")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df_v.to_parquet(SDSS_FILE, engine="pyarrow", compression="snappy")
        return df_v, "vizier_II_275"

    with tempfile.TemporaryDirectory() as tmp:
        gz = Path(tmp) / "ADR4.dat.gz"
        download_adr4(gz)
        df = load_sdss_from_adr4_gz(gz)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SDSS_FILE, engine="pyarrow", compression="snappy")
    return df, "ADR4.dat.gz (SDSS MOC4, Ivezic UW)"


def detect_number_column(df: pd.DataFrame) -> str | None:
    for name in ("number_mp", "Num", "MPC", "Numeration", "num", "NUMBER", "No"):
        if name in df.columns:
            return name
    return None


def prepare_sdss(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numbered rows, rename key, select photometry columns."""
    col_num = detect_number_column(df)
    if col_num is None:
        raise ValueError("Could not find asteroid number column in SDSS table.")

    out = df.copy()
    if col_num != "number_mp":
        out = out.rename(columns={col_num: "number_mp"})

    out["number_mp"] = pd.to_numeric(out["number_mp"], errors="coerce")
    out = out[(out["number_mp"].notna()) & (out["number_mp"] > 0)].copy()
    out["number_mp"] = out["number_mp"].astype(np.int64)

    aliases = {
        "umag": "u",
        "gmag": "g",
        "rmag": "r",
        "imag": "i",
        "zmag": "z",
        "astar": "a_star",
        "a": "a_star",
    }
    for a, b in aliases.items():
        if a in out.columns and b not in out.columns:
            out = out.rename(columns={a: b})

    found: list[str] = []
    missing: list[str] = []
    keep: list[str] = ["number_mp"]
    for logical in ("u", "g", "r", "i", "z", "a_star", "V"):
        if logical in out.columns:
            keep.append(logical)
            found.append(logical)
        else:
            missing.append(logical)

    for extra in ("e_u", "e_g", "e_r", "e_i", "e_z", "e_a_star", "B", "moID"):
        if extra in out.columns:
            keep.append(extra)

    out = out[keep].copy()
    print(f"SDSS columns kept: {found}")
    if missing:
        print(f"SDSS columns missing (optional): {missing}")

    agg_map: dict[str, str] = {c: "mean" for c in out.columns if c != "number_mp" and out[c].dtype != object}
    for c in out.columns:
        if c == "number_mp":
            continue
        if c not in agg_map:
            agg_map[c] = "first"

    sdss_g = out.groupby("number_mp", as_index=False).agg(agg_map)
    return sdss_g


def main() -> int:
    t0 = time.time()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.is_file():
        print(f"ERROR: Gaia input not found: {INPUT_FILE}", file=sys.stderr)
        return 1

    print("Downloading SDSS MOC4 via Vizier (catalog II/275) …")
    try:
        sdss_raw, prov = ensure_sdss_table()
    except Exception as e:
        print(
            f"ERROR: could not obtain SDSS MOC4: {e}\n"
            f"Fallback: download ADR4 manually from {ADR4_URL}\n"
            f"Or inspect CDS ReadMe: {CDS_README_URL}",
            file=sys.stderr,
        )
        return 1

    print(f"SDSS load provenance: {prov}")
    print(f"SDSS rows: {len(sdss_raw)}, columns: {list(sdss_raw.columns)}")
    print()
    print("SDSS dtypes:")
    print(sdss_raw.dtypes)
    print()

    col_num = detect_number_column(sdss_raw)
    print(f"Asteroid number column detected: {col_num!r}")

    n_tot = len(sdss_raw)
    if col_num:
        nmp = pd.to_numeric(sdss_raw[col_num], errors="coerce")
        n_valid = int(((nmp.notna()) & (nmp > 0)).sum())
        n_non = int(n_tot - n_valid)
    else:
        n_valid = 0
        n_non = n_tot

    print(f"Total rows in SDSS MOC4: {n_tot}")
    print(f"Rows with valid asteroid number (>0): {n_valid}")
    print(f"Rows without asteroid number: {n_non}")
    print()

    sdss = prepare_sdss(sdss_raw)
    print(f"SDSS after dedup on number_mp: {len(sdss)} rows")
    print()

    gaia = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    print(f"Loaded Gaia: shape {gaia.shape}")

    gaia["number_mp"] = pd.to_numeric(gaia["number_mp"], errors="coerce").astype("Int64")
    gaia_n = gaia[gaia["number_mp"].notna()].copy()
    gaia_n["number_mp"] = gaia_n["number_mp"].astype(np.int64)

    n_gaia = len(gaia_n)
    n_sdss = len(sdss)
    merged = gaia_n.merge(sdss, on="number_mp", how="inner", suffixes=("", "_sdss"))
    n_match = len(merged)

    rate = 100.0 * n_match / n_gaia if n_gaia else 0.0

    print("Cross-match:")
    print(f"  Gaia asteroids:              {n_gaia:,}")
    print(f"  SDSS numbered observations:  {n_sdss:,}")
    print(f"  Matched (inner join):        {n_match:,}")
    print(f"  Match rate:                  {rate:.2f}% of Gaia objects found in SDSS")
    print()

    nuv_cols = ["refl_374", "refl_418", "refl_462", "refl_506"]
    merged["nuv_negative_flag"] = (merged[nuv_cols] < 0).any(axis=1)
    merged["nuv_reliable"] = ~merged["nuv_negative_flag"]

    n_rel = int(merged["nuv_reliable"].sum())
    n_art = int(merged["nuv_negative_flag"].sum())
    pct_rel = 100.0 * n_rel / n_match if n_match else 0.0

    print("NUV quality (merged sample):")
    print(f"  Objects with reliable NUV:  {n_rel:,} ({pct_rel:.2f}%)")
    print(f"  Objects with NUV artifacts: {n_art:,} ({100.0 - pct_rel:.2f}%)")
    print()

    merged.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy")
    size_mb = OUTPUT_FILE.stat().st_size / (1024**2)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Shape: {len(merged)} rows × {len(merged.columns)} columns")
    print(f"File size: {size_mb:.2f} MB")
    print()

    cols = list(merged.columns)
    refl = [c for c in cols if c.startswith("refl_")]
    err = [c for c in cols if c.startswith("err_")]
    meta_gaia = [
        c
        for c in cols
        if c
        in (
            "num_of_spectra",
            "nb_samples",
            "quality",
            "norm_ok",
            "n_valid_bands",
        )
    ]
    nuv_meta = [
        c
        for c in cols
        if c
        in (
            "nuv_correction_applied",
            "nuv_correction_source",
            "nuv_correction_doi",
            "nuv_reliable",
            "nuv_negative_flag",
        )
    ]
    sdss_cols = [
        c
        for c in cols
        if c
        in (
            "u",
            "g",
            "r",
            "i",
            "z",
            "a_star",
            "V",
            "e_u",
            "e_g",
            "e_r",
            "e_i",
            "e_z",
            "B",
            "moID",
        )
    ]

    print("Dataset overview (column groups):")
    print(f"  Identifiers:  denomination, number_mp")
    print(f"  Gaia spectra: {sorted(refl)}")
    print(f"  Gaia errors:  {sorted(err)}")
    print(f"  Gaia meta:    {meta_gaia}")
    print(f"  NUV flags:    {nuv_meta}")
    print(f"  SDSS phot:    {sdss_cols}")
    print()

    duration = time.time() - t0
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gasp_phase": "05_crossmatch_sdss",
        "sdss_catalog": "SDSS MOC4 ADR4.dat.gz (Ivezic UW); VizieR II/275 is IRAS, not MOC4",
        "sdss_load_provenance": prov,
        "gaia_input_rows": int(n_gaia),
        "sdss_total_rows": int(n_tot),
        "sdss_numbered_rows": int(n_valid),
        "matched_rows": int(n_match),
        "match_rate_pct": round(float(rate), 4),
        "nuv_reliable_count": n_rel,
        "nuv_artifact_count": n_art,
        "output_columns": cols,
        "duration_seconds": round(duration, 3),
    }
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print("=== GASP Phase 5 complete ===")
    print("Gaia × SDSS MOC4 cross-match done.")
    print(f"Matched asteroids: {n_match:,}")
    print(f"NUV-reliable subset: {n_rel:,} ({pct_rel:.2f}%)")
    print(f"Output: {OUTPUT_FILE}")
    print("Ready for Phase 6 (asteroid metadata via rocks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

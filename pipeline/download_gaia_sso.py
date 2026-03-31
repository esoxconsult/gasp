#!/usr/bin/env python3
"""GASP Phase 2: chunked download of Gaia DR3 SSO reflectance spectra."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astroquery.gaia import Gaia
from tqdm import tqdm

RAW_DIR = Path("data/raw")
COLUMNS = [
    "denomination",
    "number_mp",
    "nb_samples",
    "num_of_spectra",
    "reflectance_spectrum",
    "reflectance_spectrum_err",
    "wavelength",
]


def retry_query(query: str, max_retries: int, label: str) -> pd.DataFrame:
    """Async TAP job with exponential backoff; sleep(2) after success."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            job = Gaia.launch_job_async(query)
            tbl = job.get_results()
            df = tbl.to_pandas()
            time.sleep(2)
            return df
        except Exception as e:
            last_exc = e
            if attempt >= max_retries:
                break
            wait_s = 10 * (2 ** (attempt - 1))
            print(
                f"[{label}] attempt {attempt}/{max_retries} failed: {e!s}; "
                f"waiting {wait_s}s before retry",
                file=sys.stderr,
            )
            time.sleep(wait_s)
    assert last_exc is not None
    raise last_exc


def chunk_boundaries(min_mp: int, max_mp: int, chunk_size: int) -> list[tuple[int, int]]:
    """Half-open ranges [start, end) in number_mp space."""
    chunks: list[tuple[int, int]] = []
    current = min_mp
    while current <= max_mp:
        chunk_end = min(current + chunk_size, max_mp + 1)
        if chunk_end <= current:
            break
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def chunk_path(start: int, end: int) -> Path:
    return RAW_DIR / f"chunk_{start}_{end}.parquet"


def chunk_path_null() -> Path:
    return RAW_DIR / "chunk_NULL.parquet"


def is_valid_nonempty_parquet(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        meta = pq.read_metadata(path)
        return meta.num_rows > 0
    except Exception:
        return False


def ensure_raw_dir() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def raw_dir_writable() -> bool:
    ensure_raw_dir()
    test = RAW_DIR / ".write_test_gasp"
    try:
        test.write_text("ok")
        test.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def plan_chunks(
    min_mp: int, max_mp: int, chunk_size: int
) -> tuple[list[tuple[int, int]], bool]:
    """Numbered chunks plus whether a NULL chunk is needed (always True per spec)."""
    numbered = chunk_boundaries(min_mp, max_mp, chunk_size)
    return numbered, True


def run_dry_run(chunk_size: int, max_retries: int) -> None:
    ensure_raw_dir()
    Gaia.TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

    bounds_q = """
SELECT MIN(number_mp) AS min_mp, MAX(number_mp) AS max_mp
FROM gaiadr3.sso_reflectance_spectrum
WHERE number_mp IS NOT NULL
"""
    df_bounds = retry_query(bounds_q, max_retries, "bounds")
    min_mp = int(df_bounds["min_mp"].iloc[0])
    max_mp = int(df_bounds["max_mp"].iloc[0])

    numbered, _has_null = plan_chunks(min_mp, max_mp, chunk_size)

    count_q = "SELECT COUNT(*) AS n FROM gaiadr3.sso_reflectance_spectrum"
    df_count = retry_query(count_q, max_retries, "count")
    est_rows = int(df_count["n"].iloc[0])

    total_chunks = len(numbered) + 1

    print("=== GASP download_gaia_sso.py — DRY RUN ===")
    print(f"min_mp (archive): {min_mp}")
    print(f"max_mp (archive): {max_mp}")
    print(f"chunk_size: {chunk_size}")
    print(f"total chunks planned (numbered + NULL): {total_chunks}")
    print(f"estimated total rows (COUNT(*)): {est_rows}")
    print()
    print("First 5 chunk ranges (number_mp):")
    for i, (a, b) in enumerate(numbered[:5]):
        print(f"  {i + 1}. [{a}, {b})  ->  chunk_{a}_{b}.parquet")
    if len(numbered) > 5:
        print("  ...")
    print("Last 5 chunk ranges (number_mp):")
    for a, b in numbered[-5:]:
        print(f"  [{a}, {b})  ->  chunk_{a}_{b}.parquet")
    print("Final chunk: WHERE number_mp IS NULL -> chunk_NULL.parquet")
    print()
    w_ok = raw_dir_writable()
    print(f"data/raw/ exists and is writable: {w_ok}")
    print(f"Absolute path: {RAW_DIR.resolve()}")
    print()
    print("Dry-run complete — no files downloaded.")


def download_chunk(
    start: int,
    end: int,
    max_retries: int,
    chunk_idx: int,
    total_numbered: int,
    cumulative_rows: list[int],
) -> tuple[bool, bool]:
    """Returns (success, was_skipped)."""
    path = chunk_path(start, end)
    label = f"chunk[{start},{end})"
    if is_valid_nonempty_parquet(path):
        print(f"Skipping chunk {start}–{end} (already downloaded)")
        try:
            n = pq.read_metadata(path).num_rows
        except Exception:
            n = 0
        cumulative_rows[0] += n
        return True, True

    q = f"""
SELECT denomination, number_mp, nb_samples, num_of_spectra,
       reflectance_spectrum, reflectance_spectrum_err, wavelength
FROM gaiadr3.sso_reflectance_spectrum
WHERE number_mp >= {start} AND number_mp < {end}
"""
    df = retry_query(q, max_retries, label)
    n = len(df)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    cumulative_rows[0] += n
    print(
        f"Chunk {chunk_idx}/{total_numbered}  range [{start}, {end})  "
        f"rows={n}  cumulative={cumulative_rows[0]}"
    )
    return True, False


def download_null_chunk(max_retries: int, cumulative_rows: list[int]) -> tuple[bool, bool]:
    """Returns (success, was_skipped)."""
    path = chunk_path_null()
    label = "chunk_NULL"
    if is_valid_nonempty_parquet(path):
        print("Skipping chunk NULL (number_mp IS NULL) (already downloaded)")
        try:
            n = pq.read_metadata(path).num_rows
        except Exception:
            n = 0
        cumulative_rows[0] += n
        return True, True

    q = """
SELECT denomination, number_mp, nb_samples, num_of_spectra,
       reflectance_spectrum, reflectance_spectrum_err, wavelength
FROM gaiadr3.sso_reflectance_spectrum
WHERE number_mp IS NULL
"""
    df = retry_query(q, max_retries, label)
    n = len(df)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")
    cumulative_rows[0] += n
    print(f"Chunk NULL  rows={n}  cumulative={cumulative_rows[0]}")
    return True, False


def concatenate_and_finalize(
    out_path: Path,
    chunks_failed: list[str],
    chunks_skipped: int,
    chunks_total: int,
    t0: float,
    max_retries: int,
) -> int:
    if out_path.exists():
        print(
            f"WARNING: {out_path} already exists — not overwriting. "
            "Remove or rename it before concatenation.",
            file=sys.stderr,
        )
        return 1

    chunk_files = sorted(RAW_DIR.glob("chunk_*.parquet"))
    if not chunk_files:
        print("No chunk_*.parquet files found to concatenate.", file=sys.stderr)
        return 1

    try:
        dfs = []
        for p in tqdm(chunk_files, desc="Loading chunks"):
            dfs.append(pd.read_parquet(p))
        combined = pd.concat(dfs, ignore_index=True)
        table = pa.Table.from_pandas(combined, preserve_index=False)
        pq.write_table(table, out_path, compression="snappy")
    except Exception as e:
        print(
            f"Concatenation failed: {e}\n"
            "Chunk parquet files were kept under data/raw/.\n"
            "Fix the issue and re-run; existing chunks will be skipped.",
            file=sys.stderr,
        )
        return 1

    for p in chunk_files:
        try:
            p.unlink()
        except OSError as e:
            print(f"Warning: could not delete {p}: {e}", file=sys.stderr)

    total_rows = len(combined)
    astro_v = importlib.metadata.version("astroquery")

    num_col = combined["number_mp"]
    unique_numbered = num_col.dropna().nunique()
    unique_denom = combined["denomination"].nunique()

    wl_min = float("inf")
    wl_max = float("-inf")
    for x in combined["wavelength"]:
        if x is None:
            continue
        a = np.asarray(x, dtype=float)
        if a.size == 0:
            continue
        wl_min = min(wl_min, float(np.nanmin(a)))
        wl_max = max(wl_max, float(np.nanmax(a)))
    if wl_min == float("inf"):
        wl_min = float("nan")
    if wl_max == float("-inf"):
        wl_max = float("nan")

    null_refl = 0
    if "reflectance_spectrum" in combined.columns:
        for v in combined["reflectance_spectrum"]:
            if v is None:
                null_refl += 1
                continue
            arr = np.asarray(v, dtype=float)
            if arr.size == 0 or np.isnan(arr).any():
                null_refl += 1

    size_mb = out_path.stat().st_size / (1024 * 1024)

    print()
    print("--- Validation ---")
    print(f"Total rows: {total_rows}")
    print(f"Unique asteroids (numbered): {unique_numbered}")
    print(f"Unique denominations: {unique_denom}")
    print(f"Wavelength range: {wl_min:.6f} – {wl_max:.6f} µm")
    print(f"Rows with null reflectance (approx): {null_refl}")
    print(f"File size: {size_mb:.2f} MB")

    duration = time.time() - t0
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gasp_phase": "02_download",
        "astroquery_version": astro_v,
        "total_rows": total_rows,
        "unique_asteroids": int(unique_numbered),
        "chunks_total": chunks_total,
        "chunks_skipped": chunks_skipped,
        "chunks_failed": chunks_failed,
        "archive_warning": True,
        "duration_seconds": round(duration, 3),
    }
    log_path = RAW_DIR / "download_log.json"
    log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print()
    print("=== GASP Phase 2 complete ===")
    print(f"Output: {out_path}")
    print(f"Rows: {total_rows}")
    print(f"Unique asteroids: {int(unique_numbered)}")
    print(f"File size: {size_mb:.2f} MB")
    print("Ready for Phase 3 (restructure to wide format).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Gaia DR3 SSO spectra (chunked).")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_raw_dir()
    Gaia.TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

    if args.dry_run:
        run_dry_run(args.chunk_size, args.max_retries)
        return 0

    t0 = time.time()
    chunks_failed: list[str] = []
    chunks_skipped = 0
    cumulative_rows: list[int] = [0]

    bounds_q = """
SELECT MIN(number_mp) AS min_mp, MAX(number_mp) AS max_mp
FROM gaiadr3.sso_reflectance_spectrum
WHERE number_mp IS NOT NULL
"""
    df_bounds = retry_query(bounds_q, args.max_retries, "bounds")
    min_mp = int(df_bounds["min_mp"].iloc[0])
    max_mp = int(df_bounds["max_mp"].iloc[0])
    numbered, _ = plan_chunks(min_mp, max_mp, args.chunk_size)
    total_numbered = len(numbered)

    for idx, (start, end) in enumerate(numbered, start=1):
        try:
            _ok, skipped = download_chunk(
                start,
                end,
                args.max_retries,
                idx,
                total_numbered,
                cumulative_rows,
            )
            if skipped:
                chunks_skipped += 1
        except Exception as e:
            chunks_failed.append(f"{start}-{end}")
            print(f"FAILED chunk [{start},{end}): {e}", file=sys.stderr)

    try:
        _ok, skipped_null = download_null_chunk(args.max_retries, cumulative_rows)
        if skipped_null:
            chunks_skipped += 1
    except Exception as e:
        chunks_failed.append("NULL")
        print(f"FAILED chunk NULL: {e}", file=sys.stderr)

    if chunks_failed:
        print(
            "Some chunks failed; not concatenating. Fix network/archive and re-run.",
            file=sys.stderr,
        )
        duration = time.time() - t0
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gasp_phase": "02_download",
            "astroquery_version": importlib.metadata.version("astroquery"),
            "total_rows": cumulative_rows[0],
            "unique_asteroids": None,
            "chunks_total": total_numbered + 1,
            "chunks_skipped": chunks_skipped,
            "chunks_failed": chunks_failed,
            "archive_warning": True,
            "duration_seconds": round(duration, 3),
        }
        (RAW_DIR / "download_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        return 1

    out_path = RAW_DIR / "gaia_sso_raw.parquet"
    return concatenate_and_finalize(
        out_path,
        chunks_failed,
        chunks_skipped,
        total_numbered + 1,
        t0,
        args.max_retries,
    )


if __name__ == "__main__":
    sys.exit(main())

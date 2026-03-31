#!/usr/bin/env python3
"""GASP Phase 6: enrich merged Gaia×SDSS catalog with SsODNet metadata via rocks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Non-interactive rocks: ensure cache dir exists before first import
_ROCKS_CACHE = Path(os.environ.get("ROCKS_CACHE_DIR", str(Path.home() / ".cache" / "rocks")))
_ROCKS_CACHE.mkdir(parents=True, exist_ok=True)

import rocks  # noqa: E402

INTERIM_DIR = Path("data/interim")
FINAL_DIR = Path("data/final")
INPUT_FILE = INTERIM_DIR / "gaia_sdss_merged.parquet"
OUTPUT_FILE = FINAL_DIR / "gasp_catalog_v1.parquet"
CACHE_FILE = INTERIM_DIR / "rocks_cache.parquet"
LOG_FILE = FINAL_DIR / "enrichment_log.json"

TEST_CACHE = INTERIM_DIR / "rocks_test_cache.parquet"
TEST_OUTPUT = INTERIM_DIR / "rocks_test.parquet"
TEST_LOG = INTERIM_DIR / "rocks_test_enrichment_log.json"

CHECKPOINT_EVERY = 500
SLEEP_S = 0.1


def _family_str(r: object) -> str | None:
    if not hasattr(r, "family") or r.family is None:
        return None
    fam = r.family
    if hasattr(fam, "name") and fam.name:
        return str(fam.name)
    if hasattr(fam, "family_name") and fam.family_name is not None:
        fn = fam.family_name
        v = getattr(fn, "value", None) if not isinstance(fn, dict) else fn.get("value")
        if v not in (None, ""):
            return str(v)
    return None


def _taxonomy_str(r: object) -> str | None:
    if not hasattr(r, "taxonomy") or r.taxonomy is None:
        return None
    t = r.taxonomy
    if isinstance(t, str):
        return t
    if hasattr(t, "class_") and t.class_ is not None:
        return str(t.class_)
    return None


def _float_attr(r: object, name: str) -> float | None:
    a = getattr(r, name, None)
    if a is None:
        return None
    if hasattr(a, "value") and a.value is not None:
        try:
            return float(a.value)
        except (TypeError, ValueError):
            return None
    return None


def rock_record(number_mp: int) -> dict:
    r = rocks.Rock(int(number_mp))
    return {
        "number_mp": int(number_mp),
        "rocks_name": str(r.name) if getattr(r, "name", None) else None,
        "family": _family_str(r),
        "taxonomy": _taxonomy_str(r),
        "albedo": _float_attr(r, "albedo"),
        "diameter_km": _float_attr(r, "diameter"),
        "rocks_error": None,
    }


def rock_record_error(number_mp: int, err: Exception) -> dict:
    return {
        "number_mp": int(number_mp),
        "rocks_name": None,
        "family": None,
        "taxonomy": None,
        "albedo": None,
        "diameter_km": None,
        "rocks_error": str(err)[:200],
    }


def load_cache(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path, engine="pyarrow")
    return pd.DataFrame(
        columns=[
            "number_mp",
            "rocks_name",
            "family",
            "taxonomy",
            "albedo",
            "diameter_km",
            "rocks_error",
        ]
    )


def save_cache(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy")


def run_enrichment(
    numbers: list[int],
    cache_path: Path,
    checkpoint_every: int,
) -> pd.DataFrame:
    cache_df = load_cache(cache_path)
    done = set(cache_df["number_mp"].astype(int).tolist()) if len(cache_df) else set()
    remaining = [n for n in numbers if n not in done]

    if done:
        print(
            f"Resuming: {len(done)} done, {len(remaining)} remaining"
        )
    else:
        print(f"Starting fresh: {len(remaining)} asteroids")

    batch: list[dict] = []

    for mp in tqdm(remaining, desc="rocks"):
        try:
            rec = rock_record(mp)
        except Exception as e:
            rec = rock_record_error(mp, e)
        batch.append(rec)
        time.sleep(SLEEP_S)

        if len(batch) >= checkpoint_every:
            new_df = pd.DataFrame(batch)
            combined = pd.concat([cache_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["number_mp"], keep="last")
            cache_df = combined
            save_cache(cache_df, cache_path)
            print(
                f"Checkpoint saved: {len(cache_df)} / {len(numbers)}"
            )
            batch = []

    if batch:
        new_df = pd.DataFrame(batch)
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        cache_df = cache_df.drop_duplicates(subset=["number_mp"], keep="last")
        save_cache(cache_df, cache_path)

    if len(cache_df):
        save_cache(cache_df, cache_path)
    return cache_df


def print_summary(df: pd.DataFrame, n_total: int) -> tuple[dict, dict]:
    def pct(x: int) -> float:
        return 100.0 * x / n_total if n_total else 0.0

    wf = int((df["family"].notna() & (df["family"].astype(str).str.len() > 0)).sum())
    wt = int((df["taxonomy"].notna() & (df["taxonomy"].astype(str).str.len() > 0)).sum())
    wa = int(df["albedo"].notna().sum())
    wd = int(df["diameter_km"].notna().sum())
    we = int((df["rocks_error"].notna() & (df["rocks_error"].astype(str).str.len() > 0)).sum())

    print(f"Total asteroids:          {n_total:,}")
    print(f"With family data:         {wf} ({pct(wf):.1f}%)")
    print(f"With taxonomy data:       {wt} ({pct(wt):.1f}%)")
    print(f"With albedo data:         {wa} ({pct(wa):.1f}%)")
    print(f"With diameter data:       {wd} ({pct(wd):.1f}%)")
    print(f"rocks errors:             {we}")
    print()

    fam_counts = (
        df["family"].dropna().replace("", np.nan).dropna().value_counts().head(10)
    )
    print("Top 10 families:")
    for name, c in fam_counts.items():
        print(f"  {name}: {c}")
    print()

    tax_counts = (
        df["taxonomy"].dropna().replace("", np.nan).dropna().value_counts()
    )
    print("Taxonomy distribution:")
    for name, c in tax_counts.items():
        print(f"  {name}: {c}")
    print()

    top_families = {str(k): int(v) for k, v in fam_counts.items()}
    tax_dist = {str(k): int(v) for k, v in tax_counts.items()}
    return top_families, tax_dist


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich GASP catalog with rocks metadata.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Process only first 50 asteroids; write rocks_test*.parquet in interim/",
    )
    args = parser.parse_args()

    test_mode = args.test
    out_file = TEST_OUTPUT if test_mode else OUTPUT_FILE
    cache_file = TEST_CACHE if test_mode else CACHE_FILE
    log_file = TEST_LOG if test_mode else LOG_FILE

    if not test_mode:
        FINAL_DIR.mkdir(parents=True, exist_ok=True)

    if out_file.exists():
        print(
            f"WARNING: {out_file} already exists — not overwriting. "
            "Remove or rename before re-running.",
            file=sys.stderr,
        )
        return 0

    if not INPUT_FILE.is_file():
        print(f"ERROR: input not found: {INPUT_FILE}", file=sys.stderr)
        return 1

    t0 = time.time()
    df_main = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    print(f"Loaded input: shape {df_main.shape}")
    if "number_mp" not in df_main.columns:
        print("ERROR: number_mp column missing", file=sys.stderr)
        return 1

    nums = (
        pd.unique(df_main["number_mp"].dropna().astype(int))
        .tolist()
    )
    nums.sort()

    if test_mode:
        nums = nums[:50]
        print(f"--test: processing first {len(nums)} asteroid numbers")
        df_main = df_main[df_main["number_mp"].isin(nums)].copy()

    n_expect = len(nums)

    df_rocks = run_enrichment(nums, cache_file, CHECKPOINT_EVERY)
    df_merged = df_main.merge(df_rocks, on="number_mp", how="left")

    n_total = len(df_merged)
    top_families, tax_dist = print_summary(df_rocks, n_expect)

    wf = int((df_rocks["family"].notna() & (df_rocks["family"].astype(str).str.len() > 0)).sum())
    wt = int((df_rocks["taxonomy"].notna() & (df_rocks["taxonomy"].astype(str).str.len() > 0)).sum())
    wa = int(df_rocks["albedo"].notna().sum())
    wd = int(df_rocks["diameter_km"].notna().sum())
    we = int((df_rocks["rocks_error"].notna() & (df_rocks["rocks_error"].astype(str).str.len() > 0)).sum())

    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(out_file, engine="pyarrow", compression="snappy")
    size_mb = out_file.stat().st_size / (1024**2)

    print(f"Saved: {out_file}")
    print(f"Shape: {len(df_merged)} rows × {len(df_merged.columns)} columns")
    print(f"File size: {size_mb:.2f} MB")

    duration = time.time() - t0
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gasp_phase": "06_enrich_rocks",
        "test_mode": test_mode,
        "input_rows": int(len(df_main)),
        "output_rows": int(len(df_merged)),
        "with_family": wf,
        "with_taxonomy": wt,
        "with_albedo": wa,
        "with_diameter": wd,
        "rocks_errors": we,
        "top_families": top_families,
        "taxonomy_distribution": tax_dist,
        "duration_seconds": round(duration, 3),
    }
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print()
    print("=== GASP Phase 6 complete ===")
    print(f"Final catalog: {out_file}")
    print(f"Asteroids: {n_total:,}")
    print(f"Columns: {len(df_merged.columns)}")
    print(f"Families identified: {wf}")
    print(f"Taxonomy assigned: {wt}")
    print("GASP v1 catalog is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""GASP Phase 6b: enrich Gaia×SDSS merge with static, citable catalogs (no live SsODNet API)."""

from __future__ import annotations

import io
import json
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import astropy.units as u  # noqa: F401 — common with Vizier workflows
from astroquery.vizier import Vizier

INTERIM_DIR = Path("data/interim")
FINAL_DIR = Path("data/final")
RAW_DIR = Path("data/raw")

INPUT_FILE = INTERIM_DIR / "gaia_sdss_merged.parquet"
OUTPUT_FILE = FINAL_DIR / "gasp_catalog_v1.parquet"
LOG_FILE = FINAL_DIR / "enrichment_log.json"

TAXONOMY_FILE = RAW_DIR / "taxonomy_mahlke2022.parquet"
FAMILIES_FILE = RAW_DIR / "families_nesvorny_pds4.parquet"
NEOWISE_FILE = RAW_DIR / "neowise_masiero2017.parquet"

FAMILIES_PDS_LISTING_URL = (
    "https://sbnarchive.psi.edu/pds4/non_mission/ast.nesvorny.families_V2_0/data/families_2015/"
)
FAMILIES_TAB_HREF_RE = re.compile(r'href="(\d+_[\w]+\.tab)"')

TAXONOMY_URL_PRIMARY = (
    "https://raw.githubusercontent.com/maxmahlke/classy/main/classy/data/taxonomy.csv"
)
TAXONOMY_URL_FALLBACK = (
    "https://raw.githubusercontent.com/maxmahlke/classy/main/classy/data/nironly_1_classified.csv"
)

def _http_get(url: str, timeout: int = 120) -> bytes:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "GASP/1.0 (enrich_static)"})
    r.raise_for_status()
    return r.content


def load_or_build_taxonomy() -> pd.DataFrame:
    if TAXONOMY_FILE.is_file():
        return pd.read_parquet(TAXONOMY_FILE, engine="pyarrow")

    last_err: Exception | None = None
    for url in (TAXONOMY_URL_PRIMARY, TAXONOMY_URL_FALLBACK):
        try:
            text = _http_get(url).decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(text))
            break
        except Exception as e:
            last_err = e
            continue
    else:
        raise RuntimeError(f"Could not download Mahlke taxonomy CSV: {last_err}")

    if "number" not in df.columns:
        raise ValueError("Taxonomy CSV missing 'number' column")

    col_class = "class_" if "class_" in df.columns else ("class" if "class" in df.columns else None)
    if col_class is None:
        raise ValueError("Taxonomy CSV missing class_/class column")

    sort_col = None
    for c in ("cluster", "probability", "prob"):
        if c in df.columns:
            sort_col = c
            break
    df["number_mp"] = pd.to_numeric(df["number"], errors="coerce").astype("Int64")
    df = df[df["number_mp"].notna() & (df["number_mp"] > 0)].copy()
    df["number_mp"] = df["number_mp"].astype(np.int64)
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)
    out = (
        df.groupby("number_mp", as_index=False)
        .agg({col_class: "first"})
        .rename(columns={col_class: "taxonomy"})
    )

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(TAXONOMY_FILE, engine="pyarrow", compression="snappy")
    print(f"Taxonomy catalog: {len(out)} asteroids (saved {TAXONOMY_FILE})")
    return out


def _family_display_name_from_stem_suffix(name_part: str) -> str:
    """e.g. nysa_polana → Nysa-Polana; vesta → Vesta."""
    hyphenated = name_part.replace("_", "-")
    return "-".join(segment.title() for segment in hyphenated.split("-"))


def _parse_family_tab_filename(filename: str) -> tuple[int, str]:
    """401_vesta.tab → (401, 'Vesta')."""
    stem = Path(filename).stem
    m = re.match(r"^(\d+)_(.+)$", stem)
    if not m:
        raise ValueError(f"Unexpected family .tab filename: {filename}")
    return int(m.group(1)), _family_display_name_from_stem_suffix(m.group(2))


def _list_pds_family_tab_files() -> list[str]:
    r = requests.get(
        FAMILIES_PDS_LISTING_URL,
        timeout=30,
        headers={"User-Agent": "GASP/1.0 (enrich_static)"},
    )
    r.raise_for_status()
    names = sorted(set(FAMILIES_TAB_HREF_RE.findall(r.text)))
    return names


def _mpc_from_family_tab_line(line: str) -> int | None:
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    token = s.split()[0]
    try:
        n = int(token)
    except ValueError:
        return None
    return n if n > 0 else None


def load_or_build_families() -> tuple[pd.DataFrame, str]:
    src = (
        "Nesvorný et al. (2015) PDS4 ast.nesvorny.families_V2_0 families_2015/ "
        "(PSI SBN archive)"
    )
    if FAMILIES_FILE.is_file():
        df = pd.read_parquet(FAMILIES_FILE, engine="pyarrow")
        n_u = int(df["number_mp"].nunique())
        n_f = int(df["family"].nunique(dropna=True))
        n_v = int((df["family"] == "Vesta").sum())
        n_fl = int((df["family"] == "Flora").sum())
        n_np = int((df["family"] == "Nysa-Polana").sum())
        print(f"Found {n_f} family files (cached)")
        print(f"Family membership loaded: {n_u} unique asteroids")
        print(f"Families represented: {n_f}")
        print(f"Largest families: Vesta={n_v}, Flora={n_fl}, Nysa-Polana={n_np}")
        return df, "cached_parquet"

    tab_files = _list_pds_family_tab_files()
    n_files = len(tab_files)
    print(f"Found {n_files} family files")
    if n_files == 0:
        raise RuntimeError("No family .tab files found in PDS listing")

    records: list[dict[str, int | str]] = []
    for fn in tqdm(tab_files, desc="Families PDS", unit="file"):
        family_id, family_name = _parse_family_tab_filename(fn)
        url = FAMILIES_PDS_LISTING_URL + fn
        resp = requests.get(url, timeout=30, headers={"User-Agent": "GASP/1.0 (enrich_static)"})
        resp.raise_for_status()
        for line in resp.text.splitlines():
            mpc = _mpc_from_family_tab_line(line)
            if mpc is None:
                continue
            records.append(
                {"number_mp": mpc, "family": family_name, "family_id": family_id}
            )
        time.sleep(0.1)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("PDS family membership parse produced no rows")

    df = df.sort_values("family_id", ascending=True)
    df = df.drop_duplicates(subset=["number_mp"], keep="last")
    df["number_mp"] = df["number_mp"].astype(np.int64)
    df["family_id"] = df["family_id"].astype(np.int64)

    n_unique = len(df)
    n_families = int(df["family"].nunique(dropna=True))
    n_v = int((df["family"] == "Vesta").sum())
    n_fl = int((df["family"] == "Flora").sum())
    n_np = int((df["family"] == "Nysa-Polana").sum())

    print(f"Family membership loaded: {n_unique} unique asteroids")
    print(f"Families represented: {n_families}")
    print(f"Largest families: Vesta={n_v}, Flora={n_fl}, Nysa-Polana={n_np}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FAMILIES_FILE, engine="pyarrow", compression="snappy")
    print(f"Saved family cache: {FAMILIES_FILE}")
    return df, src


def load_or_build_neowise() -> tuple[pd.DataFrame, str]:
    if NEOWISE_FILE.is_file():
        return pd.read_parquet(NEOWISE_FILE, engine="pyarrow"), "cached_parquet"

    v = Vizier(row_limit=-1, columns=["No", "pV", "D", "e_pV", "e_D"])
    try:
        cl = v.get_catalogs("J/ApJS/221/19")
        if len(cl) and len(cl[0]) > 0:
            t = cl[0].to_pandas()
            # column names vary — normalize
            col_no = "No" if "No" in t.columns else ([c for c in t.columns if c.lower() in ("no", "mpc", "number")] or [None])[0]
            if col_no is None:
                raise ValueError("NEOWISE table missing number column")
            out = pd.DataFrame(
                {
                    "number_mp": pd.to_numeric(t[col_no], errors="coerce"),
                    "albedo": pd.to_numeric(t.get("pV", np.nan), errors="coerce"),
                    "diameter_km": pd.to_numeric(t.get("D", np.nan), errors="coerce"),
                }
            )
            out = out[out["number_mp"].notna() & (out["number_mp"] > 0)]
            out["number_mp"] = out["number_mp"].astype(np.int64)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            out.to_parquet(NEOWISE_FILE, engine="pyarrow", compression="snappy")
            print(f"NEOWISE catalog: {len(out)} asteroids (J/ApJS/221/19 → {NEOWISE_FILE})")
            return out, "Masiero et al. (2017) J/ApJS/221/19"
    except Exception as e:
        warnings.warn(f"Vizier J/ApJS/221/19 failed ({e}); using J/ApJ/770/7.")

    v2 = Vizier(row_limit=-1, columns=["MPC", "D", "pV", "e_D", "e_pV"])
    cl = v2.get_catalogs("J/ApJ/770/7")
    if len(cl) < 2:
        raise RuntimeError("Vizier J/ApJ/770/7 missing expected tables")
    t = cl[1].to_pandas()
    mpc = t["MPC"].astype(str).str.replace(r"^0+", "", regex=True)
    out = pd.DataFrame(
        {
            "number_mp": pd.to_numeric(mpc, errors="coerce"),
            "albedo": pd.to_numeric(t["pV"], errors="coerce"),
            "diameter_km": pd.to_numeric(t["D"], errors="coerce"),
        }
    )
    out = out[out["number_mp"].notna() & (out["number_mp"] > 0)]
    out["number_mp"] = out["number_mp"].astype(np.int64)
    out = out.drop_duplicates(subset=["number_mp"], keep="first")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(NEOWISE_FILE, engine="pyarrow", compression="snappy")
    print(
        f"NEOWISE catalog: {len(out)} asteroids with albedo/diameter "
        f"(Masiero et al. 2011 J/ApJ/770/7 fallback → {NEOWISE_FILE})"
    )
    return out, "Masiero et al. (2011) J/ApJ/770/7 (J/ApJS/221/19 unavailable)"


def main() -> int:
    t0 = time.time()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        print(
            f"WARNING: {OUTPUT_FILE} already exists — not overwriting. "
            "Remove or rename before re-running.",
            file=sys.stderr,
        )
        return 0

    if not INPUT_FILE.is_file():
        print(f"ERROR: input not found: {INPUT_FILE}", file=sys.stderr)
        return 1

    df_tax = load_or_build_taxonomy()
    df_fam, fam_src = load_or_build_families()
    df_neo, neo_src = load_or_build_neowise()

    df = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    n_in = len(df)
    print(f"Gaia×SDSS dataset: {n_in} asteroids")

    if "number_mp" not in df.columns:
        print("ERROR: number_mp missing", file=sys.stderr)
        return 1

    df["number_mp"] = pd.to_numeric(df["number_mp"], errors="coerce").astype(np.int64)

    n0 = len(df)
    tax_cols = ["number_mp", "taxonomy"]
    if "taxonomy" in df.columns:
        df = df.drop(columns=["taxonomy"])
    df = pd.merge(df, df_tax[tax_cols], on="number_mp", how="left")
    n_tax = int(df["taxonomy"].notna().sum())
    print(f"Taxonomy matched: {n_tax} / {n0} ({100.0 * n_tax / n0:.2f}%)")

    df = pd.merge(df, df_fam[["number_mp", "family"]], on="number_mp", how="left")
    n_fam_m = int(df["family"].notna().sum())
    print(f"Family matched: {n_fam_m} / {n0} ({100.0 * n_fam_m / n0:.2f}%)")

    neo_cols = ["number_mp", "albedo", "diameter_km"]
    for c in ("albedo", "diameter_km"):
        if c in df.columns:
            df = df.drop(columns=[c])
    df = pd.merge(df, df_neo[neo_cols], on="number_mp", how="left")
    n_neo = int((df["albedo"].notna() & df["diameter_km"].notna()).sum())
    n_alb = int(df["albedo"].notna().sum())
    n_dia = int(df["diameter_km"].notna().sum())
    print(f"NEOWISE matched (both D and pV): {n_neo} / {n0} ({100.0 * n_neo / n0:.2f}%)")

    def pct(x: int) -> float:
        return 100.0 * x / n0 if n0 else 0.0

    with_tax = int(df["taxonomy"].notna().sum())
    with_fam = int(df["family"].notna().sum())
    with_alb = n_alb
    with_dia = n_dia
    with_all = int(
        (
            df["taxonomy"].notna()
            & df["family"].notna()
            & df["albedo"].notna()
            & df["diameter_km"].notna()
        ).sum()
    )

    print()
    print("=== Enrichment summary ===")
    print(f"Total asteroids:           {n0:,}")
    print(f"With taxonomy:             {with_tax} ({pct(with_tax):.1f}%)")
    print(f"With family:               {with_fam} ({pct(with_fam):.1f}%)")
    print(f"With albedo:               {with_alb} ({pct(with_alb):.1f}%)")
    print(f"With diameter:             {with_dia} ({pct(with_dia):.1f}%)")
    print(f"With ALL metadata:         {with_all} ({pct(with_all):.1f}%)")
    print()

    fam_vc = df["family"].dropna().astype(str)
    fam_vc = fam_vc[fam_vc.str.len() > 0].value_counts().head(10)
    print("Top 10 families:")
    for name, c in fam_vc.items():
        print(f"  {name}: {c}")
    print()

    tax_vc = df["taxonomy"].dropna().astype(str)
    tax_vc = tax_vc[tax_vc.str.len() > 0].value_counts()
    print("Taxonomy distribution:")
    for name, c in tax_vc.items():
        print(f"  {name}: {c}")
    print()

    df.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy")
    size_mb = OUTPUT_FILE.stat().st_size / (1024**2)
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Shape: {len(df)} rows × {len(df.columns)} columns")
    print(f"File size: {size_mb:.2f} MB")

    duration = time.time() - t0
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gasp_phase": "06b_enrich_static",
        "catalogs_used": [
            "Mahlke et al. (2022) DOI:10.1051/0004-6361/202243587",
            "Nesvorny et al. (2015) DOI:10.1088/0004-6256/150/2/48",
            "Masiero et al. (2017) ApJS 221/19",
        ],
        "family_source_note": fam_src,
        "neowise_source_note": neo_src,
        "input_rows": int(n_in),
        "output_rows": int(len(df)),
        "taxonomy_matched": n_tax,
        "family_matched": n_fam_m,
        "albedo_matched": n_alb,
        "diameter_matched": n_dia,
        "with_all_metadata": with_all,
        "top_families": {str(k): int(v) for k, v in fam_vc.items()},
        "taxonomy_distribution": {str(k): int(v) for k, v in tax_vc.items()},
        "duration_seconds": round(duration, 3),
    }
    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")

    print()
    print("=== GASP Phase 6 complete ===")
    print(f"Final catalog: {OUTPUT_FILE}")
    print(f"Asteroids: {n0:,}")
    print(f"Taxonomy assigned: {with_tax} ({pct(with_tax):.1f}%)")
    print(f"Family assigned:   {with_fam} ({pct(with_fam):.1f}%)")
    print(f"Albedo/diameter:   {n_neo} ({100.0 * n_neo / n0:.1f}%)")
    print("GASP v1 catalog is complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

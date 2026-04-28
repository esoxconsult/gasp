#!/usr/bin/env python3
"""
enrich_external.py
GASP — Enrich v1 catalog with external ancillary data → v2.

Adds columns from four authoritative external catalogs:
  1. LCDB + Durech+2022: rotation periods (rot_period_best, lcdb_U_qual)
  2. PDS spectral taxonomy (Bus-DeMeo > Bus > Tholen): spectral_class_best
  3. Pravec+2024 binaries: pravec_binary flag
  4. DAMIT (Durech+2010): damit_model boolean
  5. Goffin+2014: goffin_mass_1e10Msun, goffin_density_gcm3

Input:  data/final/gasp_catalog_v1.parquet  (19,190 rows)
Output: data/final/gasp_catalog_v2.parquet

GAPC raw data is shared at ../gapc/data/raw/ relative to the GASP root.
Falls back to local data/raw/ if GAPC path not available.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]       # ~/gasp/
GAPC_RAW  = ROOT.parent / "gapc" / "data" / "raw"    # ~/gapc/data/raw/
LOCAL_RAW = ROOT / "data" / "raw"                     # fallback

V1_PATH = ROOT / "data" / "final" / "gasp_catalog_v1.parquet"
V2_PATH = ROOT / "data" / "final" / "gasp_catalog_v2.parquet"
LOG_DIR = ROOT / "data" / "final"


def raw(filename: str) -> Path:
    """Return path to raw data file — prefer shared GAPC raw dir, fall back locally."""
    p = GAPC_RAW / filename
    if p.exists():
        return p
    p2 = LOCAL_RAW / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"{filename} not found in {GAPC_RAW} or {LOCAL_RAW}"
    )


# ── LCDB / Durech rotation periods ────────────────────────────────────────────
U_MIN = 2


def _parse_float(s: str) -> float:
    try:
        return float(s.strip().lstrip(">").lstrip("<").strip())
    except ValueError:
        return np.nan


def _parse_u(s: str) -> float:
    m = re.match(r"^([0-9]+)", s.strip())
    return float(m.group(1)) if m else np.nan


def load_lcdb(path: Path) -> pd.DataFrame:
    records = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            m = re.match(r"^\s*(\d+)", line)
            if not m:
                continue
            num = int(m.group(1))
            if num == 0 or len(line) < 190:
                continue
            period_str = line[145:162].strip()
            u_raw      = line[187:190].strip()
            u_val      = _parse_u(u_raw)
            if np.isnan(u_val) or u_val < U_MIN or not period_str:
                continue
            records.append(dict(
                number_mp     = num,
                lcdb_period_h = _parse_float(period_str),
                lcdb_U_qual   = u_raw[:3],
                lcdb_binary   = bool(line[197:201].strip()),
                lcdb_taxonomy = line[73:79].strip(),
            ))
    return pd.DataFrame(records)


def load_durech(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=0,
                     on_bad_lines="skip", dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    num_col = next((c for c in df.columns
                    if c in ("number", "num", "#", "number_mp")), None)
    per_col = next((c for c in df.columns
                    if "period" in c or c == "p"), None)
    if num_col is None or per_col is None:
        return pd.DataFrame(columns=["number_mp", "durech_period_h"])
    out = pd.DataFrame({
        "number_mp":       pd.to_numeric(df[num_col], errors="coerce"),
        "durech_period_h": pd.to_numeric(df[per_col], errors="coerce"),
    }).dropna()
    out["number_mp"] = out["number_mp"].astype(int)
    return out


# ── PDS spectral taxonomy ─────────────────────────────────────────────────────

def load_pds_taxonomy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().upper() for c in df.columns]
    num_col = next((c for c in df.columns
                    if "NUMBER" in c or "AST_NUMBER" in c), None)
    if num_col is None:
        return pd.DataFrame(columns=["number_mp", "spectral_class_best",
                                      "spectral_source"])
    bd_col  = next((c for c in df.columns if "BUS_DEMEO" in c
                    or "BUSDEMEO" in c), None)
    bus_col = next((c for c in df.columns if c == "BUS_CLASS"), None)
    th_col  = next((c for c in df.columns if "THOLEN" in c), None)

    out = pd.DataFrame({"number_mp": pd.to_numeric(df[num_col], errors="coerce")})
    for col, src in [(bd_col, "busdemeo"), (bus_col, "bus"), (th_col, "tholen")]:
        out[src] = df[col].replace({"": np.nan, "nan": np.nan, "-": np.nan}) \
                   if (col and col in df.columns) else np.nan

    best, source = [], []
    for _, row in out.iterrows():
        for src in ["busdemeo", "bus", "tholen"]:
            val = row.get(src, np.nan)
            if pd.notna(val) and str(val).strip() not in ("nan", ""):
                best.append(str(val).strip())
                source.append(src)
                break
        else:
            best.append(np.nan)
            source.append(np.nan)
    out["spectral_class_best"] = best
    out["spectral_source"]     = source
    out = out.dropna(subset=["number_mp"])
    out["number_mp"] = out["number_mp"].astype(int)
    return out[["number_mp", "spectral_class_best", "spectral_source"]]


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 65)
    print("  GASP enrich_external — adding external ancillary data → v2")
    print("=" * 65)

    if not V1_PATH.exists():
        print(f"  ERROR: {V1_PATH} not found")
        return

    gasp = pd.read_parquet(V1_PATH)
    print(f"\n  GASP v1: {len(gasp):,} rows, {len(gasp.columns)} cols")
    print(f"  Looking for raw data in: {GAPC_RAW}")

    # ── 1. Rotation periods ────────────────────────────────────────────────────
    print("\n  [1] Rotation periods (LCDB + Durech+2022)")
    try:
        lcdb = load_lcdb(raw("lcdb_sum.txt"))
        print(f"      LCDB: {len(lcdb):,} entries (U≥{U_MIN})")
    except FileNotFoundError as e:
        print(f"      WARN: {e}"); lcdb = pd.DataFrame()

    try:
        durech = load_durech(raw("durech2022_spins_table0.csv"))
        print(f"      Durech+2022: {len(durech):,} entries")
    except FileNotFoundError as e:
        print(f"      WARN: {e}"); durech = pd.DataFrame()

    if not lcdb.empty:
        gasp = gasp.merge(
            lcdb[["number_mp", "lcdb_period_h", "lcdb_U_qual",
                  "lcdb_binary", "lcdb_taxonomy"]],
            on="number_mp", how="left"
        )
        gasp["lcdb_binary"] = gasp["lcdb_binary"].fillna(False).astype(bool)
    else:
        for c, v in [("lcdb_period_h", np.nan), ("lcdb_U_qual", np.nan),
                     ("lcdb_binary", False), ("lcdb_taxonomy", np.nan)]:
            gasp[c] = v

    if not durech.empty:
        gasp = gasp.merge(durech[["number_mp", "durech_period_h"]],
                          on="number_mp", how="left")
    else:
        gasp["durech_period_h"] = np.nan

    gasp["rot_period_best"] = np.where(
        gasp["durech_period_h"].notna(),
        gasp["durech_period_h"],
        gasp["lcdb_period_h"]
    )
    n_rot = gasp["rot_period_best"].notna().sum()
    print(f"      rot_period_best: {n_rot:,} / {len(gasp):,} "
          f"({n_rot/len(gasp)*100:.1f}%)")

    # ── 2. PDS spectral taxonomy ───────────────────────────────────────────────
    print("\n  [2] PDS spectral taxonomy (Bus-DeMeo > Bus > Tholen)")
    try:
        spec = load_pds_taxonomy(raw("busdemeo2009_taxonomy.csv"))
        gasp = gasp.merge(spec, on="number_mp", how="left")
        n_spec = gasp["spectral_class_best"].notna().sum()
        print(f"      matched: {n_spec:,} objects ({n_spec/len(gasp)*100:.1f}%)")
        for src, n in gasp["spectral_source"].value_counts().items():
            print(f"        {src}: {n:,}")
    except FileNotFoundError as e:
        print(f"      WARN: {e}")
        gasp["spectral_class_best"] = np.nan
        gasp["spectral_source"]     = np.nan

    # ── 3. Pravec binary flag ─────────────────────────────────────────────────
    print("\n  [3] Pravec+2024 binary flag")
    try:
        pravec = pd.read_csv(raw("pravec_binaries.csv"))
        p_nums = set(pravec["number_mp"].dropna().astype(int))
        gasp["pravec_binary"] = gasp["number_mp"].isin(p_nums)
        gasp["binary_known"]  = gasp["pravec_binary"] | gasp["lcdb_binary"]
        print(f"      Pravec: {gasp['pravec_binary'].sum():,}  "
              f"LCDB: {gasp['lcdb_binary'].sum():,}  "
              f"combined: {gasp['binary_known'].sum():,}")
    except FileNotFoundError as e:
        print(f"      WARN: {e}")
        gasp["pravec_binary"] = False
        gasp["binary_known"]  = gasp["lcdb_binary"]

    # ── 4. DAMIT model flag ───────────────────────────────────────────────────
    print("\n  [4] DAMIT shape model flag")
    try:
        damit = pd.read_csv(raw("damit_models.csv"))
        d_nums = set(damit["number_mp"].dropna().astype(int))
        gasp["damit_model"] = gasp["number_mp"].isin(d_nums)
        print(f"      damit_model True: {gasp['damit_model'].sum():,} "
              f"({gasp['damit_model'].mean()*100:.1f}%)")
    except FileNotFoundError as e:
        print(f"      WARN: {e}")
        gasp["damit_model"] = False

    # ── 5. Goffin+2014 masses and densities ──────────────────────────────────
    print("\n  [5] Goffin+2014 masses & densities")
    try:
        goffin = pd.read_csv(raw("carry2021_masses.csv"))
        goffin_sub = goffin[["number_mp", "mass_1e10Msun", "e_mass_1e10Msun",
                              "density_gcm3", "Cl"]].copy()
        goffin_sub.columns = ["number_mp", "goffin_mass_1e10Msun",
                               "goffin_mass_err", "goffin_density_gcm3",
                               "goffin_density_class"]
        for c in ["goffin_mass_1e10Msun", "goffin_mass_err", "goffin_density_gcm3"]:
            goffin_sub[c] = pd.to_numeric(goffin_sub[c], errors="coerce")
        gasp = gasp.merge(goffin_sub, on="number_mp", how="left")
        n_m = gasp["goffin_mass_1e10Msun"].notna().sum()
        print(f"      matched: {n_m:,} objects with mass/density")
    except FileNotFoundError as e:
        print(f"      WARN: {e}")
        for c in ["goffin_mass_1e10Msun", "goffin_mass_err",
                  "goffin_density_gcm3", "goffin_density_class"]:
            gasp[c] = np.nan

    # ── Save ──────────────────────────────────────────────────────────────────
    gasp.to_parquet(V2_PATH, index=False)
    print(f"\n  → GASP v2: {len(gasp):,} rows, {len(gasp.columns)} cols")
    print(f"     {V2_PATH}")

    new_cols = [
        "rot_period_best", "lcdb_U_qual", "durech_period_h",
        "spectral_class_best", "spectral_source",
        "pravec_binary", "lcdb_binary", "binary_known",
        "damit_model",
        "goffin_mass_1e10Msun", "goffin_mass_err",
        "goffin_density_gcm3", "goffin_density_class",
    ]
    print(f"\n  Coverage of new columns:")
    for c in new_cols:
        if c not in gasp.columns:
            continue
        if gasp[c].dtype == bool:
            n = gasp[c].sum()
        else:
            n = gasp[c].notna().sum()
        print(f"    {c:35s}: {n:,}")

    with open(LOG_DIR / "enrich_external_log.txt", "w") as f:
        f.write("GASP enrich_external — v1 → v2\n")
        f.write("=" * 60 + "\n")
        f.write(f"GASP v1 rows: {len(gasp):,}\n")
        f.write(f"Output: {V2_PATH.name}  ({len(gasp.columns)} cols)\n\n")
        for c in new_cols:
            if c not in gasp.columns:
                continue
            n = gasp[c].sum() if gasp[c].dtype == bool else gasp[c].notna().sum()
            f.write(f"  {c}: {n:,}\n")
    print(f"  Log → data/final/enrich_external_log.txt\n")


if __name__ == "__main__":
    main()

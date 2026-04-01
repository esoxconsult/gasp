#!/usr/bin/env python3
"""GASP Phase 8: add MPC orbital class per asteroid (extended MPCORB JSON or semi-major rules)."""

from __future__ import annotations

import gzip
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    import ijson
except ImportError:
    ijson = None  # type: ignore

FINAL_DIR = Path("data/final")
RAW_DIR = Path("data/raw")
INPUT_FILE = FINAL_DIR / "gasp_catalog_v1.parquet"
OUTPUT_FILE = INPUT_FILE
BACKUP_ORBITAL = FINAL_DIR / "gasp_catalog_v1_pre_orbital.parquet"
MPC_CACHE = RAW_DIR / "mpc_orbital_classes.parquet"
LOG_FILE = FINAL_DIR / "orbital_classes_log.json"

MPC_API_URL = "https://data.minorplanetcenter.net/api/search"
MPC_EXTENDED_URL = "https://www.minorplanetcenter.net/Extended_Files/mpcorb_extended.json.gz"

MPC_CLASSES: dict[str, str] = {
    "MBA": "Main Belt Asteroid",
    "OMB": "Outer Main Belt",
    "IMB": "Inner Main Belt",
    "MCA": "Mars Crosser",
    "TJN": "Jupiter Trojan",
    "CEN": "Centaur",
    "TNO": "Trans-Neptunian Object",
    "PAA": "Parabolic Asteroid",
    "HYA": "Hyperbolic Asteroid",
    "IEO": "Interior Earth Object",
    "ATE": "Aten",
    "APO": "Apollo",
    "AMO": "Amor",
    "KBO": "Kuiper Belt Object",
}

NUMBER_RE = re.compile(r"^\((\d+)\)$")


def _parse_mpc_number(num: object) -> int | None:
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return None
    s = str(num).strip()
    m = NUMBER_RE.match(s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    return None


def _classify_from_a_au(a: float) -> str:
    """Method C: rough labels aligned with MPC_CLASSES names."""
    if a < 1.0:
        return "Interior Earth Object"
    if a < 1.665:
        return "Amor"
    if a < 2.0:
        return "Mars Crosser"
    if a < 2.5:
        return "Inner Main Belt"
    if a < 3.2:
        return "Main Belt Asteroid"
    if a < 4.1:
        return "Outer Main Belt"
    if a < 5.5:
        return "Jupiter Trojan"
    if a < 30.0:
        return "Centaur"
    return "Trans-Neptunian Object"


def _codes_to_labels(code: str | None) -> str:
    if not code:
        return "Unknown/unclassified"
    s = str(code).strip()
    return MPC_CLASSES.get(s, s)


def _test_mpc_api() -> None:
    headers = {"Content-Type": "application/json", "User-Agent": "GASP/1.0 (add_orbital_classes)"}
    payload = {"object_type": "M", "limit": 1, "number": 1}
    try:
        r = requests.post(MPC_API_URL, json=payload, headers=headers, timeout=10)
        print(f"MPC API test (Ceres): HTTP {r.status_code}")
        print(r.text[:1200])
    except Exception as e:
        print(f"MPC API test failed: {e}")


def _build_map_from_extended_gz(path: Path, wanted: set[int]) -> tuple[dict[int, str], dict[int, float]]:
    if ijson is None:
        raise RuntimeError("ijson is required (pip install ijson)")
    code_map: dict[int, str] = {}
    a_map: dict[int, float] = {}
    with gzip.open(path, "rb") as f:
        for obj in tqdm(ijson.items(f, "item"), desc="MPC extended JSON", unit="obj"):
            if not isinstance(obj, dict):
                continue
            n = _parse_mpc_number(obj.get("Number"))
            if n is None or n not in wanted:
                continue
            ot = obj.get("Orbit_type")
            if ot is not None:
                code_map[n] = str(ot).strip()
            a = obj.get("a")
            if a is not None:
                try:
                    a_map[n] = float(a)
                except (TypeError, ValueError):
                    pass
    return code_map, a_map


def _download_extended(dest_gz: Path) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "GASP/1.0 (add_orbital_classes)"}
    with requests.get(MPC_EXTENDED_URL, headers=headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        pbar = tqdm(total=total if total > 0 else None, unit="B", unit_scale=True, desc="Download mpcorb_extended.json.gz")
        with open(dest_gz, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    if total:
                        pbar.update(len(chunk))
        pbar.close()


def _row_label(
    n_int: int | None,
    code_map: dict[int, str],
    a_map: dict[int, float],
) -> tuple[str, str]:
    if n_int is None:
        return "Unknown/unclassified", "unclassified"
    if n_int in code_map:
        return _codes_to_labels(code_map[n_int]), "mpc_extended"
    a = a_map.get(n_int)
    if a is not None:
        return _classify_from_a_au(float(a)), "semi_major_axis_rule"
    return "Unknown/unclassified", "unclassified"


def main() -> int:
    t0 = time.time()
    if not INPUT_FILE.is_file():
        print(f"ERROR: input not found: {INPUT_FILE}", file=sys.stderr)
        return 1

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(INPUT_FILE, BACKUP_ORBITAL)
    print(f"Backup saved: {BACKUP_ORBITAL}")

    df = pd.read_parquet(INPUT_FILE, engine="pyarrow")
    if "number_mp" not in df.columns:
        print("ERROR: number_mp missing", file=sys.stderr)
        return 1

    nums = pd.to_numeric(df["number_mp"], errors="coerce")
    wanted = {int(x) for x in nums.dropna().unique()}

    print("MPC API probe (schema / availability):")
    _test_mpc_api()

    method_used = "mpc_extended"
    code_map: dict[int, str] = {}
    a_map: dict[int, float] = {}

    if MPC_CACHE.is_file():
        cached = pd.read_parquet(MPC_CACHE, engine="pyarrow")
        print(f"Loaded cache: {MPC_CACHE} ({len(cached)} rows)")
        for _, row in cached.iterrows():
            n = int(row["number_mp"])
            if "orbit_code" in row.index and pd.notna(row["orbit_code"]):
                code_map[n] = str(row["orbit_code"])
            if "a_au" in row.index and pd.notna(row.get("a_au")):
                try:
                    a_map[n] = float(row["a_au"])
                except (TypeError, ValueError):
                    pass
        method_used = "mpc_extended_cached"
    else:
        tmp_gz = RAW_DIR / "mpcorb_extended.json.gz"
        try:
            if not tmp_gz.is_file():
                print(f"Downloading {MPC_EXTENDED_URL} …")
                _download_extended(tmp_gz)
            print("Parsing extended MPC JSON (streaming) …")
            code_map, a_map = _build_map_from_extended_gz(tmp_gz, wanted)
            cache_rows = [
                {
                    "number_mp": n,
                    "orbit_code": code_map.get(n),
                    "a_au": a_map.get(n),
                }
                for n in sorted(wanted)
            ]
            cache_df = pd.DataFrame(cache_rows)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            cache_df.to_parquet(MPC_CACHE, engine="pyarrow", compression="snappy")
            print(f"Saved MPC cache: {MPC_CACHE}")
        except Exception as e:
            print(f"MPC extended failed ({e}); no orbit codes available.")
            method_used = "failed"
            code_map, a_map = {}, {}

    oc: list[str] = []
    om: list[str] = []
    for i in range(len(df)):
        nv = nums.iloc[i]
        n_int = int(nv) if pd.notna(nv) else None
        lab, met = _row_label(n_int, code_map, a_map)
        oc.append(lab)
        om.append(met)

    df["orbital_class"] = oc
    df["orbital_class_method"] = om

    vc = df["orbital_class"].value_counts()
    n_total = len(df)
    print()
    print("Orbital class distribution:")
    for name, c in vc.items():
        print(f"  {name}: {c} ({100.0 * c / n_total:.1f}%)")
    unk = int((df["orbital_class"] == "Unknown/unclassified").sum())
    print(f"  Unknown/unclassified: {unk} ({100.0 * unk / n_total:.1f}%)")
    print()

    df.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy")
    size_mb = OUTPUT_FILE.stat().st_size / (1024**2)
    n_cols = len(df.columns)
    print(f"Updated: {OUTPUT_FILE}")
    print("New columns: orbital_class, orbital_class_method")
    print(f"Total columns: {n_cols}")
    print(f"File size: {size_mb:.2f} MB")

    duration = time.time() - t0
    log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gasp_phase": "08_orbital_classes",
        "method_used": method_used,
        "orbital_class_distribution": {str(k): int(v) for k, v in vc.items()},
        "unclassified_count": unk,
        "duration_seconds": round(duration, 3),
    }

    LOG_FILE.write_text(json.dumps(log, indent=2), encoding="utf-8")

    top3 = vc.head(3)
    top_str = ", ".join(f"{k}={v}" for k, v in top3.items())
    print()
    print("=== GASP Phase 8 complete ===")
    print("Added: orbital_class, orbital_class_method")
    print(f"Distribution (top 3): {top_str}")
    print(f"Output: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""GASP environment and Gaia TAP connectivity verification."""

from __future__ import annotations

import sys
import traceback


def main() -> int:
    print("=== GASP — Gaia Asteroid Spectral Pipeline ===")

    try:
        import astroquery
        import astropy
        import matplotlib
        import numpy
        import pandas
        import pyarrow
        import rocks
        import sklearn
        import tqdm

        def pkg_version(mod, label: str) -> str:
            v = getattr(mod, "__version__", None)
            return f"{label}: {v}" if v is not None else f"{label}: (unknown)"

        print(pkg_version(astroquery, "astroquery"))
        print(pkg_version(astropy, "astropy"))
        print(pkg_version(pandas, "pandas"))
        print(pkg_version(numpy, "numpy"))
        print(pkg_version(matplotlib, "matplotlib"))
        print(pkg_version(sklearn, "sklearn"))
        print(pkg_version(pyarrow, "pyarrow"))
        print(pkg_version(tqdm, "tqdm"))
        print(pkg_version(rocks, "rocks"))

        rocks_ver = getattr(rocks, "__version__", "0")
        try:
            from packaging.version import Version

            if Version(rocks_ver) < Version("1.9.16"):
                print(
                    f"Warning: space-rocks >= 1.9.16 recommended (found {rocks_ver})",
                    file=sys.stderr,
                )
        except Exception:
            pass

        from astroquery.gaia import Gaia

        # ESA Gaia TAP (gea.esac.esa.int)
        if hasattr(Gaia, "TAP_URL"):
            Gaia.TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

        query = """
SELECT TOP 5 denomination, number_mp, nb_samples
FROM gaiadr3.sso_reflectance_spectrum
ORDER BY number_mp ASC
"""
        job = Gaia.launch_job(query)
        result = job.get_results()

        print()
        print(result)
        print()
        print("=== SETUP OK — ready for Phase 2 ===")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

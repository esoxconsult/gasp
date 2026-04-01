# GASP — Gaia Asteroid Spectral Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15366682.svg)](https://doi.org/10.5281/zenodo.15366682)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A reproducible, open-source pipeline to download, correct,
and cross-match asteroid reflectance spectra from the ESA
Gaia DR3 archive.

Built by an enthusiast amateur astronomer with a VPS and
an AI. No institutional affiliation required.

## What GASP produces

**gasp_catalog_v1.parquet** — 19,190 asteroids, 72 columns with:
- Gaia DR3 reflectance spectra (16 bands, 374–1034 nm, NUV-corrected)
- SDSS MOC4 photometry (u, g, r, i, z, a*)
- NEOWISE albedo and diameter (22.3% coverage)
- Spectral taxonomy — Mahlke et al. 2022 (1.9%)
- Family membership — Nesvorný et al. 2015 (39.3%, 117 families)
- Orbital classification — MPC (orbital_class, 100%)
- Spectral slopes s1–s4 (µm⁻¹, Tinaut-Ruano 2024 methodology)
- S/N quality flags (nuv_snr_score 0–3)
- SDSS C/S complex classification (sdss_complex)
- ECAS-validated: Pearson r = −0.848 vs ground-truth NUV photometry

## Why this matters

No corrected, cross-matched, publicly available catalog of
Gaia DR3 asteroid reflectance spectra existed before GASP.

Three years after the Gaia DR3 release (June 2022), the first
full taxonomic classification appeared at EPSC-DPS 2025
(Tinaut-Ruano et al.). GASP provides the open, reproducible
infrastructure that was missing: download the data, apply the
published corrections, cross-match with SDSS, enrich with
families — all in one pipeline, fully documented.

## Pipeline

| Phase | Script | Output |
|-------|--------|--------|
| 1 | verify_setup.py | environment check |
| 2 | download_gaia_sso.py | data/raw/gaia_sso_raw.parquet |
| 3 | restructure.py | data/interim/gaia_sso_wide.parquet |
| 4 | nuv_correction.py | data/interim/gaia_sso_nuv_corrected.parquet |
| 5 | crossmatch_sdss.py | data/interim/gaia_sdss_merged.parquet |
| 6 | enrich_static.py | data/final/gasp_catalog_v1.parquet |
| 7 | compute_features.py | adds slopes, S/N flags, C/S class |
| 8 | add_orbital_classes.py | adds orbital_class (MPC) |

**Validation:** `pipeline/validate_ecas.py` — cross-validates NUV correction against ECAS (Zellner et al. 1985). Produces `figures/05_ecas_validation.png`.

## Quickstart
```bash
git clone https://github.com/esoxconsult/gasp.git
cd gasp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline/verify_setup.py
```

To reproduce the full catalog (~1 hour, ~2 GB disk):
```bash
python pipeline/download_gaia_sso.py
python pipeline/restructure.py
python pipeline/nuv_correction.py
python pipeline/crossmatch_sdss.py
python pipeline/enrich_static.py
python pipeline/compute_features.py
python pipeline/add_orbital_classes.py
```

## Catalog at a glance

| Property | Value |
|----------|-------|
| Asteroids | 19,190 |
| Columns | 72 |
| Spectral bands | 16 (374–1034 nm, NUV-corrected) |
| SDSS photometry | u, g, r, i, z, a* |
| Family coverage | 39.3% (Nesvorný 2015, 117 families) |
| Taxonomy coverage | 1.9% (Mahlke 2022) |
| Albedo coverage | 22.3% (NEOWISE/Masiero 2011) |
| Orbital classes | 100% (MPC extended catalog) |
| ECAS validation | r = −0.848 (n=83, p≈0) |
| File size | ~4.9 MB (parquet) |
| DOI | 10.5281/zenodo.15366682 |

## Catalog columns

| Group | Columns |
|-------|---------|
| Identifiers | denomination, number_mp |
| Gaia spectra | refl_374 … refl_1034 (16 bands, NUV-corrected) |
| Gaia errors | err_374 … err_1034 (16 bands) |
| Gaia metadata | num_of_spectra, nb_samples, quality, norm_ok |
| NUV correction | nuv_correction_applied, nuv_reliable, nuv_negative_flag |
| NUV provenance | nuv_correction_source, nuv_correction_doi |
| SDSS photometry | u, g, r, i, z, a_star, V, B, e_u…e_z, moID |
| NEOWISE | albedo, diameter_km |
| Taxonomy | taxonomy (Mahlke et al. 2022) |
| Family | family, family_id (Nesvorný et al. 2015) |
| Orbital classification | orbital_class, orbital_class_method |
| Spectral slopes | s1, s2, s3, s4 (µm⁻¹) |
| Quality flags | nuv_snr_flag, nuv_snr_score, sdss_complex |

## Known limitations

- **Taxonomy coverage 1.9%**: reflects the actual size of
  spectroscopically classified asteroids in Mahlke (2022).
  Most Gaia asteroids are too faint or too new to have
  ground-based spectral classification.

- **Family coverage 39.3%**: based on Nesvorný et al. (2015)
  HCM classification. Asteroids outside known families
  are "background" objects — scientifically valid, not missing.

- **NUV artifacts**: 58 asteroids in the full Gaia DR3 dataset
  have negative reflectance in ≥1 NUV band (known low-S/N
  artifact per Tinaut-Ruano 2023). None appear in the final
  cross-matched sample (nuv_reliable = True for all 19,190).

- **NIR correction not applied**: Gallinier et al. (2023)
  describe a systematic reddening between 0.7–0.9 µm.
  This correction is not yet implemented in GASP v1.
  Planned for v2 with Gaia DR4 data.

- **ECAS validation**: Cross-matched with 589 ECAS asteroids
  (Zellner et al. 1985); 83 objects in common. Pearson
  correlation between ECAS b-v and GASP refl_418:
  r = −0.848 (p ≈ 0), confirming the NUV correction
  produces spectra consistent with ground-based photometry.
  The negative sign reflects the inverse relationship between
  reflectance ratio and color index conventions.

## References

| What | Citation |
|------|----------|
| Gaia DR3 SSO spectra | Galluccio et al. (2022) arXiv:2206.12174 |
| NUV correction | Tinaut-Ruano et al. (2023) DOI:10.1051/0004-6361/202245134 |
| NUV primitive asteroids | Tinaut-Ruano et al. (2024) arXiv:2403.10321 |
| SDSS MOC4 | Ivezic et al. (2001) AJ 122, 2749 |
| Taxonomy | Mahlke et al. (2022) DOI:10.1051/0004-6361/202243587 |
| Families | Nesvorný et al. (2015) DOI:10.1088/0004-6256/150/2/48 |
| Albedo/diameter | Masiero et al. (2011) DOI:10.1088/0004-637X/741/2/68 |

## License

MIT — code is freely reusable.

Data derived from:
- ESA Gaia (CC BY-SA 3.0 IGO) — please cite Gaia Collaboration
- SDSS (BSD-style) — please cite Ivezic et al.
- NEOWISE — public domain (NASA)
- Nesvorný families — public domain (NASA PDS)

## Author

Werner Scheibenpflug (Austria)
Built with curiosity, a VPS, and Claude (Anthropic).

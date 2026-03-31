# GASP — Project Notes

## Scientific context
- Gaia DR3 contains 60,518 asteroid reflectance spectra (16 bands,
  0.374–1.002 µm), published June 2022
- Two known systematic artifacts in the raw spectra:
  1. NUV reddening below 0.55 µm (Tinaut-Ruano et al. 2023)
  2. NIR reddening between 0.7–0.9 µm (Gallinier et al. 2023)
- No public corrected + cross-matched catalog exists yet
- First full taxonomy published Sept 2025 (EPSC-DPS, Helsinki)
  — GASP pipeline predates and complements this work

## Data volume estimates
- gaia_sso_raw.parquet:              ~5 MB
- gaia_sso_wide.parquet:             ~8 MB
- All interim files combined:        ~30 MB
- SDSS MOC4:                         ~40 MB
- rocks cache (~/.cache/rocks/):     ~500 MB–1 GB

## Important rules
- Raw data is NEVER overwritten — always write to new files
- All parquet files use snappy compression (pyarrow default)
- NUV correction factors must be taken from Table 2 of
  Tinaut-Ruano et al. (2023), DOI:10.1051/0004-6361/202245134
- rocks caches automatically — do not delete ~/.cache/rocks/

## Publication plan
1. GitHub: MIT license, full code + documentation
2. Zenodo: final catalog with DOI (citable dataset)
3. Research Notes of the AAS: short data paper (~1000 words)
4. Gaia Users Community: announcement on ESA mailing list

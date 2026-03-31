# GASP — Gaia Asteroid Spectral Pipeline

A pipeline to download, correct, and cross-match asteroid reflectance 
spectra from the ESA Gaia DR3 archive with the SDSS Moving Object 
Catalog (MOC4).

## What GASP does
- Downloads all 60,518 asteroid reflectance spectra from Gaia DR3
- Applies two known systematic corrections (NUV + NIR artifacts)
- Cross-matches with SDSS MOC4 photometry (~8,000–15,000 shared objects)
- Enriches with asteroid metadata via the rocks tool
- Produces a clean, reproducible, publicly available catalog

## Why this matters
No corrected, cross-matched public catalog of Gaia DR3 asteroid 
spectra currently exists. GASP fills that gap.

## Pipeline phases
1. Setup & verification
2. Download Gaia SSO spectra      → data/raw/
3. Restructure to wide format     → data/interim/
4. Apply NUV + NIR corrections    → data/interim/
5. Download & merge SDSS MOC4     → data/interim/
6. Enrich with asteroid metadata  → data/final/
7. First analysis & figures       → figures/

## Key references
- Gaia SSO spectra: Galluccio et al. (2022) arXiv:2206.12174
- NUV correction: Tinaut-Ruano et al. (2023) DOI:10.1051/0004-6361/202245218
- NUV primitive asteroids: Tinaut-Ruano et al. (2024) arXiv:2403.10321
- SDSS MOC4: Ivezic et al. (2001) AJ 122, 2749

## License
MIT

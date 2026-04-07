# GASP — Gaia DR4 Preparation Notes

## DR4 release timeline
Expected: **late 2026** (December per current ESA planning; confirm on [Gaia](https://www.cosmos.esa.int/web/gaia/release) when announced)
Coverage: 66 months of observations (vs 34 months in DR3)

## What changes in DR4

### More asteroids
DR3: 60,518 SSO reflectance spectra
DR4: expected ~150,000+ (roughly 2.5× more)
Pipeline phases 2-6 will run unchanged — just larger data.

### Better spectra
More epoch observations averaged per object → higher S/N.
The nuv_snr_score distribution will shift toward score 3.
The 58 NUV-artifact objects in DR3 will likely be resolved.

### NIR correction
Gallinier et al. (2023) describe systematic reddening between
0.7–0.9 µm. ESA has indicated DR4 will improve the SSO
processing. GASP v2 should implement the NIR correction
(not yet applied in v1) if DR4 spectra still show the artifact.

## Pipeline changes needed for DR4

### Phase 2 — download_gaia_sso.py
- Update chunk boundaries (max number_mp will be higher)
- No structural changes needed

### Phase 3 — restructure.py
- Verify wavelength grid is unchanged (16 bands, same centers)
- If new bands added: update WAVELENGTHS constant

### Phase 4 — nuv_correction.py
- Verify correction factors still apply
- Check if ESA updated the solar analogue list
- If NIR correction published: add Phase 4b

### Phase 5 — crossmatch_sdss.py
- SDSS MOC4 is static — no changes needed
- Consider adding LSST/Rubin data when available (~2027)

### Phase 6 — enrich_static.py
- Update Nesvorný catalog to latest version
- Add Mahlke taxonomy updates

## Suggested GASP v2 timeline
- Q1 2027: DR4 released → run pipeline, update catalog
- Q2 2027: Add NIR correction (Gallinier 2023)
- Q2 2027: Add LSST cross-match if data available
- Q3 2027: Submit updated paper to A&A (full data paper)

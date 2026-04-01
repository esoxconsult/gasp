# GASP v1 Catalog — Column Reference

## File
`data/final/gasp_catalog_v1.parquet`

## Quick load
```python
import pandas as pd
df = pd.read_parquet('data/final/gasp_catalog_v1.parquet')
print(df.shape)           # (19190, 70)
print(df.columns.tolist())
```

## Identifiers

| Column | Type | Description |
|--------|------|-------------|
| denomination | str | IAU name, lowercase (e.g. "ceres") |
| number_mp | int | MPC asteroid number |

## Gaia DR3 reflectance spectra

NUV correction applied per Tinaut-Ruano et al. (2023) Table 1.
All spectra normalized to 1.0 at 550 nm.

| Column | Wavelength | Note |
|--------|-----------|------|
| refl_374 | 374 nm | NUV-corrected (/1.07) |
| refl_418 | 418 nm | NUV-corrected (/1.05) |
| refl_462 | 462 nm | NUV-corrected (/1.02) |
| refl_506 | 506 nm | NUV-corrected (/1.01) |
| refl_550 | 550 nm | normalization point = 1.0 |
| refl_594 | 594 nm | |
| refl_638 | 638 nm | |
| refl_682 | 682 nm | |
| refl_726 | 726 nm | |
| refl_770 | 770 nm | |
| refl_814 | 814 nm | |
| refl_858 | 858 nm | |
| refl_902 | 902 nm | |
| refl_946 | 946 nm | |
| refl_990 | 990 nm | |
| refl_1034 | 1034 nm | |

Corresponding error columns: err_374 … err_1034

## Gaia metadata

| Column | Type | Description |
|--------|------|-------------|
| num_of_spectra | int | epoch spectra averaged |
| nb_samples | int | wavelength samples |
| n_valid_bands | int | non-null bands (16 for all) |
| quality | str | high/medium/low (all high) |
| norm_ok | bool | refl_550 within 0.95–1.05 |

## NUV correction provenance

| Column | Description |
|--------|-------------|
| nuv_correction_applied | True for all rows |
| nuv_correction_source | "Tinaut-Ruano et al. (2023) Table 1" |
| nuv_correction_doi | "10.1051/0004-6361/202245134" |
| nuv_negative_flag | True if any NUV band < 0 |
| nuv_reliable | True if no negative NUV values |

## SDSS MOC4 photometry

| Column | Description |
|--------|-------------|
| u, g, r, i, z | SDSS magnitudes |
| e_u, e_g, e_r, e_i, e_z | magnitude errors |
| a_star | color index — S/C separator |
| V, B | Johnson magnitudes |
| moID | SDSS moving object ID |

## Enrichment

| Column | Source | Coverage |
|--------|--------|----------|
| albedo | NEOWISE Masiero (2011) | 22.3% |
| diameter_km | NEOWISE Masiero (2011) | 22.3% |
| taxonomy | Mahlke et al. (2022) | 1.9% |
| family | Nesvorný et al. (2015) | 39.3% |
| family_id | Nesvorný et al. (2015) | 39.3% |

## Derived features

| Column | Description |
|--------|-------------|
| s1 | slope 374–418 nm (µm⁻¹), NUV absorption diagnostic |
| s2 | slope 418–550 nm (µm⁻¹), blue-visible |
| s3 | slope 550–700 nm (µm⁻¹), visible-red |
| s4 | slope 700–1034 nm (µm⁻¹), NIR |
| nuv_snr_flag | True if num_of_spectra >= 5 |
| nuv_snr_score | 0-3 reliability score |
| sdss_complex | C-complex / ambiguous / S-complex |

## Code examples

### Plot a spectrum
```python
import matplotlib.pyplot as plt

BANDS = [374,418,462,506,550,594,638,682,
         726,770,814,858,902,946,990,1034]
refl_cols = [f'refl_{b}' for b in BANDS]

row = df[df['denomination'] == 'ceres'].iloc[0]
plt.figure(figsize=(10, 4))
plt.plot(BANDS, row[refl_cols].values, 'o-')
plt.axvline(550, color='gray', ls='--', alpha=0.5,
            label='norm point')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance (norm. at 550 nm)')
plt.title(f"Ceres — Gaia DR3 (NUV corrected)")
plt.legend()
plt.tight_layout()
plt.show()
```

### S vs C separation
```python
c = df[df['sdss_complex'] == 'C-complex']
s = df[df['sdss_complex'] == 'S-complex']
print(f"C-complex: {len(c):,}")
print(f"S-complex: {len(s):,}")
```

### NUV absorption in primitive asteroids
```python
primitives = df[
    (df['albedo'] < 0.1) &
    (df['nuv_snr_score'] >= 2)
]
print(f"Primitive, reliable NUV: {len(primitives)}")
print(primitives['s1'].describe())
```

### Family spectral homogeneity
```python
for family in ['Vesta', 'Flora', 'Themis']:
    members = df[df['family'] == family]
    print(f"{family}: n={len(members)}, "
          f"s1_std={members['s1'].std():.4f}")
```

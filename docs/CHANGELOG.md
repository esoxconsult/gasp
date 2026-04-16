# GASP Changelog

## v2.0.0 — 2026-04-16

### Neu
- ML-Taxonomie-Klassifikator (RandomForest, Optuna-optimiert, F1-macro=0.72)
- 4 neue Spalten: `taxonomy_ml`, `taxonomy_ml_prob`, `taxonomy_ml_source`, `taxonomy_ml_x_refined`
- `taxonomy_final`: 8,186 Asteroiden (42.7%) — Mahlke-Labels + ML bei prob ≥ 0.70
- X→E/M/P Subklassifikation für 540 Asteroiden via NEOWISE-Albedo (Tholen 1984)
- Pipeline-Skript: `pipeline/09_taxonomy_classifier.py`

### Unverändert
- Alle 72 ursprünglichen Spalten aus v1.0
- NUV-Korrektur (Tinaut-Ruano et al. 2023)
- ECAS-Validierung (Pearson r = −0.848)

---

## v1.1.0 — 2026-04-01

- Orbital classification via MPC (orbital_class, 100% coverage)
- ECAS cross-validation against Zellner et al. (1985)
- RNAAS paper submitted (DOI: 10.3847/2515-5172/ae5e45)

## v1.0.0 — 2026-03-01

- Initial release: 19,190 asteroids, 72 columns
- Gaia DR3 NUV-corrected spectra, SDSS photometry, NEOWISE albedo,
  Nesvorný family membership, Mahlke taxonomy

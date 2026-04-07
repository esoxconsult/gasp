# GASP — Projekt-Kontext (Fortsetzung / KI-Assistenz)

Diese Datei fasst Identität, Infrastruktur und Arbeitsablauf zusammen. Sie kann in Chats eingebunden oder als Ausgangsprompt kopiert werden.

---

You are helping me continue development of **GASP** (Gaia Asteroid Spectral Pipeline), my open-source catalog of **19,190** corrected Gaia DR3 asteroid reflectance spectra.

## Project identity

| | |
|---|---|
| **Author** | Werner Scheibenpflug, Vienna, Austria |
| **GitHub** | https://github.com/esoxconsult/gasp — MIT, public, **v1.1.0** |
| **Zenodo DOI** | https://doi.org/10.5281/zenodo.19366681 |
| **RNAAS paper** | Submitted to AAS (v3, final) — **awaiting decision** (typ. 2–4 weeks) |

## Infrastructure

| Environment | Path / detail |
|-------------|----------------|
| **VPS** | `media.esoxconsult.com`, Ubuntu 22.04, project root `~/gasp/` |
| **Local Mac** | `/Users/wernerfhs/_media/GASP/` |
| **Python venv** | `~/gasp/.venv` on VPS; locally e.g. `GASP/.venv` |
| **space-rocks** | ≥ **1.9.16** (see `requirements.txt`) |
| **Jupyter kernel** | Register as **"GASP — Gaia Asteroid Spectral Pipeline"** (e.g. `python -m ipykernel install --user --name gasp --display-name "GASP — Gaia Asteroid Spectral Pipeline"`) |

## Catalog: `gasp_catalog_v1.parquet`

- **19,190** asteroids, **72** columns, **16** spectral bands (374–1034 nm, NUV-corrected)
- **Family coverage:** 39.3% (117 families, Nesvorný PDS4 V2.0)
- **Taxonomy:** 1.9% (Mahlke et al. 2022)
- **Albedo/diameter:** 22.3% (NEOWISE / Masiero 2011)
- **Orbital class:** 100% (MPC extended)
- **ECAS validation:** Pearson **r = −0.848** (n = 83, p ≈ 0)

## Pipeline (8 scripts + validation)

1. `verify_setup.py` — environment check  
2. `download_gaia_sso.py` → `data/raw/gaia_sso_raw.parquet`  
3. `restructure.py` → `data/interim/gaia_sso_wide.parquet`  
4. `nuv_correction.py` → `data/interim/gaia_sso_nuv_corrected.parquet`  
5. `crossmatch_sdss.py` → `data/interim/gaia_sdss_merged.parquet`  
6. `enrich_static.py` → `data/final/gasp_catalog_v1.parquet`  
7. `compute_features.py` — slopes s1–s4, SNR flags, C/S class  
8. `add_orbital_classes.py` — MPC orbital classes  

**+** `validate_ecas.py` — ECAS validation (figures under `figures/`).

## Key references

| Topic | Reference |
|-------|-----------|
| NUV correction | Tinaut-Ruano et al. (2023) DOI:10.1051/0004-6361/202245134 |
| Gaia DR3 SSO | Galluccio et al. (2022) arXiv:2206.12174 |
| SDSS MOC4 | Ivezic et al. (2001) AJ 122, 2749 |
| Families | Nesvorný et al. (2015) DOI:10.1088/0004-6256/150/2/48 |
| Taxonomy | Mahlke et al. (2022) DOI:10.1051/0004-6361/202243587 |
| Albedo | Masiero et al. (2011) DOI:10.1088/0004-637X/741/2/68 |
| ECAS | Zellner et al. (1985) Icarus 61, 355 |

## Git workflow

```bash
# Local commit & push
cd /Users/wernerfhs/_media/GASP
git add . && git commit -m "..." && git push origin HEAD:main
```

```bash
# Deploy to VPS
ssh media "cd ~/gasp && git pull origin main"
```

(`media` = SSH host alias for `media.esoxconsult.com` — adjust if your `~/.ssh/config` differs.)

## Current status & next steps

1. **RNAAS** — Awaiting decision; no action unless reviewers request changes.  
2. **NIR correction** (0.7–0.9 µm) — planned for **v2** with Gaia DR4 (see `docs/dr4_preparation.md`).  
3. **Gaia Helpdesk** — e-mail re: DR4 asteroid spectra; low priority, send when ready.  
4. **DR4** — Release expected **late 2026** → full pipeline re-run; then catalog v2.

## Commercial extension

ML taxonomy, orbital logistics, and prospecting tooling live in the private **Esox Space** repository (not in this public tree). See README section *GASP Enterprise (Esox Space)*.

---

## What I need from you now

*[Describe your current task here]*

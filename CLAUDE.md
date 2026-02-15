# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Political economy research studying how wildfire smoke exposure affects voting behavior in the United States. Uses quasi-experimental design exploiting wind-driven smoke plume dispersion to estimate causal effects on county-level presidential and district-level House election outcomes via two-way fixed effects (TWFE) panel regressions.

## Pipeline Execution

The analysis runs as a sequential pipeline. Each step depends on the prior step's output.

```bash
# 1. Download smoke PM2.5 data (Harvard Dataverse → data/smoke/)
python scripts/download_smoke_data.py

# 2. Download election returns and Census crosswalks (Harvard Dataverse, Census → data/elections/, data/crosswalks/)
python scripts/download_election_data.py

# 3. Build analysis datasets (merge smoke + elections → output/*.parquet)
python scripts/build_smoke_analysis.py        # Presidential (county-level)
python scripts/build_house_analysis.py         # House (district-level)
python scripts/build_county_house_analysis.py  # House (county-level, from precinct data)

# 4. Run analysis and generate figures (output/figures/)
python scripts/analyze_smoke_voting.py         # Main presidential analysis
python scripts/analyze_house_voting.py         # House analysis (district-level)
python scripts/analyze_county_house_voting.py  # House analysis (county-level)

# 5. Cross-validate TWFE results in R
Rscript scripts/analyze_house_voting_check.R

# 6. Generate geographic maps
Rscript scripts/map_smoke_panel.R

# 7. Compile paper
cd paper && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

There is no Makefile, test suite, or linter configured.

## Architecture

**Data flow:** Raw CSVs (`data/`) → build scripts merge smoke exposure windows with election returns → Parquet analysis datasets (`output/`) → analysis scripts run TWFE regressions and produce figures (`output/figures/`) → LaTeX paper (`paper/`).

**Key identifiers:** County FIPS codes (`fips`), congressional district IDs, election year. Smoke exposure is aggregated into pre-election windows (7d, 30d, 60d, 90d, full fire season Jun 1–Election Day).

**Statistical framework:** `linearmodels.panel.PanelOLS` for TWFE with entity (county/district) and time (year) fixed effects. Standard errors clustered by entity. Three outcome specifications: Democratic vote share, incumbent vote share, log total votes.

**Analysis datasets:**
- `output/smoke_voting_analysis.parquet` — 12,429 county-election obs (2008–2020 presidential)
- `output/smoke_voting_house_analysis.parquet` — 3,452 district-election obs (2006–2020 House)
- `output/smoke_voting_county_house_analysis.parquet` — ~9,000 county-election obs (2016–2020 House, from precinct data)

## Key Dependencies

**Python:** pandas, numpy, linearmodels, statsmodels, matplotlib, pyarrow, requests, scipy

**R:** fixest (TWFE verification), arrow, sf, tigris, ggplot2, dplyr

## Conventions

- Scripts use `BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` for path resolution
- Raw data is CSV; analysis datasets are Parquet
- Figures are PNG with matplotlib Agg backend
- Smoke variable naming: `smoke_pm25_mean_30d`, `smoke_days_30d`, `smoke_days_severe_30d` (window suffix)
- `dem_vote_share` = DEM / (DEM + REP) two-party vote share
- The R script `analyze_house_voting_check.R` exists specifically to cross-validate Python regression results

## California Tract-Level Analysis

A parallel analysis pipeline at census-tract resolution for California, using `ca_` prefixed scripts. Data in `data/california/`, output to `output/california/`, report in `report/`.

```bash
# CA Pipeline (Steps 1-3 can run in parallel)
python scripts/ca_download_smoke_data.py          # 1. Tract-level smoke PM2.5
python scripts/ca_download_election_data.py        # 2. SWDB precinct data + crosswalk
python scripts/ca_download_controls_data.py        # 3a. ACS tract-level controls
Rscript scripts/ca_download_weather_controls.R     # 3b. PRISM → tract weather

python scripts/ca_build_crosswalk.py               # 4. Precinct→tract allocation
python scripts/ca_build_smoke_analysis.py          # 5. Merge → analysis parquets

python scripts/ca_analyze_smoke_voting.py          # 6. Regressions + figures
Rscript scripts/ca_analyze_check.R                 # 6b. fixest cross-validation
Rscript scripts/ca_map_smoke_tracts.R              # 7. Maps

cd report && pdflatex ca_tract_analysis.tex        # 8. Compile report
```

**CA-specific identifiers:** Census tract GEOID (`geoid`, 11-digit), `county_fips` (GEOID[:5]).
**FE structure:** Tract + Year (base), County trends (robustness), County×Year (most demanding).
**Additional Python dependency:** geopandas (for crosswalk spatial overlay)

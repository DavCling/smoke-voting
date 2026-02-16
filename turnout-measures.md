# Turnout Measurement: Denominator Choice

**Status:** To be resolved  
**Date:** 2026-02-16

## Question

How should we construct the county-level turnout variable? The key decision is the denominator.

## Options

### 1. Voting-Age Population (VAP)

- **Source:** Census intercensal population estimates (county, annual)
- **Pros:** Available annually, standard in the county-level elections literature (e.g., Gomez et al. 2007), consistent time series
- **Cons:** Includes non-citizens, disenfranchised felons, and other ineligible populations — measurement error varies systematically across counties (e.g., high-immigration counties show artificially low turnout)
- **Note:** County fixed effects absorb the bias to the extent that the ineligible share is roughly stable within counties over time

### 2. Citizen Voting-Age Population (CVAP)

- **Source:** ACS 5-year estimates (county level, available 2005+)
- **Pros:** Conceptually cleaner — closer to the actual eligible electorate
- **Cons:** Noisy for small counties due to ACS sampling; 5-year pooling makes the denominator sluggish and slow to reflect rapid population changes

### 3. Registered Voters

- **Source:** State election administration records
- **Pros:** Avoids population estimation issues entirely
- **Cons:** Registration is potentially endogenous — smoke exposure could affect registration itself, so this denominators conditions on a post-treatment variable

## Recommendation

Use **VAP as the primary measure** and **CVAP as a robustness check**. VAP is the standard in the literature we’re engaging with, and county fixed effects handle the main source of bias (cross-sectional variation in ineligible shares). Registered voters should be avoided as the primary denominator due to the endogeneity concern.

## Data Sources

- **Numerator:** Total votes cast, from MIT Election Data + Science Lab (MEDSL)
- **VAP:** Census Bureau intercensal population estimates
- **CVAP:** ACS 5-year estimates via Census Bureau
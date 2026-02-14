# Wildfire Smoke & Voting Behavior — TODO

## Data Extensions

- [ ] **NOAA HMS smoke plume data as robustness check**
  Rationale: The Stanford Echo Lab smoke PM2.5 (V1) only covers 2006–2020, limiting us to 4 presidential elections (2008–2020) and excluding 2024 and pre-2008 cycles. NOAA Hazard Mapping System (HMS) smoke plume data is available from 2005 through the present and provides a binary smoke/no-smoke indicator at the county level. Using HMS as an alternative treatment variable serves two purposes: (1) extends temporal coverage to include 2024 and potentially 2022 midterms, and (2) provides a robustness check on the main PM2.5 results using an independent data source with a different measurement approach (satellite-detected plume presence vs. ML-estimated concentration).

- [ ] **Echo Lab V2 smoke data (10km grid, 2006–2023)**
  Would add the 2022 midterms (major Western smoke season). Requires spatial aggregation from 10km grid to county boundaries. Beta release — revisit when stable or county-level version becomes available.

- [ ] **County population controls from Census Bureau**
  Currently using total votes as a population proxy. Download Census county population estimates (intercensal) for proper population denominators and turnout rates.

## Analysis Extensions

- [ ] **Wind direction instrument**
  Use NOAA/NCEP reanalysis wind data as an instrument for smoke exposure. Wind direction is exogenous to local politics, strengthening the causal interpretation.

- [ ] **Placebo test with winter smoke**
  Compute smoke exposure during January–March (non-fire season) as a placebo treatment. Should find null effects if the mechanism is genuinely fire-smoke-related.

- [ ] **State-by-year fixed effects**
  Add state × year FE to absorb state-level trends (e.g., California vs. Montana political dynamics). Current specs use county + year FE only.

- [ ] **Congressional House race outcomes**
  Download MEDSL county-level House returns and test whether smoke affects down-ballot races differently than presidential. House races occur every 2 years, so this also adds midterm elections (2006, 2010, 2014, 2018, and 2022 with extended smoke data), roughly doubling the number of election cycles.

- [ ] **U.S. Senate race outcomes**
  County-level Senate returns from MEDSL. Staggered 6-year terms create within-state variation in whether a Senate race is on the ballot, which could interact with smoke salience effects on turnout and partisan composition.

- [ ] **State legislative race outcomes**
  State legislative returns (e.g., from Carl Klarner's dataset or Harvard Dataverse state legislative election returns). Tests whether smoke effects propagate to sub-federal offices where environmental policy may feel more proximate. Also provides much larger N (thousands of districts) and finer geographic variation than county-level federal races.

- [ ] **Event study with finer windows**
  Test 7, 14, 21, 30, 45, 60, 90, 120-day windows systematically to pin down the temporal dynamics of the smoke-voting relationship.

- [ ] **Heterogeneity by smoke novelty**
  Counties that rarely experience smoke (e.g., East Coast) may react differently than counties accustomed to fire seasons (e.g., rural West). Interact smoke exposure with historical smoke frequency.

## Robustness Checks

- [ ] **Drop 2020 as a sensitivity check**
  2020 is an outlier in both smoke (massive season) and turnout (COVID, mail voting). Verify results hold in the 2008–2016 subsample.

- [ ] **Alternative smoke thresholds**
  Test sensitivity to the "smoke day" definition (currently PM2.5 > 0). Try > 1, > 5, > 12 (EPA "moderate") µg/m³ cutoffs.

- [ ] **Spatial clustering of standard errors**
  Current SEs are clustered by county. Test robustness to clustering by state or by state-year, and to Conley spatial HAC standard errors.

## Writing

- [ ] **Literature positioning**
  Frame relative to Hazlett & Mildenberger (2020, fire perimeters, CA ballot measures), Bellani et al. (2024, PM10, German elections), and Gomez et al. (2007, rain and turnout).

- [ ] **Results interpretation**
  Main effect: ~0.9 pp DEM shift per 10 µg/m³ (60-day window). Compare to Bellani et al.'s ~2 pp per 10 µg/m³ for PM10 in Germany. Discuss why the effect may be smaller (different pollutant measure, different political context, county vs. municipality aggregation).

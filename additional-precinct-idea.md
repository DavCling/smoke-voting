# Extending the Wildfire Smoke Analysis: Precinct-Level Data Beyond California

## Overview

The California Statewide Database (SWDB) is uniquely comprehensive, but post-2016 there are good options for adding states to the analysis. The best path is through VEST (Voting and Election Science Team), which provides precinct boundaries joined with election results for all 50 states.

## Primary Data Source: VEST

- **Coverage:** All 50 states for 2016, 2018, and 2020 general elections; 2022 partially available (~13 states); 2024 forthcoming
- **Format:** State-level shapefiles with precinct boundaries and election results already joined
- **Access:** Free via [Harvard Dataverse](https://dataverse.harvard.edu/dataverse/electionscience) under Creative Commons license
- **Citation:** Amos, B., Gerontakis, S. & McDonald, M. “United States Precinct Boundaries and Statewide Partisan Election Results.” *Scientific Data* 11, 1173 (2024). https://doi.org/10.1038/s41597-024-04024-2
- **Variables:** Vote counts for candidates in all partisan statewide offices (president, Senate, governor, etc.). Exact set varies by state and year; codebooks document this.

## Supplementary Sources

- **Redistricting Data Hub (RDH):** Hosts VEST data alongside files from MGGG and the Princeton Gerrymandering Project. Good search interface for browsing by state and year. https://redistrictingdatahub.org
- **MEDSL (MIT Election Data + Science Lab):** Precinct-level *results* (tabular only, no boundaries) for 2016–2024. Useful for cross-checking or filling gaps. https://electionlab.mit.edu/data
- **NYT 2024 Precinct Map:** Presidential results with shapefiles for 2024, available on GitHub. https://github.com/nytimes/presidential-precinct-map-2024
- **Washington Secretary of State:** Publishes official statewide precinct shapefiles annually. Closest analog to the California SWDB. https://www.sos.wa.gov/elections/data-research/election-data-and-maps/reports-data-and-statistics/precinct-shapefiles

## Priority States for Wildfire Smoke Study

### High-exposure western states

- **Oregon** and **Washington** — Heavy smoke exposure, especially 2017–2020; Washington has excellent official data
- **Colorado**, **Montana**, **Idaho** — Significant wildfire smoke exposure
- **Nevada**, **Utah**, **Arizona** — Varying exposure; useful for capturing the extensive margin

### Transported smoke states

- Northern Plains and Mountain West states that receive episodic smoke from western fires without local fire effects. Useful for identification: separates smoke exposure from direct fire damage, evacuation, property loss, etc.

## Workflow

1. Download VEST shapefiles for target states and years from Harvard Dataverse
1. Spatially join smoke/PM2.5 data to precinct polygons (same approach as California)
1. Harmonize variable names across states (VEST naming conventions differ somewhat by state)
1. Merge additional precinct-level covariates as needed

## Key Differences from the California SWDB

|Feature                    |California SWDB|VEST                    |
|---------------------------|---------------|------------------------|
|Precinct boundaries        |Yes            |Yes                     |
|Election results           |Yes            |Yes                     |
|Voter registration by party|Yes            |No                      |
|Time coverage              |1992–present   |2016–present            |
|Crosswalks to Census blocks|Yes            |Some (via ALARM project)|
|Update frequency           |Every election |Rolling, with lag       |

To get voter registration data for other states, you would need state voter files, which vary in cost and access rules by state.

## The 2020 Opportunity

The 2020 wildfire season was historically extreme in the West (Oregon, Washington, Colorado, California). VEST’s 2020 data is complete for all 50 states, making the 2020 presidential election a natural starting point for a multi-state extension.

## Notes

- VEST precinct boundaries change every election, so panel analysis across years requires crosswalking precincts (or aggregating to a stable geography like Census blocks or tracts)
- North Carolina adds small noise to precinct results when a candidate receives 100% of votes, per state law
- Some states (e.g., Alabama, Louisiana, Missouri) report absentee/provisional votes at the county level rather than precinct level — the NYT 2024 data documents how they allocated these
#!/usr/bin/env python3
"""
Download tract-level economic controls for California smoke-voting analysis.

Sources:
  A. ACS 5-year estimates via Census API (tract-level for CA):
     - B23025: Employment status (unemployment rate)
     - B19013: Median household income
     - B01003: Total population
     - B15003: Educational attainment (% bachelor's+)
     - B03002: Hispanic/Latino origin by race (race/ethnicity composition)

  Years: 2009-2022 (ACS 5-year available from 2009 onward)
  For election years 2006-2008, uses the 2009 vintage.

Output: data/california/controls/
  acs_tract_controls.csv
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "california", "controls")
OUT_FILE = os.path.join(OUT_DIR, "acs_tract_controls.csv")

# Census API base URL
CENSUS_API_BASE = "https://api.census.gov/data"

# ACS 5-year vintages available (2009 = first 5-year release, covers 2005-2009)
# For each election year, use the ACS vintage covering that year
ACS_VINTAGE_MAP = {
    2006: 2009,  # ACS 2005-2009
    2008: 2009,
    2010: 2010,  # ACS 2006-2010
    2012: 2012,
    2014: 2014,
    2016: 2016,
    2018: 2018,
    2020: 2020,
    2022: 2022,
}

# Variables to download from ACS
ACS_VARIABLES = {
    # Employment status (B23025)
    "B23025_003E": "labor_force",       # In labor force: Civilian labor force
    "B23025_005E": "unemployed",        # In labor force: Civilian: Unemployed

    # Median household income (B19013)
    "B19013_001E": "median_income",

    # Total population (B01003)
    "B01003_001E": "total_population",

    # Educational attainment 25+ (B15003)
    "B15003_001E": "educ_total",        # Total
    "B15003_022E": "educ_bachelors",    # Bachelor's degree
    "B15003_023E": "educ_masters",      # Master's degree
    "B15003_024E": "educ_professional", # Professional school degree
    "B15003_025E": "educ_doctorate",    # Doctorate degree

    # Hispanic/Latino by race (B03002)
    "B03002_001E": "race_total",        # Total
    "B03002_003E": "race_white_nh",     # Not Hispanic: White alone
    "B03002_004E": "race_black_nh",     # Not Hispanic: Black alone
    "B03002_006E": "race_asian_nh",     # Not Hispanic: Asian alone
    "B03002_012E": "race_hispanic",     # Hispanic or Latino
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def fetch_acs_tract_data(vintage, variables, state_fips="06"):
    """Fetch ACS 5-year tract-level data for California from Census API."""
    var_str = ",".join(variables)
    url = (
        f"{CENSUS_API_BASE}/{vintage}/acs/acs5"
        f"?get={var_str}"
        f"&for=tract:*"
        f"&in=state:{state_fips}"
    )

    # Add API key if available
    api_key = os.environ.get("CENSUS_API_KEY")
    if api_key:
        url += f"&key={api_key}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    ERROR fetching ACS {vintage}: {e}")
        return None

    data = resp.json()
    if len(data) < 2:
        print(f"    WARNING: No data returned for ACS {vintage}")
        return None

    df = pd.DataFrame(data[1:], columns=data[0])
    return df


def get_available_variables(acs_vintage):
    """Return ACS variables available for a given vintage.

    B23025 (employment) first available in 2011 5-year release.
    B15003 (educational attainment) first available in 2012 5-year release.
    B19013 (median income), B01003 (population), B03002 (race) available from 2009.
    """
    available = {}
    for var, label in ACS_VARIABLES.items():
        table = var.split("_")[0]
        if table == "B23025" and acs_vintage < 2011:
            continue
        if table == "B15003" and acs_vintage < 2012:
            continue
        available[var] = label
    return available


def download_acs_controls():
    """Download ACS tract-level controls for all election years."""
    print("\n" + "=" * 60)
    print("Downloading ACS 5-Year Tract-Level Controls")
    print("=" * 60)

    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE)
        print(f"  Already exists: {OUT_FILE} ({len(df):,} rows)")
        return

    all_data = []

    for election_year, acs_vintage in sorted(ACS_VINTAGE_MAP.items()):
        print(f"\n  Election {election_year} (ACS vintage {acs_vintage})...")

        avail_vars = get_available_variables(acs_vintage)
        variables = list(avail_vars.keys())
        if not variables:
            print(f"    No variables available for vintage {acs_vintage}, skipping")
            continue

        missing_tables = set()
        for var in ACS_VARIABLES:
            if var not in avail_vars:
                missing_tables.add(var.split("_")[0])
        if missing_tables:
            print(f"    Note: tables {', '.join(sorted(missing_tables))} not available for {acs_vintage}")

        df = fetch_acs_tract_data(acs_vintage, variables)
        if df is None:
            continue

        # Construct GEOID (11-digit tract FIPS)
        df["state"] = df["state"].str.zfill(2)
        df["county"] = df["county"].str.zfill(3)
        df["tract"] = df["tract"].str.zfill(6)
        df["GEOID"] = df["state"] + df["county"] + df["tract"]

        # Rename variables
        rename_map = {k: v for k, v in avail_vars.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Convert to numeric
        for col in avail_vars.values():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["election_year"] = election_year
        df["acs_vintage"] = acs_vintage

        # Keep only GEOID + variable columns
        keep_cols = ["GEOID", "election_year", "acs_vintage"] + [
            v for v in avail_vars.values() if v in df.columns
        ]
        df = df[keep_cols]

        n_tracts = df["GEOID"].nunique()
        print(f"    {n_tracts:,} tracts, {len(df):,} rows")

        all_data.append(df)

        # Rate limit
        time.sleep(1)

    if not all_data:
        print("  ERROR: No ACS data downloaded!")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Compute derived variables
    print("\n  Computing derived variables...")

    # Unemployment rate (B23025 — available from 2011+)
    if "labor_force" in combined.columns and "unemployed" in combined.columns:
        combined["unemployment_rate"] = np.where(
            combined["labor_force"] > 0,
            combined["unemployed"] / combined["labor_force"],
            np.nan
        )
    else:
        combined["unemployment_rate"] = np.nan

    # Log median income (handle negative/zero/missing)
    if "median_income" in combined.columns:
        combined["log_median_income"] = np.where(
            combined["median_income"] > 0,
            np.log(combined["median_income"]),
            np.nan
        )
    else:
        combined["log_median_income"] = np.nan

    # Log population
    if "total_population" in combined.columns:
        combined["log_population"] = np.where(
            combined["total_population"] > 0,
            np.log(combined["total_population"]),
            np.nan
        )
    else:
        combined["log_population"] = np.nan

    # Pct bachelor's or higher (B15003 — available from 2012+)
    if "educ_total" in combined.columns:
        educ_higher = (
            combined["educ_bachelors"].fillna(0) +
            combined["educ_masters"].fillna(0) +
            combined["educ_professional"].fillna(0) +
            combined["educ_doctorate"].fillna(0)
        )
        combined["pct_bachelors_plus"] = np.where(
            combined["educ_total"] > 0,
            educ_higher / combined["educ_total"],
            np.nan
        )
    else:
        combined["pct_bachelors_plus"] = np.nan

    # Race/ethnicity shares
    for race_var, label in [
        ("race_white_nh", "pct_white_nh"),
        ("race_black_nh", "pct_black_nh"),
        ("race_asian_nh", "pct_asian_nh"),
        ("race_hispanic", "pct_hispanic"),
    ]:
        if race_var in combined.columns and "race_total" in combined.columns:
            combined[label] = np.where(
                combined["race_total"] > 0,
                combined[race_var].fillna(0) / combined["race_total"],
                np.nan
            )
        else:
            combined[label] = np.nan

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    combined.to_csv(OUT_FILE, index=False)
    print(f"\n  Saved: {OUT_FILE}")
    print(f"  Shape: {combined.shape}")
    print(f"  Tracts: {combined['GEOID'].nunique():,}")
    print(f"  Election years: {sorted(combined['election_year'].unique())}")

    # Coverage report
    print("\n  Coverage (% non-missing):")
    derived_cols = [
        "unemployment_rate", "log_median_income", "log_population",
        "pct_bachelors_plus", "pct_white_nh", "pct_hispanic",
    ]
    for col in derived_cols:
        if col in combined.columns:
            pct = combined[col].notna().mean() * 100
            print(f"    {col}: {pct:.1f}%")


def main():
    print("=" * 60)
    print("CA Step 3a: Download Tract-Level ACS Controls")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)
    download_acs_controls()

    print(f"\nStep 3a complete: {OUT_DIR}")


if __name__ == "__main__":
    main()

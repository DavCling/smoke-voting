#!/usr/bin/env python3
"""
Download ACS 5-year tract-level controls for all CONUS states via Census API.

Sources:
  ACS 5-year estimates (tract-level for all CONUS states):
    - B23025: Employment status (unemployment rate)
    - B19013: Median household income
    - B01003: Total population
    - B15003: Educational attainment (% bachelor's+)
    - B03002: Hispanic/Latino origin by race (race/ethnicity composition)

  Years: 2016 and 2020 election years (ACS 5-year vintages 2016 and 2020)

Output: data/national_tracts/controls/acs_tract_controls.csv
"""

import os
import time
import requests
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "national_tracts", "controls")
OUT_FILE = os.path.join(OUT_DIR, "acs_tract_controls.csv")

CENSUS_API_BASE = "https://api.census.gov/data"

# CONUS state FIPS codes (exclude AK=02, HI=15, and territories)
CONUS_FIPS = [
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
]

# Election year -> ACS vintage mapping
ACS_VINTAGE_MAP = {
    2016: 2016,
    2020: 2020,
}

# Variables to download from ACS
ACS_VARIABLES = {
    # Employment status (B23025)
    "B23025_003E": "labor_force",
    "B23025_005E": "unemployed",
    # Median household income (B19013)
    "B19013_001E": "median_income",
    # Total population (B01003)
    "B01003_001E": "total_population",
    # Educational attainment 25+ (B15003)
    "B15003_001E": "educ_total",
    "B15003_022E": "educ_bachelors",
    "B15003_023E": "educ_masters",
    "B15003_024E": "educ_professional",
    "B15003_025E": "educ_doctorate",
    # Hispanic/Latino by race (B03002)
    "B03002_001E": "race_total",
    "B03002_003E": "race_white_nh",
    "B03002_004E": "race_black_nh",
    "B03002_006E": "race_asian_nh",
    "B03002_012E": "race_hispanic",
    # Sex by age — under 18 components for VAP (B01001)
    "B01001_003E": "male_under5",
    "B01001_004E": "male_5to9",
    "B01001_005E": "male_10to14",
    "B01001_006E": "male_15to17",
    "B01001_027E": "female_under5",
    "B01001_028E": "female_5to9",
    "B01001_029E": "female_10to14",
    "B01001_030E": "female_15to17",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def fetch_acs_tract_data(vintage, variables, state_fips):
    """Fetch ACS 5-year tract-level data for a single state from Census API."""
    var_str = ",".join(variables)
    url = (
        f"{CENSUS_API_BASE}/{vintage}/acs/acs5"
        f"?get={var_str}"
        f"&for=tract:*"
        f"&in=state:{state_fips}"
    )

    api_key = os.environ.get("CENSUS_API_KEY")
    if api_key:
        url += f"&key={api_key}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"      ERROR fetching ACS {vintage} state {state_fips}: {e}")
        return None

    data = resp.json()
    if len(data) < 2:
        print(f"      WARNING: No data for state {state_fips}")
        return None

    df = pd.DataFrame(data[1:], columns=data[0])
    return df


def download_acs_controls():
    """Download ACS tract-level controls for all CONUS states."""
    print("\n" + "=" * 60)
    print("Downloading ACS 5-Year Tract-Level Controls (All CONUS)")
    print("=" * 60)

    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE)
        print(f"  Already exists: {OUT_FILE} ({len(df):,} rows)")
        return

    variables = list(ACS_VARIABLES.keys())
    all_data = []

    for election_year, acs_vintage in sorted(ACS_VINTAGE_MAP.items()):
        print(f"\n  Election {election_year} (ACS vintage {acs_vintage}):")

        for state_fips in sorted(CONUS_FIPS):
            df = fetch_acs_tract_data(acs_vintage, variables, state_fips)
            if df is None:
                continue

            # Construct 11-digit GEOID
            df["state"] = df["state"].str.zfill(2)
            df["county"] = df["county"].str.zfill(3)
            df["tract"] = df["tract"].str.zfill(6)
            df["GEOID"] = df["state"] + df["county"] + df["tract"]

            # Rename variables
            rename_map = {k: v for k, v in ACS_VARIABLES.items() if k in df.columns}
            df = df.rename(columns=rename_map)

            # Convert to numeric
            for col in ACS_VARIABLES.values():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df["election_year"] = election_year
            df["acs_vintage"] = acs_vintage

            keep_cols = ["GEOID", "election_year", "acs_vintage"] + [
                v for v in ACS_VARIABLES.values() if v in df.columns
            ]
            df = df[keep_cols]
            all_data.append(df)

            time.sleep(0.5)  # Rate limit

        n_state = sum(1 for d in all_data if d["election_year"].iloc[0] == election_year)
        print(f"    Downloaded {n_state} states")

    if not all_data:
        print("  ERROR: No ACS data downloaded!")
        return

    combined = pd.concat(all_data, ignore_index=True)

    # Compute derived variables
    print("\n  Computing derived variables...")

    # Unemployment rate
    combined["unemployment_rate"] = np.where(
        combined["labor_force"] > 0,
        combined["unemployed"] / combined["labor_force"],
        np.nan,
    )

    # Log median income
    combined["log_median_income"] = np.where(
        combined["median_income"] > 0,
        np.log(combined["median_income"]),
        np.nan,
    )

    # Log population
    combined["log_population"] = np.where(
        combined["total_population"] > 0,
        np.log(combined["total_population"]),
        np.nan,
    )

    # Pct bachelor's or higher
    educ_higher = (
        combined["educ_bachelors"].fillna(0)
        + combined["educ_masters"].fillna(0)
        + combined["educ_professional"].fillna(0)
        + combined["educ_doctorate"].fillna(0)
    )
    combined["pct_bachelors_plus"] = np.where(
        combined["educ_total"] > 0,
        educ_higher / combined["educ_total"],
        np.nan,
    )

    # Race/ethnicity shares
    for race_var, label in [
        ("race_white_nh", "pct_white_nh"),
        ("race_black_nh", "pct_black_nh"),
        ("race_asian_nh", "pct_asian_nh"),
        ("race_hispanic", "pct_hispanic"),
    ]:
        combined[label] = np.where(
            combined["race_total"] > 0,
            combined[race_var].fillna(0) / combined["race_total"],
            np.nan,
        )

    # Voting-age population (total pop minus under-18)
    under18_cols = ["male_under5", "male_5to9", "male_10to14", "male_15to17",
                    "female_under5", "female_5to9", "female_10to14", "female_15to17"]
    if all(c in combined.columns for c in under18_cols):
        under18 = sum(combined[c].fillna(0) for c in under18_cols)
        combined["voting_age_population"] = combined["total_population"] - under18
        combined["voting_age_population"] = combined["voting_age_population"].clip(lower=0)

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
        "voting_age_population",
        "pct_bachelors_plus", "pct_white_nh", "pct_hispanic",
    ]
    for col in derived_cols:
        if col in combined.columns:
            pct = combined[col].notna().mean() * 100
            print(f"    {col}: {pct:.1f}%")


def main():
    print("=" * 60)
    print("National Tract-Level: Download ACS Controls")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)
    download_acs_controls()

    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()

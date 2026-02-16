#!/usr/bin/env python3
"""
Build national tract-level analysis dataset merging smoke exposure,
elections, and controls into a single parquet file.

Input:
  data/national_tracts/elections/tract_presidential.csv
  data/national_tracts/smoke/tract_smoke_exposure.parquet
  data/national_tracts/controls/acs_tract_controls.csv
  data/national_tracts/controls/tract_weather_october.csv

Output:
  output/national_tracts/tract_smoke_voting_presidential.parquet
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input files
ELECTION_FILE = os.path.join(BASE_DIR, "data", "national_tracts", "elections",
                             "tract_presidential.csv")
SMOKE_FILE = os.path.join(BASE_DIR, "data", "national_tracts", "smoke",
                          "tract_smoke_exposure.parquet")
ACS_FILE = os.path.join(BASE_DIR, "data", "national_tracts", "controls",
                        "acs_tract_controls.csv")
WEATHER_FILE = os.path.join(BASE_DIR, "data", "national_tracts", "controls",
                            "tract_weather_october.csv")

# Output
OUT_DIR = os.path.join(BASE_DIR, "output", "national_tracts")
OUT_FILE = os.path.join(OUT_DIR, "tract_smoke_voting_presidential.parquet")

# CONUS state FIPS codes
CONUS_FIPS = {
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
}


def load_elections():
    """Load tract-level election data."""
    print("Loading election data...")
    if not os.path.exists(ELECTION_FILE):
        print(f"  ERROR: {ELECTION_FILE} not found")
        return None

    df = pd.read_csv(ELECTION_FILE, dtype={"tract_GEOID": str})
    df["tract_GEOID"] = df["tract_GEOID"].str.zfill(11)
    df = df.rename(columns={"tract_GEOID": "geoid"})

    # Filter to CONUS
    df = df[df["geoid"].str[:2].isin(CONUS_FIPS)].copy()

    print(f"  {len(df):,} rows, {df['geoid'].nunique():,} tracts, "
          f"years: {sorted(df['year'].unique())}")
    return df


def load_smoke():
    """Load tract-level smoke exposure."""
    print("\nLoading smoke exposure data...")
    if not os.path.exists(SMOKE_FILE):
        print(f"  ERROR: {SMOKE_FILE} not found")
        return None

    df = pd.read_parquet(SMOKE_FILE)
    df["geoid"] = df["geoid"].astype(str).str.zfill(11)
    print(f"  {len(df):,} rows, {df['geoid'].nunique():,} tracts")
    return df


def load_controls():
    """Load and merge ACS + weather controls."""
    print("\nLoading controls...")
    dfs = []

    # ACS controls
    if os.path.exists(ACS_FILE):
        acs = pd.read_csv(ACS_FILE, dtype={"GEOID": str})
        acs["GEOID"] = acs["GEOID"].str.zfill(11)
        acs = acs.rename(columns={"GEOID": "geoid", "election_year": "year"})

        keep_cols = ["geoid", "year", "unemployment_rate", "log_median_income",
                     "log_population", "voting_age_population", "pct_bachelors_plus",
                     "pct_white_nh", "pct_black_nh", "pct_asian_nh", "pct_hispanic"]
        keep_cols = [c for c in keep_cols if c in acs.columns]
        acs = acs[keep_cols]
        print(f"  ACS: {len(acs):,} rows, {acs['geoid'].nunique():,} tracts")
        dfs.append(acs)
    else:
        print(f"  WARNING: ACS controls not found: {ACS_FILE}")

    # Weather controls
    if os.path.exists(WEATHER_FILE):
        weather = pd.read_csv(WEATHER_FILE, dtype={"GEOID": str})
        weather["GEOID"] = weather["GEOID"].str.zfill(11)
        weather = weather.rename(columns={"GEOID": "geoid"})
        print(f"  Weather: {len(weather):,} rows, {weather['geoid'].nunique():,} tracts")
        dfs.append(weather[["geoid", "year", "october_tmean", "october_ppt"]])
    else:
        print(f"  WARNING: Weather controls not found: {WEATHER_FILE}")

    if not dfs:
        return None

    controls = dfs[0]
    for df in dfs[1:]:
        controls = controls.merge(df, on=["geoid", "year"], how="outer")

    print(f"  Combined controls: {len(controls):,} rows, "
          f"{controls['geoid'].nunique():,} tracts")
    return controls


def build_analysis_dataset(elections, smoke, controls):
    """Merge smoke, elections, and controls into analysis dataset."""
    print("\nBuilding analysis dataset...")

    # Inner join smoke + elections on (geoid, year)
    merged = elections.merge(smoke, on=["geoid", "year"], how="inner")
    print(f"  Smoke + elections: {len(merged):,} tract-year observations "
          f"({merged['geoid'].nunique():,} tracts)")

    if len(merged) == 0:
        print("  ERROR: No matches between smoke and election data!")
        return None

    # Left join controls
    if controls is not None:
        merged = merged.merge(controls, on=["geoid", "year"], how="left")
        n_ctrl = merged["unemployment_rate"].notna().sum() if "unemployment_rate" in merged.columns else 0
        print(f"  Controls matched: {n_ctrl:,}/{len(merged):,}")

    # Add identifiers
    merged["county_fips"] = merged["geoid"].str[:5]
    merged["state_fips"] = merged["geoid"].str[:2]
    merged["state_year"] = merged["state_fips"] + "_" + merged["year"].astype(str)
    merged["county_year"] = merged["county_fips"] + "_" + merged["year"].astype(str)

    # No lagged vote share (only 2 years — would require 2012 data)

    # Turnout rate = total votes / voting-age population
    if "voting_age_population" in merged.columns and "total_votes" in merged.columns:
        merged["turnout_rate"] = np.where(
            merged["voting_age_population"] > 0,
            merged["total_votes"] / merged["voting_age_population"],
            np.nan,
        )
        # Clip extreme values from allocation artifacts (VAP/vote mismatch)
        n_extreme = (merged["turnout_rate"] > 1.5).sum()
        merged.loc[merged["turnout_rate"] > 1.5, "turnout_rate"] = np.nan
        n_tr = merged["turnout_rate"].notna().sum()
        print(f"  Turnout rate computed: {n_tr:,}/{len(merged):,} "
              f"({n_extreme} extreme values dropped)")

    return merged


def print_summary(df):
    """Print summary statistics for the analysis dataset."""
    print(f"\n{'='*60}")
    print("ANALYSIS DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Observations: {len(df):,}")
    print(f"  Tracts: {df['geoid'].nunique():,}")
    print(f"  Counties: {df['county_fips'].nunique():,}")
    print(f"  States: {df['state_fips'].nunique()}")
    print(f"  Elections: {sorted(df['year'].unique())}")

    print(f"\n  Outcomes:")
    for col in ["dem_vote_share", "incumbent_vote_share", "log_total_votes", "turnout_rate"]:
        if col in df.columns:
            valid = df[col].dropna()
            print(f"    {col}: mean={valid.mean():.4f}, sd={valid.std():.4f}, "
                  f"n={len(valid):,}")

    print(f"\n  Smoke exposure (30d):")
    for col in ["smoke_pm25_mean_30d", "smoke_days_30d", "smoke_frac_haze_30d"]:
        if col in df.columns:
            print(f"    {col}: mean={df[col].mean():.4f}, "
                  f"max={df[col].max():.2f}")

    print(f"\n  Smoke by year:")
    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr]
        s30 = "smoke_pm25_mean_30d"
        if s30 in yr_df.columns:
            print(f"    {yr}: n={len(yr_df):,}, mean PM2.5={yr_df[s30].mean():.4f}, "
                  f"frac_haze={yr_df['smoke_frac_haze_30d'].mean():.4f}")

    # Controls coverage
    ctrl_cols = ["unemployment_rate", "log_median_income", "log_population",
                 "october_tmean", "october_ppt", "pct_bachelors_plus"]
    ctrl_available = [c for c in ctrl_cols if c in df.columns]
    if ctrl_available:
        print(f"\n  Controls coverage (% non-missing):")
        for col in ctrl_available:
            pct = df[col].notna().mean() * 100
            print(f"    {col}: {pct:.1f}%")

    # Per-state tract counts
    print(f"\n  Tracts per state:")
    state_counts = df.groupby("state_fips")["geoid"].nunique().sort_index()
    for fips, count in state_counts.items():
        print(f"    {fips}: {count:,}")


def main():
    print("=" * 60)
    print("National Tract-Level: Build Analysis Dataset")
    print("=" * 60)

    if os.path.exists(OUT_FILE):
        df = pd.read_parquet(OUT_FILE)
        print(f"Already exists: {OUT_FILE}")
        print(f"  {len(df):,} rows, {df['geoid'].nunique():,} tracts")
        print_summary(df)
        return

    # Load data
    elections = load_elections()
    smoke = load_smoke()
    if elections is None or smoke is None:
        print("\nERROR: Missing input data. Run download/extract scripts first.")
        return

    controls = load_controls()

    # Build
    df = build_analysis_dataset(elections, smoke, controls)
    if df is None:
        return

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(f"\n  Saved: {OUT_FILE}")

    print_summary(df)

    print(f"\nBuild complete.")


if __name__ == "__main__":
    main()

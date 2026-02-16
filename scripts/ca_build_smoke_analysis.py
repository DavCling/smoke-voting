#!/usr/bin/env python3
"""
Build CA tract-level analysis datasets merging smoke exposure with election returns.

Aggregates daily tract-level smoke PM2.5 into pre-election windows, merges with
tract-level election outcomes (from crosswalk step), and adds controls.

Input:
  data/california/smoke/smoke_pm25_tract_daily.csv
  data/california/elections/tract_presidential.csv
  data/california/elections/tract_house.csv
  data/california/controls/acs_tract_controls.csv
  data/california/controls/tract_weather_october.csv

Output:
  output/california/ca_smoke_voting_presidential.parquet
  output/california/ca_smoke_voting_house.parquet
  output/california/ca_controls_panel.parquet
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROSSWALK_DIR = os.path.join(BASE_DIR, "data", "california", "crosswalk")
TRACT_RELATIONSHIP_FILE = os.path.join(CROSSWALK_DIR, "tract_2020_to_2010_natl.txt")
SMOKE_FILE = os.path.join(BASE_DIR, "data", "california", "smoke",
                          "smoke_pm25_tract_daily.csv")
PRES_ELECTION_FILE = os.path.join(BASE_DIR, "data", "california", "elections",
                                  "tract_presidential.csv")
HOUSE_ELECTION_FILE = os.path.join(BASE_DIR, "data", "california", "elections",
                                   "tract_house.csv")
ACS_FILE = os.path.join(BASE_DIR, "data", "california", "controls",
                        "acs_tract_controls.csv")
WEATHER_FILE = os.path.join(BASE_DIR, "data", "california", "controls",
                            "tract_weather_october.csv")

OUT_DIR = os.path.join(BASE_DIR, "output", "california")
PRES_OUT = os.path.join(OUT_DIR, "ca_smoke_voting_presidential.parquet")
HOUSE_OUT = os.path.join(OUT_DIR, "ca_smoke_voting_house.parquet")
CONTROLS_OUT = os.path.join(OUT_DIR, "ca_controls_panel.parquet")

# Election dates (first Tuesday after first Monday in November)
ELECTION_DATES = {
    2006: "2006-11-07",
    2008: "2008-11-04",
    2010: "2010-11-02",
    2012: "2012-11-06",
    2014: "2014-11-04",
    2016: "2016-11-08",
    2018: "2018-11-06",
    2020: "2020-11-03",
    2022: "2022-11-08",
}

# EPA thresholds for smoke PM2.5 (µg/m³)
HAZE_THRESHOLD = 20.0      # Visible haze onset
EPA_USG_THRESHOLD = 35.5   # Unhealthy for Sensitive Groups
EPA_UNHEALTHY = 55.5       # Unhealthy


def load_smoke_data():
    """Load and standardize CA tract-level smoke PM2.5 data.

    IMPORTANT: The source file contains ONLY smoke days. Non-smoke-day
    tract-days have smokePM_pred = 0 and are not in the file. The
    compute_smoke_exposure function accounts for this by using the total
    number of calendar days in each window as the denominator for means
    and fractions, not just the number of rows present.
    """
    print("Loading CA tract-level smoke data...")
    if not os.path.exists(SMOKE_FILE):
        print(f"  ERROR: Smoke file not found: {SMOKE_FILE}")
        print("  Run ca_download_smoke_data.py first.")
        return None

    df = pd.read_csv(SMOKE_FILE, dtype={"GEOID": str})
    df = df.rename(columns={"GEOID": "geoid", "smokePM_pred": "smoke_pm25"})
    df["geoid"] = df["geoid"].str.zfill(11)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    print(f"  {len(df):,} rows (smoke days only), {df['geoid'].nunique():,} tracts")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Mean smoke PM2.5 (on smoke days): {df['smoke_pm25'].mean():.4f} µg/m³")

    return df[["geoid", "date", "smoke_pm25"]]


def compute_smoke_exposure(smoke_df, election_year, election_date, all_tracts=None):
    """Compute smoke exposure measures for a single election.

    The source data contains ONLY smoke days (non-smoke days = 0, omitted).
    This function accounts for that by:
      - Using total calendar days in each window as the denominator for means
        and fractions (not the number of smoke-day rows).
      - Including all CA tracts in the output, not just those with smoke days.
        Tracts with no smoke-day rows in a window get 0 for all measures.
    """
    edate = pd.Timestamp(election_date)

    # Define windows
    windows = {
        "7d": (edate - timedelta(days=7), edate),
        "14d": (edate - timedelta(days=14), edate),
        "21d": (edate - timedelta(days=21), edate),
        "28d": (edate - timedelta(days=28), edate),
        "30d": (edate - timedelta(days=30), edate),
        "60d": (edate - timedelta(days=60), edate),
        "90d": (edate - timedelta(days=90), edate),
    }
    season_start = pd.Timestamp(f"{election_year}-06-01")
    windows["season"] = (season_start, edate)

    # Filter to broadest window
    earliest = min(start for start, _ in windows.values())
    smoke_window = smoke_df[
        (smoke_df["date"] >= earliest) & (smoke_df["date"] <= edate)
    ].copy()

    # Start with all known tracts (from the full smoke file or provided list)
    if all_tracts is not None:
        unique_tracts = all_tracts
    else:
        unique_tracts = smoke_df["geoid"].unique()

    result_df = pd.DataFrame({"geoid": unique_tracts})
    result_df["year"] = election_year

    for label, (start, end) in windows.items():
        # Total calendar days in this window
        n_days = (end - start).days

        w = smoke_window[(smoke_window["date"] >= start) & (smoke_window["date"] <= end)]
        grp = w.groupby("geoid")["smoke_pm25"]

        # Aggregate smoke-day rows per tract
        agg = grp.agg(
            # Number of rows with smoke_pm25 > 0 (all rows are smoke days,
            # but some smoke days may have smokePM_pred = 0)
            smoke_days=lambda x: (x > 0).sum(),
            # Sum of smoke PM2.5 across smoke-day rows (for computing mean)
            smoke_sum=lambda x: x.sum(),
            smoke_max=lambda x: x.max(),
            smoke_severe=lambda x: (x > EPA_USG_THRESHOLD).sum(),
            smoke_n_haze=lambda x: (x > HAZE_THRESHOLD).sum(),
            smoke_n_usg=lambda x: (x > EPA_USG_THRESHOLD).sum(),
            smoke_n_unhealthy=lambda x: (x > EPA_UNHEALTHY).sum(),
        )

        # Mean = sum of smoke PM2.5 / total calendar days in window
        # (non-smoke days contribute 0 to the numerator)
        agg[f"smoke_pm25_mean_{label}"] = agg["smoke_sum"] / n_days
        agg[f"smoke_days_{label}"] = agg["smoke_days"]
        agg[f"smoke_pm25_max_{label}"] = agg["smoke_max"]
        agg[f"smoke_days_severe_{label}"] = agg["smoke_severe"]
        agg[f"smoke_pm25_cumul_{label}"] = agg["smoke_sum"]
        agg[f"smoke_frac_haze_{label}"] = agg["smoke_n_haze"] / n_days
        agg[f"smoke_frac_usg_{label}"] = agg["smoke_n_usg"] / n_days
        agg[f"smoke_frac_unhealthy_{label}"] = agg["smoke_n_unhealthy"] / n_days

        # Keep only the final suffixed columns (drop intermediate ones)
        keep_cols = [c for c in agg.columns if label in c]
        agg = agg[keep_cols].reset_index()

        result_df = result_df.merge(agg, on="geoid", how="left")

    # Tracts with no smoke-day rows in a window get 0
    result_df = result_df.fillna(0)
    return result_df


def load_tract_2020_to_2010_map():
    """Load Census 2020-to-2010 tract relationship file for CA.

    Returns a DataFrame mapping 2020 Census GEOIDs to 2010 Census GEOIDs
    with area weights, used to convert 2022 ACS controls (which use 2020
    Census tract geography) to 2010 Census GEOIDs matching the smoke data.
    """
    if not os.path.exists(TRACT_RELATIONSHIP_FILE):
        return None

    df = pd.read_csv(TRACT_RELATIONSHIP_FILE, sep="|", dtype=str)
    ca = df[df["GEOID_TRACT_20"].str.startswith("06")].copy()

    ca["AREALAND_PART"] = pd.to_numeric(ca["AREALAND_PART"], errors="coerce").fillna(0)
    total_area = ca.groupby("GEOID_TRACT_20")["AREALAND_PART"].transform("sum")
    ca["area_weight"] = np.where(total_area > 0, ca["AREALAND_PART"] / total_area, 0)

    result = ca[["GEOID_TRACT_20", "GEOID_TRACT_10", "area_weight"]].copy()
    result = result.rename(columns={"GEOID_TRACT_20": "geoid_2020", "GEOID_TRACT_10": "geoid_2010"})
    result = result[result["area_weight"] > 0].copy()

    return result


def convert_controls_2020_to_2010(acs_df, tract_map):
    """Convert 2022 ACS controls from 2020 Census GEOIDs to 2010 Census GEOIDs.

    For rate/proportion variables (unemployment_rate, pct_*), uses
    area-weighted averages. For log variables (log_median_income,
    log_population), uses area-weighted averages. This approximation is
    reasonable because most 2020 tracts map almost entirely to one 2010 tract.
    """
    # Split into 2022 (needs conversion) and non-2022 (already 2010 GEOIDs)
    mask_2022 = acs_df["year"] == 2022
    acs_other = acs_df[~mask_2022].copy()
    acs_2022 = acs_df[mask_2022].copy()

    if len(acs_2022) == 0 or tract_map is None:
        return acs_df

    n_before = acs_2022["geoid"].nunique()

    # Merge with mapping
    merged = acs_2022.merge(
        tract_map.rename(columns={"geoid_2020": "geoid"}),
        on="geoid",
        how="left",
    )

    # Unmapped tracts: keep as-is
    unmapped = merged["geoid_2010"].isna()
    if unmapped.any():
        merged.loc[unmapped, "geoid_2010"] = merged.loc[unmapped, "geoid"]
        merged.loc[unmapped, "area_weight"] = 1.0

    # Replace geoid with 2010 version
    merged["geoid"] = merged["geoid_2010"]

    # Weighted average for all numeric control columns
    numeric_cols = [c for c in merged.columns
                    if c not in ("geoid", "year", "geoid_2010", "area_weight", "acs_vintage")
                    and merged[c].dtype in ("float64", "int64")]

    # Apply weights
    for col in numeric_cols:
        merged[col] = merged[col] * merged["area_weight"]

    # Sum weights per 2010 tract (for normalizing weighted averages)
    merged["_weight_sum"] = merged["area_weight"]

    agg_dict = {col: "sum" for col in numeric_cols}
    agg_dict["_weight_sum"] = "sum"

    result = merged.groupby(["geoid", "year"], dropna=False).agg(agg_dict).reset_index()

    # Normalize by weight sum to get weighted averages
    for col in numeric_cols:
        result[col] = result[col] / result["_weight_sum"]

    result = result.drop(columns=["_weight_sum"])
    result = result[result["geoid"].str.startswith("06")].copy()

    n_after = result["geoid"].nunique()
    print(f"  ACS 2022: converted 2020→2010 tracts: {n_before} → {n_after}")

    # Recombine
    # Ensure same columns
    common_cols = [c for c in acs_other.columns if c in result.columns]
    combined = pd.concat([acs_other[common_cols], result[common_cols]], ignore_index=True)
    return combined


def load_controls():
    """Load and merge ACS + weather controls into a single tract-year panel."""
    print("\nLoading controls...")
    dfs = []

    # Load 2020→2010 tract mapping for ACS 2022 conversion
    tract_map = load_tract_2020_to_2010_map()

    # ACS controls
    if os.path.exists(ACS_FILE):
        acs = pd.read_csv(ACS_FILE, dtype={"GEOID": str})
        acs["GEOID"] = acs["GEOID"].str.zfill(11)
        acs = acs.rename(columns={"GEOID": "geoid", "election_year": "year"})

        # Keep derived variables
        keep_cols = ["geoid", "year", "unemployment_rate", "log_median_income",
                     "log_population", "voting_age_population", "pct_bachelors_plus",
                     "pct_white_nh", "pct_black_nh", "pct_asian_nh", "pct_hispanic"]
        keep_cols = [c for c in keep_cols if c in acs.columns]
        acs = acs[keep_cols]
        print(f"  ACS: {len(acs):,} rows, {acs['geoid'].nunique():,} tracts")

        # Convert 2022 ACS from 2020 Census GEOIDs to 2010 Census GEOIDs
        if tract_map is not None and 2022 in acs["year"].values:
            acs = convert_controls_2020_to_2010(acs, tract_map)

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

    # Merge controls
    controls = dfs[0]
    for df in dfs[1:]:
        controls = controls.merge(df, on=["geoid", "year"], how="outer")

    print(f"  Combined controls: {len(controls):,} rows, {controls['geoid'].nunique():,} tracts")
    return controls


def build_analysis_dataset(smoke_panel, election_df, controls, race_type="presidential"):
    """Merge smoke, elections, and controls into analysis dataset."""
    print(f"\nBuilding {race_type} analysis dataset...")

    if election_df is None:
        print(f"  ERROR: No {race_type} election data!")
        return None

    # Ensure consistent types
    election_df["geoid"] = election_df["geoid"].astype(str).str.zfill(11) if "geoid" in election_df.columns else election_df["GEOID"].astype(str).str.zfill(11)
    if "GEOID" in election_df.columns and "geoid" not in election_df.columns:
        election_df = election_df.rename(columns={"GEOID": "geoid"})

    # Merge smoke with elections
    merged = election_df.merge(smoke_panel, on=["geoid", "year"], how="inner")
    print(f"  Matched smoke+elections: {len(merged):,} tract-year observations")

    if len(merged) == 0:
        print("  ERROR: No matches between smoke and election data!")
        return None

    # Merge controls
    if controls is not None:
        merged = merged.merge(controls, on=["geoid", "year"], how="left")
        n_ctrl = merged["unemployment_rate"].notna().sum() if "unemployment_rate" in merged.columns else 0
        print(f"  Controls matched: {n_ctrl:,}/{len(merged):,}")

    # Add identifiers
    merged["county_fips"] = merged["geoid"].str[:5]
    merged["state_year"] = "06_" + merged["year"].astype(str)
    merged["county_year"] = merged["county_fips"] + "_" + merged["year"].astype(str)

    # Lagged vote share (prior election)
    merged = merged.sort_values(["geoid", "year"])
    merged["dem_vote_share_lag"] = merged.groupby("geoid")["dem_vote_share"].shift(1)

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


def print_summary(df, label):
    """Print summary statistics for an analysis dataset."""
    print(f"\n{'='*60}")
    print(f"{label} SUMMARY")
    print(f"{'='*60}")
    print(f"  Observations: {len(df):,}")
    print(f"  Tracts: {df['geoid'].nunique():,}")
    print(f"  Counties: {df['county_fips'].nunique():,}")
    print(f"  Elections: {sorted(df['year'].unique())}")

    print(f"\n  Outcomes:")
    if "dem_vote_share" in df.columns:
        valid = df["dem_vote_share"].dropna()
        print(f"    DEM vote share: mean={valid.mean():.3f}, sd={valid.std():.3f}")
    if "incumbent_vote_share" in df.columns:
        valid = df["incumbent_vote_share"].dropna()
        print(f"    Incumbent vote share: mean={valid.mean():.3f}, sd={valid.std():.3f}")
    if "total_votes" in df.columns:
        print(f"    Total votes: mean={df['total_votes'].mean():,.0f}, "
              f"median={df['total_votes'].median():,.0f}")

    print(f"\n  Smoke exposure (30-day window):")
    for col in ["smoke_days_30d", "smoke_pm25_mean_30d", "smoke_frac_haze_30d"]:
        if col in df.columns:
            print(f"    {col}: mean={df[col].mean():.3f}, max={df[col].max():.1f}")

    print(f"\n  Smoke by year:")
    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr]
        s30 = "smoke_pm25_mean_30d"
        if s30 in yr_df.columns:
            print(f"    {yr}: n={len(yr_df):,}, mean PM2.5={yr_df[s30].mean():.3f}, "
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


def main():
    print("=" * 60)
    print("CA Step 5: Build Tract-Level Analysis Datasets")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load smoke data
    smoke_df = load_smoke_data()
    if smoke_df is None:
        return

    # Determine which elections have smoke coverage
    smoke_min_year = smoke_df["date"].dt.year.min()
    smoke_max_year = smoke_df["date"].dt.year.max()
    print(f"\nSmoke data covers {smoke_min_year}-{smoke_max_year}")

    eligible_elections = {
        yr: dt for yr, dt in ELECTION_DATES.items()
        if smoke_min_year <= yr <= smoke_max_year
    }
    print(f"Eligible elections: {sorted(eligible_elections.keys())}")

    # Get the full set of CA tracts (all tracts that ever appear in the smoke data)
    ca_tracts = smoke_df["geoid"].unique()
    print(f"Total CA tracts in smoke data: {len(ca_tracts):,}")

    # Compute smoke exposure for each election
    print("\nComputing smoke exposure measures...")
    print("  (Non-smoke days filled with 0; means use total calendar days)")
    smoke_exposures = []
    for yr, dt in sorted(eligible_elections.items()):
        print(f"\n  {yr} election (date: {dt})...")
        exposure = compute_smoke_exposure(smoke_df, yr, dt, all_tracts=ca_tracts)
        if len(exposure) > 0:
            print(f"    {len(exposure):,} tracts")
            print(f"    Mean smoke days (30d): {exposure['smoke_days_30d'].mean():.1f}")
            print(f"    Mean smoke PM2.5 (30d): {exposure['smoke_pm25_mean_30d'].mean():.3f}")
            smoke_exposures.append(exposure)

    smoke_panel = pd.concat(smoke_exposures, ignore_index=True)
    print(f"\nSmoke panel: {len(smoke_panel):,} tract-year rows")

    # Load controls
    controls = load_controls()
    if controls is not None:
        # Save controls panel
        controls.to_parquet(CONTROLS_OUT, index=False)
        print(f"  Saved controls panel: {CONTROLS_OUT}")

    # Build presidential dataset
    if os.path.exists(PRES_ELECTION_FILE):
        pres_elections = pd.read_csv(PRES_ELECTION_FILE, dtype={"GEOID": str, "geoid": str},
                                     low_memory=False)
        pres_df = build_analysis_dataset(smoke_panel, pres_elections, controls, "presidential")
        if pres_df is not None:
            pres_df.to_parquet(PRES_OUT, index=False)
            print(f"  Saved: {PRES_OUT}")
            print_summary(pres_df, "PRESIDENTIAL")
    else:
        print(f"\n  Presidential election file not found: {PRES_ELECTION_FILE}")
        print("  Run ca_build_crosswalk.py first.")

    # Build House dataset
    if os.path.exists(HOUSE_ELECTION_FILE):
        house_elections = pd.read_csv(HOUSE_ELECTION_FILE, dtype={"GEOID": str, "geoid": str},
                                      low_memory=False)
        house_df = build_analysis_dataset(smoke_panel, house_elections, controls, "house")
        if house_df is not None:
            house_df.to_parquet(HOUSE_OUT, index=False)
            print(f"  Saved: {HOUSE_OUT}")
            print_summary(house_df, "HOUSE")
    else:
        print(f"\n  House election file not found: {HOUSE_ELECTION_FILE}")
        print("  Run ca_build_crosswalk.py first.")

    print(f"\nStep 5 complete.")


if __name__ == "__main__":
    main()

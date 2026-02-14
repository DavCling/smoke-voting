#!/usr/bin/env python3
"""
Build analysis dataset merging smoke exposure with House election returns.

Aggregates daily smoke PM2.5 into pre-election windows at the county level,
maps counties to congressional districts using Census crosswalk files, and
merges with MEDSL House election returns.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_FILE = os.path.join(BASE_DIR, "data", "smoke", "smoke_pm25_county_daily.csv")
HOUSE_FILE = os.path.join(BASE_DIR, "data", "elections", "house_district.csv")
CROSSWALK_DIR = os.path.join(BASE_DIR, "data", "crosswalks")
OUT_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_house_analysis.parquet")

# Election dates: presidential + midterm years
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

# Incumbent party (president's party — midterm referendum framing)
INCUMBENT_PARTY = {
    2006: "REPUBLICAN",   # Bush admin
    2008: "REPUBLICAN",   # Bush admin
    2010: "DEMOCRAT",      # Obama admin
    2012: "DEMOCRAT",      # Obama admin
    2014: "DEMOCRAT",      # Obama admin
    2016: "DEMOCRAT",      # Obama admin
    2018: "REPUBLICAN",    # Trump admin
    2020: "REPUBLICAN",    # Trump admin
    2022: "DEMOCRAT",      # Biden admin
}

# Redistricting cutoffs:
# Before 2012: 2000 Census districts (108th Congress crosswalk)
# 2012-2020: 2010 Census districts (113th Congress crosswalk)
# 2022+: 2020 Census districts (118th Congress crosswalk)
REDISTRICTING_CUTOFF_2010 = 2012
REDISTRICTING_CUTOFF_2020 = 2022

# EPA thresholds for smoke PM2.5 (µg/m³)
EPA_USG_THRESHOLD = 35.5
EPA_UNHEALTHY = 55.5


def load_smoke_data():
    """Load and standardize smoke PM2.5 data."""
    print("Loading smoke data...")
    # v1.0 is tab-delimited; v2.0 is comma-delimited. Auto-detect.
    with open(SMOKE_FILE) as f:
        sep = "\t" if "\t" in f.readline() else ","
    df = pd.read_csv(SMOKE_FILE, sep=sep, dtype={"GEOID": str})
    df = df.rename(columns={"GEOID": "fips", "smokePM_pred": "smoke_pm25"})
    df["fips"] = df["fips"].str.zfill(5)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

    print(f"  {len(df):,} rows, {df['fips'].nunique():,} counties")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    return df[["fips", "date", "smoke_pm25"]]


def compute_smoke_exposure(smoke_df, election_year, election_date):
    """Compute smoke exposure measures for a single election (vectorized)."""
    edate = pd.Timestamp(election_date)

    windows = {
        "7d": (edate - timedelta(days=7), edate),
        "30d": (edate - timedelta(days=30), edate),
        "60d": (edate - timedelta(days=60), edate),
        "90d": (edate - timedelta(days=90), edate),
    }
    season_start = pd.Timestamp(f"{election_year}-06-01")
    windows["season"] = (season_start, edate)

    earliest = min(start for start, _ in windows.values())
    smoke_window = smoke_df[
        (smoke_df["date"] >= earliest) & (smoke_df["date"] <= edate)
    ].copy()

    if len(smoke_window) == 0:
        print(f"  WARNING: No smoke data for {election_year} election window")
        return pd.DataFrame()

    result_df = smoke_window[["fips"]].drop_duplicates().copy()
    result_df["year"] = election_year

    for label, (start, end) in windows.items():
        w = smoke_window[(smoke_window["date"] >= start) & (smoke_window["date"] <= end)]
        grp = w.groupby("fips")["smoke_pm25"]

        agg = grp.agg(
            smoke_days=lambda x: (x > 0).sum(),
            smoke_mean=lambda x: x.mean(),
            smoke_max=lambda x: x.max(),
            smoke_severe=lambda x: (x > EPA_USG_THRESHOLD).sum(),
            smoke_cumul=lambda x: x.sum(),
            smoke_days_gt1=lambda x: (x > 1).sum(),
            smoke_days_gt5=lambda x: (x > 5).sum(),
            smoke_days_gt12=lambda x: (x > 12).sum(),
        ).rename(columns={
            "smoke_days": f"smoke_days_{label}",
            "smoke_mean": f"smoke_pm25_mean_{label}",
            "smoke_max": f"smoke_pm25_max_{label}",
            "smoke_severe": f"smoke_days_severe_{label}",
            "smoke_cumul": f"smoke_pm25_cumul_{label}",
            "smoke_days_gt1": f"smoke_days_gt1_{label}",
            "smoke_days_gt5": f"smoke_days_gt5_{label}",
            "smoke_days_gt12": f"smoke_days_gt12_{label}",
        })

        result_df = result_df.merge(agg, on="fips", how="left")

    result_df = result_df.fillna(0)
    return result_df


def load_crosswalk(year, all_county_fips=None):
    """Load the correct era's county-to-CD crosswalk.

    Returns DataFrame with columns: fips (5-digit), district_id (e.g. '06-12'),
    afact (population weight, 0-1).

    At-large states (1 CD) are absent from the Census crosswalk files.
    If all_county_fips is provided, counties not found in the crosswalk are
    assigned to their state's at-large district (CD 00).
    """
    if year < REDISTRICTING_CUTOFF_2010:
        path = os.path.join(CROSSWALK_DIR, "county_cd108.csv")
        label = "108th Congress (2000 Census)"
    elif year < REDISTRICTING_CUTOFF_2020:
        path = os.path.join(CROSSWALK_DIR, "county_cd113.csv")
        label = "113th Congress (2010 Census)"
    else:
        path = os.path.join(CROSSWALK_DIR, "county_cd118.csv")
        label = "118th Congress (2020 Census)"

    if not os.path.exists(path):
        print(f"  WARNING: Crosswalk not found for {label}: {path}")
        return None

    # The Census file has a title line before the header
    # Detect: if first line doesn't look like CSV data, skip it
    with open(path, "r") as f:
        first_line = f.readline().strip()
    skiprows = 0 if "," in first_line and len(first_line.split(",")) >= 3 else 1

    df = pd.read_csv(path, skiprows=skiprows, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Map to standard column names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "state" or ("state" in cl and "county" not in cl):
            col_map["state"] = col
        elif "county" in cl:
            col_map["county"] = col
        elif "congress" in cl or "district" in cl or cl == "cd":
            col_map["cd"] = col

    df["state_fips"] = df[col_map["state"]].str.zfill(2)
    df["county_fips_3"] = df[col_map["county"]].str.zfill(3)
    df["fips"] = df["state_fips"] + df["county_fips_3"]
    df["cd"] = df[col_map["cd"]].str.zfill(2)
    df["district_id"] = df["state_fips"] + "-" + df["cd"]

    # Compute equal allocation factor for split counties
    county_counts = df.groupby("fips").size().rename("n_cds")
    df = df.merge(county_counts, on="fips")
    df["afact"] = 1.0 / df["n_cds"]

    result = df[["fips", "district_id", "afact"]].copy()

    # Add at-large states: counties not in the crosswalk get CD 00
    if all_county_fips is not None:
        missing_fips = set(all_county_fips) - set(result["fips"])
        if missing_fips:
            at_large_rows = []
            for fips in missing_fips:
                state_fips = fips[:2]
                district_id = state_fips + "-00"
                at_large_rows.append({"fips": fips, "district_id": district_id, "afact": 1.0})
            at_large_df = pd.DataFrame(at_large_rows)
            n_at_large_states = at_large_df["district_id"].nunique()
            print(f"  Added {len(at_large_df):,} at-large county entries "
                  f"({n_at_large_states} states)")
            result = pd.concat([result, at_large_df], ignore_index=True)

    n_split = (result.groupby("fips").size() > 1).sum()
    print(f"  Loaded {label}: {len(result):,} county-CD pairs "
          f"({result['fips'].nunique():,} counties, "
          f"{result['district_id'].nunique():,} districts, "
          f"{n_split} split-county entries)")

    return result


def aggregate_smoke_to_districts(county_smoke_df, crosswalk_df):
    """Aggregate county-level smoke to congressional districts using crosswalk weights.

    For each district, compute population-weighted average of county smoke variables.
    """
    # Merge county smoke with crosswalk
    merged = county_smoke_df.merge(crosswalk_df, on="fips", how="inner")

    # Identify smoke columns (all columns that start with "smoke_")
    smoke_cols = [c for c in county_smoke_df.columns if c.startswith("smoke_")]

    # Weighted aggregation: for each district, sum(smoke * afact) / sum(afact)
    for col in smoke_cols:
        merged[f"{col}_weighted"] = merged[col] * merged["afact"]

    agg_dict = {f"{col}_weighted": "sum" for col in smoke_cols}
    agg_dict["afact"] = "sum"
    agg_dict["year"] = "first"

    district_smoke = merged.groupby("district_id").agg(agg_dict).reset_index()

    # Normalize by total weight
    for col in smoke_cols:
        district_smoke[col] = district_smoke[f"{col}_weighted"] / district_smoke["afact"]
        district_smoke = district_smoke.drop(columns=[f"{col}_weighted"])

    district_smoke = district_smoke.drop(columns=["afact"])

    return district_smoke


def load_house_data():
    """Load and process MEDSL House district returns."""
    print("\nLoading House election data...")

    # Detect delimiter
    with open(HOUSE_FILE, "r") as f:
        header = f.readline()
    sep = "\t" if "\t" in header else ","

    df = pd.read_csv(HOUSE_FILE, sep=sep, dtype={"state_fips": str})
    df.columns = df.columns.str.lower().str.strip()

    # Identify party column
    if "party_detailed" in df.columns:
        party_col = "party_detailed"
    elif "party" in df.columns:
        party_col = "party"
    else:
        raise ValueError(f"No party column found. Columns: {list(df.columns)}")

    df["party_norm"] = df[party_col].str.upper().str.strip()

    # Filter to general elections, non-special
    if "stage" in df.columns:
        df = df[df["stage"].str.lower() == "gen"].copy()
    if "special" in df.columns:
        df = df[~df["special"].astype(bool)].copy()

    # Handle mode column (keep TOTAL where available, aggregate otherwise)
    if "mode" in df.columns:
        total_mask = df["mode"].str.upper() == "TOTAL"
        keys_with_total = set(
            df.loc[total_mask, ["state_fips", "district", "year"]].apply(tuple, axis=1)
        )
        df["_has_total"] = df[["state_fips", "district", "year"]].apply(tuple, axis=1).isin(keys_with_total)
        df = df[(df["_has_total"] & total_mask) | (~df["_has_total"])].copy()
        df = df.drop(columns=["_has_total"])

    # Create district_id
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)
    df["district"] = df["district"].astype(str).str.zfill(2)
    df["district_id"] = df["state_fips"] + "-" + df["district"]

    # Filter to relevant years
    df = df[df["year"].isin(ELECTION_DATES.keys())].copy()

    # Keep only DEM and REP
    df_party = df[df["party_norm"].isin(["DEMOCRAT", "REPUBLICAN"])].copy()

    # Aggregate votes by district-year-party
    agg = df_party.groupby(["district_id", "year", "state_fips", "party_norm"]).agg(
        candidatevotes=("candidatevotes", "sum"),
        totalvotes=("totalvotes", "first"),
    ).reset_index()

    # Pivot
    pivot = agg.pivot_table(
        index=["district_id", "year", "state_fips", "totalvotes"],
        columns="party_norm",
        values="candidatevotes",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None

    pivot = pivot.rename(columns={
        "DEMOCRAT": "dem_votes",
        "REPUBLICAN": "rep_votes",
        "totalvotes": "total_votes",
    })

    # Fill missing party votes with 0
    for col in ["dem_votes", "rep_votes"]:
        if col not in pivot.columns:
            pivot[col] = 0
        pivot[col] = pivot[col].fillna(0)

    # Flag uncontested races
    pivot["uncontested"] = (pivot["dem_votes"] == 0) | (pivot["rep_votes"] == 0)

    # Two-party vote share
    two_party = pivot["dem_votes"] + pivot["rep_votes"]
    pivot["dem_vote_share"] = pivot["dem_votes"] / two_party.replace(0, np.nan)

    # Incumbent vote share (president's party)
    pivot["incumbent_party"] = pivot["year"].map(INCUMBENT_PARTY)
    pivot["incumbent_vote_share"] = np.where(
        pivot["incumbent_party"] == "DEMOCRAT",
        pivot["dem_vote_share"],
        1 - pivot["dem_vote_share"],
    )

    # Lagged vote share (prior election cycle = 2 years ago)
    pivot = pivot.sort_values(["district_id", "year"])
    pivot["dem_vote_share_lag"] = pivot.groupby("district_id")["dem_vote_share"].shift(1)

    # Null out lagged vote share at redistricting boundaries
    # Districts changed definition at each redistricting, so lag is invalid
    pivot.loc[pivot["year"].isin([REDISTRICTING_CUTOFF_2010, REDISTRICTING_CUTOFF_2020]),
              "dem_vote_share_lag"] = np.nan

    print(f"  {len(pivot):,} district-year observations")
    print(f"  Years: {sorted(pivot['year'].unique())}")
    print(f"  Districts: {pivot['district_id'].nunique():,}")
    print(f"  Uncontested: {pivot['uncontested'].sum():,} ({pivot['uncontested'].mean():.1%})")

    return pivot


def main():
    print("=" * 60)
    print("Build House Smoke-Voting Analysis Dataset")
    print("=" * 60)

    smoke_df = load_smoke_data()
    house_df = load_house_data()

    # Determine which elections have smoke data
    smoke_min_year = smoke_df["date"].dt.year.min()
    smoke_max_year = smoke_df["date"].dt.year.max()
    print(f"\nSmoke data covers {smoke_min_year}-{smoke_max_year}")

    eligible_elections = {
        yr: dt for yr, dt in ELECTION_DATES.items()
        if smoke_min_year <= yr <= smoke_max_year
    }
    print(f"Eligible elections: {sorted(eligible_elections.keys())}")

    # Process each election year
    all_merged = []
    for yr, dt in sorted(eligible_elections.items()):
        print(f"\n{'='*40}")
        print(f"Processing {yr} election (date: {dt})...")

        # Compute county-level smoke exposure
        county_smoke = compute_smoke_exposure(smoke_df, yr, dt)
        if len(county_smoke) == 0:
            print(f"  Skipping {yr}: no smoke data")
            continue
        print(f"  County smoke: {len(county_smoke):,} counties")

        # Load crosswalk for this era (pass county FIPS to add at-large states)
        crosswalk = load_crosswalk(yr, all_county_fips=set(county_smoke["fips"]))
        if crosswalk is None:
            print(f"  Skipping {yr}: no crosswalk available")
            continue

        # Aggregate to districts
        district_smoke = aggregate_smoke_to_districts(county_smoke, crosswalk)
        print(f"  District smoke: {len(district_smoke):,} districts")

        # Get House data for this year
        house_yr = house_df[house_df["year"] == yr].copy()
        print(f"  House data: {len(house_yr):,} districts")

        # Merge
        merged = house_yr.merge(district_smoke, on=["district_id", "year"], how="inner")
        print(f"  Matched: {len(merged):,} districts")

        if len(merged) > 0:
            print(f"  Mean smoke days (60d): {merged['smoke_days_60d'].mean():.1f}")
            print(f"  Mean smoke PM2.5 (60d): {merged['smoke_pm25_mean_60d'].mean():.3f}")

        all_merged.append(merged)

    if not all_merged:
        print("\nERROR: No data was merged. Check crosswalk and data files.")
        sys.exit(1)

    panel = pd.concat(all_merged, ignore_index=True)

    # Add derived variables
    panel["state_year"] = panel["state_fips"] + "_" + panel["year"].astype(str)

    # Clean total_votes: set invalid values (<=0) to NaN
    panel.loc[panel["total_votes"] <= 0, "total_votes"] = np.nan
    panel["log_total_votes"] = np.log(panel["total_votes"])

    # Summary
    print("\n" + "=" * 60)
    print("HOUSE ANALYSIS DATASET SUMMARY")
    print("=" * 60)
    print(f"  Observations: {len(panel):,}")
    print(f"  Districts: {panel['district_id'].nunique():,}")
    print(f"  Elections: {sorted(panel['year'].unique())}")
    print(f"  Uncontested: {panel['uncontested'].sum():,}")
    print(f"\n  Outcome variables (contested only):")
    contested = panel[~panel["uncontested"]]
    print(f"    DEM vote share: mean={contested['dem_vote_share'].mean():.3f}, "
          f"sd={contested['dem_vote_share'].std():.3f}")
    print(f"    Incumbent vote share: mean={contested['incumbent_vote_share'].mean():.3f}, "
          f"sd={contested['incumbent_vote_share'].std():.3f}")
    print(f"    Total votes: mean={panel['total_votes'].mean():,.0f}, "
          f"median={panel['total_votes'].median():,.0f}")
    print(f"\n  Smoke exposure (60-day window):")
    print(f"    Smoke days: mean={panel['smoke_days_60d'].mean():.1f}")
    print(f"    Mean PM2.5: mean={panel['smoke_pm25_mean_60d'].mean():.3f}")
    print(f"\n  Smoke by year:")
    for yr in sorted(panel["year"].unique()):
        yr_df = panel[panel["year"] == yr]
        print(f"    {yr}: {len(yr_df):,} districts, "
              f"{yr_df['smoke_days_60d'].mean():.1f} smoke days, "
              f"{yr_df['smoke_pm25_mean_60d'].mean():.3f} mean PM2.5")

    # Save
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    panel.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")
    print(f"  Shape: {panel.shape}")
    print(f"  Columns: {list(panel.columns)}")

    print(f"\nBuild complete.")


if __name__ == "__main__":
    main()

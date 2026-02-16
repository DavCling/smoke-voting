#!/usr/bin/env python3
"""
Build county-level House analysis dataset from MEDSL precinct returns.

Aggregates precinct-level House election data (2016, 2018, 2020, 2022) to county level,
merges with daily smoke PM2.5 exposure, and outputs the analysis panel.

Counties that span multiple House districts will have votes from all races summed.
This is the natural county-level aggregation — measuring "how did House candidates
perform in this county" rather than tracking individual district races.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_FILE = os.path.join(BASE_DIR, "data", "smoke", "smoke_pm25_county_daily.csv")
PRECINCT_DIR = os.path.join(BASE_DIR, "data", "elections")
OUT_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_county_house_analysis.parquet")

PRECINCT_FILES = {
    2016: os.path.join(PRECINCT_DIR, "house_precinct_2016.csv"),
    2018: os.path.join(PRECINCT_DIR, "house_precinct_2018.csv"),
    2020: os.path.join(PRECINCT_DIR, "house_precinct_2020.csv"),
    2022: os.path.join(PRECINCT_DIR, "house_precinct_2022.csv"),
}

ELECTION_DATES = {
    2016: "2016-11-08",
    2018: "2018-11-06",
    2020: "2020-11-03",
    2022: "2022-11-08",
}

# Incumbent party (president's party)
INCUMBENT_PARTY = {
    2016: "DEMOCRAT",     # Obama admin
    2018: "REPUBLICAN",   # Trump admin
    2020: "REPUBLICAN",   # Trump admin
    2022: "DEMOCRAT",     # Biden admin
}

# EPA thresholds for smoke PM2.5 (µg/m³)
HAZE_THRESHOLD = 20.0      # Visible haze onset
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
            smoke_frac_haze=lambda x: (x > HAZE_THRESHOLD).sum() / len(x),
            smoke_frac_usg=lambda x: (x > EPA_USG_THRESHOLD).sum() / len(x),
            smoke_frac_unhealthy=lambda x: (x > EPA_UNHEALTHY).sum() / len(x),
        ).rename(columns={
            "smoke_days": f"smoke_days_{label}",
            "smoke_mean": f"smoke_pm25_mean_{label}",
            "smoke_max": f"smoke_pm25_max_{label}",
            "smoke_severe": f"smoke_days_severe_{label}",
            "smoke_cumul": f"smoke_pm25_cumul_{label}",
            "smoke_days_gt1": f"smoke_days_gt1_{label}",
            "smoke_days_gt5": f"smoke_days_gt5_{label}",
            "smoke_days_gt12": f"smoke_days_gt12_{label}",
            "smoke_frac_haze": f"smoke_frac_haze_{label}",
            "smoke_frac_usg": f"smoke_frac_usg_{label}",
            "smoke_frac_unhealthy": f"smoke_frac_unhealthy_{label}",
        })

        result_df = result_df.merge(agg, on="fips", how="left")

    result_df = result_df.fillna(0)
    return result_df


def load_precinct_year(year):
    """Load and aggregate precinct-level House data to county level for one year."""
    path = PRECINCT_FILES[year]
    if not os.path.exists(path):
        print(f"  WARNING: Missing precinct file: {path}")
        return None

    print(f"\n  Loading {year} precinct data from {os.path.basename(path)}...")
    df = pd.read_csv(path, dtype={"county_fips": str}, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    print(f"    Raw rows: {len(df):,}")

    # Filter to US HOUSE general elections, non-special
    if "office" in df.columns:
        df = df[df["office"].str.upper() == "US HOUSE"].copy()
    print(f"    After US HOUSE filter: {len(df):,}")

    if "stage" in df.columns:
        df = df[df["stage"].str.lower() == "gen"].copy()
        print(f"    After general election filter: {len(df):,}")

    if "special" in df.columns:
        df = df[~df["special"].astype(bool)].copy()
        print(f"    After dropping specials: {len(df):,}")

    # Handle mode column (keep TOTAL where available)
    if "mode" in df.columns:
        total_mask = df["mode"].str.upper() == "TOTAL"
        keys_with_total = set(
            df.loc[total_mask, ["county_fips", "year"]].apply(tuple, axis=1)
        )
        if keys_with_total:
            df["_has_total"] = df[["county_fips", "year"]].apply(tuple, axis=1).isin(keys_with_total)
            df = df[(df["_has_total"] & total_mask) | (~df["_has_total"])].copy()
            df = df.drop(columns=["_has_total"])
            print(f"    After mode handling: {len(df):,}")

    # Normalize party names
    party_col = "party_detailed" if "party_detailed" in df.columns else "party"
    df["party_norm"] = df[party_col].astype(str).str.upper().str.strip()

    # Identify vote column
    vote_col = "votes" if "votes" in df.columns else "candidatevotes"

    # Clean and zero-pad county FIPS
    # 2016 file has floats (e.g. "1001.0") while 2018/2020 have strings ("01097")
    df["county_fips"] = (df["county_fips"].astype(str)
                         .str.replace(r'\.0$', '', regex=True)
                         .str.zfill(5))

    # Drop rows without county FIPS or votes
    df = df.dropna(subset=["county_fips", vote_col])
    df = df[df["county_fips"] != "nan00"].copy()
    df[vote_col] = pd.to_numeric(df[vote_col], errors="coerce").fillna(0)

    # Keep only DEM and REP
    dem_mask = df["party_norm"].str.contains("DEMOCRAT")
    rep_mask = df["party_norm"].str.contains("REPUBLICAN")
    df_party = df[dem_mask | rep_mask].copy()
    df_party["party_clean"] = np.where(dem_mask[df_party.index], "DEMOCRAT", "REPUBLICAN")

    print(f"    DEM+REP rows: {len(df_party):,}")

    # Aggregate to county level: sum votes by county-party
    county_agg = df_party.groupby(["county_fips", "party_clean"])[vote_col].sum().reset_index()
    county_agg = county_agg.rename(columns={vote_col: "votes"})

    # Pivot
    pivot = county_agg.pivot_table(
        index="county_fips",
        columns="party_clean",
        values="votes",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None

    pivot = pivot.rename(columns={
        "county_fips": "fips",
        "DEMOCRAT": "dem_votes",
        "REPUBLICAN": "rep_votes",
    })

    for col in ["dem_votes", "rep_votes"]:
        if col not in pivot.columns:
            pivot[col] = 0
        pivot[col] = pivot[col].fillna(0)

    pivot["year"] = year
    pivot["total_votes"] = pivot["dem_votes"] + pivot["rep_votes"]

    # Flag uncontested counties (no votes for one party)
    pivot["uncontested"] = (pivot["dem_votes"] == 0) | (pivot["rep_votes"] == 0)

    # Two-party vote share
    two_party = pivot["dem_votes"] + pivot["rep_votes"]
    pivot["dem_vote_share"] = pivot["dem_votes"] / two_party.replace(0, np.nan)

    # Incumbent vote share
    pivot["incumbent_party"] = INCUMBENT_PARTY[year]
    pivot["incumbent_vote_share"] = np.where(
        pivot["incumbent_party"] == "DEMOCRAT",
        pivot["dem_vote_share"],
        1 - pivot["dem_vote_share"],
    )

    print(f"    Counties: {len(pivot):,}")
    print(f"    Uncontested: {pivot['uncontested'].sum():,} ({pivot['uncontested'].mean():.1%})")
    print(f"    Mean DEM share: {pivot['dem_vote_share'].mean():.3f}")

    return pivot


def load_all_precinct_data():
    """Load and combine precinct data across all years."""
    print("\nLoading precinct House election data...")

    all_years = []
    for year in sorted(PRECINCT_FILES.keys()):
        year_df = load_precinct_year(year)
        if year_df is not None:
            all_years.append(year_df)

    if not all_years:
        print("ERROR: No precinct data loaded.")
        sys.exit(1)

    combined = pd.concat(all_years, ignore_index=True)

    # Compute lagged vote share (shift by 1 election within county)
    # Elections are every 2 years: 2016 → 2018 → 2020 → 2022
    combined = combined.sort_values(["fips", "year"])
    combined["dem_vote_share_lag"] = combined.groupby("fips")["dem_vote_share"].shift(1)

    print(f"\n  Combined: {len(combined):,} county-year observations")
    print(f"  Counties: {combined['fips'].nunique():,}")
    print(f"  Years: {sorted(combined['year'].unique())}")

    return combined


def main():
    print("=" * 60)
    print("Build County-Level House Smoke-Voting Analysis Dataset")
    print("=" * 60)

    smoke_df = load_smoke_data()
    election_df = load_all_precinct_data()

    # Determine which elections have smoke data
    smoke_min_year = smoke_df["date"].dt.year.min()
    smoke_max_year = smoke_df["date"].dt.year.max()
    print(f"\nSmoke data covers {smoke_min_year}-{smoke_max_year}")

    eligible_elections = {
        yr: dt for yr, dt in ELECTION_DATES.items()
        if smoke_min_year <= yr <= smoke_max_year
    }
    print(f"Eligible elections: {sorted(eligible_elections.keys())}")

    # Compute smoke exposure for each election
    print("\nComputing smoke exposure measures...")
    smoke_exposures = []
    for yr, dt in sorted(eligible_elections.items()):
        print(f"\n  {yr} election (date: {dt})...")
        exposure = compute_smoke_exposure(smoke_df, yr, dt)
        if len(exposure) > 0:
            print(f"    {len(exposure):,} counties")
            print(f"    Mean smoke days (60d): {exposure['smoke_days_60d'].mean():.1f}")
            print(f"    Mean smoke PM2.5 (60d): {exposure['smoke_pm25_mean_60d'].mean():.3f}")
            smoke_exposures.append(exposure)

    smoke_panel = pd.concat(smoke_exposures, ignore_index=True)
    print(f"\nSmoke panel: {len(smoke_panel):,} county-year rows")

    # Merge smoke exposure with election data
    print("\nMerging smoke exposure with election returns...")
    merged = election_df.merge(smoke_panel, on=["fips", "year"], how="inner")
    print(f"  Matched: {len(merged):,} county-year observations")
    print(f"  Unmatched election rows: "
          f"{len(election_df[election_df['year'].isin(eligible_elections)]) - len(merged):,}")

    # Merge controls panel (if available)
    controls_file = os.path.join(BASE_DIR, "output", "controls_panel.parquet")
    if os.path.exists(controls_file):
        controls = pd.read_parquet(controls_file)
        merged = merged.merge(controls, on=["fips", "year"], how="left")
        n_matched = merged["unemployment_rate"].notna().sum()
        print(f"  Controls: {n_matched:,}/{len(merged):,} matched")
    else:
        print("  Controls panel not found — skipping")

    # Add derived variables
    merged["state_fips"] = merged["fips"].str[:2]
    merged["state_year"] = merged["state_fips"] + "_" + merged["year"].astype(str)
    merged.loc[merged["total_votes"] <= 0, "total_votes"] = np.nan
    merged["log_total_votes"] = np.log(merged["total_votes"])

    # Turnout rate = total votes / voting-age population
    if "voting_age_population" in merged.columns:
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

    # Summary
    print("\n" + "=" * 60)
    print("COUNTY-LEVEL HOUSE ANALYSIS DATASET SUMMARY")
    print("=" * 60)
    print(f"  Observations: {len(merged):,}")
    print(f"  Counties: {merged['fips'].nunique():,}")
    print(f"  Elections: {sorted(merged['year'].unique())}")
    print(f"  Uncontested: {merged['uncontested'].sum():,} ({merged['uncontested'].mean():.1%})")
    print(f"\n  Outcome variables (contested only):")
    contested = merged[~merged["uncontested"]]
    print(f"    DEM vote share: mean={contested['dem_vote_share'].mean():.3f}, "
          f"sd={contested['dem_vote_share'].std():.3f}")
    print(f"    Incumbent vote share: mean={contested['incumbent_vote_share'].mean():.3f}, "
          f"sd={contested['incumbent_vote_share'].std():.3f}")
    print(f"    Total votes: mean={merged['total_votes'].mean():,.0f}, "
          f"median={merged['total_votes'].median():,.0f}")
    print(f"\n  Smoke exposure (60-day window):")
    print(f"    Smoke days: mean={merged['smoke_days_60d'].mean():.1f}")
    print(f"    Mean PM2.5: mean={merged['smoke_pm25_mean_60d'].mean():.3f}")
    print(f"\n  By year:")
    for yr in sorted(merged["year"].unique()):
        yr_df = merged[merged["year"] == yr]
        print(f"    {yr}: {len(yr_df):,} counties, "
              f"{yr_df['smoke_days_60d'].mean():.1f} smoke days, "
              f"{yr_df['smoke_pm25_mean_60d'].mean():.3f} mean PM2.5")

    # Save
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    merged.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved: {OUT_FILE}")
    print(f"  Shape: {merged.shape}")
    print(f"  Columns: {list(merged.columns)}")

    print(f"\nBuild complete.")


if __name__ == "__main__":
    main()

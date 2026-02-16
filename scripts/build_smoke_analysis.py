#!/usr/bin/env python3
"""
Phase 3: Build analysis dataset merging smoke exposure with election returns.

Aggregates daily smoke PM2.5 into pre-election windows, merges with county-level
presidential vote shares, and outputs the analysis panel.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_FILE = os.path.join(BASE_DIR, "data", "smoke", "smoke_pm25_county_daily.csv")
ELECTION_FILE = os.path.join(BASE_DIR, "data", "elections", "county_presidential.csv")
OUT_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_analysis.parquet")

# Election dates (first Tuesday after first Monday in November)
ELECTION_DATES = {
    2000: "2000-11-07",
    2004: "2004-11-02",
    2008: "2008-11-04",
    2012: "2012-11-06",
    2016: "2016-11-08",
    2020: "2020-11-03",
    2024: "2024-11-05",
}

# Incumbent party for each election
INCUMBENT_PARTY = {
    2000: "DEMOCRAT",     # Clinton/Gore admin
    2004: "REPUBLICAN",   # Bush admin
    2008: "REPUBLICAN",   # Bush admin
    2012: "DEMOCRAT",     # Obama admin
    2016: "DEMOCRAT",     # Obama admin
    2020: "REPUBLICAN",   # Trump admin
    2024: "DEMOCRAT",     # Biden admin
}

# EPA thresholds for smoke PM2.5 (µg/m³)
HAZE_THRESHOLD = 20.0      # Visible haze onset (Burke et al. 2021; O'Neill et al. 2013)
EPA_USG_THRESHOLD = 35.5   # "Unhealthy for Sensitive Groups"
EPA_UNHEALTHY = 55.5       # "Unhealthy"


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
    print(f"  Mean smoke PM2.5: {df['smoke_pm25'].mean():.4f} µg/m³")

    return df[["fips", "date", "smoke_pm25"]]


def load_election_data():
    """Load and process MEDSL county presidential returns."""
    print("\nLoading election data...")
    df = pd.read_csv(ELECTION_FILE, dtype={"county_fips": str})
    df["county_fips"] = df["county_fips"].str.zfill(5)

    # Handle 2020 mode column: some counties have TOTAL, others have
    # broken-out modes (ABSENTEE, ELECTION DAY, etc.). For counties with
    # TOTAL, keep that; for others, we'll aggregate by summing candidatevotes.
    if "mode" in df.columns:
        # Identify county-year-party groups that have a TOTAL row
        total_mask = df["mode"] == "TOTAL"
        keys_with_total = set(
            df.loc[total_mask, ["county_fips", "year"]].apply(tuple, axis=1)
        )
        df["_has_total"] = df[["county_fips", "year"]].apply(tuple, axis=1).isin(keys_with_total)
        # Keep TOTAL rows for counties that have them; keep all rows for others
        df = df[(df["_has_total"] & total_mask) | (~df["_has_total"])].copy()
        df = df.drop(columns=["_has_total"])

    # Keep only DEM and REP rows
    df_party = df[df["party"].isin(["DEMOCRAT", "REPUBLICAN"])].copy()

    # Aggregate: sum candidatevotes by county-year-party (handles broken-out modes)
    agg = df_party.groupby(["county_fips", "year", "state_po", "county_name", "party"]).agg(
        candidatevotes=("candidatevotes", "sum"),
        totalvotes=("totalvotes", "first"),  # totalvotes is same across parties
    ).reset_index()

    # Pivot to get DEM and REP votes per county-year
    pivot = agg.pivot_table(
        index=["county_fips", "year", "state_po", "county_name", "totalvotes"],
        columns="party",
        values="candidatevotes",
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None

    pivot = pivot.rename(columns={
        "county_fips": "fips",
        "DEMOCRAT": "dem_votes",
        "REPUBLICAN": "rep_votes",
        "totalvotes": "total_votes",
    })

    # Compute two-party vote share
    two_party = pivot["dem_votes"].fillna(0) + pivot["rep_votes"].fillna(0)
    pivot["dem_vote_share"] = pivot["dem_votes"].fillna(0) / two_party.replace(0, np.nan)

    # Incumbent vote share
    pivot["incumbent_party"] = pivot["year"].map(INCUMBENT_PARTY)
    pivot["incumbent_vote_share"] = np.where(
        pivot["incumbent_party"] == "DEMOCRAT",
        pivot["dem_vote_share"],
        1 - pivot["dem_vote_share"],
    )

    # Lagged vote share (prior election)
    pivot = pivot.sort_values(["fips", "year"])
    pivot["dem_vote_share_lag"] = pivot.groupby("fips")["dem_vote_share"].shift(1)

    print(f"  {len(pivot):,} county-year observations")
    print(f"  Years: {sorted(pivot['year'].unique())}")
    print(f"  Counties: {pivot['fips'].nunique():,}")

    return pivot


def compute_smoke_exposure(smoke_df, election_year, election_date):
    """Compute smoke exposure measures for a single election (vectorized)."""
    edate = pd.Timestamp(election_date)

    # Define windows
    windows = {
        "7d": (edate - timedelta(days=7), edate),
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

    if len(smoke_window) == 0:
        print(f"  WARNING: No smoke data for {election_year} election window")
        return pd.DataFrame()

    # Vectorized aggregation per window
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
            smoke_frac_haze=lambda x: (x > HAZE_THRESHOLD).sum() / len(x),
            smoke_frac_usg=lambda x: (x > EPA_USG_THRESHOLD).sum() / len(x),
            smoke_frac_unhealthy=lambda x: (x > EPA_UNHEALTHY).sum() / len(x),
        ).rename(columns={
            "smoke_days": f"smoke_days_{label}",
            "smoke_mean": f"smoke_pm25_mean_{label}",
            "smoke_max": f"smoke_pm25_max_{label}",
            "smoke_severe": f"smoke_days_severe_{label}",
            "smoke_cumul": f"smoke_pm25_cumul_{label}",
            "smoke_frac_haze": f"smoke_frac_haze_{label}",
            "smoke_frac_usg": f"smoke_frac_usg_{label}",
            "smoke_frac_unhealthy": f"smoke_frac_unhealthy_{label}",
        })

        result_df = result_df.merge(agg, on="fips", how="left")

    result_df = result_df.fillna(0)
    return result_df


def main():
    print("=" * 60)
    print("Phase 3: Build Smoke-Voting Analysis Dataset")
    print("=" * 60)

    smoke_df = load_smoke_data()
    election_df = load_election_data()

    # Determine which elections have smoke data coverage
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
    print(f"  Unmatched election rows: {len(election_df[election_df['year'].isin(eligible_elections)]) - len(merged):,}")

    # Merge controls panel (if available)
    controls_file = os.path.join(BASE_DIR, "output", "controls_panel.parquet")
    if os.path.exists(controls_file):
        controls = pd.read_parquet(controls_file)
        merged = merged.merge(controls, on=["fips", "year"], how="left")
        n_matched = merged["unemployment_rate"].notna().sum()
        print(f"  Controls: {n_matched:,}/{len(merged):,} matched")
    else:
        print("  Controls panel not found — skipping")

    # Add state FIPS for fixed effects
    merged["state_fips"] = merged["fips"].str[:2]
    merged["state_year"] = merged["state_fips"] + "_" + merged["year"].astype(str)

    # Add log population proxy (total votes as proxy — imperfect but available)
    merged["log_total_votes"] = np.log1p(merged["total_votes"])

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

    # Add urban/rural proxy: counties with high total votes are more urban
    # (Can be refined with Census data later)
    vote_median = merged.groupby("year")["total_votes"].transform("median")
    merged["urban_proxy"] = (merged["total_votes"] > vote_median).astype(int)

    # Add prior partisanship terciles
    merged["dem_tercile"] = merged.groupby("year")["dem_vote_share_lag"].transform(
        lambda x: pd.qcut(x, 3, labels=["R-leaning", "Swing", "D-leaning"], duplicates="drop")
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("ANALYSIS DATASET SUMMARY")
    print("=" * 60)
    print(f"  Observations: {len(merged):,}")
    print(f"  Counties: {merged['fips'].nunique():,}")
    print(f"  Elections: {sorted(merged['year'].unique())}")
    print(f"\n  Outcome variables:")
    print(f"    DEM vote share: mean={merged['dem_vote_share'].mean():.3f}, sd={merged['dem_vote_share'].std():.3f}")
    print(f"    Incumbent vote share: mean={merged['incumbent_vote_share'].mean():.3f}, sd={merged['incumbent_vote_share'].std():.3f}")
    print(f"    Total votes: mean={merged['total_votes'].mean():,.0f}, median={merged['total_votes'].median():,.0f}")
    print(f"\n  Smoke exposure (60-day window):")
    print(f"    Smoke days: mean={merged['smoke_days_60d'].mean():.1f}, max={merged['smoke_days_60d'].max()}")
    print(f"    Mean PM2.5: mean={merged['smoke_pm25_mean_60d'].mean():.3f}, max={merged['smoke_pm25_mean_60d'].max():.1f}")
    print(f"    Severe days: mean={merged['smoke_days_severe_60d'].mean():.2f}, max={merged['smoke_days_severe_60d'].max()}")
    print(f"\n  Smoke exposure by year:")
    for yr in sorted(merged["year"].unique()):
        yr_df = merged[merged["year"] == yr]
        print(f"    {yr}: {yr_df['smoke_days_60d'].mean():.1f} smoke days, "
              f"{yr_df['smoke_pm25_mean_60d'].mean():.3f} mean PM2.5, "
              f"{yr_df['smoke_days_severe_60d'].mean():.2f} severe days")

    # Save
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    merged.to_parquet(OUT_FILE, index=False)
    print(f"\nSaved analysis dataset: {OUT_FILE}")
    print(f"  Shape: {merged.shape}")
    print(f"  Columns: {list(merged.columns)}")

    print(f"\nPhase 3 complete.")


if __name__ == "__main__":
    main()

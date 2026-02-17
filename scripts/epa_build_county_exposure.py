#!/usr/bin/env python3
"""
Build county-level EPA PM2.5 exposure panel from AQS daily monitor data.

Approach:
  1. Load EPA daily PM2.5 CSVs (parameter 88101, 24-hour samples)
  2. Build county FIPS, take daily mean across monitors within county
  3. Aggregate into pre-election windows (7d, 30d, 60d, 90d, season)
  4. Apply 50% minimum coverage threshold (EPA FRM monitors sample every
     3rd or 6th day, not daily)

Outputs:
  - data/epa/epa_county_daily.csv        — county-day PM2.5
  - output/epa_pm25_exposure.parquet      — county-year panel with windowed variables
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPA_DIR = os.path.join(BASE_DIR, "data", "epa")
DAILY_OUT = os.path.join(BASE_DIR, "data", "epa", "epa_county_daily.csv")
PANEL_OUT = os.path.join(BASE_DIR, "output", "epa_pm25_exposure.parquet")

# Election dates (must match build_smoke_analysis.py)
ELECTION_DATES = {
    2008: "2008-11-04",
    2012: "2012-11-06",
    2016: "2016-11-08",
    2020: "2020-11-03",
}

# PM2.5 thresholds (matching Childs variable structure)
HAZE_THRESHOLD = 20.0       # Visible haze onset
EPA_USG_THRESHOLD = 35.5    # "Unhealthy for Sensitive Groups"
EPA_UNHEALTHY = 55.5        # "Unhealthy"

# Minimum coverage: fraction of window days with readings required
MIN_COVERAGE = 0.50


def load_epa_year(year):
    """Load and process one year of EPA daily PM2.5 data to county-day level."""
    csv_path = os.path.join(EPA_DIR, f"daily_88101_{year}.csv")
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found — skipping {year}")
        return None

    print(f"  Loading EPA {year}...")
    df = pd.read_csv(csv_path, dtype={"State Code": str, "County Code": str})

    # Filter to 24-HOUR standard daily averages only
    df = df[df["Sample Duration"] == "24 HOUR"].copy()
    print(f"    24-HOUR samples: {len(df):,}")

    # Build county FIPS
    df["fips"] = df["State Code"].str.zfill(2) + df["County Code"].str.zfill(3)

    # Filter to states with FIPS <= 56 (exclude territories)
    state_fips = df["fips"].str[:2].astype(int)
    df = df[state_fips <= 56].copy()
    print(f"    After state filter (FIPS ≤ 56): {len(df):,}")

    # Parse date
    df["date"] = pd.to_datetime(df["Date Local"])

    # Filter to election window: broadest = Jun 1 to Election Day
    edate = pd.Timestamp(ELECTION_DATES[year])
    season_start = pd.Timestamp(f"{year}-06-01")
    df = df[(df["date"] >= season_start) & (df["date"] <= edate)].copy()
    print(f"    In election window ({season_start.date()} to {edate.date()}): {len(df):,}")

    if len(df) == 0:
        return None

    # Aggregate: mean PM2.5 across monitors within county-day
    # Multiple monitors per county → take daily mean of Arithmetic Mean
    county_day = df.groupby(["fips", "date"]).agg(
        epa_pm25=("Arithmetic Mean", "mean"),
        n_monitors=("Arithmetic Mean", "count"),
    ).reset_index()

    print(f"    County-day records: {len(county_day):,}")
    print(f"    Counties: {county_day['fips'].nunique():,}")
    print(f"    Mean monitors per county-day: {county_day['n_monitors'].mean():.1f}")
    print(f"    Mean PM2.5: {county_day['epa_pm25'].mean():.2f} µg/m³")

    return county_day


def aggregate_windows(daily_df, year):
    """Aggregate county-day EPA PM2.5 into pre-election windows.

    Key difference from Childs: EPA monitors don't sample every day.
    We compute fractions relative to monitored days and enforce a
    minimum coverage threshold.
    """
    edate = pd.Timestamp(ELECTION_DATES[year])

    windows = {
        "7d": (edate - timedelta(days=7), edate),
        "30d": (edate - timedelta(days=30), edate),
        "60d": (edate - timedelta(days=60), edate),
        "90d": (edate - timedelta(days=90), edate),
        "season": (pd.Timestamp(f"{year}-06-01"), edate),
    }

    # Start with unique counties
    result_df = daily_df[["fips"]].drop_duplicates().copy()
    result_df["year"] = year

    for label, (start, end) in windows.items():
        n_calendar_days = (end - start).days + 1
        w = daily_df[(daily_df["date"] >= start) & (daily_df["date"] <= end)]

        grp = w.groupby("fips")

        agg = grp.agg(
            epa_pm25_mean=("epa_pm25", "mean"),
            epa_days_monitored=("epa_pm25", "count"),
            epa_frac_above20=("epa_pm25", lambda x: (x > HAZE_THRESHOLD).sum() / len(x)),
            epa_frac_above35=("epa_pm25", lambda x: (x > EPA_USG_THRESHOLD).sum() / len(x)),
            epa_frac_above55=("epa_pm25", lambda x: (x > EPA_UNHEALTHY).sum() / len(x)),
        )

        # Compute coverage
        agg["epa_coverage"] = agg["epa_days_monitored"] / n_calendar_days

        # Apply coverage threshold: set to NaN if < 50% of window days monitored
        low_coverage = agg["epa_coverage"] < MIN_COVERAGE
        n_dropped = low_coverage.sum()
        if n_dropped > 0:
            agg.loc[low_coverage, ["epa_pm25_mean", "epa_frac_above20",
                                    "epa_frac_above35", "epa_frac_above55"]] = np.nan

        # Rename with window suffix
        agg = agg.rename(columns={
            "epa_pm25_mean": f"epa_pm25_mean_{label}",
            "epa_days_monitored": f"epa_days_monitored_{label}",
            "epa_coverage": f"epa_coverage_{label}",
            "epa_frac_above20": f"epa_frac_above20_{label}",
            "epa_frac_above35": f"epa_frac_above35_{label}",
            "epa_frac_above55": f"epa_frac_above55_{label}",
        })

        result_df = result_df.merge(agg, on="fips", how="left")

        # Report
        valid = result_df[f"epa_pm25_mean_{label}"].notna().sum()
        total = len(result_df)
        print(f"    {label}: {valid}/{total} counties pass {MIN_COVERAGE:.0%} coverage "
              f"(dropped {n_dropped})")

    return result_df


def main():
    print("=" * 60)
    print("Build EPA County-Level PM2.5 Exposure Panel")
    print("=" * 60)

    # Build county-day exposure for each election year
    all_daily = []
    all_panels = []

    for year in sorted(ELECTION_DATES.keys()):
        print(f"\n--- {year} election ---")
        county_day = load_epa_year(year)
        if county_day is None:
            continue

        all_daily.append(county_day)

        panel = aggregate_windows(county_day, year)
        all_panels.append(panel)

    if not all_daily:
        print("ERROR: No data loaded. Run epa_download_pm25_data.py first.")
        sys.exit(1)

    # Save daily
    print(f"\n--- Saving daily exposure ---")
    daily_all = pd.concat(all_daily, ignore_index=True)
    os.makedirs(os.path.dirname(DAILY_OUT), exist_ok=True)
    daily_all.to_csv(DAILY_OUT, index=False)
    print(f"  Saved: {DAILY_OUT}")
    print(f"  {len(daily_all):,} rows, {daily_all['fips'].nunique():,} counties")

    # Save panel
    print(f"\n--- Saving panel ---")
    panel_all = pd.concat(all_panels, ignore_index=True)

    # Add state_fips and state_year for FE
    panel_all["state_fips"] = panel_all["fips"].str[:2]
    panel_all["state_year"] = panel_all["state_fips"] + "_" + panel_all["year"].astype(str)

    os.makedirs(os.path.dirname(PANEL_OUT), exist_ok=True)
    panel_all.to_parquet(PANEL_OUT, index=False)
    print(f"  Saved: {PANEL_OUT}")
    print(f"  {len(panel_all):,} county-year rows, {panel_all['fips'].nunique():,} counties")

    # Summary
    print(f"\n" + "=" * 60)
    print("EPA PM2.5 EXPOSURE SUMMARY")
    print("=" * 60)
    for year in sorted(ELECTION_DATES.keys()):
        yr_df = panel_all[panel_all["year"] == year]
        if len(yr_df) == 0:
            continue
        print(f"\n  {year}:")
        print(f"    Counties: {len(yr_df):,}")
        for w in ["30d", "season"]:
            mean_col = f"epa_pm25_mean_{w}"
            cov_col = f"epa_coverage_{w}"
            if mean_col in yr_df.columns:
                valid = yr_df[mean_col].notna().sum()
                mean_val = yr_df[mean_col].mean()
                mean_cov = yr_df[cov_col].mean() if cov_col in yr_df.columns else np.nan
                print(f"    {w}: {valid} counties valid, "
                      f"mean PM2.5 = {mean_val:.2f} µg/m³, "
                      f"mean coverage = {mean_cov:.1%}")

    print("\nBuild complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract smoke exposure windows from the national tract-level smoke file
for 2016 and 2020 presidential elections.

Reads the 2.2 GB national smoke file in chunks, filtering to date ranges
needed for election-window computation. Then computes 8 exposure measures
across 8 time windows for each tract x election year.

CRITICAL: The smoke file contains ONLY smoke days. Non-smoke days have
smokePM_pred = 0 and are omitted from the file. All means and fractions
use total calendar days in the window as denominator, not row count.

Input:
  data/smoke/tract/smokePM2pt5_predictions_daily_tract_20060101-20231231.csv

Output:
  data/national_tracts/smoke/tract_smoke_exposure.parquet
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE_FILE = os.path.join(BASE_DIR, "data", "smoke", "tract",
                          "smokePM2pt5_predictions_daily_tract_20060101-20231231.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "national_tracts", "smoke")
OUT_FILE = os.path.join(OUT_DIR, "tract_smoke_exposure.parquet")

# CONUS state FIPS codes
CONUS_FIPS = {
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
}

# Election dates
ELECTION_DATES = {
    2016: "2016-11-08",
    2020: "2020-11-03",
}

# EPA thresholds for smoke PM2.5 (ug/m3)
HAZE_THRESHOLD = 20.0
EPA_USG_THRESHOLD = 35.5
EPA_UNHEALTHY = 55.5


def compute_smoke_exposure(smoke_df, election_year, election_date, all_tracts):
    """Compute smoke exposure measures for a single election.

    The source data contains ONLY smoke days (non-smoke days = 0, omitted).
    Uses total calendar days in each window as the denominator for means
    and fractions. All tracts get output rows (0 for no smoke).
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

    result_df = pd.DataFrame({"geoid": all_tracts})
    result_df["year"] = election_year

    for label, (start, end) in windows.items():
        n_days = (end - start).days

        w = smoke_window[(smoke_window["date"] >= start) & (smoke_window["date"] <= end)]
        grp = w.groupby("geoid")["smoke_pm25"]

        agg = grp.agg(
            smoke_days=lambda x: (x > 0).sum(),
            smoke_sum=lambda x: x.sum(),
            smoke_max=lambda x: x.max(),
            smoke_severe=lambda x: (x > EPA_USG_THRESHOLD).sum(),
            smoke_n_haze=lambda x: (x > HAZE_THRESHOLD).sum(),
            smoke_n_usg=lambda x: (x > EPA_USG_THRESHOLD).sum(),
            smoke_n_unhealthy=lambda x: (x > EPA_UNHEALTHY).sum(),
        )

        agg[f"smoke_pm25_mean_{label}"] = agg["smoke_sum"] / n_days
        agg[f"smoke_days_{label}"] = agg["smoke_days"]
        agg[f"smoke_pm25_max_{label}"] = agg["smoke_max"]
        agg[f"smoke_days_severe_{label}"] = agg["smoke_severe"]
        agg[f"smoke_pm25_cumul_{label}"] = agg["smoke_sum"]
        agg[f"smoke_frac_haze_{label}"] = agg["smoke_n_haze"] / n_days
        agg[f"smoke_frac_usg_{label}"] = agg["smoke_n_usg"] / n_days
        agg[f"smoke_frac_unhealthy_{label}"] = agg["smoke_n_unhealthy"] / n_days

        keep_cols = [c for c in agg.columns if label in c]
        agg = agg[keep_cols].reset_index()

        result_df = result_df.merge(agg, on="geoid", how="left")

    result_df = result_df.fillna(0)
    return result_df


def main():
    print("=" * 60)
    print("National Tract-Level: Extract Smoke Exposure")
    print("=" * 60)

    if os.path.exists(OUT_FILE):
        df = pd.read_parquet(OUT_FILE)
        print(f"Already exists: {OUT_FILE}")
        print(f"  {len(df):,} rows, {df['geoid'].nunique():,} tracts")
        return

    if not os.path.exists(SMOKE_FILE):
        print(f"ERROR: Smoke file not found: {SMOKE_FILE}")
        print("  Run ca_download_smoke_data.py or download manually.")
        return

    # Determine date ranges needed
    # Broadest: Jun 1 of election year to election date
    date_ranges = []
    for year, edate_str in ELECTION_DATES.items():
        start = pd.Timestamp(f"{year}-06-01")
        end = pd.Timestamp(edate_str)
        date_ranges.append((start, end))

    min_date = min(s for s, _ in date_ranges)
    max_date = max(e for _, e in date_ranges)
    print(f"\n  Date range needed: {min_date.date()} to {max_date.date()}")

    # Read the large CSV in chunks, filtering to relevant date ranges
    print(f"\n  Reading {SMOKE_FILE}...")
    print("  (filtering to 2016+2020 election windows)")

    chunks = []
    n_read = 0
    n_kept = 0

    for chunk in pd.read_csv(SMOKE_FILE, chunksize=2_000_000,
                             dtype={"GEOID": str}):
        n_read += len(chunk)

        # Parse dates
        chunk["date"] = pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d")

        # Filter to relevant date ranges
        mask = pd.Series(False, index=chunk.index)
        for start, end in date_ranges:
            mask |= (chunk["date"] >= start) & (chunk["date"] <= end)

        filtered = chunk[mask].copy()

        # Filter to CONUS tracts
        filtered["state_fips"] = filtered["GEOID"].str.zfill(11).str[:2]
        filtered = filtered[filtered["state_fips"].isin(CONUS_FIPS)]
        filtered = filtered.drop(columns=["state_fips"])

        if len(filtered) > 0:
            chunks.append(filtered)
            n_kept += len(filtered)

        if n_read % 10_000_000 == 0:
            print(f"    Read {n_read / 1e6:.0f}M rows, kept {n_kept:,}")

    print(f"  Total read: {n_read:,}, kept: {n_kept:,}")

    if not chunks:
        print("  ERROR: No smoke data within election windows!")
        return

    smoke_df = pd.concat(chunks, ignore_index=True)
    smoke_df = smoke_df.rename(columns={"GEOID": "geoid", "smokePM_pred": "smoke_pm25"})
    smoke_df["geoid"] = smoke_df["geoid"].str.zfill(11)

    print(f"  Filtered smoke data: {len(smoke_df):,} rows, "
          f"{smoke_df['geoid'].nunique():,} tracts")

    # Get all unique CONUS tract GEOIDs
    all_tracts = smoke_df["geoid"].unique()
    print(f"  All CONUS tracts with smoke data: {len(all_tracts):,}")

    # Compute smoke exposure for each election
    print("\n  Computing smoke exposure measures...")
    exposures = []
    for year, edate in sorted(ELECTION_DATES.items()):
        print(f"\n  {year} (election date: {edate})...")
        exposure = compute_smoke_exposure(smoke_df, year, edate, all_tracts)
        print(f"    {len(exposure):,} tracts")
        print(f"    Mean smoke days (30d): {exposure['smoke_days_30d'].mean():.2f}")
        print(f"    Mean smoke PM2.5 (30d): {exposure['smoke_pm25_mean_30d'].mean():.4f}")
        print(f"    Frac haze nonzero (30d): "
              f"{(exposure['smoke_frac_haze_30d'] > 0).sum():,}")
        exposures.append(exposure)

    result = pd.concat(exposures, ignore_index=True)
    print(f"\n  Combined: {len(result):,} tract-year rows")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    result.to_parquet(OUT_FILE, index=False)
    print(f"  Saved: {OUT_FILE}")

    # Per-state summary
    result["state_fips"] = result["geoid"].str[:2]
    state_summary = result.groupby("state_fips").agg(
        n_tracts=("geoid", "nunique"),
        mean_smoke_30d=("smoke_pm25_mean_30d", "mean"),
    )
    print("\n  Tracts per state:")
    for fips, row in state_summary.iterrows():
        print(f"    {fips}: {int(row['n_tracts']):,} tracts, "
              f"mean smoke 30d = {row['mean_smoke_30d']:.4f}")


if __name__ == "__main__":
    main()

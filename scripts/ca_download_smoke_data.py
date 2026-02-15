#!/usr/bin/env python3
"""
Extract California tract-level daily smoke PM2.5 from the ECHO Lab dataset.

Source: Childs et al., ECHO Lab wildfire smoke predictions v2.0 (2006-2023)
  Full US tract-level CSV located at: data/smoke/tract/
  smokePM2pt5_predictions_daily_tract_20060101-20231231.csv

IMPORTANT: The source file contains ONLY smoke days. Non-smoke days have
smokePM_pred = 0 by construction and are omitted from the file. Downstream
scripts must fill missing tract-days with 0 when computing window averages.

This script filters the full US file to California tracts (GEOID starting
with "06") and saves the CA subset.

Output: data/california/smoke/smoke_pm25_tract_daily.csv
  Columns: GEOID (11-digit tract FIPS), date (YYYYMMDD), smokePM_pred (µg/m³)
  Note: Like the source, this file contains ONLY smoke days.
"""

import argparse
import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "california", "smoke")
OUT_FILE = os.path.join(OUT_DIR, "smoke_pm25_tract_daily.csv")

# Source: full US tract-level file in data/smoke/tract/
TRACT_SOURCE = os.path.join(
    BASE_DIR, "data", "smoke", "tract",
    "smokePM2pt5_predictions_daily_tract_20060101-20231231.csv"
)


def filter_california(input_path):
    """Filter tract-level smoke data to California (GEOID starting with 06)."""
    print(f"\nFiltering to California tracts from {input_path}...")

    # Read in chunks to handle large file (~2.3 GB, ~56M rows)
    chunks = []
    total_rows = 0
    ca_rows = 0

    for chunk in pd.read_csv(input_path, dtype={"GEOID": str},
                             chunksize=2_000_000):
        total_rows += len(chunk)
        chunk["GEOID"] = chunk["GEOID"].str.zfill(11)
        ca_chunk = chunk[chunk["GEOID"].str.startswith("06")]
        ca_rows += len(ca_chunk)
        if len(ca_chunk) > 0:
            chunks.append(ca_chunk)
        print(f"\r  Processed {total_rows:,} rows, CA rows so far: {ca_rows:,}",
              end="", flush=True)

    print()
    if not chunks:
        print("  ERROR: No California tracts found!")
        return None

    ca_df = pd.concat(chunks, ignore_index=True)
    print(f"  Total US rows: {total_rows:,}")
    print(f"  California rows: {len(ca_df):,} ({len(ca_df)/total_rows*100:.1f}%)")

    return ca_df


def verify_smoke_data(path):
    """Load and verify the CA tract-level smoke dataset."""
    print(f"\nVerifying CA smoke data: {path}")
    df = pd.read_csv(path, dtype={"GEOID": str})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["GEOID"] = df["GEOID"].str.zfill(11)

    print(f"  Rows: {len(df):,} (smoke days only; non-smoke days are 0)")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Unique tracts: {df['GEOID'].nunique():,}")
    print(f"  smokePM_pred: mean={df['smokePM_pred'].mean():.4f}, "
          f"max={df['smokePM_pred'].max():.1f} µg/m³")

    # Smoke by year
    df["year"] = df["date"].dt.year
    print(f"\n  Annual summary:")
    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr]
        n_tracts = yr_df["GEOID"].nunique()
        mean_smoke = yr_df["smokePM_pred"].mean()
        max_smoke = yr_df["smokePM_pred"].max()
        print(f"    {yr}: {n_tracts:,} tracts with smoke days, mean={mean_smoke:.3f}, "
              f"max={max_smoke:.1f}")

    # Spot-check: 2020 fire season
    aug_oct_2020 = df[(df["date"] >= "2020-08-01") & (df["date"] <= "2020-10-31")]
    if len(aug_oct_2020) > 0:
        top_tracts = aug_oct_2020.groupby("GEOID")["smokePM_pred"].max().nlargest(5)
        print(f"\n  Top 5 CA tracts by max smoke PM2.5, Aug-Oct 2020:")
        for geoid, val in top_tracts.items():
            print(f"    GEOID {geoid}: {val:.1f} µg/m³")

    print("\nVerification complete.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Extract CA tract-level smoke PM2.5 data from full US file."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to full US tract-level CSV (default: data/smoke/tract/...)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CA Step 1: Extract Tract-Level Smoke PM2.5 Data")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Check if CA file already exists
    if os.path.exists(OUT_FILE):
        size_mb = os.path.getsize(OUT_FILE) / 1e6
        print(f"\nCA smoke data already exists ({size_mb:.1f} MB): {OUT_FILE}")
        verify_smoke_data(OUT_FILE)
        print(f"\nStep 1 complete: {OUT_FILE}")
        return

    # Determine source file
    input_path = args.input or TRACT_SOURCE

    if not os.path.exists(input_path):
        print(f"\n  ERROR: Source file not found: {input_path}")
        print(f"  Expected full US tract-level CSV at:")
        print(f"    {TRACT_SOURCE}")
        print(f"  Or specify with: python scripts/ca_download_smoke_data.py --input /path/to/file.csv")
        return

    size_gb = os.path.getsize(input_path) / 1e9
    print(f"\nSource file: {input_path} ({size_gb:.1f} GB)")

    # Filter to California
    ca_df = filter_california(input_path)
    if ca_df is None:
        return

    # Save
    ca_df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved CA smoke data: {OUT_FILE}")
    print(f"  Shape: {ca_df.shape}")

    verify_smoke_data(OUT_FILE)
    print(f"\nStep 1 complete: {OUT_FILE}")


if __name__ == "__main__":
    main()

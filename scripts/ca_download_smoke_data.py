#!/usr/bin/env python3
"""
Download Stanford ECHO Lab v2 tract-level daily smoke PM2.5 for California.

Source: Childs et al., ECHO Lab wildfire smoke predictions
  v2.0 (2006-2023): Tract-level daily predictions from Stanford ECHO Lab
  Harvard Dataverse doi:10.7910/DVN/DJVMTV (v1 county-level)
  Tract-level data available via Dropbox (v2)

Downloads the full US tract-level file, filters to California (GEOID starting
with "06"), and saves the CA subset.

Output: data/california/smoke/smoke_pm25_tract_daily.csv
  Columns: GEOID (11-digit tract FIPS), date (YYYYMMDD), smokePM_pred (µg/m³)
"""

import argparse
import os
import sys
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "california", "smoke")
OUT_FILE = os.path.join(OUT_DIR, "smoke_pm25_tract_daily.csv")

# Stanford ECHO Lab tract-level smoke PM2.5 v2 (Dropbox)
# The tract-level file is named like:
#   smokePM2pt5_predictions_daily_tract_20060101-20231231.csv
# If this URL stops working, visit https://www.stanfordecholab.com/wildfire_smoke
TRACT_DROPBOX_FOLDER = (
    "https://www.dropbox.com/scl/fo/91k0aq80vp57qixkm508q/"
    "AKQSIJ5C1kDMQLz8oh02UAA?rlkey=nutebc9pn2vsupr0p9ks4k73u&dl=0"
)
TRACT_ORIGINAL_NAME = "smokePM2pt5_predictions_daily_tract_20060101-20231231.csv"

# Also check for a local full-US file the user may have already downloaded
FULL_US_FILE = os.path.join(OUT_DIR, "smoke_pm25_tract_daily_us.csv")


def download_file(url, dest, chunk_size=1024 * 1024):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  File already exists ({size_mb:.1f} MB): {dest}")
        return False

    print(f"  Downloading from {url}")
    resp = requests.get(url, stream=True, allow_redirects=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)",
                      end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:.1f} MB downloaded", end="", flush=True)
    print()
    return True


def filter_california(input_path):
    """Filter tract-level smoke data to California (GEOID starting with 06)."""
    print(f"\nFiltering to California tracts from {input_path}...")

    # Auto-detect delimiter
    with open(input_path) as f:
        sep = "\t" if "\t" in f.readline() else ","

    # Read in chunks to handle large file (~2GB)
    chunks = []
    total_rows = 0
    ca_rows = 0

    for chunk in pd.read_csv(input_path, sep=sep, dtype={"GEOID": str},
                             chunksize=1_000_000):
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

    print(f"  Rows: {len(df):,}")
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
        pct_nonzero = (yr_df["smokePM_pred"] > 0).mean() * 100
        print(f"    {yr}: {n_tracts:,} tracts, mean={mean_smoke:.3f}, "
              f"max={max_smoke:.1f}, {pct_nonzero:.1f}% nonzero")

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
        description="Download tract-level smoke PM2.5 data for California."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to already-downloaded full US tract-level CSV (skips download)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CA Step 1: Download Tract-Level Smoke PM2.5 Data")
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
    input_path = args.input

    if input_path is None:
        # Check for local full-US file
        local_candidates = [
            FULL_US_FILE,
            os.path.join(OUT_DIR, TRACT_ORIGINAL_NAME),
            os.path.join(BASE_DIR, "data", "smoke", TRACT_ORIGINAL_NAME),
        ]
        for candidate in local_candidates:
            if os.path.exists(candidate):
                input_path = candidate
                print(f"\nFound local US tract file: {input_path}")
                break

    if input_path is None:
        # Tract-level data requires manual download (Dropbox folder)
        print("\n  *** MANUAL DOWNLOAD REQUIRED ***")
        print()
        print("  The tract-level smoke PM2.5 data (~2GB) must be downloaded")
        print("  manually from the Stanford ECHO Lab Dropbox folder:")
        print(f"    {TRACT_DROPBOX_FOLDER}")
        print()
        print("  Steps:")
        print(f"    1. Open the Dropbox folder in your browser")
        print(f"    2. Find the tract-level CSV:")
        print(f"       {TRACT_ORIGINAL_NAME}")
        print(f"    3. Download and save to one of these locations:")
        print(f"       {FULL_US_FILE}")
        print(f"       {os.path.join(OUT_DIR, TRACT_ORIGINAL_NAME)}")
        print(f"    4. Re-run this script")
        print()
        print("  Or specify the path directly:")
        print(f"    python scripts/ca_download_smoke_data.py --input /path/to/file.csv")
        return

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

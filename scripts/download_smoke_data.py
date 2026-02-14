#!/usr/bin/env python3
"""
Phase 1: Download Stanford Echo Lab wildfire smoke PM2.5 data.

Source: Childs et al. (2022, ES&T), Harvard Dataverse doi:10.7910/DVN/DJVMTV
File: County-level daily smoke PM2.5 predictions, 2006-01-01 to 2020-12-31
"""

import os
import sys
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "smoke")
OUT_FILE = os.path.join(OUT_DIR, "smoke_pm25_county_daily.csv")

# Harvard Dataverse file ID for county-level daily CSV
DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/8550336"


def download_file(url, dest, chunk_size=1024 * 1024):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  File already exists ({size_mb:.1f} MB): {dest}")
        return False

    print(f"  Downloading from {url}")
    resp = requests.get(url, stream=True, allow_redirects=True, timeout=300)
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
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:.1f} MB downloaded", end="", flush=True)
    print()
    return True


def verify_smoke_data(path):
    """Load and verify the smoke PM2.5 dataset."""
    print("\nVerifying smoke data...")

    # File is tab-delimited with columns: GEOID, date (YYYYMMDD), smokePM_pred
    print("  Loading full dataset (this may take a moment)...")
    df = pd.read_csv(path, sep="\t", dtype={"GEOID": str})
    print(f"  Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["GEOID"] = df["GEOID"].str.zfill(5)

    print(f"  Rows: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Unique counties (GEOID): {df['GEOID'].nunique():,}")
    print(f"  smokePM_pred: mean={df['smokePM_pred'].mean():.4f}, "
          f"max={df['smokePM_pred'].max():.1f} µg/m³")

    # Spot-check: August-September 2020 Western US fires
    aug_sep_2020 = df[(df["date"] >= "2020-08-01") & (df["date"] <= "2020-09-30")]
    top_counties = aug_sep_2020.groupby("GEOID")["smokePM_pred"].max().nlargest(5)
    print(f"\n  Top 5 counties by max smoke PM2.5, Aug-Sep 2020:")
    for fips, val in top_counties.items():
        print(f"    GEOID {fips}: {val:.1f} µg/m³")

    print("\nSmoke data verification complete.")
    return True


def main():
    print("=" * 60)
    print("Phase 1: Download Stanford Echo Lab Smoke PM2.5 Data")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nDownloading county-level daily smoke PM2.5...")
    downloaded = download_file(DATAVERSE_URL, OUT_FILE)

    if downloaded:
        print(f"  Saved to {OUT_FILE}")

    verify_smoke_data(OUT_FILE)

    print(f"\nPhase 1 complete. Output: {OUT_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phase 1: Download Stanford Echo Lab wildfire smoke PM2.5 data.

Two dataset versions are supported:
  v1.0 (2006-2020): Childs et al. (2022, ES&T), Harvard Dataverse doi:10.7910/DVN/DJVMTV
  v2.0 (2006-2023+): Beta release from Stanford ECHO Lab Dropbox, extends through 2023
                      (2024 expected to be added; check stanfordecholab.com/wildfire_smoke)

Usage:
  python scripts/download_smoke_data.py           # default: downloads v1.0
  python scripts/download_smoke_data.py --v2      # downloads v2.0
  python scripts/download_smoke_data.py --both    # downloads both versions

Both versions produce county-level daily smoke PM2.5 predictions with columns:
  GEOID (5-digit county FIPS), date (YYYYMMDD), smokePM_pred (µg/m³)

The v2.0 data are preliminary and subject to change. To use v2.0 in the analysis
pipeline, set SMOKE_VERSION=v2 when running build scripts, or symlink the desired
version to smoke_pm25_county_daily.csv.
"""

import argparse
import os
import sys
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "smoke")

# --- v1.0: Harvard Dataverse (2006-2020) ---
V1_URL = "https://dataverse.harvard.edu/api/access/datafile/8550336"
V1_FILE = os.path.join(OUT_DIR, "smoke_pm25_county_daily_v1.csv")

# --- v2.0: Stanford ECHO Lab Dropbox (2006-2023+) ---
# County-level CSV from the v2.0 beta Dropbox folder.
# If this URL stops working, visit https://www.stanfordecholab.com/wildfire_smoke
# and get the updated Dropbox link for county-level data.
V2_DROPBOX_FOLDER = (
    "https://www.dropbox.com/scl/fo/91k0aq80vp57qixkm508q/"
    "AKQSIJ5C1kDMQLz8oh02UAA?rlkey=nutebc9pn2vsupr0p9ks4k73u&dl=0"
)
V2_FILE = os.path.join(OUT_DIR, "smoke_pm25_county_daily_v2.csv")
V2_ORIGINAL_NAME = "smokePM2pt5_predictions_daily_county_20060101-20231231.csv"

# Default symlink target — the analysis pipeline reads this file
DEFAULT_FILE = os.path.join(OUT_DIR, "smoke_pm25_county_daily.csv")


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
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:.1f} MB downloaded", end="", flush=True)
    print()
    return True


def verify_smoke_data(path, label=""):
    """Load and verify a smoke PM2.5 dataset."""
    print(f"\nVerifying smoke data{' (' + label + ')' if label else ''}...")

    # v1.0 is tab-delimited; v2.0 is comma-delimited with quoted fields.
    # Detect delimiter from first line.
    print("  Loading dataset (this may take a moment)...")
    with open(path) as f:
        first_line = f.readline()
    sep = "\t" if "\t" in first_line else ","
    df = pd.read_csv(path, sep=sep, dtype={"GEOID": str})
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
    if len(aug_sep_2020) > 0:
        top_counties = aug_sep_2020.groupby("GEOID")["smokePM_pred"].max().nlargest(5)
        print(f"\n  Top 5 counties by max smoke PM2.5, Aug-Sep 2020:")
        for fips, val in top_counties.items():
            print(f"    GEOID {fips}: {val:.1f} µg/m³")

    # Check for post-2020 data (v2.0)
    post_2020 = df[df["date"] >= "2021-01-01"]
    if len(post_2020) > 0:
        years = sorted(post_2020["date"].dt.year.unique())
        print(f"\n  Post-2020 data present: years {years}")
        print(f"  Post-2020 rows: {len(post_2020):,}")

    print("\nVerification complete.")
    return True


def set_default_version(version_file):
    """Point the default symlink/copy to the specified version file."""
    if not os.path.exists(version_file):
        print(f"  Warning: {version_file} does not exist, skipping default link.")
        return

    # Remove existing default if present
    if os.path.exists(DEFAULT_FILE) or os.path.islink(DEFAULT_FILE):
        os.remove(DEFAULT_FILE)

    # Use symlink on Unix, copy on Windows
    if sys.platform != "win32":
        os.symlink(os.path.basename(version_file), DEFAULT_FILE)
        print(f"  Symlinked {DEFAULT_FILE} -> {os.path.basename(version_file)}")
    else:
        import shutil
        shutil.copy2(version_file, DEFAULT_FILE)
        print(f"  Copied {version_file} -> {DEFAULT_FILE}")


def download_v1():
    """Download v1.0 data from Harvard Dataverse."""
    print("\n--- v1.0: Childs et al. (2022), 2006-2020 ---")
    print("  Source: Harvard Dataverse doi:10.7910/DVN/DJVMTV")
    downloaded = download_file(V1_URL, V1_FILE)
    if downloaded:
        print(f"  Saved to {V1_FILE}")
    verify_smoke_data(V1_FILE, label="v1.0")
    return V1_FILE


def download_v2():
    """Download v2.0 data from Stanford ECHO Lab Dropbox."""
    print("\n--- v2.0 (beta): ECHO Lab, 2006-2023+ ---")
    print("  Source: Stanford ECHO Lab Dropbox")
    print(f"  Folder: {V2_DROPBOX_FOLDER}")
    print()

    # Check for the file under either our canonical name or the original ECHO Lab name
    v2_original = os.path.join(OUT_DIR, V2_ORIGINAL_NAME)
    if not os.path.exists(V2_FILE) and os.path.exists(v2_original):
        os.symlink(V2_ORIGINAL_NAME, V2_FILE)
        print(f"  Linked {V2_FILE} -> {V2_ORIGINAL_NAME}")

    if os.path.exists(V2_FILE):
        size_mb = os.path.getsize(V2_FILE) / 1e6
        print(f"  File already exists ({size_mb:.1f} MB): {V2_FILE}")
        verify_smoke_data(V2_FILE, label="v2.0")
        return V2_FILE

    # Dropbox folder links require manual download or direct file links.
    # The folder URL above contains the county-level CSV, but Dropbox folder
    # links don't support direct programmatic download of individual files
    # without the Dropbox API.
    #
    # To download v2.0:
    #   1. Visit the Dropbox folder URL above in a browser
    #   2. Find the county-level CSV (named like
    #      smokePM2pt5_predictions_daily_county_20060101-20231231.csv)
    #   3. Download it and save to: data/smoke/smoke_pm25_county_daily_v2.csv
    #
    # Alternatively, if you have a direct file link (dl=1), set V2_DIRECT_URL
    # below and re-run this script.

    print("  *** MANUAL DOWNLOAD REQUIRED ***")
    print()
    print("  Dropbox folder links cannot be downloaded programmatically.")
    print("  Please:")
    print(f"    1. Open the Dropbox folder in your browser:")
    print(f"       {V2_DROPBOX_FOLDER}")
    print(f"    2. Find the county-level CSV file")
    print(f"       (e.g., smokePM2pt5_predictions_daily_county_20060101-20231231.csv)")
    print(f"    3. Download and save it to:")
    print(f"       {V2_FILE}")
    print(f"    4. Re-run this script to verify the data.")
    print()
    print("  Tip: If you get a direct file link (ending in ?dl=0), change dl=0")
    print("  to dl=1 and set it as V2_DIRECT_URL in this script for auto-download.")

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Download Stanford ECHO Lab wildfire smoke PM2.5 data."
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="Download v2.0 (2006-2023+) instead of v1.0"
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Download both v1.0 and v2.0"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1: Download Stanford Echo Lab Smoke PM2.5 Data")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.both:
        v1_path = download_v1()
        v2_path = download_v2()
        # Default to v1 for reproducibility; user can switch with --v2
        active = v1_path
        print("\n  Both versions downloaded. Default set to v1.0.")
        print("  To switch to v2.0, run: python scripts/download_smoke_data.py --v2")
    elif args.v2:
        v2_path = download_v2()
        active = v2_path
    else:
        v1_path = download_v1()
        active = v1_path

    if active:
        set_default_version(active)

    print(f"\nPhase 1 complete. Active data: {DEFAULT_FILE}")


if __name__ == "__main__":
    main()

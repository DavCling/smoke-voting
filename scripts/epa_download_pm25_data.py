#!/usr/bin/env python3
"""
Download EPA AQS daily PM2.5 (parameter 88101) bulk data files.

Source: EPA Air Quality System pre-generated daily summary files.
  https://aqs.epa.gov/aqsweb/airdata/daily_88101_{YEAR}.csv.zip

Parameter 88101 = PM2.5 FRM/FEM (Federal Reference/Equivalent Method),
the regulatory standard measurement from ground-based monitors.

Downloads are idempotent — existing files are skipped.

Usage:
    python scripts/epa_download_pm25_data.py
"""

import os
import zipfile
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPA_DIR = os.path.join(BASE_DIR, "data", "epa")

# Presidential election years covered by Childs et al. smoke data
YEARS = [2008, 2012, 2016, 2020]

URL_TEMPLATE = (
    "https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip"
)


def download_file(url, dest, label=""):
    """Download a file with progress reporting. Skip if exists."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Already exists ({size_mb:.1f} MB): {os.path.basename(dest)}")
        return False

    print(f"  Downloading {label or os.path.basename(dest)}...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)",
                      end="", flush=True)
            else:
                print(f"\r    {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()
    return True


def extract_csv(zip_path, dest_dir, year):
    """Extract the CSV from a ZIP file if not already extracted."""
    csv_name = f"daily_88101_{year}.csv"
    csv_path = os.path.join(dest_dir, csv_name)

    if os.path.exists(csv_path):
        size_mb = os.path.getsize(csv_path) / 1e6
        print(f"  Already extracted ({size_mb:.1f} MB): {csv_name}")
        return csv_path

    print(f"  Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the CSV file inside the ZIP
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        if not csv_files:
            print(f"    WARNING: No CSV found in {zip_path}")
            return None

        # Extract and rename to standard name
        zf.extract(csv_files[0], dest_dir)
        extracted = os.path.join(dest_dir, csv_files[0])
        if extracted != csv_path:
            os.rename(extracted, csv_path)
        print(f"    Extracted: {csv_name}")

    return csv_path


def verify_csv(csv_path, year):
    """Load and verify an EPA daily PM2.5 CSV."""
    print(f"\n  EPA {year}:")

    df = pd.read_csv(csv_path, dtype={"State Code": str, "County Code": str})
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {len(df.columns)}")

    # Date range
    df["_date"] = pd.to_datetime(df["Date Local"])
    print(f"    Date range: {df['_date'].min().date()} to {df['_date'].max().date()}")

    # Build county FIPS
    df["_fips"] = df["State Code"].str.zfill(2) + df["County Code"].str.zfill(3)
    n_counties = df["_fips"].nunique()
    n_states = df["State Code"].str.zfill(2).nunique()
    print(f"    States: {n_states}, Counties: {n_counties}")

    # PM2.5 summary
    pm25 = df["Arithmetic Mean"]
    print(f"    PM2.5 — mean: {pm25.mean():.2f}, median: {pm25.median():.2f}, "
          f"max: {pm25.max():.1f} µg/m³")

    # Sample durations
    durations = df["Sample Duration"].value_counts()
    print(f"    Sample durations:")
    for dur, cnt in durations.head(5).items():
        print(f"      {dur}: {cnt:,} ({100*cnt/len(df):.1f}%)")

    # Monitor count
    n_monitors = df.groupby("_fips").apply(
        lambda g: g[["State Code", "County Code", "Site Num", "POC"]].drop_duplicates().shape[0]
    ).sum()
    print(f"    Monitor-POC combinations: {n_monitors:,}")


def main():
    print("=" * 60)
    print("Download EPA AQS Daily PM2.5 Data (Parameter 88101)")
    print("=" * 60)

    os.makedirs(EPA_DIR, exist_ok=True)

    for year in YEARS:
        print(f"\n--- {year} ---")
        url = URL_TEMPLATE.format(year=year)
        zip_path = os.path.join(EPA_DIR, f"daily_88101_{year}.zip")

        download_file(url, zip_path, label=f"EPA PM2.5 {year}")
        csv_path = extract_csv(zip_path, EPA_DIR, year)

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    for year in YEARS:
        csv_path = os.path.join(EPA_DIR, f"daily_88101_{year}.csv")
        if os.path.exists(csv_path):
            verify_csv(csv_path, year)
        else:
            print(f"\n  WARNING: {csv_path} not found")

    print("\nDownload complete.")


if __name__ == "__main__":
    main()

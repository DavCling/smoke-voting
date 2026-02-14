#!/usr/bin/env python3
"""
Phase 2: Download MEDSL county-level presidential election returns.

Source: MIT Election Data + Science Lab, Harvard Dataverse doi:10.7910/DVN/VOQCHQ
File: countypres_2000-2024.csv — county-level presidential returns
"""

import os
import sys
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "elections")
OUT_FILE = os.path.join(OUT_DIR, "county_presidential.csv")

# Harvard Dataverse file ID for county presidential returns (original CSV format)
DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/13454740?format=original"


def download_file(url, dest, chunk_size=1024 * 1024):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  File already exists ({size_mb:.1f} MB): {dest}")
        return False

    print(f"  Downloading from {url}")
    resp = requests.get(url, stream=True, allow_redirects=True, timeout=120)
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


def verify_election_data(path):
    """Load and verify the MEDSL county presidential dataset."""
    print("\nVerifying election data...")
    df = pd.read_csv(path, dtype={"county_fips": str})

    # Zero-pad FIPS to 5 digits
    df["county_fips"] = df["county_fips"].str.zfill(5)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  States: {df['state_po'].nunique()}")

    # For 2020, check mode column
    if "mode" in df.columns:
        modes_2020 = df[df["year"] == 2020]["mode"].unique()
        print(f"  2020 vote modes: {modes_2020}")

    # Summarize by year
    print(f"\n  Counties and votes by year:")
    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr]
        # Filter to TOTAL mode if available
        if "mode" in yr_df.columns:
            total_rows = yr_df[yr_df["mode"] == "TOTAL"]
            if len(total_rows) > 0:
                yr_df = total_rows
        n_counties = yr_df["county_fips"].nunique()
        # Sum DEM + REP votes
        dem = yr_df[yr_df["party"] == "DEMOCRAT"]["candidatevotes"].sum()
        rep = yr_df[yr_df["party"] == "REPUBLICAN"]["candidatevotes"].sum()
        print(f"    {yr}: {n_counties:,} counties, DEM={dem:,.0f}, REP={rep:,.0f}")

    print("\nElection data verification complete.")
    return True


def main():
    print("=" * 60)
    print("Phase 2: Download MEDSL County Presidential Returns")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\nDownloading county presidential returns (2000-2024)...")
    downloaded = download_file(DATAVERSE_URL, OUT_FILE)

    if downloaded:
        print(f"  Saved to {OUT_FILE}")

    verify_election_data(OUT_FILE)

    print(f"\nPhase 2 complete. Output: {OUT_FILE}")


if __name__ == "__main__":
    main()

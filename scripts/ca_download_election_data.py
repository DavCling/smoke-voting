#!/usr/bin/env python3
"""
Download California precinct-level election data and precinct-to-tract crosswalk.

Sources:
  A. Pre-built precinct-to-tract crosswalk:
     Voting and Registration Tabulation (VRT) / Nature Scientific Data 2025 paper
     Harvard Dataverse doi:10.7910/DVN/NH5S2I (or similar)
     Provides precinct-to-tract allocation factors for CA elections.

  B. Statewide Database (SWDB) precinct-level election returns:
     https://statewidedatabase.org
     Statement of Vote files with precinct-level presidential and US House results
     for California general elections 2006-2022.

  C. SWDB precinct boundary shapefiles (for crosswalk validation):
     Used to independently compute area-weighted precinct→tract intersections.

Output:
  data/california/elections/
    swdb_<year>.csv           — SWDB precinct-level returns per election year
  data/california/crosswalk/
    prebuilt_crosswalk.csv    — Pre-built precinct→tract allocation
    swdb_shapefiles/          — Precinct boundary shapefiles by year
"""

import os
import sys
import io
import zipfile
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELECTIONS_DIR = os.path.join(BASE_DIR, "data", "california", "elections")
CROSSWALK_DIR = os.path.join(BASE_DIR, "data", "california", "crosswalk")

# SWDB URLs for Statement of Vote data (general elections)
# These are publicly available CSV downloads from statewidedatabase.org
# Format: statewide precinct-level returns with party/candidate breakdowns
SWDB_YEARS = {
    2006: {
        "url": "https://statewidedatabase.org/d10/g06_geo_140_sr_csv.zip",
        "label": "2006 General (House)",
    },
    2008: {
        "url": "https://statewidedatabase.org/d10/g08_geo_140_sr_csv.zip",
        "label": "2008 General (Pres + House)",
    },
    2010: {
        "url": "https://statewidedatabase.org/d10/g10_geo_140_sr_csv.zip",
        "label": "2010 General (House)",
    },
    2012: {
        "url": "https://statewidedatabase.org/d10/g12_geo_140_sr_csv.zip",
        "label": "2012 General (Pres + House)",
    },
    2014: {
        "url": "https://statewidedatabase.org/d20/g14_geo_140_sr_csv.zip",
        "label": "2014 General (House)",
    },
    2016: {
        "url": "https://statewidedatabase.org/d20/g16_geo_140_sr_csv.zip",
        "label": "2016 General (Pres + House)",
    },
    2018: {
        "url": "https://statewidedatabase.org/d20/g18_geo_140_sr_csv.zip",
        "label": "2018 General (House)",
    },
    2020: {
        "url": "https://statewidedatabase.org/d20/g20_geo_140_sr_csv.zip",
        "label": "2020 General (Pres + House)",
    },
    2022: {
        "url": "https://statewidedatabase.org/d20/g22_geo_140_sr_csv.zip",
        "label": "2022 General (House)",
    },
}

# Pre-built crosswalk sources
# Harvard Dataverse: precinct-to-census geography allocation file
PREBUILT_CROSSWALK_URL = (
    "https://dataverse.harvard.edu/api/access/datafile/:persistentId?"
    "persistentId=doi:10.7910/DVN/NH5S2I/XZPVGQ"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def download_file(url, dest, chunk_size=1024 * 1024, headers=None):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  File already exists ({size_mb:.1f} MB): {dest}")
        return False

    print(f"  Downloading: {url[:80]}...")
    hdrs = headers or HEADERS
    resp = requests.get(url, stream=True, allow_redirects=True, timeout=300,
                        headers=hdrs)
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


def download_and_extract_zip(url, dest_dir, label=""):
    """Download a ZIP file and extract its contents."""
    zip_path = os.path.join(dest_dir, "_temp_download.zip")

    # Check if we already have extracted files for this year
    existing_csvs = [f for f in os.listdir(dest_dir) if f.endswith(".csv")]
    if existing_csvs:
        # Already extracted
        return existing_csvs

    print(f"  Downloading {label}...")
    try:
        resp = requests.get(url, stream=True, allow_redirects=True, timeout=300,
                            headers=HEADERS)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR downloading {label}: {e}")
        return []

    with open(zip_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    # Extract
    extracted = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                zf.extract(name, dest_dir)
                extracted.append(name)
    except zipfile.BadZipFile:
        print(f"  ERROR: Bad ZIP file for {label}")
        os.remove(zip_path)
        return []

    os.remove(zip_path)
    print(f"  Extracted {len(extracted)} files")
    return extracted


def download_swdb_elections():
    """Download SWDB precinct-level election returns for all years."""
    print("\n" + "=" * 60)
    print("Downloading SWDB Precinct-Level Election Returns")
    print("=" * 60)

    for year, info in sorted(SWDB_YEARS.items()):
        year_dir = os.path.join(ELECTIONS_DIR, f"swdb_{year}")
        os.makedirs(year_dir, exist_ok=True)

        # Check if already processed
        processed_file = os.path.join(ELECTIONS_DIR, f"swdb_{year}.csv")
        if os.path.exists(processed_file):
            df = pd.read_csv(processed_file, nrows=5)
            print(f"\n  {year} ({info['label']}): already processed → {processed_file}")
            continue

        print(f"\n  {year}: {info['label']}")
        extracted = download_and_extract_zip(info["url"], year_dir, info["label"])

        if not extracted:
            print(f"  WARNING: No files extracted for {year}")
            continue

        # Find and read the main data CSV
        csv_files = [f for f in extracted if f.lower().endswith(".csv")]
        if csv_files:
            print(f"  CSV files: {csv_files}")


def download_prebuilt_crosswalk():
    """Download pre-built precinct-to-tract crosswalk."""
    print("\n" + "=" * 60)
    print("Downloading Pre-Built Precinct-to-Tract Crosswalk")
    print("=" * 60)

    os.makedirs(CROSSWALK_DIR, exist_ok=True)
    dest = os.path.join(CROSSWALK_DIR, "prebuilt_crosswalk.csv")

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Already exists ({size_mb:.1f} MB): {dest}")
        return

    # Try Harvard Dataverse
    print("  Attempting Harvard Dataverse download...")
    try:
        downloaded = download_file(PREBUILT_CROSSWALK_URL, dest)
        if downloaded:
            print(f"  Saved to {dest}")
            return
    except Exception as e:
        print(f"  Dataverse download failed: {e}")

    # Fallback: provide manual instructions
    print("\n  *** MANUAL DOWNLOAD MAY BE REQUIRED ***")
    print()
    print("  The pre-built precinct-to-tract crosswalk can be obtained from:")
    print("  - Harvard Dataverse: doi:10.7910/DVN/NH5S2I")
    print("  - Or the 2025 Nature Scientific Data paper supplementary materials")
    print()
    print(f"  Save the crosswalk file to: {dest}")
    print("  Then re-run this script.")


def download_swdb_shapefiles():
    """Download SWDB precinct boundary shapefiles for crosswalk validation."""
    print("\n" + "=" * 60)
    print("Downloading SWDB Precinct Boundary Shapefiles")
    print("=" * 60)

    shp_dir = os.path.join(CROSSWALK_DIR, "swdb_shapefiles")
    os.makedirs(shp_dir, exist_ok=True)

    # SWDB provides shapefiles alongside the SOV data
    # These use the same base URL pattern but with _shp suffix
    print("  Note: SWDB shapefiles are typically bundled with the SOV ZIP files")
    print("  or available separately from statewidedatabase.org.")
    print("  The crosswalk builder will use these if available.")
    print(f"  Expected location: {shp_dir}")


def verify_elections():
    """Verify downloaded election data."""
    print("\n" + "=" * 60)
    print("Verifying Election Data")
    print("=" * 60)

    for year in sorted(SWDB_YEARS.keys()):
        year_dir = os.path.join(ELECTIONS_DIR, f"swdb_{year}")
        if not os.path.exists(year_dir):
            print(f"  {year}: NOT FOUND")
            continue

        files = os.listdir(year_dir)
        csv_files = [f for f in files if f.lower().endswith(".csv")]
        shp_files = [f for f in files if f.lower().endswith(".shp")]

        print(f"  {year}: {len(csv_files)} CSVs, {len(shp_files)} shapefiles")

        # Try to read the first CSV to check structure
        for csv_f in csv_files[:1]:
            try:
                df = pd.read_csv(os.path.join(year_dir, csv_f), nrows=5,
                                 low_memory=False)
                print(f"    Columns: {list(df.columns[:10])}...")
            except Exception as e:
                print(f"    Error reading {csv_f}: {e}")


def main():
    print("=" * 60)
    print("CA Step 2: Download Election Data & Crosswalk")
    print("=" * 60)

    os.makedirs(ELECTIONS_DIR, exist_ok=True)
    os.makedirs(CROSSWALK_DIR, exist_ok=True)

    # A. Pre-built crosswalk
    download_prebuilt_crosswalk()

    # B. SWDB precinct-level election returns
    download_swdb_elections()

    # C. SWDB shapefiles
    download_swdb_shapefiles()

    # Verify
    verify_elections()

    print(f"\nStep 2 complete.")
    print(f"  Elections: {ELECTIONS_DIR}")
    print(f"  Crosswalk: {CROSSWALK_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download California precinct-level election data and precinct-to-block crosswalks.

Sources:
  A. Statewide Database (SWDB) precinct-level election returns:
     https://statewidedatabase.org
     Statement of Vote files with precinct-level presidential and US House results
     for California general elections 2006-2022.

  B. SWDB precinct-to-block mapping files (sr_blk_map):
     Maps state reporting precincts to census blocks with share weights.
     Used to allocate precinct votes to tracts via blocks.

  C. Fekrazad (2025) precinct-to-tract allocated votes (validation):
     Harvard Dataverse doi:10.7910/DVN/Z8TSH3
     Already-allocated tract-level vote tallies for 2016+2020.

Output:
  data/california/elections/
    swdb_<year>/               — SWDB precinct-level returns per election year
  data/california/crosswalk/
    sr_blk_map_<year>.csv      — Precinct-to-block mapping per year
    fekrazad_ca.zip            — Pre-allocated tract-level votes (2016+2020)
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

# SWDB precinct-level SOV data URLs (verified working)
SWDB_SOV_URLS = {
    2006: {
        "url": "https://statewidedatabase.org/pub/data/G06/state/state_g06_sov_data_by_g06_srprec.zip",
        "label": "2006 General (House)",
    },
    2008: {
        "url": "https://statewidedatabase.org/pub/data/G08/state/state_g08_sov_data_by_g08_srprec.zip",
        "label": "2008 General (Pres + House)",
    },
    2010: {
        "url": "https://statewidedatabase.org/pub/data/G10/state/state_g10_sov_data_by_g10_srprec.zip",
        "label": "2010 General (House)",
    },
    2012: {
        "url": "https://statewidedatabase.org/pub/data/G12/state/state_g12_sov_data_by_g12_srprec.zip",
        "label": "2012 General (Pres + House)",
    },
    2014: {
        "url": "https://statewidedatabase.org/pub/data/G14/state/state_g14_sov_data_by_g14_srprec.zip",
        "label": "2014 General (House)",
    },
    2016: {
        "url": "https://statewidedatabase.org/pub/data/G16/state/state_g16_sov_data_by_g16_srprec.zip",
        "label": "2016 General (Pres + House)",
    },
    2018: {
        "url": "https://statewidedatabase.org/pub/data/G18/state/state_g18_sov_data_by_g18_srprec.zip",
        "label": "2018 General (House)",
    },
    2020: {
        "url": "https://statewidedatabase.org/pub/data/G20/state/state_g20_sov_data_by_g20_srprec.zip",
        "label": "2020 General (Pres + House)",
    },
    2022: {
        "url": "https://statewidedatabase.org/pub/data/G22/state/state_g22_sov_data_by_g22_srprec.zip",
        "label": "2022 General (House)",
    },
}

# SWDB precinct-to-block mapping files
# These map state reporting precincts to census blocks with share weights
# Format varies: some are .zip, some are .csv
SWDB_BLK_MAP_URLS = {
    2006: "https://statewidedatabase.org/pub/data/D10/2001by2011/state/state_g06_2011blk_by_g06_sr.csv",
    2008: "https://statewidedatabase.org/pub/data/D10/2001by2011/state/state_g08_2011blk_by_g08_sr.csv",
    2010: "https://statewidedatabase.org/pub/data/G10/state/state_g10_sr_blk_map.zip",
    2012: "https://statewidedatabase.org/pub/data/G12/state/state_g12_sr_blk_map.zip",
    2014: "https://statewidedatabase.org/pub/data/G14/state/state_g14_sr_blk_map.zip",
    2016: "https://statewidedatabase.org/pub/data/G16/state/state_g16_sr_blk_map.zip",
    2018: "https://statewidedatabase.org/pub/data/G18/state/state_g18_sr_blk_map.csv",
    2020: "https://statewidedatabase.org/pub/data/G20/state/state_g20_sr_blk_map.csv",
    2022: "https://statewidedatabase.org/pub/data/G22/state/state_g22_sr_blk_map.csv",
}

# Fekrazad (2025) pre-allocated tract-level votes for validation
# DOI: 10.7910/DVN/Z8TSH3 — California file (060 CA.zip)
FEKRAZAD_URL = "https://dataverse.harvard.edu/api/access/datafile/11111230"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def download_file(url, dest, chunk_size=1024 * 1024, headers=None):
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Already exists ({size_mb:.1f} MB): {dest}")
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

    # Check if we already have extracted files
    existing = [f for f in os.listdir(dest_dir)
                if f.endswith(".csv") or f.endswith(".dbf")]
    if existing:
        print(f"  Already extracted ({len(existing)} files): {dest_dir}")
        return existing

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

    for year, info in sorted(SWDB_SOV_URLS.items()):
        year_dir = os.path.join(ELECTIONS_DIR, f"swdb_{year}")
        os.makedirs(year_dir, exist_ok=True)

        print(f"\n  {year}: {info['label']}")
        extracted = download_and_extract_zip(info["url"], year_dir, info["label"])

        if not extracted:
            print(f"  WARNING: No files extracted for {year}")
            continue

        csv_files = [f for f in extracted if f.lower().endswith(".csv")]
        if csv_files:
            print(f"  CSV files: {csv_files}")


def download_blk_maps():
    """Download SWDB precinct-to-block mapping files."""
    print("\n" + "=" * 60)
    print("Downloading SWDB Precinct-to-Block Mappings")
    print("=" * 60)

    os.makedirs(CROSSWALK_DIR, exist_ok=True)

    for year, url in sorted(SWDB_BLK_MAP_URLS.items()):
        dest_csv = os.path.join(CROSSWALK_DIR, f"sr_blk_map_{year}.csv")

        if os.path.exists(dest_csv):
            size_mb = os.path.getsize(dest_csv) / 1e6
            print(f"\n  {year}: Already exists ({size_mb:.1f} MB)")
            continue

        print(f"\n  {year}: {url.split('/')[-1]}")

        if url.endswith(".zip"):
            # Download ZIP, extract CSV
            tmp_dir = os.path.join(CROSSWALK_DIR, f"_tmp_blk_{year}")
            os.makedirs(tmp_dir, exist_ok=True)
            extracted = download_and_extract_zip(url, tmp_dir, f"{year} blk_map")
            if extracted:
                csv_files = [f for f in extracted if f.lower().endswith(".csv")]
                if csv_files:
                    src = os.path.join(tmp_dir, csv_files[0])
                    os.rename(src, dest_csv)
                    print(f"  Saved: {dest_csv}")
                # Clean up temp dir
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            # Direct CSV download
            try:
                download_file(url, dest_csv)
            except Exception as e:
                print(f"  ERROR: {e}")


def download_fekrazad():
    """Download Fekrazad (2025) pre-allocated tract-level votes for validation."""
    print("\n" + "=" * 60)
    print("Downloading Fekrazad Tract-Level Votes (2016+2020, validation)")
    print("=" * 60)

    os.makedirs(CROSSWALK_DIR, exist_ok=True)
    dest_zip = os.path.join(CROSSWALK_DIR, "fekrazad_ca.zip")
    dest_dir = os.path.join(CROSSWALK_DIR, "fekrazad_ca")

    if os.path.exists(dest_dir) and os.listdir(dest_dir):
        print(f"  Already downloaded: {dest_dir}")
        return

    try:
        download_file(FEKRAZAD_URL, dest_zip)
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Manual download: https://dataverse.harvard.edu/dataset.xhtml?"
              "persistentId=doi:10.7910/DVN/Z8TSH3")
        return

    if os.path.exists(dest_zip):
        os.makedirs(dest_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(dest_zip, "r") as zf:
                zf.extractall(dest_dir)
            print(f"  Extracted to {dest_dir}")
            os.remove(dest_zip)
        except zipfile.BadZipFile:
            print(f"  ERROR: Bad ZIP file")


def verify_elections():
    """Verify downloaded election data."""
    print("\n" + "=" * 60)
    print("Verifying Election Data")
    print("=" * 60)

    for year in sorted(SWDB_SOV_URLS.keys()):
        year_dir = os.path.join(ELECTIONS_DIR, f"swdb_{year}")
        if not os.path.exists(year_dir):
            print(f"  {year}: NOT FOUND")
            continue

        files = os.listdir(year_dir)
        csv_files = [f for f in files if f.lower().endswith(".csv")]

        print(f"  {year}: {len(csv_files)} CSVs")

        for csv_f in csv_files[:1]:
            try:
                df = pd.read_csv(os.path.join(year_dir, csv_f), nrows=5,
                                 low_memory=False)
                cols = df.columns.str.upper().tolist()
                # Look for presidential and house columns
                pres_cols = [c for c in cols if "PRS" in c or "PRES" in c]
                house_cols = [c for c in cols if c.startswith("USR") or "USREP" in c]
                prec_col = next((c for c in cols if c in ["SRPREC", "PRECINCT"]), "?")
                print(f"    Precinct col: {prec_col}, "
                      f"Pres cols: {len(pres_cols)}, House cols: {len(house_cols)}")
                print(f"    First 10 cols: {cols[:10]}")
            except Exception as e:
                print(f"    Error reading {csv_f}: {e}")

    # Verify block maps
    print("\n  Block mappings:")
    for year in sorted(SWDB_BLK_MAP_URLS.keys()):
        path = os.path.join(CROSSWALK_DIR, f"sr_blk_map_{year}.csv")
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            df = pd.read_csv(path, nrows=3, dtype=str)
            print(f"  {year}: {size_mb:.1f} MB, cols={list(df.columns[:5])}")
        else:
            print(f"  {year}: NOT FOUND")


def main():
    print("=" * 60)
    print("CA Step 2: Download Election Data & Crosswalk")
    print("=" * 60)

    os.makedirs(ELECTIONS_DIR, exist_ok=True)
    os.makedirs(CROSSWALK_DIR, exist_ok=True)

    # A. SWDB precinct-level election returns
    download_swdb_elections()

    # B. SWDB precinct-to-block mappings
    download_blk_maps()

    # C. Fekrazad tract-level votes (validation)
    download_fekrazad()

    # Verify
    verify_elections()

    print(f"\nStep 2 complete.")
    print(f"  Elections: {ELECTIONS_DIR}")
    print(f"  Crosswalk: {CROSSWALK_DIR}")


if __name__ == "__main__":
    main()

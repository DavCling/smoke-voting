#!/usr/bin/env python3
"""
Download NOAA HMS (Hazard Mapping System) smoke plume shapefiles and Census
county boundaries for the HMS robustness check.

HMS source: NOAA NESDIS annual bundles of satellite-observed smoke polygons.
County boundaries: Census TIGER/Line generalized (1:500k).

Downloads are idempotent — existing files are skipped.

Usage:
    python scripts/hms_download_smoke_data.py
"""

import os
import zipfile
import requests
import geopandas as gpd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HMS_DIR = os.path.join(BASE_DIR, "data", "hms", "shapefiles")
COUNTY_DIR = os.path.join(BASE_DIR, "data", "hms", "counties")

# Presidential election years covered by Childs et al. smoke data
YEARS = [2008, 2012, 2016, 2020]

HMS_URL_TEMPLATE = (
    "https://satepsanone.nesdis.noaa.gov/pub/FIRE/web/HMS/"
    "Smoke_Polygons/Shapefile/Annual_Bundles/hms_smoke{year}.zip"
)

COUNTY_URL = (
    "https://www2.census.gov/geo/tiger/GENZ2020/shp/"
    "cb_2020_us_county_500k.zip"
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


def extract_zip(zip_path, dest_dir):
    """Extract a ZIP file if not already extracted."""
    # Check if any shapefile already exists in dest
    existing = [f for f in os.listdir(dest_dir) if f.endswith(".shp")] if os.path.isdir(dest_dir) else []
    if existing:
        print(f"  Already extracted: {len(existing)} shapefile(s) in {os.path.basename(dest_dir)}")
        return

    print(f"  Extracting {os.path.basename(zip_path)}...")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"    Extracted {len(os.listdir(dest_dir))} files")


def download_hms_shapefiles():
    """Download HMS annual smoke plume shapefiles."""
    print("\n--- HMS Smoke Plume Shapefiles ---")
    os.makedirs(HMS_DIR, exist_ok=True)

    for year in YEARS:
        url = HMS_URL_TEMPLATE.format(year=year)
        zip_path = os.path.join(HMS_DIR, f"hms_smoke{year}.zip")
        extract_dir = os.path.join(HMS_DIR, str(year))

        download_file(url, zip_path, label=f"HMS {year}")
        extract_zip(zip_path, extract_dir)


def download_county_boundaries():
    """Download Census county boundary shapefile."""
    print("\n--- Census County Boundaries (500k) ---")
    os.makedirs(COUNTY_DIR, exist_ok=True)

    zip_path = os.path.join(COUNTY_DIR, "cb_2020_us_county_500k.zip")
    download_file(COUNTY_URL, zip_path, label="County boundaries")
    extract_zip(zip_path, COUNTY_DIR)


def verify_hms(year):
    """Load and verify one HMS annual shapefile."""
    extract_dir = os.path.join(HMS_DIR, str(year))
    shp_files = [f for f in os.listdir(extract_dir) if f.endswith(".shp")]
    if not shp_files:
        print(f"  WARNING: No shapefile found for {year}")
        return

    shp_path = os.path.join(extract_dir, shp_files[0])
    # Some HMS shapefiles have invalid geometries (unclosed rings)
    gdf = gpd.read_file(shp_path, engine="pyogrio", on_invalid="ignore")
    print(f"\n  HMS {year}:")
    print(f"    Polygons: {len(gdf):,}")
    print(f"    CRS: {gdf.crs}")
    print(f"    Columns: {list(gdf.columns)}")

    # Density distribution
    if "Density" in gdf.columns:
        density_counts = gdf["Density"].value_counts()
        print(f"    Density distribution:")
        for d, c in density_counts.items():
            print(f"      {d}: {c:,} ({100*c/len(gdf):.1f}%)")

    # Date range — Start column format is "YYYYddd HHmm" (Julian day of year)
    if "Start" in gdf.columns:
        gdf["_date"] = pd.to_datetime(gdf["Start"].str[:7], format="%Y%j")
        print(f"    Date range: {gdf['_date'].min().date()} to {gdf['_date'].max().date()}")


def verify_counties():
    """Load and verify county boundary shapefile."""
    shp_files = [f for f in os.listdir(COUNTY_DIR) if f.endswith(".shp")]
    if not shp_files:
        print("  WARNING: No county shapefile found")
        return

    shp_path = os.path.join(COUNTY_DIR, shp_files[0])
    gdf = gpd.read_file(shp_path)
    print(f"\n  County boundaries:")
    print(f"    Counties: {len(gdf):,}")
    print(f"    CRS: {gdf.crs}")
    print(f"    Columns: {list(gdf.columns)}")

    # Filter to states
    if "STATEFP" in gdf.columns:
        states = gdf[gdf["STATEFP"].astype(int) <= 56]
        print(f"    CONUS+AK+HI counties (STATEFP <= 56): {len(states):,}")


def main():
    import pandas as pd  # noqa: F811 — only for date parsing in verify

    print("=" * 60)
    print("Download HMS Smoke Plume Data & County Boundaries")
    print("=" * 60)

    download_hms_shapefiles()
    download_county_boundaries()

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    for year in YEARS:
        verify_hms(year)
    verify_counties()

    print("\nDownload complete.")


if __name__ == "__main__":
    import pandas as pd  # noqa: F811
    main()

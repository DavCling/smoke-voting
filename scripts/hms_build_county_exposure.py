#!/usr/bin/env python3
"""
Build county-level HMS smoke exposure panel from HMS smoke plume shapefiles.

Approach (standard in HMS literature, e.g., Borgschulte et al. 2022):
  1. Compute centroid for each county polygon
  2. For each date in the pre-election window, spatial join centroids against
     HMS smoke polygons
  3. County centroid within any smoke polygon → smoke=1; take max density

Outputs:
  - data/hms/hms_county_daily.csv        — county-day exposure
  - output/hms_smoke_exposure.parquet     — county-year panel with windowed variables
"""

import os
import sys
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HMS_DIR = os.path.join(BASE_DIR, "data", "hms", "shapefiles")
COUNTY_DIR = os.path.join(BASE_DIR, "data", "hms", "counties")
DAILY_OUT = os.path.join(BASE_DIR, "data", "hms", "hms_county_daily.csv")
PANEL_OUT = os.path.join(BASE_DIR, "output", "hms_smoke_exposure.parquet")

# Election dates (must match build_smoke_analysis.py)
ELECTION_DATES = {
    2008: "2008-11-04",
    2012: "2012-11-06",
    2016: "2016-11-08",
    2020: "2020-11-03",
}

# Density mapping: text → numeric (Unspecified treated conservatively as Light)
DENSITY_MAP = {
    "Light": 1,
    "Medium": 2,
    "Heavy": 3,
    "Unspecified": 1,
}


def load_counties():
    """Load county centroids from Census TIGER shapefile."""
    print("Loading county boundaries...")
    shp_files = [f for f in os.listdir(COUNTY_DIR) if f.endswith(".shp")]
    if not shp_files:
        print("ERROR: No county shapefile found in", COUNTY_DIR)
        sys.exit(1)

    counties = gpd.read_file(os.path.join(COUNTY_DIR, shp_files[0]))

    # Filter to states (FIPS <= 56, exclude territories)
    counties = counties[counties["STATEFP"].astype(int) <= 56].copy()
    counties["fips"] = counties["STATEFP"] + counties["COUNTYFP"]
    print(f"  {len(counties):,} counties loaded (CRS: {counties.crs})")

    # Reproject to WGS84 to match HMS
    if counties.crs and counties.crs.to_epsg() != 4326:
        counties = counties.to_crs(epsg=4326)
        print(f"  Reprojected to EPSG:4326")

    # Compute centroids
    centroids = counties[["fips"]].copy()
    centroids["geometry"] = counties.geometry.centroid
    centroids = gpd.GeoDataFrame(centroids, crs="EPSG:4326")
    print(f"  {len(centroids):,} county centroids computed")

    return centroids


def load_hms_year(year):
    """Load HMS annual smoke shapefile for one year."""
    extract_dir = os.path.join(HMS_DIR, str(year))
    shp_files = [f for f in os.listdir(extract_dir) if f.endswith(".shp")]
    if not shp_files:
        print(f"  WARNING: No shapefile for {year}")
        return None

    # Some HMS shapefiles have invalid geometries (unclosed rings)
    gdf = gpd.read_file(os.path.join(extract_dir, shp_files[0]),
                        engine="pyogrio", on_invalid="ignore")

    # Ensure WGS84
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Parse dates — HMS Start column format is "YYYYddd HHmm" (Julian day of year)
    if "Start" in gdf.columns:
        gdf["date"] = pd.to_datetime(gdf["Start"].str[:7], format="%Y%j")
    elif all(c in gdf.columns for c in ["Year", "Month", "Day"]):
        gdf["date"] = pd.to_datetime(
            gdf["Year"].astype(str) + "-" +
            gdf["Month"].astype(str).str.zfill(2) + "-" +
            gdf["Day"].astype(str).str.zfill(2)
        )
    else:
        print(f"  WARNING: Cannot parse dates for {year}. Columns: {list(gdf.columns)}")
        return None

    # Map density to numeric
    if "Density" in gdf.columns:
        gdf["density_num"] = gdf["Density"].map(DENSITY_MAP).fillna(1).astype(int)
    else:
        gdf["density_num"] = 1  # default to Light if no density info

    print(f"  HMS {year}: {len(gdf):,} polygons, "
          f"{gdf['date'].min().date()} to {gdf['date'].max().date()}")

    return gdf[["date", "density_num", "geometry"]]


def build_daily_exposure(centroids, hms_gdf, year, edate_str):
    """Build county-day smoke exposure for one election year.

    For each date from June 1 to Election Day, spatial join county centroids
    against HMS smoke polygons active on that date.
    """
    edate = pd.Timestamp(edate_str)
    season_start = pd.Timestamp(f"{year}-06-01")

    # Generate all dates in the broadest window (season = Jun 1 to Election Day)
    all_dates = pd.date_range(season_start, edate, freq="D")
    print(f"    Window: {season_start.date()} to {edate.date()} ({len(all_dates)} days)")

    records = []
    n_smoke_days = 0
    fips_list = centroids["fips"].values

    for date in all_dates:
        # Get HMS polygons for this date
        day_polygons = hms_gdf[hms_gdf["date"] == date]

        if len(day_polygons) == 0:
            # No smoke polygons this day — all counties get 0
            day_df = pd.DataFrame({
                "fips": fips_list, "date": date,
                "hms_smoke": 0, "hms_density_max": 0,
            })
            records.append(day_df)
            continue

        # Spatial join: which centroids fall within smoke polygons
        joined = gpd.sjoin(centroids, day_polygons, predicate="within", how="left")

        # Aggregate: max density per county (a centroid can be in multiple polygons)
        county_max = joined.groupby("fips")["density_num"].max().reset_index()
        county_max["density_num"] = county_max["density_num"].fillna(0).astype(int)
        county_max["hms_smoke"] = (county_max["density_num"] > 0).astype(int)
        county_max["date"] = date
        county_max = county_max.rename(columns={"density_num": "hms_density_max"})

        n_smoke_days += (county_max["hms_smoke"] > 0).sum()
        records.append(county_max[["fips", "date", "hms_smoke", "hms_density_max"]])

    daily = pd.concat(records, ignore_index=True)

    total_obs = len(centroids) * len(all_dates)
    smoke_frac = n_smoke_days / total_obs if total_obs > 0 else 0
    print(f"    {len(daily):,} county-day records, "
          f"{n_smoke_days:,} smoke exposures ({100*smoke_frac:.2f}%)")

    return daily


def aggregate_windows(daily_df, year, edate_str):
    """Aggregate county-day exposure into pre-election windows.

    Parallels compute_smoke_exposure() in build_smoke_analysis.py.
    """
    edate = pd.Timestamp(edate_str)

    windows = {
        "7d": (edate - timedelta(days=7), edate),
        "30d": (edate - timedelta(days=30), edate),
        "60d": (edate - timedelta(days=60), edate),
        "90d": (edate - timedelta(days=90), edate),
        "season": (pd.Timestamp(f"{year}-06-01"), edate),
    }

    result_df = daily_df[["fips"]].drop_duplicates().copy()
    result_df["year"] = year

    for label, (start, end) in windows.items():
        w = daily_df[(daily_df["date"] >= start) & (daily_df["date"] <= end)]
        n_days = (end - start).days + 1  # total calendar days in window

        grp = w.groupby("fips")

        agg = grp.agg(
            hms_smoke_days=("hms_smoke", "sum"),
            hms_density_max_daily=("hms_density_max", "max"),
            hms_density_mean=("hms_density_max", "mean"),
            hms_frac_days=("hms_smoke", lambda x: x.sum() / n_days),
            hms_frac_medium=("hms_density_max", lambda x: (x >= 2).sum() / n_days),
            hms_frac_heavy=("hms_density_max", lambda x: (x >= 3).sum() / n_days),
        ).rename(columns={
            "hms_smoke_days": f"hms_smoke_days_{label}",
            "hms_density_max_daily": f"hms_density_max_{label}",
            "hms_density_mean": f"hms_density_mean_{label}",
            "hms_frac_days": f"hms_frac_days_{label}",
            "hms_frac_medium": f"hms_frac_medium_{label}",
            "hms_frac_heavy": f"hms_frac_heavy_{label}",
        })

        result_df = result_df.merge(agg, on="fips", how="left")

    result_df = result_df.fillna(0)
    return result_df


def main():
    print("=" * 60)
    print("Build HMS County-Level Smoke Exposure Panel")
    print("=" * 60)

    centroids = load_counties()

    # Build daily exposure for each election year
    all_daily = []
    all_panels = []

    for year, edate_str in sorted(ELECTION_DATES.items()):
        print(f"\n--- {year} election ---")
        hms_gdf = load_hms_year(year)
        if hms_gdf is None:
            continue

        daily = build_daily_exposure(centroids, hms_gdf, year, edate_str)
        all_daily.append(daily)

        panel = aggregate_windows(daily, year, edate_str)
        all_panels.append(panel)
        print(f"    Panel: {len(panel):,} counties")

        # Summary stats
        for w in ["7d", "30d", "season"]:
            col = f"hms_smoke_days_{w}"
            if col in panel.columns:
                print(f"    {w}: mean={panel[col].mean():.1f} smoke days, "
                      f"max={panel[col].max():.0f}")

    # Save daily
    print(f"\n--- Saving daily exposure ---")
    daily_all = pd.concat(all_daily, ignore_index=True)
    os.makedirs(os.path.dirname(DAILY_OUT), exist_ok=True)
    daily_all.to_csv(DAILY_OUT, index=False)
    print(f"  Saved: {DAILY_OUT}")
    print(f"  {len(daily_all):,} rows, {daily_all['fips'].nunique():,} counties, "
          f"{daily_all['date'].nunique():,} unique dates")

    # Save panel
    print(f"\n--- Saving panel ---")
    panel_all = pd.concat(all_panels, ignore_index=True)

    # Add state_fips and state_year for FE
    panel_all["state_fips"] = panel_all["fips"].str[:2]
    panel_all["state_year"] = panel_all["state_fips"] + "_" + panel_all["year"].astype(str)

    os.makedirs(os.path.dirname(PANEL_OUT), exist_ok=True)
    panel_all.to_parquet(PANEL_OUT, index=False)
    print(f"  Saved: {PANEL_OUT}")
    print(f"  {len(panel_all):,} county-year rows")
    print(f"  Columns: {list(panel_all.columns)}")

    # Summary
    print(f"\n" + "=" * 60)
    print("HMS EXPOSURE SUMMARY")
    print("=" * 60)
    for year in sorted(ELECTION_DATES.keys()):
        yr_df = panel_all[panel_all["year"] == year]
        if len(yr_df) == 0:
            continue
        print(f"\n  {year}:")
        print(f"    Counties: {len(yr_df):,}")
        for w in ["30d", "season"]:
            frac_col = f"hms_frac_days_{w}"
            days_col = f"hms_smoke_days_{w}"
            if frac_col in yr_df.columns:
                nonzero = (yr_df[frac_col] > 0).sum()
                print(f"    {w}: {yr_df[days_col].mean():.1f} mean smoke days, "
                      f"{nonzero:,} counties with any smoke ({100*nonzero/len(yr_df):.1f}%)")

    print("\nBuild complete.")


if __name__ == "__main__":
    main()

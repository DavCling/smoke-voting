#!/usr/bin/env python3
"""
Download Fekrazad (2025) tract-level presidential votes from Harvard Dataverse
for all CONUS states. Combine into a single national CSV.

Source: Fekrazad (2025) "Precinct-Level Election Results"
        Harvard Dataverse doi:10.7910/DVN/Z8TSH3
        Pre-allocated tract-level votes using RLCR (Registration-Linked
        Crosswalk with Regression) method for 2016 and 2020 presidential.

Each state is a ZIP file containing Census Tract CSVs with columns like:
  2016: tract_GEOID, tract_population, G16PREDCli, G16PRERTru, ...
  2020: tract_GEOID, tract_population, G20PREDBID, G20PRERTRU, ...

Output:
  data/national_tracts/elections/
    fekrazad_raw/          — Per-state ZIPs from Dataverse
    tract_presidential.csv — Combined national tract-level votes (2016+2020)
"""

import os
import io
import re
import time
import zipfile
import requests
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELECTIONS_DIR = os.path.join(BASE_DIR, "data", "national_tracts", "elections")
RAW_DIR = os.path.join(ELECTIONS_DIR, "fekrazad_raw")
OUT_FILE = os.path.join(ELECTIONS_DIR, "tract_presidential.csv")

# Crosswalk: 2020 Census tracts → 2010 Census tracts (area-weighted)
CROSSWALK_FILE = os.path.join(BASE_DIR, "data", "california", "crosswalk",
                              "tract_2020_to_2010_natl.txt")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

# CONUS state FIPS codes (exclude AK=02, HI=15, and territories)
CONUS_FIPS = [
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
]

# Incumbent party for each presidential election
INCUMBENT_PARTY = {
    2016: "DEMOCRAT",    # Obama (D) was incumbent
    2020: "REPUBLICAN",  # Trump (R) was incumbent
}

# Dataverse dataset DOI
DATAVERSE_DOI = "10.7910/DVN/Z8TSH3"
DATAVERSE_API = "https://dataverse.harvard.edu/api"


def get_dataverse_file_list():
    """Query Dataverse API to get file listing with download IDs."""
    url = f"{DATAVERSE_API}/datasets/:persistentId?persistentId=doi:{DATAVERSE_DOI}"
    print(f"  Querying Dataverse: {url[:80]}...")
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    data = resp.json()["data"]
    files = data["latestVersion"]["files"]
    print(f"  Found {len(files)} files in dataset")
    return files


def find_state_zip_files(files):
    """Map state FIPS codes to Dataverse file IDs."""
    # File names like "010 AL.zip", "040 AZ.zip", "060 CA.zip"
    state_files = {}
    for f in files:
        label = f.get("label", "")
        file_id = f.get("dataFile", {}).get("id")
        if not label.endswith(".zip") or file_id is None:
            continue
        # Extract FIPS from filename: "010 AL.zip" → "01", "100 DE.zip" → "10"
        # 3-digit codes are FIPS * 10 (trailing zero), e.g. 010=AL, 100=DE, 530=WA
        match = re.match(r"^(\d{3})\s+\w+\.zip$", label)
        if match:
            fips_raw = match.group(1)
            fips = str(int(fips_raw) // 10).zfill(2)
            if fips in CONUS_FIPS:
                state_files[fips] = {"file_id": file_id, "label": label}

    print(f"  Matched {len(state_files)}/{len(CONUS_FIPS)} CONUS states")
    missing = set(CONUS_FIPS) - set(state_files.keys())
    if missing:
        print(f"  Missing states: {sorted(missing)}")
    return state_files


def download_state_zip(file_id, label, dest_path):
    """Download a single state ZIP from Dataverse."""
    if os.path.exists(dest_path):
        return True

    url = f"{DATAVERSE_API}/access/datafile/{file_id}"
    print(f"    Downloading {label}...")
    try:
        resp = requests.get(url, headers=HEADERS, stream=True, timeout=300,
                            allow_redirects=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    ERROR: {e}")
        return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"    Saved ({size_mb:.1f} MB)")
    return True


def extract_tract_csv_from_zip(zip_path, year):
    """Extract tract-level RLCR CSV from a state ZIP for a given year.

    Files are at: Main Method/Census Tracts/tracts-{year}-RLCR.csv
    """
    yy = str(year)[-2:]
    target_name = f"tracts-{year}-RLCR.csv"

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(target_name):
                    with zf.open(name) as csvfile:
                        df = pd.read_csv(io.TextIOWrapper(csvfile),
                                         dtype={"tract_GEOID": str})
                    return df
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"    ERROR reading {zip_path}: {e}")
    return None


def parse_votes(df, year):
    """Parse vote columns from Fekrazad tract CSV.

    Column naming pattern:
      2016: G16PREDCli (DEM Clinton), G16PRERTru (REP Trump)
      2020: G20PREDBID (DEM Biden), G20PRERTRU (REP Trump)

    Strategy: match G{YY}PRED* for DEM and G{YY}PRER* for REP.
    """
    yy = str(year)[-2:]
    cols = df.columns.tolist()

    # Find DEM column(s): G{YY}PRED*
    dem_cols = [c for c in cols if re.match(rf"G{yy}PRED\w+", c, re.IGNORECASE)]
    rep_cols = [c for c in cols if re.match(rf"G{yy}PRER\w+", c, re.IGNORECASE)]

    if not dem_cols:
        print(f"    WARNING: No DEM column found for {year}. Columns: {cols[:15]}")
        return None
    if not rep_cols:
        print(f"    WARNING: No REP column found for {year}. Columns: {cols[:15]}")
        return None

    # Sum all DEM and REP columns (usually just one each)
    result = pd.DataFrame()
    result["tract_GEOID"] = df["tract_GEOID"].astype(str).str.zfill(11)
    result["year"] = year
    result["dem_votes"] = df[dem_cols].sum(axis=1)
    result["rep_votes"] = df[rep_cols].sum(axis=1)
    result["total_votes"] = result["dem_votes"] + result["rep_votes"]

    if "tract_population" in df.columns:
        result["tract_population"] = pd.to_numeric(df["tract_population"],
                                                    errors="coerce")

    return result


def load_tract_2020_to_2010_map():
    """Load Census 2020-to-2010 tract relationship file with area weights."""
    if not os.path.exists(CROSSWALK_FILE):
        print(f"  WARNING: Crosswalk file not found: {CROSSWALK_FILE}")
        return None

    df = pd.read_csv(CROSSWALK_FILE, sep="|", dtype=str)
    # Compute area weights within each 2020 tract
    df["AREALAND_PART"] = pd.to_numeric(df["AREALAND_PART"], errors="coerce").fillna(0)
    total_area = df.groupby("GEOID_TRACT_20")["AREALAND_PART"].transform("sum")
    df["area_weight"] = np.where(total_area > 0, df["AREALAND_PART"] / total_area, 0)

    result = df[["GEOID_TRACT_20", "GEOID_TRACT_10", "area_weight"]].copy()
    result = result.rename(columns={
        "GEOID_TRACT_20": "geoid_2020",
        "GEOID_TRACT_10": "geoid_2010",
    })
    result = result[result["area_weight"] > 0].copy()

    print(f"  Loaded crosswalk: {result['geoid_2020'].nunique()} 2020 tracts "
          f"-> {result['geoid_2010'].nunique()} 2010 tracts")
    return result


def convert_tracts_2020_to_2010(tract_votes, tract_map):
    """Convert 2020 election tract data from 2020 Census GEOIDs to 2010 GEOIDs.

    Uses area-weighted allocation: votes are split proportionally across
    overlapping 2010 tracts based on land area overlap.
    Only applies to year=2020 rows (2016 data already uses 2010 GEOIDs).
    """
    if tract_map is None:
        print("  WARNING: No crosswalk, keeping 2020 GEOIDs for 2020 data")
        return tract_votes

    # Split 2016 (already 2010 GEOIDs) and 2020 (needs conversion)
    mask_2020 = tract_votes["year"] == 2020
    df_2016 = tract_votes[~mask_2020].copy()
    df_2020 = tract_votes[mask_2020].copy()

    if len(df_2020) == 0:
        return tract_votes

    n_before = df_2020["tract_GEOID"].nunique()

    # Merge with mapping
    merged = df_2020.merge(
        tract_map.rename(columns={"geoid_2020": "tract_GEOID"}),
        on="tract_GEOID",
        how="left",
    )

    # Unmapped tracts: keep as-is
    unmapped = merged["geoid_2010"].isna()
    if unmapped.any():
        n_unmapped = merged.loc[unmapped, "tract_GEOID"].nunique()
        print(f"    {n_unmapped} 2020 tracts have no mapping, keeping original GEOID")
        merged.loc[unmapped, "geoid_2010"] = merged.loc[unmapped, "tract_GEOID"]
        merged.loc[unmapped, "area_weight"] = 1.0

    # Apply area weights to vote columns
    for col in ["dem_votes", "rep_votes", "total_votes"]:
        merged[col] = merged[col] * merged["area_weight"]

    if "tract_population" in merged.columns:
        merged["tract_population"] = merged["tract_population"] * merged["area_weight"]

    # Replace GEOID with 2010 GEOID
    merged["tract_GEOID"] = merged["geoid_2010"]

    # Aggregate by 2010 GEOID
    agg_cols = {col: "sum" for col in ["dem_votes", "rep_votes", "total_votes"]}
    if "tract_population" in merged.columns:
        agg_cols["tract_population"] = "sum"
    result = merged.groupby(["tract_GEOID", "year"]).agg(agg_cols).reset_index()

    # Filter to CONUS only
    result = result[result["tract_GEOID"].str[:2].isin(CONUS_FIPS)].copy()

    n_after = result["tract_GEOID"].nunique()
    print(f"    2020 tracts: {n_before} (2020 Census) -> {n_after} (2010 Census)")

    # Recombine
    combined = pd.concat([df_2016, result], ignore_index=True)
    return combined


def compute_outcomes(df):
    """Compute derived outcome variables."""
    # Two-party vote share
    df["dem_vote_share"] = np.where(
        df["total_votes"] > 0,
        df["dem_votes"] / df["total_votes"],
        np.nan,
    )

    # Incumbent vote share
    for year, party in INCUMBENT_PARTY.items():
        mask = df["year"] == year
        if party == "DEMOCRAT":
            df.loc[mask, "incumbent_vote_share"] = df.loc[mask, "dem_vote_share"]
        else:
            df.loc[mask, "incumbent_vote_share"] = np.where(
                df.loc[mask, "total_votes"] > 0,
                df.loc[mask, "rep_votes"] / df.loc[mask, "total_votes"],
                np.nan,
            )

    # Log total votes
    df["log_total_votes"] = np.where(
        df["total_votes"] > 0,
        np.log(df["total_votes"]),
        np.nan,
    )

    # Uncontested flag (only one party received votes)
    df["uncontested"] = (df["dem_votes"] <= 0) | (df["rep_votes"] <= 0)

    return df


def download_and_process():
    """Main download and processing pipeline."""
    if os.path.exists(OUT_FILE):
        df = pd.read_csv(OUT_FILE, dtype={"tract_GEOID": str})
        print(f"Already exists: {OUT_FILE}")
        print(f"  {len(df):,} rows, {df['tract_GEOID'].nunique():,} tracts, "
              f"years: {sorted(df['year'].unique())}")
        return

    # 1. Get file listing from Dataverse
    print("\n1. Querying Dataverse for file listing...")
    files = get_dataverse_file_list()
    state_files = find_state_zip_files(files)

    if not state_files:
        print("ERROR: No state files found on Dataverse!")
        return

    # 2. Download state ZIPs
    print(f"\n2. Downloading {len(state_files)} state ZIPs...")
    os.makedirs(RAW_DIR, exist_ok=True)

    for fips in sorted(state_files.keys()):
        info = state_files[fips]
        dest = os.path.join(RAW_DIR, info["label"])
        if not os.path.exists(dest):
            download_state_zip(info["file_id"], info["label"], dest)
            time.sleep(0.5)  # Rate limit

    # 3. Extract and parse tract-level votes
    print("\n3. Extracting tract-level votes...")
    all_dfs = []

    for fips in sorted(state_files.keys()):
        info = state_files[fips]
        zip_path = os.path.join(RAW_DIR, info["label"])
        if not os.path.exists(zip_path):
            print(f"  SKIP {fips}: ZIP not found")
            continue

        for year in [2016, 2020]:
            raw_df = extract_tract_csv_from_zip(zip_path, year)
            if raw_df is None:
                print(f"    {fips} {year}: no tract CSV found")
                continue

            parsed = parse_votes(raw_df, year)
            if parsed is None:
                continue

            n_tracts = parsed["tract_GEOID"].nunique()
            total = parsed["total_votes"].sum()
            print(f"  {fips} {year}: {n_tracts:,} tracts, "
                  f"{total:,.0f} total votes")
            all_dfs.append(parsed)

    if not all_dfs:
        print("ERROR: No data extracted!")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Combined: {len(combined):,} rows, "
          f"{combined['tract_GEOID'].nunique():,} unique tracts")

    # 4. Convert 2020 tracts to 2010 Census GEOIDs
    print("\n4. Converting 2020 Census tracts to 2010 Census GEOIDs...")
    tract_map = load_tract_2020_to_2010_map()
    combined = convert_tracts_2020_to_2010(combined, tract_map)

    # 5. Compute derived outcomes
    print("\n5. Computing derived outcomes...")
    combined = compute_outcomes(combined)

    # 6. Save
    os.makedirs(ELECTIONS_DIR, exist_ok=True)
    combined.to_csv(OUT_FILE, index=False)
    print(f"\n  Saved: {OUT_FILE}")
    print(f"  {len(combined):,} rows, {combined['tract_GEOID'].nunique():,} tracts")


def verify():
    """Verify downloaded election data."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    if not os.path.exists(OUT_FILE):
        print("  Output file not found!")
        return

    df = pd.read_csv(OUT_FILE, dtype={"tract_GEOID": str})

    # National totals
    for year in sorted(df["year"].unique()):
        yr_df = df[df["year"] == year]
        dem_total = yr_df["dem_votes"].sum()
        rep_total = yr_df["rep_votes"].sum()
        total = yr_df["total_votes"].sum()
        n_tracts = yr_df["tract_GEOID"].nunique()
        n_states = yr_df["tract_GEOID"].str[:2].nunique()
        print(f"\n  {year}:")
        print(f"    Tracts: {n_tracts:,} across {n_states} states")
        print(f"    DEM: {dem_total:,.0f}   REP: {rep_total:,.0f}   "
              f"Total: {total:,.0f}")
        print(f"    DEM share: {dem_total / total:.3f}")

    # Per-state tract counts
    print("\n  Tracts per state (2016):")
    df_2016 = df[df["year"] == 2016]
    state_counts = df_2016.groupby(df_2016["tract_GEOID"].str[:2]).size()
    for fips, count in state_counts.items():
        print(f"    {fips}: {count:,}")


def main():
    print("=" * 60)
    print("National Tract-Level: Download Election Data (Fekrazad 2025)")
    print("=" * 60)

    os.makedirs(ELECTIONS_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    download_and_process()
    verify()

    print(f"\nDone. Output: {OUT_FILE}")


if __name__ == "__main__":
    main()

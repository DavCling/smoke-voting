#!/usr/bin/env python3
"""
Build precinct-to-tract crosswalk and allocate CA election results to census tracts.

Method: Uses SWDB precinct-to-block mapping files (sr_blk_map) to allocate
precinct-level votes to census blocks, then aggregates blocks up to tracts.
Census blocks nest perfectly within tracts (tract GEOID = first 11 digits of
15-digit block GEOID).

Key matching: Both SOV and block map files share the SRPREC_KEY column
(FIPS county code + precinct number), enabling direct joins.

Block map share column: PCTSRPREC = percentage of precinct's registered
voters in each block (0-100 scale, divided by 100 to get fraction).

Handles California's top-two primary (post-2012): if a House race has
two candidates from the same party (e.g., two Democrats), the DEM vote
share is undefined → flagged as same_party_race.

SWDB column conventions:
  Presidential: PRSDEM/PRSREP (2006-2008), PRSDEM01/PRSREP01 (2012+)
  House:        CNGDEM/CNGREP (2006-2008), CNGDEM01+02/CNGREP01+02 (2012+)
  District:     CDDIST (congressional district number in all years)

Input:
  data/california/elections/swdb_<year>/           — SWDB precinct SOV data
  data/california/crosswalk/sr_blk_map_<year>.csv  — Precinct-to-block mapping

Output:
  data/california/elections/tract_presidential.csv
  data/california/elections/tract_house.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ELECTIONS_DIR = os.path.join(BASE_DIR, "data", "california", "elections")
CROSSWALK_DIR = os.path.join(BASE_DIR, "data", "california", "crosswalk")

# Election years
PRESIDENTIAL_YEARS = [2008, 2012, 2016, 2020]
HOUSE_YEARS = list(range(2006, 2023, 2))  # 2006-2022, biennial

# Incumbent party by year
INCUMBENT_PARTY_PRES = {
    2008: "REPUBLICAN",
    2012: "DEMOCRAT",
    2016: "DEMOCRAT",
    2020: "REPUBLICAN",
}

INCUMBENT_PARTY_HOUSE = {
    2006: "REPUBLICAN",
    2008: "REPUBLICAN",
    2010: "DEMOCRAT",
    2012: "DEMOCRAT",
    2014: "DEMOCRAT",
    2016: "DEMOCRAT",
    2018: "REPUBLICAN",
    2020: "REPUBLICAN",
    2022: "DEMOCRAT",
}


def load_blk_map(year):
    """Load precinct-to-block mapping for a given year.

    Returns DataFrame with columns: srprec_key, block_key, tract_geoid, share
    """
    path = os.path.join(CROSSWALK_DIR, f"sr_blk_map_{year}.csv")
    if not os.path.exists(path):
        print(f"    Block map not found for {year}")
        return None

    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = df.columns.str.upper().str.strip()
    print(f"    Block map {year}: {len(df):,} rows")

    # Filter out rows with missing SRPREC_KEY or BLOCK_KEY
    df = df[df["SRPREC_KEY"].notna() & df["BLOCK_KEY"].notna()].copy()
    df = df[~df["SRPREC_KEY"].str.contains("nan", case=False, na=True)].copy()

    # PCTSRPREC is percentage (0-100), convert to fraction
    df["share"] = pd.to_numeric(df["PCTSRPREC"], errors="coerce").fillna(0) / 100.0

    # Derive tract GEOID from BLOCK_KEY (first 11 of 15 digits)
    df["tract_geoid"] = df["BLOCK_KEY"].str[:11]

    # Aggregate: sum shares by SRPREC_KEY → tract (blocks within a tract)
    prec_tract = (
        df.groupby(["SRPREC_KEY", "tract_geoid"])["share"]
        .sum()
        .reset_index()
        .rename(columns={"SRPREC_KEY": "srprec_key", "share": "tract_share"})
    )

    # Filter to CA tracts
    prec_tract = prec_tract[prec_tract["tract_geoid"].str.startswith("06")].copy()

    n_prec = prec_tract["srprec_key"].nunique()
    n_tract = prec_tract["tract_geoid"].nunique()
    print(f"    {n_prec:,} precincts → {n_tract:,} tracts")

    return prec_tract


def load_swdb_year(year):
    """Load SWDB precinct-level election returns for a given year."""
    year_dir = os.path.join(ELECTIONS_DIR, f"swdb_{year}")
    if not os.path.exists(year_dir):
        print(f"  SWDB data not found for {year}")
        return None

    csv_files = [f for f in os.listdir(year_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"  No CSV files found in {year_dir}")
        return None

    csv_path = os.path.join(year_dir, csv_files[0])
    print(f"  Loading SWDB {year}: {csv_files[0]}")

    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.columns = df.columns.str.upper().str.strip()

    # Basic validation
    if len(df) < 100:
        print(f"  WARNING: Only {len(df)} rows — skipping (likely county summary, not precinct)")
        return None

    print(f"    {len(df):,} precincts, {len(df.columns)} columns")
    return df


def extract_presidential_votes(swdb_df, year):
    """Extract presidential DEM/REP votes from SWDB data.

    2008: PRSDEM, PRSREP
    2012+: PRSDEM01, PRSREP01 (numbered candidates)
    """
    if swdb_df is None:
        return None

    cols = swdb_df.columns.tolist()

    # Find DEM column: try PRSDEM then PRSDEM01
    dem_col = "PRSDEM" if "PRSDEM" in cols else None
    if dem_col is None:
        dem_col = "PRSDEM01" if "PRSDEM01" in cols else None
    if dem_col is None:
        pres_cols = [c for c in cols if "PRS" in c]
        print(f"    WARNING: No presidential DEM column. Available: {pres_cols}")
        return None

    # Find REP column
    rep_col = "PRSREP" if "PRSREP" in cols else None
    if rep_col is None:
        rep_col = "PRSREP01" if "PRSREP01" in cols else None
    if rep_col is None:
        print(f"    WARNING: No presidential REP column")
        return None

    total_col = "TOTVOTE" if "TOTVOTE" in cols else None

    print(f"    Pres: DEM={dem_col}, REP={rep_col}, Total={total_col}")

    result = pd.DataFrame({
        "srprec_key": swdb_df["SRPREC_KEY"].str.strip(),
        "dem_votes": pd.to_numeric(swdb_df[dem_col], errors="coerce").fillna(0),
        "rep_votes": pd.to_numeric(swdb_df[rep_col], errors="coerce").fillna(0),
        "year": year,
    })

    if total_col:
        result["total_votes"] = pd.to_numeric(swdb_df[total_col], errors="coerce").fillna(0)
    else:
        result["total_votes"] = result["dem_votes"] + result["rep_votes"]

    print(f"    {len(result):,} precincts, "
          f"DEM={result['dem_votes'].sum():,.0f}, REP={result['rep_votes'].sum():,.0f}")

    return result


def extract_house_votes(swdb_df, year):
    """Extract US House DEM/REP votes from SWDB data.

    All years use CNG* columns + CDDIST for district:
      2006-2008: CNGDEM, CNGREP (single candidate per party)
      2012+: CNGDEM01+CNGDEM02, CNGREP01+CNGREP02 (multiple candidates, top-two primary)

    Returns one row per precinct with district assignment and party vote totals.
    """
    if swdb_df is None:
        return None

    cols = swdb_df.columns.tolist()

    if "CDDIST" not in cols:
        print(f"    WARNING: No CDDIST column for {year}")
        return None

    # Sum all DEM and REP House candidates
    dem_cols = sorted([c for c in cols if c.startswith("CNGDEM")])
    rep_cols = sorted([c for c in cols if c.startswith("CNGREP")])

    if not dem_cols and not rep_cols:
        print(f"    WARNING: No CNG House columns for {year}")
        return None

    print(f"    House: DEM cols={dem_cols}, REP cols={rep_cols}")

    # Build result
    result = pd.DataFrame({
        "srprec_key": swdb_df["SRPREC_KEY"].str.strip(),
        "cddist": swdb_df["CDDIST"].str.strip(),
        "year": year,
    })

    # Sum DEM votes across all DEM candidates
    if dem_cols:
        dem_sum = swdb_df[dem_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        result["dem_votes"] = dem_sum
    else:
        result["dem_votes"] = 0.0

    # Sum REP votes
    if rep_cols:
        rep_sum = swdb_df[rep_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        result["rep_votes"] = rep_sum
    else:
        result["rep_votes"] = 0.0

    result["total_votes"] = result["dem_votes"] + result["rep_votes"]

    # Build district ID: "06" + zero-padded district number
    result["district"] = "06" + result["cddist"].str.zfill(2)

    # Keep only precincts with any House votes
    result = result[result["total_votes"] > 0].copy()

    # Flag same-party races (top-two primary, post-2012)
    # If only DEM candidates or only REP candidates ran
    result["same_party_race"] = (result["dem_votes"] == 0) | (result["rep_votes"] == 0)

    n_districts = result["district"].nunique()
    print(f"    {n_districts} districts, {len(result):,} precinct rows, "
          f"same-party: {result['same_party_race'].sum():,}")

    return result


def allocate_votes_to_tracts(precinct_votes, blk_map, year):
    """Allocate precinct votes to tracts using SRPREC_KEY matching."""
    if blk_map is None or precinct_votes is None:
        return None

    # Merge on SRPREC_KEY
    merged = precinct_votes.merge(
        blk_map,
        on="srprec_key",
        how="inner",
    )

    n_matched = merged["srprec_key"].nunique()
    n_total = precinct_votes["srprec_key"].nunique()
    match_rate = n_matched / n_total * 100 if n_total > 0 else 0
    print(f"    Precinct match: {n_matched:,}/{n_total:,} ({match_rate:.1f}%)")

    if n_matched == 0:
        print(f"    SOV keys sample: {precinct_votes['srprec_key'].head(3).tolist()}")
        print(f"    BLK keys sample: {blk_map['srprec_key'].head(3).tolist()}")
        return None

    # Allocate votes proportionally
    for col in ["dem_votes", "rep_votes", "total_votes"]:
        merged[col] = merged[col] * merged["tract_share"]

    # Group columns
    group_cols = ["tract_geoid", "year"]
    if "district" in merged.columns:
        group_cols.append("district")

    # Aggregate to tract level
    agg_cols = {"dem_votes": "sum", "rep_votes": "sum", "total_votes": "sum"}
    if "same_party_race" in merged.columns:
        # For same_party_race, take the first value per district (same for all precincts in a district)
        group_cols_with_flag = group_cols + ["same_party_race"]
        tract_votes = merged.groupby(group_cols_with_flag, dropna=False).agg(agg_cols).reset_index()
    else:
        tract_votes = merged.groupby(group_cols, dropna=False).agg(agg_cols).reset_index()

    tract_votes.rename(columns={"tract_geoid": "GEOID"}, inplace=True)
    tract_votes = tract_votes[tract_votes["GEOID"].str.startswith("06")].copy()

    vote_pct_captured = (tract_votes["total_votes"].sum() /
                         precinct_votes["total_votes"].sum() * 100)

    print(f"    {tract_votes['GEOID'].nunique():,} tracts, "
          f"votes captured: {vote_pct_captured:.1f}%")

    return tract_votes


def build_tract_election_results(race_type="presidential"):
    """Build tract-level election results for all years."""
    years = PRESIDENTIAL_YEARS if race_type == "presidential" else HOUSE_YEARS
    incumbent_map = INCUMBENT_PARTY_PRES if race_type == "presidential" else INCUMBENT_PARTY_HOUSE

    print(f"\n{'='*60}")
    print(f"Building Tract-Level {race_type.title()} Results")
    print(f"{'='*60}")

    all_results = []

    for year in years:
        print(f"\n  --- {year} ---")

        blk_map = load_blk_map(year)
        swdb_df = load_swdb_year(year)
        if swdb_df is None:
            continue

        if race_type == "presidential":
            votes = extract_presidential_votes(swdb_df, year)
        else:
            votes = extract_house_votes(swdb_df, year)

        if votes is None:
            continue

        tract_votes = allocate_votes_to_tracts(votes, blk_map, year)

        if tract_votes is not None:
            all_results.append(tract_votes)
        else:
            print(f"    SKIPPING {year}")

    if not all_results:
        print(f"  No {race_type} results built!")
        return None

    combined = pd.concat(all_results, ignore_index=True)

    # Derived outcomes
    two_party = combined["dem_votes"] + combined["rep_votes"]
    combined["dem_vote_share"] = np.where(two_party > 0,
                                          combined["dem_votes"] / two_party, np.nan)

    combined["incumbent_party"] = combined["year"].map(incumbent_map)
    combined["incumbent_vote_share"] = np.where(
        combined["incumbent_party"] == "DEMOCRAT",
        combined["dem_vote_share"],
        1 - combined["dem_vote_share"],
    )

    combined["log_total_votes"] = np.log1p(combined["total_votes"])
    combined["uncontested"] = (combined["dem_votes"] < 1) | (combined["rep_votes"] < 1)

    # Summary
    print(f"\n  Combined: {len(combined):,} rows, "
          f"{combined['GEOID'].nunique():,} unique tracts, "
          f"{combined['year'].nunique()} years")

    return combined


def validate_against_fekrazad(tract_df):
    """Validate SWDB-derived tract totals against Fekrazad (2025) for 2016+2020."""
    fek_dir = os.path.join(CROSSWALK_DIR, "fekrazad_ca", "060 CA", "Main Method", "Census Tracts")
    if not os.path.exists(fek_dir):
        print("  Fekrazad validation data not available — skipping")
        return

    print(f"\n{'='*60}")
    print(f"Validating Against Fekrazad (2025) RLCR Method")
    print(f"{'='*60}")

    for fek_year, fek_file, dem_col, rep_col in [
        (2016, "tracts-2016-RLCR.csv", "G16PREDCli", "G16PRERTru"),
        (2020, "tracts-2020-RLCR.csv", "G20PREDBID", "G20PRERTRU"),
    ]:
        fek_path = os.path.join(fek_dir, fek_file)
        if not os.path.exists(fek_path):
            print(f"  {fek_year}: File not found")
            continue

        fek = pd.read_csv(fek_path)
        fek["GEOID"] = fek["tract_GEOID"].astype(str).str.zfill(11)
        fek["fek_dem"] = fek[dem_col]
        fek["fek_rep"] = fek[rep_col]
        fek["fek_total"] = fek["fek_dem"] + fek["fek_rep"]

        # Our data for this year
        ours = tract_df[tract_df["year"] == fek_year][["GEOID", "dem_votes", "rep_votes", "total_votes"]]

        merged = ours.merge(fek[["GEOID", "fek_dem", "fek_rep", "fek_total"]], on="GEOID", how="inner")

        if len(merged) == 0:
            print(f"  {fek_year}: No matching tracts")
            continue

        # Correlation and RMSE
        dem_corr = merged["dem_votes"].corr(merged["fek_dem"])
        rep_corr = merged["rep_votes"].corr(merged["fek_rep"])
        dem_rmse = np.sqrt(((merged["dem_votes"] - merged["fek_dem"]) ** 2).mean())

        # State totals
        our_total_dem = merged["dem_votes"].sum()
        fek_total_dem = merged["fek_dem"].sum()

        print(f"\n  {fek_year} ({len(merged):,} tracts):")
        print(f"    DEM correlation: {dem_corr:.4f}")
        print(f"    REP correlation: {rep_corr:.4f}")
        print(f"    DEM RMSE: {dem_rmse:.1f} votes/tract")
        print(f"    State DEM total — Ours: {our_total_dem:,.0f}, Fekrazad: {fek_total_dem:,.0f}, "
              f"diff: {abs(our_total_dem - fek_total_dem) / fek_total_dem * 100:.2f}%")


def main():
    print("=" * 60)
    print("CA Step 4: Build Precinct-to-Tract Crosswalk")
    print("=" * 60)

    # Presidential
    pres_out = os.path.join(ELECTIONS_DIR, "tract_presidential.csv")
    if os.path.exists(pres_out):
        print(f"\nPresidential already exists: {pres_out}")
        pres_df = pd.read_csv(pres_out)
    else:
        pres_df = build_tract_election_results("presidential")
        if pres_df is not None:
            pres_df.to_csv(pres_out, index=False)
            print(f"\n  Saved: {pres_out} ({len(pres_df):,} rows)")

    # House
    house_out = os.path.join(ELECTIONS_DIR, "tract_house.csv")
    if os.path.exists(house_out):
        print(f"\nHouse already exists: {house_out}")
    else:
        house_df = build_tract_election_results("house")
        if house_df is not None:
            house_df.to_csv(house_out, index=False)
            print(f"\n  Saved: {house_out} ({len(house_df):,} rows)")

    # Validate presidential against Fekrazad
    if pres_df is not None:
        validate_against_fekrazad(pres_df)

    print(f"\nStep 4 complete.")


if __name__ == "__main__":
    main()

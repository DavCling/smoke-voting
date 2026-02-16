#!/usr/bin/env python3
"""
Download county-level economic controls for smoke-voting analysis.

Downloads:
  A. BLS LAUS county unemployment rates (2006-2022)
  B. Census SAIPE median household income + poverty rate (2006-2022)
  C. Census Population Estimates (2006-2022)

Outputs to data/controls/:
  - bls_laus_unemployment.csv
  - census_saipe_income.csv
  - census_popest_population.csv
"""

import io
import os
import sys
import time
import requests
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "data", "controls")

YEARS = range(2006, 2023)  # 2006-2022

# Known FIPS code changes in study period
FIPS_REMAP = {
    "46113": "46102",  # Shannon Co SD → Oglala Lakota Co (2015)
    "51515": "51019",  # Bedford City VA → Bedford Co (2013)
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}


def harmonize_fips(df, fips_col="fips"):
    """Apply known FIPS code changes to ensure consistency."""
    df[fips_col] = df[fips_col].replace(FIPS_REMAP)
    return df


# ---------------------------------------------------------------------------
# A. BLS LAUS — County Unemployment Rate
# ---------------------------------------------------------------------------

def download_bls_laus():
    """Download BLS Local Area Unemployment Statistics (annual county averages)."""
    out_file = os.path.join(OUT_DIR, "bls_laus_unemployment.csv")
    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        print(f"  Already exists: {out_file} ({len(df):,} rows)")
        return

    print("\n--- BLS LAUS County Unemployment ---")
    all_data = []

    for year in YEARS:
        yy = f"{year % 100:02d}"
        url = f"https://www.bls.gov/lau/laucnty{yy}.xlsx"
        print(f"  {year}: downloading {url}")

        try:
            resp = requests.get(url, headers=HEADERS, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"    ERROR: {e}")
            continue

        # BLS Excel: title rows then data. Auto-detect structure.
        try:
            df = pd.read_excel(io.BytesIO(resp.content), header=None)
        except Exception as e:
            print(f"    ERROR parsing Excel: {e}")
            continue

        # Find the header row: look for a row containing "LAUS" in first column
        header_idx = None
        for i in range(min(10, len(df))):
            val = str(df.iloc[i, 0]).upper()
            if "LAUS" in val or "CODE" in val:
                header_idx = i
                break

        if header_idx is not None:
            # Skip everything up to and including the header row
            df = df.iloc[header_idx + 1:].reset_index(drop=True)

        # The last column should be unemployment rate
        # Columns are: LAUS Code, State FIPS, County FIPS, Name, Year, [blank?],
        #              Labor Force, Employed, Unemployed, Unemployment Rate
        ncol = df.shape[1]
        # State FIPS is always col 1, County FIPS is col 2
        # Unemployment rate is always the last column
        df = df.dropna(subset=[1])  # Drop footer rows

        state_col = df.iloc[:, 1].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
        county_col = df.iloc[:, 2].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(3)
        unemp_col = pd.to_numeric(df.iloc[:, ncol - 1], errors="coerce")

        df = pd.DataFrame({
            "state_fips": state_col,
            "county_fips": county_col,
            "unemployment_rate": unemp_col,
        })

        # Drop state-level aggregates (county FIPS = "000") and invalid rows
        df = df[df["county_fips"] != "000"].copy()
        df = df.dropna(subset=["unemployment_rate"])

        df["fips"] = df["state_fips"] + df["county_fips"]
        df["year"] = year

        n_counties = df["fips"].nunique()
        print(f"    {n_counties:,} counties, mean unemp = {df['unemployment_rate'].mean():.1f}%")
        all_data.append(df[["fips", "year", "unemployment_rate"]])

        time.sleep(0.3)

    if not all_data:
        print("  ERROR: No BLS data downloaded.")
        return

    result = pd.concat(all_data, ignore_index=True)
    result = harmonize_fips(result)
    result.to_csv(out_file, index=False)
    print(f"  Saved: {out_file} ({len(result):,} rows, {result['fips'].nunique():,} counties)")


# ---------------------------------------------------------------------------
# B. Census SAIPE — Median Household Income + Poverty Rate
# ---------------------------------------------------------------------------

def download_census_saipe():
    """Download Census Small Area Income and Poverty Estimates via API."""
    out_file = os.path.join(OUT_DIR, "census_saipe_income.csv")
    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        print(f"  Already exists: {out_file} ({len(df):,} rows)")
        return

    print("\n--- Census SAIPE (Income + Poverty) ---")
    all_data = []

    for year in YEARS:
        url = (
            f"https://api.census.gov/data/timeseries/poverty/saipe"
            f"?get=SAEMHI_PT,SAEPOVRTALL_PT&for=county:*&YEAR={year}"
        )
        print(f"  {year}: fetching SAIPE API...")

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        header = data[0]
        rows = data[1:]
        df = pd.DataFrame(rows, columns=header)

        df["fips"] = df["state"].str.zfill(2) + df["county"].str.zfill(3)
        df["year"] = year
        df["median_hh_income"] = pd.to_numeric(df["SAEMHI_PT"], errors="coerce")
        df["poverty_rate"] = pd.to_numeric(df["SAEPOVRTALL_PT"], errors="coerce")

        n = df["median_hh_income"].notna().sum()
        print(f"    {n:,} counties, median income = ${df['median_hh_income'].median():,.0f}")
        all_data.append(df[["fips", "year", "median_hh_income", "poverty_rate"]])

        time.sleep(0.5)

    if not all_data:
        print("  ERROR: No SAIPE data downloaded.")
        return

    result = pd.concat(all_data, ignore_index=True)
    result = harmonize_fips(result)
    result.to_csv(out_file, index=False)
    print(f"  Saved: {out_file} ({len(result):,} rows, {result['fips'].nunique():,} counties)")


# ---------------------------------------------------------------------------
# C. Census Population Estimates
# ---------------------------------------------------------------------------

def download_census_popest():
    """Download Census county population estimates (intercensal + postcensal)."""
    out_file = os.path.join(OUT_DIR, "census_popest_population.csv")
    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        print(f"  Already exists: {out_file} ({len(df):,} rows)")
        return

    print("\n--- Census Population Estimates ---")

    # Three vintage files covering different periods
    sources = [
        {
            "label": "2000-2010 intercensal",
            "url": "https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv",
            "years": range(2006, 2010),  # use for 2006-2009
        },
        {
            "label": "2010-2019 vintage",
            "url": "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv",
            "years": range(2010, 2020),
        },
        {
            "label": "2020-2023 postcensal",
            "url": "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv",
            "years": range(2020, 2023),
        },
    ]

    all_data = []

    for src in sources:
        print(f"  Downloading {src['label']}...")
        try:
            resp = requests.get(src["url"], headers=HEADERS, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"    ERROR: {e}")
            continue

        # Census CSV files use latin-1 encoding
        try:
            df = pd.read_csv(
                io.BytesIO(resp.content),
                encoding="latin-1",
                dtype={"STATE": str, "COUNTY": str},
            )
        except Exception as e:
            print(f"    ERROR parsing CSV: {e}")
            continue

        # Filter to county level (SUMLEV == 50)
        df = df[df["SUMLEV"] == 50].copy()
        df["STATE"] = df["STATE"].str.zfill(2)
        df["COUNTY"] = df["COUNTY"].str.zfill(3)
        df["fips"] = df["STATE"] + df["COUNTY"]

        # Reshape wide → long for the years we need
        for year in src["years"]:
            col = f"POPESTIMATE{year}"
            if col not in df.columns:
                print(f"    WARNING: Column {col} not found")
                continue
            year_df = df[["fips"]].copy()
            year_df["year"] = year
            year_df["population"] = pd.to_numeric(df[col], errors="coerce")
            all_data.append(year_df)

        print(f"    {df['fips'].nunique():,} counties, years {min(src['years'])}-{max(src['years'])-1}")

    if not all_data:
        print("  ERROR: No population data downloaded.")
        return

    result = pd.concat(all_data, ignore_index=True)
    result = harmonize_fips(result)
    result.to_csv(out_file, index=False)
    print(f"  Saved: {out_file} ({len(result):,} rows, {result['fips'].nunique():,} counties)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_census_vap():
    """Download voting-age population from ACS 5-year at county level.

    Uses B01001 (Sex by Age) to compute VAP = total pop - under 18.
    Election years: 2008, 2012, 2016, 2020 → ACS vintages match.
    """
    out_file = os.path.join(OUT_DIR, "census_county_vap.csv")
    if os.path.exists(out_file):
        df = pd.read_csv(out_file)
        print(f"\n  VAP: Already exists ({len(df):,} rows). Delete to re-download.")
        return

    print("\n  Downloading ACS county-level VAP...")

    CENSUS_API_BASE = "https://api.census.gov/data"
    # Under-18 variables from B01001
    vap_vars = {
        "B01003_001E": "total_population",
        "B01001_003E": "male_under5",
        "B01001_004E": "male_5to9",
        "B01001_005E": "male_10to14",
        "B01001_006E": "male_15to17",
        "B01001_027E": "female_under5",
        "B01001_028E": "female_5to9",
        "B01001_029E": "female_10to14",
        "B01001_030E": "female_15to17",
    }
    var_str = ",".join(vap_vars.keys())

    # Election years and their ACS vintages (ACS 5yr starts at vintage 2009)
    election_vintages = {2008: 2009, 2012: 2012, 2016: 2016, 2020: 2020}

    all_data = []
    for election_year, vintage in election_vintages.items():
        url = (
            f"{CENSUS_API_BASE}/{vintage}/acs/acs5"
            f"?get={var_str}"
            f"&for=county:*"
        )
        api_key = os.environ.get("CENSUS_API_KEY")
        if api_key:
            url += f"&key={api_key}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            header = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=header)

            # Construct FIPS
            df["fips"] = df["state"] + df["county"]

            # Convert to numeric
            for acs_var, name in vap_vars.items():
                df[name] = pd.to_numeric(df[acs_var], errors="coerce")

            # Compute VAP
            under18_cols = ["male_under5", "male_5to9", "male_10to14", "male_15to17",
                            "female_under5", "female_5to9", "female_10to14", "female_15to17"]
            under18 = sum(df[c].fillna(0) for c in under18_cols)
            df["voting_age_population"] = (df["total_population"] - under18).clip(lower=0)

            df["year"] = election_year
            all_data.append(df[["fips", "year", "voting_age_population"]])
            print(f"    {election_year} (ACS {vintage}): {len(df):,} counties")
        except Exception as e:
            print(f"    ERROR {election_year}: {e}")

        time.sleep(1)

    if all_data:
        result = pd.concat(all_data, ignore_index=True)
        result = harmonize_fips(result)
        result.to_csv(out_file, index=False)
        print(f"  Saved: {out_file} ({len(result):,} rows)")


def main():
    print("=" * 60)
    print("Download County-Level Economic Controls")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    download_bls_laus()
    download_census_saipe()
    download_census_popest()
    download_census_vap()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for fname in ["bls_laus_unemployment.csv", "census_saipe_income.csv",
                  "census_popest_population.csv", "census_county_vap.csv"]:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            print(f"  {fname}: {len(df):,} rows, years {df['year'].min()}-{df['year'].max()}")
        else:
            print(f"  {fname}: MISSING")

    print("\nDone.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build unified county-year controls panel from downloaded economic and weather data.

Merges: unemployment (BLS), income (SAIPE), population (PopEst), weather (PRISM)
Output: output/controls_panel.parquet
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLS_DIR = os.path.join(BASE_DIR, "data", "controls")
OUT_FILE = os.path.join(BASE_DIR, "output", "controls_panel.parquet")


def load_unemployment():
    """Load BLS LAUS unemployment rate."""
    path = os.path.join(CONTROLS_DIR, "bls_laus_unemployment.csv")
    if not os.path.exists(path):
        print("  WARNING: BLS unemployment file not found")
        return None
    df = pd.read_csv(path, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)
    print(f"  Unemployment: {len(df):,} rows, {df['fips'].nunique():,} counties")
    return df[["fips", "year", "unemployment_rate"]]


def load_income():
    """Load Census SAIPE income and poverty data."""
    path = os.path.join(CONTROLS_DIR, "census_saipe_income.csv")
    if not os.path.exists(path):
        print("  WARNING: SAIPE income file not found")
        return None
    df = pd.read_csv(path, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)
    print(f"  Income/poverty: {len(df):,} rows, {df['fips'].nunique():,} counties")
    return df[["fips", "year", "median_hh_income", "poverty_rate"]]


def load_population():
    """Load Census population estimates."""
    path = os.path.join(CONTROLS_DIR, "census_popest_population.csv")
    if not os.path.exists(path):
        print("  WARNING: Population estimates file not found")
        return None
    df = pd.read_csv(path, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)
    print(f"  Population: {len(df):,} rows, {df['fips'].nunique():,} counties")
    return df[["fips", "year", "population"]]


def load_vap():
    """Load ACS voting-age population at county level."""
    path = os.path.join(CONTROLS_DIR, "census_county_vap.csv")
    if not os.path.exists(path):
        print("  WARNING: County VAP file not found")
        return None
    df = pd.read_csv(path, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)
    print(f"  VAP: {len(df):,} rows, {df['fips'].nunique():,} counties")
    return df[["fips", "year", "voting_age_population"]]


def load_weather():
    """Load PRISM October weather data."""
    path = os.path.join(CONTROLS_DIR, "prism", "county_weather_october.csv")
    if not os.path.exists(path):
        print("  WARNING: PRISM weather file not found")
        return None
    df = pd.read_csv(path, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)
    print(f"  Weather: {len(df):,} rows, {df['fips'].nunique():,} counties")
    return df[["fips", "year", "october_tmean", "october_ppt"]]


def main():
    print("=" * 60)
    print("Build Controls Panel")
    print("=" * 60)

    print("\nLoading control datasets...")
    datasets = {
        "unemployment": load_unemployment(),
        "income": load_income(),
        "population": load_population(),
        "vap": load_vap(),
        "weather": load_weather(),
    }

    # Start with the first available dataset
    available = {k: v for k, v in datasets.items() if v is not None}
    if not available:
        print("ERROR: No control datasets found. Run download scripts first.")
        return

    # Full outer merge on [fips, year]
    print("\nMerging datasets...")
    panel = None
    for name, df in available.items():
        if panel is None:
            panel = df
        else:
            panel = panel.merge(df, on=["fips", "year"], how="outer")
        print(f"  After merging {name}: {len(panel):,} rows")

    # Add derived variables
    if "population" in panel.columns:
        panel["log_population"] = np.log(panel["population"].clip(lower=1))
    if "median_hh_income" in panel.columns:
        panel["log_median_income"] = np.log(panel["median_hh_income"].clip(lower=1))

    # Sort
    panel = panel.sort_values(["fips", "year"]).reset_index(drop=True)

    # Coverage report
    print("\n" + "=" * 60)
    print("CONTROLS PANEL SUMMARY")
    print("=" * 60)
    print(f"  Total rows: {len(panel):,}")
    print(f"  Counties: {panel['fips'].nunique():,}")
    print(f"  Years: {panel['year'].min()}-{panel['year'].max()}")

    print("\n  Variable coverage:")
    control_vars = [
        "unemployment_rate", "median_hh_income", "poverty_rate",
        "population", "log_population", "log_median_income",
        "voting_age_population",
        "october_tmean", "october_ppt",
    ]
    for var in control_vars:
        if var in panel.columns:
            n = panel[var].notna().sum()
            pct = n / len(panel) * 100
            print(f"    {var}: {n:,}/{len(panel):,} ({pct:.1f}%)")

    print(f"\n  By year:")
    for yr in sorted(panel["year"].unique()):
        yr_df = panel[panel["year"] == yr]
        print(f"    {yr}: {len(yr_df):,} counties")

    # Save
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    panel.to_parquet(OUT_FILE, index=False)
    print(f"\n  Saved: {OUT_FILE}")
    print(f"  Shape: {panel.shape}")
    print(f"  Columns: {list(panel.columns)}")

    print("\nDone.")


if __name__ == "__main__":
    main()

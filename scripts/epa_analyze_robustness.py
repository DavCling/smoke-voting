#!/usr/bin/env python3
"""
EPA AQS Ground Monitor PM2.5 Robustness Check: validation and regression comparison.

EPA AQS measures the same quantity as Childs et al. (ground-level PM2.5) but
from physical FRM/FEM instruments rather than ML prediction. This is a tighter
validation than HMS (which measured satellite-observed plume presence).

Part A — Validation:
  How well do Childs ML-predicted smoke PM2.5 and EPA ground monitor PM2.5
  agree at county-day and county-window levels?

Part B — Regression Comparison:
  Does the build-up table reproduce with EPA-based treatment?
  Critical: three-way subsample comparison separating sample selection
  from measurement differences.

Inputs:
  - output/smoke_voting_analysis.parquet    (Childs-based analysis panel)
  - output/epa_pm25_exposure.parquet        (EPA windowed variables)
  - data/smoke/smoke_pm25_county_daily.csv  (Childs daily for daily validation)
  - data/epa/epa_county_daily.csv           (EPA daily for daily validation)

Outputs:
  - Printed tables
  - output/figures/epa_validation_scatter.png
  - output/figures/epa_validation_timeseries.png
  - output/figures/epa_buildup_comparison.png
  - output/figures/epa_subsample_comparison.png
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_analysis.parquet")
EPA_FILE = os.path.join(BASE_DIR, "output", "epa_pm25_exposure.parquet")
CHILDS_DAILY = os.path.join(BASE_DIR, "data", "smoke", "smoke_pm25_county_daily.csv")
EPA_DAILY = os.path.join(BASE_DIR, "data", "epa", "epa_county_daily.csv")
FIG_DIR = os.path.join(BASE_DIR, "output", "figures")

ELECTION_DATES = {
    2008: "2008-11-04",
    2012: "2012-11-06",
    2016: "2016-11-08",
    2020: "2020-11-03",
}


# ======================================================================
# Regression helpers (copied from analyze_smoke_voting.py per convention)
# ======================================================================

def run_twfe(df, dep_var, smoke_var, controls=None, absorb_entity=True, absorb_time=True,
             state_year_fe=False, drop_absorbed=False, label=""):
    """Run a two-way fixed effects regression using linearmodels PanelOLS."""
    cols = [dep_var, smoke_var]
    if controls:
        cols += controls
    subset = df[cols].dropna()

    if len(subset) < 100:
        print(f"  SKIP {label}: only {len(subset)} non-missing observations")
        return None

    y = subset[dep_var]
    x_cols = [smoke_var]
    if controls:
        x_cols += controls
    x = sm.add_constant(subset[x_cols])

    try:
        if state_year_fe:
            state_year_cat = pd.Categorical(
                subset.index.get_level_values("fips").astype(str).str[:2] + "_" +
                subset.index.get_level_values("year").astype(str)
            )
            other_ef = pd.DataFrame(state_year_cat, index=subset.index, columns=["state_year"])
            mod = PanelOLS(y, x, entity_effects=True, time_effects=False,
                           other_effects=other_ef, check_rank=False,
                           drop_absorbed=drop_absorbed)
        else:
            mod = PanelOLS(
                y, x,
                entity_effects=absorb_entity,
                time_effects=absorb_time,
                check_rank=False,
                drop_absorbed=drop_absorbed,
            )
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        return res
    except Exception as e:
        print(f"  ERROR in {label}: {e}")
        return None


def print_result(res, label, smoke_var):
    """Print a compact regression result summary."""
    if res is None:
        return None
    coef = res.params.get(smoke_var, np.nan)
    se = res.std_errors.get(smoke_var, np.nan)
    pval = res.pvalues.get(smoke_var, np.nan)
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    n = int(res.nobs)
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""

    print(f"\n  {label}")
    print(f"    β = {coef:.6f} {stars}  (SE = {se:.6f})")
    print(f"    95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"    p-value = {pval:.4f}")
    print(f"    N = {n:,}")
    print(f"    R² (within) = {res.rsquared_within:.4f}")

    return {"label": label, "coef": coef, "se": se, "pval": pval,
            "ci_low": ci_low, "ci_high": ci_high, "n": n,
            "r2_within": res.rsquared_within}


def _make_state_trends(df):
    """Create state-specific linear trend columns for the panel."""
    fips_vals = df.index.get_level_values("fips").astype(str).str[:2]
    year_vals = df.index.get_level_values("year").astype(float)
    year_norm = year_vals - np.mean(year_vals)

    states = sorted(fips_vals.unique())
    if len(states) <= 1:
        return df, []

    trend_cols = []
    for st in states[1:]:
        col = f"trend_{st}"
        df[col] = (fips_vals == st).astype(float) * year_norm
        trend_cols.append(col)

    return df, trend_cols


def _extract_coef(res, smoke_var):
    """Extract coefficient info from a regression result."""
    if res is None:
        return {"coef": np.nan, "se": np.nan, "pval": np.nan, "n": 0, "stars": ""}
    coef = res.params.get(smoke_var, np.nan)
    se = res.std_errors.get(smoke_var, np.nan)
    pval = res.pvalues.get(smoke_var, np.nan)
    n = int(res.nobs)
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
    return {"coef": coef, "se": se, "pval": pval, "n": n, "stars": stars}


# ======================================================================
# Data loading
# ======================================================================

def load_data():
    """Load Childs analysis panel and merge with EPA exposure."""
    print("Loading analysis datasets...")

    # Childs-based panel
    df = pd.read_parquet(DATA_FILE)
    print(f"  Childs panel: {len(df):,} obs, {df['fips'].nunique():,} counties")

    # EPA exposure panel
    epa = pd.read_parquet(EPA_FILE)
    print(f"  EPA panel: {len(epa):,} obs, {epa['fips'].nunique():,} counties")

    # Merge
    merged = df.merge(epa, on=["fips", "year"], how="inner",
                      suffixes=("", "_epa"))
    print(f"  Merged (inner): {len(merged):,} obs, {merged['fips'].nunique():,} counties")

    # Drop duplicate columns from EPA
    drop_cols = [c for c in merged.columns if c.endswith("_epa")]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    # Set panel index
    merged = merged.set_index(["fips", "year"]).sort_index()

    return merged, df


# ======================================================================
# Part A: Validation
# ======================================================================

def coverage_summary(epa_panel):
    """EPA monitor coverage: counties per year, coverage distribution."""
    print("\n" + "=" * 70)
    print("PART A.0: EPA Monitor Coverage Summary")
    print("=" * 70)

    epa = pd.read_parquet(EPA_FILE)

    print(f"\n  Total county-year observations: {len(epa):,}")
    print(f"  Unique counties: {epa['fips'].nunique():,}")

    print(f"\n  {'Year':>6s}  {'Counties':>10s}  {'Valid 30d':>10s}  "
          f"{'Mean Coverage':>14s}  {'Median Coverage':>16s}")
    print("  " + "-" * 65)

    for year in sorted(ELECTION_DATES.keys()):
        yr = epa[epa["year"] == year]
        n_counties = len(yr)
        valid_30d = yr["epa_pm25_mean_30d"].notna().sum()
        mean_cov = yr["epa_coverage_30d"].mean() if "epa_coverage_30d" in yr.columns else np.nan
        med_cov = yr["epa_coverage_30d"].median() if "epa_coverage_30d" in yr.columns else np.nan
        print(f"  {year:>6d}  {n_counties:>10,}  {valid_30d:>10,}  "
              f"{mean_cov:>13.1%}  {med_cov:>15.1%}")

    # Geographic coverage: states represented
    print(f"\n  States represented: {epa['state_fips'].nunique()}")

    # Coverage distribution for 30d
    if "epa_coverage_30d" in epa.columns:
        cov = epa["epa_coverage_30d"].dropna()
        print(f"\n  30d coverage distribution:")
        for pct in [10, 25, 50, 75, 90]:
            print(f"    P{pct}: {cov.quantile(pct/100):.1%}")


def daily_validation():
    """Scatter + correlation: Childs smoke PM2.5 vs EPA total PM2.5 at county-day level."""
    print("\n" + "=" * 70)
    print("PART A.1: Daily Validation — Childs Smoke PM2.5 vs EPA Total PM2.5")
    print("=" * 70)

    if not os.path.exists(CHILDS_DAILY) or not os.path.exists(EPA_DAILY):
        print("  Missing daily files — skipping daily validation")
        print(f"    Childs daily: {os.path.exists(CHILDS_DAILY)}")
        print(f"    EPA daily: {os.path.exists(EPA_DAILY)}")
        return

    # Load Childs daily
    print("  Loading Childs daily data...")
    with open(CHILDS_DAILY) as f:
        sep = "\t" if "\t" in f.readline() else ","
    childs = pd.read_csv(CHILDS_DAILY, sep=sep, dtype={"GEOID": str})
    childs = childs.rename(columns={"GEOID": "fips", "smokePM_pred": "smoke_pm25"})
    childs["fips"] = childs["fips"].str.zfill(5)
    childs["date"] = pd.to_datetime(childs["date"], format="%Y%m%d")

    # Load EPA daily
    print("  Loading EPA daily data...")
    epa = pd.read_csv(EPA_DAILY, dtype={"fips": str})
    epa["fips"] = epa["fips"].str.zfill(5)
    epa["date"] = pd.to_datetime(epa["date"])

    # Filter both to election windows
    childs_filtered = []
    epa_filtered = []
    for year, edate_str in ELECTION_DATES.items():
        edate = pd.Timestamp(edate_str)
        start = pd.Timestamp(f"{year}-06-01")

        c = childs[(childs["date"] >= start) & (childs["date"] <= edate)]
        e = epa[(epa["date"] >= start) & (epa["date"] <= edate)]
        childs_filtered.append(c)
        epa_filtered.append(e)

    childs_f = pd.concat(childs_filtered, ignore_index=True)
    epa_f = pd.concat(epa_filtered, ignore_index=True)

    print(f"  Childs daily (election windows): {len(childs_f):,} rows "
          f"(smoke-only: file omits zero-smoke days)")
    print(f"  EPA daily (election windows): {len(epa_f):,} rows")

    # Merge on county-day: EPA as base, left-join Childs
    # Childs file only contains days with predicted smoke, so missing = 0
    merged = epa_f[["fips", "date", "epa_pm25"]].merge(
        childs_f[["fips", "date", "smoke_pm25"]],
        on=["fips", "date"], how="left"
    )
    merged["smoke_pm25"] = merged["smoke_pm25"].fillna(0)
    print(f"  Matched county-day observations: {len(merged):,}")

    if len(merged) == 0:
        print("  No matched observations — skipping")
        return

    # Correlation: EPA total PM2.5 vs Childs smoke PM2.5
    corr = merged["epa_pm25"].corr(merged["smoke_pm25"])
    print(f"\n  Daily Pearson correlation (EPA total vs Childs smoke): {corr:.3f}")
    print(f"  (Note: EPA = total PM2.5; Childs = smoke-attributable PM2.5 only)")

    # Summary stats
    print(f"\n  EPA PM2.5:   mean={merged['epa_pm25'].mean():.2f}, "
          f"median={merged['epa_pm25'].median():.2f}, "
          f"SD={merged['epa_pm25'].std():.2f}")
    print(f"  Childs smoke: mean={merged['smoke_pm25'].mean():.4f}, "
          f"median={merged['smoke_pm25'].median():.4f}, "
          f"SD={merged['smoke_pm25'].std():.4f}")

    # At elevated levels: restrict to EPA > 20 µg/m³
    elevated = merged[merged["epa_pm25"] > 20]
    if len(elevated) > 10:
        corr_elevated = elevated["epa_pm25"].corr(elevated["smoke_pm25"])
        print(f"\n  Correlation at elevated EPA PM2.5 (>20 µg/m³): {corr_elevated:.3f}")
        print(f"    N elevated: {len(elevated):,} ({100*len(elevated)/len(merged):.1f}%)")
        print(f"    Fraction with any Childs smoke: "
              f"{(elevated['smoke_pm25'] > 0).mean():.1%}")

    # Binary agreement: EPA > 20 vs Childs > 0
    merged["epa_elevated"] = (merged["epa_pm25"] > 20).astype(int)
    merged["childs_any"] = (merged["smoke_pm25"] > 0).astype(int)

    tp = ((merged["epa_elevated"] == 1) & (merged["childs_any"] == 1)).sum()
    fp = ((merged["epa_elevated"] == 1) & (merged["childs_any"] == 0)).sum()
    fn = ((merged["epa_elevated"] == 0) & (merged["childs_any"] == 1)).sum()
    tn = ((merged["epa_elevated"] == 0) & (merged["childs_any"] == 0)).sum()

    print(f"\n  --- EPA elevated (>20 µg/m³) vs Childs any smoke (>0) ---")
    print(f"    TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    if (tp + fn) > 0:
        print(f"    Sensitivity (EPA elevated given Childs smoke): "
              f"{tp / (tp + fn):.3f}")
    if (tn + fp) > 0:
        print(f"    Specificity: {tn / (tn + fp):.3f}")


def window_validation(df):
    """Scatter plots and correlations at 30d window level."""
    print("\n" + "=" * 70)
    print("PART A.2: Window-Level Validation (30d)")
    print("=" * 70)

    pairs = [
        ("smoke_pm25_mean_30d", "epa_pm25_mean_30d", "Mean Smoke PM2.5 vs Mean EPA PM2.5"),
        ("smoke_frac_haze_30d", "epa_frac_above20_30d", "Frac Haze (>20) vs EPA Frac >20"),
        ("smoke_frac_usg_30d", "epa_frac_above35_30d", "Frac USG (>35.5) vs EPA Frac >35.5"),
        ("smoke_frac_unhealthy_30d", "epa_frac_above55_30d", "Frac Unhealthy (>55.5) vs EPA >55.5"),
    ]

    df_reset = df.reset_index()

    print(f"\n  {'Pair':<50} {'Pearson r':>10} {'R²':>8} {'N':>8}")
    print("  " + "-" * 80)
    valid_pairs = []
    for childs_var, epa_var, label in pairs:
        if childs_var not in df_reset.columns or epa_var not in df_reset.columns:
            print(f"  {label:<50} {'(missing)':>10}")
            continue
        mask = df_reset[childs_var].notna() & df_reset[epa_var].notna()
        n = mask.sum()
        if n < 10:
            print(f"  {label:<50} {'(N<10)':>10}")
            continue
        r = df_reset.loc[mask, childs_var].corr(df_reset.loc[mask, epa_var])
        print(f"  {label:<50} {r:>10.3f} {r**2:>8.3f} {n:>8,}")
        valid_pairs.append((childs_var, epa_var, label, r))

    # Scatter plots
    if valid_pairs:
        os.makedirs(FIG_DIR, exist_ok=True)
        n_plots = len(valid_pairs)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
        if n_plots == 1:
            axes = [axes]

        for ax, (cv, ev, lbl, r) in zip(axes, valid_pairs):
            mask = df_reset[cv].notna() & df_reset[ev].notna()
            ax.scatter(df_reset.loc[mask, cv], df_reset.loc[mask, ev],
                       alpha=0.15, s=8, color="steelblue")

            # Add 45-degree line for mean PM2.5 comparison
            if "mean" in cv.lower():
                lims = [0, max(df_reset.loc[mask, cv].max(), df_reset.loc[mask, ev].max())]
                ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.8)

            ax.set_xlabel(f"Childs: {cv.replace('smoke_', '').replace('_30d', '')}")
            ax.set_ylabel(f"EPA: {ev.replace('epa_', '').replace('_30d', '')}")
            ax.set_title(f"{lbl}\nr = {r:.3f}, R² = {r**2:.3f}")

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "epa_validation_scatter.png")
        fig.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n  Saved: {fig_path}")


def timeseries_validation():
    """National daily time series (4 panels): Childs smoke PM2.5 vs EPA total PM2.5."""
    print("\n" + "=" * 70)
    print("PART A.3: Time Series Validation")
    print("=" * 70)

    if not os.path.exists(CHILDS_DAILY) or not os.path.exists(EPA_DAILY):
        print("  Missing daily files — skipping")
        return

    print("  Loading daily data...")
    with open(CHILDS_DAILY) as f:
        sep = "\t" if "\t" in f.readline() else ","
    childs = pd.read_csv(CHILDS_DAILY, sep=sep, dtype={"GEOID": str})
    childs = childs.rename(columns={"GEOID": "fips", "smokePM_pred": "smoke_pm25"})
    childs["fips"] = childs["fips"].str.zfill(5)
    childs["date"] = pd.to_datetime(childs["date"], format="%Y%m%d")

    epa = pd.read_csv(EPA_DAILY, dtype={"fips": str})
    epa["fips"] = epa["fips"].str.zfill(5)
    epa["date"] = pd.to_datetime(epa["date"])

    # Find EPA-covered counties for consistent comparison
    epa_counties = set(epa["fips"].unique())

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for idx, (year, edate_str) in enumerate(sorted(ELECTION_DATES.items())):
        ax = axes[idx // 2][idx % 2]
        edate = pd.Timestamp(edate_str)
        start = pd.Timestamp(f"{year}-06-01")

        # Filter to this year's window and EPA-covered counties only
        c = childs[(childs["date"] >= start) & (childs["date"] <= edate) &
                   (childs["fips"].isin(epa_counties))]
        e = epa[(epa["date"] >= start) & (epa["date"] <= edate)]

        # National daily means
        childs_daily_mean = c.groupby("date")["smoke_pm25"].mean()
        epa_daily_mean = e.groupby("date")["epa_pm25"].mean()

        ax2 = ax.twinx()
        ax.plot(childs_daily_mean.index, childs_daily_mean.values,
                color="steelblue", alpha=0.7, linewidth=0.8, label="Childs smoke PM2.5")
        ax2.plot(epa_daily_mean.index, epa_daily_mean.values,
                 color="firebrick", alpha=0.7, linewidth=0.8, label="EPA total PM2.5")

        ax.set_ylabel("Childs smoke PM2.5 (µg/m³)", color="steelblue", fontsize=9)
        ax2.set_ylabel("EPA total PM2.5 (µg/m³)", color="firebrick", fontsize=9)
        ax.set_title(f"{year}")
        ax.axvline(edate, color="black", linestyle="--", alpha=0.5, linewidth=0.8)

        # Correlation
        combined = pd.DataFrame({
            "childs": childs_daily_mean,
            "epa": epa_daily_mean,
        }).dropna()
        if len(combined) > 10:
            r = combined["childs"].corr(combined["epa"])
            ax.text(0.02, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                    fontsize=9, va="top")

    fig.suptitle("Childs ML-Predicted Smoke PM2.5 vs EPA Ground Monitor PM2.5\n"
                 "(National Daily Means, Jun 1 – Election Day, EPA-covered counties)",
                 fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "epa_validation_timeseries.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")


# ======================================================================
# Part B: Regression Comparison
# ======================================================================

def epa_buildup_table(df):
    """Build-up table using EPA treatment, side-by-side with Childs."""
    print("\n" + "=" * 70)
    print("PART B.1: Build-Up Table — Childs vs EPA")
    print("  (1) Raw OLS  (2) County+Year FE  (3) +Controls  (4) +State Trends")
    print("=" * 70)

    childs_var = "smoke_pm25_mean_30d"
    epa_var = "epa_pm25_mean_30d"

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    # Create state trend columns
    df, trend_cols = _make_state_trends(df)

    outcomes = [
        ("turnout_rate", "Panel A: Turnout Rate"),
        ("log_total_votes", "Panel B: Log Total Votes"),
        ("dem_vote_share", "Panel C: DEM Vote Share"),
        ("incumbent_vote_share", "Panel D: Incumbent Vote Share"),
    ]

    results = {}

    for dep_var, panel_label in outcomes:
        print(f"\n  {panel_label}")
        childs_results = []
        epa_results = []

        for smoke_var, result_list, data_label in [
            (childs_var, childs_results, "Childs"),
            (epa_var, epa_results, "EPA"),
        ]:
            # Spec 1: Raw OLS
            res1 = run_twfe(df, dep_var, smoke_var,
                            absorb_entity=False, absorb_time=False,
                            label=f"(1) Raw OLS [{data_label}]: {dep_var}")
            # Spec 2: TWFE
            res2 = run_twfe(df, dep_var, smoke_var,
                            label=f"(2) TWFE [{data_label}]: {dep_var}")
            # Spec 3: +Controls
            res3 = run_twfe(df, dep_var, smoke_var, controls=available,
                            label=f"(3) +Controls [{data_label}]: {dep_var}")
            # Spec 4: +State trends
            res4 = run_twfe(df, dep_var, smoke_var,
                            controls=available + trend_cols,
                            drop_absorbed=True,
                            label=f"(4) +State trends [{data_label}]: {dep_var}")

            for res in [res1, res2, res3, res4]:
                result_list.append(_extract_coef(res, smoke_var))

        results[dep_var] = {"childs": childs_results, "epa": epa_results}

    # Print formatted side-by-side table
    print("\n" + "=" * 100)
    print(f"  {'':35s} {'(1) Raw OLS':>14s} {'(2) TWFE':>14s} {'(3) +Controls':>14s} {'(4) +St.Trends':>14s}")
    print("=" * 100)

    for dep_var, panel_label in outcomes:
        r = results[dep_var]
        print(f"\n  {panel_label}")

        # Childs row
        coef_line = f"  {'Childs smoke PM2.5 (30d)':<35s}"
        se_line = f"  {'':35s}"
        for cr in r["childs"]:
            coef_line += f" {cr['coef']:>12.5f}{cr['stars']:<2s}"
            se_line += f" ({cr['se']:>11.5f}) "
        print(coef_line)
        print(se_line)

        # EPA row
        coef_line = f"  {'EPA total PM2.5 (30d)':<35s}"
        se_line = f"  {'':35s}"
        for er in r["epa"]:
            coef_line += f" {er['coef']:>12.5f}{er['stars']:<2s}"
            se_line += f" ({er['se']:>11.5f}) "
        print(coef_line)
        print(se_line)

        # Standardized comparison (Spec 3)
        childs_sd = df[childs_var].std()
        epa_sd = df[epa_var].dropna().std()
        childs_beta = r["childs"][2]["coef"] * childs_sd
        epa_beta = r["epa"][2]["coef"] * epa_sd
        print(f"  {'Standardized β (Spec 3)':<35s} "
              f"Childs: {childs_beta:.6f}   EPA: {epa_beta:.6f}")

    print("\n" + "-" * 100)
    print(f"  {'County FE':<35s} {'':>14s} {'Yes':>14s} {'Yes':>14s} {'Yes':>14s}")
    print(f"  {'Year FE':<35s} {'':>14s} {'Yes':>14s} {'Yes':>14s} {'Yes':>14s}")
    print(f"  {'Controls':<35s} {'':>14s} {'':>14s} {'Yes':>14s} {'Yes':>14s}")
    print(f"  {'State trends':<35s} {'':>14s} {'':>14s} {'':>14s} {'Yes':>14s}")
    print("=" * 100)

    # Treatment variable statistics
    print(f"\n  Treatment Variable Summary (merged sample):")
    print(f"    Childs smoke PM2.5 (30d): mean={df[childs_var].mean():.4f}, "
          f"SD={df[childs_var].std():.4f}, max={df[childs_var].max():.3f}")
    epa_vals = df[epa_var].dropna()
    print(f"    EPA total PM2.5 (30d):    mean={epa_vals.mean():.4f}, "
          f"SD={epa_vals.std():.4f}, max={epa_vals.max():.3f}")
    both_valid = df[[childs_var, epa_var]].dropna()
    if len(both_valid) > 10:
        r_corr = both_valid[childs_var].corr(both_valid[epa_var])
        print(f"    Correlation:              r = {r_corr:.3f}")

    return results


def epa_subsample_comparison(df, df_full):
    """Three-way comparison: (1) Childs full, (2) Childs on EPA counties, (3) EPA.

    This is the CRITICAL test. Separates sample selection from measurement
    differences by comparing:
      (1) Childs on all ~12,200 counties
      (2) Childs on the ~2,800 EPA-covered counties only
      (3) EPA on the same ~2,800 counties

    If (1) ≈ (2): full-sample results generalize to monitored counties.
    If (2) ≈ (3): Childs ML and EPA ground truth give same answer.
    If (1) ≠ (2): sample selection matters (monitored counties differ).
    If (2) ≠ (3): measurement matters (Childs smoke vs EPA total PM2.5).
    """
    print("\n" + "=" * 70)
    print("PART B.2: Three-Way Subsample Comparison (Spec 3, 30d)")
    print("  (1) Childs full sample")
    print("  (2) Childs restricted to EPA-covered counties")
    print("  (3) EPA on same counties")
    print("=" * 70)

    childs_var = "smoke_pm25_mean_30d"
    epa_var = "epa_pm25_mean_30d"

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]

    outcomes = [
        ("turnout_rate", "Turnout Rate"),
        ("log_total_votes", "Log Total Votes"),
        ("dem_vote_share", "DEM Vote Share"),
        ("incumbent_vote_share", "Incumbent V.S."),
    ]

    # (1) Childs full sample
    df_full_panel = df_full.set_index(["fips", "year"]).sort_index()
    available_full = [v for v in control_vars
                      if v in df_full_panel.columns and df_full_panel[v].notna().any()]

    # (2) and (3) use the merged panel (df) which is already EPA-restricted
    available_merged = [v for v in control_vars
                        if v in df.columns and df[v].notna().any()]

    results = {}

    print(f"\n  {'Outcome':<22s} {'(1) Childs Full':>18s} {'(2) Childs EPA-sub':>18s} {'(3) EPA':>18s}")
    print("  " + "-" * 80)

    for dep_var, dep_label in outcomes:
        # (1) Childs full
        res1 = run_twfe(df_full_panel, dep_var, childs_var, controls=available_full,
                        label=f"Childs full: {dep_var}")
        r1 = _extract_coef(res1, childs_var)

        # (2) Childs on EPA subsample
        res2 = run_twfe(df, dep_var, childs_var, controls=available_merged,
                        label=f"Childs EPA-sub: {dep_var}")
        r2 = _extract_coef(res2, childs_var)

        # (3) EPA
        res3 = run_twfe(df, dep_var, epa_var, controls=available_merged,
                        label=f"EPA: {dep_var}")
        r3 = _extract_coef(res3, epa_var)

        results[dep_var] = {"full": r1, "childs_sub": r2, "epa": r3}

        print(f"  {dep_label:<22s} "
              f"{r1['coef']:>14.5f}{r1['stars']:<3s} "
              f"{r2['coef']:>14.5f}{r2['stars']:<3s} "
              f"{r3['coef']:>14.5f}{r3['stars']:<3s}")
        print(f"  {'':22s} "
              f"({r1['se']:>13.5f})  "
              f"({r2['se']:>13.5f})  "
              f"({r3['se']:>13.5f}) ")
        print(f"  {'':22s} "
              f"N={r1['n']:>12,}  "
              f"N={r2['n']:>12,}  "
              f"N={r3['n']:>12,} ")

    # Figure: coefficient comparison
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    y_positions = np.arange(len(outcomes))
    height = 0.25
    colors = ["steelblue", "darkorange", "firebrick"]
    labels_list = ["(1) Childs Full", "(2) Childs EPA-sub", "(3) EPA"]

    for dep_idx, (dep_var, dep_label) in enumerate(outcomes):
        r = results[dep_var]

        for i, (key, color, lbl) in enumerate(
            zip(["full", "childs_sub", "epa"], colors, labels_list)
        ):
            # Standardize: multiply by SD of treatment
            if key in ("full", "childs_sub"):
                sd = df_full[childs_var].std() if key == "full" else df.reset_index()[childs_var].std()
            else:
                sd = df.reset_index()[epa_var].dropna().std()

            coef_std = r[key]["coef"] * sd
            se_std = r[key]["se"] * sd

            y = dep_idx + (i - 1) * height
            ax.barh(y, coef_std, height=height * 0.8, color=color, alpha=0.7,
                    label=lbl if dep_idx == 0 else "")
            ax.errorbar(coef_std, y, xerr=1.96 * se_std,
                        fmt="none", color=color, capsize=3)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([lbl for _, lbl in outcomes])
    ax.set_xlabel("Standardized coefficient (β × SD_treatment)")
    ax.set_title("Three-Way Subsample Comparison (Spec 3: TWFE + Controls, 30d)")
    ax.legend(loc="best", fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "epa_subsample_comparison.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  Saved: {fig_path}")

    return results


def epa_threshold_comparison(df):
    """EPA thresholds vs Childs thresholds, Spec 3."""
    print("\n" + "=" * 70)
    print("PART B.3: Threshold Comparison — EPA vs Childs (Spec 3)")
    print("=" * 70)

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    thresholds = [
        ("epa_frac_above20_30d", "EPA: >20 µg/m³", "smoke_frac_haze_30d", "Childs: Haze (>20)"),
        ("epa_frac_above35_30d", "EPA: >35.5 µg/m³", "smoke_frac_usg_30d", "Childs: USG (>35.5)"),
        ("epa_frac_above55_30d", "EPA: >55.5 µg/m³", "smoke_frac_unhealthy_30d", "Childs: Unhealthy (>55.5)"),
    ]

    outcomes = [
        ("turnout_rate", "Turnout Rate"),
        ("log_total_votes", "Log Votes"),
        ("dem_vote_share", "DEM V.S."),
        ("incumbent_vote_share", "Incumb V.S."),
    ]

    print(f"\n  {'Treatment':<30s}", end="")
    for _, dep_label in outcomes:
        print(f" {dep_label:>16s}", end="")
    print()
    print("  " + "-" * 100)

    for epa_var, epa_label, childs_var, childs_label in thresholds:
        for smoke_var, label in [(childs_var, childs_label), (epa_var, epa_label)]:
            if smoke_var not in df.columns:
                continue
            vals = df[smoke_var].dropna()
            nonzero = (vals > 0).sum()

            print(f"  {label:<30s}", end="")
            for dep_var, _ in outcomes:
                res = run_twfe(df, dep_var, smoke_var, controls=available,
                               label=f"{label}: {dep_var}")
                if res is not None:
                    coef = res.params.get(smoke_var, np.nan)
                    pval = res.pvalues.get(smoke_var, np.nan)
                    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                    print(f" {coef:>13.5f}{stars:<3s}", end="")
                else:
                    print(f" {'---':>16s}", end="")
            print(f"  (N_nz={nonzero:,})")
        print()


def epa_comparison_figure(df, results):
    """Side-by-side standardized coefficient plot across 4 specs."""
    print("\n" + "=" * 70)
    print("PART B.4: Comparison Figure")
    print("=" * 70)

    if results is None:
        print("  No results to plot — skipping")
        return

    childs_var = "smoke_pm25_mean_30d"
    epa_var = "epa_pm25_mean_30d"

    childs_sd = df.reset_index()[childs_var].std()
    epa_sd = df.reset_index()[epa_var].dropna().std()

    outcomes = [
        ("turnout_rate", "Turnout Rate"),
        ("log_total_votes", "Log Total Votes"),
        ("dem_vote_share", "DEM Vote Share"),
        ("incumbent_vote_share", "Incumbent Vote Share"),
    ]
    spec_labels = ["(1) Raw OLS", "(2) TWFE", "(3) +Controls", "(4) +St.Trends"]

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (dep_var, dep_label) in enumerate(outcomes):
        ax = axes[idx // 2][idx % 2]
        r = results[dep_var]

        x_pos = np.arange(4)
        width = 0.35

        # Standardized coefficients
        childs_coefs = [cr["coef"] * childs_sd for cr in r["childs"]]
        childs_ses = [cr["se"] * childs_sd for cr in r["childs"]]
        epa_coefs = [er["coef"] * epa_sd for er in r["epa"]]
        epa_ses = [er["se"] * epa_sd for er in r["epa"]]

        ax.bar(x_pos - width/2, childs_coefs, width, label="Childs",
               color="steelblue", alpha=0.7)
        ax.errorbar(x_pos - width/2, childs_coefs,
                    yerr=[1.96 * s for s in childs_ses],
                    fmt="none", color="steelblue", capsize=3)

        ax.bar(x_pos + width/2, epa_coefs, width, label="EPA",
               color="firebrick", alpha=0.7)
        ax.errorbar(x_pos + width/2, epa_coefs,
                    yerr=[1.96 * s for s in epa_ses],
                    fmt="none", color="firebrick", capsize=3)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["OLS", "TWFE", "+Ctrl", "+Trend"], fontsize=9)
        ax.set_title(dep_label, fontsize=11)
        ax.set_ylabel("Standardized effect (β × SD)", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Childs ML-Predicted Smoke PM2.5 vs EPA Ground Monitor PM2.5\n"
                 "Standardized Coefficients (β × SD_treatment) by Specification",
                 fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "epa_buildup_comparison.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("EPA AQS GROUND MONITOR PM2.5 ROBUSTNESS CHECK")
    print("  Childs et al. ML-Predicted Smoke PM2.5 vs EPA FRM/FEM Monitors")
    print("=" * 70)

    # Part A: Validation
    coverage_summary(EPA_FILE)
    daily_validation()

    df, df_full = load_data()
    window_validation(df)
    timeseries_validation()

    # Part B: Regression Comparison
    results = epa_buildup_table(df)
    epa_subsample_comparison(df, df_full)
    epa_threshold_comparison(df)
    epa_comparison_figure(df, results)

    print("\n" + "=" * 70)
    print("EPA robustness check complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

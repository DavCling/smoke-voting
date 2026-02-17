#!/usr/bin/env python3
"""
HMS Smoke Plume Robustness Check: validation and regression comparison.

Part A — Correlation Validation:
  How well do Childs et al. ML-predicted smoke PM2.5 and NOAA HMS satellite
  smoke plumes agree at county-day and county-window levels?

Part B — Regression Robustness:
  Does the build-up table (4 specs) reproduce with HMS-based treatment?

Inputs:
  - output/smoke_voting_analysis.parquet   (existing Childs-based analysis panel)
  - output/hms_smoke_exposure.parquet      (HMS windowed variables)
  - data/smoke/smoke_pm25_county_daily.csv (Childs daily for daily validation)
  - data/hms/hms_county_daily.csv          (HMS daily for daily validation)

Outputs:
  - Printed tables
  - output/figures/hms_validation_scatter.png
  - output/figures/hms_correlation_heatmap.png
  - output/figures/hms_validation_timeseries.png
  - output/figures/hms_buildup_comparison.png
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
HMS_FILE = os.path.join(BASE_DIR, "output", "hms_smoke_exposure.parquet")
CHILDS_DAILY = os.path.join(BASE_DIR, "data", "smoke", "smoke_pm25_county_daily.csv")
HMS_DAILY = os.path.join(BASE_DIR, "data", "hms", "hms_county_daily.csv")
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


# ======================================================================
# Data loading
# ======================================================================

def load_data():
    """Load Childs analysis panel and merge with HMS exposure."""
    print("Loading analysis datasets...")

    # Childs-based panel
    df = pd.read_parquet(DATA_FILE)
    print(f"  Childs panel: {len(df):,} obs, {df['fips'].nunique():,} counties")

    # HMS exposure panel
    hms = pd.read_parquet(HMS_FILE)
    print(f"  HMS panel: {len(hms):,} obs, {hms['fips'].nunique():,} counties")

    # Merge
    merged = df.merge(hms, on=["fips", "year"], how="inner",
                      suffixes=("", "_hms"))
    print(f"  Merged: {len(merged):,} obs")

    # Drop duplicate state columns from HMS
    drop_cols = [c for c in merged.columns if c.endswith("_hms")]
    merged = merged.drop(columns=drop_cols, errors="ignore")

    # Set panel index
    merged = merged.set_index(["fips", "year"]).sort_index()

    return merged


# ======================================================================
# Part A: Correlation Validation
# ======================================================================

def daily_validation():
    """Agreement metrics at county-day level between Childs and HMS."""
    print("\n" + "=" * 70)
    print("PART A.1: Daily Validation — Childs vs HMS at County-Day Level")
    print("=" * 70)

    if not os.path.exists(CHILDS_DAILY) or not os.path.exists(HMS_DAILY):
        print("  Missing daily files — skipping daily validation")
        print(f"    Childs: {os.path.exists(CHILDS_DAILY)}")
        print(f"    HMS: {os.path.exists(HMS_DAILY)}")
        return

    # Load Childs daily
    print("  Loading Childs daily data...")
    with open(CHILDS_DAILY) as f:
        sep = "\t" if "\t" in f.readline() else ","
    childs = pd.read_csv(CHILDS_DAILY, sep=sep, dtype={"GEOID": str})
    childs = childs.rename(columns={"GEOID": "fips", "smokePM_pred": "smoke_pm25"})
    childs["fips"] = childs["fips"].str.zfill(5)
    childs["date"] = pd.to_datetime(childs["date"], format="%Y%m%d")

    # Load HMS daily
    print("  Loading HMS daily data...")
    hms = pd.read_csv(HMS_DAILY, dtype={"fips": str})
    hms["fips"] = hms["fips"].str.zfill(5)
    hms["date"] = pd.to_datetime(hms["date"])

    # Filter to election windows (Jun 1 to Election Day for each year)
    childs_filtered = []
    hms_filtered = []
    for year, edate_str in ELECTION_DATES.items():
        edate = pd.Timestamp(edate_str)
        start = pd.Timestamp(f"{year}-06-01")

        c = childs[(childs["date"] >= start) & (childs["date"] <= edate)].copy()
        h = hms[(hms["date"] >= start) & (hms["date"] <= edate)].copy()
        childs_filtered.append(c)
        hms_filtered.append(h)

    childs_f = pd.concat(childs_filtered, ignore_index=True)
    hms_f = pd.concat(hms_filtered, ignore_index=True)

    print(f"  Childs daily (election windows): {len(childs_f):,} rows "
          f"(smoke-only: file omits zero-smoke days)")
    print(f"  HMS daily (election windows): {len(hms_f):,} rows "
          f"(complete grid: all county-days)")

    # Use HMS daily as the base (complete county-day grid) and left-join Childs.
    # Childs file only contains days with predicted smoke, so missing = 0.
    merged = hms_f[["fips", "date", "hms_smoke", "hms_density_max"]].merge(
        childs_f[["fips", "date", "smoke_pm25"]],
        on=["fips", "date"], how="left"
    )
    merged["smoke_pm25"] = merged["smoke_pm25"].fillna(0)
    print(f"  Full county-day grid: {len(merged):,}")

    if len(merged) == 0:
        print("  No matched observations — skipping")
        return

    # Binary classification: Childs smoke > 0 vs HMS smoke = 1
    merged["childs_any"] = (merged["smoke_pm25"] > 0).astype(int)

    # Confusion matrix
    tp = ((merged["childs_any"] == 1) & (merged["hms_smoke"] == 1)).sum()
    fp = ((merged["childs_any"] == 1) & (merged["hms_smoke"] == 0)).sum()
    fn = ((merged["childs_any"] == 0) & (merged["hms_smoke"] == 1)).sum()
    tn = ((merged["childs_any"] == 0) & (merged["hms_smoke"] == 0)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / len(merged)
    prevalence_childs = merged["childs_any"].mean()
    prevalence_hms = merged["hms_smoke"].mean()

    # Cohen's kappa
    po = accuracy
    pe = (prevalence_childs * prevalence_hms +
          (1 - prevalence_childs) * (1 - prevalence_hms))
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else np.nan

    print(f"\n  --- Childs (any smoke > 0) vs HMS (smoke = 1) ---")
    print(f"    TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")
    print(f"    Sensitivity (recall): {sensitivity:.3f}")
    print(f"    Specificity: {specificity:.3f}")
    print(f"    Accuracy: {accuracy:.3f}")
    print(f"    Cohen's kappa: {kappa:.3f}")
    print(f"    Prevalence — Childs: {prevalence_childs:.3f}, HMS: {prevalence_hms:.3f}")

    # At haze threshold: Childs > 20 µg/m³ vs HMS medium+heavy
    merged["childs_haze"] = (merged["smoke_pm25"] > 20).astype(int)
    merged["hms_medium_plus"] = (merged["hms_density_max"] >= 2).astype(int)

    tp2 = ((merged["childs_haze"] == 1) & (merged["hms_medium_plus"] == 1)).sum()
    fp2 = ((merged["childs_haze"] == 1) & (merged["hms_medium_plus"] == 0)).sum()
    fn2 = ((merged["childs_haze"] == 0) & (merged["hms_medium_plus"] == 1)).sum()
    tn2 = ((merged["childs_haze"] == 0) & (merged["hms_medium_plus"] == 0)).sum()

    sens2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else np.nan
    spec2 = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else np.nan
    acc2 = (tp2 + tn2) / len(merged)
    p_c2 = merged["childs_haze"].mean()
    p_h2 = merged["hms_medium_plus"].mean()
    pe2 = p_c2 * p_h2 + (1 - p_c2) * (1 - p_h2)
    kappa2 = (acc2 - pe2) / (1 - pe2) if (1 - pe2) > 0 else np.nan

    print(f"\n  --- Childs (>20 µg/m³) vs HMS (Medium + Heavy) ---")
    print(f"    TP={tp2:,}  FP={fp2:,}  FN={fn2:,}  TN={tn2:,}")
    print(f"    Sensitivity: {sens2:.3f}")
    print(f"    Specificity: {spec2:.3f}")
    print(f"    Accuracy: {acc2:.3f}")
    print(f"    Cohen's kappa: {kappa2:.3f}")

    # Correlation at daily level
    corr = merged["smoke_pm25"].corr(merged["hms_density_max"])
    print(f"\n  Daily Pearson correlation (smoke_pm25 vs hms_density_max): {corr:.3f}")


def window_validation(df):
    """Scatter plots and correlations at 30d window level."""
    print("\n" + "=" * 70)
    print("PART A.2: Window-Level Validation (30d)")
    print("=" * 70)

    # Childs vs HMS variable pairs at 30d
    pairs = [
        ("smoke_pm25_mean_30d", "hms_density_mean_30d", "Mean PM2.5 vs Mean Density"),
        ("smoke_frac_haze_30d", "hms_frac_days_30d", "Frac Haze vs Frac Days"),
        ("smoke_frac_usg_30d", "hms_frac_medium_30d", "Frac USG vs Frac Medium+"),
        ("smoke_frac_unhealthy_30d", "hms_frac_heavy_30d", "Frac Unhealthy vs Frac Heavy"),
        ("smoke_days_30d", "hms_smoke_days_30d", "Smoke Days vs HMS Days"),
    ]

    df_reset = df.reset_index()

    # Print correlations
    print(f"\n  {'Pair':<45} {'Pearson r':>10} {'R²':>8} {'N':>8}")
    print("  " + "-" * 75)
    valid_pairs = []
    for childs_var, hms_var, label in pairs:
        if childs_var not in df_reset.columns or hms_var not in df_reset.columns:
            continue
        mask = df_reset[childs_var].notna() & df_reset[hms_var].notna()
        n = mask.sum()
        if n < 10:
            continue
        r = df_reset.loc[mask, childs_var].corr(df_reset.loc[mask, hms_var])
        print(f"  {label:<45} {r:>10.3f} {r**2:>8.3f} {n:>8,}")
        valid_pairs.append((childs_var, hms_var, label, r))

    # Scatter plots
    if valid_pairs:
        os.makedirs(FIG_DIR, exist_ok=True)
        n_plots = min(len(valid_pairs), 4)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
        if n_plots == 1:
            axes = [axes]

        for ax, (cv, hv, lbl, r) in zip(axes, valid_pairs[:n_plots]):
            mask = df_reset[cv].notna() & df_reset[hv].notna()
            ax.scatter(df_reset.loc[mask, cv], df_reset.loc[mask, hv],
                       alpha=0.1, s=5, color="steelblue")
            ax.set_xlabel(f"Childs: {cv.replace('smoke_', '').replace('_30d', '')}")
            ax.set_ylabel(f"HMS: {hv.replace('hms_', '').replace('_30d', '')}")
            ax.set_title(f"{lbl}\nr = {r:.3f}")

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "hms_validation_scatter.png")
        fig.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n  Saved: {fig_path}")

    # Correlation heatmap: all Childs 30d vs all HMS 30d
    childs_30d = [c for c in df_reset.columns if c.startswith("smoke_") and c.endswith("_30d")
                  and not c.startswith("hms_")]
    hms_30d = [c for c in df_reset.columns if c.startswith("hms_") and c.endswith("_30d")]

    if childs_30d and hms_30d:
        corr_matrix = df_reset[childs_30d + hms_30d].corr().loc[childs_30d, hms_30d]

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(hms_30d)))
        ax.set_xticklabels([c.replace("hms_", "").replace("_30d", "") for c in hms_30d],
                           rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(childs_30d)))
        ax.set_yticklabels([c.replace("smoke_", "").replace("_30d", "") for c in childs_30d],
                           fontsize=8)
        ax.set_title("Childs vs HMS: 30-Day Window Correlations")
        fig.colorbar(im, label="Pearson r")

        # Annotate cells
        for i in range(len(childs_30d)):
            for j in range(len(hms_30d)):
                val = corr_matrix.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=7)

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "hms_correlation_heatmap.png")
        fig.savefig(fig_path, dpi=150)
        plt.close()
        print(f"  Saved: {fig_path}")


def timeseries_validation():
    """National daily time series: Childs PM2.5 vs HMS coverage overlaid."""
    print("\n" + "=" * 70)
    print("PART A.3: Time Series Validation")
    print("=" * 70)

    if not os.path.exists(CHILDS_DAILY) or not os.path.exists(HMS_DAILY):
        print("  Missing daily files — skipping")
        return

    # Load dailies
    print("  Loading daily data...")
    with open(CHILDS_DAILY) as f:
        sep = "\t" if "\t" in f.readline() else ","
    childs = pd.read_csv(CHILDS_DAILY, sep=sep, dtype={"GEOID": str})
    childs = childs.rename(columns={"GEOID": "fips", "smokePM_pred": "smoke_pm25"})
    childs["fips"] = childs["fips"].str.zfill(5)
    childs["date"] = pd.to_datetime(childs["date"], format="%Y%m%d")

    hms = pd.read_csv(HMS_DAILY, dtype={"fips": str})
    hms["fips"] = hms["fips"].str.zfill(5)
    hms["date"] = pd.to_datetime(hms["date"])

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for idx, (year, edate_str) in enumerate(sorted(ELECTION_DATES.items())):
        ax = axes[idx // 2][idx % 2]
        edate = pd.Timestamp(edate_str)
        start = pd.Timestamp(f"{year}-06-01")

        # Childs: national daily mean PM2.5
        c = childs[(childs["date"] >= start) & (childs["date"] <= edate)]
        childs_daily_mean = c.groupby("date")["smoke_pm25"].mean()

        # HMS: national daily fraction of counties with smoke
        h = hms[(hms["date"] >= start) & (hms["date"] <= edate)]
        n_counties = hms["fips"].nunique()
        hms_daily_frac = h.groupby("date")["hms_smoke"].sum() / n_counties

        ax2 = ax.twinx()
        ax.plot(childs_daily_mean.index, childs_daily_mean.values,
                color="steelblue", alpha=0.7, linewidth=0.8, label="Childs PM2.5")
        ax2.plot(hms_daily_frac.index, hms_daily_frac.values,
                 color="firebrick", alpha=0.7, linewidth=0.8, label="HMS frac")

        ax.set_ylabel("Mean smoke PM2.5 (µg/m³)", color="steelblue", fontsize=9)
        ax2.set_ylabel("HMS county fraction", color="firebrick", fontsize=9)
        ax.set_title(f"{year}")
        ax.axvline(edate, color="black", linestyle="--", alpha=0.5, linewidth=0.8)

        # Compute correlation for this year
        combined = pd.DataFrame({
            "childs": childs_daily_mean,
            "hms": hms_daily_frac,
        }).dropna()
        if len(combined) > 10:
            r = combined["childs"].corr(combined["hms"])
            ax.text(0.02, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                    fontsize=9, va="top")

    fig.suptitle("Childs ML-Predicted PM2.5 vs HMS Satellite Smoke Coverage\n"
                 "(National Daily Means, Jun 1 – Election Day)", fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "hms_validation_timeseries.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")


# ======================================================================
# Part B: Regression Robustness
# ======================================================================

def hms_buildup_table(df):
    """Build-up table using HMS treatment, side-by-side with Childs."""
    print("\n" + "=" * 70)
    print("PART B.1: Build-Up Table — Childs vs HMS")
    print("  (1) Raw OLS  (2) County+Year FE  (3) +Controls  (4) +State Trends")
    print("=" * 70)

    childs_var = "smoke_pm25_mean_30d"
    hms_var = "hms_frac_days_30d"

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
        hms_results = []

        for smoke_var, result_list, data_label in [
            (childs_var, childs_results, "Childs"),
            (hms_var, hms_results, "HMS"),
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

            for i, res in enumerate([res1, res2, res3, res4], 1):
                if res is not None:
                    coef = res.params.get(smoke_var, np.nan)
                    se = res.std_errors.get(smoke_var, np.nan)
                    pval = res.pvalues.get(smoke_var, np.nan)
                    n = int(res.nobs)
                    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                    result_list.append({"spec": i, "coef": coef, "se": se,
                                        "pval": pval, "n": n, "stars": stars})
                else:
                    result_list.append({"spec": i, "coef": np.nan, "se": np.nan,
                                        "pval": np.nan, "n": 0, "stars": ""})

        results[dep_var] = {"childs": childs_results, "hms": hms_results}

    # Print formatted side-by-side table
    print("\n" + "=" * 100)
    print(f"  {'':35s} {'(1) Raw OLS':>14s} {'(2) TWFE':>14s} {'(3) +Controls':>14s} {'(4) +St.Trends':>14s}")
    print("=" * 100)

    for dep_var, panel_label in outcomes:
        r = results[dep_var]
        print(f"\n  {panel_label}")

        # Childs row
        coef_line = f"  {'Childs PM2.5 (30d)':<35s}"
        se_line = f"  {'':35s}"
        for cr in r["childs"]:
            coef_line += f" {cr['coef']:>12.5f}{cr['stars']:<2s}"
            se_line += f" ({cr['se']:>11.5f}) "
        print(coef_line)
        print(se_line)

        # HMS row
        coef_line = f"  {'HMS frac days (30d)':<35s}"
        se_line = f"  {'':35s}"
        for hr in r["hms"]:
            coef_line += f" {hr['coef']:>12.5f}{hr['stars']:<2s}"
            se_line += f" ({hr['se']:>11.5f}) "
        print(coef_line)
        print(se_line)

        # Standardized comparison (Spec 3)
        childs_sd = df[childs_var].std()
        hms_sd = df[hms_var].std()
        childs_beta = r["childs"][2]["coef"] * childs_sd  # Spec 3
        hms_beta = r["hms"][2]["coef"] * hms_sd
        print(f"  {'Standardized β (Spec 3)':<35s} "
              f"Childs: {childs_beta:.6f}   HMS: {hms_beta:.6f}")

    print("\n" + "-" * 100)
    print(f"  {'County FE':<35s} {'':>14s} {'Yes':>14s} {'Yes':>14s} {'Yes':>14s}")
    print(f"  {'Year FE':<35s} {'':>14s} {'Yes':>14s} {'Yes':>14s} {'Yes':>14s}")
    print(f"  {'Controls':<35s} {'':>14s} {'':>14s} {'Yes':>14s} {'Yes':>14s}")
    print(f"  {'State trends':<35s} {'':>14s} {'':>14s} {'':>14s} {'Yes':>14s}")
    print("=" * 100)

    # Treatment variable statistics
    print(f"\n  Treatment Variable Summary:")
    print(f"    Childs PM2.5 mean (30d): mean={df[childs_var].mean():.4f}, "
          f"SD={childs_sd:.4f}, max={df[childs_var].max():.3f}")
    print(f"    HMS frac days (30d):     mean={df[hms_var].mean():.4f}, "
          f"SD={hms_sd:.4f}, max={df[hms_var].max():.3f}")
    print(f"    Correlation:             r = {df[[childs_var, hms_var]].dropna().corr().iloc[0,1]:.3f}")

    return results


def hms_comparison_figure(df, results):
    """Side-by-side coefficient plot: Childs vs HMS across 4 specs, 4 outcomes."""
    print("\n" + "=" * 70)
    print("PART B.2: Comparison Figure")
    print("=" * 70)

    if results is None:
        print("  No results to plot — skipping")
        return

    childs_var = "smoke_pm25_mean_30d"
    hms_var = "hms_frac_days_30d"

    childs_sd = df.reset_index()[childs_var].std()
    hms_sd = df.reset_index()[hms_var].std()

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
        hms_coefs = [hr["coef"] * hms_sd for hr in r["hms"]]
        hms_ses = [hr["se"] * hms_sd for hr in r["hms"]]

        ax.bar(x_pos - width/2, childs_coefs, width, label="Childs",
               color="steelblue", alpha=0.7)
        ax.errorbar(x_pos - width/2, childs_coefs,
                    yerr=[1.96 * s for s in childs_ses],
                    fmt="none", color="steelblue", capsize=3)

        ax.bar(x_pos + width/2, hms_coefs, width, label="HMS",
               color="firebrick", alpha=0.7)
        ax.errorbar(x_pos + width/2, hms_coefs,
                    yerr=[1.96 * s for s in hms_ses],
                    fmt="none", color="firebrick", capsize=3)

        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["OLS", "TWFE", "+Ctrl", "+Trend"], fontsize=9)
        ax.set_title(dep_label, fontsize=11)
        ax.set_ylabel("Standardized effect (β × SD)", fontsize=9)
        if idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("Childs ML-Predicted PM2.5 vs HMS Satellite Smoke\n"
                 "Standardized Coefficients (β × SD_treatment) by Specification",
                 fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "hms_buildup_comparison.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved: {fig_path}")


def hms_threshold_comparison(df):
    """Compare HMS density thresholds vs Childs PM2.5 thresholds, Spec 3."""
    print("\n" + "=" * 70)
    print("PART B.3: Threshold Comparison — HMS Density vs Childs PM2.5 (Spec 3)")
    print("=" * 70)

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    # Threshold pairs
    thresholds = [
        # (HMS var, HMS label, Childs var, Childs label)
        ("hms_frac_days_30d", "HMS: Any smoke", "smoke_frac_haze_30d", "Childs: Haze (>20)"),
        ("hms_frac_medium_30d", "HMS: Medium+", "smoke_frac_usg_30d", "Childs: USG (>35.5)"),
        ("hms_frac_heavy_30d", "HMS: Heavy", "smoke_frac_unhealthy_30d", "Childs: Unhealthy (>55.5)"),
    ]

    outcomes = [
        ("turnout_rate", "Turnout Rate"),
        ("log_total_votes", "Log Total Votes"),
        ("dem_vote_share", "DEM Vote Share"),
        ("incumbent_vote_share", "Incumbent V.S."),
    ]

    print(f"\n  {'Treatment':<30s}", end="")
    for _, dep_label in outcomes:
        print(f" {dep_label:>18s}", end="")
    print()
    print("  " + "-" * 105)

    for hms_var, hms_label, childs_var, childs_label in thresholds:
        # Nonzero counts
        for smoke_var, label in [(hms_var, hms_label), (childs_var, childs_label)]:
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
                    print(f" {coef:>15.5f}{stars:<3s}", end="")
                else:
                    print(f" {'---':>18s}", end="")
            print(f"  (N_nz={nonzero:,})")
        print()


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 70)
    print("HMS SMOKE PLUME ROBUSTNESS CHECK")
    print("  Childs et al. ML-Predicted PM2.5 vs NOAA HMS Satellite Smoke")
    print("=" * 70)

    # Part A: Validation
    daily_validation()

    df = load_data()
    window_validation(df)
    timeseries_validation()

    # Part B: Regression Robustness
    results = hms_buildup_table(df)
    hms_comparison_figure(df, results)
    hms_threshold_comparison(df)

    print("\n" + "=" * 70)
    print("HMS robustness check complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

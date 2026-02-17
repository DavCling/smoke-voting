#!/usr/bin/env python3
"""
Phase 4: Analyze the relationship between wildfire smoke and voting behavior.

Specifications:
  A: Two-way fixed effects (county + year FE)
  B: Incumbent punishment (does smoke shift votes against the incumbent?)
  C: Pro-environment shift (does smoke shift votes toward Democrats?)
  D: Turnout effects (does smoke suppress turnout?)

Plus heterogeneity tests and robustness checks.
"""

import os
import sys
import warnings
from datetime import timedelta
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
SMOKE_FILE = os.path.join(BASE_DIR, "data", "smoke", "smoke_pm25_county_daily.csv")
FIG_DIR = os.path.join(BASE_DIR, "output", "figures")

ELECTION_DATES = {
    2008: "2008-11-04",
    2012: "2012-11-06",
    2016: "2016-11-08",
    2020: "2020-11-03",
}


def load_data():
    """Load analysis dataset and prepare panel structure."""
    print("Loading analysis dataset...")
    df = pd.read_parquet(DATA_FILE)
    print(f"  {len(df):,} observations, {df['fips'].nunique():,} counties, "
          f"{df['year'].nunique()} elections")

    # Set panel index
    df = df.set_index(["fips", "year"])
    df = df.sort_index()

    return df


def run_twfe(df, dep_var, smoke_var, controls=None, absorb_entity=True, absorb_time=True,
             state_year_fe=False, drop_absorbed=False, label=""):
    """Run a two-way fixed effects regression using linearmodels PanelOLS."""
    # Drop rows with missing dependent or independent variable
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
        return
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


def spec_a_dem_vote_share(df):
    """Specification A: Smoke -> DEM vote share (pro-environment shift)."""
    print("\n" + "=" * 70)
    print("SPECIFICATION A: Smoke Exposure → Democratic Vote Share")
    print("  (Tests salience/pro-environment mechanism)")
    print("=" * 70)

    results = []

    # Main specification: 60-day mean smoke PM2.5
    smoke_vars = [
        ("smoke_pm25_mean_60d", "Mean smoke PM2.5 (60 days)"),
        ("smoke_days_60d", "Smoke days (60 days)"),
        ("smoke_days_severe_60d", "Severe smoke days (60 days)"),
        ("smoke_pm25_mean_7d", "Mean smoke PM2.5 (7 days)"),
        ("smoke_pm25_mean_season", "Mean smoke PM2.5 (fire season)"),
    ]

    for svar, label in smoke_vars:
        if svar not in df.columns:
            continue
        res = run_twfe(df, "dem_vote_share", svar, label=label)
        r = print_result(res, label, svar)
        if r:
            r["smoke_var"] = svar
            results.append(r)

    # With lagged vote share control
    if "dem_vote_share_lag" in df.columns:
        print("\n  --- With lagged DEM vote share control ---")
        res = run_twfe(df, "dem_vote_share", "smoke_pm25_mean_60d",
                       controls=["dem_vote_share_lag"],
                       label="Mean PM2.5 (60d) + lag control")
        r = print_result(res, "Mean PM2.5 (60d) + lag control", "smoke_pm25_mean_60d")
        if r:
            r["smoke_var"] = "smoke_pm25_mean_60d"
            results.append(r)

    return results


def spec_b_incumbent_punishment(df):
    """Specification B: Smoke -> Incumbent vote share (negative affect)."""
    print("\n" + "=" * 70)
    print("SPECIFICATION B: Smoke Exposure → Incumbent Party Vote Share")
    print("  (Tests negative emotion / incumbent punishment mechanism)")
    print("=" * 70)

    results = []
    smoke_vars = [
        ("smoke_pm25_mean_60d", "Mean smoke PM2.5 (60 days)"),
        ("smoke_days_60d", "Smoke days (60 days)"),
        ("smoke_days_severe_60d", "Severe smoke days (60 days)"),
        ("smoke_pm25_mean_7d", "Mean smoke PM2.5 (7 days)"),
    ]

    for svar, label in smoke_vars:
        if svar not in df.columns:
            continue
        res = run_twfe(df, "incumbent_vote_share", svar, label=label)
        r = print_result(res, label, svar)
        if r:
            r["smoke_var"] = svar
            results.append(r)

    return results


def spec_c_turnout(df):
    """Specification C: Smoke -> Turnout (disruption mechanism)."""
    print("\n" + "=" * 70)
    print("SPECIFICATION C: Smoke Exposure → Voter Turnout")
    print("  (Tests disruption / suppression mechanism)")
    print("=" * 70)

    # Use log total votes as turnout proxy
    results = []
    smoke_vars = [
        ("smoke_pm25_mean_60d", "Mean smoke PM2.5 (60 days)"),
        ("smoke_days_60d", "Smoke days (60 days)"),
        ("smoke_pm25_mean_7d", "Mean smoke PM2.5 (7 days)"),
        ("smoke_days_severe_60d", "Severe smoke days (60 days)"),
    ]

    for svar, label in smoke_vars:
        if svar not in df.columns:
            continue
        res = run_twfe(df, "log_total_votes", svar, label=label)
        r = print_result(res, label, svar)
        if r:
            r["smoke_var"] = svar
            results.append(r)

    return results


def threshold_comparison(df):
    """Compare fraction-above-threshold treatment at 20, 35.5, and 55.5 µg/m³."""
    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON: Fraction of Days Above Threshold (30d)")
    print("  Haze (>20), USG (>35.5), Unhealthy (>55.5)")
    print("=" * 70)

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    thresholds = [
        ("smoke_frac_haze_30d", "Haze (>20 µg/m³)"),
        ("smoke_frac_usg_30d", "USG (>35.5 µg/m³)"),
        ("smoke_frac_unhealthy_30d", "Unhealthy (>55.5 µg/m³)"),
    ]

    outcomes = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
        ("turnout_rate", "Turnout rate"),
    ]

    for smoke_var, thresh_label in thresholds:
        if smoke_var not in df.columns:
            print(f"  WARNING: {smoke_var} not found in dataset")
            continue

        vals = df[smoke_var].dropna()
        nonzero = (vals > 0).sum()
        print(f"\n  {thresh_label}: {nonzero:,}/{len(vals):,} nonzero ({100*nonzero/len(vals):.1f}%)")
        print(f"    mean={vals.mean():.4f}, max={vals.max():.4f}")

        for dep_var, dep_label in outcomes:
            # Spec 2: TWFE only
            res2 = run_twfe(df, dep_var, smoke_var,
                            label=f"{thresh_label} TWFE: {dep_label}")
            if res2 is not None:
                coef2 = res2.params.get(smoke_var, np.nan)
                se2 = res2.std_errors.get(smoke_var, np.nan)
                pval2 = res2.pvalues.get(smoke_var, np.nan)
                stars2 = "***" if pval2 < 0.01 else "**" if pval2 < 0.05 else "*" if pval2 < 0.10 else ""
            else:
                coef2, se2, pval2, stars2 = np.nan, np.nan, np.nan, ""

            # Spec 3: TWFE + controls
            res3 = run_twfe(df, dep_var, smoke_var, controls=available,
                            label=f"{thresh_label} +controls: {dep_label}")
            if res3 is not None:
                coef3 = res3.params.get(smoke_var, np.nan)
                se3 = res3.std_errors.get(smoke_var, np.nan)
                pval3 = res3.pvalues.get(smoke_var, np.nan)
                stars3 = "***" if pval3 < 0.01 else "**" if pval3 < 0.05 else "*" if pval3 < 0.10 else ""
            else:
                coef3, se3, pval3, stars3 = np.nan, np.nan, np.nan, ""

            print(f"    {dep_label:25s}  TWFE: β={coef2:.6f}{stars2:3s} (SE={se2:.6f})"
                  f"  +Ctrl: β={coef3:.6f}{stars3:3s} (SE={se3:.6f})")


def state_year_fe_regressions(df):
    """Robustness check: State×Year FE instead of Year FE."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS: State×Year Fixed Effects (Presidential)")
    print("  (Absorbs state-level time-varying shocks)")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"
    specs = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
        ("turnout_rate", "Turnout rate"),
    ]

    for dep_var, dep_label in specs:
        res = run_twfe(df, dep_var, smoke_var, state_year_fe=True,
                       label=f"State×Year FE: {dep_label}")
        print_result(res, f"State×Year FE: {dep_label}", smoke_var)


def heterogeneity_tests(df):
    """Test for heterogeneous effects by prior partisanship and urban/rural."""
    print("\n" + "=" * 70)
    print("HETEROGENEITY TESTS")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"
    dep_var = "dem_vote_share"

    # By prior partisanship
    if "dem_tercile" in df.columns:
        print("\n  --- By Prior Partisanship (lagged DEM vote share tercile) ---")
        for tercile in ["R-leaning", "Swing", "D-leaning"]:
            subset = df[df["dem_tercile"] == tercile]
            if len(subset) < 100:
                print(f"  SKIP {tercile}: {len(subset)} obs")
                continue
            res = run_twfe(subset, dep_var, smoke_var, label=f"{tercile} counties")
            print_result(res, f"{tercile} counties", smoke_var)

    # By urban/rural proxy
    if "urban_proxy" in df.columns:
        print("\n  --- By Urban/Rural ---")
        for val, label in [(1, "Urban"), (0, "Rural")]:
            subset = df[df["urban_proxy"] == val]
            if len(subset) < 100:
                continue
            res = run_twfe(subset, dep_var, smoke_var, label=f"{label} counties")
            print_result(res, f"{label} counties", smoke_var)


def dose_response_analysis(df):
    """Test for non-linear dose-response patterns."""
    print("\n" + "=" * 70)
    print("DOSE-RESPONSE ANALYSIS")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"

    # Create quintiles of smoke exposure
    df_reset = df.reset_index()
    df_reset["smoke_quintile"] = pd.qcut(
        df_reset[smoke_var], 5, labels=False, duplicates="drop"
    )

    print(f"\n  Smoke PM2.5 (30d) quintile boundaries:")
    for q in range(5):
        vals = df_reset[df_reset["smoke_quintile"] == q][smoke_var]
        print(f"    Q{q+1}: [{vals.min():.3f}, {vals.max():.3f}] µg/m³  (n={len(vals):,})")

    # Create dummies (Q1 = reference)
    for q in range(1, 5):
        df_reset[f"smoke_q{q+1}"] = (df_reset["smoke_quintile"] == q).astype(int)

    df_panel = df_reset.set_index(["fips", "year"])
    q_vars = [f"smoke_q{q}" for q in range(2, 6)]

    try:
        y = df_panel["dem_vote_share"].dropna()
        x_cols = [c for c in q_vars if c in df_panel.columns]
        common_idx = y.index.intersection(df_panel[x_cols].dropna().index)
        y = y.loc[common_idx]
        x = sm.add_constant(df_panel.loc[common_idx, x_cols])

        mod = PanelOLS(y, x, entity_effects=True, time_effects=True, check_rank=False)
        res = mod.fit(cov_type="clustered", cluster_entity=True)

        print(f"\n  DEM vote share ~ Smoke quintile dummies (Q1=reference)")
        print(f"  County FE + Year FE, clustered SEs")
        print(f"  N = {int(res.nobs):,}")
        for qvar in x_cols:
            coef = res.params.get(qvar, np.nan)
            se = res.std_errors.get(qvar, np.nan)
            pval = res.pvalues.get(qvar, np.nan)
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            print(f"    {qvar}: β={coef:.6f} {stars} (SE={se:.6f}, p={pval:.4f})")
    except Exception as e:
        print(f"  ERROR: {e}")


def event_study_windows(df):
    """Plot how the effect varies across different pre-election windows."""
    print("\n" + "=" * 70)
    print("EVENT STUDY: Effect by Pre-Election Window Length")
    print("=" * 70)

    windows = ["7d", "30d", "60d", "90d", "season"]
    window_labels = ["7 days", "30 days", "60 days", "90 days", "Fire season"]

    coefs, ses, labels_out = [], [], []

    for window, wlabel in zip(windows, window_labels):
        svar = f"smoke_pm25_mean_{window}"
        if svar not in df.columns:
            continue

        res = run_twfe(df, "dem_vote_share", svar, label=wlabel)
        if res is not None:
            coef = res.params.get(svar, np.nan)
            se = res.std_errors.get(svar, np.nan)
            pval = res.pvalues.get(svar, np.nan)
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            print(f"  {wlabel:15s}: β={coef:.6f} {stars} (SE={se:.6f})")
            coefs.append(coef)
            ses.append(se)
            labels_out.append(wlabel)

    # Plot
    if len(coefs) > 1:
        os.makedirs(FIG_DIR, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = range(len(coefs))
        ax.errorbar(x_pos, coefs, yerr=[1.96 * s for s in ses],
                     fmt="o-", capsize=5, color="steelblue", linewidth=2, markersize=8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels_out, rotation=15)
        ax.set_ylabel("Effect on DEM vote share")
        ax.set_xlabel("Pre-election smoke exposure window")
        ax.set_title("Effect of Smoke PM2.5 on Democratic Vote Share\nby Pre-Election Window")
        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "event_study_windows.png")
        fig.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n  Saved figure: {fig_path}")


def temporal_dynamics_7day(df):
    """7-day temporal dynamics: exclusive and cumulative windows for all 3 outcomes."""
    print("\n" + "=" * 70)
    print("TEMPORAL DYNAMICS: 7-Day Non-Overlapping Windows")
    print("=" * 70)

    # Load raw daily smoke data
    print("  Loading raw daily smoke data...")
    with open(SMOKE_FILE) as f:
        sep = "\t" if "\t" in f.readline() else ","
    smoke_raw = pd.read_csv(SMOKE_FILE, sep=sep, dtype={"GEOID": str})
    smoke_raw = smoke_raw.rename(columns={"GEOID": "fips", "smokePM_pred": "smoke_pm25"})
    smoke_raw["fips"] = smoke_raw["fips"].str.zfill(5)
    smoke_raw["date"] = pd.to_datetime(smoke_raw["date"], format="%Y%m%d")

    # Compute 13 non-overlapping 7-day bins (days 0-6, 7-13, ..., 84-90 before election)
    n_bins = 13
    bin_vars = []
    for yr, edate_str in ELECTION_DATES.items():
        edate = pd.Timestamp(edate_str)
        earliest = edate - timedelta(days=n_bins * 7)
        yr_smoke = smoke_raw[(smoke_raw["date"] > earliest) & (smoke_raw["date"] <= edate)].copy()

        for b in range(n_bins):
            bin_start = edate - timedelta(days=(b + 1) * 7)
            bin_end = edate - timedelta(days=b * 7)
            w = yr_smoke[(yr_smoke["date"] > bin_start) & (yr_smoke["date"] <= bin_end)]
            bin_mean = w.groupby("fips")["smoke_pm25"].mean().rename(f"smoke_bin_{b}")
            if b == 0:
                yr_bins = bin_mean.to_frame()
            else:
                yr_bins = yr_bins.join(bin_mean, how="outer")

        yr_bins["year"] = yr
        yr_bins = yr_bins.reset_index()
        bin_vars.append(yr_bins)

    bins_df = pd.concat(bin_vars, ignore_index=True).fillna(0)
    bins_df = bins_df.set_index(["fips", "year"])

    # Merge with main panel
    df_merged = df.join(bins_df, how="inner")
    bin_cols = [f"smoke_bin_{b}" for b in range(n_bins)]

    print(f"  Merged panel: {len(df_merged):,} obs")

    outcomes = [
        ("dem_vote_share", "DEM Vote Share"),
        ("incumbent_vote_share", "Incumbent Vote Share"),
        ("log_total_votes", "Log Total Votes"),
        ("turnout_rate", "Turnout Rate"),
    ]

    # Bin labels (days before election)
    bin_labels = [f"{b*7}-{(b+1)*7-1}d" for b in range(n_bins)]

    # Collect results for plotting
    all_results = {}

    for dep_var, dep_label in outcomes:
        print(f"\n  --- {dep_label} ---")

        # EXCLUSIVE: all 13 bins as simultaneous regressors
        cols_needed = [dep_var] + bin_cols
        subset = df_merged[cols_needed].dropna()
        if len(subset) < 100:
            print(f"    SKIP: only {len(subset)} obs")
            continue

        y = subset[dep_var]
        x = sm.add_constant(subset[bin_cols])

        try:
            mod = PanelOLS(y, x, entity_effects=True, time_effects=True, check_rank=False)
            res_excl = mod.fit(cov_type="clustered", cluster_entity=True)
            excl_coefs = [res_excl.params.get(c, np.nan) for c in bin_cols]
            excl_ses = [res_excl.std_errors.get(c, np.nan) for c in bin_cols]
            excl_pvals = [res_excl.pvalues.get(c, np.nan) for c in bin_cols]

            print(f"    Exclusive windows (N={int(res_excl.nobs):,}):")
            for b in range(n_bins):
                stars = "***" if excl_pvals[b] < 0.01 else "**" if excl_pvals[b] < 0.05 else "*" if excl_pvals[b] < 0.10 else ""
                print(f"      Bin {b} ({bin_labels[b]}): β={excl_coefs[b]:.6f} {stars} (SE={excl_ses[b]:.6f})")
        except Exception as e:
            print(f"    ERROR (exclusive): {e}")
            excl_coefs, excl_ses = [np.nan] * n_bins, [np.nan] * n_bins

        # CUMULATIVE: 13 separate regressions with expanding windows
        cumul_coefs, cumul_ses = [], []
        for k in range(n_bins):
            cum_cols = bin_cols[:k + 1]
            cum_var = f"smoke_cumul_{k}"
            subset[cum_var] = subset[cum_cols].mean(axis=1)
            y_k = subset[dep_var]
            x_k = sm.add_constant(subset[[cum_var]])

            try:
                mod_k = PanelOLS(y_k, x_k, entity_effects=True, time_effects=True, check_rank=False)
                res_k = mod_k.fit(cov_type="clustered", cluster_entity=True)
                cumul_coefs.append(res_k.params.get(cum_var, np.nan))
                cumul_ses.append(res_k.std_errors.get(cum_var, np.nan))
            except Exception:
                cumul_coefs.append(np.nan)
                cumul_ses.append(np.nan)

        print(f"    Cumulative windows:")
        for k in range(n_bins):
            days_end = (k + 1) * 7
            pval_k = np.nan
            if not np.isnan(cumul_coefs[k]) and cumul_ses[k] > 0:
                from scipy import stats as sp_stats
                t_stat = cumul_coefs[k] / cumul_ses[k]
                pval_k = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=100))
            stars = "***" if pval_k < 0.01 else "**" if pval_k < 0.05 else "*" if pval_k < 0.10 else ""
            print(f"      0-{days_end}d: β={cumul_coefs[k]:.6f} {stars} (SE={cumul_ses[k]:.6f})")

        all_results[dep_var] = {
            "label": dep_label,
            "excl_coefs": excl_coefs, "excl_ses": excl_ses,
            "cumul_coefs": cumul_coefs, "cumul_ses": cumul_ses,
        }

    # Plot figure: len(outcomes) rows × 2 cols
    if all_results:
        os.makedirs(FIG_DIR, exist_ok=True)
        n_rows = len(outcomes)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        x_pos = np.arange(n_bins)

        for row_idx, (dep_var, dep_label) in enumerate(outcomes):
            if dep_var not in all_results:
                continue
            r = all_results[dep_var]

            # Exclusive (left column)
            ax = axes[row_idx, 0]
            ax.errorbar(x_pos, r["excl_coefs"], yerr=[1.96 * s for s in r["excl_ses"]],
                        fmt="o-", capsize=3, color="steelblue", linewidth=1.5, markersize=5)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(bin_labels, rotation=45, fontsize=7)
            ax.set_ylabel(f"Effect on {dep_label}")
            if row_idx == 0:
                ax.set_title("Exclusive 7-Day Windows")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Days before election")

            # Cumulative (right column)
            ax = axes[row_idx, 1]
            ax.errorbar(x_pos, r["cumul_coefs"], yerr=[1.96 * s for s in r["cumul_ses"]],
                        fmt="s-", capsize=3, color="firebrick", linewidth=1.5, markersize=5)
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            cum_labels = [f"0-{(k+1)*7}d" for k in range(n_bins)]
            ax.set_xticks(x_pos)
            ax.set_xticklabels(cum_labels, rotation=45, fontsize=7)
            if row_idx == 0:
                ax.set_title("Cumulative Windows")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Cumulative window length")

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "event_study_7day_windows.png")
        fig.savefig(fig_path, dpi=150)
        plt.close()
        print(f"\n  Saved figure: {fig_path}")


def temporal_dynamics_controls(df):
    """7-day temporal dynamics with controls (Spec 3) for mean PM2.5 and frac haze."""
    print("\n" + "=" * 70)
    print("TEMPORAL DYNAMICS WITH CONTROLS: 7-Day Windows")
    print("  Treatment vars: Mean PM2.5, Fraction Haze (>20 µg/m³)")
    print("=" * 70)

    HAZE_THRESHOLD = 20.0

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]
    print(f"  Controls: {available}")

    # Load raw daily smoke data
    print("  Loading raw daily smoke data...")
    with open(SMOKE_FILE) as f:
        sep = "\t" if "\t" in f.readline() else ","
    smoke_raw = pd.read_csv(SMOKE_FILE, sep=sep, dtype={"GEOID": str})
    smoke_raw = smoke_raw.rename(columns={"GEOID": "fips", "smokePM_pred": "smoke_pm25"})
    smoke_raw["fips"] = smoke_raw["fips"].str.zfill(5)
    smoke_raw["date"] = pd.to_datetime(smoke_raw["date"], format="%Y%m%d")

    # Compute 13 non-overlapping 7-day bins
    n_bins = 13
    bin_vars = []
    for yr, edate_str in ELECTION_DATES.items():
        edate = pd.Timestamp(edate_str)
        earliest = edate - timedelta(days=n_bins * 7)
        yr_smoke = smoke_raw[(smoke_raw["date"] > earliest) & (smoke_raw["date"] <= edate)].copy()

        yr_bins = None
        for b in range(n_bins):
            bin_start = edate - timedelta(days=(b + 1) * 7)
            bin_end = edate - timedelta(days=b * 7)
            w = yr_smoke[(yr_smoke["date"] > bin_start) & (yr_smoke["date"] <= bin_end)]

            bin_mean = w.groupby("fips")["smoke_pm25"].mean().rename(f"mean_bin_{b}")
            bin_frac = w.groupby("fips")["smoke_pm25"].apply(
                lambda x: (x > HAZE_THRESHOLD).mean()
            ).rename(f"frac_bin_{b}")

            if yr_bins is None:
                yr_bins = pd.concat([bin_mean, bin_frac], axis=1)
            else:
                yr_bins = yr_bins.join(bin_mean, how="outer").join(bin_frac, how="outer")

        yr_bins["year"] = yr
        yr_bins = yr_bins.reset_index()
        bin_vars.append(yr_bins)

    bins_df = pd.concat(bin_vars, ignore_index=True).fillna(0)
    bins_df = bins_df.set_index(["fips", "year"])

    df_merged = df.join(bins_df, how="inner")
    print(f"  Merged panel: {len(df_merged):,} obs")

    outcomes = [
        ("dem_vote_share", "DEM Vote Share"),
        ("incumbent_vote_share", "Incumbent Vote Share"),
        ("log_total_votes", "Log Total Votes"),
        ("turnout_rate", "Turnout Rate"),
    ]

    treatments = [
        ("mean_bin", "Mean Smoke PM$_{2.5}$", "Mean Smoke PM2.5",
         "temporal_7day_mean_controls.png"),
        ("frac_bin", "Frac. Days > 20 µg/m³ (Haze)", "Frac. Days Haze",
         "temporal_7day_frac_controls.png"),
    ]

    bin_labels = [f"{b*7}\u2013{(b+1)*7-1}d" for b in range(n_bins)]
    cum_labels = [f"0\u2013{(k+1)*7}d" for k in range(n_bins)]

    all_treat_results = {}  # collect across treatments for combined figure

    for prefix, fig_treat_label, print_treat_label, fig_name in treatments:
        print(f"\n  === Treatment: {print_treat_label} ===")
        bin_cols = [f"{prefix}_{b}" for b in range(n_bins)]

        all_results = {}

        for dep_var, dep_label in outcomes:
            print(f"\n    --- {dep_label} ---")

            # EXCLUSIVE: all 13 bins + controls as simultaneous regressors
            cols_needed = [dep_var] + bin_cols + available
            subset = df_merged[cols_needed].dropna().copy()
            if len(subset) < 100:
                print(f"      SKIP: only {len(subset)} obs")
                continue

            y = subset[dep_var]
            x = sm.add_constant(subset[bin_cols + available])

            try:
                mod = PanelOLS(y, x, entity_effects=True, time_effects=True,
                               check_rank=False, drop_absorbed=True)
                res_excl = mod.fit(cov_type="clustered", cluster_entity=True)
                excl_coefs = [res_excl.params.get(c, np.nan) for c in bin_cols]
                excl_ses = [res_excl.std_errors.get(c, np.nan) for c in bin_cols]
                excl_pvals = [res_excl.pvalues.get(c, np.nan) for c in bin_cols]

                print(f"      Exclusive (N={int(res_excl.nobs):,}):")
                for b in range(n_bins):
                    stars = "***" if excl_pvals[b] < 0.01 else "**" if excl_pvals[b] < 0.05 else "*" if excl_pvals[b] < 0.10 else ""
                    print(f"        {bin_labels[b]}: β={excl_coefs[b]:.6f} {stars} (SE={excl_ses[b]:.6f})")
            except Exception as e:
                print(f"      ERROR (exclusive): {e}")
                excl_coefs, excl_ses = [np.nan] * n_bins, [np.nan] * n_bins

            # CUMULATIVE: expanding windows + controls
            cumul_coefs, cumul_ses = [], []
            for k in range(n_bins):
                cum_cols_k = bin_cols[:k + 1]
                cum_var = f"{prefix}_cumul_{k}"
                subset[cum_var] = subset[cum_cols_k].mean(axis=1)
                y_k = subset[dep_var]
                x_k = sm.add_constant(subset[[cum_var] + available])

                try:
                    mod_k = PanelOLS(y_k, x_k, entity_effects=True, time_effects=True,
                                     check_rank=False, drop_absorbed=True)
                    res_k = mod_k.fit(cov_type="clustered", cluster_entity=True)
                    cumul_coefs.append(res_k.params.get(cum_var, np.nan))
                    cumul_ses.append(res_k.std_errors.get(cum_var, np.nan))
                except Exception:
                    cumul_coefs.append(np.nan)
                    cumul_ses.append(np.nan)

            print(f"      Cumulative:")
            for k in range(n_bins):
                pval_k = np.nan
                if not np.isnan(cumul_coefs[k]) and cumul_ses[k] > 0:
                    from scipy import stats as sp_stats
                    t_stat = cumul_coefs[k] / cumul_ses[k]
                    pval_k = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=100))
                stars = "***" if pval_k < 0.01 else "**" if pval_k < 0.05 else "*" if pval_k < 0.10 else ""
                print(f"        0-{(k+1)*7}d: β={cumul_coefs[k]:.6f} {stars} (SE={cumul_ses[k]:.6f})")

            all_results[dep_var] = {
                "label": dep_label,
                "excl_coefs": excl_coefs, "excl_ses": excl_ses,
                "cumul_coefs": cumul_coefs, "cumul_ses": cumul_ses,
            }

        # Plot figure: len(outcomes) rows × 2 cols (exclusive + cumulative, for appendix)
        if all_results:
            os.makedirs(FIG_DIR, exist_ok=True)
            n_rows = len(outcomes)
            fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.3 * n_rows))
            x_pos = np.arange(n_bins)
            short_bin_labels = [f"{b*7}" for b in range(n_bins)]
            short_cum_labels = [f"{(k+1)*7}" for k in range(n_bins)]
            excl_color = "#2166ac"
            cumul_color = "#b2182b"

            for row_idx, (dep_var, dep_label) in enumerate(outcomes):
                if dep_var not in all_results:
                    continue
                r = all_results[dep_var]

                # Exclusive (left column)
                ax = axes[row_idx, 0]
                ax.fill_between(x_pos,
                                [c - 1.96 * s for c, s in zip(r["excl_coefs"], r["excl_ses"])],
                                [c + 1.96 * s for c, s in zip(r["excl_coefs"], r["excl_ses"])],
                                alpha=0.2, color=excl_color)
                ax.plot(x_pos, r["excl_coefs"], "o-", color=excl_color,
                        linewidth=2, markersize=6)
                ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(short_bin_labels, fontsize=8)
                ax.tick_params(axis="y", labelsize=9)
                ax.set_ylabel(dep_label, fontsize=10)
                if row_idx == 0:
                    ax.set_title("Exclusive 7-Day Windows", fontsize=12, fontweight="bold")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Window start (days before election)", fontsize=10)

                # Cumulative (right column)
                ax = axes[row_idx, 1]
                ax.fill_between(x_pos,
                                [c - 1.96 * s for c, s in zip(r["cumul_coefs"], r["cumul_ses"])],
                                [c + 1.96 * s for c, s in zip(r["cumul_coefs"], r["cumul_ses"])],
                                alpha=0.2, color=cumul_color)
                ax.plot(x_pos, r["cumul_coefs"], "o-", color=cumul_color,
                        linewidth=2, markersize=6)
                ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(short_cum_labels, fontsize=8)
                ax.tick_params(axis="y", labelsize=9)
                if row_idx == 0:
                    ax.set_title("Cumulative Windows", fontsize=12, fontweight="bold")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Cumulative window (days)", fontsize=10)

            plt.tight_layout()
            fig_path = os.path.join(FIG_DIR, fig_name)
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n  Saved figure: {fig_path}")

        # Store for combined figure
        all_treat_results[prefix] = all_results

    # Combined cumulative-only figure: len(outcomes) rows × 2 cols (mean PM2.5, frac haze)
    treat_configs = [
        ("mean_bin", "Mean Smoke PM$_{2.5}$", "#2166ac"),
        ("frac_bin", "Frac. Days > 20 µg/m$^3$ (Haze)", "#b2182b"),
    ]
    short_labels = [f"{(k+1)*7}" for k in range(n_bins)]

    if all(t[0] in all_treat_results for t in treat_configs):
        os.makedirs(FIG_DIR, exist_ok=True)
        n_rows = len(outcomes)
        fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.3 * n_rows))
        x_pos = np.arange(n_bins)

        for col_idx, (prefix, col_title, color) in enumerate(treat_configs):
            results = all_treat_results[prefix]
            for row_idx, (dep_var, dep_label) in enumerate(outcomes):
                ax = axes[row_idx, col_idx]
                if dep_var not in results:
                    ax.set_visible(False)
                    continue
                r = results[dep_var]
                coefs = r["cumul_coefs"]
                ses = r["cumul_ses"]

                ax.fill_between(x_pos,
                                [c - 1.96 * s for c, s in zip(coefs, ses)],
                                [c + 1.96 * s for c, s in zip(coefs, ses)],
                                alpha=0.2, color=color)
                ax.plot(x_pos, coefs, "o-", color=color,
                        linewidth=2, markersize=6)
                ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(short_labels, fontsize=8)
                ax.tick_params(axis="y", labelsize=9)
                if col_idx == 0:
                    ax.set_ylabel(dep_label, fontsize=10)
                if row_idx == 0:
                    ax.set_title(col_title, fontsize=12, fontweight="bold")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Cumulative window (days)", fontsize=10)

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "temporal_cumulative_controls.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved combined cumulative figure: {fig_path}")

    # --- State×Year FE cumulative temporal dynamics (overlaid with Spec 3) ---
    if all(t[0] in all_treat_results for t in treat_configs):
        print("\n" + "-" * 60)
        print("  STATE×YEAR FE: Cumulative Temporal Dynamics")
        print("-" * 60)

        from scipy import stats as sp_stats

        # Collect state×year FE results in parallel structure
        sy_treat_results = {}

        for prefix, col_title, color in treat_configs:
            print(f"\n  === State×Year FE — Treatment: {prefix} ===")
            bin_cols = [f"{prefix}_{b}" for b in range(n_bins)]
            sy_results = {}

            for dep_var, dep_label in outcomes:
                print(f"\n    --- {dep_label} ---")
                cols_needed = [dep_var] + bin_cols + available
                subset = df_merged[cols_needed].dropna().copy()
                if len(subset) < 100:
                    print(f"      SKIP: only {len(subset)} obs")
                    continue

                # Build state×year FE
                state_year_cat = pd.Categorical(
                    subset.index.get_level_values(0).astype(str).str[:2] + "_" +
                    subset.index.get_level_values(1).astype(str)
                )
                state_year_df = pd.DataFrame(
                    state_year_cat, index=subset.index, columns=["state_year"]
                )

                cumul_coefs_sy, cumul_ses_sy = [], []
                for k in range(n_bins):
                    cum_cols_k = bin_cols[:k + 1]
                    cum_var = f"{prefix}_cumul_sy_{k}"
                    subset[cum_var] = subset[cum_cols_k].mean(axis=1)
                    y_k = subset[dep_var]
                    x_k = sm.add_constant(subset[[cum_var] + available])

                    try:
                        mod_sy = PanelOLS(
                            y_k, x_k, entity_effects=True, time_effects=False,
                            other_effects=state_year_df, check_rank=False,
                            drop_absorbed=True
                        )
                        res_sy = mod_sy.fit(cov_type="clustered", cluster_entity=True)
                        cumul_coefs_sy.append(res_sy.params.get(cum_var, np.nan))
                        cumul_ses_sy.append(res_sy.std_errors.get(cum_var, np.nan))
                    except Exception as e:
                        cumul_coefs_sy.append(np.nan)
                        cumul_ses_sy.append(np.nan)

                print(f"      Cumulative (State×Year FE):")
                for k in range(n_bins):
                    pval_k = np.nan
                    if not np.isnan(cumul_coefs_sy[k]) and cumul_ses_sy[k] > 0:
                        t_stat = cumul_coefs_sy[k] / cumul_ses_sy[k]
                        pval_k = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=100))
                    stars = "***" if pval_k < 0.01 else "**" if pval_k < 0.05 else "*" if pval_k < 0.10 else ""
                    print(f"        0-{(k+1)*7}d: β={cumul_coefs_sy[k]:.6f} {stars} (SE={cumul_ses_sy[k]:.6f})")

                sy_results[dep_var] = {
                    "cumul_coefs": cumul_coefs_sy,
                    "cumul_ses": cumul_ses_sy,
                }

            sy_treat_results[prefix] = sy_results

        # Overlaid figure: Spec 3 (blue solid) vs State×Year FE (red dashed)
        os.makedirs(FIG_DIR, exist_ok=True)
        n_rows = len(outcomes)
        fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.3 * n_rows))
        x_pos = np.arange(n_bins)

        for col_idx, (prefix, col_title, color) in enumerate(treat_configs):
            spec3_results = all_treat_results[prefix]
            sy_results = sy_treat_results.get(prefix, {})

            for row_idx, (dep_var, dep_label) in enumerate(outcomes):
                ax = axes[row_idx, col_idx]

                has_spec3 = dep_var in spec3_results
                has_sy = dep_var in sy_results

                if not has_spec3 and not has_sy:
                    ax.set_visible(False)
                    continue

                # Spec 3 (TWFE + Controls): blue solid
                if has_spec3:
                    r3 = spec3_results[dep_var]
                    coefs3 = r3["cumul_coefs"]
                    ses3 = r3["cumul_ses"]
                    ax.fill_between(x_pos,
                                    [c - 1.96 * s for c, s in zip(coefs3, ses3)],
                                    [c + 1.96 * s for c, s in zip(coefs3, ses3)],
                                    alpha=0.15, color="#2166ac")
                    ax.plot(x_pos, coefs3, "o-", color="#2166ac",
                            linewidth=2, markersize=5, label="TWFE + Controls")

                # State×Year FE: red dashed
                if has_sy:
                    rsy = sy_results[dep_var]
                    coefs_sy = rsy["cumul_coefs"]
                    ses_sy = rsy["cumul_ses"]
                    ax.fill_between(x_pos,
                                    [c - 1.96 * s for c, s in zip(coefs_sy, ses_sy)],
                                    [c + 1.96 * s for c, s in zip(coefs_sy, ses_sy)],
                                    alpha=0.15, color="#b2182b")
                    ax.plot(x_pos, coefs_sy, "s--", color="#b2182b",
                            linewidth=2, markersize=5, label="State×Year FE")

                ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(short_labels, fontsize=8)
                ax.tick_params(axis="y", labelsize=9)
                if col_idx == 0:
                    ax.set_ylabel(dep_label, fontsize=10)
                if row_idx == 0:
                    ax.set_title(col_title, fontsize=12, fontweight="bold")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Cumulative window (days)", fontsize=10)
                # Legend in top-left panel only
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=8, loc="upper left")

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "temporal_cumulative_stateyear.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved overlaid cumulative figure: {fig_path}")


def temporal_closein_controls(df):
    """Daily-resolution cumulative windows for days 1-7 before election."""
    print("\n" + "=" * 70)
    print("CLOSE-IN TEMPORAL DYNAMICS: Daily Windows (1-7 days)")
    print("  Cumulative windows at 1-day resolution, Spec 3 (TWFE + controls)")
    print("=" * 70)

    HAZE_THRESHOLD = 20.0
    n_days = 7

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    # Load raw daily smoke data
    print("  Loading raw daily smoke data...")
    with open(SMOKE_FILE) as f:
        sep = "\t" if "\t" in f.readline() else ","
    smoke_raw = pd.read_csv(SMOKE_FILE, sep=sep, dtype={"GEOID": str})
    smoke_raw = smoke_raw.rename(columns={"GEOID": "fips", "smokePM_pred": "smoke_pm25"})
    smoke_raw["fips"] = smoke_raw["fips"].str.zfill(5)
    smoke_raw["date"] = pd.to_datetime(smoke_raw["date"], format="%Y%m%d")

    # Compute 7 single-day bins
    bin_vars = []
    for yr, edate_str in ELECTION_DATES.items():
        edate = pd.Timestamp(edate_str)
        earliest = edate - timedelta(days=n_days)
        yr_smoke = smoke_raw[(smoke_raw["date"] > earliest) & (smoke_raw["date"] <= edate)].copy()

        yr_bins = None
        for d in range(n_days):
            day_start = edate - timedelta(days=d + 1)
            day_end = edate - timedelta(days=d)
            w = yr_smoke[(yr_smoke["date"] > day_start) & (yr_smoke["date"] <= day_end)]

            bin_mean = w.groupby("fips")["smoke_pm25"].mean().rename(f"mean_day_{d}")
            bin_frac = w.groupby("fips")["smoke_pm25"].apply(
                lambda x: (x > HAZE_THRESHOLD).mean()
            ).rename(f"frac_day_{d}")

            if yr_bins is None:
                yr_bins = pd.concat([bin_mean, bin_frac], axis=1)
            else:
                yr_bins = yr_bins.join(bin_mean, how="outer").join(bin_frac, how="outer")

        yr_bins["year"] = yr
        yr_bins = yr_bins.reset_index()
        bin_vars.append(yr_bins)

    bins_df = pd.concat(bin_vars, ignore_index=True).fillna(0)
    bins_df = bins_df.set_index(["fips", "year"])
    df_merged = df.join(bins_df, how="inner")
    print(f"  Merged panel: {len(df_merged):,} obs")

    outcomes = [
        ("dem_vote_share", "DEM Vote Share"),
        ("incumbent_vote_share", "Incumbent Vote Share"),
        ("log_total_votes", "Log Total Votes"),
        ("turnout_rate", "Turnout Rate"),
    ]

    treatments = [
        ("mean_day", "Mean Smoke PM$_{2.5}$", "Mean Smoke PM2.5"),
        ("frac_day", "Frac. Days > 20 µg/m$^3$ (Haze)", "Frac. Days Haze"),
    ]

    from scipy import stats as sp_stats

    all_treat_results = {}
    for prefix, fig_label, print_label in treatments:
        print(f"\n  === Treatment: {print_label} ===")
        day_cols = [f"{prefix}_{d}" for d in range(n_days)]

        all_results = {}
        for dep_var, dep_label in outcomes:
            print(f"\n    --- {dep_label} ---")

            cols_needed = [dep_var] + day_cols + available
            subset = df_merged[cols_needed].dropna().copy()
            if len(subset) < 100:
                print(f"      SKIP: only {len(subset)} obs")
                continue

            # Cumulative: expanding daily windows
            cumul_coefs, cumul_ses = [], []
            for k in range(n_days):
                cum_cols_k = day_cols[:k + 1]
                cum_var = f"{prefix}_cumul_{k}"
                subset[cum_var] = subset[cum_cols_k].mean(axis=1)
                y_k = subset[dep_var]
                x_k = sm.add_constant(subset[[cum_var] + available])

                try:
                    mod_k = PanelOLS(y_k, x_k, entity_effects=True, time_effects=True,
                                     check_rank=False, drop_absorbed=True)
                    res_k = mod_k.fit(cov_type="clustered", cluster_entity=True)
                    cumul_coefs.append(res_k.params.get(cum_var, np.nan))
                    cumul_ses.append(res_k.std_errors.get(cum_var, np.nan))
                except Exception:
                    cumul_coefs.append(np.nan)
                    cumul_ses.append(np.nan)

            print(f"      Cumulative:")
            for k in range(n_days):
                pval_k = np.nan
                if not np.isnan(cumul_coefs[k]) and cumul_ses[k] > 0:
                    t_stat = cumul_coefs[k] / cumul_ses[k]
                    pval_k = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=100))
                stars = "***" if pval_k < 0.01 else "**" if pval_k < 0.05 else "*" if pval_k < 0.10 else ""
                print(f"        0-{k+1}d: β={cumul_coefs[k]:.6f} {stars} (SE={cumul_ses[k]:.6f})")

            all_results[dep_var] = {
                "label": dep_label,
                "cumul_coefs": cumul_coefs, "cumul_ses": cumul_ses,
            }

        all_treat_results[prefix] = all_results

    # Combined figure: len(outcomes) rows × 2 cols (mean PM2.5, frac haze)
    treat_configs = [
        ("mean_day", "Mean Smoke PM$_{2.5}$", "#2166ac"),
        ("frac_day", "Frac. Days > 20 µg/m$^3$ (Haze)", "#b2182b"),
    ]
    x_pos = np.arange(n_days)
    x_labels = [str(k + 1) for k in range(n_days)]

    if all(t[0] in all_treat_results for t in treat_configs):
        os.makedirs(FIG_DIR, exist_ok=True)
        n_rows = len(outcomes)
        fig, axes = plt.subplots(n_rows, 2, figsize=(11, 3.3 * n_rows))

        for col_idx, (prefix, col_title, color) in enumerate(treat_configs):
            results = all_treat_results[prefix]
            for row_idx, (dep_var, dep_label) in enumerate(outcomes):
                ax = axes[row_idx, col_idx]
                if dep_var not in results:
                    ax.set_visible(False)
                    continue
                r = results[dep_var]
                coefs = r["cumul_coefs"]
                ses = r["cumul_ses"]

                ax.fill_between(x_pos,
                                [c - 1.96 * s for c, s in zip(coefs, ses)],
                                [c + 1.96 * s for c, s in zip(coefs, ses)],
                                alpha=0.2, color=color)
                ax.plot(x_pos, coefs, "o-", color=color, linewidth=2, markersize=6)
                ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, fontsize=9)
                ax.tick_params(axis="y", labelsize=9)
                if col_idx == 0:
                    ax.set_ylabel(dep_label, fontsize=10)
                if row_idx == 0:
                    ax.set_title(col_title, fontsize=12, fontweight="bold")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Cumulative window (days)", fontsize=10)

        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "temporal_closein_daily.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved close-in figure: {fig_path}")


def temporal_drop2020(df):
    """Cumulative temporal dynamics: full sample vs. excluding 2020."""
    print("\n" + "=" * 70)
    print("TEMPORAL DYNAMICS: Full Sample vs. Excluding 2020")
    print("  Cumulative windows, Spec 3 (TWFE + controls)")
    print("=" * 70)

    HAZE_THRESHOLD = 20.0

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    # Load raw daily smoke data
    print("  Loading raw daily smoke data...")
    with open(SMOKE_FILE) as f:
        sep = "\t" if "\t" in f.readline() else ","
    smoke_raw = pd.read_csv(SMOKE_FILE, sep=sep, dtype={"GEOID": str})
    smoke_raw = smoke_raw.rename(columns={"GEOID": "fips", "smokePM_pred": "smoke_pm25"})
    smoke_raw["fips"] = smoke_raw["fips"].str.zfill(5)
    smoke_raw["date"] = pd.to_datetime(smoke_raw["date"], format="%Y%m%d")

    # Compute 13 non-overlapping 7-day bins
    n_bins = 13
    bin_vars = []
    for yr, edate_str in ELECTION_DATES.items():
        edate = pd.Timestamp(edate_str)
        earliest = edate - timedelta(days=n_bins * 7)
        yr_smoke = smoke_raw[(smoke_raw["date"] > earliest) & (smoke_raw["date"] <= edate)].copy()

        yr_bins = None
        for b in range(n_bins):
            bin_start = edate - timedelta(days=(b + 1) * 7)
            bin_end = edate - timedelta(days=b * 7)
            w = yr_smoke[(yr_smoke["date"] > bin_start) & (yr_smoke["date"] <= bin_end)]

            bin_mean = w.groupby("fips")["smoke_pm25"].mean().rename(f"mean_bin_{b}")
            bin_frac = w.groupby("fips")["smoke_pm25"].apply(
                lambda x: (x > HAZE_THRESHOLD).mean()
            ).rename(f"frac_bin_{b}")

            if yr_bins is None:
                yr_bins = pd.concat([bin_mean, bin_frac], axis=1)
            else:
                yr_bins = yr_bins.join(bin_mean, how="outer").join(bin_frac, how="outer")

        yr_bins["year"] = yr
        yr_bins = yr_bins.reset_index()
        bin_vars.append(yr_bins)

    bins_df = pd.concat(bin_vars, ignore_index=True).fillna(0)
    bins_df = bins_df.set_index(["fips", "year"])

    df_merged = df.join(bins_df, how="inner")

    outcomes = [
        ("dem_vote_share", "DEM Vote Share"),
        ("incumbent_vote_share", "Incumbent Vote Share"),
        ("log_total_votes", "Log Total Votes"),
        ("turnout_rate", "Turnout Rate"),
    ]

    treatments = [
        ("mean_bin", "Mean Smoke PM$_{2.5}$"),
        ("frac_bin", "Frac. Days > 20 µg/m$^3$ (Haze)"),
    ]

    cum_labels = [f"0\u2013{(k+1)*7}d" for k in range(n_bins)]

    samples = [
        ("Full sample", df_merged),
        ("Excl. 2020", df_merged[df_merged.index.get_level_values("year") != 2020]),
    ]

    print(f"  Full sample: {len(samples[0][1]):,} obs")
    print(f"  Excl. 2020:  {len(samples[1][1]):,} obs")

    # Run cumulative regressions for both samples
    # results[prefix][dep_var][sample_label] = {"coefs": [...], "ses": [...]}
    results = {}
    for prefix, treat_label in treatments:
        results[prefix] = {}
        bin_cols = [f"{prefix}_{b}" for b in range(n_bins)]

        for dep_var, dep_label in outcomes:
            results[prefix][dep_var] = {}

            for sample_label, sample_df in samples:
                cols_needed = [dep_var] + bin_cols + available
                subset = sample_df[cols_needed].dropna().copy()

                cumul_coefs, cumul_ses = [], []
                for k in range(n_bins):
                    cum_cols_k = bin_cols[:k + 1]
                    cum_var = f"{prefix}_cumul_{k}"
                    subset[cum_var] = subset[cum_cols_k].mean(axis=1)
                    y_k = subset[dep_var]
                    x_k = sm.add_constant(subset[[cum_var] + available])

                    try:
                        mod_k = PanelOLS(y_k, x_k, entity_effects=True, time_effects=True,
                                         check_rank=False, drop_absorbed=True)
                        res_k = mod_k.fit(cov_type="clustered", cluster_entity=True)
                        cumul_coefs.append(res_k.params.get(cum_var, np.nan))
                        cumul_ses.append(res_k.std_errors.get(cum_var, np.nan))
                    except Exception:
                        cumul_coefs.append(np.nan)
                        cumul_ses.append(np.nan)

                results[prefix][dep_var][sample_label] = {
                    "coefs": cumul_coefs, "ses": cumul_ses,
                }

                # Print summary at 30d (bin index 4, covering 0-35d ≈ 30d)
                idx_30 = 3  # 0-28d is closest to 30d
                coef_30 = cumul_coefs[idx_30] if idx_30 < len(cumul_coefs) else np.nan
                se_30 = cumul_ses[idx_30] if idx_30 < len(cumul_ses) else np.nan
                if not np.isnan(coef_30) and se_30 > 0:
                    from scipy import stats as sp_stats
                    t_stat = coef_30 / se_30
                    pval_30 = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=100))
                    stars = "***" if pval_30 < 0.01 else "**" if pval_30 < 0.05 else "*" if pval_30 < 0.10 else ""
                    print(f"  {treat_label:35s} {dep_label:25s} {sample_label:12s}  "
                          f"β(0-28d)={coef_30:.6f}{stars:3s} (SE={se_30:.6f})")

    # Plot figure: len(outcomes) rows × 2 cols, rows = outcomes, cols = treatments
    # Two distinct colors per column: dark for full, orange for excl. 2020
    os.makedirs(FIG_DIR, exist_ok=True)
    n_rows = len(outcomes)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.3 * n_rows))
    x_pos = np.arange(n_bins)

    # Use contrasting colors: dark blue/red for full, orange/coral for drop-2020
    full_colors = {"mean_bin": "#2166ac", "frac_bin": "#b2182b"}
    drop_colors = {"mean_bin": "#f4a582", "frac_bin": "#92c5de"}

    # Shorter x-labels: just show the endpoint in days
    short_labels = [f"{(k+1)*7}" for k in range(n_bins)]

    for col_idx, (prefix, col_title) in enumerate(treatments):
        fc = full_colors[prefix]
        dc = drop_colors[prefix]
        for row_idx, (dep_var, dep_label) in enumerate(outcomes):
            ax = axes[row_idx, col_idx]

            # Full sample
            r_full = results[prefix][dep_var]["Full sample"]
            ax.fill_between(x_pos,
                            [c - 1.96 * s for c, s in zip(r_full["coefs"], r_full["ses"])],
                            [c + 1.96 * s for c, s in zip(r_full["coefs"], r_full["ses"])],
                            alpha=0.2, color=fc)
            ax.plot(x_pos, r_full["coefs"], "o-", color=fc,
                    linewidth=2, markersize=6, label="2008\u20132020 (full)")

            # Excl. 2020
            r_drop = results[prefix][dep_var]["Excl. 2020"]
            ax.fill_between(x_pos,
                            [c - 1.96 * s for c, s in zip(r_drop["coefs"], r_drop["ses"])],
                            [c + 1.96 * s for c, s in zip(r_drop["coefs"], r_drop["ses"])],
                            alpha=0.2, color=dc)
            ax.plot(x_pos, r_drop["coefs"], "s--", color=dc,
                    linewidth=2, markersize=6, label="2008\u20132016 (excl. 2020)")

            ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(short_labels, fontsize=8)
            ax.tick_params(axis="y", labelsize=9)
            if col_idx == 0:
                ax.set_ylabel(dep_label, fontsize=10)
            if row_idx == 0:
                ax.set_title(col_title, fontsize=12, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Cumulative window (days)", fontsize=10)

    # Shared legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = os.path.join(FIG_DIR, "temporal_drop2020.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved figure: {fig_path}")


def placebo_test(df):
    """Test whether smoke in non-fire months predicts voting (should be null)."""
    print("\n" + "=" * 70)
    print("PLACEBO TEST: Winter Smoke (should find null)")
    print("=" * 70)

    # We only have the pre-computed windows from Phase 3, so we can't easily
    # compute winter smoke here. Instead, test whether the smallest window (7d)
    # — which is close to election day in November when fires are rare in most
    # of the country — shows a different pattern than the 60-day window.
    for svar, label in [("smoke_pm25_mean_7d", "7-day window (near election day)"),
                         ("smoke_pm25_mean_60d", "60-day window (incl. fire season)")]:
        if svar not in df.columns:
            continue
        res = run_twfe(df, "dem_vote_share", svar, label=label)
        print_result(res, label, svar)


def create_summary_table(df):
    """Create a formatted summary results table."""
    print("\n" + "=" * 70)
    print("SUMMARY RESULTS TABLE")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"
    specs = [
        ("dem_vote_share", "DEM vote share", "Spec A: Pro-environment"),
        ("incumbent_vote_share", "Incumbent vote share", "Spec B: Incumbent punishment"),
        ("log_total_votes", "Log total votes", "Spec C: Turnout"),
        ("turnout_rate", "Turnout rate (votes/VAP)", "Spec D: Turnout rate"),
    ]

    rows = []
    for dep, dep_label, spec_label in specs:
        res = run_twfe(df, dep, smoke_var, label=spec_label)
        if res:
            coef = res.params.get(smoke_var, np.nan)
            se = res.std_errors.get(smoke_var, np.nan)
            pval = res.pvalues.get(smoke_var, np.nan)
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            rows.append({
                "Specification": spec_label,
                "Dependent Variable": dep_label,
                "β": f"{coef:.6f}{stars}",
                "SE": f"({se:.6f})",
                "p-value": f"{pval:.4f}",
                "N": f"{int(res.nobs):,}",
                "R² (within)": f"{res.rsquared_within:.4f}",
            })

    if rows:
        tbl = pd.DataFrame(rows)
        print(f"\n  Treatment: Mean smoke PM2.5, 30-day pre-election window")
        print(f"  Fixed effects: County + Year")
        print(f"  Standard errors: Clustered by county")
        print(f"\n{tbl.to_string(index=False)}")

    return rows


def plot_smoke_distribution(df):
    """Plot distribution of smoke exposure across elections."""
    os.makedirs(FIG_DIR, exist_ok=True)

    df_reset = df.reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Smoke days by year
    ax = axes[0]
    years = sorted(df_reset["year"].unique())
    bp_data = [df_reset[df_reset["year"] == yr]["smoke_days_60d"].values for yr in years]
    ax.boxplot(bp_data, labels=[str(y) for y in years])
    ax.set_xlabel("Election Year")
    ax.set_ylabel("Smoke Days (60-day window)")
    ax.set_title("A. Distribution of Smoke Days Before Election")

    # Panel B: Mean smoke PM2.5 by year
    ax = axes[1]
    bp_data = [df_reset[df_reset["year"] == yr]["smoke_pm25_mean_60d"].values for yr in years]
    ax.boxplot(bp_data, labels=[str(y) for y in years])
    ax.set_xlabel("Election Year")
    ax.set_ylabel("Mean Smoke PM2.5 (µg/m³)")
    ax.set_title("B. Distribution of Smoke PM2.5 Before Election")

    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "smoke_distribution_by_year.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\n  Saved figure: {fig_path}")


def plot_binscatter(df):
    """Create binscatter of smoke PM2.5 vs. DEM vote share (residualized)."""
    os.makedirs(FIG_DIR, exist_ok=True)

    smoke_var = "smoke_pm25_mean_30d"
    dep_var = "dem_vote_share"

    df_clean = df[[smoke_var, dep_var]].dropna()
    if len(df_clean) < 100:
        print("  Not enough data for binscatter")
        return

    # Residualize: demean by county and year
    df_reset = df_clean.reset_index()
    for var in [smoke_var, dep_var]:
        county_mean = df_reset.groupby("fips")[var].transform("mean")
        year_mean = df_reset.groupby("year")[var].transform("mean")
        grand_mean = df_reset[var].mean()
        df_reset[f"{var}_resid"] = df_reset[var] - county_mean - year_mean + grand_mean

    # Bin into 50 equal-sized bins of residualized smoke
    df_reset["smoke_bin"] = pd.qcut(df_reset[f"{smoke_var}_resid"], 50, duplicates="drop")
    binned = df_reset.groupby("smoke_bin", observed=True).agg({
        f"{smoke_var}_resid": "mean",
        f"{dep_var}_resid": "mean",
    }).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(binned[f"{smoke_var}_resid"], binned[f"{dep_var}_resid"],
               s=80, color="steelblue", zorder=3)

    # Fit line
    x = binned[f"{smoke_var}_resid"].values
    y = binned[f"{dep_var}_resid"].values
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Smoke PM2.5, 30-day mean (residualized)")
    ax.set_ylabel("DEM Two-Party Vote Share (residualized)")
    ax.set_title("Binscatter: Wildfire Smoke and Democratic Vote Share\n(County and Year FE Residualized)")
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "binscatter_smoke_dem_share.png")
    fig.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  Saved figure: {fig_path}")


def robustness_controls(df):
    """Robustness check: add time-varying county controls."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS: Time-Varying County Controls (Presidential)")
    print("=" * 70)

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    if not available:
        print("  No control variables found in dataset. Skipping.")
        return

    print(f"  Controls: {available}")
    n_with_controls = df[available].dropna().shape[0]
    print(f"  Observations with all controls: {n_with_controls:,}/{len(df):,}")

    smoke_var = "smoke_pm25_mean_30d"
    specs = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
        ("turnout_rate", "Turnout rate"),
    ]

    print(f"\n  {'Outcome':<25} {'Baseline β':>12} {'+ Controls β':>14} "
          f"{'Baseline p':>12} {'+ Controls p':>14} {'N base':>8} {'N ctrl':>8}")
    print("  " + "-" * 95)

    for dep_var, dep_label in specs:
        res_base = run_twfe(df, dep_var, smoke_var, label=f"Base: {dep_label}")
        res_ctrl = run_twfe(df, dep_var, smoke_var, controls=available,
                            label=f"+Controls: {dep_label}")

        b_coef = res_base.params.get(smoke_var, np.nan) if res_base else np.nan
        b_pval = res_base.pvalues.get(smoke_var, np.nan) if res_base else np.nan
        b_n = res_base.nobs if res_base else 0
        c_coef = res_ctrl.params.get(smoke_var, np.nan) if res_ctrl else np.nan
        c_pval = res_ctrl.pvalues.get(smoke_var, np.nan) if res_ctrl else np.nan
        c_n = res_ctrl.nobs if res_ctrl else 0

        print(f"  {dep_label:<25} {b_coef:>12.6f} {c_coef:>14.6f} "
              f"{b_pval:>12.4f} {c_pval:>14.4f} {b_n:>8,} {c_n:>8,}")

    # Also print full results for the with-controls regressions
    print("\n  --- Full results with controls ---")
    for dep_var, dep_label in specs:
        res = run_twfe(df, dep_var, smoke_var, controls=available,
                       label=f"+Controls: {dep_label}")
        print_result(res, f"+Controls: {dep_label}", smoke_var)


def _make_state_trends(df):
    """Create state-specific linear trend columns for the panel.

    Creates state_i × (year - mean) interactions. Collinear columns
    are handled by drop_absorbed=True in the PanelOLS call.
    """
    fips_vals = df.index.get_level_values("fips").astype(str).str[:2]
    year_vals = df.index.get_level_values("year").astype(float)
    year_norm = year_vals - np.mean(year_vals)

    states = sorted(fips_vals.unique())
    if len(states) <= 1:
        return df, []

    # Drop first state for identification
    trend_cols = []
    for st in states[1:]:
        col = f"trend_{st}"
        df[col] = (fips_vals == st).astype(float) * year_norm
        trend_cols.append(col)

    return df, trend_cols


def create_buildup_table(df):
    """Build-up specification table: raw OLS → TWFE → +controls → +state trends."""
    print("\n" + "=" * 70)
    print("BUILD-UP SPECIFICATION TABLE (Presidential)")
    print("  (1) Raw OLS  (2) County+Year FE  (3) +Controls  (4) +State Trends")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"
    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    # Create state trend columns
    df, trend_cols = _make_state_trends(df)

    outcomes = [
        ("dem_vote_share", "Panel A: DEM Vote Share"),
        ("incumbent_vote_share", "Panel B: Incumbent Vote Share"),
        ("log_total_votes", "Panel C: Log Total Votes"),
        ("turnout_rate", "Panel D: Turnout Rate"),
    ]

    results = {}

    for dep_var, panel_label in outcomes:
        print(f"\n  {panel_label}")
        panel_results = []

        # Spec 1: Raw OLS (no FE)
        res1 = run_twfe(df, dep_var, smoke_var,
                        absorb_entity=False, absorb_time=False,
                        label=f"(1) Raw OLS: {dep_var}")
        # Spec 2: County + Year FE
        res2 = run_twfe(df, dep_var, smoke_var,
                        label=f"(2) TWFE: {dep_var}")
        # Spec 3: + Controls
        res3 = run_twfe(df, dep_var, smoke_var, controls=available,
                        label=f"(3) +Controls: {dep_var}")
        # Spec 4: + State trends (drop absorbed variables automatically)
        res4 = run_twfe(df, dep_var, smoke_var,
                        controls=available + trend_cols,
                        drop_absorbed=True,
                        label=f"(4) +State trends: {dep_var}")

        for i, res in enumerate([res1, res2, res3, res4], 1):
            if res is not None:
                coef = res.params.get(smoke_var, np.nan)
                se = res.std_errors.get(smoke_var, np.nan)
                pval = res.pvalues.get(smoke_var, np.nan)
                n = int(res.nobs)
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                panel_results.append({"spec": i, "coef": coef, "se": se,
                                      "pval": pval, "n": n, "stars": stars})
            else:
                panel_results.append({"spec": i, "coef": np.nan, "se": np.nan,
                                      "pval": np.nan, "n": 0, "stars": ""})

        results[dep_var] = panel_results

    # Print formatted table
    print("\n" + "-" * 80)
    print(f"  {'':30s} {'(1)':>12s} {'(2)':>12s} {'(3)':>12s} {'(4)':>12s}")
    print(f"  {'':30s} {'Raw OLS':>12s} {'TWFE':>12s} {'+Controls':>12s} {'+St.Trends':>12s}")
    print("-" * 80)

    for dep_var, panel_label in outcomes:
        pr = results[dep_var]
        coef_str = "  " + panel_label
        se_str = "  "
        print(coef_str)
        coef_line = f"  {'Smoke PM2.5 (30d)':<30s}"
        se_line = f"  {'':30s}"
        for r in pr:
            coef_line += f" {r['coef']:>10.5f}{r['stars']:<2s}"
            se_line += f" ({r['se']:>9.5f}) "
        print(coef_line)
        print(se_line)
        n_line = f"  {'Observations':<30s}"
        for r in pr:
            n_line += f" {r['n']:>11,} "
        print(n_line)
        print()

    print(f"  {'County FE':<30s} {'':>12s} {'Yes':>12s} {'Yes':>12s} {'Yes':>12s}")
    print(f"  {'Year FE':<30s} {'':>12s} {'Yes':>12s} {'Yes':>12s} {'Yes':>12s}")
    print(f"  {'Controls':<30s} {'':>12s} {'':>12s} {'Yes':>12s} {'Yes':>12s}")
    print(f"  {'State trends':<30s} {'':>12s} {'':>12s} {'':>12s} {'Yes':>12s}")
    print("-" * 80)

    return results


def main():
    print("=" * 70)
    print("Phase 4: Wildfire Smoke and Voting Behavior — Analysis")
    print("=" * 70)

    df = load_data()

    # Descriptive plots
    print("\n--- Descriptive Figures ---")
    plot_smoke_distribution(df)

    # Main specifications
    spec_a_dem_vote_share(df)
    spec_b_incumbent_punishment(df)
    spec_c_turnout(df)

    # Summary table
    create_summary_table(df)

    # Build-up specification table
    create_buildup_table(df)

    # Heterogeneity
    heterogeneity_tests(df)

    # Dose-response
    dose_response_analysis(df)

    # Event study windows
    event_study_windows(df)

    # 7-day temporal dynamics
    temporal_dynamics_7day(df)

    # 7-day temporal dynamics with controls (Spec 3)
    temporal_dynamics_controls(df)

    # Close-in daily temporal dynamics (1-7 days)
    temporal_closein_controls(df)

    # Temporal dynamics: full sample vs. drop 2020
    temporal_drop2020(df)

    # Threshold comparison (haze, USG, unhealthy)
    threshold_comparison(df)

    # State×Year FE robustness
    state_year_fe_regressions(df)

    # Time-varying controls robustness
    robustness_controls(df)

    # Placebo
    placebo_test(df)

    # Binscatter
    print("\n--- Binscatter ---")
    plot_binscatter(df)

    print("\n" + "=" * 70)
    print("Phase 4 complete. Figures saved to:", FIG_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()

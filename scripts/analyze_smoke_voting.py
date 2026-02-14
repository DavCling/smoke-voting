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
FIG_DIR = os.path.join(BASE_DIR, "output", "figures")


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


def run_twfe(df, dep_var, smoke_var, controls=None, absorb_entity=True, absorb_time=True, label=""):
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
        mod = PanelOLS(
            y, x,
            entity_effects=absorb_entity,
            time_effects=absorb_time,
            check_rank=False,
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


def heterogeneity_tests(df):
    """Test for heterogeneous effects by prior partisanship and urban/rural."""
    print("\n" + "=" * 70)
    print("HETEROGENEITY TESTS")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_60d"
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

    smoke_var = "smoke_pm25_mean_60d"

    # Create quintiles of smoke exposure
    df_reset = df.reset_index()
    df_reset["smoke_quintile"] = pd.qcut(
        df_reset[smoke_var], 5, labels=False, duplicates="drop"
    )

    print(f"\n  Smoke PM2.5 (60d) quintile boundaries:")
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

    smoke_var = "smoke_pm25_mean_60d"
    specs = [
        ("dem_vote_share", "DEM vote share", "Spec A: Pro-environment"),
        ("incumbent_vote_share", "Incumbent vote share", "Spec B: Incumbent punishment"),
        ("log_total_votes", "Log total votes", "Spec C: Turnout"),
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
        print(f"\n  Treatment: Mean smoke PM2.5, 60-day pre-election window")
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

    # Heterogeneity
    heterogeneity_tests(df)

    # Dose-response
    dose_response_analysis(df)

    # Event study windows
    event_study_windows(df)

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

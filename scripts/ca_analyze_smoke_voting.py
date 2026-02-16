#!/usr/bin/env python3
"""
CA Tract-Level Analysis: Wildfire Smoke and Voting Behavior.

Matches the national analysis structure for comparability:
  - Build-up table (raw OLS → TWFE → +controls → +county trends)
  - Temporal dynamics (cumulative 7-day bins with controls)
  - Threshold comparison (haze / USG / unhealthy)
  - House comparison
  - National vs. CA comparison
  - Robustness checks (county×year FE, controls, crosswalk validation)

FE structure: Tract + Year (base), County trends as robustness,
              County × Year as most demanding.
Clustering: By tract (primary), by county (robustness).
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
PRES_FILE = os.path.join(BASE_DIR, "output", "california",
                         "ca_smoke_voting_presidential.parquet")
HOUSE_FILE = os.path.join(BASE_DIR, "output", "california",
                          "ca_smoke_voting_house.parquet")
SMOKE_FILE = os.path.join(BASE_DIR, "data", "california", "smoke",
                          "smoke_pm25_tract_daily.csv")
NATIONAL_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_analysis.parquet")
FIG_DIR = os.path.join(BASE_DIR, "output", "california", "figures")

ELECTION_DATES = {
    2006: "2006-11-07",
    2008: "2008-11-04",
    2010: "2010-11-02",
    2012: "2012-11-06",
    2014: "2014-11-04",
    2016: "2016-11-08",
    2018: "2018-11-06",
    2020: "2020-11-03",
    2022: "2022-11-08",
}

# EPA thresholds
HAZE_THRESHOLD = 20.0
EPA_USG_THRESHOLD = 35.5
EPA_UNHEALTHY = 55.5


# ============================================================
# Data Loading
# ============================================================

def load_data(filepath, id_col="geoid"):
    """Load analysis dataset and prepare panel structure."""
    print(f"Loading: {filepath}")
    df = pd.read_parquet(filepath)
    print(f"  {len(df):,} observations, {df[id_col].nunique():,} units, "
          f"{df['year'].nunique()} elections")

    df = df.set_index([id_col, "year"])
    df = df.sort_index()
    return df


# ============================================================
# TWFE Regression
# ============================================================

def run_twfe(df, dep_var, smoke_var, controls=None, absorb_entity=True,
             absorb_time=True, county_year_fe=False, drop_absorbed=False,
             cluster_col=None, label=""):
    """Run a two-way fixed effects regression using linearmodels PanelOLS.

    For county×year FE, pass county_year_fe=True (analogous to state×year
    in the national analysis, but using county since this is single-state).
    """
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
        if county_year_fe:
            # County×Year FE (most demanding — absorbs all county-year variation)
            county_year_cat = pd.Categorical(
                subset.index.get_level_values(0).astype(str).str[:5] + "_" +
                subset.index.get_level_values(1).astype(str)
            )
            other_ef = pd.DataFrame(county_year_cat, index=subset.index,
                                    columns=["county_year"])
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


# ============================================================
# Summary Statistics
# ============================================================

def summary_statistics(df, label="CA Tract-Level"):
    """Print summary statistics, comparable with national county stats."""
    print(f"\n{'='*70}")
    print(f"SUMMARY STATISTICS: {label}")
    print(f"{'='*70}")

    smoke_vars = [
        ("smoke_pm25_mean_30d", "Smoke PM2.5 mean (30d)"),
        ("smoke_days_30d", "Smoke days (30d)"),
        ("smoke_frac_haze_30d", "Frac. haze days (30d)"),
        ("smoke_frac_usg_30d", "Frac. USG days (30d)"),
        ("smoke_frac_unhealthy_30d", "Frac. unhealthy days (30d)"),
    ]

    outcome_vars = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
        ("turnout_rate", "Turnout rate"),
    ]

    control_vars = [
        ("unemployment_rate", "Unemployment rate"),
        ("log_median_income", "Log median income"),
        ("log_population", "Log population"),
        ("pct_bachelors_plus", "% Bachelor's+"),
        ("pct_white_nh", "% White non-Hispanic"),
        ("pct_hispanic", "% Hispanic"),
        ("october_tmean", "October mean temp"),
        ("october_ppt", "October precip"),
    ]

    print(f"\n  {'Variable':<30s} {'N':>8s} {'Mean':>10s} {'SD':>10s} {'Min':>10s} {'Max':>10s}")
    print("  " + "-" * 78)

    for var_list, header in [(smoke_vars, "Smoke Exposure"),
                             (outcome_vars, "Outcomes"),
                             (control_vars, "Controls")]:
        print(f"\n  {header}")
        for var, label in var_list:
            if var not in df.columns:
                continue
            s = df[var].dropna()
            if len(s) == 0:
                continue
            print(f"  {label:<30s} {len(s):>8,} {s.mean():>10.4f} {s.std():>10.4f} "
                  f"{s.min():>10.4f} {s.max():>10.4f}")


# ============================================================
# Build-Up Table
# ============================================================

def _make_county_trends(df):
    """Create county-specific linear trend columns."""
    df = df.copy()
    county_fips = df["county_fips"] if "county_fips" in df.columns else df.index.get_level_values(0).str[:5]

    years = df.index.get_level_values(1) if isinstance(df.index, pd.MultiIndex) else df["year"]
    unique_counties = county_fips.unique()

    trend_cols = []
    for c in unique_counties:
        col_name = f"trend_{c}"
        df[col_name] = np.where(county_fips == c, years.astype(float), 0.0)
        trend_cols.append(col_name)

    return df, trend_cols


def create_buildup_table(df):
    """Build-up specification table: raw OLS → TWFE → +controls → +county trends."""
    print(f"\n{'='*70}")
    print("BUILD-UP SPECIFICATION TABLE (CA Tract-Level Presidential)")
    print("  (1) Raw OLS  (2) Tract+Year FE  (3) +Controls  (4) +County Trends")
    print(f"{'='*70}")

    smoke_var = "smoke_pm25_mean_30d"
    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]
    print(f"  Available controls: {available}")

    # Create county trend columns
    df, trend_cols = _make_county_trends(df)

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
        # Spec 2: Tract + Year FE
        res2 = run_twfe(df, dep_var, smoke_var,
                        label=f"(2) TWFE: {dep_var}")
        # Spec 3: + Controls
        res3 = run_twfe(df, dep_var, smoke_var, controls=available,
                        label=f"(3) +Controls: {dep_var}")
        # Spec 4: + County trends
        res4 = run_twfe(df, dep_var, smoke_var,
                        controls=available + trend_cols,
                        drop_absorbed=True,
                        label=f"(4) +County trends: {dep_var}")

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
    print(f"\n{'-'*80}")
    print(f"  {'':30s} {'(1)':>12s} {'(2)':>12s} {'(3)':>12s} {'(4)':>12s}")
    print(f"  {'':30s} {'Raw OLS':>12s} {'TWFE':>12s} {'+Controls':>12s} {'+Cty.Trends':>12s}")
    print("-" * 80)

    for dep_var, panel_label in outcomes:
        pr = results[dep_var]
        print(f"  {panel_label}")
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

    print(f"  {'Tract FE':<30s} {'':>12s} {'Yes':>12s} {'Yes':>12s} {'Yes':>12s}")
    print(f"  {'Year FE':<30s} {'':>12s} {'Yes':>12s} {'Yes':>12s} {'Yes':>12s}")
    print(f"  {'Controls':<30s} {'':>12s} {'':>12s} {'Yes':>12s} {'Yes':>12s}")
    print(f"  {'County trends':<30s} {'':>12s} {'':>12s} {'':>12s} {'Yes':>12s}")
    print("-" * 80)

    return results


# ============================================================
# Temporal Dynamics (7-day bins with controls)
# ============================================================

def temporal_dynamics_controls(df):
    """7-day temporal dynamics with controls for mean PM2.5 and frac haze."""
    print(f"\n{'='*70}")
    print("TEMPORAL DYNAMICS WITH CONTROLS: 7-Day Windows (CA Tracts)")
    print(f"{'='*70}")

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    # Load raw daily smoke data (smoke days only — non-smoke days = 0, omitted)
    print("  Loading raw daily smoke data (smoke days only)...")
    if not os.path.exists(SMOKE_FILE):
        print(f"  ERROR: Smoke file not found: {SMOKE_FILE}")
        return

    smoke_raw = pd.read_csv(SMOKE_FILE, dtype={"GEOID": str})
    smoke_raw = smoke_raw.rename(columns={"GEOID": "geoid", "smokePM_pred": "smoke_pm25"})
    smoke_raw["geoid"] = smoke_raw["geoid"].str.zfill(11)
    smoke_raw["date"] = pd.to_datetime(smoke_raw["date"], format="%Y%m%d")

    # Only use election years that are in the data
    pres_dates = {yr: dt for yr, dt in ELECTION_DATES.items()
                  if yr in df.index.get_level_values(1).unique()}

    # Compute 13 non-overlapping 7-day bins
    # NOTE: Each bin has 7 calendar days. Since the smoke file has only smoke-day
    # rows, mean = sum / 7 (not sum / n_rows), and frac = n_above / 7.
    n_bins = 13
    bin_days = 7  # calendar days per bin
    bin_vars = []
    for yr, edate_str in pres_dates.items():
        edate = pd.Timestamp(edate_str)
        earliest = edate - timedelta(days=n_bins * bin_days)
        yr_smoke = smoke_raw[(smoke_raw["date"] > earliest) &
                             (smoke_raw["date"] <= edate)].copy()

        yr_bins = None
        for b in range(n_bins):
            bin_start = edate - timedelta(days=(b + 1) * bin_days)
            bin_end = edate - timedelta(days=b * bin_days)
            w = yr_smoke[(yr_smoke["date"] > bin_start) & (yr_smoke["date"] <= bin_end)]

            # Sum smoke PM2.5 across smoke-day rows, divide by 7 calendar days
            bin_sum = w.groupby("geoid")["smoke_pm25"].sum()
            bin_mean = (bin_sum / bin_days).rename(f"mean_bin_{b}")

            # Count days above haze threshold, divide by 7 calendar days
            bin_n_haze = w.groupby("geoid")["smoke_pm25"].apply(
                lambda x: (x > HAZE_THRESHOLD).sum()
            )
            bin_frac = (bin_n_haze / bin_days).rename(f"frac_bin_{b}")

            if yr_bins is None:
                yr_bins = pd.concat([bin_mean, bin_frac], axis=1)
            else:
                yr_bins = yr_bins.join(bin_mean, how="outer").join(bin_frac, how="outer")

        yr_bins["year"] = yr
        yr_bins = yr_bins.reset_index()
        yr_bins = yr_bins.rename(columns={"geoid": "geoid"})
        yr_bins = yr_bins.set_index(["geoid", "year"])
        bin_vars.append(yr_bins)

    bins_df = pd.concat(bin_vars).fillna(0)
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
         "ca_temporal_7day_mean_controls.png"),
        ("frac_bin", "Frac. Days > 20 µg/m³ (Haze)", "Frac. Days Haze",
         "ca_temporal_7day_frac_controls.png"),
    ]

    cum_labels = [f"0\u2013{(k+1)*7}d" for k in range(n_bins)]

    all_treat_results = {}  # collect across treatments for county×year overlay figure

    for prefix, fig_treat_label, print_treat_label, fig_name in treatments:
        print(f"\n  === Treatment: {print_treat_label} ===")
        bin_cols = [f"{prefix}_{b}" for b in range(n_bins)]

        all_results = {}

        for dep_var, dep_label in outcomes:
            print(f"\n    --- {dep_label} ---")

            # CUMULATIVE: expanding windows + controls
            cols_needed = [dep_var] + bin_cols + available
            subset = df_merged[[c for c in cols_needed if c in df_merged.columns]].dropna().copy()
            if len(subset) < 100:
                print(f"      SKIP: only {len(subset)} obs")
                continue

            cumul_coefs, cumul_ses = [], []
            for k in range(n_bins):
                cum_cols_k = bin_cols[:k + 1]
                cum_var = f"{prefix}_cumul_{k}"
                subset[cum_var] = subset[cum_cols_k].mean(axis=1)
                y_k = subset[dep_var]
                avail_in_subset = [v for v in available if v in subset.columns]
                x_k = sm.add_constant(subset[[cum_var] + avail_in_subset])

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
                "cumul_coefs": cumul_coefs, "cumul_ses": cumul_ses,
            }

        # Plot cumulative figure
        if all_results:
            os.makedirs(FIG_DIR, exist_ok=True)
            n_outcomes = len(outcomes)
            fig, axes = plt.subplots(1, n_outcomes, figsize=(4.5 * n_outcomes, 4.5))
            if n_outcomes == 1:
                axes = [axes]
            x_pos = np.arange(n_bins)
            short_cum_labels = [f"{(k+1)*7}" for k in range(n_bins)]
            cumul_color = "#b2182b"

            for col_idx, (dep_var, dep_label) in enumerate(outcomes):
                if dep_var not in all_results:
                    continue
                r = all_results[dep_var]
                ax = axes[col_idx]
                ax.fill_between(x_pos,
                                [c - 1.96 * s for c, s in zip(r["cumul_coefs"], r["cumul_ses"])],
                                [c + 1.96 * s for c, s in zip(r["cumul_coefs"], r["cumul_ses"])],
                                alpha=0.2, color=cumul_color)
                ax.plot(x_pos, r["cumul_coefs"], "o-", color=cumul_color,
                        linewidth=2, markersize=5)
                ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(short_cum_labels, fontsize=8)
                ax.set_title(dep_label, fontsize=11, fontweight="bold")
                ax.set_xlabel("Cumulative window (days)", fontsize=9)
                ax.tick_params(axis="y", labelsize=9)

            fig.suptitle(f"CA Tract-Level: Temporal Dynamics ({print_treat_label})",
                         fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            fig_path = os.path.join(FIG_DIR, fig_name)
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n  Saved figure: {fig_path}")

        all_treat_results[prefix] = all_results

    # ------------------------------------------------------------------
    # County×Year FE cumulative temporal dynamics (overlay figure)
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("COUNTY×YEAR FE: Cumulative Temporal Dynamics (CA Tracts)")
    print(f"{'='*70}")

    # Run county×year FE regressions for each treatment × outcome × window
    cy_treat_results = {}
    for prefix, fig_treat_label, print_treat_label, fig_name in treatments:
        print(f"\n  === County×Year FE — Treatment: {print_treat_label} ===")
        bin_cols = [f"{prefix}_{b}" for b in range(n_bins)]

        cy_results = {}
        for dep_var, dep_label in outcomes:
            print(f"\n    --- {dep_label} ---")

            cols_needed = [dep_var] + bin_cols + available
            subset = df_merged[[c for c in cols_needed if c in df_merged.columns]].dropna().copy()
            if len(subset) < 100:
                print(f"      SKIP: only {len(subset)} obs")
                continue

            cumul_coefs, cumul_ses = [], []
            for k in range(n_bins):
                cum_cols_k = bin_cols[:k + 1]
                cum_var = f"{prefix}_cy_cumul_{k}"
                subset[cum_var] = subset[cum_cols_k].mean(axis=1)
                y_k = subset[dep_var]
                avail_in_subset = [v for v in available if v in subset.columns]
                x_k = sm.add_constant(subset[[cum_var] + avail_in_subset])

                try:
                    county_year_cat = pd.Categorical(
                        subset.index.get_level_values(0).astype(str).str[:5] + "_" +
                        subset.index.get_level_values(1).astype(str)
                    )
                    county_year_df = pd.DataFrame(county_year_cat, index=subset.index,
                                                  columns=["county_year"])
                    mod_cy = PanelOLS(y_k, x_k, entity_effects=True, time_effects=False,
                                      other_effects=county_year_df, check_rank=False,
                                      drop_absorbed=True)
                    res_cy = mod_cy.fit(cov_type="clustered", cluster_entity=True)
                    cumul_coefs.append(res_cy.params.get(cum_var, np.nan))
                    cumul_ses.append(res_cy.std_errors.get(cum_var, np.nan))
                except Exception as e:
                    print(f"        Window 0-{(k+1)*7}d ERROR: {e}")
                    cumul_coefs.append(np.nan)
                    cumul_ses.append(np.nan)

            # Print county×year FE results
            print(f"      County×Year FE cumulative:")
            for k in range(n_bins):
                pval_k = np.nan
                if not np.isnan(cumul_coefs[k]) and cumul_ses[k] > 0:
                    from scipy import stats as sp_stats
                    t_stat = cumul_coefs[k] / cumul_ses[k]
                    pval_k = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=100))
                stars = "***" if pval_k < 0.01 else "**" if pval_k < 0.05 else "*" if pval_k < 0.10 else ""
                print(f"        0-{(k+1)*7}d: β={cumul_coefs[k]:.6f} {stars} (SE={cumul_ses[k]:.6f})")

            cy_results[dep_var] = {
                "label": dep_label,
                "cumul_coefs": cumul_coefs, "cumul_ses": cumul_ses,
            }

        cy_treat_results[prefix] = cy_results

    # --- Combined overlay figure: Spec 3 (TWFE+Controls) vs County×Year FE ---
    # Layout: len(outcomes) rows × 2 cols (mean PM2.5, frac haze)
    treat_order = [("mean_bin", "Mean Smoke PM$_{2.5}$"),
                   ("frac_bin", "Frac. Days > 20 µg/m³ (Haze)")]
    n_rows = len(outcomes)
    n_cols = len(treat_order)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    x_pos = np.arange(n_bins)
    short_cum_labels = [f"{(k+1)*7}" for k in range(n_bins)]
    color_spec3 = "#2166ac"  # blue
    color_cy = "#b2182b"     # red

    for col_idx, (prefix, treat_label) in enumerate(treat_order):
        spec3_data = all_treat_results.get(prefix, {})
        cy_data = cy_treat_results.get(prefix, {})

        for row_idx, (dep_var, dep_label) in enumerate(outcomes):
            ax = axes[row_idx, col_idx]

            # Spec 3 (TWFE + Controls) — blue solid
            if dep_var in spec3_data:
                r = spec3_data[dep_var]
                coefs = r["cumul_coefs"]
                ses = r["cumul_ses"]
                ax.fill_between(x_pos,
                                [c - 1.96 * s for c, s in zip(coefs, ses)],
                                [c + 1.96 * s for c, s in zip(coefs, ses)],
                                alpha=0.15, color=color_spec3)
                ax.plot(x_pos, coefs, "o-", color=color_spec3, linewidth=2,
                        markersize=4, label="TWFE + Controls")

            # County×Year FE — red dashed
            if dep_var in cy_data:
                r = cy_data[dep_var]
                coefs = r["cumul_coefs"]
                ses = r["cumul_ses"]
                ax.fill_between(x_pos,
                                [c - 1.96 * s for c, s in zip(coefs, ses)],
                                [c + 1.96 * s for c, s in zip(coefs, ses)],
                                alpha=0.10, color=color_cy)
                ax.plot(x_pos, coefs, "s--", color=color_cy, linewidth=2,
                        markersize=4, label="County×Year FE")

            ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(short_cum_labels, fontsize=7)
            ax.tick_params(axis="y", labelsize=8)

            if row_idx == 0:
                ax.set_title(treat_label, fontsize=11, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Cumulative window (days)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(dep_label, fontsize=9)

            # Legend in top-left panel only
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle("CA Tract-Level: Cumulative Temporal Dynamics\n"
                 "TWFE + Controls vs. County×Year FE",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    cy_fig_path = os.path.join(FIG_DIR, "ca_temporal_cumulative_countyyear.png")
    fig.savefig(cy_fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved overlay figure: {cy_fig_path}")


# ============================================================
# Temporal Dynamics: Drop 2020
# ============================================================

def temporal_drop2020(df):
    """Compare temporal dynamics: full sample vs excluding 2020."""
    print(f"\n{'='*70}")
    print("TEMPORAL DYNAMICS: Full Sample vs. Excluding 2020 (CA)")
    print(f"{'='*70}")

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    # Load raw daily smoke data (smoke days only — non-smoke days = 0, omitted)
    if not os.path.exists(SMOKE_FILE):
        print(f"  ERROR: Smoke file not found")
        return

    smoke_raw = pd.read_csv(SMOKE_FILE, dtype={"GEOID": str})
    smoke_raw = smoke_raw.rename(columns={"GEOID": "geoid", "smokePM_pred": "smoke_pm25"})
    smoke_raw["geoid"] = smoke_raw["geoid"].str.zfill(11)
    smoke_raw["date"] = pd.to_datetime(smoke_raw["date"], format="%Y%m%d")

    pres_dates = {yr: dt for yr, dt in ELECTION_DATES.items()
                  if yr in df.index.get_level_values(1).unique()}

    n_bins = 13
    bin_days = 7
    smoke_var = "smoke_pm25_mean"

    # Build bins (mean = sum / 7 calendar days, not sum / n_smoke_rows)
    bin_vars = []
    for yr, edate_str in pres_dates.items():
        edate = pd.Timestamp(edate_str)
        earliest = edate - timedelta(days=n_bins * bin_days)
        yr_smoke = smoke_raw[(smoke_raw["date"] > earliest) &
                             (smoke_raw["date"] <= edate)].copy()

        yr_bins = None
        for b in range(n_bins):
            bin_start = edate - timedelta(days=(b + 1) * bin_days)
            bin_end = edate - timedelta(days=b * bin_days)
            w = yr_smoke[(yr_smoke["date"] > bin_start) & (yr_smoke["date"] <= bin_end)]
            bin_mean = (w.groupby("geoid")["smoke_pm25"].sum() / bin_days).rename(f"mean_bin_{b}")
            if yr_bins is None:
                yr_bins = pd.DataFrame(bin_mean)
            else:
                yr_bins = yr_bins.join(bin_mean, how="outer")

        yr_bins["year"] = yr
        yr_bins = yr_bins.reset_index().rename(columns={"geoid": "geoid"})
        yr_bins = yr_bins.set_index(["geoid", "year"])
        bin_vars.append(yr_bins)

    bins_df = pd.concat(bin_vars).fillna(0)
    df_merged = df.join(bins_df, how="inner")

    outcomes = [
        ("dem_vote_share", "DEM Vote Share"),
        ("incumbent_vote_share", "Incumbent Vote Share"),
        ("log_total_votes", "Log Total Votes"),
        ("turnout_rate", "Turnout Rate"),
    ]

    bin_cols = [f"mean_bin_{b}" for b in range(n_bins)]

    def run_cumul_series(data, dep_var):
        cols_needed = [dep_var] + bin_cols + available
        cols_present = [c for c in cols_needed if c in data.columns]
        subset = data[cols_present].dropna().copy()
        coefs, ses = [], []
        for k in range(n_bins):
            cum_var = f"cumul_{k}"
            subset[cum_var] = subset[bin_cols[:k+1]].mean(axis=1)
            avail = [v for v in available if v in subset.columns]
            x_k = sm.add_constant(subset[[cum_var] + avail])
            try:
                mod = PanelOLS(subset[dep_var], x_k, entity_effects=True,
                               time_effects=True, check_rank=False, drop_absorbed=True)
                res = mod.fit(cov_type="clustered", cluster_entity=True)
                coefs.append(res.params.get(cum_var, np.nan))
                ses.append(res.std_errors.get(cum_var, np.nan))
            except Exception:
                coefs.append(np.nan)
                ses.append(np.nan)
        return coefs, ses

    # Run for full sample and drop-2020
    os.makedirs(FIG_DIR, exist_ok=True)
    n_outcomes = len(outcomes)
    fig, axes = plt.subplots(1, n_outcomes, figsize=(4.5 * n_outcomes, 4.5))
    if n_outcomes == 1:
        axes = [axes]
    x_pos = np.arange(n_bins)
    short_labels = [f"{(k+1)*7}" for k in range(n_bins)]

    for col_idx, (dep_var, dep_label) in enumerate(outcomes):
        ax = axes[col_idx]

        # Full sample
        coefs_full, ses_full = run_cumul_series(df_merged, dep_var)
        ax.fill_between(x_pos,
                        [c - 1.96*s for c,s in zip(coefs_full, ses_full)],
                        [c + 1.96*s for c,s in zip(coefs_full, ses_full)],
                        alpha=0.15, color="#2166ac")
        ax.plot(x_pos, coefs_full, "o-", color="#2166ac", linewidth=2,
                markersize=5, label="Full sample")

        # Drop 2020
        df_no2020 = df_merged[df_merged.index.get_level_values(1) != 2020]
        if len(df_no2020) > 100:
            coefs_no20, ses_no20 = run_cumul_series(df_no2020, dep_var)
            ax.fill_between(x_pos,
                            [c - 1.96*s for c,s in zip(coefs_no20, ses_no20)],
                            [c + 1.96*s for c,s in zip(coefs_no20, ses_no20)],
                            alpha=0.15, color="#b2182b")
            ax.plot(x_pos, coefs_no20, "s--", color="#b2182b", linewidth=2,
                    markersize=5, label="Excl. 2020")

        ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, fontsize=8)
        ax.set_title(dep_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Cumulative window (days)", fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle("CA Tract-Level: Full Sample vs. Excluding 2020",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "ca_temporal_drop2020.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# ============================================================
# Threshold Comparison
# ============================================================

def threshold_comparison(df):
    """Compare fraction-above-threshold treatment at 20, 35.5, 55.5 µg/m³."""
    print(f"\n{'='*70}")
    print("THRESHOLD COMPARISON: Fraction of Days Above Threshold (30d, CA)")
    print(f"{'='*70}")

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
            continue
        vals = df[smoke_var].dropna()
        nonzero = (vals > 0).sum()
        print(f"\n  {thresh_label}: {nonzero:,}/{len(vals):,} nonzero "
              f"({100*nonzero/len(vals):.1f}%)")

        for dep_var, dep_label in outcomes:
            res2 = run_twfe(df, dep_var, smoke_var,
                            label=f"{thresh_label} TWFE: {dep_label}")
            res3 = run_twfe(df, dep_var, smoke_var, controls=available,
                            label=f"{thresh_label} +controls: {dep_label}")

            coef2 = res2.params.get(smoke_var, np.nan) if res2 else np.nan
            se2 = res2.std_errors.get(smoke_var, np.nan) if res2 else np.nan
            pval2 = res2.pvalues.get(smoke_var, np.nan) if res2 else np.nan
            stars2 = "***" if pval2 < 0.01 else "**" if pval2 < 0.05 else "*" if pval2 < 0.10 else ""

            coef3 = res3.params.get(smoke_var, np.nan) if res3 else np.nan
            se3 = res3.std_errors.get(smoke_var, np.nan) if res3 else np.nan
            pval3 = res3.pvalues.get(smoke_var, np.nan) if res3 else np.nan
            stars3 = "***" if pval3 < 0.01 else "**" if pval3 < 0.05 else "*" if pval3 < 0.10 else ""

            print(f"    {dep_label:25s}  TWFE: β={coef2:.6f}{stars2:3s} (SE={se2:.6f})"
                  f"  +Ctrl: β={coef3:.6f}{stars3:3s} (SE={se3:.6f})")


# ============================================================
# County×Year FE (most demanding, analogous to state×year in national)
# ============================================================

def county_year_fe_regressions(df):
    """Robustness check: County×Year FE."""
    print(f"\n{'='*70}")
    print("ROBUSTNESS: County×Year Fixed Effects (CA Tract-Level)")
    print(f"{'='*70}")

    smoke_var = "smoke_pm25_mean_30d"
    specs = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
        ("turnout_rate", "Turnout rate"),
    ]

    for dep_var, dep_label in specs:
        res = run_twfe(df, dep_var, smoke_var, county_year_fe=True,
                       drop_absorbed=True,
                       label=f"County×Year FE: {dep_label}")
        print_result(res, f"County×Year FE: {dep_label}", smoke_var)


# ============================================================
# Controls Robustness
# ============================================================

def robustness_controls(df):
    """Robustness: add/remove individual controls."""
    print(f"\n{'='*70}")
    print("ROBUSTNESS: Controls Sensitivity (CA Tract-Level)")
    print(f"{'='*70}")

    smoke_var = "smoke_pm25_mean_30d"
    all_controls = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt", "pct_bachelors_plus",
                    "pct_white_nh", "pct_hispanic"]
    available = [v for v in all_controls if v in df.columns and df[v].notna().any()]

    outcomes = [
        ("dem_vote_share", "DEM"),
        ("incumbent_vote_share", "Inc."),
        ("log_total_votes", "Turn."),
        ("turnout_rate", "T.Rate"),
    ]

    # Base (no controls), full controls, leave-one-out
    print(f"\n  {'Specification':<35s}", end="")
    for _, short in outcomes:
        print(f" {short:>12s}", end="")
    print()
    print("  " + "-" * 71)

    configs = [("No controls", [])] + [("Full controls", available)]
    for ctrl in available:
        configs.append((f"Drop {ctrl}", [c for c in available if c != ctrl]))

    for config_name, ctrls in configs:
        print(f"  {config_name:<35s}", end="")
        for dep_var, _ in outcomes:
            res = run_twfe(df, dep_var, smoke_var,
                           controls=ctrls if ctrls else None,
                           label=config_name)
            if res is not None:
                coef = res.params.get(smoke_var, np.nan)
                pval = res.pvalues.get(smoke_var, np.nan)
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                print(f" {coef:>10.5f}{stars:<2s}", end="")
            else:
                print(f" {'N/A':>12s}", end="")
        print()


# ============================================================
# House Comparison
# ============================================================

def house_comparison(pres_df, house_df):
    """Compare tract-level presidential and House results."""
    print(f"\n{'='*70}")
    print("HOUSE vs. PRESIDENTIAL COMPARISON (CA Tract-Level)")
    print(f"{'='*70}")

    smoke_var = "smoke_pm25_mean_30d"
    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]

    outcomes = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
        ("turnout_rate", "Turnout rate"),
    ]

    print(f"\n  {'Outcome':<30s} {'Presidential':>15s} {'House':>15s}")
    print("  " + "-" * 60)

    for dep_var, dep_label in outcomes:
        results = []
        for df, label in [(pres_df, "Presidential"), (house_df, "House")]:
            if df is None:
                results.append(("N/A", ""))
                continue
            avail = [v for v in control_vars if v in df.columns and df[v].notna().any()]

            # For House, exclude uncontested races for vote share outcomes
            run_df = df
            if label == "House" and "uncontested" in df.columns and dep_var not in ("log_total_votes", "turnout_rate"):
                run_df = df[~df["uncontested"]]

            res = run_twfe(run_df, dep_var, smoke_var, controls=avail,
                           label=f"{label}: {dep_label}")
            if res is not None:
                coef = res.params.get(smoke_var, np.nan)
                pval = res.pvalues.get(smoke_var, np.nan)
                se = res.std_errors.get(smoke_var, np.nan)
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                results.append((f"{coef:.5f}{stars}", f"({se:.5f})"))
            else:
                results.append(("N/A", ""))

        print(f"  {dep_label:<30s} {results[0][0]:>15s} {results[1][0]:>15s}")
        print(f"  {'':30s} {results[0][1]:>15s} {results[1][1]:>15s}")


# ============================================================
# National vs. CA Comparison
# ============================================================

def national_comparison(ca_df):
    """Compare CA tract-level with national county-level results."""
    print(f"\n{'='*70}")
    print("NATIONAL (County) vs. CA (Tract) COMPARISON")
    print(f"{'='*70}")

    if not os.path.exists(NATIONAL_FILE):
        print(f"  National data not found: {NATIONAL_FILE}")
        return

    nat_df = pd.read_parquet(NATIONAL_FILE)
    nat_df = nat_df.set_index(["fips", "year"]).sort_index()

    smoke_var = "smoke_pm25_mean_30d"
    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]

    outcomes = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
        ("turnout_rate", "Turnout rate"),
    ]

    # Run both with Spec 3 (TWFE + controls)
    nat_results = {}
    ca_results = {}

    for dep_var, dep_label in outcomes:
        # National
        nat_avail = [v for v in control_vars if v in nat_df.columns and nat_df[v].notna().any()]
        nat_res = run_twfe(nat_df, dep_var, smoke_var, controls=nat_avail,
                           label=f"National: {dep_label}")
        if nat_res is not None:
            nat_results[dep_var] = {
                "coef": nat_res.params.get(smoke_var, np.nan),
                "se": nat_res.std_errors.get(smoke_var, np.nan),
                "n": int(nat_res.nobs),
            }

        # CA
        ca_avail = [v for v in control_vars if v in ca_df.columns and ca_df[v].notna().any()]
        ca_res = run_twfe(ca_df, dep_var, smoke_var, controls=ca_avail,
                          label=f"CA Tract: {dep_label}")
        if ca_res is not None:
            ca_results[dep_var] = {
                "coef": ca_res.params.get(smoke_var, np.nan),
                "se": ca_res.std_errors.get(smoke_var, np.nan),
                "n": int(ca_res.nobs),
            }

    # Print comparison table
    print(f"\n  Spec 3 (TWFE + Controls), 30d Mean PM2.5")
    print(f"  {'Outcome':<25s} {'National (County)':>20s} {'CA (Tract)':>20s}")
    print("  " + "-" * 65)

    for dep_var, dep_label in outcomes:
        nat = nat_results.get(dep_var, {})
        ca = ca_results.get(dep_var, {})
        nat_str = f"{nat.get('coef', np.nan):.5f}" if nat else "N/A"
        ca_str = f"{ca.get('coef', np.nan):.5f}" if ca else "N/A"
        print(f"  {dep_label:<25s} {nat_str:>20s} {ca_str:>20s}")
        nat_se = f"({nat.get('se', np.nan):.5f})" if nat else ""
        ca_se = f"({ca.get('se', np.nan):.5f})" if ca else ""
        print(f"  {'':25s} {nat_se:>20s} {ca_se:>20s}")

    # Coefficient comparison plot
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    y_positions = np.arange(len(outcomes))
    offset = 0.15

    for dep_idx, (dep_var, dep_label) in enumerate(outcomes):
        nat = nat_results.get(dep_var, {})
        ca = ca_results.get(dep_var, {})

        if nat:
            ax.errorbar(nat["coef"], dep_idx - offset,
                        xerr=1.96 * nat["se"],
                        fmt="o", color="#2166ac", markersize=8, capsize=4,
                        label="National (county)" if dep_idx == 0 else "")
        if ca:
            ax.errorbar(ca["coef"], dep_idx + offset,
                        xerr=1.96 * ca["se"],
                        fmt="s", color="#b2182b", markersize=8, capsize=4,
                        label="CA (tract)" if dep_idx == 0 else "")

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in outcomes])
    ax.set_xlabel("Coefficient on Smoke PM2.5 (30d mean)")
    ax.set_title("National County-Level vs. CA Tract-Level Estimates", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "ca_national_comparison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("CA Tract-Level Analysis: Wildfire Smoke and Voting Behavior")
    print("=" * 70)

    os.makedirs(FIG_DIR, exist_ok=True)

    # Load presidential data
    pres_df = None
    if os.path.exists(PRES_FILE):
        pres_df = load_data(PRES_FILE, id_col="geoid")
    else:
        print(f"\nPresidential data not found: {PRES_FILE}")

    # Load house data
    house_df = None
    if os.path.exists(HOUSE_FILE):
        house_df = load_data(HOUSE_FILE, id_col="geoid")
    else:
        print(f"\nHouse data not found: {HOUSE_FILE}")

    if pres_df is None and house_df is None:
        print("\nERROR: No analysis data available. Run build scripts first.")
        return

    # Use presidential as primary
    df = pres_df if pres_df is not None else house_df

    # Summary statistics
    summary_statistics(df)

    # Build-up specification table
    if pres_df is not None:
        create_buildup_table(pres_df)

    # Temporal dynamics with controls
    if pres_df is not None:
        temporal_dynamics_controls(pres_df)

    # Drop-2020 temporal comparison
    if pres_df is not None:
        temporal_drop2020(pres_df)

    # Threshold comparison
    threshold_comparison(df)

    # County×Year FE
    county_year_fe_regressions(df)

    # Controls robustness
    robustness_controls(df)

    # House comparison
    if pres_df is not None and house_df is not None:
        house_comparison(pres_df, house_df)

    # National comparison
    if pres_df is not None:
        national_comparison(pres_df)

    print(f"\n{'='*70}")
    print(f"CA analysis complete. Figures saved to: {FIG_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

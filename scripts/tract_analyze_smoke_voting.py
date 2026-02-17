#!/usr/bin/env python3
"""
National Tract-Level Analysis: Wildfire Smoke and Voting Behavior.

Matches the CA tract analysis structure for comparability:
  - Build-up table (raw OLS -> TWFE -> +controls -> +state x year FE)
  - Temporal dynamics (cumulative 7-day bins with controls)
  - Threshold comparison (haze / USG / unhealthy)
  - Robustness checks (state x year FE, controls sensitivity)
  - Heterogeneity tests
  - Comparison with county-level and CA tract-level results

FE structure: Tract + Year (base), +Controls, State x Year (most demanding).
  With only 2 years, state trends = state x year FE, so Spec 4 uses state x year.
Clustering: By tract (primary).
"""

import os
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
DATA_FILE = os.path.join(BASE_DIR, "output", "national_tracts",
                         "tract_smoke_voting_presidential.parquet")
SMOKE_FILE = os.path.join(BASE_DIR, "data", "smoke", "tract",
                          "smokePM2pt5_predictions_daily_tract_20060101-20231231.csv")
NATIONAL_COUNTY_FILE = os.path.join(BASE_DIR, "output",
                                     "smoke_voting_analysis.parquet")
CA_TRACT_FILE = os.path.join(BASE_DIR, "output", "california",
                              "ca_smoke_voting_presidential.parquet")
FIG_DIR = os.path.join(BASE_DIR, "output", "national_tracts", "figures")

ELECTION_DATES = {
    2016: "2016-11-08",
    2020: "2020-11-03",
}

# EPA thresholds
HAZE_THRESHOLD = 20.0
EPA_USG_THRESHOLD = 35.5
EPA_UNHEALTHY = 55.5

# CONUS state FIPS codes
CONUS_FIPS = {
    "01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
    "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
    "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
    "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
    "47", "48", "49", "50", "51", "53", "54", "55", "56",
}


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
             absorb_time=True, state_year_fe=False, drop_absorbed=False,
             cluster_by_county=True, label=""):
    """Run a two-way fixed effects regression using linearmodels PanelOLS.

    Clusters by county FIPS (first 5 digits of tract GEOID) by default,
    since smoke treatment varies at the county/regional level and tract-level
    clustering understates standard errors (Moulton 1990).

    For state x year FE, pass state_year_fe=True. This uses tract entity FE
    plus state x year as other_effects (absorbing all state-year variation).
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
        if state_year_fe:
            state_year_cat = pd.Categorical(
                subset.index.get_level_values(0).astype(str).str[:2] + "_" +
                subset.index.get_level_values(1).astype(str)
            )
            other_ef = pd.DataFrame(state_year_cat, index=subset.index,
                                    columns=["state_year"])
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

        if cluster_by_county:
            # Cluster by county FIPS (first 5 digits of tract GEOID)
            county_clusters = pd.DataFrame(
                subset.index.get_level_values(0).astype(str).str[:5].values,
                index=subset.index, columns=["county"]
            )
            res = mod.fit(cov_type="clustered", clusters=county_clusters)
        else:
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
    print(f"    beta = {coef:.6f} {stars}  (SE = {se:.6f})")
    print(f"    95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"    p-value = {pval:.4f}")
    print(f"    N = {n:,}")
    print(f"    R2 (within) = {res.rsquared_within:.4f}")

    return {"label": label, "coef": coef, "se": se, "pval": pval,
            "ci_low": ci_low, "ci_high": ci_high, "n": n,
            "r2_within": res.rsquared_within}


# ============================================================
# Summary Statistics
# ============================================================

def summary_statistics(df):
    """Print summary statistics."""
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS: National Tract-Level")
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
        ("turnout_rate", "Turnout rate (votes/VAP)"),
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

    print(f"\n  {'Variable':<30s} {'N':>8s} {'Mean':>10s} {'SD':>10s} "
          f"{'Min':>10s} {'Max':>10s}")
    print("  " + "-" * 78)

    for var_list, header in [(smoke_vars, "Smoke Exposure"),
                             (outcome_vars, "Outcomes"),
                             (control_vars, "Controls")]:
        print(f"\n  {header}")
        for var, vlabel in var_list:
            if var not in df.columns:
                continue
            s = df[var].dropna()
            if len(s) == 0:
                continue
            print(f"  {vlabel:<30s} {len(s):>8,} {s.mean():>10.4f} "
                  f"{s.std():>10.4f} {s.min():>10.4f} {s.max():>10.4f}")


# ============================================================
# Build-Up Table
# ============================================================

def create_buildup_table(df):
    """Build-up specification table.

    (1) Raw OLS  (2) Tract+Year FE  (3) +Controls  (4) +State x Year FE
    With 2 periods, state trends = state x year FE, so Spec 4 uses the latter.
    """
    print(f"\n{'='*70}")
    print("BUILD-UP SPECIFICATION TABLE (National Tract-Level Presidential)")
    print("  (1) Raw OLS  (2) Tract+Year FE  (3) +Controls  (4) +State x Year FE")
    print(f"{'='*70}")

    smoke_var = "smoke_pm25_mean_30d"
    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]
    print(f"  Available controls: {available}")

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

        # Spec 1: Raw OLS
        res1 = run_twfe(df, dep_var, smoke_var,
                        absorb_entity=False, absorb_time=False,
                        label=f"(1) Raw OLS: {dep_var}")
        # Spec 2: Tract + Year FE
        res2 = run_twfe(df, dep_var, smoke_var,
                        label=f"(2) TWFE: {dep_var}")
        # Spec 3: + Controls
        res3 = run_twfe(df, dep_var, smoke_var, controls=available,
                        label=f"(3) +Controls: {dep_var}")
        # Spec 4: + State x Year FE (replaces state trends with 2 periods)
        res4 = run_twfe(df, dep_var, smoke_var, controls=available,
                        state_year_fe=True, drop_absorbed=True,
                        label=f"(4) +State x Year FE: {dep_var}")

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
    print(f"\n{'-'*85}")
    print(f"  {'':30s} {'(1)':>12s} {'(2)':>12s} {'(3)':>12s} {'(4)':>12s}")
    print(f"  {'':30s} {'Raw OLS':>12s} {'TWFE':>12s} {'+Controls':>12s} {'+St.xYr FE':>12s}")
    print("-" * 85)

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
    print(f"  {'Year FE':<30s} {'':>12s} {'Yes':>12s} {'Yes':>12s} {'':>12s}")
    print(f"  {'Controls':<30s} {'':>12s} {'':>12s} {'Yes':>12s} {'Yes':>12s}")
    print(f"  {'State x Year FE':<30s} {'':>12s} {'':>12s} {'':>12s} {'Yes':>12s}")
    print(f"  {'Cluster by':<30s} {'County':>12s} {'County':>12s} {'County':>12s} {'County':>12s}")
    print("-" * 85)

    return results


# ============================================================
# Temporal Dynamics (7-day bins with controls)
# ============================================================

def temporal_dynamics_controls(df):
    """7-day temporal dynamics with controls for mean PM2.5 and frac haze."""
    print(f"\n{'='*70}")
    print("TEMPORAL DYNAMICS WITH CONTROLS: 7-Day Windows (National Tracts)")
    print(f"{'='*70}")

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    # Load raw daily smoke data
    print("  Loading raw daily smoke data...")
    if not os.path.exists(SMOKE_FILE):
        print(f"  ERROR: Smoke file not found: {SMOKE_FILE}")
        return

    # Read only the date ranges we need (chunked for memory)
    date_ranges = []
    for yr, edate_str in ELECTION_DATES.items():
        edate = pd.Timestamp(edate_str)
        earliest = edate - timedelta(days=13 * 7)
        date_ranges.append((earliest, edate))

    min_date = min(s for s, _ in date_ranges)
    max_date = max(e for _, e in date_ranges)

    chunks = []
    for chunk in pd.read_csv(SMOKE_FILE, chunksize=2_000_000,
                             dtype={"GEOID": str}):
        chunk["date"] = pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d")
        mask = pd.Series(False, index=chunk.index)
        for start, end in date_ranges:
            mask |= (chunk["date"] >= start) & (chunk["date"] <= end)
        filtered = chunk[mask].copy()
        filtered["state_fips"] = filtered["GEOID"].str.zfill(11).str[:2]
        filtered = filtered[filtered["state_fips"].isin(CONUS_FIPS)]
        filtered = filtered.drop(columns=["state_fips"])
        if len(filtered) > 0:
            chunks.append(filtered)

    smoke_raw = pd.concat(chunks, ignore_index=True)
    smoke_raw = smoke_raw.rename(columns={"GEOID": "geoid", "smokePM_pred": "smoke_pm25"})
    smoke_raw["geoid"] = smoke_raw["geoid"].str.zfill(11)
    print(f"  Loaded {len(smoke_raw):,} smoke-day rows for temporal analysis")

    # Only use election years in the data
    pres_dates = {yr: dt for yr, dt in ELECTION_DATES.items()
                  if yr in df.index.get_level_values(1).unique()}

    # Compute 13 non-overlapping 7-day bins
    n_bins = 13
    bin_days = 7
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
            bin_n_haze = w.groupby("geoid")["smoke_pm25"].apply(
                lambda x: (x > HAZE_THRESHOLD).sum()
            )
            bin_frac = (bin_n_haze / bin_days).rename(f"frac_bin_{b}")

            if yr_bins is None:
                yr_bins = pd.concat([bin_mean, bin_frac], axis=1)
            else:
                yr_bins = yr_bins.join(bin_mean, how="outer").join(bin_frac, how="outer")

        yr_bins["year"] = yr
        yr_bins = yr_bins.reset_index().rename(columns={"geoid": "geoid"})
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
         "tract_temporal_7day_mean_controls.png"),
        ("frac_bin", "Frac. Days > 20 \u00b5g/m\u00b3 (Haze)", "Frac. Days Haze",
         "tract_temporal_7day_frac_controls.png"),
    ]

    all_treat_results = {}  # Collect results across treatments for combined figure

    for prefix, fig_treat_label, print_treat_label, fig_name in treatments:
        print(f"\n  === Treatment: {print_treat_label} ===")
        bin_cols = [f"{prefix}_{b}" for b in range(n_bins)]

        all_results = {}

        for dep_var, dep_label in outcomes:
            print(f"\n    --- {dep_label} ---")

            cols_needed = [dep_var] + bin_cols + available
            subset = df_merged[[c for c in cols_needed if c in df_merged.columns]].dropna().copy()
            if len(subset) < 100:
                print(f"      SKIP: only {len(subset)} obs")
                continue

            cumul_coefs, cumul_ses = [], []
            for k in range(n_bins):
                cum_var = f"{prefix}_cumul_{k}"
                subset[cum_var] = subset[bin_cols[:k + 1]].mean(axis=1)
                avail_in_subset = [v for v in available if v in subset.columns]
                x_k = sm.add_constant(subset[[cum_var] + avail_in_subset])

                try:
                    mod_k = PanelOLS(subset[dep_var], x_k,
                                     entity_effects=True, time_effects=True,
                                     check_rank=False, drop_absorbed=True)
                    county_clusters = pd.DataFrame(
                        subset.index.get_level_values(0).astype(str).str[:5].values,
                        index=subset.index, columns=["county"]
                    )
                    res_k = mod_k.fit(cov_type="clustered", clusters=county_clusters)
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
                strs = "***" if pval_k < 0.01 else "**" if pval_k < 0.05 else "*" if pval_k < 0.10 else ""
                print(f"        0-{(k+1)*7}d: beta={cumul_coefs[k]:.6f} "
                      f"{strs} (SE={cumul_ses[k]:.6f})")

            all_results[dep_var] = {
                "label": dep_label,
                "cumul_coefs": cumul_coefs, "cumul_ses": cumul_ses,
            }

        # Plot cumulative figure
        if all_results:
            os.makedirs(FIG_DIR, exist_ok=True)
            fig, axes = plt.subplots(1, len(outcomes), figsize=(4.5 * len(outcomes), 4.5))
            x_pos = np.arange(n_bins)
            short_labels = [f"{(k+1)*7}" for k in range(n_bins)]
            color = "#b2182b"

            for col_idx, (dep_var, dep_label) in enumerate(outcomes):
                if dep_var not in all_results:
                    continue
                r = all_results[dep_var]
                ax = axes[col_idx]
                ax.fill_between(x_pos,
                                [c - 1.96 * s for c, s in zip(r["cumul_coefs"], r["cumul_ses"])],
                                [c + 1.96 * s for c, s in zip(r["cumul_coefs"], r["cumul_ses"])],
                                alpha=0.2, color=color)
                ax.plot(x_pos, r["cumul_coefs"], "o-", color=color,
                        linewidth=2, markersize=5)
                ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(short_labels, fontsize=8)
                ax.set_title(dep_label, fontsize=11, fontweight="bold")
                ax.set_xlabel("Cumulative window (days)", fontsize=9)
                ax.tick_params(axis="y", labelsize=9)

            fig.suptitle(f"National Tract-Level: Temporal Dynamics ({print_treat_label})",
                         fontsize=13, fontweight="bold", y=1.02)
            plt.tight_layout()
            fig_path = os.path.join(FIG_DIR, fig_name)
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"\n  Saved figure: {fig_path}")

        all_treat_results[prefix] = all_results

    # ---- State x Year FE cumulative temporal dynamics ----
    print(f"\n  === State x Year FE Cumulative Temporal Dynamics ===")

    sy_treat_results = {}  # prefix -> {dep_var -> {cumul_coefs, cumul_ses}}

    for prefix, fig_treat_label, print_treat_label, fig_name in treatments:
        print(f"\n  --- State x Year FE: {print_treat_label} ---")
        bin_cols = [f"{prefix}_{b}" for b in range(n_bins)]
        sy_results = {}

        for dep_var, dep_label in outcomes:
            print(f"\n    {dep_label}")
            cols_needed = [dep_var] + bin_cols + available
            subset = df_merged[[c for c in cols_needed if c in df_merged.columns]].dropna().copy()
            if len(subset) < 100:
                print(f"      SKIP: only {len(subset)} obs")
                continue

            cumul_coefs, cumul_ses = [], []
            for k in range(n_bins):
                cum_var = f"{prefix}_cumul_sy_{k}"
                subset[cum_var] = subset[bin_cols[:k + 1]].mean(axis=1)
                avail_in_subset = [v for v in available if v in subset.columns]
                x_k = sm.add_constant(subset[[cum_var] + avail_in_subset])
                y_k = subset[dep_var]

                try:
                    state_year_cat = pd.Categorical(
                        subset.index.get_level_values(0).astype(str).str[:2] + "_" +
                        subset.index.get_level_values(1).astype(str)
                    )
                    state_year_df = pd.DataFrame(state_year_cat, index=subset.index,
                                                 columns=["state_year"])
                    mod_sy = PanelOLS(y_k, x_k, entity_effects=True, time_effects=False,
                                      other_effects=state_year_df, check_rank=False,
                                      drop_absorbed=True)
                    county_clusters = pd.DataFrame(
                        subset.index.get_level_values(0).astype(str).str[:5].values,
                        index=subset.index, columns=["county"]
                    )
                    res_sy = mod_sy.fit(cov_type="clustered", clusters=county_clusters)
                    cumul_coefs.append(res_sy.params.get(cum_var, np.nan))
                    cumul_ses.append(res_sy.std_errors.get(cum_var, np.nan))
                except Exception:
                    cumul_coefs.append(np.nan)
                    cumul_ses.append(np.nan)

            print(f"      Cumulative (State x Year FE):")
            for k in range(n_bins):
                pval_k = np.nan
                if not np.isnan(cumul_coefs[k]) and cumul_ses[k] > 0:
                    from scipy import stats as sp_stats
                    t_stat = cumul_coefs[k] / cumul_ses[k]
                    pval_k = 2 * (1 - sp_stats.t.cdf(abs(t_stat), df=100))
                strs = "***" if pval_k < 0.01 else "**" if pval_k < 0.05 else "*" if pval_k < 0.10 else ""
                print(f"        0-{(k+1)*7}d: beta={cumul_coefs[k]:.6f} "
                      f"{strs} (SE={cumul_ses[k]:.6f})")

            sy_results[dep_var] = {
                "label": dep_label,
                "cumul_coefs": cumul_coefs, "cumul_ses": cumul_ses,
            }

        sy_treat_results[prefix] = sy_results

    # ---- Combined figure: Spec 3 vs State x Year FE ----
    treat_prefixes = [t[0] for t in treatments]
    treat_labels_short = [t[2] for t in treatments]

    # Only plot if we have 2 treatments (mean_bin, frac_bin)
    n_treat = len(treat_prefixes)
    n_out = len(outcomes)
    fig, axes = plt.subplots(n_out, n_treat, figsize=(5.5 * n_treat, 3.8 * n_out),
                              squeeze=False)
    x_pos = np.arange(n_bins)
    short_labels = [f"{(k+1)*7}" for k in range(n_bins)]

    for col_idx, prefix in enumerate(treat_prefixes):
        spec3 = all_treat_results.get(prefix, {})
        sy = sy_treat_results.get(prefix, {})

        for row_idx, (dep_var, dep_label) in enumerate(outcomes):
            ax = axes[row_idx, col_idx]

            # Spec 3 (TWFE + Controls) — blue solid
            if dep_var in spec3:
                r3 = spec3[dep_var]
                c3 = np.array(r3["cumul_coefs"])
                s3 = np.array(r3["cumul_ses"])
                ax.fill_between(x_pos, c3 - 1.96 * s3, c3 + 1.96 * s3,
                                alpha=0.15, color="#2166ac")
                ax.plot(x_pos, c3, "o-", color="#2166ac", linewidth=2,
                        markersize=4, label="TWFE + Controls")

            # State x Year FE — red dashed
            if dep_var in sy:
                rsy = sy[dep_var]
                csy = np.array(rsy["cumul_coefs"])
                ssy = np.array(rsy["cumul_ses"])
                ax.fill_between(x_pos, csy - 1.96 * ssy, csy + 1.96 * ssy,
                                alpha=0.15, color="#b2182b")
                ax.plot(x_pos, csy, "s--", color="#b2182b", linewidth=2,
                        markersize=4, label="State x Year FE")

            ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(short_labels, fontsize=7)
            ax.tick_params(axis="y", labelsize=8)

            if row_idx == 0:
                ax.set_title(treat_labels_short[col_idx], fontsize=11, fontweight="bold")
            if row_idx == n_out - 1:
                ax.set_xlabel("Cumulative window (days)", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(dep_label, fontsize=9)

            # Legend in top-left panel only
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("National Tract-Level: Cumulative Temporal Dynamics\n"
                 "TWFE + Controls vs. State x Year FE",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, "tract_temporal_cumulative_stateyear.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved combined figure: {fig_path}")


# ============================================================
# Close-In Daily Temporal Dynamics (1-7 days)
# ============================================================

def temporal_closein_controls(df):
    """Daily-resolution cumulative windows for days 1-7 before election."""
    print(f"\n{'='*70}")
    print("CLOSE-IN TEMPORAL DYNAMICS: Daily Windows (1-7 days, National Tracts)")
    print(f"{'='*70}")

    n_days = 7

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    # Load raw daily smoke data (chunked for memory)
    print("  Loading raw daily smoke data...")
    if not os.path.exists(SMOKE_FILE):
        print(f"  ERROR: Smoke file not found: {SMOKE_FILE}")
        return

    date_ranges = []
    for yr, edate_str in ELECTION_DATES.items():
        edate = pd.Timestamp(edate_str)
        earliest = edate - timedelta(days=n_days)
        date_ranges.append((earliest, edate))

    chunks = []
    for chunk in pd.read_csv(SMOKE_FILE, chunksize=2_000_000,
                             dtype={"GEOID": str}):
        chunk["date"] = pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d")
        mask = pd.Series(False, index=chunk.index)
        for start, end in date_ranges:
            mask |= (chunk["date"] >= start) & (chunk["date"] <= end)
        filtered = chunk[mask].copy()
        filtered["state_fips"] = filtered["GEOID"].str.zfill(11).str[:2]
        filtered = filtered[filtered["state_fips"].isin(CONUS_FIPS)]
        filtered = filtered.drop(columns=["state_fips"])
        if len(filtered) > 0:
            chunks.append(filtered)

    smoke_raw = pd.concat(chunks, ignore_index=True)
    smoke_raw = smoke_raw.rename(columns={"GEOID": "geoid", "smokePM_pred": "smoke_pm25"})
    smoke_raw["geoid"] = smoke_raw["geoid"].str.zfill(11)
    print(f"  Loaded {len(smoke_raw):,} smoke-day rows")

    pres_dates = {yr: dt for yr, dt in ELECTION_DATES.items()
                  if yr in df.index.get_level_values(1).unique()}

    # Compute 7 single-day bins (smoke-days-only file: divide by 1 calendar day)
    bin_vars = []
    for yr, edate_str in pres_dates.items():
        edate = pd.Timestamp(edate_str)
        earliest = edate - timedelta(days=n_days)
        yr_smoke = smoke_raw[(smoke_raw["date"] > earliest) &
                             (smoke_raw["date"] <= edate)].copy()

        yr_bins = None
        for d in range(n_days):
            day_start = edate - timedelta(days=d + 1)
            day_end = edate - timedelta(days=d)
            w = yr_smoke[(yr_smoke["date"] > day_start) & (yr_smoke["date"] <= day_end)]

            bin_mean = (w.groupby("geoid")["smoke_pm25"].sum() / 1).rename(f"mean_day_{d}")
            bin_n_haze = w.groupby("geoid")["smoke_pm25"].apply(
                lambda x: (x > HAZE_THRESHOLD).sum()
            )
            bin_frac = (bin_n_haze / 1).rename(f"frac_day_{d}")

            if yr_bins is None:
                yr_bins = pd.concat([bin_mean, bin_frac], axis=1)
            else:
                yr_bins = yr_bins.join(bin_mean, how="outer").join(bin_frac, how="outer")

        yr_bins["year"] = yr
        yr_bins = yr_bins.reset_index().rename(columns={"geoid": "geoid"})
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
                    county_clusters = pd.DataFrame(
                        subset.index.get_level_values(0).astype(str).str[:5].values,
                        index=subset.index, columns=["county"]
                    )
                    res_k = mod_k.fit(cov_type="clustered", clusters=county_clusters)
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
                print(f"        0-{k+1}d: beta={cumul_coefs[k]:.6f} {stars} (SE={cumul_ses[k]:.6f})")

            all_results[dep_var] = {
                "label": dep_label,
                "cumul_coefs": cumul_coefs, "cumul_ses": cumul_ses,
            }

        all_treat_results[prefix] = all_results

    # Figure
    treat_configs = [
        ("mean_day", "Mean Smoke PM$_{2.5}$", "#2166ac"),
        ("frac_day", "Frac. Days Haze", "#b2182b"),
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

        fig.suptitle("Close-In Temporal Dynamics (National Tracts)", fontsize=14,
                     fontweight="bold", y=1.01)
        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, "tract_temporal_closein_daily.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved close-in figure: {fig_path}")


# ============================================================
# Threshold Comparison
# ============================================================

def threshold_comparison(df):
    """Compare fraction-above-threshold treatment at 20, 35.5, 55.5 ug/m3."""
    print(f"\n{'='*70}")
    print("THRESHOLD COMPARISON: Fraction of Days Above Threshold (30d)")
    print(f"{'='*70}")

    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    thresholds = [
        ("smoke_frac_haze_30d", "Haze (>20 ug/m3)"),
        ("smoke_frac_usg_30d", "USG (>35.5 ug/m3)"),
        ("smoke_frac_unhealthy_30d", "Unhealthy (>55.5 ug/m3)"),
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

            print(f"    {dep_label:25s}  TWFE: beta={coef2:.6f}{stars2:3s} "
                  f"(SE={se2:.6f})  +Ctrl: beta={coef3:.6f}{stars3:3s} "
                  f"(SE={se3:.6f})")


# ============================================================
# State x Year FE (most demanding)
# ============================================================

def state_year_fe_regressions(df):
    """Robustness check: State x Year FE."""
    print(f"\n{'='*70}")
    print("ROBUSTNESS: State x Year Fixed Effects (National Tract-Level)")
    print(f"{'='*70}")

    smoke_var = "smoke_pm25_mean_30d"
    specs = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
    ]

    for dep_var, dep_label in specs:
        res = run_twfe(df, dep_var, smoke_var, state_year_fe=True,
                       drop_absorbed=True,
                       label=f"State x Year FE: {dep_label}")
        print_result(res, f"State x Year FE: {dep_label}", smoke_var)


# ============================================================
# Controls Robustness
# ============================================================

def robustness_controls(df):
    """Robustness: add/remove individual controls."""
    print(f"\n{'='*70}")
    print("ROBUSTNESS: Controls Sensitivity (National Tract-Level)")
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
# Heterogeneity Tests
# ============================================================

def heterogeneity_tests(df):
    """Subgroup analysis: by state smoke tercile, urbanicity, partisanship."""
    print(f"\n{'='*70}")
    print("HETEROGENEITY TESTS (National Tract-Level)")
    print(f"{'='*70}")

    smoke_var = "smoke_pm25_mean_30d"
    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]
    available = [v for v in control_vars if v in df.columns and df[v].notna().any()]

    outcomes = [
        ("dem_vote_share", "DEM"),
        ("incumbent_vote_share", "Inc."),
        ("log_total_votes", "Turn."),
        ("turnout_rate", "T.Rate"),
    ]

    # 1. By state smoke tercile (based on state-level mean smoke exposure)
    print("\n  --- By State Smoke Tercile ---")
    state_smoke = df.groupby(df.index.get_level_values(0).str[:2])["smoke_pm25_mean_30d"].mean()
    tercile_cuts = state_smoke.quantile([1/3, 2/3])
    low_states = set(state_smoke[state_smoke <= tercile_cuts.iloc[0]].index)
    high_states = set(state_smoke[state_smoke > tercile_cuts.iloc[1]].index)
    mid_states = set(state_smoke.index) - low_states - high_states

    for group_name, states in [("Low smoke states", low_states),
                                ("Mid smoke states", mid_states),
                                ("High smoke states", high_states)]:
        mask = df.index.get_level_values(0).str[:2].isin(states)
        sub = df[mask]
        print(f"\n    {group_name} (n={len(sub):,}, {len(states)} states):")
        for dep_var, dep_label in outcomes:
            res = run_twfe(sub, dep_var, smoke_var, controls=available,
                           label=f"{group_name}: {dep_label}")
            if res is not None:
                coef = res.params.get(smoke_var, np.nan)
                pval = res.pvalues.get(smoke_var, np.nan)
                se = res.std_errors.get(smoke_var, np.nan)
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                print(f"      {dep_label}: beta={coef:.6f}{stars} (SE={se:.6f})")

    # 2. By urbanicity (log_population tercile)
    if "log_population" in df.columns:
        print("\n  --- By Urbanicity (Log Population Tercile) ---")
        pop_cuts = df["log_population"].dropna().quantile([1/3, 2/3])
        for group_name, mask in [
            ("Low population", df["log_population"] <= pop_cuts.iloc[0]),
            ("High population", df["log_population"] > pop_cuts.iloc[1]),
        ]:
            sub = df[mask]
            print(f"\n    {group_name} (n={len(sub):,}):")
            for dep_var, dep_label in outcomes:
                res = run_twfe(sub, dep_var, smoke_var, controls=available,
                               label=f"{group_name}: {dep_label}")
                if res is not None:
                    coef = res.params.get(smoke_var, np.nan)
                    pval = res.pvalues.get(smoke_var, np.nan)
                    se = res.std_errors.get(smoke_var, np.nan)
                    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                    print(f"      {dep_label}: beta={coef:.6f}{stars} (SE={se:.6f})")

    # 3. By partisanship (2016 DEM vote share)
    if "dem_vote_share" in df.columns:
        print("\n  --- By Partisanship (2016 DEM Vote Share) ---")
        dem_2016 = df.xs(2016, level=1)["dem_vote_share"] if 2016 in df.index.get_level_values(1) else None
        if dem_2016 is not None:
            median_dem = dem_2016.median()
            blue_tracts = set(dem_2016[dem_2016 > median_dem].index)
            red_tracts = set(dem_2016[dem_2016 <= median_dem].index)

            for group_name, tract_set in [("Dem-leaning tracts", blue_tracts),
                                           ("Rep-leaning tracts", red_tracts)]:
                mask = df.index.get_level_values(0).isin(tract_set)
                sub = df[mask]
                print(f"\n    {group_name} (n={len(sub):,}):")
                for dep_var, dep_label in outcomes:
                    res = run_twfe(sub, dep_var, smoke_var, controls=available,
                                   label=f"{group_name}: {dep_label}")
                    if res is not None:
                        coef = res.params.get(smoke_var, np.nan)
                        pval = res.pvalues.get(smoke_var, np.nan)
                        se = res.std_errors.get(smoke_var, np.nan)
                        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                        print(f"      {dep_label}: beta={coef:.6f}{stars} (SE={se:.6f})")


# ============================================================
# County-Level Comparison
# ============================================================

def county_level_comparison(df):
    """Compare national tract-level with national county-level results."""
    print(f"\n{'='*70}")
    print("NATIONAL TRACT vs. COUNTY-LEVEL COMPARISON")
    print(f"{'='*70}")

    if not os.path.exists(NATIONAL_COUNTY_FILE):
        print(f"  County data not found: {NATIONAL_COUNTY_FILE}")
        return

    county_df = pd.read_parquet(NATIONAL_COUNTY_FILE)
    # Filter to same years (2016, 2020)
    county_df = county_df[county_df["year"].isin([2016, 2020])].copy()
    county_df = county_df.set_index(["fips", "year"]).sort_index()
    print(f"  County data: {len(county_df):,} obs, {county_df.index.get_level_values(0).nunique():,} counties")

    smoke_var = "smoke_pm25_mean_30d"
    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]

    outcomes = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
        ("turnout_rate", "Turnout rate"),
    ]

    tract_results = {}
    county_results = {}

    for dep_var, dep_label in outcomes:
        # Tract
        tr_avail = [v for v in control_vars if v in df.columns and df[v].notna().any()]
        tr_res = run_twfe(df, dep_var, smoke_var, controls=tr_avail,
                          label=f"Tract: {dep_label}")
        if tr_res is not None:
            tract_results[dep_var] = {
                "coef": tr_res.params.get(smoke_var, np.nan),
                "se": tr_res.std_errors.get(smoke_var, np.nan),
                "n": int(tr_res.nobs),
            }

        # County
        ct_avail = [v for v in control_vars if v in county_df.columns and county_df[v].notna().any()]
        ct_res = run_twfe(county_df, dep_var, smoke_var, controls=ct_avail,
                          label=f"County: {dep_label}")
        if ct_res is not None:
            county_results[dep_var] = {
                "coef": ct_res.params.get(smoke_var, np.nan),
                "se": ct_res.std_errors.get(smoke_var, np.nan),
                "n": int(ct_res.nobs),
            }

    # Print comparison
    print(f"\n  Spec 3 (TWFE + Controls), 30d Mean PM2.5")
    print(f"  {'Outcome':<25s} {'Tract':>20s} {'County':>20s}")
    print("  " + "-" * 65)

    for dep_var, dep_label in outcomes:
        tr = tract_results.get(dep_var, {})
        ct = county_results.get(dep_var, {})
        tr_str = f"{tr.get('coef', np.nan):.5f}" if tr else "N/A"
        ct_str = f"{ct.get('coef', np.nan):.5f}" if ct else "N/A"
        print(f"  {dep_label:<25s} {tr_str:>20s} {ct_str:>20s}")
        tr_se = f"({tr.get('se', np.nan):.5f})" if tr else ""
        ct_se = f"({ct.get('se', np.nan):.5f})" if ct else ""
        print(f"  {'':25s} {tr_se:>20s} {ct_se:>20s}")
        tr_n = f"N={tr.get('n', 0):,}" if tr else ""
        ct_n = f"N={ct.get('n', 0):,}" if ct else ""
        print(f"  {'':25s} {tr_n:>20s} {ct_n:>20s}")

    # Plot comparison
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    y_positions = np.arange(len(outcomes))
    offset = 0.15

    for dep_idx, (dep_var, dep_label) in enumerate(outcomes):
        tr = tract_results.get(dep_var, {})
        ct = county_results.get(dep_var, {})

        if tr:
            ax.errorbar(tr["coef"], dep_idx - offset,
                        xerr=1.96 * tr["se"],
                        fmt="o", color="#b2182b", markersize=8, capsize=4,
                        label="Tract-level" if dep_idx == 0 else "")
        if ct:
            ax.errorbar(ct["coef"], dep_idx + offset,
                        xerr=1.96 * ct["se"],
                        fmt="s", color="#2166ac", markersize=8, capsize=4,
                        label="County-level" if dep_idx == 0 else "")

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in outcomes])
    ax.set_xlabel("Coefficient on Smoke PM2.5 (30d mean)")
    ax.set_title("National: Tract-Level vs. County-Level Estimates", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "tract_comparison_county.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# ============================================================
# CA Comparison
# ============================================================

def ca_comparison(df):
    """Compare national tract-level with CA tract-level results."""
    print(f"\n{'='*70}")
    print("NATIONAL TRACT vs. CA TRACT COMPARISON")
    print(f"{'='*70}")

    if not os.path.exists(CA_TRACT_FILE):
        print(f"  CA tract data not found: {CA_TRACT_FILE}")
        return

    ca_df = pd.read_parquet(CA_TRACT_FILE)
    # Filter to same years (2016, 2020)
    ca_df = ca_df[ca_df["year"].isin([2016, 2020])].copy()
    ca_df = ca_df.set_index(["geoid", "year"]).sort_index()
    print(f"  CA tract data: {len(ca_df):,} obs, "
          f"{ca_df.index.get_level_values(0).nunique():,} tracts")

    smoke_var = "smoke_pm25_mean_30d"
    control_vars = ["unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt"]

    outcomes = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
        ("turnout_rate", "Turnout rate"),
    ]

    natl_results = {}
    ca_results = {}

    for dep_var, dep_label in outcomes:
        # National
        n_avail = [v for v in control_vars if v in df.columns and df[v].notna().any()]
        n_res = run_twfe(df, dep_var, smoke_var, controls=n_avail,
                         label=f"National: {dep_label}")
        if n_res is not None:
            natl_results[dep_var] = {
                "coef": n_res.params.get(smoke_var, np.nan),
                "se": n_res.std_errors.get(smoke_var, np.nan),
                "n": int(n_res.nobs),
            }

        # CA
        ca_avail = [v for v in control_vars if v in ca_df.columns and ca_df[v].notna().any()]
        ca_res = run_twfe(ca_df, dep_var, smoke_var, controls=ca_avail,
                          label=f"CA: {dep_label}")
        if ca_res is not None:
            ca_results[dep_var] = {
                "coef": ca_res.params.get(smoke_var, np.nan),
                "se": ca_res.std_errors.get(smoke_var, np.nan),
                "n": int(ca_res.nobs),
            }

    # Print comparison
    print(f"\n  Spec 3 (TWFE + Controls), 30d Mean PM2.5")
    print(f"  {'Outcome':<25s} {'National (Tract)':>20s} {'CA (Tract)':>20s}")
    print("  " + "-" * 65)

    for dep_var, dep_label in outcomes:
        n = natl_results.get(dep_var, {})
        ca = ca_results.get(dep_var, {})
        n_str = f"{n.get('coef', np.nan):.5f}" if n else "N/A"
        ca_str = f"{ca.get('coef', np.nan):.5f}" if ca else "N/A"
        print(f"  {dep_label:<25s} {n_str:>20s} {ca_str:>20s}")

    # Plot comparison
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    y_positions = np.arange(len(outcomes))
    offset = 0.15

    for dep_idx, (dep_var, dep_label) in enumerate(outcomes):
        n = natl_results.get(dep_var, {})
        ca = ca_results.get(dep_var, {})

        if n:
            ax.errorbar(n["coef"], dep_idx - offset,
                        xerr=1.96 * n["se"],
                        fmt="o", color="#2166ac", markersize=8, capsize=4,
                        label="National (tract)" if dep_idx == 0 else "")
        if ca:
            ax.errorbar(ca["coef"], dep_idx + offset,
                        xerr=1.96 * ca["se"],
                        fmt="s", color="#b2182b", markersize=8, capsize=4,
                        label="CA (tract)" if dep_idx == 0 else "")

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in outcomes])
    ax.set_xlabel("Coefficient on Smoke PM2.5 (30d mean)")
    ax.set_title("National vs. CA Tract-Level Estimates", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, "tract_comparison_ca.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("National Tract-Level Analysis: Wildfire Smoke and Voting Behavior")
    print("=" * 70)

    os.makedirs(FIG_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print(f"\nERROR: Analysis data not found: {DATA_FILE}")
        print("Run tract_build_smoke_analysis.py first.")
        return

    df = load_data(DATA_FILE, id_col="geoid")

    # Summary statistics
    summary_statistics(df)

    # Build-up specification table
    create_buildup_table(df)

    # Temporal dynamics with controls
    temporal_dynamics_controls(df)

    # Close-in daily temporal dynamics (1-7 days)
    temporal_closein_controls(df)

    # Threshold comparison
    threshold_comparison(df)

    # State x Year FE
    state_year_fe_regressions(df)

    # Controls robustness
    robustness_controls(df)

    # Heterogeneity tests
    heterogeneity_tests(df)

    # County-level comparison
    county_level_comparison(df)

    # CA comparison
    ca_comparison(df)

    print(f"\n{'='*70}")
    print(f"National tract analysis complete. Figures saved to: {FIG_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

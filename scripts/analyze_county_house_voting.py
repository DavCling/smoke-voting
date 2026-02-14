#!/usr/bin/env python3
"""
Analyze the relationship between wildfire smoke and House election voting behavior
at the county level (using precinct-aggregated data).

Specifications:
  A: DEM vote share ~ smoke (pro-environment shift)
  B: Incumbent punishment (president's party vote share ~ smoke)
  C: Turnout effects (log total votes ~ smoke)

Plus robustness checks and comparison with district-level and presidential results.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_county_house_analysis.parquet")
DISTRICT_DATA_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_house_analysis.parquet")
PRES_DATA_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_analysis.parquet")


def load_data():
    """Load county-level House analysis dataset and prepare panel structure."""
    print("Loading county-level House analysis dataset...")
    df = pd.read_parquet(DATA_FILE)
    print(f"  {len(df):,} observations, {df['fips'].nunique():,} counties, "
          f"{df['year'].nunique()} elections")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Uncontested: {df['uncontested'].sum():,}")

    df = df.set_index(["fips", "year"])
    df = df.sort_index()
    return df


def run_twfe(df, dep_var, smoke_var, controls=None, absorb_entity=True, absorb_time=True,
             state_year_fe=False, label=""):
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
                           other_effects=other_ef, check_rank=False)
        else:
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


def spec_a_dem_vote_share(df):
    """Specification A: Smoke -> DEM vote share."""
    print("\n" + "=" * 70)
    print("SPECIFICATION A: Smoke Exposure → Democratic Vote Share (County-Level House)")
    print("=" * 70)

    df_cont = df[~df["uncontested"]].copy()
    print(f"  Using {len(df_cont):,} contested county-year observations")

    results = []
    smoke_vars = [
        ("smoke_pm25_mean_60d", "Mean smoke PM2.5 (60 days)"),
        ("smoke_days_60d", "Smoke days (60 days)"),
        ("smoke_days_severe_60d", "Severe smoke days (60 days)"),
        ("smoke_pm25_mean_7d", "Mean smoke PM2.5 (7 days)"),
        ("smoke_pm25_mean_season", "Mean smoke PM2.5 (fire season)"),
    ]

    for svar, label in smoke_vars:
        if svar not in df_cont.columns:
            continue
        res = run_twfe(df_cont, "dem_vote_share", svar, label=label)
        r = print_result(res, label, svar)
        if r:
            r["smoke_var"] = svar
            results.append(r)

    # With lagged vote share control
    if "dem_vote_share_lag" in df_cont.columns:
        print("\n  --- With lagged DEM vote share control ---")
        res = run_twfe(df_cont, "dem_vote_share", "smoke_pm25_mean_60d",
                       controls=["dem_vote_share_lag"],
                       label="Mean PM2.5 (60d) + lag control")
        r = print_result(res, "Mean PM2.5 (60d) + lag control", "smoke_pm25_mean_60d")
        if r:
            r["smoke_var"] = "smoke_pm25_mean_60d"
            results.append(r)

    return results


def spec_b_incumbent_punishment(df):
    """Specification B: Smoke -> Incumbent (president's party) vote share."""
    print("\n" + "=" * 70)
    print("SPECIFICATION B: Smoke Exposure → Incumbent Party Vote Share (County-Level House)")
    print("=" * 70)

    df_cont = df[~df["uncontested"]].copy()
    print(f"  Using {len(df_cont):,} contested county-year observations")

    results = []
    smoke_vars = [
        ("smoke_pm25_mean_60d", "Mean smoke PM2.5 (60 days)"),
        ("smoke_days_60d", "Smoke days (60 days)"),
        ("smoke_days_severe_60d", "Severe smoke days (60 days)"),
        ("smoke_pm25_mean_7d", "Mean smoke PM2.5 (7 days)"),
    ]

    for svar, label in smoke_vars:
        if svar not in df_cont.columns:
            continue
        res = run_twfe(df_cont, "incumbent_vote_share", svar, label=label)
        r = print_result(res, label, svar)
        if r:
            r["smoke_var"] = svar
            results.append(r)

    return results


def spec_c_turnout(df):
    """Specification C: Smoke -> Turnout (includes uncontested)."""
    print("\n" + "=" * 70)
    print("SPECIFICATION C: Smoke Exposure → Voter Turnout (County-Level House)")
    print("=" * 70)

    print(f"  Using all {len(df):,} county-year observations (incl. uncontested)")

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


def create_summary_table(df):
    """Create a formatted summary results table."""
    print("\n" + "=" * 70)
    print("COUNTY-LEVEL HOUSE SUMMARY RESULTS TABLE")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"

    df_cont = df[~df["uncontested"]].copy()

    specs = [
        (df_cont, "dem_vote_share", "DEM vote share", "Spec A: Pro-environment"),
        (df_cont, "incumbent_vote_share", "Incumbent vote share", "Spec B: Incumbent punishment"),
        (df, "log_total_votes", "Log total votes", "Spec C: Turnout"),
    ]

    rows = []
    for data, dep, dep_label, spec_label in specs:
        res = run_twfe(data, dep, smoke_var, label=spec_label)
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


def frac_unhealthy_regressions(df):
    """Alternative treatment: fraction of days exceeding EPA unhealthy threshold."""
    print("\n" + "=" * 70)
    print("ALTERNATIVE TREATMENT: Fraction Unhealthy Days (County-Level House)")
    print("  (Fraction of days with smoke PM2.5 > 55.5 µg/m³)")
    print("=" * 70)

    smoke_var = "smoke_frac_unhealthy_30d"
    if smoke_var not in df.columns:
        print(f"  WARNING: {smoke_var} not found in dataset")
        return

    vals = df[smoke_var].dropna()
    print(f"  Distribution: mean={vals.mean():.4f}, median={vals.median():.4f}, "
          f"max={vals.max():.4f}, >0: {(vals > 0).sum():,}/{len(vals):,}")

    df_cont = df[~df["uncontested"]].copy()

    specs = [
        (df_cont, "dem_vote_share", "DEM vote share"),
        (df_cont, "incumbent_vote_share", "Incumbent vote share"),
        (df, "log_total_votes", "Log total votes"),
    ]

    for data, dep_var, dep_label in specs:
        res = run_twfe(data, dep_var, smoke_var,
                       label=f"Frac unhealthy: {dep_label}")
        print_result(res, f"Frac unhealthy: {dep_label}", smoke_var)


def state_year_fe_regressions(df):
    """Robustness check: State×Year FE instead of Year FE."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS: State×Year Fixed Effects (County-Level House)")
    print("  (Absorbs state-level time-varying shocks)")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"
    df_cont = df[~df["uncontested"]].copy()

    specs = [
        (df_cont, "dem_vote_share", "DEM vote share"),
        (df_cont, "incumbent_vote_share", "Incumbent vote share"),
        (df, "log_total_votes", "Log total votes"),
    ]

    for data, dep_var, dep_label in specs:
        res = run_twfe(data, dep_var, smoke_var, state_year_fe=True,
                       label=f"State×Year FE: {dep_label}")
        print_result(res, f"State×Year FE: {dep_label}", smoke_var)


def comparison_three_way(df):
    """Compare county-level House vs. district-level House vs. presidential coefficients."""
    print("\n" + "=" * 70)
    print("THREE-WAY COMPARISON: County House vs. District House vs. Presidential")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"

    # --- County-level House ---
    df_cont = df[~df["uncontested"]].copy()

    county_house = {}
    for dep_var, dep_label in [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
    ]:
        data = df_cont if dep_var != "log_total_votes" else df
        res = run_twfe(data, dep_var, smoke_var, label=f"County House: {dep_label}")
        if res:
            county_house[dep_label] = {
                "coef": res.params.get(smoke_var, np.nan),
                "se": res.std_errors.get(smoke_var, np.nan),
                "pval": res.pvalues.get(smoke_var, np.nan),
                "n": int(res.nobs),
            }

    # --- District-level House ---
    district_house = {}
    if os.path.exists(DISTRICT_DATA_FILE):
        dist_df = pd.read_parquet(DISTRICT_DATA_FILE)
        dist_df = dist_df.set_index(["district_id", "year"]).sort_index()
        dist_cont = dist_df[~dist_df["uncontested"]].copy()

        for dep_var, dep_label in [
            ("dem_vote_share", "DEM vote share"),
            ("incumbent_vote_share", "Incumbent vote share"),
            ("log_total_votes", "Log total votes"),
        ]:
            data = dist_cont if dep_var != "log_total_votes" else dist_df
            res = run_twfe(data, dep_var, smoke_var, label=f"District House: {dep_label}")
            if res:
                district_house[dep_label] = {
                    "coef": res.params.get(smoke_var, np.nan),
                    "se": res.std_errors.get(smoke_var, np.nan),
                    "pval": res.pvalues.get(smoke_var, np.nan),
                    "n": int(res.nobs),
                }

    # --- Presidential ---
    pres_results = {}
    if os.path.exists(PRES_DATA_FILE):
        pres_df = pd.read_parquet(PRES_DATA_FILE)
        pres_df = pres_df.set_index(["fips", "year"]).sort_index()

        for dep_var, dep_label in [
            ("dem_vote_share", "DEM vote share"),
            ("incumbent_vote_share", "Incumbent vote share"),
            ("log_total_votes", "Log total votes"),
        ]:
            res = run_twfe(pres_df, dep_var, smoke_var, label=f"Pres: {dep_label}")
            if res:
                pres_results[dep_label] = {
                    "coef": res.params.get(smoke_var, np.nan),
                    "se": res.std_errors.get(smoke_var, np.nan),
                    "pval": res.pvalues.get(smoke_var, np.nan),
                    "n": int(res.nobs),
                }

    # Print comparison table
    header = (f"  {'Outcome':<25s} "
              f"{'Cty House β':>13s} {'p':>8s} {'N':>7s}   "
              f"{'Dist House β':>13s} {'p':>8s} {'N':>7s}   "
              f"{'Pres β':>10s} {'p':>8s} {'N':>7s}")
    print(f"\n{header}")
    print("  " + "-" * 110)

    for dep_label in ["DEM vote share", "Incumbent vote share", "Log total votes"]:
        ch = county_house.get(dep_label, {})
        dh = district_house.get(dep_label, {})
        pr = pres_results.get(dep_label, {})

        def fmt(d):
            coef = d.get("coef", np.nan)
            pval = d.get("pval", np.nan)
            n = d.get("n", 0)
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            return f"{coef:>10.6f}{stars:<3s}", f"{pval:>8.4f}", f"{n:>7,}"

        ch_c, ch_p, ch_n = fmt(ch)
        dh_c, dh_p, dh_n = fmt(dh)
        pr_c, pr_p, pr_n = fmt(pr)

        print(f"  {dep_label:<25s} {ch_c} {ch_p} {ch_n}   {dh_c} {dh_p} {dh_n}   {pr_c} {pr_p} {pr_n}")


def robustness_drop_uncontested(df):
    """Show results are not sensitive to including/excluding uncontested counties."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS: Including vs. Excluding Uncontested Counties")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"

    df_all = df.copy()
    df_cont = df[~df["uncontested"]].copy()

    print(f"\n  All counties: {len(df_all):,} | Contested only: {len(df_cont):,}")

    specs = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
        ("log_total_votes", "Log total votes"),
    ]

    header = f"  {'Outcome':<25s} {'All β':>12s} {'All p':>10s} {'Contested β':>14s} {'Contested p':>12s}"
    print(f"\n{header}")
    print("  " + "-" * 73)

    for dep_var, dep_label in specs:
        res_all = run_twfe(df_all, dep_var, smoke_var, label=f"All: {dep_label}")
        res_cont = run_twfe(df_cont, dep_var, smoke_var, label=f"Contested: {dep_label}")

        coef_a = res_all.params.get(smoke_var, np.nan) if res_all else np.nan
        pval_a = res_all.pvalues.get(smoke_var, np.nan) if res_all else np.nan
        coef_c = res_cont.params.get(smoke_var, np.nan) if res_cont else np.nan
        pval_c = res_cont.pvalues.get(smoke_var, np.nan) if res_cont else np.nan

        stars_a = "***" if pval_a < 0.01 else "**" if pval_a < 0.05 else "*" if pval_a < 0.10 else ""
        stars_c = "***" if pval_c < 0.01 else "**" if pval_c < 0.05 else "*" if pval_c < 0.10 else ""

        print(f"  {dep_label:<25s} {coef_a:>10.6f}{stars_a:<2s} {pval_a:>10.4f} {coef_c:>12.6f}{stars_c:<2s} {pval_c:>12.4f}")


def robustness_controls(df):
    """Robustness check: add time-varying county controls."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS: Time-Varying County Controls (County House)")
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
    # Use contested-only for vote share outcomes (consistent with summary table)
    df_cont = df[~df["uncontested"]].copy()
    specs = [
        ("dem_vote_share", "DEM vote share", df_cont),
        ("incumbent_vote_share", "Incumbent vote share", df_cont),
        ("log_total_votes", "Log total votes", df),
    ]

    print(f"\n  {'Outcome':<25} {'Baseline β':>12} {'+ Controls β':>14} "
          f"{'Baseline p':>12} {'+ Controls p':>14} {'N base':>8} {'N ctrl':>8}")
    print("  " + "-" * 95)

    for dep_var, dep_label, data in specs:
        res_base = run_twfe(data, dep_var, smoke_var, label=f"Base: {dep_label}")
        res_ctrl = run_twfe(data, dep_var, smoke_var, controls=available,
                            label=f"+Controls: {dep_label}")

        b_coef = res_base.params.get(smoke_var, np.nan) if res_base else np.nan
        b_pval = res_base.pvalues.get(smoke_var, np.nan) if res_base else np.nan
        b_n = res_base.nobs if res_base else 0
        c_coef = res_ctrl.params.get(smoke_var, np.nan) if res_ctrl else np.nan
        c_pval = res_ctrl.pvalues.get(smoke_var, np.nan) if res_ctrl else np.nan
        c_n = res_ctrl.nobs if res_ctrl else 0

        print(f"  {dep_label:<25} {b_coef:>12.6f} {c_coef:>14.6f} "
              f"{b_pval:>12.4f} {c_pval:>14.4f} {b_n:>8,} {c_n:>8,}")

    # Full results for with-controls regressions
    print("\n  --- Full results with controls ---")
    for dep_var, dep_label, data in specs:
        res = run_twfe(data, dep_var, smoke_var, controls=available,
                       label=f"+Controls: {dep_label}")
        print_result(res, f"+Controls: {dep_label}", smoke_var)


def main():
    print("=" * 70)
    print("County-Level House Election Analysis: Wildfire Smoke and Voting")
    print("=" * 70)

    df = load_data()

    # Main specifications
    spec_a_dem_vote_share(df)
    spec_b_incumbent_punishment(df)
    spec_c_turnout(df)

    # Summary table
    create_summary_table(df)

    # Fraction unhealthy
    frac_unhealthy_regressions(df)

    # State×Year FE robustness
    state_year_fe_regressions(df)

    # Three-way comparison
    comparison_three_way(df)

    # Robustness
    print("\n\n" + "#" * 70)
    print("# ROBUSTNESS CHECKS")
    print("#" * 70)
    robustness_controls(df)
    robustness_drop_uncontested(df)

    print("\n" + "=" * 70)
    print("County-level House analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

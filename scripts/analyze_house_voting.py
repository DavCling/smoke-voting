#!/usr/bin/env python3
"""
Analyze the relationship between wildfire smoke and House election voting behavior.

Specifications:
  A: DEM vote share ~ smoke (pro-environment shift)
  B: Incumbent punishment (president's party vote share ~ smoke)
  C: Turnout effects (log total votes ~ smoke)

Plus robustness checks: drop uncontested, midterm vs. presidential split.
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
DATA_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_house_analysis.parquet")
PRES_DATA_FILE = os.path.join(BASE_DIR, "output", "smoke_voting_analysis.parquet")


def load_data():
    """Load House analysis dataset and prepare panel structure."""
    print("Loading House analysis dataset...")
    df = pd.read_parquet(DATA_FILE)
    print(f"  {len(df):,} observations, {df['district_id'].nunique():,} districts, "
          f"{df['year'].nunique()} elections")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Uncontested: {df['uncontested'].sum():,}")

    df = df.set_index(["district_id", "year"])
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
            # For district-level data, extract state from district_id (first 2 chars)
            entity_idx = subset.index.get_level_values(0).astype(str)
            state_codes = entity_idx.str[:2]
            state_year_cat = pd.Categorical(
                state_codes + "_" +
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
    print("SPECIFICATION A: Smoke Exposure → Democratic Vote Share (House)")
    print("=" * 70)

    # Drop uncontested races
    df_cont = df[~df["uncontested"]].copy()
    print(f"  Using {len(df_cont):,} contested district-year observations")

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
    print("SPECIFICATION B: Smoke Exposure → Incumbent Party Vote Share (House)")
    print("=" * 70)

    df_cont = df[~df["uncontested"]].copy()
    print(f"  Using {len(df_cont):,} contested district-year observations")

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
    print("SPECIFICATION C: Smoke Exposure → Voter Turnout (House)")
    print("=" * 70)

    print(f"  Using all {len(df):,} district-year observations (incl. uncontested)")

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
    """Create a formatted summary results table for House regressions."""
    print("\n" + "=" * 70)
    print("HOUSE SUMMARY RESULTS TABLE")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"

    # Contested only for vote share specs
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
        print(f"  Fixed effects: District + Year")
        print(f"  Standard errors: Clustered by district")
        print(f"\n{tbl.to_string(index=False)}")

    return rows


def state_year_fe_regressions(df):
    """Robustness check: State×Year FE instead of Year FE."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS: State×Year Fixed Effects (District-Level House)")
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


def comparison_with_presidential(df):
    """Compare House vs. presidential-level coefficients."""
    print("\n" + "=" * 70)
    print("COMPARISON: House vs. Presidential Regressions")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"

    # House results (contested only for vote share)
    df_cont = df[~df["uncontested"]].copy()

    house_specs = [
        (df_cont, "dem_vote_share", "DEM vote share"),
        (df_cont, "incumbent_vote_share", "Incumbent vote share"),
        (df, "log_total_votes", "Log total votes"),
    ]

    house_results = {}
    for data, dep_var, dep_label in house_specs:
        res = run_twfe(data, dep_var, smoke_var, label=f"House: {dep_label}")
        if res:
            house_results[dep_label] = {
                "coef": res.params.get(smoke_var, np.nan),
                "se": res.std_errors.get(smoke_var, np.nan),
                "pval": res.pvalues.get(smoke_var, np.nan),
                "n": int(res.nobs),
            }

    # Presidential results
    pres_results = {}
    if os.path.exists(PRES_DATA_FILE):
        pres_df = pd.read_parquet(PRES_DATA_FILE)
        pres_df = pres_df.set_index(["fips", "year"]).sort_index()

        pres_specs = [
            ("dem_vote_share", "DEM vote share"),
            ("incumbent_vote_share", "Incumbent vote share"),
            ("log_total_votes", "Log total votes"),
        ]

        for dep_var, dep_label in pres_specs:
            res = run_twfe(pres_df, dep_var, smoke_var, label=f"Pres: {dep_label}")
            if res:
                pres_results[dep_label] = {
                    "coef": res.params.get(smoke_var, np.nan),
                    "se": res.std_errors.get(smoke_var, np.nan),
                    "pval": res.pvalues.get(smoke_var, np.nan),
                    "n": int(res.nobs),
                }

    # Print comparison
    header = f"  {'Outcome':<25s} {'House β':>12s} {'House p':>10s} {'House N':>8s}   {'Pres β':>12s} {'Pres p':>10s} {'Pres N':>8s}"
    print(f"\n{header}")
    print("  " + "-" * 90)

    for dep_label in ["DEM vote share", "Incumbent vote share", "Log total votes"]:
        h = house_results.get(dep_label, {})
        p = pres_results.get(dep_label, {})

        h_coef = h.get("coef", np.nan)
        h_pval = h.get("pval", np.nan)
        h_n = h.get("n", 0)
        p_coef = p.get("coef", np.nan)
        p_pval = p.get("pval", np.nan)
        p_n = p.get("n", 0)

        h_stars = "***" if h_pval < 0.01 else "**" if h_pval < 0.05 else "*" if h_pval < 0.10 else ""
        p_stars = "***" if p_pval < 0.01 else "**" if p_pval < 0.05 else "*" if p_pval < 0.10 else ""

        print(f"  {dep_label:<25s} {h_coef:>10.6f}{h_stars:<2s} {h_pval:>10.4f} {h_n:>8,}   "
              f"{p_coef:>10.6f}{p_stars:<2s} {p_pval:>10.4f} {p_n:>8,}")


def robustness_drop_uncontested(df):
    """Show results are not sensitive to including/excluding uncontested races."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS: Including vs. Excluding Uncontested Races")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"

    df_all = df.copy()
    df_cont = df[~df["uncontested"]].copy()

    print(f"\n  All districts: {len(df_all):,} | Contested only: {len(df_cont):,}")

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


def robustness_midterm_vs_presidential(df):
    """Split sample: midterm years vs. presidential years."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS: Midterm vs. Presidential Election Years")
    print("=" * 70)

    smoke_var = "smoke_pm25_mean_30d"

    midterm_years = {2006, 2010, 2014, 2018}
    presidential_years = {2008, 2012, 2016, 2020}

    df_cont = df[~df["uncontested"]].copy()
    df_mid = df_cont[df_cont.index.get_level_values("year").isin(midterm_years)].copy()
    df_pres = df_cont[df_cont.index.get_level_values("year").isin(presidential_years)].copy()

    print(f"\n  Midterm: {len(df_mid):,} obs ({sorted(df_mid.index.get_level_values('year').unique())})")
    print(f"  Presidential: {len(df_pres):,} obs ({sorted(df_pres.index.get_level_values('year').unique())})")

    specs = [
        ("dem_vote_share", "DEM vote share"),
        ("incumbent_vote_share", "Incumbent vote share"),
    ]

    header = f"  {'Outcome':<25s} {'Midterm β':>12s} {'Mid p':>10s} {'Pres-yr β':>12s} {'Pres p':>10s}"
    print(f"\n{header}")
    print("  " + "-" * 69)

    for dep_var, dep_label in specs:
        res_mid = run_twfe(df_mid, dep_var, smoke_var, label=f"Midterm: {dep_label}")
        res_pres = run_twfe(df_pres, dep_var, smoke_var, label=f"Presidential: {dep_label}")

        coef_m = res_mid.params.get(smoke_var, np.nan) if res_mid else np.nan
        pval_m = res_mid.pvalues.get(smoke_var, np.nan) if res_mid else np.nan
        coef_p = res_pres.params.get(smoke_var, np.nan) if res_pres else np.nan
        pval_p = res_pres.pvalues.get(smoke_var, np.nan) if res_pres else np.nan

        stars_m = "***" if pval_m < 0.01 else "**" if pval_m < 0.05 else "*" if pval_m < 0.10 else ""
        stars_p = "***" if pval_p < 0.01 else "**" if pval_p < 0.05 else "*" if pval_p < 0.10 else ""

        print(f"  {dep_label:<25s} {coef_m:>10.6f}{stars_m:<2s} {pval_m:>10.4f} {coef_p:>10.6f}{stars_p:<2s} {pval_p:>10.4f}")


def main():
    print("=" * 70)
    print("House Election Analysis: Wildfire Smoke and Voting Behavior")
    print("=" * 70)

    df = load_data()

    # Main specifications
    spec_a_dem_vote_share(df)
    spec_b_incumbent_punishment(df)
    spec_c_turnout(df)

    # Summary table
    create_summary_table(df)

    # State×Year FE robustness
    state_year_fe_regressions(df)

    # Comparison with presidential
    comparison_with_presidential(df)

    # Robustness checks
    print("\n\n" + "#" * 70)
    print("# ROBUSTNESS CHECKS")
    print("#" * 70)
    robustness_drop_uncontested(df)
    robustness_midterm_vs_presidential(df)

    print("\n" + "=" * 70)
    print("House analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

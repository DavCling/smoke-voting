#!/usr/bin/env Rscript
#
# Cross-check: Replicate TWFE regressions in R using fixest.
#
# Compares R results against Python (linearmodels PanelOLS) to verify
# that coefficients match to ~4 decimal places.
#
# Covers: District House, County House, Presidential
# Specs: 30d base, state×year FE, fraction unhealthy
#
# Dependencies: fixest, arrow, data.table

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(data.table)
})

cat("=" , rep("=", 69), "\n", sep = "")
cat("R Cross-Check: TWFE Regressions (fixest)\n")
cat("=" , rep("=", 69), "\n\n", sep = "")

# ---- Helper: print regression result ----
print_reg <- function(label, res, var) {
  cat(sprintf("%s\n  β = %.6f  (SE = %.6f)  p = %.4f  N = %d\n\n",
              label,
              coef(res)[var],
              se(res)[var],
              pvalue(res)[var],
              nobs(res)))
}

# ============================================================
# DISTRICT-LEVEL HOUSE
# ============================================================
house_file <- file.path(dirname(dirname(tryCatch(sys.frame(1)$ofile, error = function(e) "."))), "output", "smoke_voting_house_analysis.parquet")
if (!file.exists(house_file)) {
  house_file <- "output/smoke_voting_house_analysis.parquet"
}

cat("Loading House data:", house_file, "\n")
house <- as.data.table(read_parquet(house_file))

cat(sprintf("  %d observations, %d districts, %d years\n",
            nrow(house), uniqueN(house$district_id), uniqueN(house$year)))
cat(sprintf("  Years: %s\n", paste(sort(unique(house$year)), collapse = ", ")))
cat(sprintf("  Uncontested: %d\n\n", sum(house$uncontested)))

# Extract state from district_id for state×year FE
house[, state_fips_2 := substr(district_id, 1, 2)]

contested <- house[uncontested == FALSE]

# ---- House TWFE regressions (30d base spec) ----
cat("--- House Regressions: 30d Base Spec (R / fixest) ---\n\n")

cat("Spec A: DEM vote share ~ smoke_pm25_mean_30d | district_id + year\n")
res_a <- feols(dem_vote_share ~ smoke_pm25_mean_30d | district_id + year,
               data = contested, cluster = ~district_id)
print_reg("", res_a, "smoke_pm25_mean_30d")

cat("Spec B: incumbent_vote_share ~ smoke_pm25_mean_30d | district_id + year\n")
res_b <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d | district_id + year,
               data = contested, cluster = ~district_id)
print_reg("", res_b, "smoke_pm25_mean_30d")

cat("Spec C: log_total_votes ~ smoke_pm25_mean_30d | district_id + year\n")
res_c <- feols(log_total_votes ~ smoke_pm25_mean_30d | district_id + year,
               data = house, cluster = ~district_id)
print_reg("", res_c, "smoke_pm25_mean_30d")

if ("turnout_rate" %in% names(house)) {
  cat("Spec D: turnout_rate ~ smoke_pm25_mean_30d | district_id + year\n")
  res_d <- feols(turnout_rate ~ smoke_pm25_mean_30d | district_id + year,
                 data = house, cluster = ~district_id)
  print_reg("", res_d, "smoke_pm25_mean_30d")
} else {
  cat("Spec D: turnout_rate not available in House data (no VAP at district level)\n\n")
}

# ---- House State×Year FE ----
cat("--- House: State×Year FE ---\n\n")

cat("State×Year FE A: dem_vote_share ~ smoke_pm25_mean_30d | district_id + state_fips_2^year\n")
res_a_sy <- feols(dem_vote_share ~ smoke_pm25_mean_30d | district_id + state_fips_2^year,
                  data = contested, cluster = ~district_id)
print_reg("", res_a_sy, "smoke_pm25_mean_30d")

cat("State×Year FE B: incumbent_vote_share ~ smoke_pm25_mean_30d | district_id + state_fips_2^year\n")
res_b_sy <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d | district_id + state_fips_2^year,
                  data = contested, cluster = ~district_id)
print_reg("", res_b_sy, "smoke_pm25_mean_30d")

cat("State×Year FE C: log_total_votes ~ smoke_pm25_mean_30d | district_id + state_fips_2^year\n")
res_c_sy <- feols(log_total_votes ~ smoke_pm25_mean_30d | district_id + state_fips_2^year,
                  data = house, cluster = ~district_id)
print_reg("", res_c_sy, "smoke_pm25_mean_30d")

if ("turnout_rate" %in% names(house)) {
  cat("State×Year FE D: turnout_rate ~ smoke_pm25_mean_30d | district_id + state_fips_2^year\n")
  res_d_sy <- feols(turnout_rate ~ smoke_pm25_mean_30d | district_id + state_fips_2^year,
                    data = house, cluster = ~district_id)
  print_reg("", res_d_sy, "smoke_pm25_mean_30d")
}

# ---- House Fraction Unhealthy ----
cat("--- House: Fraction Unhealthy ---\n\n")

cat("Frac Unhealthy A: dem_vote_share ~ smoke_frac_unhealthy_30d | district_id + year\n")
res_a_fu <- feols(dem_vote_share ~ smoke_frac_unhealthy_30d | district_id + year,
                  data = contested, cluster = ~district_id)
print_reg("", res_a_fu, "smoke_frac_unhealthy_30d")

cat("Frac Unhealthy C: log_total_votes ~ smoke_frac_unhealthy_30d | district_id + year\n")
res_c_fu <- feols(log_total_votes ~ smoke_frac_unhealthy_30d | district_id + year,
                  data = house, cluster = ~district_id)
print_reg("", res_c_fu, "smoke_frac_unhealthy_30d")

if ("turnout_rate" %in% names(house)) {
  cat("Frac Unhealthy D: turnout_rate ~ smoke_frac_unhealthy_30d | district_id + year\n")
  res_d_fu <- feols(turnout_rate ~ smoke_frac_unhealthy_30d | district_id + year,
                    data = house, cluster = ~district_id)
  print_reg("", res_d_fu, "smoke_frac_unhealthy_30d")
}


# ============================================================
# PRESIDENTIAL
# ============================================================
pres_file <- file.path(dirname(dirname(tryCatch(sys.frame(1)$ofile, error = function(e) "."))), "output", "smoke_voting_analysis.parquet")
if (!file.exists(pres_file)) {
  pres_file <- "output/smoke_voting_analysis.parquet"
}

if (file.exists(pres_file)) {
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("--- Presidential Regressions (R / fixest) ---\n\n")
  pres <- as.data.table(read_parquet(pres_file))

  cat(sprintf("  %d observations, %d counties, %d years\n",
              nrow(pres), uniqueN(pres$fips), uniqueN(pres$year)))
  cat(sprintf("  Years: %s\n\n", paste(sort(unique(pres$year)), collapse = ", ")))

  # ---- Build-up Spec 1: Raw OLS (no FE) ----
  cat("--- Presidential: Build-Up Spec (1) Raw OLS ---\n\n")

  cat("Pres Raw OLS A: dem_vote_share ~ smoke_pm25_mean_30d\n")
  pres_a_raw <- feols(dem_vote_share ~ smoke_pm25_mean_30d,
                      data = pres, cluster = ~fips)
  print_reg("", pres_a_raw, "smoke_pm25_mean_30d")

  cat("Pres Raw OLS B: incumbent_vote_share ~ smoke_pm25_mean_30d\n")
  pres_b_raw <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d,
                      data = pres, cluster = ~fips)
  print_reg("", pres_b_raw, "smoke_pm25_mean_30d")

  cat("Pres Raw OLS C: log_total_votes ~ smoke_pm25_mean_30d\n")
  pres_c_raw <- feols(log_total_votes ~ smoke_pm25_mean_30d,
                      data = pres, cluster = ~fips)
  print_reg("", pres_c_raw, "smoke_pm25_mean_30d")

  cat("Pres Raw OLS D: turnout_rate ~ smoke_pm25_mean_30d\n")
  pres_d_raw <- feols(turnout_rate ~ smoke_pm25_mean_30d,
                      data = pres, cluster = ~fips)
  print_reg("", pres_d_raw, "smoke_pm25_mean_30d")

  # ---- Build-up Spec 2: 30d base spec (County + Year FE) ----
  cat("--- Presidential: Build-Up Spec (2) TWFE ---\n\n")

  cat("Pres A: dem_vote_share ~ smoke_pm25_mean_30d | fips + year\n")
  pres_a <- feols(dem_vote_share ~ smoke_pm25_mean_30d | fips + year,
                  data = pres, cluster = ~fips)
  print_reg("", pres_a, "smoke_pm25_mean_30d")

  cat("Pres B: incumbent_vote_share ~ smoke_pm25_mean_30d | fips + year\n")
  pres_b <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d | fips + year,
                  data = pres, cluster = ~fips)
  print_reg("", pres_b, "smoke_pm25_mean_30d")

  cat("Pres C: log_total_votes ~ smoke_pm25_mean_30d | fips + year\n")
  pres_c <- feols(log_total_votes ~ smoke_pm25_mean_30d | fips + year,
                  data = pres, cluster = ~fips)
  print_reg("", pres_c, "smoke_pm25_mean_30d")

  cat("Pres D: turnout_rate ~ smoke_pm25_mean_30d | fips + year\n")
  pres_d <- feols(turnout_rate ~ smoke_pm25_mean_30d | fips + year,
                  data = pres, cluster = ~fips)
  print_reg("", pres_d, "smoke_pm25_mean_30d")

  # State×Year FE
  cat("--- Presidential: State×Year FE ---\n\n")

  cat("Pres State×Year A: dem_vote_share ~ smoke_pm25_mean_30d | fips + state_fips^year\n")
  pres_a_sy <- feols(dem_vote_share ~ smoke_pm25_mean_30d | fips + state_fips^year,
                     data = pres, cluster = ~fips)
  print_reg("", pres_a_sy, "smoke_pm25_mean_30d")

  cat("Pres State×Year B: incumbent_vote_share ~ smoke_pm25_mean_30d | fips + state_fips^year\n")
  pres_b_sy <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d | fips + state_fips^year,
                     data = pres, cluster = ~fips)
  print_reg("", pres_b_sy, "smoke_pm25_mean_30d")

  cat("Pres State×Year C: log_total_votes ~ smoke_pm25_mean_30d | fips + state_fips^year\n")
  pres_c_sy <- feols(log_total_votes ~ smoke_pm25_mean_30d | fips + state_fips^year,
                     data = pres, cluster = ~fips)
  print_reg("", pres_c_sy, "smoke_pm25_mean_30d")

  cat("Pres State×Year D: turnout_rate ~ smoke_pm25_mean_30d | fips + state_fips^year\n")
  pres_d_sy <- feols(turnout_rate ~ smoke_pm25_mean_30d | fips + state_fips^year,
                     data = pres, cluster = ~fips)
  print_reg("", pres_d_sy, "smoke_pm25_mean_30d")

  # Fraction Unhealthy
  cat("--- Presidential: Fraction Unhealthy ---\n\n")

  cat("Pres Frac Unhealthy A: dem_vote_share ~ smoke_frac_unhealthy_30d | fips + year\n")
  pres_a_fu <- feols(dem_vote_share ~ smoke_frac_unhealthy_30d | fips + year,
                     data = pres, cluster = ~fips)
  print_reg("", pres_a_fu, "smoke_frac_unhealthy_30d")

  cat("Pres Frac Unhealthy C: log_total_votes ~ smoke_frac_unhealthy_30d | fips + year\n")
  pres_c_fu <- feols(log_total_votes ~ smoke_frac_unhealthy_30d | fips + year,
                     data = pres, cluster = ~fips)
  print_reg("", pres_c_fu, "smoke_frac_unhealthy_30d")

  cat("Pres Frac Unhealthy D: turnout_rate ~ smoke_frac_unhealthy_30d | fips + year\n")
  pres_d_fu <- feols(turnout_rate ~ smoke_frac_unhealthy_30d | fips + year,
                     data = pres, cluster = ~fips)
  print_reg("", pres_d_fu, "smoke_frac_unhealthy_30d")

  # ---- Presidential: With Controls ----
  ctrl_vars <- c("unemployment_rate", "log_median_income", "log_population",
                 "october_tmean", "october_ppt")
  has_controls <- all(ctrl_vars %in% names(pres)) && any(!is.na(pres$unemployment_rate))

  if (has_controls) {
    cat("--- Presidential: With Time-Varying Controls ---\n\n")

    pres_a_ctrl <- feols(dem_vote_share ~ smoke_pm25_mean_30d + unemployment_rate +
                         log_median_income + log_population + october_tmean + october_ppt | fips + year,
                         data = pres, cluster = ~fips)
    print_reg("Pres +Controls A: DEM vote share", pres_a_ctrl, "smoke_pm25_mean_30d")

    pres_b_ctrl <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d + unemployment_rate +
                         log_median_income + log_population + october_tmean + october_ppt | fips + year,
                         data = pres, cluster = ~fips)
    print_reg("Pres +Controls B: Incumbent vote share", pres_b_ctrl, "smoke_pm25_mean_30d")

    pres_c_ctrl <- feols(log_total_votes ~ smoke_pm25_mean_30d + unemployment_rate +
                         log_median_income + log_population + october_tmean + october_ppt | fips + year,
                         data = pres, cluster = ~fips)
    print_reg("Pres +Controls C: Log total votes", pres_c_ctrl, "smoke_pm25_mean_30d")

    pres_d_ctrl <- feols(turnout_rate ~ smoke_pm25_mean_30d + unemployment_rate +
                         log_median_income + log_population + october_tmean + october_ppt | fips + year,
                         data = pres, cluster = ~fips)
    print_reg("Pres +Controls D: Turnout rate", pres_d_ctrl, "smoke_pm25_mean_30d")

    # ---- Build-up Spec 4: TWFE + Controls + State Linear Trends ----
    cat("--- Presidential: Build-Up Spec (4) +Controls +State Trends ---\n\n")

    pres[, state_fips_2 := substr(fips, 1, 2)]

    cat("Pres +Trends A: dem_vote_share ~ smoke_pm25_mean_30d + controls | fips + year + state_fips_2[year]\n")
    pres_a_trend <- feols(dem_vote_share ~ smoke_pm25_mean_30d + unemployment_rate +
                          log_median_income + log_population + october_tmean + october_ppt | fips + year + state_fips_2[year],
                          data = pres, cluster = ~fips)
    print_reg("", pres_a_trend, "smoke_pm25_mean_30d")

    cat("Pres +Trends B: incumbent_vote_share ~ smoke_pm25_mean_30d + controls | fips + year + state_fips_2[year]\n")
    pres_b_trend <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d + unemployment_rate +
                          log_median_income + log_population + october_tmean + october_ppt | fips + year + state_fips_2[year],
                          data = pres, cluster = ~fips)
    print_reg("", pres_b_trend, "smoke_pm25_mean_30d")

    cat("Pres +Trends C: log_total_votes ~ smoke_pm25_mean_30d + controls | fips + year + state_fips_2[year]\n")
    pres_c_trend <- feols(log_total_votes ~ smoke_pm25_mean_30d + unemployment_rate +
                          log_median_income + log_population + october_tmean + october_ppt | fips + year + state_fips_2[year],
                          data = pres, cluster = ~fips)
    print_reg("", pres_c_trend, "smoke_pm25_mean_30d")

    cat("Pres +Trends D: turnout_rate ~ smoke_pm25_mean_30d + controls | fips + year + state_fips_2[year]\n")
    pres_d_trend <- feols(turnout_rate ~ smoke_pm25_mean_30d + unemployment_rate +
                          log_median_income + log_population + october_tmean + october_ppt | fips + year + state_fips_2[year],
                          data = pres, cluster = ~fips)
    print_reg("", pres_d_trend, "smoke_pm25_mean_30d")
  } else {
    cat("--- Presidential: Controls not available, skipping ---\n\n")
  }

  # ---- Comparison table ----
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("COMPARISON TABLE: House vs. Presidential — 30d (R / fixest)\n")
  cat(rep("=", 70), "\n\n", sep = "")

  fmt <- "  %-25s %12.6f %10.4f %8d   %12.6f %10.4f %8d\n"
  cat(sprintf("  %-25s %12s %10s %8s   %12s %10s %8s\n",
              "Outcome", "House β", "House p", "House N",
              "Pres β", "Pres p", "Pres N"))
  cat("  ", rep("-", 90), "\n", sep = "")

  cat(sprintf(fmt, "DEM vote share",
              coef(res_a)["smoke_pm25_mean_30d"],
              pvalue(res_a)["smoke_pm25_mean_30d"],
              nobs(res_a),
              coef(pres_a)["smoke_pm25_mean_30d"],
              pvalue(pres_a)["smoke_pm25_mean_30d"],
              nobs(pres_a)))

  cat(sprintf(fmt, "Incumbent vote share",
              coef(res_b)["smoke_pm25_mean_30d"],
              pvalue(res_b)["smoke_pm25_mean_30d"],
              nobs(res_b),
              coef(pres_b)["smoke_pm25_mean_30d"],
              pvalue(pres_b)["smoke_pm25_mean_30d"],
              nobs(pres_b)))

  cat(sprintf(fmt, "Log total votes",
              coef(res_c)["smoke_pm25_mean_30d"],
              pvalue(res_c)["smoke_pm25_mean_30d"],
              nobs(res_c),
              coef(pres_c)["smoke_pm25_mean_30d"],
              pvalue(pres_c)["smoke_pm25_mean_30d"],
              nobs(pres_c)))

  if (exists("res_d") && exists("pres_d")) {
    cat(sprintf(fmt, "Turnout rate",
                coef(res_d)["smoke_pm25_mean_30d"],
                pvalue(res_d)["smoke_pm25_mean_30d"],
                nobs(res_d),
                coef(pres_d)["smoke_pm25_mean_30d"],
                pvalue(pres_d)["smoke_pm25_mean_30d"],
                nobs(pres_d)))
  } else if (exists("pres_d")) {
    cat(sprintf("  %-25s %10s %10s %7s     %10.6f %10.4f %7d\n",
                "Turnout rate", "N/A", "N/A", "N/A",
                coef(pres_d)["smoke_pm25_mean_30d"],
                pvalue(pres_d)["smoke_pm25_mean_30d"],
                nobs(pres_d)))
  }
} else {
  cat("Presidential data not found, skipping comparison.\n")
}


# ============================================================
# COUNTY-LEVEL HOUSE
# ============================================================
county_house_file <- file.path(dirname(dirname(tryCatch(sys.frame(1)$ofile, error = function(e) "."))), "output", "smoke_voting_county_house_analysis.parquet")
if (!file.exists(county_house_file)) {
  county_house_file <- "output/smoke_voting_county_house_analysis.parquet"
}

if (file.exists(county_house_file)) {
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("--- County-Level House Regressions (R / fixest) ---\n\n")
  ch <- as.data.table(read_parquet(county_house_file))

  cat(sprintf("  %d observations, %d counties, %d years\n",
              nrow(ch), uniqueN(ch$fips), uniqueN(ch$year)))
  cat(sprintf("  Years: %s\n\n", paste(sort(unique(ch$year)), collapse = ", ")))

  ch_cont <- ch[uncontested == FALSE]

  # 30d base spec
  cat("County House A: dem_vote_share ~ smoke_pm25_mean_30d | fips + year\n")
  ch_a <- feols(dem_vote_share ~ smoke_pm25_mean_30d | fips + year,
                data = ch_cont, cluster = ~fips)
  print_reg("", ch_a, "smoke_pm25_mean_30d")

  cat("County House B: incumbent_vote_share ~ smoke_pm25_mean_30d | fips + year\n")
  ch_b <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d | fips + year,
                data = ch_cont, cluster = ~fips)
  print_reg("", ch_b, "smoke_pm25_mean_30d")

  cat("County House C: log_total_votes ~ smoke_pm25_mean_30d | fips + year\n")
  ch_c <- feols(log_total_votes ~ smoke_pm25_mean_30d | fips + year,
                data = ch, cluster = ~fips)
  print_reg("", ch_c, "smoke_pm25_mean_30d")

  if ("turnout_rate" %in% names(ch) && sum(!is.na(ch$turnout_rate)) > 0) {
    cat("County House D: turnout_rate ~ smoke_pm25_mean_30d | fips + year\n")
    ch_d <- feols(turnout_rate ~ smoke_pm25_mean_30d | fips + year,
                  data = ch, cluster = ~fips)
    print_reg("", ch_d, "smoke_pm25_mean_30d")
  } else {
    cat("County House D: turnout_rate not available\n\n")
  }

  # State×Year FE
  cat("--- County House: State×Year FE ---\n\n")

  cat("County House State×Year A: dem_vote_share ~ smoke_pm25_mean_30d | fips + state_fips^year\n")
  ch_a_sy <- feols(dem_vote_share ~ smoke_pm25_mean_30d | fips + state_fips^year,
                   data = ch_cont, cluster = ~fips)
  print_reg("", ch_a_sy, "smoke_pm25_mean_30d")

  cat("County House State×Year B: incumbent_vote_share ~ smoke_pm25_mean_30d | fips + state_fips^year\n")
  ch_b_sy <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d | fips + state_fips^year,
                   data = ch_cont, cluster = ~fips)
  print_reg("", ch_b_sy, "smoke_pm25_mean_30d")

  cat("County House State×Year C: log_total_votes ~ smoke_pm25_mean_30d | fips + state_fips^year\n")
  ch_c_sy <- feols(log_total_votes ~ smoke_pm25_mean_30d | fips + state_fips^year,
                   data = ch, cluster = ~fips)
  print_reg("", ch_c_sy, "smoke_pm25_mean_30d")

  if ("turnout_rate" %in% names(ch) && sum(!is.na(ch$turnout_rate)) > 0) {
    cat("County House State×Year D: turnout_rate ~ smoke_pm25_mean_30d | fips + state_fips^year\n")
    ch_d_sy <- feols(turnout_rate ~ smoke_pm25_mean_30d | fips + state_fips^year,
                     data = ch, cluster = ~fips)
    print_reg("", ch_d_sy, "smoke_pm25_mean_30d")
  }

  # Fraction Unhealthy
  cat("--- County House: Fraction Unhealthy ---\n\n")

  cat("County House Frac Unhealthy A: dem_vote_share ~ smoke_frac_unhealthy_30d | fips + year\n")
  ch_a_fu <- feols(dem_vote_share ~ smoke_frac_unhealthy_30d | fips + year,
                   data = ch_cont, cluster = ~fips)
  print_reg("", ch_a_fu, "smoke_frac_unhealthy_30d")

  cat("County House Frac Unhealthy C: log_total_votes ~ smoke_frac_unhealthy_30d | fips + year\n")
  ch_c_fu <- feols(log_total_votes ~ smoke_frac_unhealthy_30d | fips + year,
                   data = ch, cluster = ~fips)
  print_reg("", ch_c_fu, "smoke_frac_unhealthy_30d")

  if ("turnout_rate" %in% names(ch) && sum(!is.na(ch$turnout_rate)) > 0) {
    cat("County House Frac Unhealthy D: turnout_rate ~ smoke_frac_unhealthy_30d | fips + year\n")
    ch_d_fu <- feols(turnout_rate ~ smoke_frac_unhealthy_30d | fips + year,
                     data = ch, cluster = ~fips)
    print_reg("", ch_d_fu, "smoke_frac_unhealthy_30d")
  }

  # ---- County House: With Controls ----
  ch_ctrl_vars <- c("unemployment_rate", "log_median_income", "log_population",
                    "october_tmean", "october_ppt")
  ch_has_controls <- all(ch_ctrl_vars %in% names(ch)) && any(!is.na(ch$unemployment_rate))

  if (ch_has_controls) {
    cat("--- County House: With Time-Varying Controls ---\n\n")

    ch_a_ctrl <- feols(dem_vote_share ~ smoke_pm25_mean_30d + unemployment_rate +
                       log_median_income + log_population + october_tmean + october_ppt | fips + year,
                       data = ch_cont, cluster = ~fips)
    print_reg("County House +Controls A: DEM vote share", ch_a_ctrl, "smoke_pm25_mean_30d")

    ch_b_ctrl <- feols(incumbent_vote_share ~ smoke_pm25_mean_30d + unemployment_rate +
                       log_median_income + log_population + october_tmean + october_ppt | fips + year,
                       data = ch_cont, cluster = ~fips)
    print_reg("County House +Controls B: Incumbent vote share", ch_b_ctrl, "smoke_pm25_mean_30d")

    ch_c_ctrl <- feols(log_total_votes ~ smoke_pm25_mean_30d + unemployment_rate +
                       log_median_income + log_population + october_tmean + october_ppt | fips + year,
                       data = ch, cluster = ~fips)
    print_reg("County House +Controls C: Log total votes", ch_c_ctrl, "smoke_pm25_mean_30d")

    if ("turnout_rate" %in% names(ch) && sum(!is.na(ch$turnout_rate)) > 0) {
      ch_d_ctrl <- feols(turnout_rate ~ smoke_pm25_mean_30d + unemployment_rate +
                         log_median_income + log_population + october_tmean + october_ppt | fips + year,
                         data = ch, cluster = ~fips)
      print_reg("County House +Controls D: Turnout rate", ch_d_ctrl, "smoke_pm25_mean_30d")
    }
  } else {
    cat("--- County House: Controls not available, skipping ---\n\n")
  }

} else {
  cat("County House data not found, skipping.\n")
}

cat("\n", rep("=", 70), "\n", sep = "")
cat("R cross-check complete.\n")
cat(rep("=", 70), "\n", sep = "")

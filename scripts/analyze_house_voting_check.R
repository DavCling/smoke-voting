#!/usr/bin/env Rscript
#
# Cross-check: Replicate House TWFE regressions in R using fixest.
#
# Compares R results against Python (linearmodels PanelOLS) to verify
# that coefficients match to ~4 decimal places.
#
# Dependencies: fixest, arrow, data.table

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(data.table)
})

cat("=" , rep("=", 69), "\n", sep = "")
cat("R Cross-Check: House TWFE Regressions (fixest)\n")
cat("=" , rep("=", 69), "\n\n", sep = "")

# ---- Load House data ----
house_file <- file.path(dirname(dirname(tryCatch(sys.frame(1)$ofile, error = function(e) "."))), "output", "smoke_voting_house_analysis.parquet")
if (!file.exists(house_file)) {
  # Try relative path from working directory
  house_file <- "output/smoke_voting_house_analysis.parquet"
}

cat("Loading House data:", house_file, "\n")
house <- as.data.table(read_parquet(house_file))

cat(sprintf("  %d observations, %d districts, %d years\n",
            nrow(house), uniqueN(house$district_id), uniqueN(house$year)))
cat(sprintf("  Years: %s\n", paste(sort(unique(house$year)), collapse = ", ")))
cat(sprintf("  Uncontested: %d\n\n", sum(house$uncontested)))

# Contested subset for vote share regressions
contested <- house[uncontested == FALSE]

# ---- House TWFE regressions ----
cat("--- House Regressions (R / fixest) ---\n\n")

# Spec A: DEM vote share
cat("Spec A: DEM vote share ~ smoke_pm25_mean_60d | district_id + year\n")
res_a <- feols(dem_vote_share ~ smoke_pm25_mean_60d | district_id + year,
               data = contested, cluster = ~district_id)
cat(sprintf("  β = %.6f  (SE = %.6f)  p = %.4f  N = %d\n\n",
            coef(res_a)["smoke_pm25_mean_60d"],
            se(res_a)["smoke_pm25_mean_60d"],
            pvalue(res_a)["smoke_pm25_mean_60d"],
            nobs(res_a)))

# Spec B: Incumbent vote share
cat("Spec B: incumbent_vote_share ~ smoke_pm25_mean_60d | district_id + year\n")
res_b <- feols(incumbent_vote_share ~ smoke_pm25_mean_60d | district_id + year,
               data = contested, cluster = ~district_id)
cat(sprintf("  β = %.6f  (SE = %.6f)  p = %.4f  N = %d\n\n",
            coef(res_b)["smoke_pm25_mean_60d"],
            se(res_b)["smoke_pm25_mean_60d"],
            pvalue(res_b)["smoke_pm25_mean_60d"],
            nobs(res_b)))

# Spec C: Turnout (all districts, including uncontested)
cat("Spec C: log_total_votes ~ smoke_pm25_mean_60d | district_id + year\n")
res_c <- feols(log_total_votes ~ smoke_pm25_mean_60d | district_id + year,
               data = house, cluster = ~district_id)
cat(sprintf("  β = %.6f  (SE = %.6f)  p = %.4f  N = %d\n\n",
            coef(res_c)["smoke_pm25_mean_60d"],
            se(res_c)["smoke_pm25_mean_60d"],
            pvalue(res_c)["smoke_pm25_mean_60d"],
            nobs(res_c)))

# ---- Presidential regressions for comparison ----
pres_file <- file.path(dirname(dirname(tryCatch(sys.frame(1)$ofile, error = function(e) "."))), "output", "smoke_voting_analysis.parquet")
if (!file.exists(pres_file)) {
  pres_file <- "output/smoke_voting_analysis.parquet"
}

if (file.exists(pres_file)) {
  cat("--- Presidential Regressions (R / fixest) ---\n\n")
  pres <- as.data.table(read_parquet(pres_file))

  cat(sprintf("  %d observations, %d counties, %d years\n",
              nrow(pres), uniqueN(pres$fips), uniqueN(pres$year)))
  cat(sprintf("  Years: %s\n\n", paste(sort(unique(pres$year)), collapse = ", ")))

  # DEM vote share
  cat("Pres A: dem_vote_share ~ smoke_pm25_mean_60d | fips + year\n")
  pres_a <- feols(dem_vote_share ~ smoke_pm25_mean_60d | fips + year,
                  data = pres, cluster = ~fips)
  cat(sprintf("  β = %.6f  (SE = %.6f)  p = %.4f  N = %d\n\n",
              coef(pres_a)["smoke_pm25_mean_60d"],
              se(pres_a)["smoke_pm25_mean_60d"],
              pvalue(pres_a)["smoke_pm25_mean_60d"],
              nobs(pres_a)))

  # Incumbent vote share
  cat("Pres B: incumbent_vote_share ~ smoke_pm25_mean_60d | fips + year\n")
  pres_b <- feols(incumbent_vote_share ~ smoke_pm25_mean_60d | fips + year,
                  data = pres, cluster = ~fips)
  cat(sprintf("  β = %.6f  (SE = %.6f)  p = %.4f  N = %d\n\n",
              coef(pres_b)["smoke_pm25_mean_60d"],
              se(pres_b)["smoke_pm25_mean_60d"],
              pvalue(pres_b)["smoke_pm25_mean_60d"],
              nobs(pres_b)))

  # Turnout
  cat("Pres C: log_total_votes ~ smoke_pm25_mean_60d | fips + year\n")
  pres_c <- feols(log_total_votes ~ smoke_pm25_mean_60d | fips + year,
                  data = pres, cluster = ~fips)
  cat(sprintf("  β = %.6f  (SE = %.6f)  p = %.4f  N = %d\n\n",
              coef(pres_c)["smoke_pm25_mean_60d"],
              se(pres_c)["smoke_pm25_mean_60d"],
              pvalue(pres_c)["smoke_pm25_mean_60d"],
              nobs(pres_c)))

  # ---- Comparison table ----
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("COMPARISON TABLE: House vs. Presidential (R / fixest)\n")
  cat(rep("=", 70), "\n\n", sep = "")

  fmt <- "  %-25s %12.6f %10.4f %8d   %12.6f %10.4f %8d\n"
  cat(sprintf("  %-25s %12s %10s %8s   %12s %10s %8s\n",
              "Outcome", "House β", "House p", "House N",
              "Pres β", "Pres p", "Pres N"))
  cat("  ", rep("-", 90), "\n", sep = "")

  cat(sprintf(fmt, "DEM vote share",
              coef(res_a)["smoke_pm25_mean_60d"],
              pvalue(res_a)["smoke_pm25_mean_60d"],
              nobs(res_a),
              coef(pres_a)["smoke_pm25_mean_60d"],
              pvalue(pres_a)["smoke_pm25_mean_60d"],
              nobs(pres_a)))

  cat(sprintf(fmt, "Incumbent vote share",
              coef(res_b)["smoke_pm25_mean_60d"],
              pvalue(res_b)["smoke_pm25_mean_60d"],
              nobs(res_b),
              coef(pres_b)["smoke_pm25_mean_60d"],
              pvalue(pres_b)["smoke_pm25_mean_60d"],
              nobs(pres_b)))

  cat(sprintf(fmt, "Log total votes",
              coef(res_c)["smoke_pm25_mean_60d"],
              pvalue(res_c)["smoke_pm25_mean_60d"],
              nobs(res_c),
              coef(pres_c)["smoke_pm25_mean_60d"],
              pvalue(pres_c)["smoke_pm25_mean_60d"],
              nobs(pres_c)))
} else {
  cat("Presidential data not found, skipping comparison.\n")
}

cat("\n", rep("=", 70), "\n", sep = "")
cat("R cross-check complete.\n")
cat(rep("=", 70), "\n", sep = "")

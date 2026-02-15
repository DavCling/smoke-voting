#!/usr/bin/env Rscript
#
# Cross-check: Replicate CA tract-level TWFE regressions in R using fixest.
#
# Compares R results against Python (linearmodels PanelOLS) to verify
# that coefficients match to ~4 decimal places.
#
# Covers: Presidential and House (CA tract-level)
# Specs: 30d base, county×year FE, frac haze, +controls, build-up
#
# Dependencies: fixest, arrow, data.table

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(data.table)
})

cat("=", rep("=", 69), "\n", sep = "")
cat("R Cross-Check: CA Tract-Level TWFE (fixest)\n")
cat("=", rep("=", 69), "\n\n", sep = "")

# ---- Resolve base directory ----
get_base_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1]))
    return(dirname(dirname(script_path)))
  }
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(dirname(normalizePath(ofile))))
  return(getwd())
}
base_dir <- get_base_dir()

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
# PRESIDENTIAL (CA TRACT-LEVEL)
# ============================================================
pres_file <- file.path(base_dir, "output", "california",
                       "ca_smoke_voting_presidential.parquet")

if (file.exists(pres_file)) {
  cat("Loading presidential data:", pres_file, "\n")
  pres <- as.data.table(read_parquet(pres_file))

  cat(sprintf("  %d observations, %d tracts, %d years\n",
              nrow(pres), uniqueN(pres$geoid), uniqueN(pres$year)))
  cat(sprintf("  Years: %s\n\n", paste(sort(unique(pres$year)), collapse = ", ")))

  # Extract county from tract GEOID
  pres[, county_fips := substr(geoid, 1, 5)]

  # ---- Build-up table ----
  cat("--- Build-Up Table (Presidential, 30d) ---\n\n")

  # Spec 1: Raw OLS
  cat("(1) Raw OLS: dem_vote_share ~ smoke_pm25_mean_30d\n")
  res1 <- feols(dem_vote_share ~ smoke_pm25_mean_30d, data = pres)
  print_reg("", res1, "smoke_pm25_mean_30d")

  # Spec 2: TWFE (tract + year FE)
  cat("(2) TWFE: dem_vote_share ~ smoke_pm25_mean_30d | geoid + year\n")
  res2 <- feols(dem_vote_share ~ smoke_pm25_mean_30d | geoid + year,
                data = pres, cluster = ~geoid)
  print_reg("", res2, "smoke_pm25_mean_30d")

  # Spec 3: TWFE + controls
  ctrl_cols <- c("unemployment_rate", "log_median_income", "log_population",
                 "october_tmean", "october_ppt")
  avail_ctrls <- ctrl_cols[ctrl_cols %in% names(pres)]
  avail_ctrls <- avail_ctrls[sapply(avail_ctrls, function(x) sum(!is.na(pres[[x]])) > 0)]

  if (length(avail_ctrls) > 0) {
    ctrl_formula <- paste(avail_ctrls, collapse = " + ")
    fml <- as.formula(paste("dem_vote_share ~ smoke_pm25_mean_30d +",
                            ctrl_formula, "| geoid + year"))
    cat(sprintf("(3) +Controls: %s\n", deparse(fml)))
    res3 <- feols(fml, data = pres, cluster = ~geoid)
    print_reg("", res3, "smoke_pm25_mean_30d")
  }

  # ---- TWFE regressions (30d base spec, Spec 3) ----
  cat("--- Presidential TWFE (Spec 3, 30d) ---\n\n")

  outcomes <- list(
    list(var = "dem_vote_share", label = "DEM vote share"),
    list(var = "incumbent_vote_share", label = "Incumbent vote share"),
    list(var = "log_total_votes", label = "Log total votes")
  )

  for (out in outcomes) {
    if (out$var %in% names(pres)) {
      if (length(avail_ctrls) > 0) {
        fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d +",
                                ctrl_formula, "| geoid + year"))
      } else {
        fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d | geoid + year"))
      }
      cat(sprintf("%s: %s\n", out$label, deparse(fml)))
      res <- feols(fml, data = pres, cluster = ~geoid)
      print_reg("", res, "smoke_pm25_mean_30d")
    }
  }

  # ---- County×Year FE ----
  cat("--- County×Year FE ---\n\n")

  for (out in outcomes) {
    if (out$var %in% names(pres)) {
      fml <- as.formula(paste(out$var,
                              "~ smoke_pm25_mean_30d | geoid + county_fips^year"))
      cat(sprintf("County×Year FE: %s\n", out$label))
      res_cy <- feols(fml, data = pres, cluster = ~geoid)
      print_reg("", res_cy, "smoke_pm25_mean_30d")
    }
  }

  # ---- Frac haze threshold (30d) ----
  if ("smoke_frac_haze_30d" %in% names(pres)) {
    cat("--- Frac Haze (>20 µg/m³) 30d ---\n\n")
    for (out in outcomes) {
      if (out$var %in% names(pres)) {
        if (length(avail_ctrls) > 0) {
          fml <- as.formula(paste(out$var, "~ smoke_frac_haze_30d +",
                                  ctrl_formula, "| geoid + year"))
        } else {
          fml <- as.formula(paste(out$var, "~ smoke_frac_haze_30d | geoid + year"))
        }
        cat(sprintf("Frac haze: %s\n", out$label))
        res_h <- feols(fml, data = pres, cluster = ~geoid)
        print_reg("", res_h, "smoke_frac_haze_30d")
      }
    }
  }

  # ---- Cluster by county robustness ----
  cat("--- Cluster by County (robustness) ---\n\n")
  for (out in outcomes) {
    if (out$var %in% names(pres)) {
      if (length(avail_ctrls) > 0) {
        fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d +",
                                ctrl_formula, "| geoid + year"))
      } else {
        fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d | geoid + year"))
      }
      cat(sprintf("Cluster county: %s\n", out$label))
      res_cc <- feols(fml, data = pres, cluster = ~county_fips)
      print_reg("", res_cc, "smoke_pm25_mean_30d")
    }
  }

} else {
  cat("Presidential data not found:", pres_file, "\n\n")
}

# ============================================================
# HOUSE (CA TRACT-LEVEL)
# ============================================================
house_file <- file.path(base_dir, "output", "california",
                        "ca_smoke_voting_house.parquet")

if (file.exists(house_file)) {
  cat("\n", "=", rep("=", 69), "\n", sep = "")
  cat("HOUSE (CA Tract-Level)\n")
  cat("=", rep("=", 69), "\n\n", sep = "")

  house <- as.data.table(read_parquet(house_file))
  cat(sprintf("  %d observations, %d tracts, %d years\n",
              nrow(house), uniqueN(house$geoid), uniqueN(house$year)))

  house[, county_fips := substr(geoid, 1, 5)]

  # Filter contested for vote share outcomes
  if ("uncontested" %in% names(house)) {
    contested <- house[uncontested == FALSE]
    cat(sprintf("  Uncontested: %d, Contested: %d\n\n",
                sum(house$uncontested), nrow(contested)))
  } else {
    contested <- house
  }

  ctrl_cols <- c("unemployment_rate", "log_median_income", "log_population",
                 "october_tmean", "october_ppt")
  avail_ctrls <- ctrl_cols[ctrl_cols %in% names(house)]
  avail_ctrls <- avail_ctrls[sapply(avail_ctrls, function(x) sum(!is.na(house[[x]])) > 0)]

  cat("--- House TWFE (30d, Spec 3) ---\n\n")

  # DEM vote share (contested only)
  if (length(avail_ctrls) > 0) {
    ctrl_formula <- paste(avail_ctrls, collapse = " + ")
    fml_dem <- as.formula(paste("dem_vote_share ~ smoke_pm25_mean_30d +",
                                ctrl_formula, "| geoid + year"))
  } else {
    fml_dem <- as.formula("dem_vote_share ~ smoke_pm25_mean_30d | geoid + year")
  }
  cat("DEM vote share (contested):\n")
  res_h_dem <- feols(fml_dem, data = contested, cluster = ~geoid)
  print_reg("", res_h_dem, "smoke_pm25_mean_30d")

  # Incumbent vote share (contested only)
  if ("incumbent_vote_share" %in% names(contested)) {
    if (length(avail_ctrls) > 0) {
      fml_inc <- as.formula(paste("incumbent_vote_share ~ smoke_pm25_mean_30d +",
                                  ctrl_formula, "| geoid + year"))
    } else {
      fml_inc <- as.formula("incumbent_vote_share ~ smoke_pm25_mean_30d | geoid + year")
    }
    cat("Incumbent vote share (contested):\n")
    res_h_inc <- feols(fml_inc, data = contested, cluster = ~geoid)
    print_reg("", res_h_inc, "smoke_pm25_mean_30d")
  }

  # Log total votes (all races)
  if (length(avail_ctrls) > 0) {
    fml_turn <- as.formula(paste("log_total_votes ~ smoke_pm25_mean_30d +",
                                 ctrl_formula, "| geoid + year"))
  } else {
    fml_turn <- as.formula("log_total_votes ~ smoke_pm25_mean_30d | geoid + year")
  }
  cat("Log total votes (all):\n")
  res_h_turn <- feols(fml_turn, data = house, cluster = ~geoid)
  print_reg("", res_h_turn, "smoke_pm25_mean_30d")

} else {
  cat("House data not found:", house_file, "\n")
}

cat("\nR cross-check complete.\n")

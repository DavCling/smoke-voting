#!/usr/bin/env Rscript
#
# Cross-check: Replicate national tract-level TWFE regressions in R using fixest.
#
# Compares R results against Python (linearmodels PanelOLS) to verify
# that coefficients match to ~4 decimal places.
#
# Covers: Presidential (national tract-level)
# Specs: 30d base, build-up (specs 1-3), state x year FE, frac haze, +controls
#
# Dependencies: fixest, arrow, data.table

suppressPackageStartupMessages({
  library(arrow)
  library(fixest)
  library(data.table)
})

cat("=", rep("=", 69), "\n", sep = "")
cat("R Cross-Check: National Tract-Level TWFE (fixest)\n")
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
  cat(sprintf("%s\n  beta = %.6f  (SE = %.6f)  p = %.4f  N = %d\n\n",
              label,
              coef(res)[var],
              se(res)[var],
              pvalue(res)[var],
              nobs(res)))
}

# ============================================================
# PRESIDENTIAL (NATIONAL TRACT-LEVEL)
# ============================================================
pres_file <- file.path(base_dir, "output", "national_tracts",
                       "tract_smoke_voting_presidential.parquet")

if (!file.exists(pres_file)) {
  cat("Data not found:", pres_file, "\n")
  cat("Run tract_build_smoke_analysis.py first.\n")
  quit(save = "no")
}

cat("Loading data:", pres_file, "\n")
pres <- as.data.table(read_parquet(pres_file))

cat(sprintf("  %d observations, %d tracts, %d years\n",
            nrow(pres), uniqueN(pres$geoid), uniqueN(pres$year)))
cat(sprintf("  Years: %s\n\n", paste(sort(unique(pres$year)), collapse = ", ")))

# Extract state and county from tract GEOID
pres[, state_fips := substr(geoid, 1, 2)]
pres[, county_fips := substr(geoid, 1, 5)]

# Construct state_year interaction
pres[, state_year := paste0(state_fips, "_", year)]

# ---- Build-up table ----
cat("--- Build-Up Table (Presidential, 30d) ---\n\n")

# Spec 1: Raw OLS
cat("(1) Raw OLS: dem_vote_share ~ smoke_pm25_mean_30d\n")
res1 <- feols(dem_vote_share ~ smoke_pm25_mean_30d, data = pres)
print_reg("", res1, "smoke_pm25_mean_30d")

# Spec 2: TWFE (tract + year FE), cluster by county
cat("(2) TWFE: dem_vote_share ~ smoke_pm25_mean_30d | geoid + year [cluster: county]\n")
res2 <- feols(dem_vote_share ~ smoke_pm25_mean_30d | geoid + year,
              data = pres, cluster = ~county_fips)
print_reg("", res2, "smoke_pm25_mean_30d")

# Spec 3: TWFE + controls, cluster by county
ctrl_cols <- c("unemployment_rate", "log_median_income", "log_population",
               "october_tmean", "october_ppt")
avail_ctrls <- ctrl_cols[ctrl_cols %in% names(pres)]
avail_ctrls <- avail_ctrls[sapply(avail_ctrls, function(x) sum(!is.na(pres[[x]])) > 0)]

if (length(avail_ctrls) > 0) {
  ctrl_formula <- paste(avail_ctrls, collapse = " + ")
  fml <- as.formula(paste("dem_vote_share ~ smoke_pm25_mean_30d +",
                          ctrl_formula, "| geoid + year"))
  cat(sprintf("(3) +Controls [cluster: county]: %s\n", deparse(fml)))
  res3 <- feols(fml, data = pres, cluster = ~county_fips)
  print_reg("", res3, "smoke_pm25_mean_30d")
}

# ---- TWFE regressions (30d base spec, Spec 3, cluster by county) ----
cat("--- Presidential TWFE (Spec 3, 30d, cluster by county) ---\n\n")

outcomes <- list(
  list(var = "dem_vote_share", label = "DEM vote share"),
  list(var = "incumbent_vote_share", label = "Incumbent vote share"),
  list(var = "log_total_votes", label = "Log total votes"),
  list(var = "turnout_rate", label = "Turnout rate (votes/VAP)")
)

for (out in outcomes) {
  if (out$var %in% names(pres)) {
    if (length(avail_ctrls) > 0) {
      fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d +",
                              ctrl_formula, "| geoid + year"))
    } else {
      fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d | geoid + year"))
    }
    cat(sprintf("%s [cluster: county]: %s\n", out$label, deparse(fml)))
    res <- feols(fml, data = pres, cluster = ~county_fips)
    print_reg("", res, "smoke_pm25_mean_30d")
  }
}

# ---- State x Year FE (cluster by county) ----
cat("--- State x Year FE (cluster by county) ---\n\n")

for (out in outcomes) {
  if (out$var %in% names(pres)) {
    fml <- as.formula(paste(out$var,
                            "~ smoke_pm25_mean_30d | geoid + state_fips^year"))
    cat(sprintf("State x Year FE: %s\n", out$label))
    res_sy <- feols(fml, data = pres, cluster = ~county_fips)
    print_reg("", res_sy, "smoke_pm25_mean_30d")
  }
}

# ---- Frac haze threshold (30d, cluster by county) ----
if ("smoke_frac_haze_30d" %in% names(pres)) {
  cat("--- Frac Haze (>20 ug/m3) 30d (cluster by county) ---\n\n")
  for (out in outcomes) {
    if (out$var %in% names(pres)) {
      if (length(avail_ctrls) > 0) {
        fml <- as.formula(paste(out$var, "~ smoke_frac_haze_30d +",
                                ctrl_formula, "| geoid + year"))
      } else {
        fml <- as.formula(paste(out$var, "~ smoke_frac_haze_30d | geoid + year"))
      }
      cat(sprintf("Frac haze: %s\n", out$label))
      res_h <- feols(fml, data = pres, cluster = ~county_fips)
      print_reg("", res_h, "smoke_frac_haze_30d")
    }
  }
}

# ---- State x Year FE + controls (cluster by county) ----
cat("--- State x Year FE + Controls (cluster by county) ---\n\n")

for (out in outcomes) {
  if (out$var %in% names(pres)) {
    if (length(avail_ctrls) > 0) {
      fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d +",
                              ctrl_formula,
                              "| geoid + state_fips^year"))
    } else {
      fml <- as.formula(paste(out$var,
                              "~ smoke_pm25_mean_30d | geoid + state_fips^year"))
    }
    cat(sprintf("State x Year FE + Controls: %s\n", out$label))
    res_syc <- feols(fml, data = pres, cluster = ~county_fips)
    print_reg("", res_syc, "smoke_pm25_mean_30d")
  }
}

# ---- Cluster by tract (robustness — for comparison with county clustering) ----
cat("--- Cluster by Tract (robustness) ---\n\n")
for (out in outcomes) {
  if (out$var %in% names(pres)) {
    if (length(avail_ctrls) > 0) {
      fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d +",
                              ctrl_formula, "| geoid + year"))
    } else {
      fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d | geoid + year"))
    }
    cat(sprintf("Cluster tract: %s\n", out$label))
    res_ct <- feols(fml, data = pres, cluster = ~geoid)
    print_reg("", res_ct, "smoke_pm25_mean_30d")
  }
}

# ---- Conley Spatial HAC Standard Errors ----
cat("--- Conley Spatial HAC SEs (Spec 3, 30d) ---\n\n")

# Load tract centroids (computed from tigris, cached to CSV)
cent_cache <- file.path(base_dir, "data", "national_tracts", "controls",
                         "tract_centroids.csv")

if (!file.exists(cent_cache)) {
  cat("  Computing tract centroids via tigris (first run only)...\n")
  suppressPackageStartupMessages({
    library(sf)
    library(tigris)
  })
  options(tigris_use_cache = TRUE)

  conus_fips <- c("01","04","05","06","08","09","10","11","12","13",
                  "16","17","18","19","20","21","22","23","24","25",
                  "26","27","28","29","30","31","32","33","34","35",
                  "36","37","38","39","40","41","42","44","45","46",
                  "47","48","49","50","51","53","54","55","56")
  tracts_list <- lapply(conus_fips, function(st) {
    tracts(state = st, cb = TRUE, year = 2019)
  })
  all_tracts <- do.call(rbind, tracts_list)
  suppressWarnings(tract_cents <- st_centroid(all_tracts))
  coords <- st_coordinates(tract_cents)
  cent_out <- data.table(
    geoid = tract_cents$GEOID,
    latitude = coords[, 2],
    longitude = coords[, 1]
  )
  dir.create(dirname(cent_cache), recursive = TRUE, showWarnings = FALSE)
  fwrite(cent_out, cent_cache)
  cat(sprintf("  Saved %d tract centroids to %s\n", nrow(cent_out), cent_cache))
} else {
  cat("  Loading cached centroids:", cent_cache, "\n")
}

centroids <- fread(cent_cache, colClasses = c(geoid = "character"))
pres[centroids, on = "geoid", `:=`(latitude = i.latitude, longitude = i.longitude)]
n_matched <- sum(!is.na(pres$latitude))
cat(sprintf("  Matched centroids: %d / %d observations (%.1f%%)\n",
            n_matched, nrow(pres), 100 * n_matched / nrow(pres)))

# Subset to tracts with coordinates for fair comparison
pres_geo <- pres[!is.na(latitude)]
cat(sprintf("  Using %d observations with coordinates\n\n", nrow(pres_geo)))

# Compare SEs: County cluster vs Conley at 100, 200, 500 km
bandwidths <- c(100, 200, 500)

cat("  SE comparison (Spec 3: TWFE + Controls, smoke_pm25_mean_30d):\n\n")
cat(sprintf("  %-25s %10s %10s %10s %10s %10s\n",
            "Outcome", "County", "100km", "200km", "500km", "Tract"))
cat("  ", strrep("-", 75), "\n", sep = "")

for (out in outcomes) {
  if (!(out$var %in% names(pres_geo))) next
  if (length(avail_ctrls) > 0) {
    fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d +",
                            ctrl_formula, "| geoid + year"))
  } else {
    fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d | geoid + year"))
  }

  res_fit <- feols(fml, data = pres_geo)

  se_county <- se(summary(res_fit, cluster = ~county_fips))["smoke_pm25_mean_30d"]
  se_tract <- se(summary(res_fit, cluster = ~geoid))["smoke_pm25_mean_30d"]

  se_conley <- sapply(bandwidths, function(bw) {
    se(summary(res_fit,
               vcov = conley(cutoff = bw)))["smoke_pm25_mean_30d"]
  })

  cat(sprintf("  %-25s %10.6f %10.6f %10.6f %10.6f %10.6f\n",
              out$label, se_county, se_conley[1], se_conley[2], se_conley[3], se_tract))
}

cat("\n")

# Detailed results at 200km bandwidth
cat("  Detailed results (Conley 200km):\n\n")
for (out in outcomes) {
  if (!(out$var %in% names(pres_geo))) next
  if (length(avail_ctrls) > 0) {
    fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d +",
                            ctrl_formula, "| geoid + year"))
  } else {
    fml <- as.formula(paste(out$var, "~ smoke_pm25_mean_30d | geoid + year"))
  }

  res_fit <- feols(fml, data = pres_geo)
  s_con <- summary(res_fit, vcov = conley(cutoff = 200))
  cat(sprintf("  %s (Conley 200km):\n", out$label))
  cat(sprintf("    beta = %.6f  (SE = %.6f)  p = %.4f  N = %d\n\n",
              coef(s_con)["smoke_pm25_mean_30d"],
              se(s_con)["smoke_pm25_mean_30d"],
              pvalue(s_con)["smoke_pm25_mean_30d"],
              nobs(s_con)))
}

cat("  Note: Conley SEs use spherical (Haversine) distance.\n")
cat("  Bandwidth = cutoff in km for the spatial HAC kernel.\n\n")

cat("\nR cross-check complete.\n")

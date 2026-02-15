#!/usr/bin/env Rscript
# Download PRISM October weather data and extract tract-level averages for California.
#
# Same approach as national download_weather_controls.R, but extracts to
# CA census tract polygons instead of county polygons.
#
# Requires: install.packages(c("terra", "prism", "sf", "tigris"))
#
# Output: data/california/controls/tract_weather_october.csv
#   Columns: GEOID (11-digit tract), year, october_tmean, october_ppt

library(terra)
library(prism)
library(sf)
library(tigris)

# --- Paths ---
get_base_dir <- function() {
  # Try commandArgs (works with Rscript)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1]))
    return(dirname(dirname(script_path)))
  }
  # Try sys.frame (works when source()'d)
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(dirname(normalizePath(ofile))))
  # Fallback to working directory
  return(getwd())
}
base_dir <- get_base_dir()
prism_dir <- file.path(base_dir, "data", "california", "controls", "prism")
out_file <- file.path(base_dir, "data", "california", "controls",
                      "tract_weather_october.csv")

if (file.exists(out_file)) {
  cat("Already exists:", out_file, "\n")
  df <- read.csv(out_file)
  cat(sprintf("  %d rows, years %d-%d, %d tracts\n",
              nrow(df), min(df$year), max(df$year),
              length(unique(df$GEOID))))
  quit(save = "no")
}

dir.create(prism_dir, recursive = TRUE, showWarnings = FALSE)

# --- Set PRISM download directory ---
prism_set_dl_dir(prism_dir)
cat("PRISM download directory:", prism_dir, "\n")

# --- Download October monthly rasters (2006-2022) ---
years <- 2006:2022

cat("\nDownloading PRISM October mean temperature...\n")
get_prism_monthlys(type = "tmean", years = years, mon = 10, keepZip = FALSE)

cat("\nDownloading PRISM October total precipitation...\n")
get_prism_monthlys(type = "ppt", years = years, mon = 10, keepZip = FALSE)

# --- Load CA tract polygons ---
# Use 2019 vintage (2010 Census tracts, most stable for 2010-2019)
# For 2006-2009, these boundaries are approximate (2000 Census tracts differ slightly)
cat("\nLoading California tract polygons (2019 vintage)...\n")
options(tigris_use_cache = TRUE)
tracts <- tracts(state = "CA", cb = TRUE, year = 2019)

cat(sprintf("  %d tracts\n", nrow(tracts)))

# Convert to terra SpatVector for extraction
tracts_vect <- vect(tracts)

# --- Extract tract-level weather for each year ---
cat("\nExtracting tract averages...\n")

results <- data.frame()

for (yr in years) {
  cat(sprintf("  %d: ", yr))

  # Find PRISM raster paths for this year/month
  tmean_dirs <- prism_archive_subset("tmean", "monthly", years = yr, mon = 10,
                                     resolution = "4km")
  ppt_dirs <- prism_archive_subset("ppt", "monthly", years = yr, mon = 10,
                                   resolution = "4km")

  if (length(tmean_dirs) == 0 || length(ppt_dirs) == 0) {
    cat("WARNING - missing rasters\n")
    next
  }

  # Load rasters
  tmean_rast <- rast(pd_to_file(tmean_dirs))
  ppt_rast <- rast(pd_to_file(ppt_dirs))

  # Extract area-weighted tract means
  tmean_vals <- extract(tmean_rast, tracts_vect, fun = mean, na.rm = TRUE)
  ppt_vals <- extract(ppt_rast, tracts_vect, fun = mean, na.rm = TRUE)

  yr_df <- data.frame(
    GEOID = tracts$GEOID,
    year = yr,
    october_tmean = tmean_vals[, 2],
    october_ppt = ppt_vals[, 2]
  )

  cat(sprintf("tmean=%.1f°C, ppt=%.0fmm (n=%d tracts)\n",
              mean(yr_df$october_tmean, na.rm = TRUE),
              mean(yr_df$october_ppt, na.rm = TRUE),
              sum(!is.na(yr_df$october_tmean))))

  results <- rbind(results, yr_df)
}

# --- Save ---
dir.create(dirname(out_file), recursive = TRUE, showWarnings = FALSE)
write.csv(results, out_file, row.names = FALSE)
cat(sprintf("\nSaved: %s (%d rows, %d tracts, years %d-%d)\n",
            out_file, nrow(results), length(unique(results$GEOID)),
            min(results$year), max(results$year)))

# Coverage report
cat("\nCoverage (% non-missing):\n")
cat(sprintf("  october_tmean: %.1f%%\n",
            100 * sum(!is.na(results$october_tmean)) / nrow(results)))
cat(sprintf("  october_ppt: %.1f%%\n",
            100 * sum(!is.na(results$october_ppt)) / nrow(results)))

cat("Done.\n")

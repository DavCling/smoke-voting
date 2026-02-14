#!/usr/bin/env Rscript
# Download PRISM October weather data and extract county-level averages.
#
# Requires: install.packages(c("terra", "prism"))
# Also uses: sf, tigris (already installed for map scripts)
#
# Outputs: data/controls/prism/county_weather_october.csv
#   Columns: fips, year, october_tmean, october_ppt

library(terra)
library(prism)
library(sf)
library(tigris)

# --- Paths ---
# Resolve base directory robustly (works with Rscript and source())
get_base_dir <- function() {
  # Try sys.frame (works when source()'d)
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile)) return(dirname(dirname(normalizePath(ofile))))
  # Try commandArgs (works with Rscript)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1]))
    return(dirname(dirname(script_path)))
  }
  # Fallback to working directory
  return(getwd())
}
base_dir <- get_base_dir()
prism_dir <- file.path(base_dir, "data", "controls", "prism")
out_file <- file.path(prism_dir, "county_weather_october.csv")

if (file.exists(out_file)) {
  cat("Already exists:", out_file, "\n")
  df <- read.csv(out_file)
  cat(sprintf("  %d rows, years %d-%d\n", nrow(df), min(df$year), max(df$year)))
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

# --- Load county polygons (lower-48, 2019 vintage) ---
cat("\nLoading county polygons...\n")
options(tigris_use_cache = TRUE)
counties <- counties(cb = TRUE, year = 2019, resolution = "20m")

# Keep lower-48 only (exclude AK=02, HI=15, territories)
counties <- counties[!counties$STATEFP %in% c("02", "15", "60", "66", "69", "72", "78"), ]
counties$fips <- paste0(counties$STATEFP, counties$COUNTYFP)
cat(sprintf("  %d counties\n", nrow(counties)))

# Convert to terra SpatVector for extraction
counties_vect <- vect(counties)

# --- Extract county-level weather for each year ---
cat("\nExtracting county averages...\n")

results <- data.frame()

for (yr in years) {
  cat(sprintf("  %d: ", yr))

  # Find PRISM raster paths for this year/month
  tmean_dirs <- prism_archive_subset("tmean", "monthly", years = yr, mon = 10, resolution = "4km")
  ppt_dirs <- prism_archive_subset("ppt", "monthly", years = yr, mon = 10, resolution = "4km")

  if (length(tmean_dirs) == 0 || length(ppt_dirs) == 0) {
    cat("WARNING - missing rasters\n")
    next
  }

  # Load rasters
  tmean_rast <- rast(pd_to_file(tmean_dirs))
  ppt_rast <- rast(pd_to_file(ppt_dirs))

  # Extract area-weighted county means
  tmean_vals <- extract(tmean_rast, counties_vect, fun = mean, na.rm = TRUE)
  ppt_vals <- extract(ppt_rast, counties_vect, fun = mean, na.rm = TRUE)

  yr_df <- data.frame(
    fips = counties$fips,
    year = yr,
    october_tmean = tmean_vals[, 2],
    october_ppt = ppt_vals[, 2]
  )

  cat(sprintf("tmean=%.1f°C, ppt=%.0fmm\n",
              mean(yr_df$october_tmean, na.rm = TRUE),
              mean(yr_df$october_ppt, na.rm = TRUE)))

  results <- rbind(results, yr_df)
}

# --- Save ---
write.csv(results, out_file, row.names = FALSE)
cat(sprintf("\nSaved: %s (%d rows, %d counties, years %d-%d)\n",
            out_file, nrow(results), length(unique(results$fips)),
            min(results$year), max(results$year)))

cat("Done.\n")

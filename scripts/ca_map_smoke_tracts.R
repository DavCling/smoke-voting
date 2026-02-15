#!/usr/bin/env Rscript
#
# CA Tract-Level Smoke Exposure Maps
#
# Creates:
#   1. Choropleth maps of smoke exposure by year (faceted)
#   2. Residualized smoke map (after removing tract + year means)
#   3. Regional focus maps (Northern CA, Central Valley)
#
# Dependencies: arrow, sf, tigris, ggplot2, dplyr

suppressPackageStartupMessages({
  library(arrow)
  library(sf)
  library(tigris)
  library(ggplot2)
  library(dplyr)
  library(RColorBrewer)
})

options(tigris_use_cache = TRUE)

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
fig_dir <- file.path(base_dir, "output", "california", "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# ---- Load data ----
pres_file <- file.path(base_dir, "output", "california",
                       "ca_smoke_voting_presidential.parquet")
house_file <- file.path(base_dir, "output", "california",
                        "ca_smoke_voting_house.parquet")

df <- NULL
if (file.exists(pres_file)) {
  df <- read_parquet(pres_file)
  cat(sprintf("Presidential data: %d rows\n", nrow(df)))
}
if (file.exists(house_file)) {
  house <- read_parquet(house_file)
  cat(sprintf("House data: %d rows\n", nrow(house)))
  if (is.null(df)) {
    df <- house
  } else {
    df <- bind_rows(
      df |> select(geoid, year, smoke_pm25_mean_30d),
      house |> select(geoid, year, smoke_pm25_mean_30d)
    ) |> distinct(geoid, year, .keep_all = TRUE)
  }
}

if (is.null(df)) {
  cat("ERROR: No analysis data found. Run build scripts first.\n")
  quit(save = "no")
}

# Ensure GEOID is zero-padded
df <- df |> mutate(geoid = sprintf("%011s", geoid))

# ---- Load CA tract shapefile ----
cat("Loading CA tract boundaries...\n")
tracts_sf <- tracts(state = "CA", cb = TRUE, year = 2019) |>
  st_transform(3310)  # CA Albers Equal Area

cat(sprintf("  %d tracts in shapefile\n", nrow(tracts_sf)))

# Merge data with geometry
map_data <- tracts_sf |>
  inner_join(df |> select(geoid, year, smoke_pm25_mean_30d),
             by = c("GEOID" = "geoid"))

cat(sprintf("  %d tract-year observations mapped\n", nrow(map_data)))

# ---- State outline for overlay ----
ca_outline <- states(cb = TRUE, year = 2019) |>
  filter(STATEFP == "06") |>
  st_transform(3310)

# County outlines for context
ca_counties <- counties(state = "CA", cb = TRUE, year = 2019) |>
  st_transform(3310)

# ============================================================
# 1. Smoke exposure by year (faceted map)
# ============================================================
cat("\nCreating faceted smoke exposure map...\n")

# Cap extreme values for better color discrimination
q99 <- quantile(map_data$smoke_pm25_mean_30d, 0.99, na.rm = TRUE)
map_data <- map_data |>
  mutate(smoke_capped = pmin(smoke_pm25_mean_30d, q99))

p1 <- ggplot(map_data) +
  geom_sf(aes(fill = smoke_capped), color = NA, linewidth = 0) +
  geom_sf(data = ca_counties, fill = NA, color = "gray60", linewidth = 0.1) +
  geom_sf(data = ca_outline, fill = NA, color = "black", linewidth = 0.3) +
  facet_wrap(~year, nrow = 2) +
  scale_fill_distiller(
    palette = "Spectral",
    direction = -1,
    name = expression("Mean smoke PM"[2.5] ~ "(30d, " * mu * "g/m"^3 * ")"),
    limits = c(0, q99),
    na.value = "gray90"
  ) +
  theme_void() +
  theme(
    strip.text = element_text(size = 11, face = "bold"),
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    legend.title = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
  ) +
  labs(title = "California Tract-Level Wildfire Smoke Exposure")

ggsave(file.path(fig_dir, "ca_smoke_map_by_year.png"), p1,
       width = 12, height = 8, dpi = 200)
cat("  Saved: ca_smoke_map_by_year.png\n")

# ============================================================
# 2. Residualized smoke map (remove tract + year means)
# ============================================================
cat("\nCreating residualized smoke map...\n")

# Compute residuals: smoke_resid = smoke - tract_mean - year_mean + grand_mean
resid_data <- map_data |>
  group_by(GEOID) |>
  mutate(tract_mean = mean(smoke_pm25_mean_30d, na.rm = TRUE)) |>
  ungroup() |>
  group_by(year) |>
  mutate(year_mean = mean(smoke_pm25_mean_30d, na.rm = TRUE)) |>
  ungroup() |>
  mutate(
    grand_mean = mean(smoke_pm25_mean_30d, na.rm = TRUE),
    smoke_resid = smoke_pm25_mean_30d - tract_mean - year_mean + grand_mean
  )

# Cap residuals for visualization
q_low <- quantile(resid_data$smoke_resid, 0.01, na.rm = TRUE)
q_high <- quantile(resid_data$smoke_resid, 0.99, na.rm = TRUE)
resid_data <- resid_data |>
  mutate(smoke_resid_capped = pmin(pmax(smoke_resid, q_low), q_high))

p2 <- ggplot(resid_data) +
  geom_sf(aes(fill = smoke_resid_capped), color = NA, linewidth = 0) +
  geom_sf(data = ca_counties, fill = NA, color = "gray60", linewidth = 0.1) +
  geom_sf(data = ca_outline, fill = NA, color = "black", linewidth = 0.3) +
  facet_wrap(~year, nrow = 2) +
  scale_fill_distiller(
    palette = "RdBu",
    direction = -1,
    name = expression("Residualized smoke PM"[2.5] ~ "(" * mu * "g/m"^3 * ")"),
    limits = c(q_low, q_high),
    na.value = "gray90"
  ) +
  theme_void() +
  theme(
    strip.text = element_text(size = 11, face = "bold"),
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    legend.title = element_text(size = 10),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
  ) +
  labs(title = "Residualized Smoke Exposure (Tract + Year Means Removed)")

ggsave(file.path(fig_dir, "ca_smoke_map_residualized.png"), p2,
       width = 12, height = 8, dpi = 200)
cat("  Saved: ca_smoke_map_residualized.png\n")

# ============================================================
# 3. Regional focus: Northern CA (high smoke area)
# ============================================================
cat("\nCreating Northern CA regional map...\n")

# Northern CA counties (approximate: north of Sacramento)
norcal_counties <- c("Butte", "Shasta", "Tehama", "Glenn", "Lake", "Mendocino",
                     "Humboldt", "Trinity", "Siskiyou", "Del Norte", "Lassen",
                     "Modoc", "Plumas", "Sierra", "Nevada", "Placer", "El Dorado",
                     "Yuba", "Sutter", "Colusa")

norcal_fips <- ca_counties |>
  filter(NAME %in% norcal_counties) |>
  pull(GEOID)

norcal_tracts <- map_data |>
  filter(substr(GEOID, 1, 5) %in% norcal_fips)

norcal_outline <- ca_counties |>
  filter(GEOID %in% norcal_fips)

if (nrow(norcal_tracts) > 0) {
  p3 <- ggplot(norcal_tracts) +
    geom_sf(aes(fill = smoke_capped), color = NA, linewidth = 0) +
    geom_sf(data = norcal_outline, fill = NA, color = "gray40", linewidth = 0.2) +
    facet_wrap(~year, nrow = 2) +
    scale_fill_distiller(
      palette = "Spectral",
      direction = -1,
      name = expression("Mean smoke PM"[2.5] ~ "(30d)"),
      limits = c(0, q99),
      na.value = "gray90"
    ) +
    theme_void() +
    theme(
      strip.text = element_text(size = 11, face = "bold"),
      legend.position = "bottom",
      legend.key.width = unit(2, "cm"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    ) +
    labs(title = "Northern California: Tract-Level Smoke Exposure")

  ggsave(file.path(fig_dir, "ca_smoke_map_norcal.png"), p3,
         width = 10, height = 8, dpi = 200)
  cat("  Saved: ca_smoke_map_norcal.png\n")
}

cat("\nMap generation complete.\n")

library(arrow)
library(sf)
library(tigris)
library(ggplot2)
library(dplyr)
library(RColorBrewer)

options(tigris_use_cache = TRUE)

# Resolve base directory (works with both Rscript and source())
base_dir <- tryCatch({
  normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."))
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("--file=", "", args[grep("--file=", args)])
  normalizePath(file.path(dirname(script_path), ".."))
})

# --- Load analysis datasets (presidential + county house for 2018 coverage) ---
pres_df <- read_parquet(file.path(base_dir, "output", "smoke_voting_analysis.parquet"))
house_df <- read_parquet(file.path(base_dir, "output", "smoke_voting_county_house_analysis.parquet"))

# Combine and keep unique fips-year with 30-day smoke measure
panel <- bind_rows(
  pres_df |> select(fips, year, smoke_pm25_mean_30d),
  house_df |> select(fips, year, smoke_pm25_mean_30d)
) |>
  mutate(fips = sprintf("%05s", fips)) |>
  distinct(fips, year, .keep_all = TRUE)

# --- Get county shapefile (lower 48 + DC) ---
counties_sf <- counties(cb = TRUE, resolution = "20m", year = 2019) |>
  st_transform(5070) |>  # Albers Equal Area
  mutate(fips = GEOID) |>
  filter(!STATEFP %in% c("02", "15", "60", "66", "69", "72", "78"))  # drop AK, HI, territories

# --- Merge ---
map_data <- counties_sf |>
  inner_join(panel, by = "fips")

# --- Cap extreme values for better color discrimination ---
q99 <- quantile(map_data$smoke_pm25_mean_30d, 0.99, na.rm = TRUE)
map_data <- map_data |>
  mutate(smoke_capped = pmin(smoke_pm25_mean_30d, q99))

# --- State outlines for overlay ---
states_sf <- states(cb = TRUE, resolution = "20m", year = 2019) |>
  st_transform(5070) |>
  filter(!STATEFP %in% c("02", "15", "60", "66", "69", "72", "78"))

# --- Plot ---
p <- ggplot(map_data) +
  geom_sf(aes(fill = smoke_capped), color = NA, linewidth = 0) +
  geom_sf(data = states_sf, fill = NA, color = "gray40", linewidth = 0.15) +
  facet_wrap(~year, nrow = 2) +
  scale_fill_distiller(
    palette = "Spectral",
    direction = -1,
    name = expression("Mean smoke PM"[2.5] ~ "(30 days, " * mu * "g/m"^3 * ")"),
    limits = c(0, q99),
    oob = scales::squish,
    breaks = scales::pretty_breaks(5)
  ) +
  labs(
    title = "Pre-Election Wildfire Smoke Exposure by County",
    subtitle = "Mean wildfire-attributed PM2.5 in the 30 days before election day"
  ) +
  theme_void(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14,
                              margin = margin(t = 2, b = 2)),
    plot.subtitle = element_text(hjust = 0.5, color = "gray30", size = 10,
                                 margin = margin(b = 4)),
    strip.text = element_text(face = "bold", size = 13, margin = margin(t = 2, b = 2)),
    legend.position = "bottom",
    legend.key.width = unit(2.5, "cm"),
    legend.key.height = unit(0.4, "cm"),
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    legend.margin = margin(t = 0),
    plot.margin = margin(2, 5, 5, 5)
  )

out_path <- file.path(base_dir, "output", "figures", "smoke_exposure_map_panel.png")
ggsave(out_path, p, width = 14, height = 8, dpi = 200, bg = "white")
cat("Saved:", out_path, "\n")

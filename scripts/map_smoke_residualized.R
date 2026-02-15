library(arrow)
library(sf)
library(tigris)
library(ggplot2)
library(dplyr)

options(tigris_use_cache = TRUE)

# Resolve base directory
base_dir <- tryCatch({
  normalizePath(file.path(dirname(sys.frame(1)$ofile), ".."))
}, error = function(e) {
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("--file=", "", args[grep("--file=", args)])
  normalizePath(file.path(dirname(script_path), ".."))
})

# --- Load analysis datasets ---
pres_df <- read_parquet(file.path(base_dir, "output", "smoke_voting_analysis.parquet"))
house_df <- read_parquet(file.path(base_dir, "output", "smoke_voting_county_house_analysis.parquet"))

# Combine and keep unique fips-year
panel <- bind_rows(
  pres_df |> select(fips, year, state_fips, smoke_pm25_mean_30d),
  house_df |> select(fips, year, state_fips, smoke_pm25_mean_30d)
) |>
  mutate(fips = sprintf("%05s", fips),
         state_fips = sprintf("%02s", state_fips)) |>
  distinct(fips, year, .keep_all = TRUE)

# --- Residualize on state×year means ---
panel <- panel |>
  group_by(state_fips, year) |>
  mutate(state_year_mean = mean(smoke_pm25_mean_30d, na.rm = TRUE)) |>
  ungroup() |>
  mutate(smoke_resid = smoke_pm25_mean_30d - state_year_mean)

cat(sprintf("Residualized smoke: mean=%.4f, sd=%.4f, min=%.3f, max=%.3f\n",
            mean(panel$smoke_resid, na.rm = TRUE),
            sd(panel$smoke_resid, na.rm = TRUE),
            min(panel$smoke_resid, na.rm = TRUE),
            max(panel$smoke_resid, na.rm = TRUE)))

# --- Get county shapefile (lower 48 + DC) ---
counties_sf <- counties(cb = TRUE, resolution = "20m", year = 2019) |>
  st_transform(5070) |>
  mutate(fips = GEOID) |>
  filter(!STATEFP %in% c("02", "15", "60", "66", "69", "72", "78"))

states_sf <- states(cb = TRUE, resolution = "20m", year = 2019) |>
  st_transform(5070) |>
  filter(!STATEFP %in% c("02", "15", "60", "66", "69", "72", "78"))

# --- Merge ---
map_data <- counties_sf |>
  inner_join(panel, by = "fips")

# --- Cap at symmetric percentiles for diverging scale ---
q995 <- quantile(abs(map_data$smoke_resid), 0.995, na.rm = TRUE)
map_data <- map_data |>
  mutate(smoke_resid_capped = pmax(pmin(smoke_resid, q995), -q995))

# --- Plot: residualized smoke ---
p <- ggplot(map_data) +
  geom_sf(aes(fill = smoke_resid_capped), color = NA, linewidth = 0) +
  geom_sf(data = states_sf, fill = NA, color = "gray40", linewidth = 0.15) +
  facet_wrap(~year, nrow = 2) +
  scale_fill_gradient2(
    low = "#2166AC", mid = "white", high = "#B2182B",
    midpoint = 0,
    name = expression("Residual smoke PM"[2.5] ~ "(" * mu * "g/m"^3 * ")"),
    limits = c(-q995, q995),
    oob = scales::squish,
    breaks = scales::pretty_breaks(5)
  ) +
  labs(
    title = "Within-State Variation in Pre-Election Smoke Exposure",
    subtitle = "County smoke PM2.5 (30d) residualized on state×year means"
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

out_path <- file.path(base_dir, "output", "figures", "smoke_exposure_map_residualized.png")
ggsave(out_path, p, width = 14, height = 8, dpi = 200, bg = "white")
cat("Saved:", out_path, "\n")

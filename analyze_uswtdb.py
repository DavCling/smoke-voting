"""
Analyze USGS Wind Turbine Database - California subset
Feasibility investigation for Project 3: Wind Turbines and Voting Behavior
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('data/uswtdb/uswtdb_V8_2_20251210.csv')
print(f"Total US turbines: {len(df):,}")
print(f"States represented: {df['t_state'].nunique()}")

# California subset
ca = df[df['t_state'] == 'CA'].copy()
print(f"\n=== CALIFORNIA SUBSET ===")
print(f"Total CA turbines: {len(ca):,}")
print(f"Counties with turbines: {ca['t_county'].nunique()}")

# Year distribution
print(f"\n--- Installation Year Distribution ---")
print(f"Earliest: {ca['p_year'].min()}")
print(f"Latest: {ca['p_year'].max()}")
print(f"Missing p_year: {ca['p_year'].isna().sum()}")

year_counts = ca['p_year'].dropna().astype(int).value_counts().sort_index()
print(f"\nTurbines per year:")
for year, count in year_counts.items():
    print(f"  {year}: {count}")

# Capacity
print(f"\n--- Capacity Summary ---")
print(f"Total CA capacity: {ca['t_cap'].sum()/1000:,.1f} MW")
print(f"Mean turbine capacity: {ca['t_cap'].mean():,.0f} kW")
print(f"Median turbine capacity: {ca['t_cap'].median():,.0f} kW")

# Geographic clustering by county
print(f"\n--- Geographic Concentration (by county) ---")
county_summary = ca.groupby('t_county').agg(
    n_turbines=('case_id', 'count'),
    total_cap_mw=('t_cap', lambda x: x.sum()/1000),
    earliest=('p_year', 'min'),
    latest=('p_year', 'max')
).sort_values('n_turbines', ascending=False)

for county, row in county_summary.iterrows():
    earliest = int(row['earliest']) if pd.notna(row['earliest']) else 'N/A'
    latest = int(row['latest']) if pd.notna(row['latest']) else 'N/A'
    print(f"  {county}: {row['n_turbines']:,} turbines, {row['total_cap_mw']:,.0f} MW, {earliest}-{latest}")

# Key wind farm regions
print(f"\n--- Key Wind Farm Regions ---")
regions = {
    'Tehachapi Pass (Kern)': ca[ca['t_county'].str.contains('Kern', na=False)],
    'Altamont Pass (Alameda)': ca[ca['t_county'].str.contains('Alameda', na=False)],
    'San Gorgonio Pass (Riverside)': ca[ca['t_county'].str.contains('Riverside', na=False)],
    'Solano County': ca[ca['t_county'].str.contains('Solano', na=False)],
}
for name, subset in regions.items():
    print(f"\n  {name}:")
    print(f"    Turbines: {len(subset):,}")
    print(f"    Capacity: {subset['t_cap'].sum()/1000:,.0f} MW")
    yr_min = int(subset['p_year'].min()) if len(subset) > 0 and pd.notna(subset['p_year'].min()) else 'N/A'
    yr_max = int(subset['p_year'].max()) if len(subset) > 0 and pd.notna(subset['p_year'].max()) else 'N/A'
    print(f"    Year range: {yr_min}-{yr_max}")
    yr = subset['p_year'].dropna().astype(int).value_counts().sort_index()
    print(f"    Installation years: {dict(yr)}")

# Top 4 regions share
top4_counties = ['Kern County', 'Alameda County', 'Riverside County', 'Solano County']
top4 = ca[ca['t_county'].isin(top4_counties)]
print(f"\nTop 4 regions: {len(top4):,} turbines ({len(top4)/len(ca)*100:.1f}% of CA total)")
print(f"Top 4 capacity: {top4['t_cap'].sum()/1000:,.0f} MW ({top4['t_cap'].sum()/ca['t_cap'].sum()*100:.1f}% of CA total)")

# Project-level info
print(f"\n--- Project-Level Summary ---")
projects = ca.groupby('p_name').agg(
    n_turbines=('case_id', 'count'),
    total_cap_mw=('t_cap', lambda x: x.sum()/1000),
    county=('t_county', 'first'),
    year=('p_year', 'first')
).sort_values('total_cap_mw', ascending=False)
print(f"Total projects: {len(projects)}")
print(f"\nTop 20 projects by capacity:")
for name, row in projects.head(20).iterrows():
    print(f"  {name}: {row['n_turbines']} turbines, {row['total_cap_mw']:.0f} MW, {row['county']}, {int(row['year']) if pd.notna(row['year']) else 'N/A'}")

# Temporal analysis - installations by election cycle
print(f"\n--- Installations by Election Cycle ---")
ca_with_year = ca.dropna(subset=['p_year'])
election_years = [2000, 2002, 2004, 2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024]
for i in range(len(election_years) - 1):
    start = election_years[i]
    end = election_years[i+1]
    mask = (ca_with_year['p_year'] >= start) & (ca_with_year['p_year'] < end)
    n = mask.sum()
    cap = ca_with_year.loc[mask, 't_cap'].sum() / 1000
    print(f"  {start}-{end-1}: {n:,} turbines, {cap:,.0f} MW added")

# Coordinate bounds (for mapping)
print(f"\n--- Coordinate Bounds ---")
print(f"Latitude: {ca['ylat'].min():.4f} to {ca['ylat'].max():.4f}")
print(f"Longitude: {ca['xlong'].min():.4f} to {ca['xlong'].max():.4f}")

# Confidence scores
print(f"\n--- Data Quality (Confidence Scores) ---")
print(f"Location confidence (1=best, 3=worst):")
print(ca['t_conf_loc'].value_counts().sort_index().to_string())
print(f"\nAttribute confidence (1=best, 3=worst):")
print(ca['t_conf_atr'].value_counts().sort_index().to_string())

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('California Wind Turbines - USWTDB Analysis', fontsize=14, fontweight='bold')

# 1. Installations by year
ax = axes[0, 0]
year_counts.plot(kind='bar', ax=ax, color='steelblue', width=0.8)
ax.set_title('Turbines Installed per Year')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Turbines')
ax.tick_params(axis='x', rotation=45, labelsize=7)

# 2. Geographic distribution scatter
ax = axes[0, 1]
for county in top4_counties:
    subset = ca[ca['t_county'] == county]
    ax.scatter(subset['xlong'], subset['ylat'], s=3, alpha=0.5, label=county)
other = ca[~ca['t_county'].isin(top4_counties)]
ax.scatter(other['xlong'], other['ylat'], s=3, alpha=0.3, c='gray', label='Other')
ax.set_title('Turbine Locations by Region')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(fontsize=8, markerscale=3)

# 3. Capacity by county (top 10)
ax = axes[1, 0]
top10_cap = county_summary.head(10)['total_cap_mw']
top10_cap.plot(kind='barh', ax=ax, color='darkorange')
ax.set_title('Top 10 Counties by Capacity (MW)')
ax.set_xlabel('Capacity (MW)')

# 4. Cumulative capacity over time
ax = axes[1, 1]
ca_yearly = ca_with_year.groupby(ca_with_year['p_year'].astype(int))['t_cap'].sum() / 1000
cum_cap = ca_yearly.sort_index().cumsum()
cum_cap.plot(ax=ax, color='green', linewidth=2)
ax.set_title('Cumulative CA Wind Capacity (MW)')
ax.set_xlabel('Year')
ax.set_ylabel('Cumulative MW')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('data/uswtdb/ca_wind_turbines_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to data/uswtdb/ca_wind_turbines_analysis.png")

"""
Date: 250912
Author: danikae
Purpose: Visualize LCB production costs from agricultural and forest residues
         using uploaded biomass_cost_data.csv

Input CSV columns (observed):
- Country
- Biomass type
- Cost
- Cost units  (one of: "2018 USD / GJ" or "2024 USD / dry tonne")
- Source
"""

from load_inputs import load_global_parameters
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from common_tools import get_top_dir

top_dir = get_top_dir()
plots_dir = os.path.join(top_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# -----------------------------
# Constants & inflation
# -----------------------------
KG_PER_TONNE = 1000
F_RS_ENERGY_CONTENT = 20.0     # GJ/tonne (bone dry)
A_RS_ENERGY_CONTENT = 16.2     # GJ/tonne (bone dry)

glob = load_global_parameters()
USD_2018_TO_2024 = glob["2018_to_2024_USD"]["value"]

# -----------------------------
# Mappings (from your snippet)
# -----------------------------
country_to_continent = {
    "Norway": "Europe",
    "Finland": "Europe",
    "Germany": "Europe",
    "France": "Europe",
    "Brazil": "Latin America",
    "UK": "Europe",
    "Sweden": "Europe",
    "Japan": "Asia",
    "India": "Asia",
    "China": "Asia",
    "USA": "North America",
    "Turkey": "Europe",
    "Malaysia": "Asia",
    "Canada": "North America",
    "Mexico": "Latin America",
    "Thailand": "Asia",
    "Russia": "Former Soviet Union",
    "Chile": "Latin America",
    "Egypt": "Middle East",
    "Ghana": "Africa",
}

country_to_region = {
    "Norway": "Scandinavia",
    "Finland": "Scandinavia",
    "Germany": "Rest of Europe",
    "France": "Rest of Europe",
    "Brazil": "Brazil",
    "UK": "Rest of Europe",
    "Sweden": "Scandinavia",
    "Japan": "Japan",
    "India": "India",
    "China": "China",
    "USA": "North America",
    "Turkey": "Middle East",
    "Malaysia": "Southeast Asia",
    "Canada": "North America",
    "Mexico": "Mexico",
    "Thailand": "Southeast Asia",
    "Russia": "Former Soviet Union",
    "Chile": "Rest of Latin America",
    "Egypt": "Middle East",
    "Ghana": "Africa",
}

# -----------------------------
# Load & clean input
# -----------------------------
df = pd.read_csv(f"{top_dir}/data/biomass_cost_data.csv")

# Normalize
df["Country"] = df["Country"].astype(str).str.strip()
df["Biomass type"] = df["Biomass type"].astype(str).str.strip()
df["Cost units"] = df["Cost units"].astype(str).str.strip()

# Expected residue types and units
valid_residue_types = {"Ag_rs", "F_rs"}
UNIT_2018 = "2018 USD / GJ"
UNIT_2024_T = "2024 USD / dry tonne"

# Drop unknown residue types (report once)
unknown_types = sorted(set(df["Biomass type"]) - valid_residue_types)
if unknown_types:
    print(f"Note: Dropping rows with unknown Biomass type values: {unknown_types}")
    df = df[df["Biomass type"].isin(valid_residue_types)].copy()

# -----------------------------
# Convert to 2024 USD per dry tonne (row-wise)
# -----------------------------
energy_map = {"Ag_rs": A_RS_ENERGY_CONTENT, "F_rs": F_RS_ENERGY_CONTENT}
df["Energy_GJ_per_t"] = df["Biomass type"].map(energy_map)

def row_to_2024_usd_per_tonne(row):
    units = row["Cost units"]
    cost = row["Cost"]
    if pd.isna(cost):
        return np.nan
    # Normalize units for robust matching
    units_norm = str(units).strip().lower()
    if units_norm == UNIT_2018.lower():
        # Convert 2018 USD/GJ -> 2024 USD/tonne
        energy = row["Energy_GJ_per_t"]
        if pd.isna(energy):
            return np.nan
        return cost * energy * USD_2018_TO_2024
    elif units_norm == UNIT_2024_T.lower():
        # Already 2024 USD/tonne
        return cost
    else:
        # Unknown units: warn and skip
        return np.nan

df["Cost_2024_USD_per_tonne"] = df.apply(row_to_2024_usd_per_tonne, axis=1)

# Report unknown/unsupported units (if any)
unsupported_mask = ~df["Cost units"].str.lower().isin({UNIT_2018.lower(), UNIT_2024_T.lower()})
if unsupported_mask.any():
    bad_units = sorted(df.loc[unsupported_mask, "Cost units"].unique().tolist())
    print(f"Warning: Found rows with unsupported 'Cost units': {bad_units}. These rows were skipped in plotting.")

# Add group labels from mappings
df["Continent"] = df["Country"].map(country_to_continent)
df["Region"] = df["Country"].map(country_to_region)

# -----------------------------
# Plot helpers (Matplotlib-only)
# -----------------------------
def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s).strip()).strip("_")

def violin_scatter_by_group(
    data: pd.DataFrame,
    group_col: str,
    residue_type: str,  # "Ag_rs" or "F_rs"
    title_prefix: str
):
    """
    Make a horizontal violin plot with scatter overlay for the given group_col.
    Includes a legend with per-group medians.
    Saves as PNG and PDF in {top_dir}/plots.
    """
    # Filter
    sub = data[(data["Biomass type"] == residue_type) & (~data[group_col].isna())].copy()
    sub = sub[~sub["Cost_2024_USD_per_tonne"].isna()].copy()

    if sub.empty:
        print(f"(No data to plot) {title_prefix} — {residue_type}")
        return

    # Prepare grouped values and medians
    group_order = sorted(sub[group_col].dropna().unique().tolist())
    values_by_group = []
    medians_by_group = []

    for g in group_order:
        vals = sub.loc[sub[group_col] == g, "Cost_2024_USD_per_tonne"].dropna().values
        if len(vals) > 0:
            values_by_group.append(vals)
            medians_by_group.append(np.median(vals))

    # Drop groups with zero datapoints (should be none after filtering)
    if len(values_by_group) == 0:
        print(f"(No data to plot after filtering) {title_prefix} — {residue_type}")
        return

    # Build the plot
    fig_height = max(4, 0.4 * len(values_by_group))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Horizontal violin
    positions = np.arange(len(values_by_group))
    ax.violinplot(
        values_by_group,
        positions=positions,
        vert=False,
        showmeans=False,
        showextrema=False,
        showmedians=False
    )

    # Overlay scatter with small jitter
    rng = np.random.default_rng(42)
    for i, vals in enumerate(values_by_group):
        y = np.full_like(vals, i, dtype=float) + rng.uniform(-0.08, 0.08, size=len(vals))
        ax.plot(vals, y, linestyle="None", marker="o", markersize=3, alpha=0.6)

    # Add median markers and legend entries
    legend_handles = []
    for i, med in enumerate(medians_by_group):
        ax.plot([med], [i], marker="|", markersize=12)
        legend_handles.append(
            plt.Line2D([0],[0], marker="|", linestyle="None", markersize=12,
                       label=f"{group_order[i]} median: {med:.0f}")
        )

    # Axis cosmetics
    ax.set_yticks(np.arange(len(group_order)))
    ax.set_yticklabels(group_order)
    nice_res = "Agricultural residues" if residue_type == "Ag_rs" else "Forestry residues"
    ax.set_title(f"{title_prefix}: {nice_res}")
    ax.set_xlabel("Cost (2024 USD / dry tonne)")
    ax.set_ylabel(group_col)

    # Legend with medians
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()

    # Save figures
    base = f"{_slug(title_prefix)}__{_slug(residue_type)}__by_{_slug(group_col)}"
    png_path = os.path.join(plots_dir, base + ".png")
    pdf_path = os.path.join(plots_dir, base + ".pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.show()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

# -----------------------------
# 1) By Continent (Ag_rs & F_rs)
# -----------------------------
violin_scatter_by_group(df, group_col="Continent", residue_type="Ag_rs",
                        title_prefix="Residue cost by continent")
violin_scatter_by_group(df, group_col="Continent", residue_type="F_rs",
                        title_prefix="Residue cost by continent")

# -----------------------------
# 2) By Region (Ag_rs & F_rs)
# -----------------------------
violin_scatter_by_group(df, group_col="Region", residue_type="Ag_rs",
                        title_prefix="Residue cost by region")
violin_scatter_by_group(df, group_col="Region", residue_type="F_rs",
                        title_prefix="Residue cost by region")

# -----------------------------
# 3) By Country (Ag_rs & F_rs)
# -----------------------------
violin_scatter_by_group(df, group_col="Country", residue_type="Ag_rs",
                        title_prefix="Residue cost by country")
violin_scatter_by_group(df, group_col="Country", residue_type="F_rs",
                        title_prefix="Residue cost by country")

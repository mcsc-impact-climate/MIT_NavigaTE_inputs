"""
Date: 250527
Author: danikae
Purpose: Quick script to plot the breakdown of vessel costs to carry out annual LSFO trade for a given fuel production pathway
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from common_tools import get_top_dir
import os
import numpy as np

top_dir = get_top_dir()
RESULTS_DIR = f"{top_dir}/processed_results"

pathways = {
    "ammonia": "lowcost",
    "FTdiesel": "lowcost",
    "liquid_hydrogen": "lowcost",
    "lng": "lowcost",
    "lsfo": "fossil",
    "methanol": "lowcost",
}

fuels = ["ammonia", "liquid_hydrogen", "methanol", "FTdiesel", "lng", "lsfo"]

vessel_types = ["bulk_carrier_ice", "container_ice", "tanker_ice", "gas_carrier_ice"]

vessel_sizes = {
    "bulk_carrier_ice": [
        "bulk_carrier_capesize_ice",
        "bulk_carrier_handy_ice",
        "bulk_carrier_panamax_ice",
    ],
    "container_ice": [
        "container_15000_teu_ice",
        "container_8000_teu_ice",
        "container_3500_teu_ice",
    ],
    "tanker_ice": ["tanker_100k_dwt_ice", "tanker_300k_dwt_ice", "tanker_35k_dwt_ice"],
    "gas_carrier_ice": ["gas_carrier_100k_cbm_ice"],
}

vessel_type_title = {
    "bulk_carrier_ice": "Bulk Carrier",
    "container_ice": "Container",
    "tanker_ice": "Tanker",
    "gas_carrier_ice": "Gas Carrier",
}

vessel_size_title = {
    "bulk_carrier_capesize_ice": "Capesize",
    "bulk_carrier_handy_ice": "Handy",
    "bulk_carrier_panamax_ice": "Panamax",
    "container_15000_teu_ice": "15,000 TEU",
    "container_8000_teu_ice": "8,000 TEU",
    "container_3500_teu_ice": "3,500 TEU",
    "tanker_100k_dwt_ice": "100k DWT",
    "tanker_300k_dwt_ice": "300k DWT",
    "tanker_35k_dwt_ice": "35k DWT",
    "gas_carrier_100k_cbm_ice": "100k m$^3$",
}

CAPEX_components = ["BaseCAPEX", "TankCAPEX", "PowerCAPEX"]
OPEX_components = ["BaseOPEX", "TankOPEX", "PowerOPEX", "FuelOPEX"]

# Grouped structure: CAPEX first, then OPEX of same base, then FuelOPEX raw and delta
component_order = [
    ("BaseCAPEX", "BaseOPEX"),
    ("TankCAPEX", "TankOPEX"),
    ("PowerCAPEX", "PowerOPEX"),
    ("FuelOPEX_raw", "FuelOPEX_delta"),
]

# Color scheme
component_colors = {
    "Base": "tab:blue",
    "Tank": "tab:orange",
    "Power": "tab:green",
    "Fuel_raw": "tab:red",
    "Fuel_delta": "darkred",
}

# Legend labels
component_labels = {
    "BaseCAPEX": "Base Vessel CAPEX",
    "BaseOPEX": "Base Vessel OPEX",
    "TankCAPEX": "Tank CAPEX",
    "TankOPEX": "Tank OPEX",
    "PowerCAPEX": "Power Systems CAPEX",
    "PowerOPEX": "Power Systems OPEX",
    "FuelOPEX_raw": "Fuel OPEX (original cargo capacity)",
    "FuelOPEX_delta": "Fuel OPEX (accounting for tank displacement)",
}

def load_quantity_data(quantity):
    dfs = {}
    for fuel in fuels:
        pathway = pathways[fuel]
        fname = f"{RESULTS_DIR}/{fuel}-{pathway}-{quantity}-vessel.csv"
        df = pd.read_csv(fname, index_col=0)
        dfs[fuel] = df.loc["Global Average"]
    return dfs

# Load all required data once
all_data = {}
for quantity in CAPEX_components + OPEX_components + ["FinalTonneMiles"]:
    all_data[quantity] = load_quantity_data(quantity)

lsfo_miles = all_data["FinalTonneMiles"]["lsfo"]

# Main plotting loop
for vessel_type in vessel_types:
    for vessel_size in vessel_sizes[vessel_type]:
        fig, ax = plt.subplots(figsize=(12, 6))
        bottoms = np.zeros(len(fuels))
        plotted_labels = set()

        for group in component_order:
            for comp in group:
                values = []
                annotations = []

                for i, fuel in enumerate(fuels):
                    try:
                        if comp == "FuelOPEX_raw":
                            val = all_data["FuelOPEX"][fuel][vessel_size]
                            values.append(val)
                            annotations.append((i, False))

                        elif comp == "FuelOPEX_delta":
                            raw = all_data["FuelOPEX"][fuel][vessel_size]
                            fuel_miles = all_data["FinalTonneMiles"][fuel][vessel_size]
                            if fuel_miles == 0:
                                values.append(0)
                                annotations.append((i, True))
                                continue
                            norm_factor = lsfo_miles[vessel_size] / fuel_miles
                            normalized = raw * norm_factor
                            delta = max(normalized - raw, 0)
                            values.append(delta)
                            annotations.append((i, False))

                        else:
                            val = all_data[comp][fuel][vessel_size]
                            values.append(val)
                            annotations.append((i, False))

                    except Exception:
                        values.append(0)
                        annotations.append((i, True))

                if "FuelOPEX" in comp:
                    base = "Fuel_raw" if "raw" in comp else "Fuel_delta"
                else:
                    base = comp.replace("CAPEX", "").replace("OPEX", "")

                hatch = '///' if "OPEX" in comp and "delta" not in comp else ''
                color = component_colors.get(base, 'gray')
                label = component_labels.get(comp, comp) if comp not in plotted_labels else "_nolegend_"

                ax.barh(fuels, values, left=bottoms, label=label, color=color, hatch=hatch, edgecolor='k')
                bottoms += np.array(values)
                plotted_labels.add(comp)

                if comp == "FuelOPEX_delta":
                    for idx, needs_annotation in annotations:
                        if needs_annotation:
                            ax.text(
                                bottoms[idx] + 100000,
                                idx,
                                r"$\infty$ FuelOPEX after accounting for tank displacement",
                                va='center',
                                ha='left',
                                fontsize=12,
                                color='darkred'
                            )

        ax.set_xlabel("Annual Vessel Cost ($/year)", fontsize=16)
        ax.set_title(f"{vessel_type_title[vessel_type]} – {vessel_size_title[vessel_size]}", fontsize=16)
        ax.legend(
            title="Cost Component",
            loc="center left",
            bbox_to_anchor=(1.05, 0.8),
            borderaxespad=0.0,
            fontsize=12,
            title_fontsize=14,
        )
        plt.tight_layout()
        #plt.show()
        plt.savefig(f"{top_dir}/plots/cost_breakdown_{vessel_type}_{vessel_size}.png", dpi=300)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from common_tools import get_top_dir, get_fuel_label

top_dir = get_top_dir()
RESULTS_DIR = f"{top_dir}/processed_results_mod_caps_with_boiloff"

pathways = {
    "ammonia": "LTE_H_solar_E",
    "FTdiesel": "LTE_H_DAC_C_solar_E",
    "liquid_hydrogen": "LTE_H_solar_E",
    "lng": "fossil",
    "lsfo": "fossil",
    "methanol": "LTE_H_DAC_C_solar_E",
}

fuels = ["ammonia", "liquid_hydrogen", "methanol", "FTdiesel", "lng", "lsfo"]

vessel_types = ["bulk_carrier_ice", "container_ice", "tanker_ice", "gas_carrier_ice"]

vessel_sizes = {
    "bulk_carrier_ice": ["bulk_carrier_capesize_ice", "bulk_carrier_handy_ice", "bulk_carrier_panamax_ice"],
    "container_ice": ["container_15000_teu_ice", "container_8000_teu_ice", "container_3500_teu_ice"],
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

component_order = [
    ("BaseCAPEX", "BaseOPEX"),
    ("TankCAPEX", "TankOPEX"),
    ("PowerCAPEX", "PowerOPEX"),
    ("FuelOPEX_raw", "FuelOPEX_delta"),
]

component_colors = {
    "Base": "tab:blue",
    "Tank": "tab:orange",
    "Power": "tab:green",
    "Fuel_raw": "tab:red",
    "Fuel_delta": "darkred",
}

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

def plot_vessel_cost_breakdown(no_fuel=False):
    # Load required data
    all_data = {}
    required_quantities = CAPEX_components + OPEX_components + ["FinalTonneMiles"]
    for quantity in required_quantities:
        all_data[quantity] = load_quantity_data(quantity)

    lsfo_miles = all_data["FinalTonneMiles"]["lsfo"]

    for vessel_type in vessel_types:
        for vessel_size in vessel_sizes[vessel_type]:
            fig, ax = plt.subplots(figsize=(12, 6))
            bottoms = np.zeros(len(fuels))
            plotted_labels = set()
            fuel_labels = [get_fuel_label(f) for f in fuels]

            for group in component_order:
                for comp in group:
                    if no_fuel and "FuelOPEX" in comp:
                        continue

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
                                delta = max(raw * norm_factor - raw, 0)
                                values.append(delta)
                                annotations.append((i, False))
                            else:
                                val = all_data[comp][fuel][vessel_size]
                                values.append(val)
                                annotations.append((i, False))
                        except Exception:
                            values.append(0)
                            annotations.append((i, True))

                    base = "Fuel_raw" if comp == "FuelOPEX_raw" else \
                           "Fuel_delta" if comp == "FuelOPEX_delta" else \
                           comp.replace("CAPEX", "").replace("OPEX", "")
                    hatch = '///' if "OPEX" in comp and "delta" not in comp else ''
                    color = component_colors.get(base, 'gray')
                    label = component_labels.get(comp, comp) if comp not in plotted_labels else "_nolegend_"

                    ax.barh(fuel_labels, values, left=bottoms, label=label, color=color, hatch=hatch, edgecolor='k')
                    bottoms += np.array(values)
                    plotted_labels.add(comp)

                    if comp == "FuelOPEX_delta":
                        for idx, needs_annotation in annotations:
                            if needs_annotation:
                                ax.text(
                                    bottoms[idx] + 100000,
                                    fuel_labels[idx],
                                    r"$\infty$ FuelOPEX after accounting for tank displacement",
                                    va='center',
                                    ha='left',
                                    fontsize=12,
                                    color='darkred'
                                )

            ax.set_xlim([0, max(bottoms) * 1.15])
            ax.set_xlabel("Annual Vessel Cost ($/year)", fontsize=22, labelpad=20)
            ax.tick_params(axis="both", labelsize=20)
            ax.xaxis.get_offset_text().set_fontsize(20)
            ax.grid(True, axis='x', color='gray', ls='--')
            ax.set_title(f"{vessel_type_title[vessel_type]}: {vessel_size_title[vessel_size]}", fontsize=24)
            ax.legend(
                title="Cost Component",
                loc="center left",
                bbox_to_anchor=(1.05, 0.5),
                borderaxespad=0.0,
                fontsize=18,
                title_fontsize=20,
            )
            plt.tight_layout()
            outpath = f"{top_dir}/plots/cost_breakdown_{vessel_type}_{vessel_size}"
            outpath += "_nofuel.png" if no_fuel else ".png"
            plt.savefig(outpath, dpi=300)
            print(f"Plot saved to {outpath}")

def main():
    plot_vessel_cost_breakdown(no_fuel=True)

if __name__ == "__main__":
    main()

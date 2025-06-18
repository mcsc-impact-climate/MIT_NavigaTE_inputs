"""
Date: Jun 2, 2025
Author: danikam
Purpose: Make plots to summarize the tank sizes and cargo capacities going into the Singapore-Rotterdam simulation
"""

from common_tools import get_top_dir
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from pathlib import Path

top_dir = get_top_dir()

vessel_types = ["bulk_carrier_ice", "container_ice", "tanker_ice", "gas_carrier_ice"]

vessel_sizes = {
    "bulk_carrier_ice": [
        "bulk_carrier_capesize_ice",
        "bulk_carrier_handy_ice",
        "bulk_carrier_panamax_ice",
    ],
    "container_ice": [
        "container_8000_teu_ice",
        "container_3500_teu_ice",
    ],
    "tanker_ice": ["tanker_100k_dwt_ice", "tanker_35k_dwt_ice"],
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
    "container_8000_teu_ice": "8,000 TEU",
    "container_3500_teu_ice": "3,500 TEU",
    "tanker_100k_dwt_ice": "100k DWT",
    "tanker_35k_dwt_ice": "35k DWT",
    "gas_carrier_100k_cbm_ice": "100k m$^3$",
}

fuel_colors = {
    "ammonia": "tab:blue",
    "hydrogen": "tab:orange",
    "methanol": "tab:green",
    "diesel": "tab:red",
    "oil": "black",
    "lsfo": "black",
    "lng": "tab:grey",
    "methane": "tab:grey"
}

power_systems_names = {
    "ammonia": "Ammonia (dual fuel)",
    "hydrogen": "Liquid Hydrogen (dual fuel)",
    "methanol": "Methanol (dual fuel)",
    "diesel": "FT Diesel (single fuel)",
    "oil": "LSFO (single fuel)",
    "methane": "Methane (dual fuel)"
}

power_system_colors = {
    "Ammonia (dual fuel)": "tab:blue",
    "Liquid Hydrogen (dual fuel)": "tab:orange",
    "Methanol (dual fuel)": "tab:green",
    "FT Diesel (single fuel)": "tab:red",
    "LSFO (single fuel)": "black",
    "Methane (dual fuel)": "tab:grey"
}

nominal_capacity_units = {
    "bulk_carrier_ice": "DWT",
    "container_ice": "TEU",
    "tanker_ice": "DWT",
    "gas_carrier_ice": "m$^3$",
}

# -- Load vessel info --
vessel_info_path = Path(top_dir) / "includes_global" / "vessels_singapore_rotterdam.inc"
vessel_content = vessel_info_path.read_text()

# -- Parse data --
vessel_data = defaultdict(lambda: defaultdict(dict))
pattern = r'Vessel\s+"([^"]+)"\s+\{[^}]*?NominalCapacity\s*=\s*([\d\.]+)'

for match in re.finditer(pattern, vessel_content):
    name, cap = match.groups()
    cap = float(cap)
    for vtype, sizes in vessel_sizes.items():
        for size in sizes:
            if name.startswith(size):
                for k, fuel in power_systems_names.items():
                    if k in name:
                        vessel_data[vtype][size][fuel] = cap

# -- Plotting routine with centered y-axis labels and vertical gaps --
def plot_nominal_capacity_histograms():
    for vclass, sizes in vessel_data.items():
        fuels = sorted(set(f for s in sizes.values() for f in s))
        fig, ax = plt.subplots(figsize=(10, 2 + len(sizes)))

        bar_height = 0.1                # Height of each individual bar
        gap_between_clusters = 0.3     # Gap between vessel size clusters
        yticks, yticklabels = [], []
        y_base = 0

        for idx, (size, fuel_caps) in enumerate(sizes.items()):
            for i, fuel in enumerate(fuels):
                val = fuel_caps.get(fuel, 0)
                y_pos = y_base + i * bar_height
                ax.barh(y_pos, val, height=bar_height,
                        color=power_system_colors.get(fuel, "gray"),
                        label=fuel if idx == 0 else "")
            # Add label at center of this group
            cluster_height = len(fuels) * bar_height
            yticks.append(y_base + cluster_height / 2 - bar_height / 2)
            yticklabels.append(vessel_size_title.get(size, size))
            # Advance y_base to leave room for the next cluster + gap
            y_base += cluster_height + gap_between_clusters

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(f"Nominal Cargo Capacity ({nominal_capacity_units[vclass]})")
        ax.set_title(f"{vessel_type_title[vclass]} Vessels: Nominal Cargo Capacity by Fuel")

        # De-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.tight_layout()
        #plt.show()
        plt.savefig(f"{top_dir}/plots/nominal_capacities_{vclass}_singapore_rotterdam.pdf")
        
# ------------------- Tank Size Plot -------------------

tank_fuels = {
    "liquid_hydrogen": "Liquid Hydrogen",
    "ammonia": "Ammonia",
    "methanol": "Methanol",
    "oil": "LSFO",
    "methane": "Methane",
    "FTdiesel": "Diesel"
}

power_system_tank_fuel_mapping = {
    "ammonia": "ammonia",
    "hydrogen": "liquid_hydrogen",
    "methanol": "methanol",
    "diesel": "FTdiesel",
    "oil": "oil",
    "methane": "methane"
}

tank_fuel_colors = {
    "Ammonia": "tab:blue",
    "Liquid Hydrogen": "tab:orange",
    "Methanol": "tab:green",
    "Diesel": "tab:red",
    "LSFO": "black",
    "Methane": "tab:grey"
}

def parse_tank_size(file_path):
    text = Path(file_path).read_text()
    match = re.search(r'Size\s*=\s*([0-9\.]+)', text)
    return float(match.group(1)) if match else 0

def read_tank_data(top_dir):
    tank_dir = Path(top_dir) / "NavigaTE/navigate/defaults/user/Tank"
    tank_data = defaultdict(lambda: defaultdict(dict))  # vessel_type → vessel_size → fuel → {"main", "pilot"}

    for vessel_type, sizes in vessel_sizes.items():
        for size in sizes:
            short_size = size.replace("_ice", "")
            for short_fuel, tank_fuel in power_system_tank_fuel_mapping.items():
                tank_file = tank_dir / f"main_{tank_fuel}_{short_size}.inc"
                if tank_file.exists():
                    fuel_label = tank_fuels[tank_fuel]
                    tank_data[vessel_type][size][fuel_label] = {
                        "main": parse_tank_size(tank_file),
                        "pilot": 0
                    }

                    # Dual-fuel? Add pilot
                    if short_fuel in ["ammonia", "hydrogen", "methanol", "methane"]:
                        pilot_file = tank_dir / f"pilot_oil_{short_size}.inc"
                        if pilot_file.exists():
                            tank_data[vessel_type][size][fuel_label]["pilot"] = parse_tank_size(pilot_file)

    return tank_data

def plot_tank_sizes(top_dir, tank_data):
    for vclass, size_dict in tank_data.items():
        fig, ax = plt.subplots(figsize=(12, 2 + len(size_dict)))

        y_pos = []
        yticklabels = []
        bars_main, bars_pilot, bar_colors, fuel_labels = [], [], [], []
        tick_centers = []
        y_base = 0
        gap_between_clusters = 0.5
        bar_height = 0.2

        for size in size_dict:
            fuels = sorted(size_dict[size].keys())
            num_fuels = len(fuels)

            for i, fuel in enumerate(fuels):
                fuel_info = size_dict[size][fuel]
                y = y_base + i * bar_height
                bars_main.append((y, fuel_info["main"]))
                bars_pilot.append(fuel_info["pilot"])
                bar_colors.append(tank_fuel_colors.get(fuel, "gray"))
                fuel_labels.append(fuel)
                y_pos.append(y)

            cluster_height = num_fuels * bar_height
            tick_centers.append(y_base + cluster_height / 2 - bar_height / 2)
            yticklabels.append(vessel_size_title.get(size, size))
            y_base += cluster_height + gap_between_clusters

        # Plot main tanks with hatch
        for (y, main), color in zip(bars_main, bar_colors):
            ax.barh(y, main, height=bar_height, facecolor=color, edgecolor="black", hatch="xx")

        # Plot pilot tanks stacked
        for (y, main), pilot, color in zip(bars_main, bars_pilot, bar_colors):
            ax.barh(y, pilot, left=main, height=bar_height, facecolor=color, edgecolor="black")

        ax.set_yticks(tick_centers)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Tank Size (m³)")
        ax.set_title(f"{vessel_type_title[vclass]} Vessels: Main + Pilot Tank Sizes")

        # Legend 1: Fuel colors
        unique_fuels = sorted(set(fuel_labels))
        fuel_patches = [
            plt.Rectangle((0, 0), 1, 1, facecolor=tank_fuel_colors[f], edgecolor="black", label=f)
            for f in unique_fuels
        ]
        fuel_legend = ax.legend(handles=fuel_patches, title="Fuel Type",
                                loc='center left', bbox_to_anchor=(1.01, 0.65))
        ax.add_artist(fuel_legend)

        # Legend 2: Hatch = Main, Solid = Pilot
        tank_patches = [
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="xx", label="Main Tank"),
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", label="Pilot Tank")
        ]
        ax.legend(handles=tank_patches, title="Tank Type",
                  loc='center left', bbox_to_anchor=(1.01, 0.25))

        plt.tight_layout()
        plt.subplots_adjust(right=0.7)
        outpath = Path(top_dir) / f"plots/tank_sizes_{vclass}_singapore_rotterdam.pdf"
        print(f"Saving figure to {outpath}")
        plt.savefig(outpath)


# ------------------- Run All -------------------

if __name__ == "__main__":
    #plot_nominal_capacity_histograms()
    tank_data = read_tank_data(top_dir)
    plot_tank_sizes(top_dir, tank_data)

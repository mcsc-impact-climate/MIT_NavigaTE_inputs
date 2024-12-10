"""
Date: Dec. 9, 2024
Author: danikam
Purpose: Analyze the effect of varying design range on the impact of tank size modifications to available cargo capacity.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from common_tools import get_top_dir, create_directory_if_not_exists, generate_blue_shades, get_pathway_label, read_pathway_labels, read_fuel_labels, get_fuel_label

import matplotlib.colors as mcolors
from parse import parse
import numpy as np

vessel_class_titles = {
    "bulk_carrier_capesize": "Bulk Carrier (Capesize)",
    "bulk_carrier_handy": "Bulk Carrier (Handy)",
    "bulk_carrier_panamax": "Bulk Carrier (Panamax)",
    "container_15000_teu": "Container (15,000 TEU)",
    "container_8000_teu": "Container (8,000 TEU)",
    "container_3500_teu": "Container (3,500 TEU)",
    "tanker_100k_dwt": "Tanker (100k DWT)",
    "tanker_300k_dwt": "Tanker (300k DWT)",
    "tanker_35k_dwt": "Tanker (35k DWT)",
    "gas_carrier_100k_cbm": "Gas Carrier (100k m³)",
}

def generate_distinct_shades_from_cmap(cmap_name, num_shades, min_intensity=0.2, max_intensity=1.0):
    """
    Generates a list of distinct color shades using a specified colormap,
    ensuring that the lightest shade is not too bright.

    Parameters
    ----------
    cmap_name : str
        Name of the matplotlib colormap.
    num_shades : int
        Number of shades to generate.
    min_intensity : float, optional
        Minimum intensity for the darkest color (default: 0.2).
    max_intensity : float, optional
        Maximum intensity for the brightest color (default: 1.0).

    Returns
    -------
    shades : list of str
        A list of distinct colors in hex format.
    """
    cmap = colormaps.get_cmap(cmap_name)
    # Adjust the intensity range to avoid very light or very dark colors
    return [
        mcolors.to_hex(cmap(min_intensity + (i / (num_shades - 1)) * (max_intensity - min_intensity)))
        for i in range(num_shades)
    ]
    

def plot_days_to_empty_tank(top_dir):
    # Define the parse pattern for matching filenames
    pattern = "days_to_empty_tank_{vessel_range:d}.csv"
    
    # Loop through all files in the given directory
    days_to_empty_tank_dfs = []
    vessel_ranges = []
    for filename in os.listdir(f"{top_dir}/tables"):
        # Attempt to parse the filename
        result = parse(pattern, filename)
        if result:
            # Extract the vessel range from the parsed result
            vessel_range = result["vessel_range"]
            print(f"Processing file: {filename} with vessel range: {vessel_range}")
            
            file_path = os.path.join(top_dir, "tables", filename)

            days_to_empty_tank_df = pd.read_csv(file_path)
            days_to_empty_tank_dfs.append(days_to_empty_tank_df)
            vessel_ranges.append(vessel_range)
    
    # Combine all data into a single DataFrame with vessel_range as a column
    all_data = pd.concat(
        [
            df.assign(vessel_range=vr)
            for df, vr in zip(days_to_empty_tank_dfs, vessel_ranges)
        ]
    )

    # Ensure vessel_range is treated as a numeric value
    all_data["vessel_range"] = pd.to_numeric(all_data["vessel_range"])

    # Define colors for each vessel type
    container_shades = generate_distinct_shades_from_cmap("Greens", 5)
    bulk_shades = generate_distinct_shades_from_cmap("Blues", 5)
    tanker_shades = generate_distinct_shades_from_cmap("Oranges", 5)
    gas_carrier_color = "black"  # Gray for gas carriers

    # Assign colors based on vessel type
    vessel_colors = {}
    idx = 0
    for vessel_class in vessel_class_titles.keys():
        if "container" in vessel_class:
            vessel_colors[vessel_class] = container_shades[idx % len(container_shades)]
        elif "bulk" in vessel_class:
            vessel_colors[vessel_class] = bulk_shades[idx % len(bulk_shades)]
        elif "tanker" in vessel_class:
            vessel_colors[vessel_class] = tanker_shades[idx % len(tanker_shades)]
        elif "gas_carrier" in vessel_class:
            vessel_colors[vessel_class] = gas_carrier_color
        idx += 1

    # Plot only Total Days for each vessel class
    plt.figure(figsize=(14, 8))
    for vessel_class in vessel_class_titles:
        vessel_data = all_data[all_data["Vessel Class"] == vessel_class]
        plt.plot(
            vessel_data["vessel_range"],
            vessel_data["Total Days"],
            linestyle="-",
            label=vessel_class_titles[vessel_class],
            linewidth=1.5,
            color=vessel_colors[vessel_class],
        )

    # Add legend, labels, and title
    plt.legend(title="Vessel Class", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=18, title_fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel("Vessel Range (nautical miles)", fontsize=20)
    plt.ylabel("Total Days", fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    create_directory_if_not_exists("plots")
    plt.savefig("plots/total_days_vs_range.png", dpi=300)
    
def plot_tank_size_correction_factors(top_dir):
    # Define the parse pattern for matching filenames
    pattern = "tank_size_factors_range_{vessel_range:d}.csv"
    
    # Loop through all files in the given directory
    tank_size_factors_dfs = []
    vessel_ranges = []
    for filename in os.listdir(f"{top_dir}/tables"):
        # Attempt to parse the filename
        result = parse(pattern, filename)
        if result:
            # Extract the vessel range from the parsed result
            vessel_range = result["vessel_range"]
            print(f"Processing file: {filename} with vessel range: {vessel_range}")
            
            file_path = os.path.join(top_dir, "tables", filename)

            tank_size_factors_df = pd.read_csv(file_path)
            tank_size_factors_dfs.append(tank_size_factors_df.assign(vessel_range=vessel_range))
            vessel_ranges.append(vessel_range)
    
    # Combine all data into a single DataFrame with vessel_range as a column
    all_data = pd.concat(tank_size_factors_dfs)

    # Ensure vessel_range is treated as a numeric value and sort the entire DataFrame
    all_data["vessel_range"] = pd.to_numeric(all_data["vessel_range"])
    all_data = all_data.sort_values(by="vessel_range")

    # Get unique fuels
    fuels = all_data["Fuel"].unique()

    # Create directory for saving plots
    create_directory_if_not_exists("plots")
    
    # Define colors for each vessel type
    container_shades = generate_distinct_shades_from_cmap("Greens", 5)
    bulk_shades = generate_distinct_shades_from_cmap("Blues", 5)
    tanker_shades = generate_distinct_shades_from_cmap("Oranges", 5)
    gas_carrier_color = "black"  # Gray for gas carriers
    
    # Assign colors based on vessel type
    vessel_colors = {}
    idx = 0
    for vessel_class in vessel_class_titles.keys():
        if "container" in vessel_class:
            vessel_colors[vessel_class] = container_shades[idx % len(container_shades)]
        elif "bulk" in vessel_class:
            vessel_colors[vessel_class] = bulk_shades[idx % len(bulk_shades)]
        elif "tanker" in vessel_class:
            vessel_colors[vessel_class] = tanker_shades[idx % len(tanker_shades)]
        elif "gas_carrier" in vessel_class:
            vessel_colors[vessel_class] = gas_carrier_color
        idx += 1

    # Generate a plot for each fuel
    for fuel in fuels:
        fuel_data = all_data[all_data["Fuel"] == fuel]

        # Create a figure with two subplots
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot Total correction factor
        for vessel_class in vessel_class_titles:
            vessel_data = fuel_data[fuel_data["Vessel Class"] == vessel_class].sort_values(by="vessel_range")
            ax.plot(
                vessel_data["vessel_range"],
                vessel_data["Total"],
                label=vessel_class_titles[vessel_class],
                linewidth=1.5,
                color=vessel_colors[vessel_class],
            )

        # Configure the upper plot
        fuel_label = get_fuel_label(fuel)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_title(fuel_label, fontsize=24)
        ax.set_ylabel("Tank Size Correction Factor", fontsize=20)
        ax.set_xlabel("Vessel Range (nautical miles)", fontsize=20)
        ax.legend(title="Vessel Class", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=18, title_fontsize=22)
        ax.grid(True)

        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(f"plots/tank_size_correction_factor_{fuel}.png", dpi=300)
        plt.close(fig)
        
def plot_cargo_loss(top_dir):
    # Define the parse pattern for matching filenames
    pattern = "modified_capacities_{vessel_range:d}.csv"

    # Loop through all files in the given directory
    modified_capacity_dfs = []
    vessel_ranges = []
    for filename in os.listdir(f"{top_dir}/tables"):
        # Attempt to parse the filename
        result = parse(pattern, filename)
        if result:
            # Extract the vessel range from the parsed result
            vessel_range = result["vessel_range"]
            print(f"Processing file: {filename} with vessel range: {vessel_range}")
            
            file_path = os.path.join(top_dir, "tables", filename)

            modified_capacity_df = pd.read_csv(file_path)
            modified_capacity_dfs.append(modified_capacity_df.assign(vessel_range=vessel_range))
            vessel_ranges.append(vessel_range)

    # Combine all data into a single DataFrame with vessel_range as a column
    all_data = pd.concat(modified_capacity_dfs)

    # Ensure vessel_range is treated as a numeric value and sort the entire DataFrame
    all_data["vessel_range"] = pd.to_numeric(all_data["vessel_range"])
    all_data = all_data.sort_values(by="vessel_range")

    # Get unique fuels
    fuels = all_data["Fuel"].unique()

    # Create directory for saving plots
    create_directory_if_not_exists("plots")
    
    # Define colors for each vessel type
    container_shades = generate_distinct_shades_from_cmap("Greens", 5)
    bulk_shades = generate_distinct_shades_from_cmap("Blues", 5)
    tanker_shades = generate_distinct_shades_from_cmap("Oranges", 5)
    gas_carrier_color = "black"  # Gray for gas carriers
    
    # Assign colors based on vessel type
    vessel_colors = {}
    idx = 0
    for vessel_class in vessel_class_titles.keys():
        if "container" in vessel_class:
            vessel_colors[vessel_class] = container_shades[idx % len(container_shades)]
        elif "bulk" in vessel_class:
            vessel_colors[vessel_class] = bulk_shades[idx % len(bulk_shades)]
        elif "tanker" in vessel_class:
            vessel_colors[vessel_class] = tanker_shades[idx % len(tanker_shades)]
        elif "gas_carrier" in vessel_class:
            vessel_colors[vessel_class] = gas_carrier_color
        idx += 1

    # Generate a plot for each fuel
    for fuel in fuels:
        fuel_data = all_data[all_data["Fuel"] == fuel]

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot Percent Volume Difference
        for vessel_class in vessel_class_titles:
            vessel_data = fuel_data[fuel_data["Vessel"] == vessel_class].sort_values(by="vessel_range")
            ax1.plot(
                vessel_data["vessel_range"],
                -1*vessel_data["Percent volume difference (%)"],
                label=vessel_class_titles[vessel_class],
                linewidth=1.5,
                color=vessel_colors[vessel_class],
            )

        # Configure the upper plot
        fuel_label = get_fuel_label(fuel)
        ax1.tick_params(axis='both', which='major', labelsize=18)
        ax1.set_title(fuel_label, fontsize=24)
        ax1.set_ylabel("Volume Capacity Loss (%)", fontsize=20)
        ax1.legend(title="Vessel Class", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=18, title_fontsize=22)
        ax1.grid(True)

        # Plot Percent Mass Difference
        for vessel_class in vessel_class_titles:
            vessel_data = fuel_data[fuel_data["Vessel"] == vessel_class].sort_values(by="vessel_range")
            ax2.plot(
                vessel_data["vessel_range"],
                -1*vessel_data["Percent mass difference (%)"],
                label=vessel_class_titles[vessel_class],
                linewidth=1.5,
                color=vessel_colors[vessel_class],
            )

        # Configure the lower plot
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.set_xlabel("Vessel Range (nautical miles)", fontsize=20)
        ax2.set_ylabel("Mass Capacity Loss (%)", fontsize=20)
        ax2.grid(True)

        # Adjust layout and save plot
        plt.tight_layout()
        plt.savefig(f"plots/cargo_capacity_loss_{fuel}.png", dpi=300)
        plt.close(fig)
            

if __name__ == "__main__":
    top_dir = get_top_dir()
    #plot_days_to_empty_tank(top_dir)
    #plot_tank_size_correction_factors(top_dir)
    plot_cargo_loss(top_dir)

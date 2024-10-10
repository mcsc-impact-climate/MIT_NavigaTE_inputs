"""
Date: Oct. 9, 2024
Author: danikam
Purpose: Modify tanks sizes and vessel cargo capacities based on the fuel density.
"""

import pandas as pd
import os
from common_tools import get_top_dir, get_fuel_label
import matplotlib.pyplot as plt
import parse
import numpy as np
import matplotlib.patches as mpatches

M3_PER_TEU = 38.28
L_PER_M3 = 1000
KG_PER_DWT = 1000
VESSELS_ORIG_DIR = "includes_global/vessels_orig_capacity"
VESSELS_MODIFIED_DIR = "includes_global/vessels_modified_capacity"
TANKS_ORIG_DIR = "includes_global/tanks_orig_size"
TANKS_MODIFIED_DIR = "includes_global/tanks_modified_size"

# Dictionary containing the keyword for the given fuel in vessel file names
fuel_vessel_dict = {
    "ammonia": "ammonia",
    "methanol": "methanol",
    "FTdiesel": "diesel",
    "compressed_hydrogen": "hydrogen",
    "liquid_hydrogen": "hydrogen",
    "lsfo": "oil",
}

# Vessel type and size information
vessels = {
    "bulk_carrier": [
        "bulk_carrier_capesize",
        "bulk_carrier_handy",
        "bulk_carrier_panamax",
    ],
    "container": [
        "container_15000_teu",
        "container_8000_teu",
        "container_3500_teu",
    ],
    "tanker": ["tanker_100k_dwt", "tanker_300k_dwt", "tanker_35k_dwt"],
    "gas_carrier": ["gas_carrier_100k_cbm"],
}

vessel_type_title = {
    "bulk_carrier": "Bulk Carrier (ICE)",
    "container": "Container (ICE)",
    "tanker": "Tanker (ICE)",
    "gas_carrier": "Gas Carrier (ICE)",
}

vessel_size_title = {
    "bulk_carrier_capesize": "Capesize",
    "bulk_carrier_handy": "Handy",
    "bulk_carrier_panamax": "Panamax",
    "container_15000_teu": "15,000 TEU",
    "container_8000_teu": "8,000 TEU",
    "container_3500_teu": "3,500 TEU",
    "tanker_100k_dwt": "100k DWT",
    "tanker_300k_dwt": "300k DWT",
    "tanker_35k_dwt": "35k DWT",
    "gas_carrier_100k_cbm": "100k m$^3$",
}

def get_fuel_info_dict(info_filepath, column_name):
    """
    Collects the specified fuel info from an info csv file

    Parameters
    ----------
    info_filepath : str
        Path to a csv file containing info for each fuel

    Returns
    -------
    fuel_info_dict : Dictionary
        Dictionary containing the fuel info corresponding to each fuel
    """
    # Read the CSV file
    df = pd.read_csv(info_filepath)
    
    # Create a dictionary from the 'Fuel' column and 'Mass density (kg/L)' column
    fuel_info_dict = dict(zip(df['Fuel'], df[column_name]))
    
    return fuel_info_dict
    
def get_tank_size(tank_size_lsfo, LHV_lsfo, mass_density_lsfo, LHV_fuel, mass_density_fuel):
    """
    Calculates the size of the tank needed to provide equivalent fuel energy to the vessel as LSFO

    Parameters
    ----------
    tank_size_lsfo : float
        Size of the tank for the LSFO vessel

    LHV_lsfo : float
        LHV of LSFO
        
    mass_density_lsfo : float
        Mass density of LSFO

    LHV_fuel : float
        LHV of the fuel
        
    mass_density_fuel : float
        Mass density of the fuel

    Returns
    -------
    tank_size_fuel : float
        Size of the tank with an equivalent fuel energy to the LSFO tank
    """
    
    tank_size_fuel = ( tank_size_lsfo * LHV_lsfo * mass_density_lsfo ) / ( LHV_fuel * mass_density_fuel )
    
    return tank_size_fuel
    
def get_tank_size_factors(fuels, LHV_dict, mass_density_dict):
    """
    Creates a dictionary of tank size scaling factors for each fuel relative to LSFO
    
    Parameters
    ----------
    fuels : list of str
        List of fuels to include as keys in the dictionary

    Returns
    -------
    tank_size_factors_dict : Dictionary
        Dictionary containing the tank size factor for each fuel
    """
    LHV_lsfo = LHV_dict["lsfo"]
    mass_density_lsfo = mass_density_dict["lsfo"]
    
    tank_size_factors_dict = {}
    for fuel in fuels:
        LHV_fuel = LHV_dict[fuel]
        mass_density_fuel = mass_density_dict[fuel]
        tank_size_factors_dict[fuel] = get_tank_size(1, LHV_lsfo, mass_density_lsfo, LHV_fuel, mass_density_fuel)
        
    return tank_size_factors_dict
    
def plot_tank_size_factors(tank_size_factors_dict):
    """
    Plots a horizontal bar chart of tank size scaling factors.

    Parameters
    ----------
    tank_size_factors_dict : dict
        A dictionary with fuel types as keys and tank size scaling factors as values.
    """
    # Extracting keys and values from the dictionary
    fuels = list(tank_size_factors_dict.keys())
    factors = list(tank_size_factors_dict.values())

    # Obtain labels for each fuel using get_fuel_label
    labels = [get_fuel_label(fuel) for fuel in fuels]
    
    # Creating the plot
    plt.figure(figsize=(10, 5))
    bars = plt.barh(labels, factors, color='skyblue')
    plt.xlabel('Tank Size Scaling Factor', fontsize=22)
    plt.xticks(fontsize=20)  # Set font size for x-ticks
    plt.yticks(fontsize=20)  # Set font size for y-ticks
    plt.axvline(1, color='black', linestyle='--')  # Adding a dashed line at x=1
    plt.gca().invert_yaxis()  # Invert y-axis to display the first item at the top
    plt.tight_layout()
    plt.savefig("plots/tank_size_scaling_factors.png", dpi=300)
    
def modify_cargo_capacity(vessel_type, fuel, cargo_capacity_lsfo, tank_size_lsfo, tank_size_factors_dict, mass_density_dict):
    """
    Calculates the modified cargo capacity based on the change in tank size for the given fuel relative to LSFO.

    Parameters
    ----------
    vessel_type : str
        Filename keyword indicating the vessel type, which will determine how the cargo capacity is modified

    fuel : str
        Name of the fuel
        
    cargo_capacity_lsfo : float
        Nominal cargo capacity for the LSFO vessel
        
    tank_size_lsfo : float
        Nominal tank size for the LSFO vessel
        
    tank_size_factors_dict : Dictionary
        Dictionary of scaling factors that modify the size of the tank for the given fuel relative to an LSFO tank
        
    mass_density_dict : Dictionary
        Dictionary containing the mass density of each fuel
        
    Returns
    ----------
    modified_capacity : float
        Modified capacity of the vessel, accounting for cargo displacement from the fuel tanks
    """
    tank_size_fuel = tank_size_lsfo * tank_size_factors_dict[fuel]
    
    # For gas carrier, the cargo displacment is just the change in tank size (volume) since the units of capacity are m^3 of natural gas
    if vessel_type == "gas_carrier":
        cargo_displacement = tank_size_fuel - tank_size_lsfo
        
    # For containerships, capacity is measured in TEU (volumetric), so the cargo displacement is the change in tank size measured in TEU
    elif vessel_type == "container":
        cargo_displacement = ( tank_size_fuel - tank_size_lsfo ) / M3_PER_TEU
        
    elif (vessel_type == "bulk_carrier") or (vessel_type == "tanker"):
        cargo_displacement = (tank_size_fuel * mass_density_dict[fuel] * L_PER_M3 - tank_size_lsfo * mass_density_dict["lsfo"] * L_PER_M3) / KG_PER_DWT
        
    modified_capacity = cargo_capacity_lsfo - cargo_displacement
    
    return modified_capacity
    
    
def collect_nominal_capacity(top_dir, type_class_keyword):
    """
    Reads in the vessel .inc file for the given vessel type and size class, and collects the nominal capacity.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
    
    type_class_keyword : str
        Unique keyword in the filename for the given type and class.
        
    Returns
    ----------
    nom_capacity : float
        Nominal capacity of the given vessel, as defined in the vessel .inc file.
    """
    
    # Construct the filepath to the vessel .inc file for the given vessel
    filepath = f"{top_dir}/{VESSELS_ORIG_DIR}/{type_class_keyword}_ice_oil.inc"
    
    # Initialize the nominal capacity variable
    nom_capacity = None
    
    # Define the parse format for the line containing the nominal capacity
    capacity_format = "NominalCapacity = {}"
    
    # Try to open the file and read its content
    try:
        with open(filepath, 'r') as file:
            for line in file:
                # Parse the line using the format string
                result = parse.parse(capacity_format, line.strip())
                if result:
                    nom_capacity = float(result[0])
                    break
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return nom_capacity
    
def collect_nominal_tank_size(top_dir, type_class_keyword):
    """
    Reads in the tank .inc file for the given vessel type and size class, and collects the nominal tank size.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
    
    type_class_keyword : str
        Unique keyword in the filename for the given type and class.
        
    Returns
    ----------
    nom_tank_size : float
        Nominal tank size of the given vessel, as defined in the tank .inc file.
    """
    
    # Construct the filepath to the vessel .inc file for the given vessel
    filepath = f"{top_dir}/{TANKS_ORIG_DIR}/main_oil_{type_class_keyword}.inc"
    
    # Initialize the nominal tank size variable
    nom_tank_size = None
    
    # Define the parse format for the line containing the nominal tank size
    tank_size_format = "Size={}"
    
    # Try to open the file and read its content
    try:
        with open(filepath, 'r') as file:
            for line in file:
                # Parse the line using the format string
                result = parse.parse(tank_size_format, line.strip())
                if result:
                    nom_tank_size = float(result[0])
                    break
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return nom_tank_size
    
def make_modified_capacities_df(top_dir, vessels, fuels, LHV_dict, mass_density_dict):
    """
    Creates a DataFrame containing the nominal and modified capacities for each vessel and fuel combination.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
    
    vessels : dict
        Dictionary containing vessel types and their associated vessel size classes.

    fuels : list
        List of fuels to include in the calculations.

    LHV_dict : dict
        Dictionary containing the LHV values for each fuel.

    mass_density_dict : dict
        Dictionary containing the mass density values for each fuel.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ["Vessel", "Vessel Type", "Fuel", "Nominal Capacity", "Modified Capacity", "% Difference"]
    """
    
    # Create an empty list to hold rows for the DataFrame
    data = []
    
    # Get the tank size scaling factors
    tank_size_factors_dict = get_tank_size_factors(fuels, LHV_dict, mass_density_dict)
    
    # Loop through each vessel type and size
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            # Collect the nominal capacity for the LSFO vessel
            nominal_capacity_lsfo = collect_nominal_capacity(top_dir, vessel_class)
            
            # Collect the nominal tank size for the LSFO vessel
            tank_size_lsfo = collect_nominal_tank_size(top_dir, vessel_class)
            
            # Loop through each fuel
            for fuel in fuels:
                # Calculate the modified cargo capacity for the current fuel
                modified_capacity = modify_cargo_capacity(
                    vessel_type=vessel_type,
                    fuel=fuel,
                    cargo_capacity_lsfo=nominal_capacity_lsfo,
                    tank_size_lsfo=tank_size_lsfo,
                    tank_size_factors_dict=tank_size_factors_dict,
                    mass_density_dict=mass_density_dict
                )
                
                # Calculate the percentage difference
                percent_difference = 100 * (modified_capacity - nominal_capacity_lsfo) / nominal_capacity_lsfo
                
                # Append the data to the list
                data.append({
                    "Vessel": vessel_class,
                    "Vessel Type": vessel_type,
                    "Fuel": fuel,
                    "Nominal Capacity": nominal_capacity_lsfo,
                    "Modified Capacity": modified_capacity,
                    "% Difference": percent_difference
                })
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    
    return df
    

def plot_vessel_capacities(modified_capacities_df):
    """
    Plots vertical bar plots of nominal and modified capacities for each Vessel Type and Fuel,
    with percentage difference plotted as markers below the bars.
    
    Parameters
    ----------
    modified_capacities_df : pd.DataFrame
        DataFrame containing the columns: ["Vessel", "Vessel Type", "Fuel", "Nominal Capacity", "Modified Capacity", "% Difference"]
    """
    
    # Get the unique vessel types from the DataFrame
    vessel_types = modified_capacities_df["Vessel Type"].unique()

    # Define the colors for each fuel, consistent across all vessels
    fuel_colors = {
        "ammonia": "blue",
        "methanol": "green",
        "FTdiesel": "orange",
        "liquid_hydrogen": "red",
        "compressed_hydrogen": "purple"
    }

    # Loop through each vessel type to plot
    for vessel_type in vessel_types:
        # Filter the dataframe for the current vessel type
        df_vessel_type = modified_capacities_df[modified_capacities_df["Vessel Type"] == vessel_type]
        
        # Get the unique vessels for this vessel type
        vessels = df_vessel_type["Vessel"].unique()

        # Set up the figure and axis for the bar plots and percentage difference panel
        fig, (ax_bars, ax_diff) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(14, 8), sharex=True
        )
        
        plt.subplots_adjust(right=0.7)
        
        # Set font size for the x-ticks and y-ticks on ax_bars
        ax_bars.tick_params(axis='x', labelsize=20)
        ax_bars.tick_params(axis='y', labelsize=20)

        # Set font size for the x-ticks and y-ticks on ax_diff
        ax_diff.tick_params(axis='x', labelsize=20)
        ax_diff.tick_params(axis='y', labelsize=20)
        
        # Initialize the x position
        x_pos = np.arange(len(vessels)) * 6  # Give some space between vessels
        
        bar_width = 1  # Width of each bar for fuel
        
        # Loop through each vessel and plot its bars and markers for each fuel
        for i, vessel in enumerate(vessels):
            # Get the data for the current vessel
            df_vessel = df_vessel_type[df_vessel_type["Vessel"] == vessel]
            
            # Plot bars for each fuel
            for j, fuel in enumerate(fuel_colors.keys()):
                # Get the data for the specific fuel for this vessel
                df_fuel = df_vessel[df_vessel["Fuel"] == fuel]

                if not df_fuel.empty:
                    # Calculate the x position for the current fuel bar within the cluster
                    x_bar = x_pos[i] + j * bar_width
                    
                    # Plot the solid bar for the nominal capacity
                    ax_bars.bar(x_bar, df_fuel["Nominal Capacity"].values[0],
                                width=bar_width, color=fuel_colors[fuel], label=get_fuel_label(fuel) if i == 0 else "", alpha=0.7, edgecolor='black')
                    
                    # Plot the hatched bar for the modified capacity overlaid
                    ax_bars.bar(x_bar, df_fuel["Modified Capacity"].values[0],
                                width=bar_width, color='none', edgecolor='black', hatch='xxx')

                    # Plot the % difference marker on the lower axis
                    ax_diff.plot(x_bar, df_fuel["% Difference"].values[0],
                                 marker='o', color=fuel_colors[fuel], markersize=8)
                    ax_diff.hlines(df_fuel["% Difference"].values[0],
                                   x_bar - bar_width / 2, x_bar + bar_width / 2, color=fuel_colors[fuel])

        # Adjust the layout of the bar axis
        ax_bars.set_ylabel("Capacity", fontsize=22)
        ax_bars.set_title(f"{vessel_type_title[vessel_type]}", fontsize=24)

        # Create the first legend for fuel types
        fuel_legend = ax_bars.legend(title="Fuel", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=18, title_fontsize=22)

        # Create custom legend handles for nominal and modified capacities
        solid_patch = mpatches.Patch(facecolor='grey', edgecolor='black', label='Nominal Capacity', alpha=0.7)
        hatched_patch = mpatches.Patch(facecolor='none', edgecolor='black', hatch='xxx', label='Modified Capacity')
        
        # Add the second legend for nominal and modified capacities
        capacity_legend = ax_bars.legend(handles=[solid_patch, hatched_patch], bbox_to_anchor=(1.01, 0.3), loc='upper left', fontsize=18, title_fontsize=22)
        
        # Add the fuel legend back to avoid being overwritten by the second legend
        ax_bars.add_artist(fuel_legend)

        # Adjust the layout of the % difference panel
        ax_diff.set_ylabel("% Diff", fontsize=22)
        ax_diff.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax_diff.set_ylim([modified_capacities_df[modified_capacities_df["Vessel Type"] == vessel_type]["% Difference"].min() - 5,
                          modified_capacities_df[modified_capacities_df["Vessel Type"] == vessel_type]["% Difference"].max() + 5])

        # Set the x-axis ticks and labels
        ax_diff.set_xticks(x_pos + (bar_width * len(fuel_colors) / 2 - bar_width / 2))
        vessel_labels = [vessel_size_title[vessel] for vessel in vessels]
        ax_diff.set_xticklabels(vessel_labels, rotation=0, ha="center", fontsize=22)
        
        #plt.tight_layout()
        plt.savefig(f"plots/modified_capacities_{vessel_type}.png", dpi=300)
        
def make_modified_vessel_incs(top_dir, vessels, fuels, fuel_vessel_dict, modified_capacities_df):
    """
    Modifies vessel .inc files by updating the "NominalCapacity" with the modified capacity for each fuel and vessel,
    and also updates the "Vessel" line to use the correct fuel name.

    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.

    vessels : dict
        Dictionary containing vessel types and their associated vessel size classes.

    fuels : list
        List of fuels to include in the calculations.

    fuel_vessel_dict : dict
        Dictionary containing the keyword for the given fuel in vessel file names.

    modified_capacities_df : pd.DataFrame
        DataFrame containing the columns: ["Vessel", "Vessel Type", "Fuel", "Nominal Capacity", "Modified Capacity", "% Difference"]
    """
    
    # Loop through each vessel type and vessel size class
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            
            # Loop through each fuel
            for fuel in fuels:
                # Define the original and modified file paths
                original_filepath = f"{top_dir}/{VESSELS_ORIG_DIR}/{vessel_class}_ice_{fuel_vessel_dict[fuel]}.inc"
                modified_filepath = f"{top_dir}/{VESSELS_MODIFIED_DIR}/{vessel_class}_ice_{fuel}.inc"

                try:
                    # Read the original .inc file
                    with open(original_filepath, 'r') as file:
                        content = file.readlines()

                    # Get the modified capacity for this vessel and fuel combination from the DataFrame
                    modified_capacity = modified_capacities_df[
                        (modified_capacities_df["Vessel"] == vessel_class) &
                        (modified_capacities_df["Fuel"] == fuel)
                    ]["Modified Capacity"].values[0]

                    # Loop through the lines and make necessary updates
                    for i, line in enumerate(content):
                        # Update the "NominalCapacity" line
                        if line.strip().startswith("NominalCapacity"):
                            content[i] = f"    NominalCapacity = {modified_capacity}\n"
                        
                        # Update the "Vessel" line to replace the fuel keyword with the actual fuel name
                        if line.strip().startswith("Vessel"):
                            content[i] = f'Vessel "{vessel_class}_ice_{fuel}" {{\n'
                            
                        # Update the "Tanks" line
                        if line.strip().startswith("Tanks"):
                            content[i] = f'    Tanks = [Tank("main_{fuel}_{vessel_class}"), Tank("pilot_oil_bulk_carrier_capesize")]\n'

                    # Write the modified content to the new file
                    os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
                    with open(modified_filepath, 'w') as file:
                        file.writelines(content)
                        
                    print(f"Modified file written to: {modified_filepath}")
                
                except FileNotFoundError:
                    print(f"File not found: {original_filepath}")
                except Exception as e:
                    print(f"An error occurred while processing {original_filepath}: {e}")


def make_modified_tank_incs(top_dir, vessels, fuels, fuel_vessel_dict, modified_capacities_df):
    """
    Modifies tank .inc files by updating the "Size" with the modified tank size for each fuel and vessel.
    
    Parameters
    ----------
    top_dir : str
        The top directory where tank files are located.

    vessels : dict
        Dictionary containing vessel types and their associated vessel size classes.

    fuels : list
        List of fuels to include in the calculations.

    fuel_vessel_dict : dict
        Dictionary containing the keyword for the given fuel in vessel file names.

    modified_capacities_df : pd.DataFrame
        DataFrame containing the columns: ["Vessel", "Vessel Type", "Fuel", "Nominal Capacity", "Modified Capacity", "% Difference"]
    """
    
    # Loop through each vessel type and vessel size class
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            
            # Loop through each fuel
            for fuel in fuels:
                # Define the original and modified file paths for the tank .inc file
                original_filepath = f"{top_dir}/{TANKS_ORIG_DIR}/main_{fuel_vessel_dict[fuel]}_{vessel_class}.inc"
                modified_filepath = f"{top_dir}/{TANKS_MODIFIED_DIR}/main_{fuel}_{vessel_class}.inc"

                try:
                    # Read the original .inc file
                    with open(original_filepath, 'r') as file:
                        content = file.readlines()

                    # Get the modified tank size for this vessel and fuel combination from the DataFrame
                    modified_tank_size = modified_capacities_df[
                        (modified_capacities_df["Vessel"] == vessel_class) &
                        (modified_capacities_df["Fuel"] == fuel)
                    ]["Modified Capacity"].values[0]

                    # Replace the "Size" line with the modified tank size
                    for i, line in enumerate(content):
                        # Update the "Size" line
                        if line.strip().startswith("Size"):
                            content[i] = f"    Size={modified_tank_size}\n"
                            break
                            
                        # Update the "Tank" line
                        if line.strip().startswith("Tank"):
                            content[i] = f'Tank "main_{fuel}_{vessel_class}" {{\n'
                            break

                    # Write the modified content to the new file
                    os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
                    with open(modified_filepath, 'w') as file:
                        file.writelines(content)
                        
                    print(f"Modified tank file written to: {modified_filepath}")
                
                except FileNotFoundError:
                    print(f"File not found: {original_filepath}")
                except Exception as e:
                    print(f"An error occurred while processing {original_filepath}: {e}")



def main():
    # List of fuels to consider
    fuels = ["ammonia", "methanol", "FTdiesel", "liquid_hydrogen", "compressed_hydrogen"]

    top_dir = get_top_dir()
    mass_density_dict = get_fuel_info_dict(f"{top_dir}/info_files/fuel_info.csv", "Mass density (kg/L)")
    
    LHV_dict = get_fuel_info_dict(f"{top_dir}/info_files/fuel_info.csv", "Lower Heating Value (MJ / kg)")
    
    tank_size_factors_dict = get_tank_size_factors(fuels, LHV_dict, mass_density_dict)
    plot_tank_size_factors(tank_size_factors_dict)

    modified_capacities_df = make_modified_capacities_df(top_dir, vessels, fuels, LHV_dict, mass_density_dict)
    plot_vessel_capacities(modified_capacities_df)
    
    make_modified_vessel_incs(top_dir, vessels, fuels, fuel_vessel_dict, modified_capacities_df)
    make_modified_tank_incs(top_dir, vessels, fuels, fuel_vessel_dict, modified_capacities_df)
    
if __name__ == "__main__":
    main()

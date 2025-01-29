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
from parse import search
import numpy as np
import matplotlib.patches as mpatches
import re
from scipy.interpolate import RegularGridInterpolator, interp1d

M3_PER_TEU = 38.28
L_PER_M3 = 1000
KG_PER_DWT = 1000
VESSELS_DIR_NAVIGATE = "NavigaTE/navigate/defaults/installation/Vessel"
VESSELS_DIR_LOCAL = "includes_global/vessels_orig_capacity"
VESSELS_MODIFIED_DIR = "includes_global/vessels_modified_capacity"
TANKS_DIR_NAVIGATE = "NavigaTE/navigate/defaults/installation/Tank"
TANKS_DIR_LOCAL = "includes_global/tanks_orig_size"
TANKS_MODIFIED_DIR = "includes_global/tanks_modified_size"
PROPULSION_EFF_DIR_NAVIGATE="NavigaTE/navigate/defaults/installation/Forecast"
PROPULSION_EFF_DIR_LOCAL="includes_global/vessels_orig_capacity"
ROUTE_DIR_NAVIGATE = "NavigaTE/navigate/defaults/installation/Route"
SURFACE_DIR_NAVIGATE = "NavigaTE/navigate/defaults/installation/Surface"
CURVE_DIR_NAVIGATE = "NavigaTE/navigate/defaults/installation/Curve"
S_PER_MIN = 60
MIN_PER_H = 60
H_PER_DAY = 24
SECONDS_PER_HOUR = 3600
DAYS_PER_YEAR = 365.25
HOURS_PER_DAY = 24

# Dictionary containing the keyword for the given fuel in vessel file names
fuel_vessel_dict = {
    "ammonia": "ammonia",
    "methanol": "methanol",
    "FTdiesel": "diesel",
    "compressed_hydrogen": "hydrogen",
    "liquid_hydrogen": "hydrogen",
    "lsfo": "oil",
}

# Dictionary to indicate whether to look for input files in the default NavigaTE inputs, or in the relevant local dir with custom inputs
input_file_types = {
    "ammonia": "NavigaTE",
    "methanol": "NavigaTE",
    "diesel": "local",
    "hydrogen": "local",
    "oil": "NavigaTE",
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
    "bulk_carrier": "Bulk Carrier",
    "container": "Container",
    "tanker": "Tanker",
    "gas_carrier": "Gas Carrier",
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
    
def get_tank_size_factor_energy(LHV_lsfo, mass_density_lsfo, LHV_fuel, mass_density_fuel):
    """
    Calculates the multiplying factor to the tank needed to provide equivalent fuel energy to the vessel as LSFO

    Parameters
    ----------

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
    tank_size_factor : float
        Size of the tank with an equivalent fuel energy to the LSFO tank
    """
    
    tank_size_factor = ( LHV_lsfo * mass_density_lsfo ) / ( LHV_fuel * mass_density_fuel )
    
    return tank_size_factor
    
def get_tank_size_factor_propulsion_eff(propulsion_eff_lsfo, propulsion_eff_fuel):
    """
    Calculates the multiplicative scaling factor to the tank size needed to account for a different engine efficiency relative to LSFO

    Parameters
    ----------
    propulsion_eff_lsfo : float
        Efficiency of an engine running on LSFO

    propulsion_eff_fuel : float
        Efficiency of an engine running on the given fuel

    Returns
    -------
    tank_size_factor : float
        Tank size scaling factor needed to correct for the different propulsion efficiency relative to LSFO
    """
    tank_size_factor = propulsion_eff_lsfo / propulsion_eff_fuel
    
    return tank_size_factor
    
def get_route_properties(top_dir, type_class_keyword):
    """
    Fetches relevant route properties for the given vessel type and class.
    
    Parameters
    ----------
    type_class_keyword : str
        Unique keyword in the filename for the given type and class.

    Returns
    -------
    route_properties_dict : dict
        Dictionary containing the route properties for the given vessel and size.
    """
    
    # Construct the filepath to the vessel .inc file for the given vessel
    filepath = f"{top_dir}/{ROUTE_DIR_NAVIGATE}/globalized_{type_class_keyword}.inc"
    
    # Initialize dictionary to store the extracted properties
    route_properties_dict = {}
    
    # Define regex patterns for flexible matching
    time_at_sea_pattern = re.compile(r"TimeAtSea\s*=\s*([\d.]+)")
    condition_distribution_pattern = re.compile(r"ConditionDistribution\s*=\s*\[([\d.,\s]+)\]")
    speeds_pattern = re.compile(r"Speeds\s*=\s*\[([\d.,\s]+)\]")
    capacity_utilization_pattern = re.compile(r"CapacityUtilizations\s*=\s*\[([\d.,\s]+)\]")
    
    # Read and parse the file
    try:
        with open(filepath, 'r') as file:
            content = file.readlines()
            
            for line in content:
                # Ignore lines that start with '#' or remove comments within lines
                line = line.split('#')[0].strip()
                if not line:
                    continue
                
                # Extract TimeAtSea
                time_at_sea_match = time_at_sea_pattern.search(line)
                if time_at_sea_match:
                    route_properties_dict['TimeAtSea'] = float(time_at_sea_match.group(1))
                
                # Extract ConditionDistribution
                condition_distribution_match = condition_distribution_pattern.search(line)
                if condition_distribution_match:
                    route_properties_dict['ConditionDistribution'] = np.asarray([
                        float(value.strip()) for value in condition_distribution_match.group(1).split(',')
                    ])
                
                # Extract Speeds
                speeds_match = speeds_pattern.search(line)
                if speeds_match:
                    route_properties_dict['Speeds'] = np.asarray([
                        float(value.strip()) for value in speeds_match.group(1).split(',')
                    ])
                
                # Extract CapacityUtilizations
                capacity_utilization_match = capacity_utilization_pattern.search(line)
                if capacity_utilization_match:
                    route_properties_dict['CapacityUtilizations'] = np.asarray([
                        float(value.strip()) for value in capacity_utilization_match.group(1).split(',')
                    ])
                
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
        return None
    
    return route_properties_dict

    
def calculate_propulsion_power_distribution(speed_distribution, utilization_distribution, type_class_keyword):
    """
    Calculates the distribution of propulsion power for a given vessel type and size class with respect to speed and capacity utilization
    
    Parameters
    ----------
    speed_distribution : numpy array
        Distribution of vessel speeds
        
    utilization_distribution : numpy array
        Distribution of capacity utilization

    Returns
    -------
    propulsion_power_distribution : numpy array
        Distribution of propulsion powers
    """
    
    #print(f"Speed distribution: {speed_distribution}")
    #print(f"Utilization distribution: {utilization_distribution}")
    
    top_dir = get_top_dir()
    
    # Attempt to read in a speed-draft-power surface.
    try:
        filepath = f"{top_dir}/{SURFACE_DIR_NAVIGATE}/speed_draft_power_surface_{type_class_keyword}.inc"
        
        with open(filepath, 'r') as file:
            lines = file.readlines()
            # Find the starting line after the header that contains 'Table'
            start_idx = next(i for i, line in enumerate(lines) if "Table" in line) + 1
            
            speeds = []
            power_values = []
            
            # Process the header to extract drafts
            header_parts = lines[start_idx].split('#')[0].strip().split()
            drafts = [float(value) for value in header_parts]  # Skip the first column (x-axis label)
            
            # Process the subsequent rows to extract speeds and power values
            for line in lines[start_idx + 1:]:
                # Ignore comments and empty lines
                line = line.split('#')[0].strip()
                if not line:
                    continue
                parts = line.split()
                
                if len(parts) > 1:
                    speeds.append(float(parts[0]))  # First column is the speed
                    power_values.append([float(val) for val in parts[1:]])  # Remaining columns are power values
            
            speeds = np.array(speeds)
            drafts = np.array(drafts)
            power_values = np.array(power_values)
            
            expected_shape = (len(speeds), len(drafts))
            if power_values.shape != expected_shape:
                raise ValueError(f"Expected power_values shape {expected_shape}, but got {power_values.shape}")
            
            # Create a 2D interpolator for the speed-draft-power surface
            interp_func = RegularGridInterpolator((speeds, drafts), power_values, method='linear')
            
            # Calculate propulsion power distribution
            propulsion_power_distribution = np.array([
                interp_func([spd, util]) for spd, util in zip(speed_distribution, utilization_distribution)
            ])
            
            return propulsion_power_distribution
        
    except FileNotFoundError:
        # Fallback to the speed-power curve
        filepath = f"{top_dir}/{CURVE_DIR_NAVIGATE}/speed_power_curve_{type_class_keyword}.inc"
        
        with open(filepath, 'r') as file:
            speed = []
            power = []
            
            for line in file:
                # Ignore comments and empty lines
                line = line.split('#')[0].strip()
                if not line:
                    continue
                parts = line.split()
                
                if len(parts) == 2 and all(p.replace('.', '', 1).isdigit() for p in parts):
                    speed.append(float(parts[0]))
                    power.append(float(parts[1]))
            
            # Create a 1D interpolator for the speed-power curve
            interp_func = interp1d(speed, power, kind='linear', fill_value="extrapolate")
            
            # Calculate propulsion power distribution
            propulsion_power_distribution = np.array([interp_func(spd) for spd in speed_distribution])
            
            return propulsion_power_distribution
    
    return None


def calculate_average_propulsion_power(type_class_keyword, route_properties_dict):
    """
    Calculates the average propulsion power of a given vessel type and size class over all voyage conditions
    Units: MW
    
    Parameters
    ----------
    type_class_keyword : str
        Unique keyword in the filename for the given type and class.
        
    route_properties_dict : dict
        Dictionary containing the route properties for the given vessel and size.

    Returns
    -------
    average_propulsion_power : float
        Average propulsion power over all voyage conditions
    """
    
    propulsion_power_distribution = calculate_propulsion_power_distribution(route_properties_dict["Speeds"], route_properties_dict["CapacityUtilizations"], type_class_keyword)
    
    average_propulsion_power = np.sum(route_properties_dict["ConditionDistribution"] * propulsion_power_distribution)
    
    return average_propulsion_power
    
def calculate_average_propulsion_energy_per_distance(type_class_keyword, route_properties_dict):
    """
    Calculates the average propulsion power divided by speed for a given vessel type and size class over all voyage conditions.
    Units: MJ / nautical miles
    
    Parameters
    ----------
    type_class_keyword : str
        Unique keyword in the filename for the given type and class.
        
    route_properties_dict : dict
        Dictionary containing the route properties for the given vessel and size.

    Returns
    -------
    average_propulsion_power_over_speed : float
        Average propulsion power divided by vessel speed over all voyage conditions
    """
    
    propulsion_power_distribution = calculate_propulsion_power_distribution(route_properties_dict["Speeds"], route_properties_dict["CapacityUtilizations"], type_class_keyword)
    
    # Propulsion power has units of MW (MJ/s) and speed has units of nautical miles / h, so need to multiply by SECONDS_PER_HOUR to convert to MJ / nm
    average_propulsion_energy_per_distance = np.sum(route_properties_dict["ConditionDistribution"] * propulsion_power_distribution / route_properties_dict["Speeds"]) * SECONDS_PER_HOUR
    
    return average_propulsion_energy_per_distance
    
def calculate_fuel_usage_rate(fuel, LHV_fuel, propulsion_eff_fuel, type_class_keyword, route_properties_dict):
    """
    Calculates the usage rate of a given fuel in a given vessel type and size class
    
    Parameters
    ----------
    fuel : str
        Name of the fuel
        
    LHV_fuel : float
        Lower heating value of the given fuel (in MJ / kg)
        
    propulsion_eff_fuel : float
        Efficiency of an engine running on the given fuel
        
    type_class_keyword : str
        Unique keyword in the filename for the given type and class.
        
    route_properties_dict : dict
        Dictionary containing the route properties for the given vessel and size.

    Returns
    -------
    fuel_usage_rate : float
        Usage rate of the given fuel in the given vessel type and size class
    """
    
    # Calculate the average propulsion power, in MW
    average_propulsion_power = calculate_average_propulsion_power(type_class_keyword, route_properties_dict)
    
    # Use the propulsion efficiency and the fuel LHV to convert to kg/s
    fuel_usage_rate = average_propulsion_power / (propulsion_eff_fuel * LHV_fuel)
    
    # Convert to kg per day
    fuel_usage_rate = fuel_usage_rate * (S_PER_MIN * MIN_PER_H * H_PER_DAY)
    
    return fuel_usage_rate
    
    
def calculate_days_to_empty_tank(fuel, tank_size_lsfo, tank_size_factors_dict, mass_density_fuel, propulsion_eff_fuel, LHV_fuel, type_class_keyword):
    """
    Calculate the number of days it would take to empty the fuel tank, neglecting boil-off
    
    Parameters
    ----------
    fuel : str
        Name of the fuel
        
    tank_size_lsfo : float
        Nominal size of the tank for an LSFO vessel

    tank_size_factors_dict : Dictionary
        Dictionary containing tank size scaling factors for the given fuel, neglecting boil-off
        
    mass_density_fuel : float
        Mass density of the given fuel (in kg/L)

    propulsion_eff_fuel : float
        Efficiency of an engine running on the given fuel
        
    LHV_fuel : float
        Lower heating value of the given fuel (in MJ / kg)
        
    type_class_keyword : str
        Unique keyword in the filename for the given type and class.

    Returns
    -------
    N_days : float
        Number of days it would take to empty the tank without boil-off
    """
    
    top_dir = get_top_dir()
    
    # Collect the route properties for the given vessel
    route_properties_dict = get_route_properties(top_dir, type_class_keyword)
    
    # Get the corrected tank size, measured in m^3 of fuel
    tank_size_corrected = tank_size_lsfo
    for key in tank_size_factors_dict[fuel][type_class_keyword]:
        tank_size_corrected = tank_size_corrected * tank_size_factors_dict[fuel][type_class_keyword][key]

    #print(f"Corrected tank size: {tank_size_corrected} m^3")
    
    # Multiply by the mass density of the fuel to get the mass of fuel in the tank
    mass_density_fuel_per_m3 = mass_density_fuel * L_PER_M3     # Convert mass density from kg/L to kg/m^3
    m_fuel_tank = tank_size_corrected * mass_density_fuel_per_m3       # Mass of fuel in the tank, in kg

    # Divide the mass of fuel by the average daily fuel usage rate (in kg/day) to get the number of days at sea
    fuel_usage_rate = calculate_fuel_usage_rate(fuel, LHV_fuel, propulsion_eff_fuel, type_class_keyword, route_properties_dict)
    #print(f"Mass of fuel tank: {m_fuel_tank} kg")
    #print(f"Fuel usage rate (kg/day): {fuel_usage_rate}")
    
    N_days_at_sea = m_fuel_tank / fuel_usage_rate

    # Calculate the number of days at port based on the average fraction of time the vessel spends at port vs. at sea
    TimeAtPort = 1 - route_properties_dict["TimeAtSea"]
    N_days_at_port = N_days_at_sea * TimeAtPort / route_properties_dict["TimeAtSea"]
    
    N_days = N_days_at_sea + N_days_at_port
    
    return N_days, N_days_at_sea, N_days_at_port
    
def get_tank_size_factor_boiloff(boiloff_rate, N_days):
    """
    Calculates the multiplicative scaling factor to the tank size needed to account for boil-off of liquefied fuels.

    Parameters
    ----------
    boiloff_rate : float
        Average boil-off rate of the fuel, in %/day

    N_days : float
        Number of days it would take to empty the tank without boil-off

    Returns
    -------
    tank_size_fuel : float
        Tank size scaling factor needed to correct for fuel loss due to boil-off
    """
    
    # Convert the % boil-off rate to a fractional rate relative to 1
    boiloff_rate_rel = boiloff_rate / 100
    
    # The tank size factor gets incrased by a factor of 1/(1 - boiloff_rate_rel) for every day that the boil-off occurs
    tank_size_factor = 1 / (1 - boiloff_rate_rel)**N_days
    
    return tank_size_factor
    
def get_tank_size_factors(fuels, LHV_dict, mass_density_dict, propulsion_eff_dict, boiloff_rate_dict, vessel_range=None):
    """
    Creates a dictionary of tank size scaling factors for each fuel relative to LSFO
    
    Parameters
    ----------
    fuels : list of str
        List of fuels to include as keys in the dictionary
        
    LHV_dict : dictionary of float
        Dictionary containing the lower heating value for each fuel
        
    mass_density_dict : dictionary of floatf
        Dictionary containing the mass density of each fuel
        
    propulsion_eff_dict : dictionary of float
        Dictionary containing the engine efficiency for each fuel

    boiloff_rate_dict : dictionary of float
        Dictionary containing the boiloff rate for each fuel
        
    vessel_range: Design range of the vessel, in nautical miles

    Returns
    -------
    tank_size_factors_dict : Dictionary
        Dictionary containing the tank size factor for each fuel
        
    days_to_empty_tank_dict : Dictionary
        Dictionary containing the number of days it takes to deplete a tank without accounting for boiloff
    """
    top_dir = get_top_dir()
    LHV_lsfo = LHV_dict["lsfo"]
    mass_density_lsfo = mass_density_dict["lsfo"]
    propulsion_eff_lsfo = propulsion_eff_dict["lsfo"]
    
    tank_size_factors_dict = {}
    days_to_empty_tank_dict = {}
    for fuel in fuels:
        tank_size_factors_dict[fuel] = {}
        days_to_empty_tank_dict[fuel] = {}
        # Loop through each vessel type and size
        for vessel_type, vessel_classes in vessels.items():
            for vessel_class in vessel_classes:
            
                if vessel_range is None:
                    tank_size_lsfo = collect_tank_size(top_dir, vessel_class)
                else:
                    tank_size_lsfo = get_lsfo_tank_size(vessel_range, vessel_class)
                #print(f"Tank size: {tank_size_lsfo}")
            
                #print(f"\n\nFuel: {fuel}")
                #print(f"Vessel class: {vessel_class}")
                tank_size_factors_dict[fuel][vessel_class] = {}
                days_to_empty_tank_dict[fuel][vessel_class] = {}
                LHV_fuel = LHV_dict[fuel]
                mass_density_fuel = mass_density_dict[fuel]
                propulsion_eff_fuel = propulsion_eff_dict[fuel]
                boiloff_rate_fuel = boiloff_rate_dict[fuel]
                tank_size_factors_dict[fuel][vessel_class]["Consistent Energy Density"] = get_tank_size_factor_energy(LHV_lsfo, mass_density_lsfo, LHV_fuel, mass_density_fuel)
                tank_size_factors_dict[fuel][vessel_class]["Engine Efficiency"] = get_tank_size_factor_propulsion_eff(propulsion_eff_lsfo, propulsion_eff_fuel)
                
                days_to_empty_tank, days_at_sea, days_at_port = calculate_days_to_empty_tank(fuel, tank_size_lsfo, tank_size_factors_dict, mass_density_fuel, propulsion_eff_fuel, LHV_fuel, vessel_class)
                days_to_empty_tank_dict[fuel][vessel_class]["Days at Sea"] = days_at_sea
                days_to_empty_tank_dict[fuel][vessel_class]["Days at Port"] = days_at_port
                days_to_empty_tank_dict[fuel][vessel_class]["Total Days"] = days_to_empty_tank
                
                tank_size_factors_dict[fuel][vessel_class]["Boil-off"] = get_tank_size_factor_boiloff(boiloff_rate_fuel, days_to_empty_tank)
                tank_size_factors_dict[fuel][vessel_class]["Total"] = tank_size_factors_dict[fuel][vessel_class]["Consistent Energy Density"] * tank_size_factors_dict[fuel][vessel_class]["Engine Efficiency"] * tank_size_factors_dict[fuel][vessel_class]["Boil-off"]
                        
    return tank_size_factors_dict, days_to_empty_tank_dict
    
def plot_tank_size_factors_boiloff(tank_size_factors_dict, days_to_empty_tank_dict):
    """
    Plots the days to empty tank and resulting boil-off tank size factor for each fuel and vessel
    
    Parameters
    ----------
    tank_size_factors_dict : Dictionary
        Dictionary containing the tank size factor for each fuel
        
    days_to_empty_tank_dict : Dictionary
        Dictionary containing the number of days it takes to deplete a tank without accounting for boiloff

    Returns
    -------
    None
    """
    
    # Define colors for fuels
    fuel_colors = {
        "ammonia": "blue",
        "methanol": "green",
        "FTdiesel": "orange",
        "liquid_hydrogen": "red",
        "compressed_hydrogen": "purple"
    }

    # Plot both Boil-off Tank Size Factor and Days to Empty Tank side by side
    for fuel, vessel_data in tank_size_factors_dict.items():
        
        vessel_labels = []
        boiloff_factors = []
        days_at_sea = []
        days_at_port = []
        
        # Prepare data and labels for plotting
        for vessel_type, vessel_sizes in vessels.items():
            for vessel_size in vessel_sizes:
                # Label format: "Vessel Type (Vessel Size)"
                vessel_labels.append(f"{vessel_type_title[vessel_type]} ({vessel_size_title[vessel_size]})")
                boiloff_factors.append(vessel_data[vessel_size]['Boil-off'])
                days_at_sea.append(days_to_empty_tank_dict[fuel][vessel_size]['Days at Sea'])
                days_at_port.append(days_to_empty_tank_dict[fuel][vessel_size]['Days at Port'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        fig.suptitle(f"{get_fuel_label(fuel)} Tank Size Factor and Days to Empty Tank", fontsize=22)
        
        # Plot Boil-off Tank Size Factor
        y_pos = np.arange(len(vessel_labels))
        ax1.barh(y_pos, boiloff_factors, color=fuel_colors[fuel], edgecolor='black', alpha=0.7)
        ax1.set_title("Boil-off Tank Size Factor", fontsize=18)
        ax1.set_xlabel("Boil-off Factor", fontsize=18)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(vessel_labels, fontsize=16)
        ax1.invert_yaxis()
        ax1.axvline(1, ls='--', color='black')

        # Plot Days to Empty Tank (stacked bar for Days at Sea and Days at Port)
        ax2.barh(y_pos, days_at_sea, color='skyblue', edgecolor='black', label="Days at Sea")
        ax2.barh(y_pos, days_at_port, left=days_at_sea, color='steelblue', edgecolor='black', label="Days at Port")
        ax2.set_title("Days to Empty Tank", fontsize=18)
        ax2.set_xlabel("Days", fontsize=18)
        ax2.legend(loc="upper right", fontsize=16)

        # Label each bar with values for Days at Sea and Days at Port
        for i, (sea, port) in enumerate(zip(days_at_sea, days_at_port)):
            ax2.text(sea / 2, y_pos[i], f"{sea:.1f}", va="center", ha="center", color="black", fontsize=10)
            ax2.text(sea + port / 2, y_pos[i], f"{port:.1f}", va="center", ha="center", color="white", fontsize=10)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"plots/boiloff_tank_size_factor_{fuel}.png", dpi=300)
        plt.savefig(f"plots/boiloff_tank_size_factor_{fuel}.pdf")
    
def save_tank_size_factors(top_dir, tank_size_factors_dict, vessel_range=None):
    """
    Reformat the tank size factors dict so it can be saved as a csv file.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
        
    tank_size_factors_dict : Dictionary
        Dictionary containing the tank size factor for each fuel
    
    vessel_range : str
        Vessel design range (to modify the filename that the factors get saved to)
        
    Returns
    -------
    None
    """
    
    # Flatten the nested dictionary
    data = []
    for fuel, vessels in tank_size_factors_dict.items():
        for vessel_class, factors in vessels.items():
            row = {"Fuel": fuel, "Vessel Class": vessel_class}
            row.update(factors)  # Add all the factors as columns
            data.append(row)

    # Convert to a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    if vessel_range is None:
        df.to_csv(f"{top_dir}/tables/tank_size_factors.csv", index=False)
    else:
        df.to_csv(f"{top_dir}/tables/tank_size_factors_range_{vessel_range}.csv", index=False)
        
def save_days_to_empty_tank(top_dir, days_to_empty_tank_dict, vessel_range=None):
    """
    Reformat the days to empty tank dict so it can be saved as a csv file.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
        
    days_to_empty_tank_dict : Dictionary
        Dictionary containing the days to empty the tank (both at sea and at port)
    
    vessel_range : str
        Vessel design range (to modify the filename that the factors get saved to)
        
    Returns
    -------
    None
    """
    
    # Flatten the nested dictionary
    data = []
    for fuel, vessels in days_to_empty_tank_dict.items():
        for vessel_class, days in vessels.items():
            row = {"Fuel": fuel, "Vessel Class": vessel_class}
            row.update(days)  # Add all the factors as columns
            data.append(row)

    # Convert to a DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    if vessel_range is None:
        df.to_csv(f"{top_dir}/tables/days_to_empty_tank.csv", index=False)
    else:
        df.to_csv(f"{top_dir}/tables/days_to_empty_tank_{vessel_range}.csv", index=False)
    
def plot_tank_size_factors(tank_size_factors_dict):
    """
    Plots a horizontal bar chart of tank size scaling factors for each fuel, averaging across vessel types and sizes,
    with error bars representing the range from minimum to maximum scaling factors for each correction.

    Parameters
    ----------
    tank_size_factors_dict : dict
        A dictionary with fuel types as keys and dictionaries of tank size scaling factors for each vessel type and size.

    Returns
    -------
    None
    """
    # Set up lists for plotting
    fuels = list(tank_size_factors_dict.keys())
    corrections = list(next(iter(tank_size_factors_dict.values())).values())[0].keys()  # Get corrections from the first vessel entry
    other_corrections = [c for c in corrections if c != 'Total']
    
    colors = ["green", "blue", "magenta"]
    total_color = "red"
    bar_width = 0.5
    buffer_width = 1.0  # Space between groups of bars for different fuels

    # Initialize lists for plotting data
    avg_factors = {corr: [] for corr in corrections}
    error_bars = {corr: [] for corr in corrections}

    # Calculate the average, minimum, and maximum for each correction factor across all vessel types and sizes
    for fuel, vessel_data in tank_size_factors_dict.items():
        for corr in corrections:
            values = [vessel_data[vessel][corr] for vessel in vessel_data]
            avg_factors[corr].append(np.mean(values))
            min_error = np.mean(values) - np.min(values)
            max_error = np.max(values) - np.mean(values)
            error_bars[corr].append([abs(min_error), abs(max_error)])  # Ensure positive values for error bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot 'Total' correction separately
    for i, fuel in enumerate(fuels):
        pos = i * (len(corrections) * bar_width + buffer_width)
        ax.barh(pos, avg_factors['Total'][i], color=total_color, height=bar_width, label='Total' if i == 0 else "")
        ax.errorbar(x=avg_factors['Total'][i], y=pos, xerr=[[error_bars['Total'][i][0]], [error_bars['Total'][i][1]]],
                    fmt='none', ecolor='black', capsize=5)

    # Plot other corrections with unique colors and error bars
    for j, correction in enumerate(other_corrections):
        for i, fuel in enumerate(fuels):
            pos = i * (len(corrections) * bar_width + buffer_width) + (j + 1) * bar_width
            ax.barh(pos, avg_factors[correction][i], color=colors[j], height=bar_width, label=correction if i == 0 else "")
            ax.errorbar(x=avg_factors[correction][i], y=pos,
                        xerr=[[error_bars[correction][i][0]], [error_bars[correction][i][1]]],
                        fmt='none', ecolor='black', capsize=5)

    # Add a reference line at scaling factor 1
    ax.axvline(1, color='black', linestyle='--')

    # Setting labels, title, and ticks
    ax.set_yticks([i * (len(corrections) * bar_width + buffer_width) + bar_width for i in range(len(fuels))])
    ax.set_yticklabels([get_fuel_label(fuel).replace(' ', '\n') for fuel in fuels], fontsize=20)
    ax.tick_params(axis='x', labelsize=18)
    ax.set_xlabel("Tank Size Scaling Factor", fontsize=22)
    ax.legend(title="Correction", fontsize=18, title_fontsize=20)

    plt.tight_layout()
    plt.savefig("plots/tank_size_scaling_factors.png", dpi=300)
    plt.savefig("plots/tank_size_scaling_factors.pdf")
    
    
def modify_cargo_capacity(vessel_class, fuel, cargo_capacity_lsfo, tank_size_lsfo, tank_size_factors_dict, mass_density_dict):
    """
    Calculates the modified cargo capacity based on the change in tank size for the given fuel relative to LSFO.

    Parameters
    ----------
    vessel_class : str
        Filename keyword indicating the vessel type and class, which will determine how the cargo capacity is modified

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
    tank_size_fuel = tank_size_lsfo * tank_size_factors_dict[fuel][vessel_class]["Total"]
    
    # For gas carrier, the cargo displacment is just the change in tank size (volume) since the units of capacity are m^3 of natural gas
    if "gas_carrier" in vessel_class:
        cargo_displacement = tank_size_fuel - tank_size_lsfo
        
    # For containerships, capacity is measured in TEU (volumetric), so the cargo displacement is the change in tank size measured in TEU
    elif "container" in vessel_class:
        cargo_displacement = ( tank_size_fuel - tank_size_lsfo ) / M3_PER_TEU
        
    elif ("bulk_carrier" in vessel_class) or ("tanker" in vessel_class):
        cargo_displacement = (tank_size_fuel * mass_density_dict[fuel] * L_PER_M3 - tank_size_lsfo * mass_density_dict["lsfo"] * L_PER_M3) / KG_PER_DWT
        
    modified_capacity = cargo_capacity_lsfo - cargo_displacement
    
    return modified_capacity
    
    
def collect_nominal_capacity(top_dir, type_class_keyword, fuel="oil", modified_capacity=False):
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
    if modified_capacity:
        filepath = f"{top_dir}/{VESSELS_MODIFIED_DIR}/{type_class_keyword}_ice_{fuel}.inc"
    else:
        filepath = f"{top_dir}/{VESSELS_DIR_NAVIGATE}/{type_class_keyword}_ice_oil.inc"
    
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
    
def collect_propulsion_eff(top_dir, fuel):
    """
    Reads in the propulsion efficiency for the given fuel in 2024 from the relevant NavigaTE input file.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
    
    fuel : str
        Name of the fuel to collect the propulsion efficiency for
        
    Returns
    ----------
    propulsion_eff : float
        Propulsion efficiency for the given fuel in 2024
    """
    
    # Construct the filepath to the vessel .inc file for the given vessel
    fuel_vessel = fuel_vessel_dict[fuel]        # Fuel name as defined in vessel input filenames
    input_type = input_file_types[fuel_vessel]
    if input_type == "local":
        filepath = f"{top_dir}/{VESSELS_DIR_LOCAL}/bulk_carrier_capesize_ice_{fuel_vessel}.inc"
    else:
        filepath = f"{top_dir}/{PROPULSION_EFF_DIR_NAVIGATE}/propulsion_ice_{fuel_vessel}_thermal_efficiency.inc"
    
    # Initialize the variable to contain the propulsion efficiency
    propulsion_eff = None
    
    # Define a regex pattern to match the date "01-01-2024" with any amount of surrounding whitespace and capture the efficiency value
    pattern = r'^\s*"01-01-2024"\s+([\d.]+)'
    
    # Try to open the file and read its content
    try:
        with open(filepath, 'r') as file:
            for line in file:
                match = re.match(pattern, line)
                if match:
                    propulsion_eff = float(match.group(1))
                    break
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return propulsion_eff
    
def collect_tank_size(top_dir, type_class_keyword, fuel="oil", modified_size=False):
    """
    Reads in the tank .inc file for the given vessel type and size class, and collects the nominal tank size.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
    
    type_class_keyword : str
        Unique keyword in the filename for the given type and class.
        
    fuel : str
        Fuel to read the tank size for
        
    modified_size : str
        Boolean variable to indicate whether we're reading the tank size before or after modification
        
    Returns
    ----------
    tank_size : float
        Nominal tank size of the given vessel, as defined in the tank .inc file.
    """
    
    # Construct the filepath to the vessel .inc file for the given vessel
    if modified_size:
        filepath = f"{top_dir}/{TANKS_MODIFIED_DIR}/main_{fuel}_{type_class_keyword}.inc"
    else:
        filepath = f"{top_dir}/{TANKS_DIR_NAVIGATE}/main_oil_{type_class_keyword}.inc"
    
    # Initialize the tank size variable
    tank_size = None
    
    # Define the parse format for the line containing the nominal tank size
    tank_size_format = "Size={}"
    
    # Try to open the file and read its content
    try:
        with open(filepath, 'r') as file:
            for line in file:
                # Parse the line using the format string
                result = parse.parse(tank_size_format, line.strip())
                if result:
                    tank_size = float(result[0])
                    break
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    
    return tank_size
    
def get_propulsion_eff_dict(top_dir, fuels):
    """
    Constructs a dictionary containing the 2024 engine efficiency for each fuel
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
    
    fuels : list of str
        List of fuels to collect engine efficiencies for
        
    Returns
    ----------
    propulsion_eff_dict : dictionary of float
        Dictionary containing the 2024 engine efficiency for each fuel
    """
    propulsion_eff_dict = {}
    for fuel in fuels:
        propulsion_eff_dict[fuel] = collect_propulsion_eff(top_dir, fuel)
    
    return propulsion_eff_dict
        
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
        DataFrame containing the modified cargo capacity for each vessel (by both mass and volume) and related info
    """
    
    # Loop through each vessel type and vessel size class
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            
            # Loop through each fuel
            for fuel in fuels:
                # Define the original and modified file paths
                input_type = input_file_types[fuel_vessel_dict[fuel]]
                if input_type == "local":
                    original_filepath = f"{top_dir}/{VESSELS_DIR_LOCAL}/{vessel_class}_ice_{fuel_vessel_dict[fuel]}.inc"
                else:
                    original_filepath = f"{top_dir}/{VESSELS_DIR_NAVIGATE}/{vessel_class}_ice_{fuel_vessel_dict[fuel]}.inc"
                    
                modified_filepath = f"{top_dir}/{VESSELS_MODIFIED_DIR}/{vessel_class}_ice_{fuel}.inc"

                try:
                    # Read the original .inc file
                    with open(original_filepath, 'r') as file:
                        content = file.readlines()

                    # Get the modified capacity for this vessel and fuel combination from the DataFrame
                    modified_capacity = modified_capacities_df[
                        (modified_capacities_df["Vessel"] == vessel_class) &
                        (modified_capacities_df["Fuel"] == fuel)
                    ]["Modified capacity (m^3)"].values[0]

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
        DataFrame containing the modified cargo capacity for each vessel (by both mass and volume) and related info
    """
    
    # Loop through each vessel type and vessel size class
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            
            # Loop through each fuel
            for fuel in fuels:
                # Define the original and modified file paths for the tank .inc file
                input_type = input_file_types[fuel_vessel_dict[fuel]]
                if input_type == "local":
                    original_filepath = f"{top_dir}/{TANKS_DIR_LOCAL}/main_{fuel_vessel_dict[fuel]}_{vessel_class}.inc"
                else:
                    original_filepath = f"{top_dir}/{TANKS_DIR_NAVIGATE}/main_{fuel_vessel_dict[fuel]}_{vessel_class}.inc"
                modified_filepath = f"{top_dir}/{TANKS_MODIFIED_DIR}/main_{fuel}_{vessel_class}.inc"

                try:
                    # Read the original .inc file
                    with open(original_filepath, 'r') as file:
                        content = file.readlines()

                    # Get the modified tank size for this vessel and fuel combination from the DataFrame
                    modified_tank_size = modified_capacities_df[
                        (modified_capacities_df["Vessel"] == vessel_class) &
                        (modified_capacities_df["Fuel"] == fuel)
                    ]["Modified tank size (m^3)"].values[0]
                    
                    # Replace the "Size" line with the modified tank size
                    for i, line in enumerate(content):
                        # Update the "Size" line
                        if line.strip().startswith("Size"):
                            content[i] = f"    Size={modified_tank_size}\n"
                            break
                            
                        # Update the "Tank" line
                        if line.strip().startswith("Tank"):
                            content[i] = f'Tank "main_{fuel}_{vessel_class}" {{\n'

                    # Write the modified content to the new file
                    os.makedirs(os.path.dirname(modified_filepath), exist_ok=True)
                    with open(modified_filepath, 'w') as file:
                        file.writelines(content)
                        
                    print(f"Modified tank file written to: {modified_filepath}")
                
                except FileNotFoundError:
                    print(f"File not found: {original_filepath}")
                except Exception as e:
                    print(f"An error occurred while processing {original_filepath}: {e}")
                    
def calculate_average_fuel_per_distance(fuel, LHV_fuel, propulsion_eff_fuel, vessel_type_class, route_properties_dict):
    """
    Calculates the fuel consumed per unit distance traveled by the vessel.
    Units: kg / nautical mile
    
    Parameters
    ----------
    fuel : str
        Name of the fuel
        
    LHV_fuel : float
        Lower heating value of the given fuel (in MJ / kg)
        
    propulsion_eff_fuel : float
        Efficiency of an engine running on the given fuel
        
    vessel_type_class : str
        Unique keyword in the filename for the given type and class.

    route_properties_dict : dict
        Dictionary containing the route properties for the given vessel and size.

    Returns
    -------
    fuel_per_distance : float
        Fuel usage (kg) per distance traveled (nautical miles) of the vessel
    """

    # Calculate the average propulsion power over speed, in MW / nautical miles
    average_propulsion_energy_per_distance = calculate_average_propulsion_energy_per_distance(vessel_type_class, route_properties_dict)
    
    # Use the propulsion efficiency and the fuel LHV to convert to kg/nm
    fuel_per_distance = average_propulsion_energy_per_distance / (propulsion_eff_fuel * LHV_fuel)
    
    return fuel_per_distance

def calculate_vessel_range(top_dir, vessel_type_class, fuel, LHV_fuel, boiloff_factor, mass_density_fuel, propulsion_eff_fuel, route_properties_dict):
    """
    Function to calculate the range of a given vessel.
    
    Parameters
    ----------
    vessel_type_class : str
        String specifying the vessel's type and class

    fuel : str
        Fuel to evaluate the range for
        
    boiloff_factor : float
        Factor used to scale up the tank size to compensate for boiloff
        
    mass_density_fuel : float
        Mass density of the fuel, in kg/m^3
        
    route_properties_dict : dict
        Dictionary containing the route properties for the given vessel and size.

    Returns
    -------
    vessel_range : float
        Range of the vessel, in nautical miles
    """
    
    # Collect the modified tank size
    modified_size = False if fuel == "lsfo" else True
    tank_size = collect_tank_size(top_dir, vessel_type_class, fuel, modified_size=modified_size)
    
    # Reduce the available fuel to account for boil-off, if needed
    available_fuel_volume = tank_size / boiloff_factor
    
    # Multiply by the mass density of the fuel to get the mass of fuel in the tank
    m_available_fuel = available_fuel_volume * mass_density_fuel      # Mass of fuel in the tank, in kg
    average_fuel_per_distance = calculate_average_fuel_per_distance(fuel, LHV_fuel, propulsion_eff_fuel, vessel_type_class, route_properties_dict)               # Calculate the fuel use per unit distance traveled, in kg / nautical mile
    vessel_range = m_available_fuel / average_fuel_per_distance
    
    return vessel_range
    
def get_lsfo_tank_size(vessel_range, vessel_type_class):
    """
    Calculates the required size of a tank carrying LSFO fuel corresponding to a specified design range for a given vessel.
    
    Parameters
    ----------
    vessel_range : float
        Range of the vessel, in nautical miles

    vessel_type_class : str
        String specifying the vessel's type and class

    Returns
    -------
    tank_size : float
        Required size of the tank, in m^3
    """
    top_dir = get_top_dir()
    fuel_properties_dict = get_fuel_properties("lsfo")
    route_properties_dict = get_route_properties(top_dir, vessel_type_class)
    propulsion_eff = get_propulsion_eff_dict(top_dir, ["lsfo"])["lsfo"]
    
    average_fuel_per_distance = calculate_average_fuel_per_distance("lsfo", fuel_properties_dict["Lower Heating Value (MJ / kg)"], propulsion_eff, vessel_type_class, route_properties_dict)               # Calculate the fuel use per unit distance traveled, in kg / nautical mile
    
    # Calculate the mass of fuel needed for the given vessel design range
    m_fuel = vessel_range * average_fuel_per_distance
    
    # Calculate the needed tank volume based on the fuel density
    tank_size = (m_fuel / fuel_properties_dict["Mass density (kg/L)"]) / L_PER_M3
    
    return tank_size
    
def get_modified_tank_size(vessel_range, vessel_type_class, fuel):
    """
    Calculates the required size of a tank carrying the given fuel fuel corresponding to a specified design range for a given vessel, modified from LSFO tank size to maintain the same range.
    
    Parameters
    ----------
    vessel_range : float
        Range of the vessel, in nautical miles

    vessel_type_class : str
        String specifying the vessel's type and class
    
    fuel : str
        String specifying the name of the fuel

    Returns
    -------
    modified_tank_size : float
        Required size of the tank, in m^3
    """
    lsfo_tank_size = get_lsfo_tank_size(vessel_range, vessel_type_class)
    
    
    
    return tank_size
    
def calculate_all_vessel_ranges(top_dir, fuels, tank_size_factors_dict, LHV_dict, mass_density_dict, propulsion_eff_dict, boiloff_rate_dict):
    """
    Function to calculate the ranges of all vessels and fuel types, in nautical miles
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.

    Returns
    -------
    ranges_df : pandas DataFrame
        Ranges of all vessels and fuel types
    """
    
    # Loop through each vessel type and vessel size class
    vessel_ranges_df = pd.DataFrame(columns = ["Vessel"] + fuels)
    vessel_range_rows = []
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            range_dict = {}
            range_dict["Vessel"] = f"{vessel_type_title[vessel_type]} ({vessel_size_title[vessel_class]})"
            
            # Collect the route properties for the given vessel
            route_properties_dict = get_route_properties(top_dir, vessel_class)
            
            # Loop through each fuel
            for fuel in ["lsfo"] + fuels:
                boiloff_factor = 1 if fuel == "lsfo" else tank_size_factors_dict[fuel][vessel_class]["Boil-off"]
                fuel_mass_density = mass_density_dict[fuel] * L_PER_M3      # Fuel mass density, converted from kg/L to kg/m^3
                LHV_fuel = LHV_dict[fuel]
                propulsion_eff_fuel = propulsion_eff_dict[fuel]
                vessel_range = calculate_vessel_range(top_dir, vessel_class, fuel, LHV_fuel, boiloff_factor, fuel_mass_density, propulsion_eff_fuel, route_properties_dict)
                range_dict[fuel] = vessel_range
            vessel_range_rows.append(range_dict)
    vessel_ranges_df = pd.concat([vessel_ranges_df, pd.DataFrame(vessel_range_rows)], ignore_index=True)
    return vessel_ranges_df
    
def get_nominal_cargo_capacity_mass_volume(top_dir, cargo_info_df, vessel_type_class):
    """
    Gets the cargo capacity of a vessel in both tonnes and m^3, based on the average density of cargo that it carries
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
        
    vessel_type_class : str
        String specifying the vessel's type and class
        
    fuel : str
        Fuel used by the vessel
        
    cargo_info_df : pandas DataFrame
        Dataframe containing cargo-related info and assumptions for each vessel

    Returns
    -------
    cargo_capacity_tonnes : float
        Cargo capacity by mass, in tonnes
        
    cargo_capacity_cbm : float
        Cargo capacity by volume, in m^3
    """
    
    # Nominal capacity, in whatever units
    nominal_capacity = collect_nominal_capacity(top_dir, vessel_type_class)
    
    cargo_capacity_cbm = None
    cargo_capacity_tonnes = None
    if "bulk" in vessel_type_class:
        # Nominal capacity in NavigaTE is measured in deadweight tonnes
        cargo_capacity_tonnes = nominal_capacity
        
        # Volumetric capacity is read in from the info file, in m^3
        cargo_capacity_cbm = cargo_info_df.loc[cargo_info_df["Vessel class"] == vessel_type_class, "Max capacity (m^3)"].values[0]
        
    elif "container" in vessel_type_class:
        # Nominal capacity in NavigaTE is measured in twenty foot equivalent units (TEUs)
        cargo_capacity_cbm = nominal_capacity * M3_PER_TEU
        
        # Mass cargo capacity is calculated based on the maximum allowed mass density of twenty foot equivalent cargo containers
        cargo_capacity_tonnes = cargo_capacity_cbm * cargo_info_df.loc[cargo_info_df["Vessel class"] == vessel_type_class, "Max density (tonnes/m^3)"].values[0]
        
    elif "tanker" in vessel_type_class:
        # Nominal capacity in NavigaTE is measured in deadweight tonnes
        cargo_capacity_tonnes = nominal_capacity
        
        # Volumetric capacity is evaluated from the mass capacity based on the minimum density of crude oil
        cargo_capacity_cbm = cargo_capacity_tonnes / cargo_info_df.loc[cargo_info_df["Vessel class"] == vessel_type_class, "Min density (tonnes/m^3)"].values[0]
        
    elif "gas_carrier" in vessel_type_class:
        # Nominal capacity in NavigaTE is measured in m^3
        cargo_capacity_cbm = nominal_capacity
        
        # Mass capacity is evaluated from the volumetric capacity baed on the mass density of LNG
        cargo_capacity_tonnes = cargo_capacity_cbm * cargo_info_df.loc[cargo_info_df["Vessel class"] == vessel_type_class, "Max density (tonnes/m^3)"].values[0]
    
    return cargo_capacity_cbm, cargo_capacity_tonnes
    
def calculate_cargo_miles(top_dir, fuel, vessel_class, modified_capacities_df):
    """
    Calculates the cargo miles that a vessel travels annually, either in terms of tonne-miles or m^3-miles
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
        
    fuel : str
        Name of the fuel
        
    vessel_class : str
        Unique keyword for the given type and class

    modified_capacities_df : pd.DataFrame
        DataFrame containing the modified cargo capacity for each vessel (by both mass and volume) and related info

    Returns
    -------
    cargo_capacity_tonnes : float
        Cargo miles carried by mass, in tonne-miles
        
    cargo_capacity_cbm : float
        Cargo miles carried by volume, in tonne-m^3
        
    """
    route_properties_dict = get_route_properties(top_dir, vessel_class)
    
    if fuel == "lsfo":
        cargo_capacity_cbm = modified_capacities_df.loc[
        (modified_capacities_df["Fuel"] == "ammonia") & (modified_capacities_df["Vessel"] == vessel_class),
        "Nominal capacity (m^3)"].values[0]
        
        cargo_capacity_tonnes = modified_capacities_df.loc[
        (modified_capacities_df["Fuel"] == "ammonia") & (modified_capacities_df["Vessel"] == vessel_class),
        "Nominal capacity (tonnes)"].values[0]
    else:
        cargo_capacity_cbm = modified_capacities_df.loc[
        (modified_capacities_df["Fuel"] == fuel) & (modified_capacities_df["Vessel"] == vessel_class),
        "Modified capacity (m^3)"].values[0]
        
        cargo_capacity_tonnes = modified_capacities_df.loc[
        (modified_capacities_df["Fuel"] == fuel) & (modified_capacities_df["Vessel"] == vessel_class),
        "Modified capacity (tonnes)"].values[0]
    
    hours_at_sea = DAYS_PER_YEAR * HOURS_PER_DAY * route_properties_dict["TimeAtSea"]
    loaded_miles = hours_at_sea * np.sum(route_properties_dict["ConditionDistribution"] * route_properties_dict["Speeds"] * route_properties_dict["CapacityUtilizations"])
    
    cargo_miles_cbm = cargo_capacity_cbm * loaded_miles
    cargo_miles_tonnes = cargo_capacity_tonnes * loaded_miles
    return cargo_miles_cbm, cargo_miles_tonnes
    
def make_cargo_miles_df(top_dir, vessels, fuels, modified_capacities_df):
    """
    Creates a DataFrame containing the cargo miles for each vessel and fuel, with tank displacement expressed either in a weight-constrained or volume-constrained scenario.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
    
    vessels : dict
        Dictionary containing vessel types and their associated vessel size classes.

    fuels : list
        List of fuels to include in the calculations.
        
    modified_capacities_df : pd.DataFrame
        DataFrame containing the modified cargo capacity for each vessel (by both mass and volume) and related info

    Returns
    -------
    cargo_miles_df : pd.DataFrame
        DataFrame with the cargo miles for each vessel and fuel
    """
    
    # Create an empty list to hold rows for the DataFrame
    data = []

    # Loop through each vessel type and size
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            # Loop through each fuel
            for fuel in ["lsfo"] + fuels:
                cargo_miles_dict = {}
                cargo_miles_dict["Vessel"] = vessel_class
                cargo_miles_dict["Fuel"] = fuel
                
                # Get the cargo miles for the given vessel type and fuel
                cargo_miles_cbm, cargo_miles_tonnes = calculate_cargo_miles(top_dir, fuel, vessel_class, modified_capacities_df)
                
                cargo_miles_dict["Cargo miles (m^3-miles)"] = cargo_miles_cbm
                cargo_miles_dict["Cargo miles (tonne-miles)"] = cargo_miles_tonnes
                data.append(cargo_miles_dict)
    
    # Create a DataFrame from the collected info
    cargo_miles_df = pd.DataFrame(data)
    cargo_miles_df.to_csv("tables/cargo_miles.csv")
    
    return cargo_miles_df

    
def get_nominal_cargo_capacity_mass_volume(top_dir, cargo_info_df, vessel_type_class):
    """
    Gets the cargo capacity of a vessel in both tonnes and m^3, based on the average density of cargo that it carries
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
        
    vessel_type_class : str
        String specifying the vessel's type and class
        
    fuel : str
        Fuel used by the vessel
        
    cargo_info_df : pandas DataFrame
        Dataframe containing cargo-related info and assumptions for each vessel

    Returns
    -------
    cargo_capacity_tonnes : float
        Cargo capacity by mass, in tonnes
        
    cargo_capacity_cbm : float
        Cargo capacity by volume, in m^3
    """
    
    # Nominal capacity, in whatever units
    nominal_capacity = collect_nominal_capacity(top_dir, vessel_type_class)
    
    cargo_capacity_cbm = None
    cargo_capacity_tonnes = None
    if "bulk" in vessel_type_class:
        # Nominal capacity in NavigaTE is measured in deadweight tonnes
        cargo_capacity_tonnes = nominal_capacity
        
        # Volumetric capacity is read in from the info file, in m^3
        #print(cargo_info_df)
        cargo_capacity_cbm = cargo_info_df.loc[cargo_info_df["Vessel class"] == vessel_type_class, "Max capacity (m^3)"].values[0]
        
    elif "container" in vessel_type_class:
        # Nominal capacity in NavigaTE is measured in twenty foot equivalent units (TEUs)
        cargo_capacity_cbm = nominal_capacity * M3_PER_TEU
        
        # Mass cargo capacity is calculated based on the maximum allowed mass density of twenty foot equivalent cargo containers
        cargo_capacity_tonnes = cargo_capacity_cbm * cargo_info_df.loc[cargo_info_df["Vessel class"] == vessel_type_class, "Max density (tonnes/m^3)"].values[0]
        
    elif "tanker" in vessel_type_class:
        # Nominal capacity in NavigaTE is measured in deadweight tonnes
        cargo_capacity_tonnes = nominal_capacity
        
        # Volumetric capacity is evaluated from the mass capacity based on the minimum density of crude oil
        cargo_capacity_cbm = cargo_capacity_tonnes / cargo_info_df.loc[cargo_info_df["Vessel class"] == vessel_type_class, "Min density (tonnes/m^3)"].values[0]
        
    elif "gas_carrier" in vessel_type_class:
        # Nominal capacity in NavigaTE is measured in m^3
        cargo_capacity_cbm = nominal_capacity
        
        # Mass capacity is evaluated from the volumetric capacity baed on the mass density of LNG
        cargo_capacity_tonnes = cargo_capacity_cbm * cargo_info_df.loc[cargo_info_df["Vessel class"] == vessel_type_class, "Max density (tonnes/m^3)"].values[0]
    
    return cargo_capacity_cbm, cargo_capacity_tonnes
    
def calculate_modified_cargo_capacities(top_dir, vessel_type_class, fuel, cargo_info_df, mass_density_dict, tank_size_factors_dict, vessel_range=None):
    """
    Calculates the modified cargo capacities (both by mass and by volume) for the given vessel type and class.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
        
    vessel_type_class : str
        String specifying the vessel's type and class
        
    cargo_info_df : pandas DataFrame
        Dataframe containing cargo-related info and assumptions for each vessel
        
    mass_density_dict : dictionary of float
        Dictionary containing the mass density of each fuel

    Returns
    -------
    capacity_dict : Dictionary
        Dictionary containing the nominal and modified cargo capacities, along with the nominal and modified tank sizes
    """
    
    capacity_dict = {}
    
    # Get the nominal cargo capacity for the given vessel and class, in both mass and volume
    capacity_dict["Nominal capacity (m^3)"], capacity_dict["Nominal capacity (tonnes)"] = get_nominal_cargo_capacity_mass_volume(top_dir, cargo_info_df, vessel_type_class)
    
    # Get the nominal and modified tank sizes, in m^3
    if vessel_range is None:
        capacity_dict["Nominal tank size (m^3)"] = collect_tank_size(top_dir, vessel_type_class)
    else:
        capacity_dict["Nominal tank size (m^3)"] = get_lsfo_tank_size(vessel_range, vessel_type_class)

    capacity_dict["Modified tank size (m^3)"] = capacity_dict["Nominal tank size (m^3)"] * tank_size_factors_dict[fuel][vessel_type_class]["Total"]
    
    # Calculate the change in tank size, in both cbm and tonnes
    capacity_dict["Tank size difference (m^3)"] = capacity_dict["Modified tank size (m^3)"] - capacity_dict["Nominal tank size (m^3)"]
    capacity_dict["Tank size difference (tonnes)"] = capacity_dict["Modified tank size (m^3)"] * mass_density_dict[fuel] - capacity_dict["Nominal tank size (m^3)"] * mass_density_dict["lsfo"]
    
    # Calculate the modified cargo capacity, in both cbm and tonnes
    capacity_dict["Modified capacity (m^3)"] = capacity_dict["Nominal capacity (m^3)"] - capacity_dict["Tank size difference (m^3)"]
    capacity_dict["Modified capacity (tonnes)"] = capacity_dict["Nominal capacity (tonnes)"] - capacity_dict["Tank size difference (tonnes)"]
    
    capacity_dict["Percent volume difference (%)"] = 100 * (capacity_dict["Modified tank size (m^3)"] - capacity_dict["Nominal tank size (m^3)"]) / capacity_dict["Nominal capacity (m^3)"]
    #print("Cargo loss (%): ", capacity_dict["Percent volume difference (%)"])

    capacity_dict["Percent mass difference (%)"] = 100 * capacity_dict["Tank size difference (tonnes)"] / capacity_dict["Nominal capacity (tonnes)"]
    
    return capacity_dict
    
def make_modified_capacities_df(top_dir, vessels, fuels, mass_density_dict, cargo_info_df, tank_size_factors_dict, vessel_range=None):
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
        
    mass_density_dict : dictionary of float
        Dictionary containing the mass density of each fuel
        
    cargo_info_df : pandas DataFrame
        Dataframe containing cargo-related info and assumptions for each vessel

    Returns
    -------
    capacities_df : pd.DataFrame
        DataFrame with the nominal and modified capacities
    """
    
    # Create an empty list to hold rows for the DataFrame
    data = []
    
    # Loop through each vessel type and size
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            # Loop through each fuel
            for fuel in fuels:
                # Calculate the modified capacities for the given vessel type and fuel
                capacities_dict = calculate_modified_cargo_capacities(top_dir, vessel_class, fuel, cargo_info_df, mass_density_dict, tank_size_factors_dict, vessel_range)
                
                capacities_dict["Vessel"] = vessel_class
                capacities_dict["Vessel Type"] = vessel_type
                capacities_dict["Fuel"] = fuel
                                
                # Append the data to the list
                data.append(capacities_dict)
    
    # Create a DataFrame from the collected data
    capacities_df = pd.DataFrame(data)
    
    # Save to a csv file
    capacities_df.to_csv("tables/modified_tank_sizes_and_capacities.csv")
    
    return capacities_df
    
def plot_vessel_capacities(modified_capacities_df, capacity_type="mass"):
    """
    Plots vertical bar plots of nominal and modified capacities for each Vessel Type and Fuel,
    with percentage difference plotted as markers below the bars.
    
    Parameters
    ----------
    modified_capacities_df : pd.DataFrame
        DataFrame containing the modified cargo capacity for each vessel (by both mass and volume) and related info
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
                    if capacity_type == "mass":
                        nominal_capacity_label = "Nominal capacity (tonnes)"
                        modified_capacity_label = "Modified capacity (tonnes)"
                        perc_diff_label = "Percent mass difference (%)"
                    elif capacity_type == "volume":
                        nominal_capacity_label = "Nominal capacity (m^3)"
                        modified_capacity_label = "Modified capacity (m^3)"
                        perc_diff_label = "Percent volume difference (%)"
                    else:
                        raise Exception(f"Error: capacity type {capacity_type} supplied to plot_vessel_capacities is not recognized. Accepted types are 'mass' and 'volume'")
                    
                    ax_bars.bar(x_bar, df_fuel[nominal_capacity_label].values[0],
                                width=bar_width, color=fuel_colors[fuel], label=get_fuel_label(fuel) if i == 0 else "", alpha=0.7, edgecolor='black')
                    
                    # Plot the hatched bar for the modified capacity overlaid
                    ax_bars.bar(x_bar, df_fuel[modified_capacity_label].values[0],
                                width=bar_width, color='none', edgecolor='black', hatch='xxx')

                    # Plot the % difference marker on the lower axis
                    ax_diff.plot(x_bar, df_fuel[perc_diff_label].values[0],
                                 marker='o', color=fuel_colors[fuel], markersize=8)
                    ax_diff.hlines(df_fuel[perc_diff_label].values[0],
                                   x_bar - bar_width / 2, x_bar + bar_width / 2, color=fuel_colors[fuel])

        # Adjust the layout of the bar axis
        if capacity_type == "mass":
            ax_bars.set_ylabel("Capacity (tonnes)", fontsize=22)
        else:
            ax_bars.set_ylabel("Capacity (m$^3$)", fontsize=22)
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
        ax_diff.set_ylim([modified_capacities_df[modified_capacities_df["Vessel Type"] == vessel_type][perc_diff_label].min() - 5,
                          modified_capacities_df[modified_capacities_df["Vessel Type"] == vessel_type][perc_diff_label].max() + 5])

        # Set the x-axis ticks and labels
        ax_diff.set_xticks(x_pos + (bar_width * len(fuel_colors) / 2 - bar_width / 2))
        vessel_labels = [vessel_size_title[vessel] for vessel in vessels]
        ax_diff.set_xticklabels(vessel_labels, rotation=0, ha="center", fontsize=22)
        
        #plt.tight_layout()
        plt.savefig(f"plots/modified_capacities_{vessel_type}_{capacity_type}.png", dpi=300)
        ax_bars.set_title("")
        plt.savefig(f"plots/modified_capacities_{vessel_type}_{capacity_type}.pdf")
        plt.close()
    
def calculate_cargo_miles(top_dir, fuel, vessel_class, modified_capacities_df):
    """
    Calculates the cargo miles that a vessel travels annually, either in terms of tonne-miles or m^3-miles
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
        
    fuel : str
        Name of the fuel
        
    vessel_class : str
        Unique keyword for the given type and class

    modified_capacities_df : pd.DataFrame
        DataFrame containing the modified cargo capacity for each vessel (by both mass and volume) and related info

    Returns
    -------
    cargo_capacity_tonnes : float
        Cargo miles carried by mass, in tonne-miles
        
    cargo_capacity_cbm : float
        Cargo miles carried by volume, in tonne-m^3
        
    """
    route_properties_dict = get_route_properties(top_dir, vessel_class)
    
    if fuel == "lsfo":
        cargo_capacity_cbm = modified_capacities_df.loc[
        (modified_capacities_df["Fuel"] == "ammonia") & (modified_capacities_df["Vessel"] == vessel_class),
        "Nominal capacity (m^3)"].values[0]
        
        cargo_capacity_tonnes = modified_capacities_df.loc[
        (modified_capacities_df["Fuel"] == "ammonia") & (modified_capacities_df["Vessel"] == vessel_class),
        "Nominal capacity (tonnes)"].values[0]
    else:
        cargo_capacity_cbm = modified_capacities_df.loc[
        (modified_capacities_df["Fuel"] == fuel) & (modified_capacities_df["Vessel"] == vessel_class),
        "Modified capacity (m^3)"].values[0]
        
        cargo_capacity_tonnes = modified_capacities_df.loc[
        (modified_capacities_df["Fuel"] == fuel) & (modified_capacities_df["Vessel"] == vessel_class),
        "Modified capacity (tonnes)"].values[0]
    
    hours_at_sea = DAYS_PER_YEAR * HOURS_PER_DAY * route_properties_dict["TimeAtSea"]
    loaded_miles = hours_at_sea * np.sum(route_properties_dict["ConditionDistribution"] * route_properties_dict["Speeds"] * route_properties_dict["CapacityUtilizations"])
    
    cargo_miles_cbm = cargo_capacity_cbm * loaded_miles
    cargo_miles_tonnes = cargo_capacity_tonnes * loaded_miles
    return cargo_miles_cbm, cargo_miles_tonnes
    
def make_cargo_miles_df(top_dir, vessels, fuels, modified_capacities_df):
    """
    Creates a DataFrame containing the cargo miles for each vessel and fuel, with tank displacement expressed either in a weight-constrained or volume-constrained scenario.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.
    
    vessels : dict
        Dictionary containing vessel types and their associated vessel size classes.

    fuels : list
        List of fuels to include in the calculations.
        
    modified_capacities_df : pd.DataFrame
        DataFrame containing the modified cargo capacity for each vessel (by both mass and volume) and related info

    Returns
    -------
    cargo_miles_df : pd.DataFrame
        DataFrame with the cargo miles for each vessel and fuel
    """
    
    # Create an empty list to hold rows for the DataFrame
    data = []

    # Loop through each vessel type and size
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            # Loop through each fuel
            for fuel in ["lsfo"] + fuels:
                cargo_miles_dict = {}
                cargo_miles_dict["Vessel"] = vessel_class
                cargo_miles_dict["Fuel"] = fuel
                
                # Get the cargo miles for the given vessel type and fuel
                cargo_miles_cbm, cargo_miles_tonnes = calculate_cargo_miles(top_dir, fuel, vessel_class, modified_capacities_df)
                
                cargo_miles_dict["Cargo miles (m^3-miles)"] = cargo_miles_cbm
                cargo_miles_dict["Cargo miles (tonne-miles)"] = cargo_miles_tonnes
                data.append(cargo_miles_dict)
    
    # Create a DataFrame from the collected info
    cargo_miles_df = pd.DataFrame(data)
    cargo_miles_df.to_csv("tables/cargo_miles.csv")
    
    return cargo_miles_df
    
def get_fuel_properties(fuel):
    """
    Reads the LHV, mass density, and boil-off rate of the given fuel
        
    Parameters
    ----------
    fuel : str
        Name of the fuel to get the LHV for.

    Returns
    -------
    fuel_properties : Dictionary
        Dictionary containing:
            - Mass density of the fuel, in kg/L
            - Lower heating value of the fuel, in MJ / kg
            - Boil-off rate of the fuel, in %/day
            - Tank size scaling factor for energy density
            - Tank size scaling factor for engine efficiency
    """
    
    top_dir = get_top_dir()
    fuel_properties_dict = {}
    fuel_info = pd.read_csv(f"{top_dir}/info_files/fuel_info.csv", index_col="Fuel")
    fuel_properties_dict["Mass density (kg/L)"] = fuel_info.loc[fuel, "Mass density (kg/L)"]
    fuel_properties_dict["Lower Heating Value (MJ / kg)"] = fuel_info.loc[fuel, "Lower Heating Value (MJ / kg)"]
    fuel_properties_dict["Boil-off Rate (%/day)"] = fuel_info.loc[fuel, "Boil-off Rate (%/day)"]
    fuel_properties_dict["Engine efficiency"] = collect_propulsion_eff(top_dir, fuel)
    fuel_properties_dict["Tank size scaling factor (energy density)"] = get_tank_size_factor_energy(fuel_info.loc["lsfo", "Lower Heating Value (MJ / kg)"], fuel_info.loc["lsfo", "Mass density (kg/L)"], fuel_info.loc[fuel, "Lower Heating Value (MJ / kg)"], fuel_info.loc[fuel, "Mass density (kg/L)"])
    fuel_properties_dict["Tank size scaling factor (engine efficiency)"] = get_tank_size_factor_propulsion_eff(collect_propulsion_eff(top_dir, "lsfo"), collect_propulsion_eff(top_dir, fuel))
    
    return fuel_properties_dict
    
def fetch_and_save_vessel_info(cargo_info_df):
    """
    Fetches info used to define each vessel type and class, and saves it to a csv file for reference.
    
    Parameters
    ----------
    vessels : dict
        Dictionary containing vessel types and their associated vessel size classes.
    
    cargo_info_df : DataFrame
        Pandas dataframe containing the cargo capacity for each vessel

    Returns
    -------
    vessel_info_df : pd.DataFrame
        DataFrame with the info collected for each vessel
    """
    
    top_dir = get_top_dir()
    
    # Create an empty list to hold rows for the DataFrame
    data = []
    
    for vessel_type, vessel_classes in vessels.items():
        for vessel_class in vessel_classes:
            vessel_info_dict = {}
            nominal_capacity_cbm, nominal_capacity_tonnes = get_nominal_cargo_capacity_mass_volume(top_dir, cargo_info_df, vessel_class)
            nominal_tank_size_cbm = collect_tank_size(top_dir, vessel_class, fuel="oil")
            route_properties_dict = get_route_properties(top_dir, vessel_class)
            average_speed = np.sum(route_properties_dict["ConditionDistribution"] * route_properties_dict["Speeds"])
            average_utilization = np.sum(route_properties_dict["ConditionDistribution"] * route_properties_dict["CapacityUtilizations"])
            average_propulsion_power = calculate_average_propulsion_power(vessel_class, route_properties_dict)
            average_propulsion_energy_per_distance = calculate_average_propulsion_energy_per_distance(vessel_class, route_properties_dict)
            fuel_properties_lsfo = get_fuel_properties("lsfo")
            propulsion_eff_lsfo = collect_propulsion_eff(top_dir, "lsfo")
            nominal_vessel_range = calculate_vessel_range(top_dir, vessel_class, "lsfo", fuel_properties_lsfo["Lower Heating Value (MJ / kg)"], 1, fuel_properties_lsfo["Mass density (kg/L)"] * L_PER_M3, propulsion_eff_lsfo, route_properties_dict)
            
            vessel_info_dict["Vessel"] = f"{vessel_type_title[vessel_type]} ({vessel_size_title[vessel_class]})"
            vessel_info_dict["Nominal Cargo Capacity (m^3)"] = nominal_capacity_cbm
            vessel_info_dict["Nominal Cargo Capacity (tonnes)"] = nominal_capacity_tonnes
            vessel_info_dict["Nominal Tank Capacity (m^3)"] = nominal_tank_size_cbm
            vessel_info_dict["Nominal Range (nautical miles)"] = nominal_vessel_range
            vessel_info_dict["Average Speed (knots)"] = average_speed
            vessel_info_dict["Average Power over Speed (MJ / nautical mile)"] = average_propulsion_energy_per_distance
            vessel_info_dict["Average Propulsion Power (MW)"] = average_propulsion_power
            vessel_info_dict["Average Utilization"] = average_utilization
            vessel_info_dict["Fraction Year at Sea"] = route_properties_dict["TimeAtSea"]

            data.append(vessel_info_dict)
    
    # Create a DataFrame from the collected info
    vessel_info_df = pd.DataFrame(data)
    vessel_info_df.to_csv(f"{top_dir}/tables/vessel_info.csv")
    return vessel_info_df
    
def fetch_and_save_fuel_properties(fuels):
    """
    Fetches info for each fuel relevant for assessing the impact of tank displacement on cargo capacity, and saves it to a csv file for reference.
    
    Parameters
    ----------
    fuels : list of str
        List of fuels to consider

    Returns
    -------
    fuel_info_df : pd.DataFrame
        DataFrame with the info collected for each fuel
    """
    top_dir = get_top_dir()
    data = []
    
    for fuel in fuels + ["lsfo"]:
        fuel_properties_fuel = {}
        fuel_properties_fuel["Fuel"] = fuel
        fuel_properties_fuel.update(get_fuel_properties(fuel))
        data.append(fuel_properties_fuel)
    
    fuel_info_df = pd.DataFrame(data)
    fuel_info_df.to_csv(f"{top_dir}/tables/fuel_info.csv")
    return fuel_info_df
    
def main():
    # List of fuels to consider
    fuels = ["ammonia", "methanol", "FTdiesel", "liquid_hydrogen", "compressed_hydrogen"]

    top_dir = get_top_dir()
    
    mass_density_dict = get_fuel_info_dict(f"{top_dir}/info_files/fuel_info.csv", "Mass density (kg/L)")
    LHV_dict = get_fuel_info_dict(f"{top_dir}/info_files/fuel_info.csv", "Lower Heating Value (MJ / kg)")
    boiloff_rate_dict = get_fuel_info_dict(f"{top_dir}/info_files/fuel_info.csv", "Boil-off Rate (%/day)")
    propulsion_eff_dict = get_propulsion_eff_dict(top_dir, fuels + ["lsfo"])
    
    propulsion_power_distribution = calculate_propulsion_power_distribution([12, 22], [0, 0.5], "container_15000_teu")
 
    tank_size_factors_dict, days_to_empty_tank_dict = get_tank_size_factors(fuels, LHV_dict, mass_density_dict, propulsion_eff_dict, boiloff_rate_dict)
    
    cargo_info_df = pd.read_csv(f"{top_dir}/info_files/assumed_cargo_density.csv")
    fetch_and_save_vessel_info(cargo_info_df)
    fetch_and_save_fuel_properties(fuels)
    

#    tank_size_factors_dict, days_to_empty_tank_dict = get_tank_size_factors(fuels, LHV_dict, mass_density_dict, propulsion_eff_dict, boiloff_rate_dict, vessel_range=50000)
#    calculate_modified_cargo_capacities(top_dir, "tanker_35k_dwt", "liquid_hydrogen", cargo_info_df, mass_density_dict, tank_size_factors_dict, vessel_range=50000)
    
    #propulsion_power_distribution = calculate_propulsion_power_distribution([12, 22], [0, 0.5], "container_15000_teu")
    
    # First, consider original vessel design ranges
    tank_size_factors_dict, days_to_empty_tank_dict = get_tank_size_factors(fuels, LHV_dict, mass_density_dict, propulsion_eff_dict, boiloff_rate_dict, vessel_range=None)
    save_tank_size_factors(top_dir, tank_size_factors_dict, vessel_range=None)
    plot_tank_size_factors_boiloff(tank_size_factors_dict, days_to_empty_tank_dict)
    plot_tank_size_factors(tank_size_factors_dict)
    modified_capacities_df = make_modified_capacities_df(top_dir, vessels, fuels, mass_density_dict, cargo_info_df, tank_size_factors_dict)
    plot_vessel_capacities(modified_capacities_df, capacity_type="mass")
    plot_vessel_capacities(modified_capacities_df, capacity_type="volume")

    # Next, consider a range of vessel design ranges
    vessel_ranges = range(5000, 55000, 5000)
    
    tank_size_factors_dicts = {}
    days_to_empty_tank_dicts = {}
    modified_capacities_dfs = {}
    
#    for vessel_range in vessel_ranges:
#        print(f"Processing vessel range {vessel_range} nm")
#        tank_size_factors_dict, days_to_empty_tank_dict = get_tank_size_factors(fuels, LHV_dict, mass_density_dict, propulsion_eff_dict, boiloff_rate_dict, vessel_range)
#
#        # Save the tank size factors to a csv
#        save_tank_size_factors(top_dir, tank_size_factors_dict, vessel_range)
#        save_days_to_empty_tank(top_dir, days_to_empty_tank_dict, vessel_range)
#
#        # Modified vessel capacities
#        modified_capacities_df = make_modified_capacities_df(top_dir, vessels, fuels, mass_density_dict, cargo_info_df, tank_size_factors_dict, vessel_range)
#        modified_capacities_df.to_csv(f"{top_dir}/tables/modified_capacities_{vessel_range}.csv", index=False)
    
    make_modified_vessel_incs(top_dir, vessels, fuels, fuel_vessel_dict, modified_capacities_df)
    make_modified_tank_incs(top_dir, vessels, fuels, fuel_vessel_dict, modified_capacities_df)

    # Collect the range (in nautical miles) for each fuel
    vessel_ranges_df = calculate_all_vessel_ranges(top_dir, fuels, tank_size_factors_dict, LHV_dict, mass_density_dict, propulsion_eff_dict, boiloff_rate_dict)
    
    # Save the ranges to a csv file
    vessel_ranges_df.to_csv(f"{top_dir}/data/vessel_ranges.csv")
    
    #cargo_miles_cbm, cargo_miles_tonnes = calculate_cargo_miles(top_dir, "lsfo", "bulk_carrier_handy", modified_capacities_df)
    
    #cargo_miles_df = make_cargo_miles_df(top_dir, vessels, fuels, modified_capacities_df)
        
if __name__ == "__main__":
    main()

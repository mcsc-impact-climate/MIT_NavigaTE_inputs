"""
Date: Dec. 17, 2024
Author: danikam
Purpose: Analyze the impact of vessel design parameters on the cargo displacement due to increased tank sizes needed for alternative fuels.
"""

import pandas as pd
import os
from common_tools import get_top_dir, get_fuel_label
import matplotlib.pyplot as plt
import numpy as np

KG_PER_TONNE=1000
L_PER_CBM=1000
S_PER_DAY=86400

def collect_nominal_vessel_design_params(top_dir):
    """
    Collects the nominal values of vessel design parameters considered in this analysis for each vessel defined in NavigaTE.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.

    Returns
    -------
    vessel_design_params : pd.DataFrame
        DataFrame with the info collected for each vessel
    """
    vessel_design_params_df = pd.read_csv(f"{top_dir}/tables/vessel_info.csv")
    vessel_design_params_df["Nominal Cargo Capacity (kg)"] = vessel_design_params_df["Nominal Cargo Capacity (tonnes)"] * KG_PER_TONNE
    return vessel_design_params_df
    
def extract_vessel_design_params(vessel_design_params_df, vessel):
    """
    Extracts the vessel design parameters for a given vessel as a dictionary, using the dataframe containing parameters for all vessels.
    
    Parameters
    ----------
    vessel_design_params_df : pandas.DataFrame
        DataFrame containing the design parameters for all vessels.
    vessel : str
        Name of the vessel, as it's stored in the dataframe.

    Returns
    -------
    vessel_design_params_dict : dict
        Dictionary with the nominal design parameters for the given vessel.
    """
    # Filter the DataFrame for the specified vessel
    vessel_row = vessel_design_params_df[vessel_design_params_df['Vessel'] == vessel]
    
    # Check if the vessel exists in the DataFrame
    if vessel_row.empty:
        raise ValueError(f"Vessel '{vessel}' not found in the dataframe.")
    
    # Extract the row as a dictionary (reset index to ensure correct format)
    vessel_design_params_dict = vessel_row.iloc[0].to_dict()
    
    # Drop the index column (e.g., "Unnamed: 0")
    vessel_design_params_dict.pop("Unnamed: 0", None)
    
    return vessel_design_params_dict

def collect_fuel_params(top_dir):
    """
    Collects the vessel design parameters considered in this analysis for each vessel defined in NavigaTE.
    
    Parameters
    ----------
    top_dir : str
        The top directory where vessel files are located.

    Returns
    -------
    fuel_params : pd.DataFrame
        DataFrame with the info collected for each vessel
    """
    fuel_params_df = pd.read_csv(f"{top_dir}/tables/fuel_info.csv")
    fuel_params_df["Mass density (kg/m^3)"] = fuel_params_df["Mass density (kg/L)"] * L_PER_CBM
    
    return fuel_params_df
    
def extract_fuel_params(fuel_params_df, fuel):
    """
    Extracts the fuel parameters for a given fuel as a dictionary, using the dataframe containing parameters for all fuels.
    
    Parameters
    ----------
    fuel_params_df : pandas.DataFrame
        DataFrame containing the fuel parameters for all fuels.
    fuel : str
        Name of the fuel, as it's stored in the dataframe.

    Returns
    -------
    fuel_params_dict : dict
        Dictionary with the parameters for the given fuel.
    """
    # Filter the DataFrame for the specified fuel
    fuel_row = fuel_params_df[fuel_params_df['Fuel'] == fuel]
    
    # Check if the vessel exists in the DataFrame
    if fuel_row.empty:
        raise ValueError(f"Fuel '{fuel}' not found in the dataframe.")
    
    # Extract the row as a dictionary (reset index to ensure correct format)
    fuel_params_dict = fuel_row.iloc[0].to_dict()
    
    # Drop the index column (e.g., "Unnamed: 0")
    fuel_params_dict.pop("Unnamed: 0", None)
    
    return fuel_params_dict

def calculate_days_to_empty_tank(R, P_s_av, f_s, P_av):
    """
    Calculates the total number of days needed to empty the vessel's tank
    
    Parameters
    ----------
    R : float
        Vessel design range, in nautical miles
    
    P_s_av : float
        Average ratio of main engine power to vessel speed, in MJ / nautical mile
        
    f_s : float
        Fraction of time that the vessel spends at sea (vs. at port)
        
    P_av : float
        Average main engine power, in MW (MJ/s)

    Returns
    -------
    N_days : float
        Average number of days the vessel takes to empty its tank
    """
    
    N_seconds = R * P_s_av / (f_s * P_av)
    N_days = N_seconds / S_PER_DAY
    
    return N_days
    
def calculate_tank_size(R, P_s_av, rho_l, e_l, L_l):
    """
    Calculates the tank size based on the range, average main engine power, fuel density, and lower heating value.
    
    Parameters
    ----------
    R : float
        Vessel design range, in nautical miles
    
    P_s_av : float
        Average ratio of main engine power to vessel speed, in MJ / nautical mile
        
    rho_l : float
        Mass density of LSFO fuel (kg/m^3)
        
    e_l : float
        Energy conversion efficiency of LSFO fuel by the main engine
        
    L_l : float
        Lower heating value of the given fuel (MJ/kg)

    Returns
    -------
    tank_size : float
        Tank size, in m^3
    """
    tank_size = R * P_s_av / (rho_l * e_l * L_l)
    return tank_size

def calculate_fuel_term(L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="mass"):
    """
    Calculates the dimensionless fuel term in the equation for fractional cargo capacity loss
    
    Parameters
    ----------
    L_l : float
        Lower heating value of LSFO fuel (MJ/kg)
    
    rho_l : float
        Mass density of LSFO fuel (kg/m^3)
        
    e_l : float
        Energy conversion efficiency of LSFO fuel by the main engine
        
    L_l : float
        Lower heating value of the given fuel (MJ/kg)
    
    rho_l : float
        Mass density of the given fuel (kg/m^3)
        
    e_l : float
        Energy conversion efficiency of the given fuel by the main engine

    Returns
    -------
    fuel_term : float
        Dimensionless fuel term
    """
    
    if cargo_type == "mass":
        fuel_term = L_l * e_l / (L_f * e_f)
        return fuel_term
        
    elif cargo_type == "volume":
        fuel_term = L_l * rho_l * e_l / (L_f * rho_f * e_f)
        return fuel_term
    
    else:
        print(f"Error: Cargo type {cargo_type} not recognized. Currently accepted cargo types are mass and volume.")
        return

def calculate_scaling_term(R, P_s_av, e_l, L_l, rho_l, m_c, V_c, cargo_type="mass"):
    """
    Calculates the dimensionless scaling term in the equation for fractional cargo capacity loss
    
    Parameters
    ----------
    L_l : float
        Lower heating value of LSFO fuel (MJ/kg)
    
    rho_l : float
        Mass density of LSFO fuel (kg/m^3)
        
    e_l : float
        Energy conversion efficiency of LSFO fuel by the main engine
        
    L_l : float
        Lower heating value of the given fuel (MJ/kg)
    
    rho_l : float
        Mass density of the given fuel (kg/m^3)
        
    e_l : float
        Energy conversion efficiency of the given fuel by the main engine

    Returns
    -------
    fuel_term : float
        Dimensionless fuel term
    """

    if cargo_type == "mass":
        return R * P_s_av / (e_l * L_l * m_c)
        
    elif cargo_type == "volume":
        return R * P_s_av / (rho_l * e_l * L_l * V_c)

    else:
        print(f"Error: Cargo type {cargo_type} not recognized. Currently accepted cargo types are mass and volume.")
        return

def calculate_boiloff_term(r_f, N_days):
    """
    Calculates the dimensionless boil-off term in the equation for fractional cargo capacity loss
    
    Parameters
    ----------
    r_f : float
        Fractional loss of fuel to boil-off per day
    
    N_days : float
        Number of days the vessel takes to empty its fuel tank

    Returns
    -------
    boiloff_term : float
        Dimensionless boiloff term
    """
    
    boiloff_term = 1 / (1 - r_f/100)**N_days
    return boiloff_term
    
def calculate_cargo_loss(R, P_s_av, P_av, r_f, f_s, e_l, L_l, rho_l, e_f, L_f, rho_f, m_c, V_c, cargo_type = "mass"):
    N_days_to_empty = calculate_days_to_empty_tank(R, P_s_av, f_s, P_av)
    
    scaling_term = calculate_scaling_term(R, P_s_av, e_l, L_l, rho_l, m_c, V_c, cargo_type)
    fuel_term = calculate_fuel_term(L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type)
    boiloff_term = calculate_boiloff_term(r_f, N_days_to_empty)
        
    cargo_loss = scaling_term * (fuel_term * boiloff_term - 1)
    return cargo_loss
    
def get_cargo_loss_info(nominal_vessel_params_df, fuel_params_df):
    """
    Calculates parameters related to cargo loss for all fuels and vessels, and saves them to a dataframe.
    
    Parameters
    ----------
    nominal_vessel_params_df : pd.DataFrame
        Dataframe containing nominal parameters for each vessel
    
    fuel_params_df : pd.DataFrame
        Dataframe containing parameters for each fuel

    Returns
    -------
    cargo_loss_info_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss
    """
    
    top_dir = get_top_dir()
    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    data = []
    for vessel in nominal_vessel_params_df["Vessel"]:
        nominal_vessel_params = extract_vessel_design_params(nominal_vessel_params_df, vessel)
        
        tank_size = calculate_tank_size(nominal_vessel_params["Nominal Range (nautical miles)"], nominal_vessel_params["Average Power over Speed (MJ / nautical mile)"], fuel_params_lsfo["Mass density (kg/m^3)"], fuel_params_lsfo["Engine efficiency"], fuel_params_lsfo["Lower Heating Value (MJ / kg)"])
        
        N_days_to_empty = calculate_days_to_empty_tank(nominal_vessel_params["Nominal Range (nautical miles)"], nominal_vessel_params["Average Power over Speed (MJ / nautical mile)"], nominal_vessel_params["Fraction Year at Sea"], nominal_vessel_params["Average Propulsion Power (MW)"])
        
        for fuel in fuel_params_df["Fuel"]:
            cargo_loss_info_dict = {}
            cargo_loss_info_dict["Vessel"] = vessel
            cargo_loss_info_dict["Fuel"] = fuel
            cargo_loss_info_dict["Days to Empty"] = N_days_to_empty
            cargo_loss_info_dict["Tank size"] = tank_size
            
            if fuel == "lsfo":
                continue
            fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)
            
            for cargo_type in ["mass", "volume"]:
            
                cargo_loss_info_dict[f"Fuel term ({cargo_type})"] = calculate_fuel_term(fuel_params_lsfo["Lower Heating Value (MJ / kg)"], fuel_params_lsfo["Mass density (kg/m^3)"], fuel_params_lsfo["Engine efficiency"], fuel_params_fuel["Lower Heating Value (MJ / kg)"], fuel_params_fuel["Mass density (kg/m^3)"], fuel_params_fuel["Engine efficiency"], cargo_type=cargo_type)
            
                cargo_loss_info_dict[f"Scaling term ({cargo_type})"] = calculate_scaling_term(nominal_vessel_params["Nominal Range (nautical miles)"], nominal_vessel_params["Average Power over Speed (MJ / nautical mile)"], fuel_params_lsfo["Engine efficiency"], fuel_params_lsfo["Lower Heating Value (MJ / kg)"], fuel_params_lsfo["Mass density (kg/m^3)"],  nominal_vessel_params["Nominal Cargo Capacity (kg)"], nominal_vessel_params["Nominal Cargo Capacity (m^3)"], cargo_type=cargo_type)
            
            cargo_loss_info_dict[f"Boil-off term"] = calculate_boiloff_term(fuel_params_fuel["Boil-off Rate (%/day)"], N_days_to_empty)
            
            cargo_loss_info_dict[f"Fractional loss (mass)"] = cargo_loss_info_dict[f"Scaling term (mass)"] * (cargo_loss_info_dict[f"Fuel term (mass)"] * cargo_loss_info_dict[f"Boil-off term"] - 1)
            
            cargo_loss_info_dict[f"Fractional loss (volume)"] = cargo_loss_info_dict[f"Scaling term (volume)"] * (cargo_loss_info_dict[f"Fuel term (volume)"] *  cargo_loss_info_dict[f"Boil-off term"] - 1)
            
            data.append(cargo_loss_info_dict)
            
    cargo_loss_info_df = pd.DataFrame(data)
    cargo_loss_info_df.to_csv(f"{top_dir}/tables/cargo_loss_info.csv", index=False)
    return cargo_loss_info_df
    
def get_parameter_values(nominal_vessel_params_df, fuel_params_df, fuel="liquid_hydrogen"):
    """
    Calculates all parameters related to cargo loss for all vessels for the given fuel, evaluates the maximum and minimum value of each parameter over all vessels, and saves the results to a dataframe.
    
    Parameters
    ----------
    nominal_vessel_params_df : pd.DataFrame
        Dataframe containing nominal parameters for each vessel
    
    fuel_params_df : pd.DataFrame
        Dataframe containing parameters for each fuel

    fuel : str
        Fuel to get parameter values for

    Returns
    -------
    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima
    """
    top_dir = get_top_dir()
    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)
    
    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]
    e_l = fuel_params_lsfo["Engine efficiency"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["Engine efficiency"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]
    fuel_term_mass = calculate_fuel_term(L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="mass")
    fuel_term_volume = calculate_fuel_term(L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="volume")

    minima = {
        "Vessel": "Minimum",
        "R (nm)": 9e9,
        "P_s_av (MJ/nm)": 9e9,
        "P_av (MW)": 9e9,
        "f_s": 9e9,
        "m_c (kg)": 9e9,
        "V_c (m^3)": 9e9,
        "N_days": 9e9,
        "r_f": 9e9,
        "scaling_term_volume": 9e9,
        "fuel_term_volume": 9e9,
        "scaling_term_mass": 9e9,
        "fuel_term_mass": 9e9,
        "cargo_loss_volume": 9e9,
        "cargo_loss_mass": 9e9,
    }

    maxima = {
        "Vessel": "Maximum",
        "R (nm)": -9e9,
        "P_s_av (MJ/nm)": -9e9,
        "P_av (MW)": -9e9,
        "f_s": -9e9,
        "m_c (kg)": -9e9,
        "V_c (m^3)": -9e9,
        "N_days": -9e9,
        "r_f": -9e9,
        "scaling_term_volume": -9e9,
        "fuel_term_volume": -9e9,
        "scaling_term_mass": -9e9,
        "fuel_term_mass": -9e9,
        "cargo_loss_volume": -9e9,
        "cargo_loss_mass": -9e9,
    }
    
    data=[]
    for vessel in nominal_vessel_params_df["Vessel"]:
        parameter_value_dict = {}
        nominal_vessel_params = extract_vessel_design_params(nominal_vessel_params_df, vessel)
        parameter_value_dict["Vessel"] = vessel
        parameter_value_dict["R (nm)"] = nominal_vessel_params["Nominal Range (nautical miles)"]
        parameter_value_dict["P_s_av (MJ/nm)"] = nominal_vessel_params["Average Power over Speed (MJ / nautical mile)"]
        parameter_value_dict["P_av (MW)"] = nominal_vessel_params["Average Propulsion Power (MW)"]
        parameter_value_dict["f_s"] = nominal_vessel_params["Fraction Year at Sea"]
        parameter_value_dict["m_c (kg)"] = nominal_vessel_params["Nominal Cargo Capacity (kg)"]
        parameter_value_dict["V_c (m^3)"] = nominal_vessel_params["Nominal Cargo Capacity (m^3)"]
        parameter_value_dict["N_days"] = calculate_days_to_empty_tank(parameter_value_dict["R (nm)"], parameter_value_dict["P_s_av (MJ/nm)"], parameter_value_dict["f_s"], parameter_value_dict["P_av (MW)"])
        parameter_value_dict["r_f"] = r_f
        parameter_value_dict["scaling_term_volume"] = calculate_scaling_term(parameter_value_dict["R (nm)"], parameter_value_dict["P_s_av (MJ/nm)"], e_l, L_l, rho_l,  parameter_value_dict["m_c (kg)"], parameter_value_dict["V_c (m^3)"], cargo_type="volume")
        parameter_value_dict["scaling_term_mass"] = calculate_scaling_term(parameter_value_dict["R (nm)"], parameter_value_dict["P_s_av (MJ/nm)"], e_l, L_l, rho_l,  parameter_value_dict["m_c (kg)"], parameter_value_dict["V_c (m^3)"], cargo_type="mass")
        parameter_value_dict["fuel_term_volume"] = fuel_term_volume
        parameter_value_dict["fuel_term_mass"] = fuel_term_mass
        parameter_value_dict["cargo_loss_volume"] = calculate_cargo_loss(parameter_value_dict["R (nm)"], parameter_value_dict["P_s_av (MJ/nm)"], parameter_value_dict["P_av (MW)"], r_f, parameter_value_dict["f_s"], e_l, L_l, rho_l, e_f, L_f, rho_f, parameter_value_dict["m_c (kg)"], parameter_value_dict["V_c (m^3)"], cargo_type = "volume")
        parameter_value_dict["cargo_loss_mass"] = calculate_cargo_loss(parameter_value_dict["R (nm)"], parameter_value_dict["P_s_av (MJ/nm)"], parameter_value_dict["P_av (MW)"], r_f, parameter_value_dict["f_s"], e_l, L_l, rho_l, e_f, L_f, rho_f, parameter_value_dict["m_c (kg)"], parameter_value_dict["V_c (m^3)"], cargo_type = "mass")
        
        data.append(parameter_value_dict)
        
        # Set the minimum and maximum values for each parameter over all vessels
        for parameter in minima:
            if parameter == "Vessel":
                continue
            if parameter_value_dict[parameter] < minima[parameter]:
                minima[parameter] = parameter_value_dict[parameter]
                
            if parameter_value_dict[parameter] > maxima[parameter]:
                maxima[parameter] = parameter_value_dict[parameter]
        
    data.append(minima)
    data.append(maxima)
    
    parameter_values_df = pd.DataFrame(data)
    parameter_values_df.to_csv(f"{top_dir}/tables/parameter_values_{fuel}.csv", index=False)
    
    return parameter_values_df

def plot_cargo_loss_vs_range(vessel, fuel, cargo_type):
    """
    Plots the cargo loss as a function of range.
    
    Parameters
    ----------
    vessel : str
        Name of the vessel to consider
    
    fuel : str
        Name of the fuel to consider
        
    cargo_type : str
        Type of cargo (mass or volume)

    Returns
    -------
    None
    """
    top_dir = get_top_dir()
    nominal_vessel_params_df = collect_nominal_vessel_design_params(top_dir)
    nominal_vessel_params = extract_vessel_design_params(nominal_vessel_params_df, vessel)
    fuel_params_df = collect_fuel_params(top_dir)
    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)
    
    P_s_av = nominal_vessel_params["Average Power over Speed (MJ / nautical mile)"]
    P_av = nominal_vessel_params["Average Propulsion Power (MW)"]
    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]
    f_s = nominal_vessel_params["Fraction Year at Sea"]
    e_l = fuel_params_lsfo["Engine efficiency"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["Engine efficiency"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]
    m_c = nominal_vessel_params["Nominal Cargo Capacity (kg)"]
    V_c = nominal_vessel_params["Nominal Cargo Capacity (m^3)"]
    
    vessel_ranges = range(5000, 55000, 5000)
    N_days_to_empty = []
    total_size_corr_factor = []
    cargo_losses = []
    for R in vessel_ranges:
        N_days = calculate_days_to_empty_tank(R, P_s_av, f_s, P_av)
        N_days_to_empty.append(N_days)
        
        fuel_term = calculate_fuel_term(L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type=cargo_type)
        boiloff_term = calculate_boiloff_term(r_f, N_days)
        
        total_size_corr_factor.append(fuel_term * boiloff_term)
        
        cargo_losses.append(calculate_cargo_loss(R, P_s_av, P_av, r_f, f_s, e_l, L_l, rho_l, e_f, L_f, rho_f, m_c, V_c, cargo_type = cargo_type))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(vessel_ranges, total_size_corr_factor)
    ax.set_xlabel("Range (nautical miles)")
    ax.set_ylabel(f"Tank size correction factor")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(vessel_ranges, N_days_to_empty)
    ax.set_xlabel("Range (nautical miles)")
    ax.set_ylabel(f"Days to empty tank")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(vessel_ranges, cargo_losses)
    ax.set_xlabel("Range (nautical miles)")
    ax.set_ylabel(f"Fractional cargo loss ({cargo_type})")
    plt.show()
    
def plot_cargo_loss_vs_dimensionless_params(parameter_values_df, fuel, cargo_type):
    """
    Plots the cargo loss as a function of dimensionless parameters that vary between different vessel designs.

    Parameters
    ----------
    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    fuel : str
        Name of the fuel to consider

    cargo_type : str
        Type of cargo (mass or volume)

    Returns
    -------
    None
    """
    
    # Determine column names based on cargo_type
    if cargo_type == "volume":
        cargo_loss_col = "cargo_loss_volume"
        scaling_term_col = "scaling_term_volume"
        fuel_term_col = "fuel_term_volume"
    elif cargo_type == "mass":
        cargo_loss_col = "cargo_loss_mass"
        scaling_term_col = "scaling_term_mass"
        fuel_term_col = "fuel_term_mass"
    else:
        raise ValueError("Invalid cargo_type. Choose 'volume' or 'mass'.")

    # Remove "Minimum" and "Maximum" rows from the vessel data
    vessel_data = parameter_values_df[~parameter_values_df["Vessel"].isin(["Minimum", "Maximum"])]

    # Extract relevant columns
    N_days = vessel_data["N_days"]
    scaling_term = vessel_data[scaling_term_col]
    fuel_term = parameter_values_df.loc[parameter_values_df["Vessel"] == "Minimum", fuel_term_col].iloc[0]
    r_f = parameter_values_df.loc[parameter_values_df["Vessel"] == "Minimum", "r_f"].iloc[0]
    cargo_loss = vessel_data[cargo_loss_col]

    # Define grid for heatmap with extended ranges
    N_days_min, N_days_max = N_days.min(), N_days.max()
    scaling_term_min, scaling_term_max = scaling_term.min(), scaling_term.max()
    
    N_days_range = N_days_max - N_days_min
    scaling_term_range = scaling_term_max - scaling_term_min

    N_days_grid = np.linspace(N_days_min - 0.1 * N_days_range, N_days_max + 0.1 * N_days_range, 100)
    scaling_term_grid = np.linspace(scaling_term_min - 0.1 * scaling_term_range, scaling_term_max + 0.1 * scaling_term_range, 100)

    N_days_mesh, scaling_term_mesh = np.meshgrid(N_days_grid, scaling_term_grid)
    cargo_loss_mesh = scaling_term_mesh * (fuel_term / (1 - r_f / 100)**N_days_mesh - 1)

    # Find the location of the absolute minimum cargo loss
    min_loss_index = np.unravel_index(np.nanargmin(cargo_loss_mesh), cargo_loss_mesh.shape)
    min_N_days = N_days_mesh[min_loss_index]
    min_scaling_term = scaling_term_mesh[min_loss_index]
    min_cargo_loss = cargo_loss_mesh[min_loss_index]

    # Plot the heatmap
    plt.figure(figsize=(15, 8))
    plt.contourf(N_days_mesh, scaling_term_mesh, cargo_loss_mesh, levels=20, cmap="viridis")
    cbar = plt.colorbar(label=f"Cargo Loss ({cargo_type})")
    cbar.set_label(f"Cargo Loss ({cargo_type})", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    # Overlay vessel points with shades of orange/red/magenta
    colors = plt.cm.magma(np.linspace(0.2, 0.8, len(vessel_data)))
    for i, (vessel, n_days, scale, color) in enumerate(zip(vessel_data["Vessel"], N_days, scaling_term, colors)):
        plt.scatter(n_days, scale, color=color, label=vessel, alpha=0.8)

    # Highlight the absolute minimum cargo loss
    #plt.scatter(min_N_days, min_scaling_term, color="red", s=200, marker="*", label="Minimum Cargo Loss")

    # Add labels and title
    plt.xlabel("Days to empty tank", fontsize=20)
    plt.ylabel(f"Scaling term ({cargo_type})".replace("_", " ").capitalize(), fontsize=20)
    plt.title(f"Fuel: {get_fuel_label(fuel)}. Cargo type: {cargo_type}", fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.legend(fontsize=18, bbox_to_anchor=(1.25, 1))

    # Adjust grid to span parameter ranges proportionally (rectangular grid)
    plt.gca().set_aspect('auto')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f"plots/cargo_loss_vs_dimensionless_params_{fuel}_{cargo_type}.png", dpi=300)
    plt.savefig(f"plots/cargo_loss_vs_dimensionless_params_{fuel}_{cargo_type}.pdf")
    plt.close()
    

def main():
    top_dir = get_top_dir()
    
    # Collect vessel design and fuel parameters for each vessel+fuel
    nominal_vessel_params_df = collect_nominal_vessel_design_params(top_dir)
    fuel_params_df = collect_fuel_params(top_dir)
    
    # Get cargo loss info
    cargo_loss_info_df = get_cargo_loss_info(nominal_vessel_params_df, fuel_params_df)
    
    # Plot cargo loss as a function of range
    #plot_cargo_loss_vs_range("Tanker (35k DWT)", "liquid_hydrogen", "volume")
    
    for fuel in fuel_params_df["Fuel"]:
        if fuel == "lsfo":
            continue
            
        parameter_values_df = get_parameter_values(nominal_vessel_params_df, fuel_params_df, fuel)
        plot_cargo_loss_vs_dimensionless_params(parameter_values_df, fuel, "volume")
        plot_cargo_loss_vs_dimensionless_params(parameter_values_df, fuel, "mass")

if __name__ == "__main__":
    main()

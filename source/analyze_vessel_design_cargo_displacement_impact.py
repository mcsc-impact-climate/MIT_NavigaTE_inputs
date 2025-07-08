"""
Date: Dec. 17, 2024
Author: danikam
Purpose: Analyze the impact of vessel design parameters on the cargo displacement due to increased tank sizes needed for alternative fuels.
"""

import os
import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common_tools import get_fuel_label, get_top_dir
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import differential_evolution, minimize

KG_PER_TONNE = 1000
L_PER_CBM = 1000
S_PER_DAY = 86400

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
    vessel_design_params_df["Nominal Cargo Capacity (kg)"] = (
        vessel_design_params_df["Nominal Cargo Capacity (tonnes)"] * KG_PER_TONNE
    )
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
    vessel_row = vessel_design_params_df[vessel_design_params_df["Vessel"] == vessel]

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
    fuel_params_df["Mass density (kg/m^3)"] = (
        fuel_params_df["Mass density (kg/L)"] * L_PER_CBM
    )

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
    fuel_row = fuel_params_df[fuel_params_df["Fuel"] == fuel]

    # Check if the vessel exists in the DataFrame
    if fuel_row.empty:
        raise ValueError(f"Fuel '{fuel}' not found in the dataframe.")

    # Extract the row as a dictionary (reset index to ensure correct format)
    fuel_params_dict = fuel_row.iloc[0].to_dict()

    # Drop the index column (e.g., "Unnamed: 0")
    fuel_params_dict.pop("Unnamed: 0", None)

    return fuel_params_dict


def calculate_days_to_empty_tank(R, P_s_av, P_av):
    """
    Calculates the total number of days needed to empty the vessel's tank

    Parameters
    ----------
    R : float
        Vessel design range, in nautical miles

    P_s_av : float
        Average ratio of main engine power to vessel speed, in MJ / nautical mile

    P_av : float
        Average main engine power, in MW (MJ/s)

    Returns
    -------
    N_days : float
        Average number of days the vessel takes to empty its tank
    """

    N_seconds = R * P_s_av / (P_av)
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
        print(
            f"Error: Cargo type {cargo_type} not recognized. Currently accepted cargo types are mass and volume."
        )
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

    m_c : float
        Mass cargo capacity of the vessel, in kg

    V_c : float
        Volume cargo capacity of the vessel, in m^3

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
        print(
            f"Error: Cargo type {cargo_type} not recognized. Currently accepted cargo types are mass and volume."
        )
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
    
    if r_f == 0:
        boiloff_term = 1
    else:
        boiloff_term = (1 / (1 - r_f / 100) ** N_days) * (1 - (1 - r_f/100) ** N_days) / (N_days * r_f/100)
    return boiloff_term


def calculate_cargo_loss(
    R,
    P_s_av,
    P_av,
    r_f,
    e_l,
    L_l,
    rho_l,
    e_f,
    L_f,
    rho_f,
    m_c,
    V_c,
    f_p,
    cargo_type="mass",
):
    """
    Calculates the fractional cargo loss due to tank displacement.

    Parameters
    ----------
    R : float
        Vessel design range, in nautical miles

    P_s_av : float
        Average ratio of main engine power to vessel speed, in MJ / nautical mile

    P_av : float
        Average main engine power, in MW (MJ/s)

    r_f : float
        Fractional loss of fuel to boil-off per day

    e_l : float
        Energy conversion efficiency of LSFO fuel by the main engine

    L_l : float
        Lower heating value of LSFO fuel (MJ/kg)

    rho_l : float
        Mass density of LSFO fuel (kg/m^3)

    e_f : float
        Energy conversion efficiency of the given fuel by the main engine

    L_f : float
        Lower heating value of the given fuel (MJ/kg)

    rho_f : float
        Mass density of the given fuel (kg/m^3)

    m_c : float
        Mass cargo capacity of the vessel, in kg

    V_c : float
        Volume cargo capacity of the vessel, in m^3
    
    f_p : float
        Pilot tank size ratio

    Returns
    -------
    cargo_loss : float
        Fractional cargo loss
    """
    scaling_term = calculate_scaling_term(
        R, P_s_av, e_l, L_l, rho_l, m_c, V_c, cargo_type
    )
    fuel_term = calculate_fuel_term(L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type)

    # If the daily fuel loss to boil-off is zero, don't bother calculating the boil-off term
    if np.absolute(r_f) < 0.000001:
        cargo_loss = scaling_term * (fuel_term + f_p - 1)

    else:
        N_days_to_empty = calculate_days_to_empty_tank(R, P_s_av, P_av)
        boiloff_term = calculate_boiloff_term(r_f, N_days_to_empty)
        cargo_loss = scaling_term * (fuel_term * boiloff_term + f_p - 1)
    
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
        nominal_vessel_params = extract_vessel_design_params(
            nominal_vessel_params_df, vessel
        )

        tank_size = calculate_tank_size(
            nominal_vessel_params["Nominal Range (nautical miles)"],
            nominal_vessel_params["Average Power over Speed (MJ / nautical mile)"],
            fuel_params_lsfo["Mass density (kg/m^3)"],
            fuel_params_lsfo["f_eff_mean"],
            fuel_params_lsfo["Lower Heating Value (MJ / kg)"],
        )

        N_days_to_empty = calculate_days_to_empty_tank(
            nominal_vessel_params["Nominal Range (nautical miles)"],
            nominal_vessel_params["Average Power over Speed (MJ / nautical mile)"],
            nominal_vessel_params["Fraction Year at Sea"],
            nominal_vessel_params["Average Propulsion Power (MW)"],
        )

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
                cargo_loss_info_dict[f"Fuel term ({cargo_type})"] = calculate_fuel_term(
                    fuel_params_lsfo["Lower Heating Value (MJ / kg)"],
                    fuel_params_lsfo["Mass density (kg/m^3)"],
                    fuel_params_lsfo["f_eff_mean"],
                    fuel_params_fuel["Lower Heating Value (MJ / kg)"],
                    fuel_params_fuel["Mass density (kg/m^3)"],
                    fuel_params_fuel["f_eff_mean"],
                    cargo_type=cargo_type,
                )

                cargo_loss_info_dict[f"Scaling term ({cargo_type})"] = (
                    calculate_scaling_term(
                        nominal_vessel_params["Nominal Range (nautical miles)"],
                        nominal_vessel_params[
                            "Average Power over Speed (MJ / nautical mile)"
                        ],
                        fuel_params_lsfo["f_eff_mean"],
                        fuel_params_lsfo["Lower Heating Value (MJ / kg)"],
                        fuel_params_lsfo["Mass density (kg/m^3)"],
                        nominal_vessel_params["Nominal Cargo Capacity (kg)"],
                        nominal_vessel_params["Nominal Cargo Capacity (m^3)"],
                        cargo_type=cargo_type,
                    )
                )

            cargo_loss_info_dict[f"Boil-off term"] = calculate_boiloff_term(
                fuel_params_fuel["Boil-off Rate (%/day)"], N_days_to_empty
            )

            cargo_loss_info_dict[f"Fractional loss (mass)"] = cargo_loss_info_dict[
                f"Scaling term (mass)"
            ] * (
                cargo_loss_info_dict[f"Fuel term (mass)"]
                * cargo_loss_info_dict[f"Boil-off term"]
                - 1
            )

            cargo_loss_info_dict[f"Fractional loss (volume)"] = cargo_loss_info_dict[
                f"Scaling term (volume)"
            ] * (
                cargo_loss_info_dict[f"Fuel term (volume)"]
                * cargo_loss_info_dict[f"Boil-off term"]
                - 1
            )

            data.append(cargo_loss_info_dict)

    cargo_loss_info_df = pd.DataFrame(data)
    cargo_loss_info_df.to_csv(f"{top_dir}/tables/cargo_loss_info.csv", index=False)
    return cargo_loss_info_df


def get_parameter_values(
    nominal_vessel_params_df, fuel_params_df, fuel="liquid_hydrogen"
):
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
    e_l = fuel_params_lsfo["eta_eff_mean"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["eta_eff_mean"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]
    fuel_term_mass = calculate_fuel_term(
        L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="mass"
    )
    fuel_term_volume = calculate_fuel_term(
        L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="volume"
    )

    minima = {
        "Vessel": "Minimum",
        "R (nm)": 9e9,
        "P_s_av (MJ/nm)": 9e9,
        "P_av (MW)": 9e9,
        "m_c (kg)": 9e9,
        "V_c (m^3)": 9e9,
        "f_p": 9e9,
        "N_days": 9e9,
        "r_f": 9e9,
        "scaling_term_volume": 9e9,
        "fuel_term_volume": 9e9,
        "scaling_term_mass": 9e9,
        "fuel_term_mass": 9e9,
        "boiloff_term": 9e9,
        "cargo_loss_volume": 9e9,
        "cargo_loss_mass": 9e9,
    }

    maxima = {
        "Vessel": "Maximum",
        "R (nm)": -9e9,
        "P_s_av (MJ/nm)": -9e9,
        "P_av (MW)": -9e9,
        "m_c (kg)": -9e9,
        "V_c (m^3)": -9e9,
        "f_p": -9e9,
        "N_days": -9e9,
        "r_f": -9e9,
        "scaling_term_volume": -9e9,
        "fuel_term_volume": -9e9,
        "boiloff_term": -9e9,
        "scaling_term_mass": -9e9,
        "fuel_term_mass": -9e9,
        "cargo_loss_volume": -9e9,
        "cargo_loss_mass": -9e9,
    }

    data = []
    for vessel in nominal_vessel_params_df["Vessel"]:
        parameter_value_dict = {}
        nominal_vessel_params = extract_vessel_design_params(
            nominal_vessel_params_df, vessel
        )
        parameter_value_dict["Vessel"] = vessel
        parameter_value_dict["R (nm)"] = nominal_vessel_params[
            "Nominal Range (nautical miles)"
        ]
        parameter_value_dict["P_s_av (MJ/nm)"] = nominal_vessel_params[
            "Average Power over Speed (MJ / nautical mile)"
        ]
        parameter_value_dict["P_av (MW)"] = nominal_vessel_params[
            "Average Vessel Power (MW)"
        ]
        parameter_value_dict["m_c (kg)"] = nominal_vessel_params[
            "Nominal Cargo Capacity (kg)"
        ]
        parameter_value_dict["V_c (m^3)"] = nominal_vessel_params[
            "Nominal Cargo Capacity (m^3)"
        ]
        parameter_value_dict["f_p"] = nominal_vessel_params["Pilot Fuel Tank Capacity (m^3)"] / nominal_vessel_params["Nominal Tank Capacity (m^3)"]
        parameter_value_dict["N_days"] = calculate_days_to_empty_tank(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            parameter_value_dict["P_av (MW)"],
        )
        parameter_value_dict["boiloff_term"] = calculate_boiloff_term(
            r_f, parameter_value_dict["N_days"]
        )
        parameter_value_dict["r_f"] = r_f
        parameter_value_dict["scaling_term_volume"] = calculate_scaling_term(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            e_l,
            L_l,
            rho_l,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            cargo_type="volume",
        )
        parameter_value_dict["scaling_term_mass"] = calculate_scaling_term(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            e_l,
            L_l,
            rho_l,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            cargo_type="mass",
        )
        parameter_value_dict["fuel_term_volume"] = fuel_term_volume
        parameter_value_dict["fuel_term_mass"] = fuel_term_mass
        parameter_value_dict["cargo_loss_volume"] = calculate_cargo_loss(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            parameter_value_dict["P_av (MW)"],
            r_f,
            e_l,
            L_l,
            rho_l,
            e_f,
            L_f,
            rho_f,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            parameter_value_dict["f_p"],
            cargo_type="volume",
        )
        parameter_value_dict["cargo_loss_mass"] = calculate_cargo_loss(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            parameter_value_dict["P_av (MW)"],
            r_f,
            e_l,
            L_l,
            rho_l,
            e_f,
            L_f,
            rho_f,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            parameter_value_dict["f_p"],
            cargo_type="mass",
        )

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
    parameter_values_df.to_csv(
        f"{top_dir}/tables/parameter_values_{fuel}.csv", index=False
    )
    print(f"Saved parameter values to {top_dir}/tables/parameter_values_{fuel}.csv")

    return parameter_values_df


def make_mc(
    nominal_vessel_params_df, fuel_params_df, fuel="liquid_hydrogen", N_events=1000
):
    """
    Make Monte Carlo parameter value dictionaries for cargo loss without boil-off, for either mass or volume cargo types.

    Parameters
    ----------

    fuel : str
        Fuel to get parameter values for

    cargo_type : str
        Type of cargo (mass or volume)

    Returns
    -------
    None
    """
    top_dir = get_top_dir()

    # Get a summary of parameter values, along with maxima and minima
    vessel_parameter_values_df = get_parameter_values(
        nominal_vessel_params_df, fuel_params_df, fuel
    )

    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)
    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]
    e_l = fuel_params_lsfo["f_eff_mean"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["f_eff_mean"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]
    fuel_term_mass = calculate_fuel_term(
        L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="mass"
    )
    fuel_term_volume = calculate_fuel_term(
        L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type="volume"
    )

    # Set the relevant MC parameters
    parameters = [
        "R (nm)",
        "P_s_av (MJ/nm)",
        "P_av (MW)",
        "m_c (kg)",
        "V_c (m^3)",
        "f_p"
    ]
    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]

    # Get the bounds on all parameters
    param_mins = {}
    param_maxes = {}

    Rs, P_s_avs, P_avs, m_cs, V_cs, f_ps = get_cargo_loss_parameter_extrema(
        vessel_parameter_values_df
    )
    param_mins["R (nm)"] = Rs[0]
    param_maxes["R (nm)"] = Rs[1]

    param_mins["P_s_av (MJ/nm)"] = P_s_avs[0]
    param_maxes["P_s_av (MJ/nm)"] = P_s_avs[1]

    param_mins["P_av (MW)"] = P_avs[0]
    param_maxes["P_av (MW)"] = P_avs[1]

    param_mins["m_c (kg)"] = m_cs[0]
    param_maxes["m_c (kg)"] = m_cs[1]

    param_mins["V_c (m^3)"] = V_cs[0]
    param_maxes["V_c (m^3)"] = V_cs[1]

    param_mins["f_p"] = f_ps[0]
    param_maxes["f_p"] = f_ps[1]

    data = []
    for event in range(N_events):
        parameter_value_dict = {}
        for parameter in parameters:
            parameter_value_dict[parameter] = random.uniform(
                param_mins[parameter], param_maxes[parameter]
            )

        parameter_value_dict["N_days"] = calculate_days_to_empty_tank(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            parameter_value_dict["P_av (MW)"],
        )
        parameter_value_dict["scaling_term_volume"] = calculate_scaling_term(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            e_l,
            L_l,
            rho_l,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            cargo_type="volume",
        )
        parameter_value_dict["scaling_term_mass"] = calculate_scaling_term(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            e_l,
            L_l,
            rho_l,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            cargo_type="mass",
        )
        parameter_value_dict["fuel_term_volume"] = fuel_term_volume
        parameter_value_dict["fuel_term_mass"] = fuel_term_mass

        parameter_value_dict["cargo_loss_volume"] = calculate_cargo_loss(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            parameter_value_dict["P_av (MW)"],
            r_f,
            e_l,
            L_l,
            rho_l,
            e_f,
            L_f,
            rho_f,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            parameter_value_dict["f_p"],
            cargo_type="volume",
        )
        parameter_value_dict["cargo_loss_mass"] = calculate_cargo_loss(
            parameter_value_dict["R (nm)"],
            parameter_value_dict["P_s_av (MJ/nm)"],
            parameter_value_dict["P_av (MW)"],
            r_f,
            e_l,
            L_l,
            rho_l,
            e_f,
            L_f,
            rho_f,
            parameter_value_dict["m_c (kg)"],
            parameter_value_dict["V_c (m^3)"],
            parameter_value_dict["f_p"],
            cargo_type="mass",
        )

        data.append(parameter_value_dict)

    parameter_values_mc_df = pd.DataFrame(data)
    parameter_values_mc_df.to_csv(
        f"{top_dir}/tables/mc_parameter_values_{fuel}.csv", index=False
    )

    return parameter_values_mc_df


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
    nominal_vessel_params = extract_vessel_design_params(
        nominal_vessel_params_df, vessel
    )
    fuel_params_df = collect_fuel_params(top_dir)
    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)

    P_s_av = nominal_vessel_params["Average Power over Speed (MJ / nautical mile)"]
    P_av = nominal_vessel_params["Average Propulsion Power (MW)"]
    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]
    e_l = fuel_params_lsfo["f_eff_mean"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["f_eff_mean"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]
    m_c = nominal_vessel_params["Nominal Cargo Capacity (kg)"]
    V_c = nominal_vessel_params["Nominal Cargo Capacity (m^3)"]
    f_p = nominal_vessel_params["Pilot Fuel Tank Capacity (m^3)"] / nominal_vessel_params["Nominal Tank Capacity (m^3)"]

    vessel_ranges = range(5000, 55000, 5000)
    N_days_to_empty = []
    total_size_corr_factor = []
    cargo_losses = []
    for R in vessel_ranges:
        N_days = calculate_days_to_empty_tank(R, P_s_av, P_av)
        N_days_to_empty.append(N_days)

        fuel_term = calculate_fuel_term(
            L_l, rho_l, e_l, L_f, rho_f, e_f, cargo_type=cargo_type
        )
        boiloff_term = calculate_boiloff_term(r_f, N_days)

        total_size_corr_factor.append(fuel_term * boiloff_term)

        cargo_losses.append(
            calculate_cargo_loss(
                R,
                P_s_av,
                P_av,
                r_f,
                e_l,
                L_l,
                rho_l,
                e_f,
                L_f,
                rho_f,
                m_c,
                V_c,
                f_p,
                cargo_type=cargo_type,
            )
        )

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


def plot_cargo_loss_vs_param_with_mc(
    fuel,
    parameter_values_df,
    parameter_values_mc_df,
    params_powers,
    param_name_title,
    param_title_short,
    show_median_line,
    x_lim=None,
):
    """
    Plots the cargo loss as a function of the given parameter, using values for both the defined vessels and MC values

    Parameters
    ----------
    fuel : str
        Name of the fuel to consider

    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    parameter_values_mc_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for MC-generated events

    params_powers : dictionary
        Dictionary of the following form:
        {
            param1 (str): power1 (float),
            param2 (str): power2 (float)
        }

    where each param is the name of a model parameter as it's saved in parameter_values_df, and the corresponding power represents the power to raise the param to when multiplied with other params in the dictionary. Eg. the above dictionary would resolve to the following expression: (param1^power1) * (param2^power2).

    param_name_title : str
        Name of the parameter, as it will be shown in the plot

    param_title_short: str
        Short version of the parameter name

    show_median_line : bool
        Boolean parameter to determine whether to show the median line

    x_lim : list of floats
        If x_lim is not None, it should take the form [x_min, x_max]

    Returns
    -------
    None
    """

    # Make copies of the vessel and MC parameter value dfs
    parameter_values_df_cp = parameter_values_df.copy(deep=True)
    parameter_values_mc_df_cp = parameter_values_mc_df.copy(deep=True)

    # Add a new column to parameter_values_df_cp and parameter_values_mc_df_cp with the combined parameter products specified in params_powers
    parameter_values_df_cp["product"] = 1
    parameter_values_mc_df_cp["product"] = 1

    for param in params_powers:
        parameter_values_df_cp["product"] *= (
            parameter_values_df_cp[param] ** params_powers[param]
        )
        parameter_values_mc_df_cp["product"] *= (
            parameter_values_mc_df_cp[param] ** params_powers[param]
        )

    if "m_c (kg)" in params_powers or "V_c (m^3)" in params_powers:
        fig, ax = plt.subplots(figsize=(12, 6))
        axes = [ax, ax]
        plt.title(f"{get_fuel_label(fuel)}: {param_title_short}", fontsize=24)

    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns
        fig.suptitle(f"{get_fuel_label(fuel)}: {param_title_short}", fontsize=24)
    if x_lim is not None:
        axes[0].set_xlim(x_lim[0], x_lim[1])
        axes[1].set_xlim(x_lim[0], x_lim[1])
    axes[0].tick_params(axis="both", which="major", labelsize=16)
    axes[1].tick_params(axis="both", which="major", labelsize=16)

    # Use a qualitative colormap for distinct vessel colors
    vessels = [
        vessel
        for vessel in parameter_values_df_cp["Vessel"]
        if vessel not in {"Minimum", "Maximum"}
    ]
    colors = plt.cm.tab10(
        np.linspace(0, 1, len(vessels))
    )  # Use 'tab10' for categorical data
    vessel_color_map = dict(zip(vessels, colors))

    # Bin the MC points into 10 bins and calculate the average cargo loss in each bin
    num_bins = 10
    if x_lim is None:
        bins = np.linspace(
            parameter_values_mc_df_cp["product"].min(),
            parameter_values_mc_df_cp["product"].max(),
            num_bins + 1,
        )
    else:
        bins = np.linspace(x_lim[0], x_lim[1], num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Calculate bin centers

    # Calculate binned averages for volume and mass
    binned_avg_volume = parameter_values_mc_df_cp.groupby(
        np.digitize(parameter_values_mc_df_cp["product"], bins) - 1
    )["cargo_loss_volume"].median()
    binned_avg_mass = parameter_values_mc_df_cp.groupby(
        np.digitize(parameter_values_mc_df_cp["product"], bins) - 1
    )["cargo_loss_mass"].median()

    # Plot on the first subplot
    if not ("m_c (kg)" in params_powers):
        axes[0].scatter(
            parameter_values_mc_df_cp["product"],
            parameter_values_mc_df_cp["cargo_loss_volume"],
            label="MC",
            color="black",
            marker="o",
            s=3,
            alpha=0.3,
        )
        for vessel in vessels:
            vessel_data = parameter_values_df_cp[
                parameter_values_df_cp["Vessel"] == vessel
            ]
            axes[0].scatter(
                vessel_data["product"],
                vessel_data["cargo_loss_volume"],
                label=vessel,
                color=vessel_color_map[vessel],
                marker="o",
                edgecolors="white",
                linewidth=1,
                s=300,
            )
        # Add the average cargo loss line
        if show_median_line:
            axes[0].plot(
                bin_centers,
                binned_avg_volume[:num_bins],
                label="Median over MC",
                color="red",
                linestyle="-",
                linewidth=4,
            )
        axes[0].set_xlabel(param_name_title, fontsize=20)
        axes[0].set_ylabel("Fractional Cargo Loss (Volume)", fontsize=20)
        ymin, ymax = axes[0].get_ylim()
        cargo_loss_min = parameter_values_df_cp.loc[
            parameter_values_df_cp["Vessel"] == "Minimum", "cargo_loss_volume"
        ].iloc[0]
        cargo_loss_max = parameter_values_df_cp.loc[
            parameter_values_df_cp["Vessel"] == "Maximum", "cargo_loss_volume"
        ].iloc[0]
        axes[0].set_ylim(
            cargo_loss_min - 0.5 * (cargo_loss_max - cargo_loss_min),
            cargo_loss_max + 0.5 * (cargo_loss_max - cargo_loss_min),
        )

    if not ("V_c (m^3)" in params_powers):
        # Plot on the second subplot
        axes[1].scatter(
            parameter_values_mc_df_cp["product"],
            parameter_values_mc_df_cp["cargo_loss_mass"],
            color="black",
            marker="o",
            s=3,
            label="MC",
            alpha=0.3,
        )
        for vessel in vessels:
            if vessel == "Minimum" or vessel == "Maximum":
                continue
            vessel_data = parameter_values_df_cp[
                parameter_values_df_cp["Vessel"] == vessel
            ]
            axes[1].scatter(
                vessel_data["product"],
                vessel_data["cargo_loss_mass"],
                label=vessel,
                color=vessel_color_map[vessel],
                marker="o",
                edgecolors="white",
                linewidth=1,
                s=300,
            )
        # Add the average cargo loss line
        if show_median_line:
            axes[1].plot(
                bin_centers,
                binned_avg_mass[:num_bins],
                label="Median over MC",
                color="red",
                linestyle="-",
                linewidth=4,
            )
        axes[1].set_xlabel(param_name_title, fontsize=20)
        axes[1].set_ylabel("Fractional Cargo Loss (Mass)", fontsize=20)
        ymin, ymax = axes[0].get_ylim()
        cargo_loss_min = parameter_values_df_cp.loc[
            parameter_values_df_cp["Vessel"] == "Minimum", "cargo_loss_mass"
        ].iloc[0]
        cargo_loss_max = parameter_values_df_cp.loc[
            parameter_values_df_cp["Vessel"] == "Maximum", "cargo_loss_mass"
        ].iloc[0]
        axes[1].set_ylim(
            cargo_loss_min - 0.5 * (cargo_loss_max - cargo_loss_min),
            cargo_loss_max + 0.5 * (cargo_loss_max - cargo_loss_min),
        )

    # Combine legends and place them to the right of the second plot
    if "V_c (m^3)" in params_powers:
        lines_labels = [axes[1].get_legend_handles_labels()]
    else:
        lines_labels = [axes[0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1, 0.5), fontsize=16)

    # Adjust layout to prevent overlapping
    if "m_c (kg)" in params_powers or "V_c (m^3)" in params_powers:
        plt.tight_layout(
            rect=[0, 0, 0.68, 1]
        )  # Leave space for the legend on the right
    else:
        plt.tight_layout(
            rect=[0, 0, 0.82, 1]
        )  # Leave space for the legend on the right
    param_name_save_plot = ""
    for param in params_powers:
        param_name_save_plot += "_" + param.split(" ")[0] + str(params_powers[param])
    plt.savefig(f"plots/cargo_loss_vs{param_name_save_plot}_{fuel}.png", dpi=300)
    plt.savefig(f"plots/cargo_loss_vs{param_name_save_plot}_{fuel}.pdf")
    print(f"Plot saved to plots/cargo_loss_vs{param_name_save_plot}_{fuel}.png and .pdf")
    plt.close()


def plot_x_vs_y_with_mc(
    fuel,
    parameter_values_df,
    parameter_values_mc_df,
    x_powers,
    y_powers,
    x_name_title,
    y_name_title,
    x_title_short,
    y_title_short,
    show_best_fit_line,
    color_gradient_param=None,
    color_gradient_title=None,
):
    """
    Plots the cargo loss as a function of the given parameter, using values for both the defined vessels and MC values

    Parameters
    ----------
    fuel : str
        Name of the fuel to consider

    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    parameter_values_mc_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for MC-generated events

    x_powers, y_powers : dictionaries
        Dictionary of the following form for the x and y parameters:
        {
            param1 (str): power1 (float),
            param2 (str): power2 (float)
        }

    where each param is the name of a model parameter as it's saved in parameter_values_df, and the corresponding power represents the power to raise the param to when multiplied with other params in the dictionary. Eg. the above dictionary would resolve to the following expression: (param1^power1) * (param2^power2).

    x_name_title, y_name_title : strs
        Names of the x and y parameters, as they will be shown in the plot

    x_title_short, y_title_short: str
        Short versions of the x and y parameter names

    show_median_line : bool
        Boolean parameter to determine whether to show the median line

    color_gradient_param : str
        Parameter whose value to use for defining a red-blue color gradient for the color of vessel markers

    color_gradient_title : str
        Name to put on the color gradient for the vessel markers

    Returns
    -------
    None
    """

    # Make copies of the vessel and MC parameter value dfs
    parameter_values_df_cp = parameter_values_df.copy(deep=True)
    parameter_values_mc_df_cp = parameter_values_mc_df.copy(deep=True)

    # Add a new column to parameter_values_df_cp and parameter_values_mc_df_cp with the combined parameter products specified in params_powers
    parameter_values_df_cp["x_product"] = 1
    parameter_values_mc_df_cp["x_product"] = 1
    parameter_values_df_cp["y_product"] = 1
    parameter_values_mc_df_cp["y_product"] = 1

    for param in x_powers:
        parameter_values_df_cp["x_product"] *= (
            parameter_values_df_cp[param] ** x_powers[param]
        )
        parameter_values_mc_df_cp["x_product"] *= (
            parameter_values_mc_df_cp[param] ** x_powers[param]
        )

    for param in y_powers:
        parameter_values_df_cp["y_product"] *= (
            parameter_values_df_cp[param] ** y_powers[param]
        )
        parameter_values_mc_df_cp["y_product"] *= (
            parameter_values_mc_df_cp[param] ** y_powers[param]
        )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(
        f"{get_fuel_label(fuel)}: {x_title_short} vs. {y_title_short}", fontsize=24
    )
    ax.tick_params(axis="both", which="major", labelsize=16)

    # Use a qualitative colormap for distinct vessel colors
    vessels = [
        vessel
        for vessel in parameter_values_df_cp["Vessel"]
        if vessel not in {"Minimum", "Maximum"}
    ]

    if color_gradient_param is None:
        colors = plt.cm.tab10(
            np.linspace(0, 1, len(vessels))
        )  # Use 'tab10' for categorical data
        vessel_color_map = dict(zip(vessels, colors))

    else:
        # Normalize the values of the color_gradient_param for the colormap
        norm = plt.Normalize(
            parameter_values_df_cp[color_gradient_param].min(),
            parameter_values_df_cp[color_gradient_param].max(),
        )
        cmap = plt.cm.coolwarm  # Use a red-blue colormap

        # Create a color map for vessels based on the color_gradient_param
        vessel_color_map = {
            vessel: cmap(
                norm(
                    parameter_values_df_cp.loc[
                        parameter_values_df_cp["Vessel"] == vessel, color_gradient_param
                    ].iloc[0]
                )
            )
            for vessel in vessels
        }

        # Add a colorbar for the gradient
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # ScalarMappable needs a dummy array
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02)
        cbar.set_label(color_gradient_title, fontsize=18)
        cbar.ax.tick_params(labelsize=16)

    # Plot the vessel and MC data as scatterplots
    ax.scatter(
        parameter_values_mc_df_cp["x_product"],
        parameter_values_mc_df_cp["y_product"],
        label="MC",
        color="black",
        marker="o",
        s=3,
        alpha=0.3,
    )
    for vessel in vessels:
        vessel_data = parameter_values_df_cp[parameter_values_df_cp["Vessel"] == vessel]
        ax.scatter(
            vessel_data["x_product"],
            vessel_data["y_product"],
            label=vessel,
            color=vessel_color_map[vessel],
            marker="o",
            edgecolors="white",
            linewidth=1,
            s=300,
        )

    # Add a best fit line
    if show_best_fit_line:
        # Use only the vessel data for regression
        vessel_data_x = parameter_values_df_cp["x_product"]
        vessel_data_y = parameter_values_df_cp["y_product"]

        # Fit a linear regression model to vessel data
        coefficients = np.polyfit(
            vessel_data_x, vessel_data_y, 1
        )  # Linear fit: degree=1
        best_fit_line = np.poly1d(coefficients)  # Create a polynomial function

        # Generate x values for the best fit line
        x_fit = np.linspace(vessel_data_x.min(), vessel_data_x.max(), 500)
        y_fit = best_fit_line(x_fit)

        # Plot the best fit line
        ax.plot(
            x_fit,
            y_fit,
            color="red",
            linestyle="--",
            linewidth=4,
            label="Best Fit Line (Vessels)",
        )

    ax.set_xlabel(x_name_title, fontsize=20)
    ax.set_ylabel(y_name_title, fontsize=20)

    # Place the legend to the right of the plot
    lines_labels = [ax.get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="center right", bbox_to_anchor=(1, 0.5), fontsize=16)

    # Adjust layout to prevent overlapping
    plt.tight_layout(rect=[0, 0, 0.68, 1])  # Leave space for the legend on the right
    param_name_save_plot = ""
    y_param_keys = list(y_powers.keys())
    param_name_save_plot += y_param_keys[0].split(" ")[0] + str(
        y_powers[y_param_keys[0]]
    )

    if len(y_param_keys) > 1:
        for param in y_powers[1:]:
            param_name_save_plot += "_" + param.split(" ")[0] + str(y_powers[param])

    param_name_save_plot += "_vs"
    for param in x_powers:
        param_name_save_plot += "_" + param.split(" ")[0] + str(x_powers[param])

    plt.savefig(f"plots/{param_name_save_plot}_{fuel}.png", dpi=300)
    plt.savefig(f"plots/{param_name_save_plot}_{fuel}.pdf")
    print(f"Plot saved to plots/{param_name_save_plot}_{fuel}.png and .pdf")
    plt.close()


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
        raise ValueError("Invalid cargo_type. Please choose 'volume' or 'mass'.")

    # Remove "Minimum" and "Maximum" rows from the vessel data
    vessel_data = parameter_values_df[
        ~parameter_values_df["Vessel"].isin(["Minimum", "Maximum"])
    ]

    # Extract relevant columns
    N_days = vessel_data["N_days"]
    scaling_term = vessel_data[scaling_term_col]
    fuel_term = parameter_values_df.loc[
        parameter_values_df["Vessel"] == "Minimum", fuel_term_col
    ].iloc[0]
    r_f = parameter_values_df.loc[
        parameter_values_df["Vessel"] == "Minimum", "r_f"
    ].iloc[0]
    cargo_loss = vessel_data[cargo_loss_col]

    # Define grid for heatmap with extended ranges
    N_days_min, N_days_max = N_days.min(), N_days.max()
    scaling_term_min, scaling_term_max = scaling_term.min(), scaling_term.max()

    N_days_range = N_days_max - N_days_min
    scaling_term_range = scaling_term_max - scaling_term_min

    N_days_grid = np.linspace(
        N_days_min - 0.1 * N_days_range, N_days_max + 0.1 * N_days_range, 100
    )
    scaling_term_grid = np.linspace(
        scaling_term_min - 0.1 * scaling_term_range,
        scaling_term_max + 0.1 * scaling_term_range,
        100,
    )

    N_days_mesh, scaling_term_mesh = np.meshgrid(N_days_grid, scaling_term_grid)
    cargo_loss_mesh = scaling_term_mesh * (
        fuel_term / (1 - r_f / 100) ** N_days_mesh - 1
    )

    # Find the location of the absolute minimum cargo loss
    min_loss_index = np.unravel_index(
        np.nanargmin(cargo_loss_mesh), cargo_loss_mesh.shape
    )
    min_N_days = N_days_mesh[min_loss_index]
    min_scaling_term = scaling_term_mesh[min_loss_index]
    min_cargo_loss = cargo_loss_mesh[min_loss_index]

    # Plot the heatmap
    plt.figure(figsize=(15, 8))
    plt.contourf(
        N_days_mesh, scaling_term_mesh, cargo_loss_mesh, levels=20, cmap="viridis"
    )
    cbar = plt.colorbar(label=f"Cargo Loss ({cargo_type})")
    cbar.set_label(f"Cargo Loss ({cargo_type})", fontsize=20)
    cbar.ax.tick_params(labelsize=18)

    # Overlay vessel points with shades of orange/red/magenta
    colors = plt.cm.magma(np.linspace(0.2, 0.8, len(vessel_data)))
    for i, (vessel, n_days, scale, color) in enumerate(
        zip(vessel_data["Vessel"], N_days, scaling_term, colors)
    ):
        plt.scatter(n_days, scale, color=color, label=vessel, alpha=0.8)

    # Highlight the absolute minimum cargo loss
    # plt.scatter(min_N_days, min_scaling_term, color="red", s=200, marker="*", label="Minimum Cargo Loss")

    # Add labels and title
    plt.xlabel("Days to empty tank", fontsize=20)
    plt.ylabel(
        f"Scaling term ({cargo_type})".replace("_", " ").capitalize(), fontsize=20
    )
    plt.title(f"Fuel: {get_fuel_label(fuel)}. Cargo type: {cargo_type}", fontsize=22)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.legend(fontsize=18, bbox_to_anchor=(1.25, 1))

    # Adjust grid to span parameter ranges proportionally (rectangular grid)
    plt.gca().set_aspect("auto")

    # Show the plot
    plt.tight_layout()
    plt.savefig(
        f"plots/cargo_loss_vs_dimensionless_params_{fuel}_{cargo_type}.png", dpi=300
    )
    plt.savefig(f"plots/cargo_loss_vs_dimensionless_params_{fuel}_{cargo_type}.pdf")
    plt.close()


def get_cargo_loss_parameter_extrema(parameter_values_df):
    """
    Gets the upper and lower extrema of tunable vessel design parameters used to evaluate the fractional cargo loss.

    Parameters
    ----------
    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    Returns
    -------
    Rs : list of floats
        Extreme values of the vessel design range, in nautical miles

    P_s_avs : list of floats
        Extreme values of the average ratio of main engine power to vessel speed, in MJ / nautical mile

    P_avs : list of floats
        Extreme values of the average main engine power, in MW (MJ/s)

    m_cs : float
        Extreme values of the mass cargo capacity of the vessel, in kg

    V_cs : float
        Extreme values of the volume cargo capacity of the vessel, in m^3
    
    f_ps : float
        Extreme values of the pilot tank to nominal tank size ratio
    """
    def get_bounds(parameter_str):
        param_min = parameter_values_df.loc[
            parameter_values_df["Vessel"] == "Minimum", parameter_str
        ].iloc[0]
        param_max = parameter_values_df.loc[
            parameter_values_df["Vessel"] == "Maximum", parameter_str
        ].iloc[0]
        return [param_min, param_max]

    Rs = get_bounds("R (nm)")
    P_s_avs = get_bounds("P_s_av (MJ/nm)")
    P_avs = get_bounds("P_av (MW)")
    m_cs = get_bounds("m_c (kg)")
    V_cs = get_bounds("V_c (m^3)")
    f_ps = get_bounds("f_p")

    return Rs, P_s_avs, P_avs, m_cs, V_cs, f_ps


def minimize_cargo_loss(fuel, fuel_params_df, parameter_values_df, cargo_type):
    """
    Finds the values of tunable vessel design parameters (R, P_s_av, P_av, m_c, V_c) that minimize fractional cargo loss for the given cargo type.

    Parameters
    ----------
    fuel : str
        Name of the fuel to consider

    fuel_params_df : pandas.DataFrame
        DataFrame containing the fuel parameters for all fuels.

    cargo_type : str
        Type of cargo (mass or volume)

    Returns
    -------
    R_min : float
        Vessel design range that minimizes cargo loss, in nautical miles

    P_s_av_min : float
        Average ratio of main engine power to vessel speed that minimizes cargo loss, in MJ / nautical mile

    P_av_min : float
        Average main engine power that minimizes cargo loss, in MW (MJ/s)

    m_c_min : float
        Mass cargo capacity of the vessel that minimizes cargo loss, in kg

    V_c_min : float
        Volume cargo capacity of the vessel that minimizes cargo loss, in m^3

    cargo_loss_min : float
        Minimum cargo loss
    """

    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)

    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]
    e_l = fuel_params_lsfo["f_eff_mean"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["f_eff_mean"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]

    Rs, P_s_avs, P_avs, m_cs, V_cs, f_ps = get_cargo_loss_parameter_extrema(
        parameter_values_df
    )

    def objective_function(params):
        R, P_s_av, P_av, m_c, V_c, f_p = params

        return calculate_cargo_loss(
            R,
            P_s_av,
            P_av,
            r_f,
            e_l,
            L_l,
            rho_l,
            e_f,
            L_f,
            rho_f,
            m_c,
            V_c,
            f_p,
            cargo_type=cargo_type,
        )

    result_minimize = minimize(
        objective_function,
        x0=[mean(Rs), mean(P_s_avs), mean(P_avs), mean(m_cs), mean(V_cs), mean(f_ps)],
        bounds=[
            (Rs[0], Rs[1]),
            (P_s_avs[0], P_s_avs[1]),
            (P_avs[0], P_avs[1]),
            (m_cs[0], m_cs[1]),
            (V_cs[0], V_cs[1]),
            (f_ps[0], f_ps[1]),
        ],
    )

    bounds = [
        (Rs[0], Rs[1]),
        (P_s_avs[0], P_s_avs[1]),
        (P_avs[0], P_avs[1]),
        (m_cs[0], m_cs[1]),
        (V_cs[0], V_cs[1]),
        (f_ps[0], f_ps[1]),
    ]
    result_global = differential_evolution(objective_function, bounds)
    result_local = minimize(objective_function, x0=result_global.x, bounds=bounds)

    R_min = result_local.x[0]
    P_s_av_min = result_local.x[1]
    P_av_min = result_local.x[2]
    m_c_min = result_local.x[4]
    V_c_min = result_local.x[5]
    f_p_min = result_local.x[6]
    cargo_loss_min = result_local.fun

    return R_min, P_s_av_min, P_av_min, m_c_min, V_c_min, f_p_min, cargo_loss_min


def plot_parameter_profiles(
    fuel,
    fuel_params_df,
    parameter_values_df,
    R_min,
    P_s_av_min,
    P_av_min,
    m_c_min,
    V_c_min,
    f_p_min,
    cargo_loss_min,
    cargo_type,
):
    """
    Plot profiles of each vessel design parameter within its extrema, with the values of all other tunable parameters set to the values that minimize the cargo loss.

    Parameters
    ----------
    fuel : str
        Name of the fuel to consider

    fuel_params_df : pandas.DataFrame
        DataFrame containing the fuel parameters for all fuels.

    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    R_min : float
        Vessel design range that minimizes cargo loss, in nautical miles

    P_s_av_min : float
        Average ratio of main engine power to vessel speed that minimizes cargo loss, in MJ / nautical mile

    P_av_min : float
        Average main engine power that minimizes cargo loss, in MW (MJ/s)

    m_c_min : float
        Mass cargo capacity of the vessel that minimizes cargo loss, in kg

    V_c_min : float
        Volume cargo capacity of the vessel that minimizes cargo loss, in m^3

    cargo_loss_min : float
        Minimum cargo loss

    cargo_type : str
        Type of cargo (mass or volume)

    Returns
    -------
    None
    """

    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)

    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]
    e_l = fuel_params_lsfo["f_eff_mean"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["f_eff_mean"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]

    Rs, P_s_avs, P_avs, m_cs, V_cs, f_ps = get_cargo_loss_parameter_extrema(
        parameter_values_df
    )

    # Set up parameter names and their corresponding ranges
    parameters = {
        "Range (nautical miles)": (Rs, R_min, "R (nm)"),
        "Average Power / Speed (MJ/nm)": (P_s_avs, P_s_av_min, "P_s_av (MJ/nm)"),
        "Average power (MW)": (P_avs, P_av_min, "P_av (MW)"),
        "Mass cargo capacity (kg)": (m_cs, m_c_min, "m_c (kg)"),
        "Volume cargo capacity (m$^3$)": (V_cs, V_c_min, "V_c (m^3)"),
        "Pilot tank size ratio": (f_ps, f_p_min, "f_p")
    }

    # Filter parameters based on cargo type
    if cargo_type == "volume":
        parameters.pop("Mass cargo capacity (kg)", None)
    elif cargo_type == "mass":
        parameters.pop("Volume cargo capacity (m^3)", None)

    # Create subplots
    fig, axes = plt.subplots(
        1, len(parameters), figsize=(5 * len(parameters), 6), sharey=True
    )
    fig.suptitle(f"Fuel: {get_fuel_label(fuel)}", fontsize=22)

    all_handles, all_labels = [], []

    i = 0
    for ax, (param_name, (param_range, param_min, param_shortname)) in zip(
        axes, parameters.items()
    ):
        ax.tick_params(axis="both", which="major", labelsize=18)

        param_values = np.linspace(param_range[0], param_range[1], 1000)
        cargo_losses_profiled = np.zeros(0)

        for val in param_values:
            cargo_loss_profiled = calculate_cargo_loss(
                R_min if param_name != "Range (nautical miles)" else val,
                P_s_av_min if param_name != "Average Power / Speed (MJ/nm)" else val,
                P_av_min if param_name != "Average power (MW)" else val,
                r_f,
                e_l,
                L_l,
                rho_l,
                e_f,
                L_f,
                rho_f,
                m_c_min if param_name != "Mass cargo capacity (kg)" else val,
                V_c_min if param_name != "Volume cargo capacity (m$^3$)" else val,
                f_p_min if param_name != "Pilot tank size ratio" else val,
                cargo_type=cargo_type,
            )
            cargo_losses_profiled = np.append(
                cargo_losses_profiled, cargo_loss_profiled
            )

        # Plot cargo loss vs parameter
        ax.plot(param_values, cargo_losses_profiled, color="black")
        line = ax.axvline(param_min, color="red", linestyle="--")
        if i == 0:
            all_handles.append(line)
            all_labels.append("Minimizing Value")

        for vessel in parameter_values_df["Vessel"]:
            if vessel == "Minimum" or vessel == "Maximum":
                continue
            param_value = parameter_values_df.loc[
                parameter_values_df["Vessel"] == vessel, param_shortname
            ].iloc[0]
            cargo_loss_value = parameter_values_df.loc[
                parameter_values_df["Vessel"] == vessel, f"cargo_loss_{cargo_type}"
            ].iloc[0]

            (point,) = ax.plot(param_value, cargo_loss_value, "o")

            if i == 0:
                all_handles.append(point)
                all_labels.append(vessel)
        i += 1

        ax.set_xlabel(param_name, fontsize=20)

    # Set shared vertical axis label
    fig.text(
        0.03,
        0.5,
        f"Fractional Cargo Loss ({cargo_type})",
        va="center",
        rotation="vertical",
        fontsize=20,
    )

    # Add a legend below all plots
    handles, labels = axes[-1].get_legend_handles_labels()
    all_handles.extend(handles)
    all_labels.extend(labels)
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        fontsize=16,
        ncol=6,
        bbox_to_anchor=(0.5, -0.1),
    )

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    ymin, ymax = ax.get_ylim()
    if ymax > 1:
        ax.set_ylim(-1, 1)

    filename_png = f"plots/parameter_profiles_minimum_{fuel}_{cargo_type}.png"
    filename_pdf = f"plots/parameter_profiles_minimum_{fuel}_{cargo_type}.pdf"
    print(f"Saving figure to {filename_png}")
    plt.savefig(filename_png, dpi=300, bbox_inches="tight")
    plt.savefig(filename_pdf, bbox_inches="tight")
    plt.close()


def plot_parameter_profiles_vessel(
    fuel, vessel_to_profile, fuel_params_df, parameter_values_df, cargo_type="mass"
):
    """
    Plot profiles of each vessel design parameter within its extrema, with the values of all other tunable parameters set to their values for the given vessel.

    Parameters
    ----------
    fuel : str
        Name of the fuel to consider

    vessel_to_profile : str
        Name of the vessel to consider

    fuel_params_df : pandas.DataFrame
        DataFrame containing the fuel parameters for all fuels.

    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    cargo_loss_min : float
        Minimum cargo loss

    cargo_type : str
        Type of cargo (mass or volume)

    Returns
    -------
    None
    """

    fuel_params_lsfo = extract_fuel_params(fuel_params_df, "lsfo")
    fuel_params_fuel = extract_fuel_params(fuel_params_df, fuel)

    r_f = fuel_params_fuel["Boil-off Rate (%/day)"]
    e_l = fuel_params_lsfo["f_eff_mean"]
    L_l = fuel_params_lsfo["Lower Heating Value (MJ / kg)"]
    rho_l = fuel_params_lsfo["Mass density (kg/m^3)"]
    e_f = fuel_params_fuel["f_eff_mean"]
    L_f = fuel_params_fuel["Lower Heating Value (MJ / kg)"]
    rho_f = fuel_params_fuel["Mass density (kg/m^3)"]

    R_vessel = parameter_values_df.loc[
        parameter_values_df["Vessel"] == vessel_to_profile, "R (nm)"
    ].iloc[0]
    P_s_av_vessel = parameter_values_df.loc[
        parameter_values_df["Vessel"] == vessel_to_profile, "P_s_av (MJ/nm)"
    ].iloc[0]
    P_av_vessel = parameter_values_df.loc[
        parameter_values_df["Vessel"] == vessel_to_profile, "P_av (MW)"
    ].iloc[0]
    m_c_vessel = parameter_values_df.loc[
        parameter_values_df["Vessel"] == vessel_to_profile, "m_c (kg)"
    ].iloc[0]
    V_c_vessel = parameter_values_df.loc[
        parameter_values_df["Vessel"] == vessel_to_profile, "V_c (m^3)"
    ].iloc[0]
    f_p_vessel = parameter_values_df.loc[
        parameter_values_df["Vessel"] == vessel_to_profile, "f_p"
    ].iloc[0]

    Rs, P_s_avs, P_avs, m_cs, V_cs, f_ps = get_cargo_loss_parameter_extrema(
        parameter_values_df
    )

    # Set up parameter names and their corresponding ranges
    parameters = {
        "Range (nautical miles)": (Rs, R_vessel, "R (nm)"),
        "Average Power / Speed (MJ/nm)": (P_s_avs, P_s_av_vessel, "P_s_av (MJ/nm)"),
        "Average power (MW)": (P_avs, P_av_vessel, "P_av (MW)"),
        "Mass cargo capacity (kg)": (m_cs, m_c_vessel, "m_c (kg)"),
        "Volume cargo capacity (m$^3$)": (V_cs, V_c_vessel, "V_c (m^3)"),
        "Pilot tank size ratio": (f_ps, f_p_vessel, "f_p"),
    }

    # Filter parameters based on cargo type
    if cargo_type == "volume":
        parameters.pop("Mass cargo capacity (kg)", None)
    elif cargo_type == "mass":
        parameters.pop("Volume cargo capacity (m$^3$)", None)

    # Create subplots
    fig, axes = plt.subplots(
        1, len(parameters), figsize=(5 * len(parameters), 6), sharey=True
    )
    fig.suptitle(
        f"Fuel: {get_fuel_label(fuel)}. Vessel profiled: {vessel_to_profile}",
        fontsize=22,
    )

    all_handles, all_labels = [], []

    i = 0
    for ax, (param_name, (param_range, param_vessel, param_shortname)) in zip(
        axes, parameters.items()
    ):
        ax.tick_params(axis="both", which="major", labelsize=18)

        param_values = np.linspace(param_range[0], param_range[1], 1000)
        cargo_losses_profiled = np.zeros(0)

        param_min = 9e9
        cargo_loss_profiled_min = 9e9

        for val in param_values:
            cargo_loss_profiled = calculate_cargo_loss(
                R_vessel if param_name != "Range (nautical miles)" else val,
                P_s_av_vessel if param_name != "Average Power / Speed (MJ/nm)" else val,
                P_av_vessel if param_name != "Average power (MW)" else val,
                r_f,
                e_l,
                L_l,
                rho_l,
                e_f,
                L_f,
                rho_f,
                m_c_vessel if param_name != "Mass cargo capacity (kg)" else val,
                V_c_vessel if param_name != "Volume cargo capacity (m$^3$)" else val,
                f_p_vessel if param_name != "Pilot tank size ratio" else val,
                cargo_type=cargo_type,
            )
            if cargo_loss_profiled < cargo_loss_profiled_min:
                cargo_loss_profiled_min = cargo_loss_profiled
                param_min = val
            cargo_losses_profiled = np.append(
                cargo_losses_profiled, cargo_loss_profiled
            )

        # Plot cargo loss vs parameter
        ax.plot(param_values, cargo_losses_profiled, color="black")
        line_vessel = ax.axvline(param_vessel, color="blue", linestyle="--")
        (star_param_min,) = ax.plot(
            param_min, cargo_loss_profiled_min, "*", markersize=20, color="red"
        )
        if i == 0:
            all_handles.append(line_vessel)
            all_labels.append(f"Value for Profiled Vessel: {param_vessel:.2e}")
            all_handles.append(star_param_min)
            all_labels.append(f"Minimizing value: {param_min:.2e}")

        for vessel in parameter_values_df["Vessel"]:
            if vessel == "Minimum" or vessel == "Maximum":
                continue
            param_value = parameter_values_df.loc[
                parameter_values_df["Vessel"] == vessel, param_shortname
            ].iloc[0]
            cargo_loss_value = parameter_values_df.loc[
                parameter_values_df["Vessel"] == vessel, f"cargo_loss_{cargo_type}"
            ].iloc[0]

            (point,) = ax.plot(param_value, cargo_loss_value, "o")

            if i == 0:
                all_handles.append(point)
                all_labels.append(vessel)
        i += 1

        ax.set_xlabel(param_name, fontsize=20)

    # Set shared vertical axis label
    fig.text(
        0.03,
        0.5,
        f"Fractional Cargo Loss ({cargo_type})",
        va="center",
        rotation="vertical",
        fontsize=20,
    )

    # Add a legend below all plots
    handles, labels = axes[-1].get_legend_handles_labels()
    all_handles.extend(handles)
    all_labels.extend(labels)
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        fontsize=16,
        ncol=6,
        bbox_to_anchor=(0.5, -0.1),
    )

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.96])
    ymin, ymax = ax.get_ylim()
    if ymax > 1:
        ax.set_ylim(-1, 1)

    vessel_to_profile_save = (
        vessel_to_profile.replace(" ", "_").replace("(", "").replace(")", "")
    )
    filename_png = (
        f"plots/parameter_profiles_{vessel_to_profile_save}_{fuel}_{cargo_type}.png"
    )
    filename_pdf = (
        f"plots/parameter_profiles_{vessel_to_profile_save}_{fuel}_{cargo_type}.pdf"
    )
    print(f"Saving figure to {filename_png}")
    plt.savefig(filename_png, dpi=300, bbox_inches="tight")
    plt.savefig(filename_pdf, bbox_inches="tight")
    plt.close()


def plot_parameter_profiles_all_vessels(
    fuel, fuel_params_df, parameter_values_df, cargo_type="mass"
):
    """
    Creates plots of parameter profiles for each vessel.

    Parameters
    ----------
    fuel : str
        Name of the fuel to consider

    fuel_params_df : pandas.DataFrame
        DataFrame containing the fuel parameters for all fuels.

    parameter_values_df : pd.DataFrame
        Dataframe containing parameters related to cargo loss for each vessel, along with maxima and minima

    cargo_loss_min : float
        Minimum cargo loss

    cargo_type : str
        Type of cargo (mass or volume)

    Returns
    -------
    None
    """

    for vessel in parameter_values_df["Vessel"]:
        if vessel == "Minimum" or vessel == "Maximum":
            continue

        plot_parameter_profiles_vessel(
            fuel, vessel, fuel_params_df, parameter_values_df, cargo_type
        )


def main():
    top_dir = get_top_dir()

    # Collect vessel design and fuel parameters for each vessel+fuel
    nominal_vessel_params_df = collect_nominal_vessel_design_params(top_dir)
    fuel_params_df = collect_fuel_params(top_dir)

    # Get cargo loss info
#    cargo_loss_info_df = get_cargo_loss_info(nominal_vessel_params_df, fuel_params_df)

    # Plot cargo loss as a function of range
    # plot_cargo_loss_vs_range("Tanker (35k DWT)", "liquid_hydrogen", "volume")

    ################ Make plots of cargo loss vs. various parameters ################

    params_powers = {
        "Vessel Design Range (nautical miles)": {"R (nm)": 1},
        "Average Power / Speed (MJ/nm)": {"P_s_av (MJ/nm)": 1},
        "Volume cargo capacity (m$^3$)": {"V_c (m^3)": 1},
        "Mass cargo capacity (kg)": {"m_c (kg)": 1},
        "Average Propulsion Power (MW)": {"P_av (MW)": 1},
        r"R $\times$ (P/s)$_\text{av}$ (MJ)": {"R (nm)": 1, "P_s_av (MJ/nm)": 1},
        r"R $\times$ (P/s)$_\text{av}$ / V$_c$ (MJ/m$^3$)": {
            "R (nm)": 1,
            "P_s_av (MJ/nm)": 1,
            "V_c (m^3)": -1,
        },
        r"R $\times$ (P/s)$_\text{av}$ / m$_c$ (MJ/kg)": {
            "R (nm)": 1,
            "P_s_av (MJ/nm)": 1,
            "m_c (kg)": -1,
        },
        "Days to Empty Tank": {"N_days": 1},
    }

    param_titles_short = {
        "Vessel Design Range (nautical miles)": "R",
        "Average Power / Speed (MJ/nm)": r"(P/s)$_\text{av}$",
        "Volume cargo capacity (m$^3$)": r"V$_c$",
        "Mass cargo capacity (kg)": r"m$_c$",
        "Average Propulsion Power (MW)": r"P$_\text{av}$",
        r"R $\times$ (P/s)$_\text{av}$ (MJ)": r"R $\times$ (P/s)$_\text{av}$",
        r"R $\times$ (P/s)$_\text{av}$ / V$_c$ (MJ/m$^3$)": r"R $\times$ (P/s)$_\text{av}$ / V$_c$",
        r"R $\times$ (P/s)$_\text{av}$ / m$_c$ (MJ/kg)": r"R $\times$ (P/s)$_\text{av}$ / m$_c$",
        "Days to Empty Tank": "N",
    }

    show_median_line = {
        "Vessel Design Range (nautical miles)": True,
        "Average Power / Speed (MJ/nm)": True,
        "Volume cargo capacity (m$^3$)": True,
        "Mass cargo capacity (kg)": True,
        "Average Propulsion Power (MW)": True,
        r"R $\times$ (P/s)$_\text{av}$ (MJ)": True,
        r"R $\times$ (P/s)$_\text{av}$ / V$_c$ (MJ/m$^3$)": False,
        r"R $\times$ (P/s)$_\text{av}$ / m$_c$ (MJ/kg)": False,
        "Days to Empty Tank": {"N_days": True},
    }

    x_lims = {
        "Vessel Design Range (nautical miles)": None,
        "Average Power / Speed (MJ/nm)": None,
        "Volume cargo capacity (m$^3$)": None,
        "Mass cargo capacity (kg)": None,
        "Average Propulsion Power (MW)": None,
        r"R $\times$ (P/s)$_\text{av}$ (MJ)": None,
        r"R $\times$ (P/s)$_\text{av}$ / V$_c$ (MJ/m$^3$)": None,
        r"R $\times$ (P/s)$_\text{av}$ / m$_c$ (MJ/kg)": None,
        "Days to Empty Tank": [0, 500],
    }

    for fuel in ["methanol", "ammonia", "liquid_hydrogen", "lng"]:
        parameter_values_df = get_parameter_values(
            nominal_vessel_params_df, fuel_params_df, fuel
        )
        
        
        parameter_values_mc_df = make_mc(
            nominal_vessel_params_df, fuel_params_df, fuel, 10000
        )

        plot_x_vs_y_with_mc(
            fuel,
            parameter_values_df,
            parameter_values_mc_df,
            {"P_s_av (MJ/nm)": 1},
            {"R (nm)": 1},
            "Average Power / Speed (MJ/nm)",
            "Vessel Design Range (nautical miles)",
            r"(P/s)$_\text{av}$",
            "R",
            show_best_fit_line=False,
        )

        plot_x_vs_y_with_mc(
            fuel,
            parameter_values_df,
            parameter_values_mc_df,
            {"P_s_av (MJ/nm)": 1},
            {"V_c (m^3)": 1},
            "Average Power / Speed (MJ/nm)",
            "Volume cargo capacity (m$^3$)",
            r"(P/s)$_\text{av}$",
            r"V$_c$",
            show_best_fit_line=True,
        )

        plot_x_vs_y_with_mc(
            fuel,
            parameter_values_df,
            parameter_values_mc_df,
            {"P_s_av (MJ/nm)": 1},
            {"m_c (kg)": 1},
            "Average Power / Speed (MJ/nm)",
            "Mass cargo capacity (kg)",
            r"(P/s)$_\text{av}$",
            r"m$_c$",
            show_best_fit_line=True,
        )

        plot_x_vs_y_with_mc(
            fuel,
            parameter_values_df,
            parameter_values_mc_df,
            {"R (nm)": 1},
            {"V_c (m^3)": 1},
            "Vessel Design Range (nautical miles)",
            "Volume cargo capacity (m$^3$)",
            "R",
            r"V$_c$",
            show_best_fit_line=True,
        )

        plot_x_vs_y_with_mc(
            fuel,
            parameter_values_df,
            parameter_values_mc_df,
            {"R (nm)": 1},
            {"m_c (kg)": 1},
            "Vessel Design Range (nautical miles)",
            "Mass cargo capacity (kg)",
            "R",
            r"m$_c$",
            show_best_fit_line=True,
        )

        plot_x_vs_y_with_mc(
            fuel,
            parameter_values_df,
            parameter_values_mc_df,
            {"R (nm)": 1, "P_s_av (MJ/nm)": 1},
            {"V_c (m^3)": 1},
            r"R $\times$ (P/s)$_\text{av}$ [Energy capacity] (MJ)",
            "Volume cargo capacity (m$^3$)",
            r"R $\times$ (P/s)$_\text{av}$",
            r"V$_c$",
            show_best_fit_line=True,
            color_gradient_param="cargo_loss_volume",
            color_gradient_title="Fractional Cargo Loss (volume)",
        )

        plot_x_vs_y_with_mc(
            fuel,
            parameter_values_df,
            parameter_values_mc_df,
            {"R (nm)": 1, "P_s_av (MJ/nm)": 1},
            {"m_c (kg)": 1},
            r"R $\times$ (P/s)$_\text{av}$ [Energy capacity] (MJ)",
            "Mass cargo capacity (kg)",
            r"R $\times$ (P/s)$_\text{av}$",
            r"m$_c$",
            show_best_fit_line=True,
            color_gradient_param="cargo_loss_mass",
            color_gradient_title="Fractional Cargo Loss (mass)",
        )

        for param_title in params_powers:
            plot_cargo_loss_vs_param_with_mc(
                fuel,
                parameter_values_df,
                parameter_values_mc_df,
                params_powers[param_title],
                param_title,
                param_titles_short[param_title],
                show_median_line=show_median_line[param_title],
                x_lim=x_lims[param_title],
            )

#    for fuel in fuel_params_df["Fuel"]:
#        if fuel == "lsfo":
#            continue
#
#        print(fuel)
#        print()
#
#        parameter_values_df = get_parameter_values(nominal_vessel_params_df, fuel_params_df, fuel)
#        plot_cargo_loss_vs_dimensionless_params(parameter_values_df, fuel, "volume")
#        plot_cargo_loss_vs_dimensionless_params(parameter_values_df, fuel, "mass")
#
#        R_min, P_s_av_min, P_av_min, m_c_min, V_c_min, cargo_loss_min = minimize_cargo_loss(fuel, fuel_params_df, parameter_values_df, "mass")
#        plot_parameter_profiles(fuel, fuel_params_df, parameter_values_df, R_min, P_s_av_min, P_av_min, m_c_min, V_c_min, cargo_loss_min, "mass")
#        plot_parameter_profiles_all_vessels(fuel, fuel_params_df, parameter_values_df, "mass")
#
#        R_min, P_s_av_min, P_av_min, m_c_min, V_c_min, cargo_loss_min = minimize_cargo_loss(fuel, fuel_params_df, parameter_values_df, "volume")
#        plot_parameter_profiles(fuel, fuel_params_df, parameter_values_df, R_min, P_s_av_min, P_av_min, m_c_min, V_c_min, cargo_loss_min, "volume")
#        plot_parameter_profiles_all_vessels(fuel, fuel_params_df, parameter_values_df, "volume")



if __name__ == "__main__":
    main()

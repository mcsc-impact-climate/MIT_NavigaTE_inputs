"""
Date: Oct. 17, 2025
Author: danikae
Purpose: Evaluate emission rates for each vessel and fuel for commodities included in the Freight Analysis Framework.
"""

from common_tools import get_top_dir, get_fuel_density, ensure_directory_exists
from get_sf_distributions import get_commodities_class, get_commodity_info, get_gas_carrier_sfs, calculate_crude_petrol_sfs
from modify_tanks_and_cargo_capacity import get_fuel_info_dict, calculate_modified_cargo_capacities_no_sf,  calculate_modified_cargo_capacities_with_sf, get_eff_dict, get_tank_size_factors, get_route_properties
import pandas as pd
from parse import parse
import os
import shutil

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

fuels = [
    "ammonia",
    "methanol",
    "liquid_hydrogen",
    "compressed_hydrogen",
    "lng",
    "FTdiesel",
    "lsfo",
    "bio_cfp"
]

# Constants for fuel densities (kg/L)
GASOLINE_DENSITY = 0.749  # kg/L (GREET 2024)
FUEL_OIL_DENSITY = get_fuel_density("lsfo")  # kg/L
LNG_DENSITY = get_fuel_density("lng") # kg/L

# Calculate stowage factors (m³/tonne = 1 / density in kg/L)
GASOLINE_SF = 1 / GASOLINE_DENSITY
FUEL_OIL_SF = 1 / FUEL_OIL_DENSITY
LNG_SF = 1 / LNG_DENSITY

top_dir = get_top_dir()
commodity_info_df = get_commodity_info()


def get_commodity_sfs_by_class():
    """
    Compiles the max, min, and central stowage factor (SF) values for each vessel class,
    and concatenates them into a single dataframe.
    
    Returns
    -------
    combined_sfs_df : pandas.DataFrame
        Combined dataframe containing vessel class, commodity, and lower/upper/central
        stowage factors for all vessel types.
    """
    dfs = []  # list to collect per-vessel dataframes

    for vessel_type in vessels:
        commodity_info_class = commodity_info_df[commodity_info_df["Vessel Class"] == vessel_type].copy()
        
        if vessel_type == "tanker":
            # Assign SFs for liquid fuels
            commodity_info_class.loc["Gasoline", "Lower Stowage Factor (m^3/tonne)"] = GASOLINE_SF
            commodity_info_class.loc["Gasoline", "Upper Stowage Factor (m^3/tonne)"] = GASOLINE_SF
            commodity_info_class.loc["Gasoline", "Central Stowage Factor (m^3/tonne)"] = GASOLINE_SF

            commodity_info_class.loc["Fuel oils", "Lower Stowage Factor (m^3/tonne)"] = FUEL_OIL_SF
            commodity_info_class.loc["Fuel oils", "Upper Stowage Factor (m^3/tonne)"] = FUEL_OIL_SF
            commodity_info_class.loc["Fuel oils", "Central Stowage Factor (m^3/tonne)"] = FUEL_OIL_SF

            # Handle crude petroleum dynamically
            central_crude_petrol_df, lower_crude_petrol_df, upper_crude_petrol_df = calculate_crude_petrol_sfs()
            commodity_info_class.loc["Crude petroleum", "Lower Stowage Factor (m^3/tonne)"] = lower_crude_petrol_df["Stowage Factor (m^3/tonne)"].min()
            commodity_info_class.loc["Crude petroleum", "Upper Stowage Factor (m^3/tonne)"] = upper_crude_petrol_df["Stowage Factor (m^3/tonne)"].max()
            commodity_info_class.loc["Crude petroleum", "Central Stowage Factor (m^3/tonne)"] = (
                central_crude_petrol_df["Stowage Factor (m^3/tonne)"] * central_crude_petrol_df["Weight"]
            ).sum()

        elif vessel_type == "gas_carrier":
            # Assign LNG SF
            commodity_info_class.loc["Natural gas and other fossil products", "Lower Stowage Factor (m^3/tonne)"] = LNG_SF
            commodity_info_class.loc["Natural gas and other fossil products", "Upper Stowage Factor (m^3/tonne)"] = LNG_SF
            commodity_info_class.loc["Natural gas and other fossil products", "Central Stowage Factor (m^3/tonne)"] = LNG_SF

        else:
            # Clean and average for bulk and container
            commodity_info_class = commodity_info_class.dropna(subset=["Lower Stowage Factor (m^3/tonne)", "Upper Stowage Factor (m^3/tonne)"])
            commodity_info_class["Central Stowage Factor (m^3/tonne)"] = 0.5 * (
                commodity_info_class["Lower Stowage Factor (m^3/tonne)"] + commodity_info_class["Upper Stowage Factor (m^3/tonne)"]
            )

        dfs.append(commodity_info_class)

    combined_sfs_df = pd.concat(dfs, axis=0)
    return combined_sfs_df

def extract_fuel_name(filepath):

    # Extract the filename
    filename = filepath.split("/")[-1]

    # define the expected filename pattern
    pattern = "{fuel}-{pathway}-{result}-per_mile.csv"

    # attempt to parse the filename
    parsed = parse(pattern, filename)

    if not parsed:
        raise ValueError(f"Filename '{filename}' does not match expected pattern: {pattern}")

    # return the fuel name
    return parsed["fuel"]
    
def get_vessel_type_carrying_commodity(commodity):
    """
    Extract the vessel type that is assumed to carry the given commodity.
    
    Parameters
    ----------
    commodity : string
        Name of the commodity

    Returns
    -------
    vessel_type : string
        Name of the vessel type assumed to carry the given commodity
    """
    
    vessel_type = commodity_info_df[commodity_info_df.index == commodity]["Vessel Class"].iloc[0]
    return vessel_type
    

def list_matching_files(directory):
    """
    Lists all files in the given directory that match one of the expected naming patterns:
      {fuel}-{process}-TotalEquivalentWTW-per_mile.csv
      {fuel}-{process}-TotalEquivalentWTT-per_mile.csv
      {fuel}-{process}-TotalEquivalentTTW-per_mile.csv

    Returns
    -------
    list of str
        Full file paths of matching files.
    """
    patterns = [
        "{fuel}-{process}-TotalEquivalentWTW-per_mile.csv",
        "{fuel}-{process}-TotalEquivalentWTT-per_mile.csv",
        "{fuel}-{process}-TotalEquivalentTTW-per_mile.csv"
    ]

    matching_files = []

    for filename in os.listdir(directory):
        for pattern in patterns:
            if parse(pattern, filename):
                full_path = os.path.join(directory, filename)
                matching_files.append(full_path)
                break  # Stop after first match for this file

    return matching_files
    
def get_all_route_properties():
    """
    Makes a dictionary of route properties (speeds, time at sea, capacity utilization, condition distribution) for each vessel type and class.

    Returns
    -------
    all_route_properties : Dictionary
        Dictionary containing the route property dictionary for each vessel class.
    """
    all_route_properties = {}
    for vessel_type in vessels:
        for vessel_class in vessels[vessel_type]:
            all_route_properties[vessel_type] = get_route_properties(vessel_class)
    return all_route_properties
    

def main():
    mass_density_dict = get_fuel_info_dict(
        f"{top_dir}/info_files/fuel_info.csv", "Mass density (kg/L)"
    )
    LHV_dict = get_fuel_info_dict(
        f"{top_dir}/info_files/fuel_info.csv", "Lower Heating Value (MJ / kg)"
    )
    boiloff_rate_dict = get_fuel_info_dict(
        f"{top_dir}/info_files/fuel_info.csv", "Boil-off Rate (%/day)"
    )

    sec_dict = get_fuel_info_dict(
        f"{top_dir}/info_files/fuel_info.csv", "Reliquefaction SEC (kWh/kg)"
    )

    eff_dict = get_eff_dict(fuels + ["lsfo"])
    
    tank_size_factors_dict, days_to_empty_tank_dict = get_tank_size_factors(
        fuels, LHV_dict, mass_density_dict, eff_dict, boiloff_rate_dict, sec_dict
    )
    
    #print(tank_size_factors_dict)
    
    cargo_info_df = pd.read_csv(f"{top_dir}/info_files/assumed_cargo_density.csv")
    
    commodity_sfs = get_commodity_sfs_by_class()
    
    def get_modified_cargo_capacity(vessel_class, fuel, sf):
        """
        Calculates modified cargo capacities for a given vessel class operating on a given fuel type and carrying a commodity with stowage factor SF. The fuel type determines how much space is taken up by tanks and the SF determines whether the cargo is mass- or volume-limited.

        Parameters
        ----------
        vessel_class : string
            Name of the vessel class to get the modified cargo capacity for
        
        fuel : string
            Fuel name
            
        sf : float
            Stowage factor (SF) for of the commodity being carried by the vessel

        Returns
        -------
        volume_capacity_fuel : float
            Volume capacity of the vessel carrying the given commodity, in m^3
        
        mass_capacity_fuel : float
            Mass capacity of the vessel carrying the given commodity, in tonnes
        """
        capacity_dict = calculate_modified_cargo_capacities_no_sf(
            vessel_class,
            fuel,
            cargo_info_df,
            mass_density_dict,
            tank_size_factors_dict)

        volume_capacity_fuel, mass_capacity_fuel, limitation_fuel = calculate_modified_cargo_capacities_with_sf(capacity_dict, sf)
                
        return volume_capacity_fuel, mass_capacity_fuel

    def evaluate_per_cargo_mile(filepath, commodity):
        """
        Given the filename for a given per-mile result, evaluate the result per cargo-mile (m^3-mile or tonne-mile) for the given commodity.
        
        Parameters
        ----------
        filepath : string
            Full path to the file per-mile results
        
        commodity : string
            Name of the commodity being carried
            

        Returns
        -------
        per_tonne_mile_df : Pandas Dataframe
            Dataframe containing the results per tonne-mile
        
        per-cargo_mile_df : Pandas Dataframe
            Dataframe containing the results per m^3-mile
        """
        per_cargo_mile = {}
        
        # Extract the fuel name from the filename and confirm that the filename has the expected format
        fuel = extract_fuel_name(filepath)
        
        # Get the vessel class that carries the given commodity
        vessel_type = get_vessel_type_carrying_commodity(commodity)
        
        # Read in the file contents as a csf
        data_df = pd.read_csv(filepath)
        
        # Get the upper, lower, and central SF for the given commodity
        sfs = commodity_sfs[commodity_sfs.index == commodity]
        lower_sf = sfs["Lower Stowage Factor (m^3/tonne)"]
        upper_sf = sfs["Upper Stowage Factor (m^3/tonne)"]
        central_sf = sfs["Central Stowage Factor (m^3/tonne)"]
        
        # Initialize dataframes to contain the data per tonne mile  with the region names
        per_tonne_mile_df = data_df[["Region"]].copy()
        per_cbm_mile_df = data_df[["Region"]].copy()
        
        for sf_opt in ["upper", "lower", "central"]:
            for vessel_class in vessels[vessel_type]:
                sf = sfs[f"{sf_opt.title()} Stowage Factor (m^3/tonne)"].iloc[0]
                volume_capacity, mass_capacity = get_modified_cargo_capacity(vessel_class, fuel, sf)
                
                # Get the fraction of time the vessel is fully loaded vs. unloaded
                
                
                per_tonne_mile_df[vessel_class] = data_df[vessel_class + "_ice"] / mass_capacity
                per_cbm_mile_df[vessel_class] = data_df[vessel_class + "_ice"] / volume_capacity
            per_cargo_mile[f"per tonne-mile ({sf_opt})"] = per_tonne_mile_df
            per_cargo_mile[f"per cbm-mile ({sf_opt})"] = per_cbm_mile_df
            
        return per_cargo_mile
    
    # Loop through all commodities and make files for emissions per tonne-mile and per cbm-mile for the commodity's upper, lower, and central stowage factor.
    commodities = commodity_sfs.index

    # Make a list of all files containing WTT, WTW, and TTW emissions results per mile
    emissions_filepaths = list_matching_files(f"{top_dir}/processed_results")
    output_dir = f"{top_dir}/emissions_by_commodity"
    ensure_directory_exists(output_dir)
    
    i_file = 0
    n_files = len(emissions_filepaths)
    for emissions_filepath in emissions_filepaths:
        if not "bio_cfp" in emissions_filepath:
            continue
        # Get the filename without the full path
        emissions_filename = emissions_filepath.split("/")[-1]
        print(f"------> Processing emissions file {i_file} of {n_files}")

        # Copy the original per-mile file to the output dir
        shutil.copy(emissions_filepath, f"{output_dir}/{emissions_filename}")
        print(f"Saved original file to {output_dir}/{emissions_filename}")

        i_file += 1
        for commodity in commodities:
            per_cargo_mile = evaluate_per_cargo_mile(emissions_filepath, commodity)
            
            # Ensure the commodity name included in the filename doesn't include any spaces, periods, or slashes
            commodity_save = commodity.replace(" ", "").replace(".", "").replace("/", "")
            
            for sf_opt in ["upper", "lower", "central"]:
                for cargo_metric in ["tonne-mile", "cbm-mile"]:
                    emission_filename_base = emissions_filename.replace("_mile.csv", "")
                    output_filepath = f"{output_dir}/{emission_filename_base}_{cargo_metric}_commodity_{commodity_save}_{sf_opt}_sf.csv"
                    per_cargo_mile[f"per {cargo_metric} ({sf_opt})"].to_csv(output_filepath)
                    print(f"Saved augmented file to {output_filepath}")

if __name__ == "__main__":
    main()

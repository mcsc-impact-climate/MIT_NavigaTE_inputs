"""
Date: April 22, 2025
Author: danikae
Purpose: Calculates the cargo-miles (either tonne-miles or cbm-miles) per MJ of fuel energy for each vessel class and size modeled in NavigaTE.
"""

import glob
import os
import pandas as pd
from common_tools import get_top_dir, ensure_directory_exists
top_dir = get_top_dir()

fuels = ["lsfo", "liquid_hydrogen", "ammonia", "methanol", "FTdiesel"]

def get_sample_file(pattern):
    """
    Gets the first file matching the given pattern
    
    Parameters
    ----------
    pattern: str
        Filename pattern for the files of interest

    Returns
    -------
    sample_filename : str
        Sample filename matching the provided pattern
    """
    
    # Use glob to find matching files
    matching_files = glob.glob(pattern)

    # Check and select the first file
    if matching_files:
        sample_file = matching_files[0]
    else:
        print("No matching files found.")
    return sample_file
    
def get_info_row(filename):
    info_df = pd.read_csv(filename, nrows=1)
    info_df.drop(columns=["Region"])
    
    return info_df
    
def add_info_row(info_df, pattern, fuel):

    lsfo_info_file = get_sample_file(pattern.replace("_fueltype-", "_lsfo-"))
    lsfo_info = get_info_row(lsfo_info_file)
    
    # Don't double count for lsfo since it's also the main fuel
    if fuel == "lsfo":
        main_info = lsfo_info * 0
    else:
        main_info_file = get_sample_file(pattern.replace("_fueltype-", "_main-"))
        main_info = get_info_row(main_info_file)

    # Add the numeric columns together
    combined = lsfo_info.add(main_info, fill_value=0).select_dtypes(include=[float, int])
    
    # Add in a Fuel column
    combined["Fuel"] = fuel
    
    # Reorder columns to put "Fuel" first
    cols = ["Fuel"] + [col for col in combined.columns if col != "Fuel"]
    combined = combined[cols]
    
    # Initialize the dataframe containing this info if it's the first fuel, and otherwise append to the existing dataframe
    if info_df.empty:
            info_df = combined
    else:
        info_df = pd.concat([info_df, combined], ignore_index=True)
        
    return info_df
    

def main():
    # Loop through all modeled fuels that the vessel could run on
    total_energy_per_mile = pd.DataFrame()
    total_energy_per_tonne_mile_max_cap = pd.DataFrame()
    total_energy_per_cbm_mile_max_cap = pd.DataFrame()
    total_energy_per_tonne_mile_mod_cap = pd.DataFrame()
    total_energy_per_cbm_mile_mod_cap = pd.DataFrame()
    
    for fuel in fuels:

        ##################################################################### Evaluate per mile #####################################################################
        # Energy in GJ (fuel energy) and miles are nautical miles (1.852 km)
        pattern = os.path.join(top_dir, "processed_results", f"{fuel}-*-ConsumedEnergy_fueltype-per_mile.csv")
        total_energy_per_mile = add_info_row(total_energy_per_mile, pattern, fuel)
        ###########################################################################################################################################################
        
        ####################### Evaluate per tonne-mile and per cbm-mile assuming the vessel is loaded to its max weight or volume capacity, without accounting for changes in max cargo capacity due to different tank sizes for alternative fuels #######################
        # Get the name of a sample processed file with the LSFO and main fuel energy consumed per tonne-mile or per cargo-mile (doesn't matter which fuel production pathway)
        pattern = os.path.join(top_dir, "processed_results", f"{fuel}-*-ConsumedEnergy_fueltype-per_tonne_mile_lsfo.csv")
        total_energy_per_tonne_mile_max_cap = add_info_row(total_energy_per_tonne_mile_max_cap, pattern, fuel)
        
        pattern = os.path.join(top_dir, "processed_results", f"{fuel}-*-ConsumedEnergy_fueltype-per_cbm_mile_lsfo.csv")
        total_energy_per_cbm_mile_max_cap = add_info_row(total_energy_per_cbm_mile_max_cap, pattern, fuel)
        ###########################################################################################################################################################
        
        ########## Evaluate per tonne-mile and per cbm-mile assuming the vessel is loaded to its typical capacity accounting for cargo stowage factors, but not accounting for changes in max cargo capacity due to different tank sizes for alternative fuels ############
        pattern = os.path.join(top_dir, "processed_results", f"{fuel}-*-ConsumedEnergy_fueltype-per_tonne_mile_lsfo_final.csv")
        total_energy_per_tonne_mile_mod_cap = add_info_row(total_energy_per_tonne_mile_mod_cap, pattern, fuel)
        
        pattern = os.path.join(top_dir, "processed_results", f"{fuel}-*-ConsumedEnergy_fueltype-per_cbm_mile_lsfo_final.csv")
        total_energy_per_cbm_mile_mod_cap = add_info_row(total_energy_per_cbm_mile_mod_cap, pattern, fuel)
        ###########################################################################################################################################################


    # Save all results to csv
    total_energy_per_mile.to_csv("tables/fuel_GJ_per_nautical_mile.csv")
    total_energy_per_tonne_mile_max_cap.to_csv("tables/fuel_GJ_per_tonne_nautical_mile_max_cap.csv")
    total_energy_per_cbm_mile_max_cap.to_csv("tables/fuel_GJ_per_cbm_nautical_mile_max_cap.csv")
    total_energy_per_tonne_mile_mod_cap.to_csv("tables/fuel_GJ_per_tonne_nautical_mile_lsfo_cap.csv")
    total_energy_per_cbm_mile_mod_cap.to_csv("tables/fuel_GJ_per_cbm_nautical_mile_lsfo_cap.csv")
    
main()

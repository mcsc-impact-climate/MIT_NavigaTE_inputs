"""
Date: 251208
Author: danikae
Purpose: Combine costs and emissions from all sources (fuel production, transportation, and storage) to get the final fuel cost at a given port
"""
import pandas as pd
import os
from common_tools import get_top_dir
import parse

ports = ["Singapore", "Rotterdam"]

fuels = [
    "FTdiesel",
    "lng",
    "ammonia",
    "compressed_hydrogen",
    "liquid_hydrogen",
    "methanol",
    "bio_cfp",
#        "bio_leo"
]

top_dir = get_top_dir()

def add_transportation_storage_costs(costs_emissions_df, fuel, port):
    """

    """
    transport_file = os.path.join(top_dir, "input_fuel_pathway_data", "transport", f"{fuel}_{port}.csv")
    transport_df = pd.read_csv(transport_file)
    
    merged_df = pd.merge(costs_emissions_df, transport_df, on=["Region"], how="left")
    
    # Make a final LCOF and Emissions column that sums up all relevant costs and emissions
    merged_df["Final LCOF [$/tonne]"] = merged_df["LCOF [$/tonne]"] + merged_df["Land Transport Cost [$/tonne]"] + merged_df["Fuel Storage Cost [$/tonne]"] + merged_df["Tanker Transport Cost [$/tonne]"]
    merged_df["Final Emissions [kg CO2e / kg fuel]"] = merged_df["Emissions [kg CO2e / kg fuel]"] + merged_df["Land Transport Emissions [kg CO2e / kg fuel]"] + merged_df["Fuel Storage Emissions [kg CO2e / kg fuel]"] + merged_df["Tanker Transport Emissions [kg CO2e / kg fuel]"]

    # Rename LCOF [$/tonne] to Fuel Production LCOF [$/tonne] and Emissions [kg CO2e / kg fuel] to Fuel Production Emissions [kg CO2e / kg fuel] for clarity
    merged_df.rename(columns={"LCOF [$/tonne]": "Fuel Production LCOF [$/tonne]", "Emissions [kg CO2e / kg fuel]": "Fuel Production Emissions [kg CO2e / kg fuel]", "Fuel_x": "Fuel"}, inplace=True)
    merged_df.drop(columns=["Fuel_y", "Comment"], inplace=True)

    return merged_df

def main():
    input_dir = os.path.join(top_dir, "input_fuel_pathway_data", "production")
    for file_name in os.listdir(input_dir):
        # Only process CSV files matching the following structure: {fuel}_costs_emissions.csv using the parse package
        result = parse.parse("{fuel}_costs_emissions.csv", file_name)
        if result:
            fuel = result["fuel"]
            if fuel in fuels:
                for port in ports:
                    costs_emissions_df = pd.read_csv(os.path.join(top_dir, "input_fuel_pathway_data", "production", file_name))
                    costs_emissions_df = add_transportation_storage_costs(costs_emissions_df, fuel, port)

                    # Make sure the output directory exists
                    output_dir = os.path.join(top_dir, "input_fuel_pathway_data/final_fuel", port)
                    os.makedirs(output_dir, exist_ok=True)
                    filepath_save = os.path.join(output_dir, f"{fuel}_{port}_costs_emissions.csv")
                    costs_emissions_df.to_csv(filepath_save, index=False)
                    print(f"Combined cost-emissions table saved to {filepath_save}")

if __name__ == "__main__":
    main()
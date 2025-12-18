"""
Date: 251106
Author: danikae
Purpose: Tabulate annual fuel demands to operate the global shipping fleet on each fuel
"""

import pandas as pd
import glob
import os

processed_results = "processed_results"  # set your actual directory path

fuels = ["lsfo", "liquid_hydrogen", "ammonia", "lng", "lsng", "FTdiesel", "methanol", "bio_cfp"]

results = []  # list to hold results for all fuels

for fuel in fuels:
    # Use wildcard for 'process' so it matches any process name
    consumedEnergy_pattern = os.path.join(processed_results, f"{fuel}-*-ConsumedEnergy_main-fleet.csv")
    consumedMass_pattern = os.path.join(processed_results, f"{fuel}-*-ConsumedMass_main-fleet.csv")
    
    # Find all matching files
    matches_energy = glob.glob(consumedEnergy_pattern)
    matches_mass = glob.glob(consumedMass_pattern)
    
    if matches_energy and matches_mass:
        # Read the first match of each
        df_energy = pd.read_csv(matches_energy[0])
        df_mass = pd.read_csv(matches_mass[0])

        # Ensure the data can be indexed by row label
        if "Unnamed: 0" in df_energy.columns:
            df_energy = df_energy.set_index("Unnamed: 0")
        if "Unnamed: 0" in df_mass.columns:
            df_mass = df_mass.set_index("Unnamed: 0")

        # Extract global totals (assuming "Global Average" is an index label)
        try:
            global_energy_demand = df_energy.loc[df_energy["Region"].astype(str).str.strip() == "Global Average", "fleet"].iloc[0]
            global_mass_demand = df_mass.loc[df_mass["Region"].astype(str).str.strip() == "Global Average", "fleet"].iloc[0]
        except KeyError:
            print(f"'Global Average' not found in {fuel} files, skipping.")
            continue

        # Store results
        results.append({
            "Fuel name": fuel,
            "Annual fuel energy consumed (GJ/year)": global_energy_demand,
            "Annual fuel mass consumed (kg/year)": global_mass_demand
        })

    else:
        print(f"No matching files found for {fuel}")

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = os.path.join("tables", "global_fleet_fuel_demands.csv")
results_df.to_csv(output_path, index=False)

print(f"\nSaved results to: {output_path}")
print(results_df)



    

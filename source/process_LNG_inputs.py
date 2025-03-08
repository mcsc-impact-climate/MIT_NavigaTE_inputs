"""
Date: Mar 7, 2025
Purpose: Process NG related inputs from GREET to get them in the form needed for use in calculate_fuel_costs_emissions.py
"""

import pandas as pd
from common_tools import get_top_dir, get_fuel_LHV

BTU_PER_MJ = 947.817
BTU_PER_MMBTU = 1e6
MJ_PER_GJ = 1000
GAL_PER_M3 = 264.172
NG_LHV = get_fuel_LHV("LNG")  # LHV of NG, in MJ/kg
G_PER_KG = 1000

def Btu_per_mmBtu_to_kg_per_kg(value):
    """Convert NG consumption from Btu/mmBtu to kg/kg."""
    return value / BTU_PER_MMBTU

def kg_per_kg_to_GJ_per_kg(value):
    """Convert NG consumption from kg/kg to GJ/kg."""
    return value * NG_LHV / MJ_PER_GJ

def gal_per_mmBtu_to_m3_per_kg(value):
    """Convert water consumption from gal/mmBtu to m³/kg."""
    return value / GAL_PER_M3 * NG_LHV / BTU_PER_MMBTU * BTU_PER_MJ

def g_per_mmBtu_to_kg_per_kg(value):
    """Convert emissions from g/mmBtu to kg/kg."""
    return value / G_PER_KG * NG_LHV / BTU_PER_MMBTU * BTU_PER_MJ

def main():
    top_dir = get_top_dir()
    input_path = f"{top_dir}/input_fuel_pathway_data/LNG_inputs_GREET.csv"
    
    # Load data
    df = pd.read_csv(input_path)

    # Define mappings
    stage_mapping = {
        "Recovery": "Production",
        "Processing": "Production",
        "Liquefaction": "Liquefaction"
    }

    # Apply stage mapping
    df["Stage"] = df["Stage"].map(stage_mapping)

    # Aggregate production stages
    df_grouped = df.groupby(["Stage", "Input"], as_index=False)["Value"].sum()

    # Pivot table to correctly align columns
    df_pivot = df_grouped.pivot_table(index="Stage", columns="Input", values="Value", aggfunc="sum").reset_index()

    # Rename columns explicitly
    df_pivot = df_pivot.rename(columns={
        "NG Consumption": "NG Consumption (Btu/mmBtu)",
        "Water Consumption": "Water Consumption (gal/mmBtu)",
        "CH4 Emissions": "CH4 Emissions (g/mmBtu)",
        "CO2 Emissions": "CO2 Emissions (g/mmBtu)"
    })

    print("Pivoted Table:")
    print(df_pivot)

    # Apply conversions
    df_pivot["NG Consumption (kg/kg)"] = df_pivot["NG Consumption (Btu/mmBtu)"].apply(Btu_per_mmBtu_to_kg_per_kg)
    df_pivot["NG Consumption (MJ/kg)"] = df_pivot["NG Consumption (kg/kg)"].apply(kg_per_kg_to_GJ_per_kg)
    df_pivot["Water Consumption (m^3/kg)"] = df_pivot["Water Consumption (gal/mmBtu)"].apply(gal_per_mmBtu_to_m3_per_kg)
    df_pivot["CH4 Emissions (kg/kg)"] = df_pivot["CH4 Emissions (g/mmBtu)"].apply(g_per_mmBtu_to_kg_per_kg)
    df_pivot["CO2 Emissions (kg/kg)"] = df_pivot["CO2 Emissions (g/mmBtu)"].apply(g_per_mmBtu_to_kg_per_kg)

    # Select final columns
    df_final = df_pivot[["Stage", "NG Consumption (kg/kg)", "NG Consumption (MJ/kg)",
                          "Water Consumption (m^3/kg)", "CH4 Emissions (kg/kg)", "CO2 Emissions (kg/kg)"]]

    # Save to CSV
    output_path = f"{top_dir}/input_fuel_pathway_data/LNG_inputs_GREET_processed.csv"
    df_final.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    main()



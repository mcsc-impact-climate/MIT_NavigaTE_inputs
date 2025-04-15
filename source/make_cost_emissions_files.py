"""
Date: July 23, 2024
Purpose: Prepare .inc files containing cost and emissions NavigaTE inputs for custom fuel production pathways defined in the csv files contained in input_fuel_pathway_data.
"""

import pandas as pd
import os
from common_tools import get_top_dir


# Function to create content of cost and emissions .inc file
def create_cost_emissions_file_content(row):
    content = f"""# Definition of cost and emissions for {row['Fuel']} from {row['Region']} for pathway {row['Pathway Name']}

Port "port" {{
    
# Emissions
set_bunker_WTT_overwrite("{row['Fuel']}", "carbon_dioxide", {row['Emissions [kg CO2e / kg fuel]']})

# Costs
set_bunker_price_overwrite("{row['Fuel']}", {row['LCOF [$/tonne]']})

}}
"""
    return content
    
# Function to create content of .inc file for excel report
def create_report_file_content(row, top_dir):
    
    # Make the directory to contain the report if it doesn't already exist
    report_output_dir = f"{top_dir}/navigate_reports/{row['Fuel']}-{row['Pathway Name']}-{row['Region']}-{row['Number']}"
    os.makedirs(report_output_dir, exist_ok=True)
    
    content = f"""
Report "excel_report" {{
    
    Directory="../../all_outputs_full_fleet"
    
    # Vessel properties
    add_vessel_property("*", TotalEquivalentWTT)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentTTW)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentWTW)        # tonnes CO2e/year
    add_vessel_property("*", Miles)                     # miles/year
    add_vessel_property("*", CargoMiles)                # tonne/miles per year
    add_vessel_property("*", SpendEnergy, BOTH)         # Energy demand / year (GJ)
        
    add_vessel_property("*", TotalCAPEX)
    add_vessel_property("*", TotalExcludingFuelOPEX)
    add_vessel_property("*", TotalFuelOPEX)
    add_vessel_property("*", ConsumedEnergy)            # Dictionary for each fuel
    
    # Fleet properties
    #add_fleet_property("*", Vessels)                   # DMM: Currently produces an error
    add_fleet_property("*", TotalEquivalentWTT)
    add_fleet_property("*", TotalEquivalentTTW)
    add_fleet_property("*", TotalEquivalentWTW)
    #add_fleet_property("*", Miles)
    #add_fleet_property("*", SpendEnergy, BOTH)         # DMM: Absolute number in GJ
    
    #add_fleet_property("*", TotalCAPEX)
    #add_fleet_property("*", TotalExcludingFuelOPEX)
    #add_fleet_property("*", TotalFuelOPEX)
    add_fleet_property("*", ConsumedEnergy)

}}
"""
    return content
    
def create_inc_file(row, output_dir, file_content):
    inc_file_name = f"{output_dir}{row['Fuel']}-{row['Pathway Name']}-{row['Region']}-{row['Number']}.inc"

    # Write content to .inc file
    with open(inc_file_name, "w") as file:
        file.write(file_content)

    print(f"File created: {inc_file_name}")

    
def create_nav_file(row, top_dir, cost_emissions_dir, report_dir):
    # Define the directory and ensure it exists
    navs_dir = f"{top_dir}/single_pathway_full_fleet/{row['Fuel']}/navs"
    os.makedirs(navs_dir, exist_ok=True)

    # Define paths for the original and modified .nav files
    original_nav_file = f"{top_dir}/single_pathway_full_fleet/{row['Fuel']}/{row['Fuel']}.nav"
    modified_nav_file = f"{navs_dir}/{row['Fuel']}-{row['Pathway Name']}-{row['Region']}-{row['Number']}.nav"

    # Read the content of the original .nav file
    with open(original_nav_file, 'r') as file:
        nav_content = file.readlines()

    # Define the replacement string
    cost_emissions_include_line = f'    INCLUDE "{cost_emissions_dir}{row["Fuel"]}-{row["Pathway Name"]}-{row["Region"]}-{row["Number"]}.inc"\n'
    report_include_line = f'    INCLUDE "{report_dir}{row["Fuel"]}-{row["Pathway Name"]}-{row["Region"]}-{row["Number"]}.inc"\n'

    # Modify the content of the .nav file
    modified_content = []
    for line in nav_content:
        # Replace the line that includes the cost_emissions row['Fuel'].inc
        if f'INCLUDE "../../includes_global/cost_emissions_{row["Fuel"]}.inc"' in line:
            modified_content.append(cost_emissions_include_line)
        elif f'INCLUDE "../../includes_global/make_excel.inc"' in line:
            modified_content.append(report_include_line)
        else:
            modified_content.append(line)

    # Write the modified content to the new .nav file
    with open(modified_nav_file, 'w') as file:
        file.writelines(modified_content)
        
    print(f"File created: {modified_nav_file}")


def main():
    top_dir = get_top_dir()
    input_dir = f"{top_dir}/input_fuel_pathway_data/production"
    cost_emissions_dir = f"{top_dir}/includes_global/all_costs_emissions/"
    report_dir = f"{top_dir}/includes_global/all_reports/"
        
    # Ensure the directories to contain the cost/emissions and report inc files exist
    os.makedirs(cost_emissions_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Delete all existing .inc files
    for inc_dir in [cost_emissions_dir, report_dir]:
        for file_name in os.listdir(inc_dir):
            if file_name.endswith(".inc"):
                os.remove(os.path.join(inc_dir, file_name))
                print(f"Deleted file: {file_name}")

    # Loop through all CSV files in the input directory
    fuels = ["FTdiesel", "lng", "ammonia", "methanol"]
    for fuel in fuels:
        file_name = f"{fuel}_costs_emissions.csv"
        csv_file = os.path.join(input_dir, file_name)
        df = pd.read_csv(csv_file)

        # Iterate over rows in the dataframe and create .inc files
        for index, row in df.iterrows():
            cost_emissions_file_content = create_cost_emissions_file_content(row)
            report_file_content = create_report_file_content(row, top_dir)
            
            create_inc_file(row, cost_emissions_dir, cost_emissions_file_content)
            create_inc_file(row, report_dir, report_file_content)

            # Create a modified version of the nav file that uses the costs and emissions .inc and the report .inc file
            create_nav_file(row, top_dir, cost_emissions_dir, report_dir)


if __name__ == "__main__":
    main()

"""
Date: July 23, 2024
Purpose: Prepare .inc files containing cost and emissions NavigaTE inputs for custom fuel production pathways defined in the csv files contained in input_fuel_pathway_data.
"""

import os
import pandas as pd
import argparse
from common_tools import get_top_dir


def create_forecast_inc_file_content(fuel, pathway, region, yearly_values):
    # Build table lines for forecasts
    forecast_lines_price = []
    forecast_lines_emiss = []
    for year in sorted(yearly_values.keys()):
        date_str = f'"01-01-{year}"'
        forecast_lines_emiss.append(f"        {date_str}    {yearly_values[year]['Emissions']}")
        forecast_lines_price.append(f"        {date_str}    {yearly_values[year]['LCOF']}")

    fuel_lower = fuel.lower()
    content = f"""# Definition of cost and emissions for {fuel} from {region} for pathway {pathway}

Port "port" {{
    
# Emissions
set_bunker_WTT_overwrite("{fuel}", "carbon_dioxide", Forecast("{fuel_lower}_WTT"))

# Costs
set_bunker_price_overwrite("{fuel}", Forecast("{fuel_lower}_price"))

}}

Forecast "{fuel_lower}_WTT" {{
    Table [
{chr(10).join(forecast_lines_emiss)}
    ]
    Extrapolate=FLAT
    Interpolate=LINEAR
}}

Forecast "{fuel_lower}_price" {{
    Table [
{chr(10).join(forecast_lines_price)}
    ]
    Extrapolate=FLAT
    Interpolate=LINEAR
}}
"""
    return content


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


def create_inc_file(row, output_dir, file_content):
    inc_file_name = f"{output_dir}{row['Fuel']}-{row['Pathway Name']}-{row['Region']}-{row['Number']}.inc"
    with open(inc_file_name, "w") as file:
        file.write(file_content)
    print(f"File created: {inc_file_name}")


def create_nav_file(row, top_dir, cost_emissions_dir, orig_caps=False):
    navs_dir = f"{top_dir}/single_pathway_full_fleet/{row['Fuel']}/navs"
    os.makedirs(navs_dir, exist_ok=True)

    if orig_caps:
        #print("Using original capacities")
        original_nav_file = (
            f"{top_dir}/single_pathway_full_fleet/{row['Fuel']}/{row['Fuel']}_orig_caps.nav"
        )
    else:
        original_nav_file = (
            f"{top_dir}/single_pathway_full_fleet/{row['Fuel']}/{row['Fuel']}.nav"
        )
    modified_nav_file = f"{navs_dir}/{row['Fuel']}-{row['Pathway Name']}-{row['Region']}-{row['Number']}.nav"

    with open(original_nav_file, "r") as file:
        nav_content = file.readlines()

    cost_emissions_include_line = f'    INCLUDE "{cost_emissions_dir}{row["Fuel"]}-{row["Pathway Name"]}-{row["Region"]}-{row["Number"]}.inc"\n'

    modified_content = []
    for line in nav_content:
        if f'INCLUDE "../../includes_global/cost_emissions_{row["Fuel"]}.inc"' in line:
            modified_content.append(cost_emissions_include_line)
        else:
            modified_content.append(line)

    with open(modified_nav_file, "w") as file:
        file.writelines(modified_content)
    print(f"File created: {modified_nav_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yearly", action="store_true", help="Enable yearly mode using *_costs_emissions_YEAR.csv")
    parser.add_argument("-o", "--orig_caps", action="store_true", help="Use original tank and cargo capacity")
    args = parser.parse_args()

    top_dir = get_top_dir()
    input_dir = f"{top_dir}/input_fuel_pathway_data/production"
    cost_emissions_dir = f"{top_dir}/includes_global/all_costs_emissions/"

    os.makedirs(cost_emissions_dir, exist_ok=True)
    for file_name in os.listdir(cost_emissions_dir):
        if file_name.endswith(".inc"):
            os.remove(os.path.join(cost_emissions_dir, file_name))
            print(f"Deleted file: {file_name}")

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

    for fuel in fuels:
        if args.yearly:
            # Collect yearly data across all files
            yearly_dfs = {}
            for file_name in os.listdir(input_dir):
                if file_name.startswith(f"{fuel}_costs_emissions_") and file_name.endswith(".csv"):
                    try:
                        year_str = file_name.rsplit("_", 1)[-1].replace(".csv", "")
                        year = int(year_str)
                        file_path = os.path.join(input_dir, file_name)
                        yearly_dfs[year] = pd.read_csv(file_path)
                    except ValueError:
                        continue

            if not yearly_dfs:
                print(f"No yearly data found for {fuel}, skipping.")
                continue

            # Group by common identifying columns
            first_year = sorted(yearly_dfs.keys())[0]
            for i in range(len(yearly_dfs[first_year])):
                row_key = (
                    yearly_dfs[first_year].loc[i, "Fuel"],
                    yearly_dfs[first_year].loc[i, "Pathway Name"],
                    yearly_dfs[first_year].loc[i, "Region"],
                )
                sample_row = yearly_dfs[first_year].loc[i]
                yearly_values = {
                    year: {
                        "LCOF": yearly_dfs[year].loc[i, "LCOF [$/tonne]"],
                        "Emissions": yearly_dfs[year].loc[i, "Emissions [kg CO2e / kg fuel]"]
                    }
                    for year in yearly_dfs
                }
                content = create_forecast_inc_file_content(*row_key, yearly_values)
                create_inc_file(sample_row, cost_emissions_dir, content)
                create_nav_file(sample_row, top_dir, cost_emissions_dir, args.orig_caps)

        else:
            file_path = os.path.join(input_dir, f"{fuel}_costs_emissions.csv")
            if not os.path.exists(file_path):
                continue
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                content = create_cost_emissions_file_content(row)
                create_inc_file(row, cost_emissions_dir, content)
                create_nav_file(row, top_dir, cost_emissions_dir, args.orig_caps)

if __name__ == "__main__":
    main()

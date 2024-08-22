"""
Date: July 23, 2024
Purpose: Prepare .inc files containing cost and emissions NavigaTE inputs for custom fuel production pathways defined in the csv files contained in input_fuel_pathway_data.
"""

import pandas as pd
import os
from common_tools import get_top_dir


# Function to create .inc file content
def create_inc_file_content(row):
    content = f"""# Definition of cost and emissions for {row['Fuel']} from {row['Region']} for pathway {row['Pathway Name']}

Port "port" {{
    
# Emissions
set_bunker_WTT_overwrite("{row['Fuel']}", "carbon_dioxide", {row['Emissions [kg CO2e / kg fuel]']})

# Costs
set_bunker_price_overwrite("{row['Fuel']}", {row['LCOF [$/tonne]']})

}}
"""
    return content


def main():
    top_dir = get_top_dir()
    input_dir = f"{top_dir}/input_fuel_pathway_data/production"
    output_dir = f"{top_dir}/includes_global/all_costs_emissions/"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Delete all existing .inc files in the output directory
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".inc"):
            os.remove(os.path.join(output_dir, file_name))
            print(f"Deleted file: {file_name}")

    # Loop through all CSV files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith("costs_emissions.csv"):
            csv_file = os.path.join(input_dir, file_name)
            df = pd.read_csv(csv_file)

            # Iterate over rows in the dataframe and create .inc files
            for index, row in df.iterrows():
                inc_file_name = f"{output_dir}{row['Fuel']}-{row['Pathway Name']}-{row['Region']}-{row['Number']}.inc"
                file_content = create_inc_file_content(row)

                # Write content to .inc file
                with open(inc_file_name, "w") as file:
                    file.write(file_content)

                print(f"File created: {inc_file_name}")


if __name__ == "__main__":
    main()

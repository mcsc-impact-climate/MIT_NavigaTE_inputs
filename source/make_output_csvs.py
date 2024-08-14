"""
Date: July 25, 2024
Author: danikam
Purpose: Reads in NavigaTE outputs from a 2024 run including tankers, bulk vessels, containerships and gas carriers, and produces CSV files to process and structure the outputs for visualization.
"""

from common_tools import get_top_dir
import pandas as pd
from parse import parse
import os

# Constants
TONNES_PER_TEU = 14
LB_PER_GAL_LNG = 3.49
GAL_PER_M3 = 264.172
LB_PER_TONNE = 2204.62
TONNES_PER_M3_LNG = LB_PER_GAL_LNG * GAL_PER_M3 / LB_PER_TONNE

# Vessel type and size information
vessels = {
    "bulk_carrier_ice": [
        "bulk_carrier_capesize_ice",
        "bulk_carrier_handy_ice",
        "bulk_carrier_panamax_ice",
    ],
    "container_ice": [
        "container_15000_teu_ice",
        "container_8000_teu_ice",
        "container_3500_teu_ice",
    ],
    "tanker_ice": ["tanker_100k_dwt_ice", "tanker_300k_dwt_ice", "tanker_35k_dwt_ice"],
    "gas_carrier_ice": ["gas_carrier_100k_cbm_ice"],
}

# Number of vessels for each type and size
vessel_size_number = {
    "bulk_carrier_capesize_ice": 2013,
    "bulk_carrier_handy_ice": 4186,
    "bulk_carrier_panamax_ice": 6384,
    "container_15000_teu_ice": 464,
    "container_8000_teu_ice": 1205,
    "container_3500_teu_ice": 3896,
    "tanker_100k_dwt_ice": 3673,
    "tanker_300k_dwt_ice": 866,
    "tanker_35k_dwt_ice": 8464,
    "gas_carrier_100k_cbm_ice": 2156,
}

# Quantities of interest
quantities = [
    "ConsumedEnergy_lsfo",
    "ConsumedEnergy_main",
    "TotalCAPEX",
    "TotalExcludingFuelOPEX",
    "TotalFuelOPEX",
    "TotalCost",
    "TotalEquivalentWTT",
    "TotalEquivalentTTW",
    "TotalEquivalentWTW",
]

# Evaluation choices
evaluation_choices = ["per_year", "per_mile", "per_tonne_mile"]


def read_results(fuel, fuel_type, pathway, country, number, filename, all_results_df):
    # Define columns to read based on the fuel type
    if fuel == "lsfo":
        results_df_columns = [
            "Date",
            "Time (days)",
            "TotalEquivalentWTT",
            "TotalEquivalentTTW",
            "TotalEquivalentWTW",
            "Miles",
            "CargoMiles",
            "SpendEnergy",
            "TotalCAPEX",
            "TotalExcludingFuelOPEX",
            "TotalFuelOPEX",
            "ConsumedEnergy_lsfo",
        ]
    else:
        results_df_columns = [
            "Date",
            "Time (days)",
            "TotalEquivalentWTT",
            "TotalEquivalentTTW",
            "TotalEquivalentWTW",
            "Miles",
            "CargoMiles",
            "SpendEnergy",
            "TotalCAPEX",
            "TotalExcludingFuelOPEX",
            "TotalFuelOPEX",
            f"ConsumedEnergy_{fuel}",
            "ConsumedEnergy_lsfo",
        ]

    # Read the results from the Excel file
    results = pd.ExcelFile(filename)
    results_df = pd.read_excel(results, "Vessels")

    # Extract relevant data for each vessel type and size
    results_dict = {}
    for vessel_type in vessels:
        for vessel in vessels[vessel_type]:
            results_df_vessel = results_df.filter(regex=f"Date|Time|{vessel}").drop(
                [0, 1, 2]
            )
            results_df_vessel.columns = results_df_columns
            results_df_vessel = results_df_vessel.set_index("Date")

            results_dict["Vessel"] = f"{vessel}_{fuel}"
            results_dict["Fuel"] = fuel
            results_dict["FuelType"] = fuel_type
            results_dict["Pathway"] = pathway
            results_dict["Country"] = country
            results_dict["Number"] = number
            results_dict["TotalEquivalentWTT"] = float(
                results_df_vessel["TotalEquivalentWTT"].loc["2024-01-01"]
            )
            results_dict["TotalEquivalentTTW"] = float(
                results_df_vessel["TotalEquivalentTTW"].loc["2024-01-01"]
            )
            results_dict["TotalEquivalentWTW"] = float(
                results_df_vessel["TotalEquivalentWTW"].loc["2024-01-01"]
            )
            results_dict["TotalCAPEX"] = float(
                results_df_vessel["TotalCAPEX"].loc["2024-01-01"]
            )
            results_dict["TotalFuelOPEX"] = float(
                results_df_vessel["TotalFuelOPEX"].loc["2024-01-01"]
            )
            results_dict["TotalExcludingFuelOPEX"] = float(
                results_df_vessel["TotalExcludingFuelOPEX"].loc["2024-01-01"]
            )
            results_dict["TotalCost"] = (
                float(results_df_vessel["TotalCAPEX"].loc["2024-01-01"])
                + float(results_df_vessel["TotalFuelOPEX"].loc["2024-01-01"])
                + float(results_df_vessel["TotalExcludingFuelOPEX"].loc["2024-01-01"])
            )
            results_dict["Miles"] = float(results_df_vessel["Miles"].loc["2024-01-01"])
            if vessel_type == "container":
                results_dict["CargoMiles"] = (
                    float(results_df_vessel["CargoMiles"].loc["2024-01-01"])
                    * TONNES_PER_TEU
                )
            elif vessel_type == "gas_carrier":
                results_dict["CargoMiles"] = (
                    float(results_df_vessel["CargoMiles"].loc["2024-01-01"])
                    * TONNES_PER_M3_LNG
                )
            else:
                results_dict["CargoMiles"] = float(
                    results_df_vessel["CargoMiles"].loc["2024-01-01"]
                )

            if fuel == "lsfo":
                results_dict["ConsumedEnergy_main"] = float(
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"]
                )
                results_dict["ConsumedEnergy_lsfo"] = (
                    float(results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"])
                    * 0
                )
            else:
                results_dict["ConsumedEnergy_main"] = float(
                    results_df_vessel[f"ConsumedEnergy_{fuel}"].loc["2024-01-01"]
                )
                results_dict["ConsumedEnergy_lsfo"] = float(
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"]
                )

            results_row_df = pd.DataFrame([results_dict])
            all_results_df = pd.concat(
                [all_results_df, results_row_df], ignore_index=True
            )

    return all_results_df


def extract_info_from_filename(filename):
    pattern = "report_{fuel}-{fuel_type}-{pathway}-{country}-{number}.xlsx"
    result = parse(pattern, filename)
    if result:
        return result.named
    return None


def collect_all_results(top_dir):
    # List all files in the output directory
    files = os.listdir(f"{top_dir}/all_outputs_full_fleet/")
    fuel_pathway_country_tuples = [
        extract_info_from_filename(file)
        for file in files
        if extract_info_from_filename(file)
    ]

    # Initialize DataFrame to store all results
    columns = [
        "Vessel",
        "Fuel",
        "FuelType",
        "Pathway",
        "Country",
        "Number",
        "TotalEquivalentWTT",
        "TotalEquivalentTTW",
        "TotalEquivalentWTW",
        "TotalCAPEX",
        "TotalFuelOPEX",
        "TotalExcludingFuelOPEX",
        "TotalCost",
        "Miles",
        "CargoMiles",
        "ConsumedEnergy_main",
        "ConsumedEnergy_lsfo",
    ]

    all_results_df = pd.DataFrame(columns=columns)

    # Read results for each file and add to the DataFrame
    results_filename = f"{top_dir}/all_outputs_full_fleet/report_lsfo-1.xlsx"
    all_results_df = read_results(
        "lsfo", "grey", "fossil", "Global", 1, results_filename, all_results_df
    )

    for fuel_pathway_country in fuel_pathway_country_tuples:
        fuel = fuel_pathway_country["fuel"]
        fuel_type = fuel_pathway_country["fuel_type"]
        pathway = fuel_pathway_country["pathway"]
        country = fuel_pathway_country["country"]
        number = fuel_pathway_country["number"]
        results_filename = f"{top_dir}/all_outputs_full_fleet/report_{fuel}-{fuel_type}-{pathway}-{country}-{number}.xlsx"
        all_results_df = read_results(
            fuel, fuel_type, pathway, country, number, results_filename, all_results_df
        )

    return all_results_df


def add_number_of_vessels(all_results_df):
    def extract_base_vessel_name(vessel_name):
        return "_".join(vessel_name.split("_")[:-1])

    # Map the number of vessels to each row in the DataFrame
    all_results_df["base_vessel_name"] = all_results_df["Vessel"].apply(
        extract_base_vessel_name
    )
    all_results_df["n_vessels"] = (
        all_results_df["base_vessel_name"].map(vessel_size_number).astype(float)
    )
    all_results_df.drop("base_vessel_name", axis=1, inplace=True)

    return all_results_df


def add_evaluated_quantities(all_results_df):
    for quantity in quantities:
        for evaluation_choice in evaluation_choices:
            # Determine the column to divide by based on the evaluation choice
            if evaluation_choice == "per_mile":
                column_divide = "Miles"
            elif evaluation_choice == "per_tonne_mile":
                column_divide = "CargoMiles"
            else:
                continue

            # Compute evaluated quantity
            all_results_df[f"{quantity}-{evaluation_choice}"] = (
                all_results_df[f"{quantity}"] / all_results_df[column_divide]
                if column_divide
                else all_results_df[f"{quantity}"]
            )

def add_fleet_quantities(all_results_df):
    for quantity in quantities + ["Miles", "CargoMiles"]:
        # Multiply by the number of vessels to sum the quantity to the full fleet
        all_results_df[f"{quantity}-fleet"] = all_results_df[quantity] * all_results_df["n_vessels"]
    
    return all_results_df
        
def add_vessel_type_quantities(all_results_df):
    # List of quantities to sum
    quantities_fleet = [
        column for column in all_results_df.columns if "-fleet" in column
    ]

    new_rows = []

    # Iterate over each fuel, pathway, country, and number combination
    for (fuel, pathway, country, number), group_df in all_results_df.groupby(
        ["Fuel", "Pathway", "Country", "Number"]
    ):
        for vessel_type, vessel_names in vessels.items():
            # Filter the DataFrame for the current vessel type
            vessel_type_df = group_df[
                group_df["Vessel"].str.contains("|".join(vessel_names))
            ]

            if not vessel_type_df.empty:
                # Sum quantities for the vessel type
                vessel_type_row = vessel_type_df[
                    quantities_fleet + ["Miles", "CargoMiles"]
                ].sum()
                vessel_type_row["Fuel"] = fuel
                vessel_type_row["FuelType"] = vessel_type_df["FuelType"]
                vessel_type_row["Pathway"] = pathway
                vessel_type_row["Country"] = country
                vessel_type_row["Number"] = number
                vessel_type_row["Vessel"] = f"{vessel_type}_{fuel}"
                vessel_type_row["n_vessels"] = vessel_type_df["n_vessels"].sum()

                # Evaluate the average based on the fleet sum for each vessel-level quantity
                for quantity in quantities + ["Miles", "CargoMiles"]:
                    vessel_type_row[f"{quantity}"] = (
                        vessel_type_row[f"{quantity}-fleet"]
                        / vessel_type_row["n_vessels"]
                    )

                # Append the new row to the list
                new_rows.append(vessel_type_row)

    # Convert the list of new rows to a DataFrame and concatenate with the original DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    all_results_df = pd.concat([all_results_df, new_rows_df], ignore_index=True)

    return all_results_df


def mark_countries_with_multiples(all_results_df):
    # Iterate over each row in the DataFrame
    for index, row in all_results_df.iterrows():
        if int(row["Number"]) > 1:
            all_results_df.at[index, "Country"] = f"{row['Country']}_{row['Number']}"


def add_fleet_level_quantities(all_results_df):
    # Get a list of all vessels considered in the global fleet
    all_vessels = list(vessel_size_number.keys())

    # List of quantities to sum
    quantities_fleet = [
        column for column in all_results_df.columns if "-fleet" in column
    ]

    new_rows = []

    # Iterate over each fuel, pathway, country, and number combination
    for (fuel, pathway, country, number), group_df in all_results_df.groupby(
        ["Fuel", "Pathway", "Country", "Number"]
    ):
        fleet_df = group_df[group_df["Vessel"].str.contains("|".join(all_vessels))]

        if not fleet_df.empty:
            # Sum quantities for the full fleet
            fleet_row = fleet_df[quantities_fleet + ["Miles", "CargoMiles"]].sum()
            fleet_row["Fuel"] = fuel
            fleet_row["FuelType"] = fleet_df["FuelType"]
            fleet_row["Pathway"] = pathway
            fleet_row["Country"] = country
            fleet_row["Number"] = number
            fleet_row["Vessel"] = f"fleet_{fuel}"
            fleet_row["n_vessels"] = fleet_df["n_vessels"].sum()

            # Evaluate the average based on the fleet sum for each vessel-level quantity
            for quantity in quantities + ["Miles", "CargoMiles"]:
                fleet_row[f"{quantity}"] = (
                    fleet_row[f"{quantity}-fleet"] / fleet_row["n_vessels"]
                )

            # Append the new row to the list
            new_rows.append(fleet_row)

    # Convert the list of new rows to a DataFrame and concatenate with the original DataFrame
    new_rows_df = pd.DataFrame(new_rows)
    all_results_df = pd.concat([all_results_df, new_rows_df], ignore_index=True)

    return all_results_df


def generate_csv_files(all_results_df, top_dir):
    quantities_of_interest = list(
        all_results_df.drop(
            columns=["Vessel", "Fuel", "FuelType", "Pathway", "Country", "Number", "n_vessels"]
        ).columns
    )
    
    unique_fuels = all_results_df["Fuel"].unique()

    os.makedirs(f"{top_dir}/processed_results", exist_ok=True)

    for fuel in unique_fuels:
        unique_pathways = all_results_df["Pathway"][
            all_results_df["Fuel"] == fuel
        ].unique()
        for pathway in unique_pathways:
            filter = (all_results_df["Fuel"] == fuel) & (
                all_results_df["Pathway"] == pathway
            )
            all_selected_results_df = all_results_df[filter]
            fuel_type = all_selected_results_df["FuelType"].iloc[0]
            for quantity in quantities_of_interest:
                quantity_selected_results_df = all_selected_results_df[
                    ["Vessel", "Country", quantity]
                ]

                # Pivot the DataFrame
                pivot_df = quantity_selected_results_df.pivot(
                    index="Country", columns="Vessel", values=quantity
                )

                # Replace NaN with zeros or any other value as needed
                pivot_df = pivot_df.fillna(0)

                # Ensure all data are numeric
                pivot_df = pivot_df.apply(pd.to_numeric, errors="coerce").fillna(0)

                # Identify countries with multiple entries
                countries_with_multiple_entries = quantity_selected_results_df[
                    "Country"
                ][quantity_selected_results_df["Country"].str.contains("_2")]
                base_countries_with_multiple_entries = (
                    countries_with_multiple_entries.apply(
                        lambda x: x.split("_")[0]
                    ).unique()
                )

                # Rename base country rows with multiple entries
                for base_country in base_countries_with_multiple_entries:
                    if base_country in pivot_df.index:
                        pivot_df.rename(
                            index={base_country: f"{base_country}_1"}, inplace=True
                        )

                # Calculate the average for each base country with multiple entries and add as a new row
                avg_rows = []
                for base_country in base_countries_with_multiple_entries:
                    matching_rows = pivot_df.loc[
                        pivot_df.index.str.startswith(base_country + "_")
                    ]
                    if not matching_rows.empty:
                        avg_values = matching_rows.mean()
                        avg_values.name = base_country
                        avg_rows.append(avg_values)
                if avg_rows:
                    avg_df = pd.DataFrame(avg_rows)
                    pivot_df = pd.concat([pivot_df, avg_df])

                # Calculate the weighted average for each column, excluding the 'Weight' column itself
                global_avg = pivot_df.loc[~pivot_df.index.str.contains("_")].mean()

                # Add the weighted averages as a new row
                pivot_df.loc["Global Average"] = global_avg

                # If no modifier specified, add a modifier to indicate that the quantity is per-vessel
                if not "-" in quantity:
                    quantity = f"{quantity}-vessel"
                    
                # Generate the filename
                
                # Specify whether electro fuel type is from grid or renewables
                if fuel_type == "electro":
                    if "grid" in pathway:
                        fuel_type = "electro_grid"
                    else:
                        fuel_type = "electro_renew"
                
                filename = f"{fuel}-{fuel_type}-{pathway}-{quantity}.csv"
                filepath = f"{top_dir}/processed_results/{filename}"

                # Save the DataFrame to a CSV file
                pivot_df.to_csv(filepath)
                print(f"Saved {filename}")


def main():
    # Get the path to the top level of the Git repo
    top_dir = get_top_dir()

    # Collect all results from the Excel files
    all_results_df = collect_all_results(top_dir)

    # Add the number of vessels to the DataFrame
    all_results_df = add_number_of_vessels(all_results_df)

    # Multiply by number of vessels of each type+size the fleet to get fleet-level quantities
    all_results_df = add_fleet_quantities(all_results_df)
    
    #print(all_results_df.columns)

    # Group vessels by type to get type-level quantities
    all_results_df = add_vessel_type_quantities(all_results_df)

    # Group all vessel together to get fleet-level quantities
    all_results_df = add_fleet_level_quantities(all_results_df)

    # Add evaluated quantities (per mile and per tonne-mile) to the dataframe
    add_evaluated_quantities(all_results_df)

    # Append the country number to countries for which there's data for >1 country
    mark_countries_with_multiples(all_results_df)

    all_results_df.to_csv('all_results_df.csv')

    # Generate CSV files for each combination of fuel pathway, quantity, and evaluation choice
    generate_csv_files(all_results_df, top_dir)


main()

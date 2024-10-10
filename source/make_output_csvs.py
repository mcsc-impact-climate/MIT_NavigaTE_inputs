"""	
Date: July 25, 2024
Author: danikam
Purpose: Reads in NavigaTE outputs from a 2024 run including tankers, bulk vessels, containerships and gas carriers, and produces CSV files to process and structure the outputs for visualization.
"""

# GE - import useful python libraries
from common_tools import get_top_dir
import pandas as pd
from parse import parse
import os
import glob
import time
import functools

# Constants
TONNES_PER_TEU = 14 # GE - TEU = twenty-foot equivalent
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
    "tanker_ice": ["tanker_100k_dwt_ice", "tanker_300k_dwt_ice", "tanker_35k_dwt_ice"], # GE - dwt = deadweight tonnage
    "gas_carrier_ice": ["gas_carrier_100k_cbm_ice"], # GE - cbm = cubic meter
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
    "ConsumedEnergy_lsfo", # GE - low sulfur fuel oil
    "ConsumedEnergy_main",
    "TotalCAPEX", # GE - CAPEX = Capital Expenditure
    "TotalExcludingFuelOPEX", # GE - OPEX = Operating Expense
    "TotalFuelOPEX",
    "TotalCost",
    "TotalEquivalentWTT", # GE - wtt = well-to-tank emissions
    "TotalEquivalentTTW", # GE - ttw = tank-to-wake emissions
    "TotalEquivalentWTW", #GE - wtw = well-to-wake emissions
]

# Evaluation choices
evaluation_choices = ["per_year", "per_mile", "per_tonne_mile"] # per tonne-mile = multiply weight by distance

def time_function(func):
    """A decorator that logs the time a function takes to execute."""
    @functools.wraps(func)
    def wrapper_time_function(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds")
        return result
    return wrapper_time_function

# GE - returns DataFrame with new data appended
def read_results(fuel, pathway, region, number, filename, all_results_df):
    """
    Reads the results from an Excel file and extracts relevant data for each vessel type and size.

    Parameters
    ----------
    fuel : str
        The type of fuel being used (eg. ammonia, hydrogen)

    pathway : str
        The fuel production pathway (e.g., fossil, SMR).

    region : str
        The region associated with the results.

    number : int
        The number of instances or scenarios for this configuration.

    filename : str
        The path to the Excel file containing the results.

    all_results_df : pandas.DataFrame
        The DataFrame to which results will be appended.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the new results appended.
    """
    
    # Replace 'compressed_hydrogen' and 'liquid_hydrogen' in all_results_df with compressedhydrogen and liquidhydrogen to facilitate vessel name parsing
    fuel_orig = fuel
    fuel = fuel.replace('compressed_hydrogen', 'compressedhydrogen').replace('liquid_hydrogen', 'liquidhydrogen')
    
    # Define columns to read based on the fuel type
    if fuel == "lsfo":
        results_df_columns = [
            "Date",
            "Time (days)",
            "TotalEquivalentWTT",
            "TotalEquivalentTTW",
            "TotalEquivalentWTW",
            "Miles",
            "CargoMiles", # GE - another term for ton-miles
            "SpendEnergy",
            "TotalCAPEX",
            "TotalExcludingFuelOPEX",
            "TotalFuelOPEX",
            "ConsumedEnergy_lsfo",
        ]
    elif fuel == "FTdiesel": # GE - FT = Fischer-Tropsch: synthetic, biomass fuel
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
            f"ConsumedEnergy_{fuel_orig}",
            "ConsumedEnergy_lsfo",
        ]
    # Read the results from the csv file
    results_df = pd.read_csv(filename)

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
            results_dict["Pathway"] = pathway
            results_dict["Region"] = region
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
            if "container" in vessel_type:
                results_dict["CargoMiles"] = (
                    float(results_df_vessel["CargoMiles"].loc["2024-01-01"])
                    * TONNES_PER_TEU
                )
            elif "gas_carrier" in vessel_type:
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
            if fuel == "FTdiesel":
                results_dict["ConsumedEnergy_main"] = float(
                    results_df_vessel[f"ConsumedEnergy_{fuel}"].loc["2024-01-01"]
                )
                results_dict["ConsumedEnergy_lsfo"] = (
                    float(results_df_vessel[f"ConsumedEnergy_{fuel}"].loc["2024-01-01"])
                    * 0
                )
            else:
                results_dict["ConsumedEnergy_main"] = float(
                    results_df_vessel[f"ConsumedEnergy_{fuel_orig}"].loc["2024-01-01"]
                )
                results_dict["ConsumedEnergy_lsfo"] = float(
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"]
                ) 

            results_row_df = pd.DataFrame([results_dict])
            all_results_df = pd.concat(
                [all_results_df, results_row_df], ignore_index=True
            ) 

    return all_results_df
    
def read_results_dask(fuel, pathway, region, number, filename, all_results_df):
    """
    Reads the results from an Excel file and extracts relevant data for each vessel type and size.

    Parameters
    ----------
    fuel : str
        The type of fuel being used (eg. ammonia, hydrogen)

    pathway : str
        The fuel production pathway (e.g., fossil, SMR).

    region : str
        The region associated with the results.

    number : int
        The number of instances or scenarios for this configuration.

    filename : str
        The path to the Excel file containing the results.

    all_results_df : pandas.DataFrame
        The DataFrame to which results will be appended.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the new results appended.
    """
    
    # Replace 'compressed_hydrogen' and 'liquid_hydrogen' in all_results_df with compressedhydrogen and liquidhydrogen to facilitate vessel name parsing
    fuel_orig = fuel
    fuel = fuel.replace('compressed_hydrogen', 'compressedhydrogen').replace('liquid_hydrogen', 'liquidhydrogen')
    
    # Define columns to read based on the fuel type
    if fuel == "lsfo":
        results_df_columns = [
            "Date",
            "Time (days)",
            "TotalEquivalentWTT",
            "TotalEquivalentTTW",
            "TotalEquivalentWTW",
            "Miles",
            "CargoMiles", # GE - another term for ton-miles
            "SpendEnergy",
            "TotalCAPEX",
            "TotalExcludingFuelOPEX",
            "TotalFuelOPEX",
            "ConsumedEnergy_lsfo",
        ]
    elif fuel == "FTdiesel": # GE - FT = Fischer-Tropsch: synthetic, biomass fuel
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
            f"ConsumedEnergy_{fuel_orig}",
            "ConsumedEnergy_lsfo",
        ]
    # Read the results from the Excel file
    #results = pd.ExcelFile(filename)
    results_df = pd.read_excel(filename, sheet_name="Vessels")
    results_df = dd.from_pandas(results_df, npartitions=8)  # npartitions can be adjusted

    # Extract relevant data for each vessel type and size
    results_dict = {}
    for vessel_type in vessels:
        for vessel in vessels[vessel_type]:
#            results_df_vessel = results_df.filter(regex=f"Date|Time|{vessel}").drop(
#                [0, 1, 2]
#            )
            # Get the columns that match the pattern "Date|Time|<vessel>"
            matching_columns = [col for col in results_df.columns if pd.Series(col).str.contains(f"Date|Time|{vessel}").any()]

            # Select those columns
            results_df_vessel = results_df[matching_columns]

            # Filter out the first three rows (equivalent to drop([0, 1, 2], axis=0))
            results_df_vessel = results_df_vessel.loc[3:]
            
            results_df_vessel.columns = results_df_columns
            results_df_vessel = results_df_vessel.set_index("Date")

            results_dict["Vessel"] = f"{vessel}_{fuel}"
            results_dict["Fuel"] = fuel
            results_dict["Pathway"] = pathway
            results_dict["Region"] = region
            results_dict["Number"] = number
            results_dict["TotalEquivalentWTT"] = float(
                results_df_vessel["TotalEquivalentWTT"].loc["2024-01-01"].compute().iloc[0]
            )
            results_dict["TotalEquivalentTTW"] = float(
                results_df_vessel["TotalEquivalentTTW"].loc["2024-01-01"].compute().iloc[0]
            )
            results_dict["TotalEquivalentWTW"] = float(
                results_df_vessel["TotalEquivalentWTW"].loc["2024-01-01"].compute().iloc[0]
            )
            results_dict["TotalCAPEX"] = float(
                results_df_vessel["TotalCAPEX"].loc["2024-01-01"].compute().iloc[0]
            )
            results_dict["TotalFuelOPEX"] = float(
                results_df_vessel["TotalFuelOPEX"].loc["2024-01-01"].compute().iloc[0]
            )
            results_dict["TotalExcludingFuelOPEX"] = float(
                results_df_vessel["TotalExcludingFuelOPEX"].loc["2024-01-01"].compute().iloc[0]
            )
            results_dict["TotalCost"] = (
                float(results_df_vessel["TotalCAPEX"].loc["2024-01-01"].compute().iloc[0])
                + float(results_df_vessel["TotalFuelOPEX"].loc["2024-01-01"].compute().iloc[0])
                + float(results_df_vessel["TotalExcludingFuelOPEX"].loc["2024-01-01"].compute().iloc[0])
            )
            results_dict["Miles"] = float(results_df_vessel["Miles"].loc["2024-01-01"].compute().iloc[0])
            if vessel_type == "container":
                results_dict["CargoMiles"] = (
                    float(results_df_vessel["CargoMiles"].loc["2024-01-01"].compute().iloc[0])
                    * TONNES_PER_TEU
                )
            elif vessel_type == "gas_carrier":
                results_dict["CargoMiles"] = (
                    float(results_df_vessel["CargoMiles"].loc["2024-01-01"].compute().iloc[0])
                    * TONNES_PER_M3_LNG
                )
            else:
                results_dict["CargoMiles"] = float(
                    results_df_vessel["CargoMiles"].loc["2024-01-01"].compute().iloc[0]
                )

            if fuel == "lsfo":
                results_dict["ConsumedEnergy_main"] = float(
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"].compute().iloc[0]
                )
                results_dict["ConsumedEnergy_lsfo"] = (
                    float(results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"].compute().iloc[0])
                    * 0
                )
            if fuel == "FTdiesel":
                results_dict["ConsumedEnergy_main"] = float(
                    results_df_vessel[f"ConsumedEnergy_{fuel}"].loc["2024-01-01"].compute().iloc[0]
                )
                results_dict["ConsumedEnergy_lsfo"] = (
                    float(results_df_vessel[f"ConsumedEnergy_{fuel}"].loc["2024-01-01"].compute().iloc[0])
                    * 0
                )
            else:
                results_dict["ConsumedEnergy_main"] = float(
                    results_df_vessel[f"ConsumedEnergy_{fuel_orig}"].loc["2024-01-01"].compute().iloc[0]
                )
                results_dict["ConsumedEnergy_lsfo"] = float(
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"].compute().iloc[0]
                )

            results_row_df = pd.DataFrame([results_dict])
            all_results_df = dd.concat(
                [all_results_df, results_row_df], ignore_index=True
            )

    return all_results_df


def extract_info_from_filename(filename):
    """
    Extracts fuel, pathway, country, and number information from the given filename.

    Parameters
    ----------
    filename : str
        The filename from which to extract the information.

    Returns
    -------
    result.named : dict
        A dictionary containing the extracted information, or None if the pattern doesn't match.
    """
    pattern = "{fuel}-{pathway}-{region}-{number}_excel_report.csv"
    result = parse(pattern, filename)
    if result:
        return result.named
    return None

@time_function
def collect_all_results(top_dir):
    """
    Collects all results from Excel files in the specified directory and compiles them into a DataFrame.

    Parameters
    ----------
    top_dir : str
        The top-level directory containing the output files.

    Returns
    -------
    all_results_df : pandas.DataFrame
        A DataFrame containing all the collected results.
    """
    # List all files in the output directory
    files = os.listdir(f"{top_dir}/all_outputs_full_fleet_csv/")
    fuel_pathway_region_tuples = [
        extract_info_from_filename(file)
        for file in files
        if extract_info_from_filename(file)
    ]

    # Initialize DataFrame to store all results
    columns = [
        "Vessel",
        "Fuel",
        "Pathway",
        "Region",
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
    results_filename = f"{top_dir}/all_outputs_full_fleet_csv/lsfo-1_excel_report.csv"
    all_results_df = read_results(
        "lsfo", "fossil", "Global", 1, results_filename, all_results_df
    )

    for fuel_pathway_region in fuel_pathway_region_tuples:
        fuel = fuel_pathway_region["fuel"]
        pathway = fuel_pathway_region["pathway"]
        region = fuel_pathway_region["region"]
        number = fuel_pathway_region["number"]
        results_filename = f"{top_dir}/all_outputs_full_fleet_csv/{fuel}-{pathway}-{region}-{number}_excel_report.csv"
        all_results_df = read_results(
            fuel, pathway, region, number, results_filename, all_results_df
        )
    return all_results_df

@time_function
def add_number_of_vessels(all_results_df):
    """
    Maps the number of vessels to each row in the DataFrame.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which the number of vessels will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the number of vessels added.
    """

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


# GE - divides by specific quantity modifier (per mile, per tonne-mile)
@time_function
def add_quantity_modifiers(all_results_df):
    """
    Adds quantity modifiers (e.g., per year, per mile, per tonne-mile) to the DataFrame based on the existing quantities.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which evaluated quantities will be added.

    Returns
    -------
    None
    """
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


# GE - calculates quantities for the entire fleet
@time_function
def scale_quantities_to_fleet(all_results_df):
    """
    Scales quantities to the global fleet within each vessel type and size class by multiplying by the number of vessels of that type and class.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which fleet quantities will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with fleet-level quantities added.
    """
    for quantity in quantities + ["Miles", "CargoMiles"]:
        # Multiply by the number of vessels to sum the quantity to the full fleet
        all_results_df[f"{quantity}-fleet"] = (
            all_results_df[quantity] * all_results_df["n_vessels"]
        )

    return all_results_df

@time_function
def add_vessel_type_quantities(all_results_df):
    quantities_fleet = [col for col in all_results_df.columns if "-fleet" in col]
    
    # Perform groupby once
    grouped = all_results_df.groupby(["Fuel", "Pathway", "Region", "Number"])

    new_rows = []
    for (fuel, pathway, region, number), group_df in grouped:
        for vessel_type, vessel_names in vessels.items():
            vessel_type_df = group_df[group_df["Vessel"].str.contains("|".join(vessel_names))]

            if not vessel_type_df.empty:
                vessel_type_row = vessel_type_df[quantities_fleet + ["Miles", "CargoMiles"]].sum()
                vessel_type_row["Fuel"] = fuel
                vessel_type_row["Pathway"] = pathway
                vessel_type_row["Region"] = region
                vessel_type_row["Number"] = number
                vessel_type_row["Vessel"] = f"{vessel_type}_{fuel}"
                vessel_type_row["n_vessels"] = vessel_type_df["n_vessels"].sum()

                # Add rows in bulk
                new_rows.append(vessel_type_row)

    new_rows_df = pd.DataFrame(new_rows)
    return pd.concat([all_results_df, new_rows_df], ignore_index=True)


# GE - not used now but included just in case will be needed in the future
@time_function
def mark_countries_with_multiples(all_results_df):
    """
    Marks countries with multiple entries in the DataFrame by appending '_Number' to the country name.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results in which countries with multiples will be marked.

    Returns
    -------
    None
    """
    # Iterate over each row in the DataFrame
    for index, row in all_results_df.iterrows():
        if int(row["Number"]) > 1:
            all_results_df.at[index, "Region"] = f"{row['Region']}_{row['Number']}"

@time_function
def add_fleet_level_quantities(all_results_df):
    """
    Sums quantities in DataFrame to the full fleet, aggregating over all vessel types and sizes considered in the global fleet

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which fleet-level quantities will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with fleet-level quantities added.
    """
    # Get a list of all vessels considered in the global fleet
    all_vessels = list(vessel_size_number.keys())

    # List of quantities that have already been scaled to the fleet level for individual vessel types and sizes (searchable with the '-fleet' keyword)
    quantities_fleet = [
        column for column in all_results_df.columns if "-fleet" in column
    ]

    new_rows = []

    # Iterate over each fuel, pathway, region, and number combination in the dataframe.
    # For each such combo, the rows in all_results_df with vessels that match this combo are grouped into a single dataframe group_df
    for (fuel, pathway, region, number), group_df in all_results_df.groupby(
        ["Fuel", "Pathway", "Region", "Number"]
    ):
        # This filter ensures that we're only including base vessels (defined by both vessel type and size) when summing to the global fleet
        # This is necessary because we previously grouped base vessels into vessel types in add_vessel_type_quantities
        fleet_df = group_df[group_df["Vessel"].str.contains("|".join(all_vessels))]

        if not fleet_df.empty:
            # Sum quantities for the full fleet
            fleet_row = fleet_df[quantities_fleet + ["Miles", "CargoMiles"]].sum()
            fleet_row["Fuel"] = fuel
            fleet_row["Pathway"] = pathway
            fleet_row["Region"] = region
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

@time_function
def add_cac(all_results_df):
    """
    Adds the cost of carbon abatement (CAC) to all_results_df, where:
        CAC = (cost increase of the fuel relative to LSFO) / (WTW emission reduction relative to LSFO),
        but only if the WTW reduction is negative and its magnitude is at least 10% of the LSFO total cost.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which fleet-level quantities will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the cost of carbon abatement added.
        
    NOTE: This function is not currently being used. Instead, the product of total cost and emissions is being included in the output csvs (see function add_cost_times_emissions).
    """

    # Mapping vessels to LSFO equivalents
    lsfo_vessels = all_results_df["Vessel"].str.replace(
        r"(_[^_]+)$", "_lsfo", regex=True
    )

    # Adding LSFO vessel names to the DataFrame for comparison
    all_results_df["lsfo_vessel"] = lsfo_vessels

    # Find LSFO baseline for comparison
    lsfo_baseline = all_results_df[
        (all_results_df["Fuel"] == "lsfo")
        & (all_results_df["Pathway"] == "fossil")
        & (all_results_df["Region"] == "Global")
        & (all_results_df["Number"] == 1)
    ].set_index("Vessel")

    # Merge to find the matching LSFO baseline data for each vessel
    merged_df = all_results_df.merge(
        lsfo_baseline[["TotalCost", "TotalEquivalentWTW"]],
        left_on="lsfo_vessel",
        right_index=True,
        suffixes=("", "_lsfo"),
    )

    # Calculate the change in cost relative to LSFO
    merged_df["DeltaCost"] = merged_df["TotalCost"] - merged_df["TotalCost_lsfo"]

    merged_df["DeltaWTW"] = (
        merged_df["TotalEquivalentWTW"] - merged_df["TotalEquivalentWTW_lsfo"]
    )

    # Condition for calculating CAC: DeltaWTW is negative and its magnitude is at least 10% of TotalEquivalentWTW_lsfo
    condition = (merged_df["DeltaWTW"] < -0.1 * merged_df["TotalEquivalentWTW_lsfo"])

    # Calculate the cost of carbon abatement (CAC) based on the condition
    merged_df.loc[condition, "CAC"] = merged_df["DeltaCost"] / (-merged_df["DeltaWTW"])

    # Drop the temporary LSFO vessel column
    merged_df = merged_df.drop(
        columns=["lsfo_vessel", "TotalCost_lsfo", "TotalEquivalentWTW_lsfo", "DeltaCost", "DeltaWTW"]
    )

    return merged_df


@time_function
def add_av_cost_emissions_ratios(all_results_df):
    """
    Adds the average of cost and emissions ratios relative to LSFO:
        Average ratio = (1/2) * (cost for alt fuel / cost for LSFO) + (emissions for alt fuel / emissions for LSFO)

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which vessel-level quantities will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the cost of carbon abatement added.
        
    NOTE: This function is not currently being used. Instead, the product of total cost and emissions is being included in the output csvs (see function add_cost_times_emissions).
    """

    # Mapping vessels to LSFO equivalents
    lsfo_vessels = all_results_df["Vessel"].str.replace(
        r"(_[^_]+)$", "_lsfo", regex=True
    )

    # Adding LSFO vessel names to the DataFrame for comparison
    all_results_df["lsfo_vessel"] = lsfo_vessels

    # Find LSFO baseline for comparison
    lsfo_baseline = all_results_df[
        (all_results_df["Fuel"] == "lsfo")
        & (all_results_df["Pathway"] == "fossil")
        & (all_results_df["Region"] == "Global")
        & (all_results_df["Number"] == 1)
    ].set_index("Vessel")

    # Merge to find the matching LSFO baseline data for each vessel
    merged_df = all_results_df.merge(
        lsfo_baseline[["TotalCost", "TotalEquivalentWTW"]],
        left_on="lsfo_vessel",
        right_index=True,
        suffixes=("", "_lsfo"),
    )

    # Calculate the change in cost relative to LSFO
    merged_df["HalfCostRatio"] = 0.5 * merged_df["TotalCost"] / merged_df["TotalCost_lsfo"]

    merged_df["HalfWTWRatio"] = (
        0.5 * merged_df["TotalEquivalentWTW"] / merged_df["TotalEquivalentWTW_lsfo"]
    )

    # Calculate the average of the two ratios
    merged_df["AverageCostEmissionsRatio"] = merged_df["HalfCostRatio"] + merged_df["HalfWTWRatio"]

    # Drop the temporary LSFO vessel column
    merged_df = merged_df.drop(
        columns=["lsfo_vessel", "TotalCost_lsfo", "TotalEquivalentWTW_lsfo"]
    )

    return merged_df

@time_function
def add_cost_times_emissions(all_results_df):
    """
    Adds the product of cost and emissions to all_results_df.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the results to which fleet-level quantities will be added.

    Returns
    -------
    all_results_df : pandas.DataFrame
        The updated DataFrame with the cost of carbon abatement added.
        
    """
    
    # Get a list of all modifiers to handle
    all_modifiers = ["per_mile", "per_tonne_mile", "fleet"]
    
    # Calculate the product of cost times emissions
    all_results_df["CostTimesEmissions"] = all_results_df["TotalCost"] * all_results_df["TotalEquivalentWTW"]
    
    # Repeat for all modifiers
    for modifier in all_modifiers:
        all_results_df[f"CostTimesEmissions-{modifier}"] = all_results_df[f"TotalCost-{modifier}"] * all_results_df[f"TotalEquivalentWTW"]

    return all_results_df

# GE - thi function is called in generate_csv_files
def remove_all_files_in_directory(directory_path):
    """
    Removes all files in the specified directory.

    Parameters
    ----------
    directory_path : str
        The path to the directory where all files will be removed.

    Returns
    -------
    None
    """
    files = glob.glob(os.path.join(directory_path, "*"))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error removing {f}: {e}")

@time_function
def generate_csv_files(all_results_df, top_dir):
    """
    Generates and saves CSV files from the processed results DataFrame, organized by fuel, pathway, and quantity.

    Parameters
    ----------
    all_results_df : pandas.DataFrame
        The DataFrame containing the processed results.

    top_dir : str
        The top-level directory where the CSV files will be saved.

    Returns
    -------
    None
    """
    quantities_of_interest = list(
        all_results_df.drop(
            columns=[
                "Vessel",
                "Fuel",
                "Pathway",
                "Region",
                "Number",
                "n_vessels",
            ]
        ).columns
    )
    
    # Remove the fuel name from the vessel name since it's included in the filename
    all_results_df['Vessel'] = all_results_df['Vessel'].str.replace(r'_[^_]*$', '', regex=True)

    unique_fuels = all_results_df["Fuel"].unique()

    remove_all_files_in_directory(f"{top_dir}/processed_results")
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
            for quantity in quantities_of_interest:
                quantity_selected_results_df = all_selected_results_df[
                    ["Vessel", "Region", quantity]
                ]

                # Pivot the DataFrame
                pivot_df = quantity_selected_results_df.pivot(
                    index="Region", columns="Vessel", values=quantity
                )

                # Replace NaN with zeros or any other value as needed
                pivot_df = pivot_df.fillna(0)

                # Ensure all data are numeric
                pivot_df = pivot_df.apply(pd.to_numeric, errors="coerce").fillna(0)

                # Identify countries with multiple entries
                countries_with_multiple_entries = quantity_selected_results_df[
                    "Region"
                ][quantity_selected_results_df["Region"].str.contains("_2")]
                base_countries_with_multiple_entries = (
                    countries_with_multiple_entries.apply(
                        lambda x: x.split("_")[0]
                    ).unique()
                )

                # Rename base region rows with multiple entries
                for base_region in base_countries_with_multiple_entries:
                    if base_region in pivot_df.index:
                        pivot_df.rename(
                            index={base_region: f"{base_region}_1"}, inplace=True
                        )

                # Calculate the average for each base region with multiple entries and add as a new row
                avg_rows = []
                for base_region in base_countries_with_multiple_entries:
                    matching_rows = pivot_df.loc[
                        pivot_df.index.str.startswith(base_region + "_")
                    ]
                    if not matching_rows.empty:
                        avg_values = matching_rows.mean()
                        avg_values.name = base_region
                        avg_rows.append(avg_values)
                if avg_rows:
                    avg_df = pd.DataFrame(avg_rows)
                    pivot_df = pd.concat([pivot_df, avg_df])

                # Calculate the weighted average for each column, excluding the 'Weight' column itself
                global_avg = pivot_df.loc[~pivot_df.index.str.contains("_")].mean()

                # Add the weighted averages as a new row
                pivot_df.loc["Global Average"] = global_avg

                # If no modifier specified, add a modifier to indicate that the quantity is per-vessel
                if "-" not in quantity:
                    quantity = f"{quantity}-vessel"
                    
                # Update the fuel name for compressed/liquified hydrogen back to its original form with a '_' for file saving
                fuel_save = fuel
                if fuel == "compressedhydrogen":
                    fuel_save = "compressed_hydrogen"
                if fuel == "liquidhydrogen":
                    fuel_save = "liquid_hydrogen"

                # Generate the filename
                filename = f"{fuel_save}-{pathway}-{quantity}.csv"
                filepath = f"{top_dir}/processed_results/{filename}"

                # Save the DataFrame to a CSV file
                pivot_df.to_csv(filepath)
    print(f"Saved processed csv files to {top_dir}/processed_results")

# GE - function where the code is actually being run and where the functions above are called
def main():
    # Get the path to the top level of the Git repo
    top_dir = get_top_dir()
    
    # Collect all results from the Excel files in parallel
    all_results_df = collect_all_results(top_dir)

    # Add the number of vessels to the DataFrame
    all_results_df = add_number_of_vessels(all_results_df)

    # Multiply by number of vessels of each type+size the fleet to get fleet-level quantities
    all_results_df = scale_quantities_to_fleet(all_results_df)

    # Group vessels by type to get type-level quantities
    all_results_df = add_vessel_type_quantities(all_results_df)

    # Group all vessel together to get fleet-level quantities
    all_results_df = add_fleet_level_quantities(all_results_df)

    # Add evaluated quantities (per mile and per tonne-mile) to the dataframe
    add_quantity_modifiers(all_results_df)

    # Append the region number to countries for which there's data for >1 region
    mark_countries_with_multiples(all_results_df)
    
    # Add a column quantifying the cost of carbon abatement
    all_results_df = add_cac(all_results_df)
    
    # Add a column for cost times emissions
    all_results_df = add_cost_times_emissions(all_results_df)
    
    # Add a column for the average ratios of cost and emissions relative to LSFO
    all_results_df = add_av_cost_emissions_ratios(all_results_df)

    # Output all_results_df to a csv file to help with debugging
    all_results_df.to_csv("all_results_df.csv")

    # Generate CSV files for each combination of fuel pathway, quantity, and evaluation choice
    generate_csv_files(all_results_df, top_dir)


main()

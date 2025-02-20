"""
Author: danikam
Date: 250203
Purpose: Plot initial results from a multi-fuel simulation
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

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
    "tanker": [
        "tanker_100k_dwt",
        "tanker_300k_dwt",
        "tanker_35k_dwt"
    ],
    "gas_carrier": ["gas_carrier_100k_cbm"], # GE - cbm = cubic meter
}

# Fuels consumed by the global fleet
fuels_consumed = ["ammonia", "low_sulfur_fuel_oil"]

# Designated primary fuels for vessels
fuels_main = ["ammonia", "oil"]

# Plot the total number of vessels in the fleet vs. time
def read_results_fleet(filename):
    """
    Reads the results from an Excel file and extracts relevant data for each vessel type and size.

    Parameters
    ----------
    filename : str
        The path to the Excel file containing the results.

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame containing the results
    """
    
    # Read the results from the excel file
    results_df = pd.read_excel(filename, "Fleets")
    results_dict = {}
    
    for vessel_type in vessels:
        results_dict[vessel_type] = {}
        for vessel in vessels[vessel_type]:
            results_dict[vessel_type][vessel] = {}
            
            results_df_vessel = results_df.filter(regex=f"Date|Time|{vessel}")
            
            ####################### Add a df containing fleet info #######################
            # Identify columns with NaN in row index 1
            nan_columns = results_df_vessel.columns[results_df_vessel.iloc[1].isna()]

            # Retain 'Date' and 'Time (days)' columns
            columns_to_keep = list(nan_columns)

            # Filter the dataframe
            filtered_df = results_df_vessel[columns_to_keep]

            # Rename the columns using the values from row index 1, except for 'Date' and 'Time (days)'
            new_column_names = {}
            for col in nan_columns:
                cell_value = filtered_df.iloc[0][col]
                if isinstance(cell_value, (str, int, float)) and pd.notna(cell_value):
                    new_column_names[col] = str(cell_value)

            filtered_df = filtered_df.rename(columns=new_column_names)

            # Drop rows with index 0, 1, and 2
            results_dict[vessel_type][vessel]["fleet info"] = filtered_df.drop(index=[0, 1, 2]).reset_index(drop=True)
            ##############################################################################
            
            ############## Add dfs containing info about each vessel type ################
            results_dict[vessel_type][vessel]["main fuel info"] = {}
            
            # Identify columns where the value in row index 1 contains the vessel name
            vessel_columns = [col for col in results_df_vessel.columns
                               if isinstance(results_df_vessel.iloc[1][col], str) and
                               vessel in results_df_vessel.iloc[1][col]]

            # Retain relevant columns
            columns_to_keep = vessel_columns

            # Filter the dataframe
            filtered_df = results_df_vessel[columns_to_keep]

            # Get unique values in row 1 for the vessel columns
            unique_main_fuels_long = filtered_df.iloc[1][vessel_columns].unique()

            # Parse for the part of the column name that comes after 'ice'
            unique_main_fuels = [value.split('_ice_')[-1] for value in unique_main_fuels_long if isinstance(value, str) and 'ice' in value]

            # Loop to create one dataframe for each unique vessel type
            for main_fuel_long, main_fuel in zip(unique_main_fuels_long, unique_main_fuels):
                # Identify columns where the value in row index 1 matches main_fuel_long
                main_fuel_columns = [col for col in filtered_df.columns
                                      if filtered_df.iloc[1][col] == main_fuel_long]

                # Filter the dataframe for the current main fuel
                main_fuel_df = filtered_df[main_fuel_columns]

                # Set values of row index 0 as the new column names
                main_fuel_df.columns = main_fuel_df.iloc[0]

                # Drop rows with index 0, 1, and 2
                main_fuel_df = main_fuel_df.drop(index=[0, 1, 2]).reset_index(drop=True)

                # Add 'Date' and 'Time (days)' columns at the far left
                main_fuel_df.insert(0, 'Time (days)', results_df_vessel['Time (days)'].iloc[3:].reset_index(drop=True))
                main_fuel_df.insert(0, 'Date', results_df_vessel['Date'].iloc[3:].reset_index(drop=True))

                # Add the dataframe to the dictionary
                results_dict[vessel_type][vessel]["main fuel info"][main_fuel] = main_fuel_df
            
            ##############################################################################
            
            ############### Add dfs containing info about fuel consumption ###############
            results_dict[vessel_type][vessel]["consumed fuel info"] = {}

            # Identify columns where the value in row index 1 does not contain the vessel name
            consumed_fuel_columns = [col for col in results_df_vessel.columns
                               if isinstance(results_df_vessel.iloc[1][col], str) and
                               vessel not in results_df_vessel.iloc[1][col]]

            # Retain relevant columns
            columns_to_keep = consumed_fuel_columns

            # Filter the dataframe
            filtered_df = results_df_vessel[columns_to_keep]

            # Get unique fuel names
            unique_consumed_fuels = filtered_df.iloc[1][consumed_fuel_columns].unique()
            
            for consumed_fuel in unique_consumed_fuels:
                # Identify columns where the value in row index 1 matches fuel
                consumed_fuel_columns = [col for col in filtered_df.columns
                                      if filtered_df.iloc[1][col] == consumed_fuel]

                # Filter the dataframe for the current main fuel
                main_fuel_df = filtered_df[consumed_fuel_columns]

                # Set values of row index 0 as the new column names
                main_fuel_df.columns = main_fuel_df.iloc[0]

                # Drop rows with index 0, 1, and 2
                main_fuel_df = main_fuel_df.drop(index=[0, 1, 2]).reset_index(drop=True)

                # Add 'Date' and 'Time (days)' columns at the far left
                main_fuel_df.insert(0, 'Time (days)', results_df_vessel['Time (days)'].iloc[3:].reset_index(drop=True))
                main_fuel_df.insert(0, 'Date', results_df_vessel['Date'].iloc[3:].reset_index(drop=True))

                # Add the dataframe to the dictionary
                results_dict[vessel_type][vessel]["consumed fuel info"][main_fuel] = main_fuel_df
            ##############################################################################

    return results_dict
    

def read_results_global(filename):
    """
    Reads the results from an Excel file and extracts relevant global data.

    Parameters
    ----------
    filename : str
        The path to the Excel file containing the results.

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame containing the results
    """
    
    # Read the results from the excel file
    results_df = pd.read_excel(filename, "Global")
    #print(results_df)
    results_dict = {}

    ########### Add a df containing info for the entire global fleet ############
    # Filter columns where both row index 1 and 2 are NaN, but index 0 isn't NaN
    filtered_columns = results_df.columns[(results_df.loc[1].isna()) & (results_df.loc[2].isna()) & ~(results_df.loc[0].isna())]

    # Select the filtered columns and set row index 0 as the new header
    filtered_df = results_df[filtered_columns].copy()
    filtered_df.columns = filtered_df.loc[0]

    # Drop the first three rows (headers and NaN rows)
    filtered_df = filtered_df.drop([0, 1, 2]).reset_index(drop=True)

    # Add the 'Date' and 'Time (days)' columns back to the left side
    results_dict["fleet info"] = pd.concat([results_df[['Date', 'Time (days)']].iloc[3:].reset_index(drop=True), filtered_df], axis=1)
    ##############################################################################
    
    ############### Add dfs containing info about individual fuels ###############
    # Extract unique fuel names, excluding all caps
    fuel_names = results_df.iloc[1].dropna().unique()
    fuel_names = [name for name in fuel_names if not name.isupper()]

    # Dictionary to store separate DataFrames for each fuel
    results_dict["fuel info"] = {}

    # Iterate over each unique fuel name
    for fuel in fuel_names:
        # Select columns where row index 1 matches the fuel name and row index 2 is NaN
        fuel_columns = results_df.columns[(results_df.iloc[1] == fuel) & (results_df.iloc[2].isna())]

        # Create new DataFrame with Date, Time (days), and relevant columns
        fuel_df = results_df[['Date', 'Time (days)'] + list(fuel_columns)].copy()

        # Rename columns using row index 0 values
        fuel_df.rename(columns={col: results_df.at[0, col] for col in fuel_columns}, inplace=True)

        # Remove rows 0 to 2
        fuel_df = fuel_df.iloc[3:].reset_index(drop=True)

        # Store the resulting DataFrame
        results_dict["fuel info"][fuel] = fuel_df
    ##############################################################################
    
    return results_dict
    
#def plot_global_results():
    
def plot_fleet_info_column(data_dict, column_name):
    # Access the "fleet info" DataFrame
    fleet_df = data_dict['fleet info']
    
    # Check if the column exists
    if column_name not in fleet_df.columns:
        print(f"Column '{column_name}' not found in 'fleet info'.")
        return
    
    # Plot the column as a function of Date
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(fleet_df['Date']), fleet_df[column_name], marker='o')
    plt.xlabel('Date', fontsize=22)
    plt.ylabel(column_name, fontsize=22)
    #plt.title(f"{column_name} Over Time")
    plt.grid(True)
    plt.tight_layout()
    
    # Update tick label font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.tight_layout()
    plt.savefig(f"plots/multifuel/global_{column_name}.png", dpi=300)
    plt.savefig(f"plots/multifuel/global_{column_name}.pdf")
    plt.close()
    
def plot_stacked_fuel_info(global_results_dict, column_name):
    # Access the 'fuel info' dictionary
    fuel_info = global_results_dict['fuel info']
    
    # Initialize an empty DataFrame to collect data for stacking
    stacked_data = pd.DataFrame()

    # Loop through each fuel and extract the column if it exists
    for fuel, df in fuel_info.items():
        if column_name in df.columns:
            # Ensure Date is datetime for proper plotting
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Add data to stacked_data, using Date as the index
            if stacked_data.empty:
                stacked_data['Date'] = df['Date']
                stacked_data[fuel] = df[column_name]
            else:
                stacked_data = pd.merge(stacked_data, df[['Date', column_name]].rename(columns={column_name: fuel}),
                                        on='Date', how='outer')

    # Sort by Date to ensure proper stacking
    stacked_data = stacked_data.sort_values('Date').fillna(0)
    stacked_data.set_index('Date', inplace=True)

    # Plotting the stacked area chart
    plt.figure(figsize=(12, 6))
    plt.stackplot(stacked_data.index, stacked_data.T, labels=stacked_data.columns, alpha=0.8)

    # Labeling
    plt.xlabel('Date', fontsize=22)
    plt.ylabel(column_name, fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title='Fuel', fontsize=14, title_fontsize=16, loc='upper left')

    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"plots/multifuel/stacked_{column_name}.png", dpi=300)
    plt.savefig(f"plots/multifuel/stacked_{column_name}.pdf")
    plt.close()


def main():
    #fleet_results_dict = read_results_fleet("multi_fuel_full_fleet/ammonia_lsfo/plots/ammonia_lsfo_excel_report.xlsx")
    global_results_dict = read_results_global("multi_fuel_full_fleet/all_fuels/plots/all_fuels_excel_report.xlsx")
    
    print(global_results_dict["fleet info"].columns)
    print(global_results_dict["fuel info"]["ammonia"].columns)
    plot_fleet_info_column(global_results_dict, "TotalEquivalentWTW")
    plot_fleet_info_column(global_results_dict, "RegulationExpenses")

    plot_stacked_fuel_info(global_results_dict, "ConsumedEnergy")
#    plot_stacked_fuel_info(global_results_dict, "FuelRelatedExpenses")
#    plot_stacked_fuel_info(global_results_dict, "VesselThreshold")

main()

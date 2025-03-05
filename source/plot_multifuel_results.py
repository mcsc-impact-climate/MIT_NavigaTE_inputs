"""
Author: danikam
Date: 250203
Purpose: Plot results from a multi-fuel simulation
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from common_tools import get_top_dir, get_pathway_type, get_pathway_type_label, get_pathway_label, read_pathway_labels, read_fuel_labels, get_fuel_label, create_directory_if_not_exists

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
    "gas_carrier": ["gas_carrier_100k_cbm"],
}

vessel_type_title = {
    "bulk_carrier_ice": "Bulk Carrier",
    "container_ice": "Container",
    "tanker_ice": "Tanker",
    "gas_carrier_ice": "Gas Carrier",
}

vessel_size_title = {
    "bulk_carrier_capesize_ice": "Capesize",
    "bulk_carrier_handy_ice": "Handy",
    "bulk_carrier_panamax_ice": "Panamax",
    "container_15000_teu_ice": "15,000 TEU",
    "container_8000_teu_ice": "8,000 TEU",
    "container_3500_teu_ice": "3,500 TEU",
    "tanker_100k_dwt_ice": "100k DWT",
    "tanker_300k_dwt_ice": "300k DWT",
    "tanker_35k_dwt_ice": "35k DWT",
    "gas_carrier_100k_cbm_ice": "100k m$^3$",
}

# Read in results for the fleet sheet
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

            # Identify all consumed fuel columns by excluding vessel-specific columns
            consumed_fuel_columns = [col for col in results_df_vessel.columns
                                     if isinstance(results_df_vessel.iloc[1][col], str) and
                                     vessel not in results_df_vessel.iloc[1][col]]

            # Retain relevant columns
            filtered_df = results_df_vessel[consumed_fuel_columns]

            # Get unique fuel names
            unique_consumed_fuels = filtered_df.iloc[1].dropna().unique()  # Drop NaN values before extracting unique names

            for consumed_fuel in unique_consumed_fuels:
                # Identify columns where the value in row index 1 matches `consumed_fuel`
                consumed_fuel_columns = [col for col in filtered_df.columns
                                         if isinstance(filtered_df.iloc[1][col], str) and
                                         filtered_df.iloc[1][col] == consumed_fuel]

                # Ensure at least one column is found before processing
                if consumed_fuel_columns:
                    # Filter the dataframe for the current consumed fuel
                    consumed_fuel_df = filtered_df[consumed_fuel_columns].copy()

                    # Set values of row index 0 as the new column names
                    consumed_fuel_df.columns = consumed_fuel_df.iloc[0]

                    # Drop rows with index 0, 1, and 2
                    consumed_fuel_df = consumed_fuel_df.drop(index=[0, 1, 2]).reset_index(drop=True)

                    # Add 'Date' and 'Time (days)' columns at the far left
                    consumed_fuel_df.insert(0, 'Time (days)', results_df_vessel['Time (days)'].iloc[3:].reset_index(drop=True))
                    consumed_fuel_df.insert(0, 'Date', results_df_vessel['Date'].iloc[3:].reset_index(drop=True))

                    # Add the dataframe to the dictionary
                    results_dict[vessel_type][vessel]["consumed fuel info"][consumed_fuel] = consumed_fuel_df
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

def read_results_regulation(filename):
    """
    Reads the results from the Regulations sheet in an Excel file and extracts relevant regulation data.

    Parameters
    ----------
    filename : str
        The path to the Excel file containing the results.

    Returns
    -------
    results_dict : dict
        Dictionary containing the extracted results structured by regulation names.
    """

    # Read the results from the "Regulations" sheet
    results_df = pd.read_excel(filename, "Regulations", header=None)

    # Extract the actual regulation names from row 0 without the .number suffix
    regulation_names = results_df.iloc[0].dropna().unique()
    regulation_names = [name for name in regulation_names if name not in ["Date", "Time (days)"]]

    # Create a mapping from existing column names to their actual regulation names
    col_regulation_mapping = {}
    for col in results_df.columns:
        col_name = results_df.iloc[0, col]
        if pd.notna(col_name):  # Only map non-NaN values
            col_regulation_mapping[col] = col_name

    # Initialize dictionary to store results
    results_dict = {}

    for regulation in regulation_names:
        # Select columns that belong to the current regulation
        regulation_columns = [col for col, name in col_regulation_mapping.items() if name == regulation]

        # Create new DataFrame with Date, Time (days), and relevant columns
        regulation_df = results_df[[0, 1] + regulation_columns].copy()

        # Rename columns: use row 1 values for meaningful names
        col_names = ['Date', 'Time (days)'] + [f"{results_df.iloc[1, col]}" for col in regulation_columns]
        regulation_df.columns = col_names

        # Separate fleet info (columns where row 2 is NaN) and vessel info (columns where row 2 has vessel names)
        fleet_columns = [col_names[i] for i, col in enumerate(regulation_columns, start=2) if pd.isna(results_df.iloc[2, col])]

        # Create fleet info DataFrame
        fleet_df = regulation_df[['Date', 'Time (days)'] + fleet_columns].copy()
        fleet_df = fleet_df.iloc[4:].reset_index(drop=True)  # Drop header rows

        # Create vessel info dictionary
        vessel_info = {}
        unique_vessels = results_df.iloc[2, regulation_columns].dropna().unique()

        for vessel in unique_vessels:
            # Get only the column indices corresponding to this vessel, ensuring uniqueness
            vessel_col_indices = [col for col in range(regulation_df.shape[1]) if regulation_df.iloc[2, col] == vessel]

            # Ensure 'Date' and 'Time (days)' are included
            vessel_df = regulation_df.iloc[:, [0, 1] + vessel_col_indices].copy()

            vessel_df = vessel_df.iloc[4:].reset_index(drop=True)
            vessel_info[vessel] = vessel_df

        # Store results in dictionary
        results_dict[regulation] = {
            "fleet info": fleet_df,
            "vessel info": vessel_info
        }

    return results_dict


    
def plot_fleet_info_column(data_dict, column_name, info_type="global", ylabel=None, save_label=None):
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
    if ylabel is None:
        plt.ylabel(column_name, fontsize=22)
    else:
        plt.ylabel(ylabel, fontsize=22)
    #plt.title(f"{column_name} Over Time")
    plt.grid(True)
    plt.tight_layout()
    
    # Update tick label font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.tight_layout()
    
    # Save the plot
    if save_label is not None:
        save_label = "_" + save_label
    
    save_path_png = f"plots/multifuel/{info_type}_{column_name}{save_label}.png"
    save_path_pdf = f"plots/multifuel/{info_type}_{column_name}{save_label}.pdf"
    
    print(f"Saving to {save_path_png}")
    plt.savefig(save_path_png, dpi=300)
    
    print(f"Saving to {save_path_pdf}")
    plt.savefig(save_path_pdf)
    plt.close()
    
def plot_regulation_vessel_info(data_dict, column_name, regulation_name="net_zero_regulation", vessel_name="container_3500_teu_ice_liquid_hydrogen", ylabel=None, save_label=None):

    # Access the regulation dict
    vessel_df = data_dict[regulation_name]["vessel info"][vessel_name]
    
    # Check if the column exists
    if column_name not in vessel_df.columns:
        print(f"Column '{column_name}' not found in 'vessel info'.")
        return
    
    # Plot the column as a function of Date
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(vessel_df['Date']), vessel_df[column_name], marker='o')
    plt.xlabel('Date', fontsize=22)
    if ylabel is None:
        plt.ylabel(column_name, fontsize=22)
    else:
        plt.ylabel(ylabel, fontsize=22)
    #plt.title(f"{column_name} Over Time")
    plt.grid(True)
    plt.tight_layout()
    
    # Update tick label font size
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    plt.tight_layout()
    
    # Save the plot
    if save_label is not None:
        save_label = "_" + save_label
    
    # Save the plot
    print(f"Saving to plots/multifuel/regulation_{regulation_name}_{vessel_name}_{column_name}{save_label}.png")
    plt.savefig(f"plots/multifuel/regulation_{regulation_name}_{vessel_name}_{column_name}{save_label}.png", dpi=300)
    
    print(f"Saving to plots/multifuel/regulation_{regulation_name}_{vessel_name}_{column_name}{save_label}.pdf")
    plt.savefig(f"plots/multifuel/regulation_{regulation_name}_{vessel_name}_{column_name}{save_label}.pdf")
    plt.close()
    
    
def plot_global_stacked_fuel_info(global_results_dict, column_name, ylabel=None, save_label=None):
    # Access the 'fuel info' dictionary
    fuel_info = global_results_dict['fuel info']
    
    # Initialize an empty DataFrame to collect data for stacking
    stacked_data = pd.DataFrame()

    # Loop through each fuel and extract the column if it exists
    for fuel, df in fuel_info.items():
        if column_name in df.columns:
            # Ensure Date is datetime for proper plotting
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Get the formatted fuel label
            fuel_label = get_fuel_label(fuel)

            # Add data to stacked_data, using Date as the index
            if stacked_data.empty:
                stacked_data['Date'] = df['Date']
                stacked_data[fuel_label] = df[column_name]
            else:
                stacked_data = pd.merge(
                    stacked_data,
                    df[['Date', column_name]].rename(columns={column_name: fuel_label}),
                    on='Date',
                    how='outer'
                )

    # Ensure all columns except Date are numeric
    stacked_data = stacked_data.sort_values('Date')

    # Convert all non-Date columns to numeric before filling NaNs
    for col in stacked_data.columns:
        if col != 'Date':
            stacked_data[col] = pd.to_numeric(stacked_data[col], errors='coerce')

    # Fill NaN values with 0
    stacked_data = stacked_data.fillna(0)

    stacked_data.set_index('Date', inplace=True)

    # Plotting the stacked area chart
    plt.figure(figsize=(14, 6))
    plt.stackplot(stacked_data.index, stacked_data.T, labels=stacked_data.columns, alpha=0.8)

    # Labeling
    plt.xlabel('Date', fontsize=22)
    if ylabel is None:
        plt.ylabel(column_name, fontsize=22)
    else:
        plt.ylabel(ylabel, fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title='Fuel', fontsize=20, title_fontsize=22, loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    if save_label is not None:
        save_label = "_" + save_label
        
    save_path_png = f"plots/multifuel/global_stacked_{column_name}{save_label}.png"
    save_path_pdf = f"plots/multifuel/global_stacked_{column_name}{save_label}.pdf"
        
    print(f"Saving to {save_path_png}")
    plt.savefig(save_path_png, dpi=300)
    
    print(f"Saving to {save_path_pdf}")
    plt.savefig(save_path_pdf)
    plt.close()
    

def plot_main_fuel_info_fleet(fleet_results_dict, column, vessel_classes=["bulk_carrier", "container", "tanker", "gas_carrier"], level="size", ylabel=None, save_label=None):
    """
    Plots a stacked area chart of the specified column for different fuels at different levels.

    Parameters
    ----------
    fleet_results_dict : dict
        Dictionary containing fleet results, structured by vessel class and size.
    column : str
        The column to be plotted (e.g., "ExistingVessels").
    vessel_classes : list, optional
        List of vessel classes to include in the plot.
    level : str, optional
        Determines the stacking level: "size" (per vessel size), "class" (per vessel class), or "fleet" (full fleet).
    """

    fleet_stacked_data = None  # Ensure fleet_stacked_data is initialized properly

    for vessel_class in vessel_classes:
        fleet_results_class_dict = fleet_results_dict[vessel_class]
        
        class_stacked_data = None  # Initialize for class-level aggregation

        for vessel_size in fleet_results_class_dict:
            
            main_fuel_size_dict = fleet_results_class_dict[vessel_size]["main fuel info"]
            
            stacked_data = None  # Initialize for size-level aggregation

            for fuel, df in main_fuel_size_dict.items():
                if column in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is datetime
                    
                    # Extract relevant data
                    df_subset = df[['Date', column]].rename(columns={column: fuel})
                    df_subset.set_index("Date", inplace=True)

                    # Ensure numeric values for addition
                    df_subset[fuel] = pd.to_numeric(df_subset[fuel], errors='coerce')

                    if stacked_data is None:
                        stacked_data = df_subset
                    else:
                        stacked_data = stacked_data.add(df_subset, fill_value=0)

            # Fill NaNs with 0 after summation
            if stacked_data is not None:
                stacked_data = stacked_data.fillna(0)

                # If level="size", plot immediately
                if level == "size":
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.stackplot(stacked_data.index, stacked_data.T, labels=stacked_data.columns, alpha=0.8)

                    ax.set_xlabel('Date', fontsize=22)
                    if ylabel is None:
                        ax.set_ylabel(column, fontsize=22)
                    else:
                        ax.set_ylabel(ylabel, fontsize=22)
                    ax.set_title(f"{vessel_class}: {vessel_size}", fontsize=24)
                    ax.tick_params(axis='both', labelsize=18)
                    ax.grid(True)

                    ax.legend(title='Fuel', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
                    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legend

                    # Save the plot
                    if save_label is not None and not save_label.startswith("_"):
                        save_label = "_" + save_label

                    save_path_png = f"plots/multifuel/size_{vessel_size}_{column}{save_label}.png"
                    save_path_pdf = f"plots/multifuel/size_{vessel_size}_{column}{save_label}.pdf"

                    print(f"Saving to {save_path_png}")
                    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
                    print(f"Saving to {save_path_pdf}")
                    plt.savefig(save_path_pdf, bbox_inches="tight")
                    plt.close()

            # Accumulate into class-level data if level="class" or "fleet"
            if level in ["class", "fleet"] and stacked_data is not None:
                if class_stacked_data is None:
                    class_stacked_data = stacked_data.copy()
                else:
                    # Ensure numeric values before adding
                    stacked_data = stacked_data.apply(pd.to_numeric, errors='coerce')
                    class_stacked_data = class_stacked_data.add(stacked_data, fill_value=0)

        # If level="class", plot the results after iterating over all vessel sizes
        if level == "class" and class_stacked_data is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.stackplot(class_stacked_data.index, class_stacked_data.T, labels=class_stacked_data.columns, alpha=0.8)

            ax.set_xlabel('Date', fontsize=22)
            if ylabel is None:
                ax.set_ylabel(column, fontsize=22)
            else:
                ax.set_ylabel(ylabel, fontsize=22)
            ax.set_title(f"{vessel_class} Fleet", fontsize=24)
            ax.tick_params(axis='both', labelsize=18)
            ax.grid(True)

            ax.legend(title='Fuel', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legend

            # Save the plot
            if save_label is not None and not save_label.startswith("_"):
                save_label = "_" + save_label
            
            save_path_png = f"plots/multifuel/class_{vessel_class}_{column}{save_label}.png"
            save_path_pdf = f"plots/multifuel/class_{vessel_class}_{column}{save_label}.pdf"

            print(f"Saving to {save_path_png}")
            plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
            print(f"Saving to {save_path_pdf}")
            plt.savefig(save_path_pdf, bbox_inches="tight")
            plt.close()

        # Accumulate into fleet-level data if level="fleet"
        if level == "fleet" and class_stacked_data is not None:
            if fleet_stacked_data is None:
                fleet_stacked_data = class_stacked_data.copy()
            else:
                # Ensure numeric values before adding
                class_stacked_data = class_stacked_data.apply(pd.to_numeric, errors='coerce')
                fleet_stacked_data = fleet_stacked_data.add(class_stacked_data, fill_value=0)

    # If level="fleet", plot after iterating over all vessel classes
    if level == "fleet" and fleet_stacked_data is not None:
        fleet_stacked_data = fleet_stacked_data.sort_index()  # Ensure chronological order
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.stackplot(fleet_stacked_data.index, fleet_stacked_data.T, labels=fleet_stacked_data.columns, alpha=0.8)

        ax.set_xlabel('Date', fontsize=22)
        if ylabel is None:
            ax.set_ylabel(column, fontsize=22)
        else:
            ax.set_ylabel(ylabel, fontsize=22)
        ax.set_title("Full Fleet", fontsize=24)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True)

        ax.legend(title='Fuel', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legend

        # Save the plot
        if save_label is not None and not save_label.startswith("_"):
            save_label = "_" + save_label

        save_path_png = f"plots/multifuel/fleet_{column}{save_label}.png"
        save_path_pdf = f"plots/multifuel/fleet_{column}{save_label}.pdf"

        print(f"Saving to {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        print(f"Saving to {save_path_pdf}")
        plt.savefig(save_path_pdf, bbox_inches="tight")
        plt.close()
        
def make_all_plots(results_file, results_label=None, regulations=None):
    """
    Run all plotting functions on the given results file.
    
    Parameters
    ----------
    results_file : str
        Path to the excel file containing results from the NavigaTE simulation
    results_label : str
        Label to append to filenames of plots produced for the given simulation result
    """

    ################################## Global results ##################################
#    global_results_dict = read_results_global(results_file)
    
#    # Fleet info
#    plot_fleet_info_column(global_results_dict, "TotalEquivalentWTW", info_type="global", ylabel="WTW Emissions (tonnes CO2e)", save_label=results_label)
#    plot_fleet_info_column(global_results_dict, "RegulationExpenses", info_type="global", save_label=results_label)
#    plot_fleet_info_column(global_results_dict, "VesselExpenses", info_type="global", ylabel="Vessel Expenses (USD)", save_label=results_label)
#    plot_fleet_info_column(global_results_dict, "Expenses", info_type="global", ylabel="Total Expenses (USD)", save_label=results_label)

#    # Global info
#    plot_global_stacked_fuel_info(global_results_dict, "ConsumedEnergy", ylabel="Fuel Energy Consumed (GJ)", save_label=results_label)
#    plot_global_stacked_fuel_info(global_results_dict, "FuelRelatedExpenses", save_label=results_label)
    ####################################################################################
#
#    ################################ Regulation results ################################
#    regulation_results_dict = read_results_regulation(results_file)
#    for regulation in regulations:
#        if regulation in regulation_results_dict:
#            plot_fleet_info_column(regulation_results_dict["net_zero_regulation"], "FlexibilityCost", info_type = "regulation", save_label=results_label)
#            plot_regulation_vessel_info(regulation_results_dict, "VesselThreshold", ylabel="Max kg CO2e WTW / GJ fuel", save_label=results_label)
#        else:
#            print(f"Regulation {regulation} is not in the simulation output. No plots produced.")
#    ####################################################################################
#
#    ################################## Fleet results ###################################
    fleet_results_dict = read_results_fleet(results_file)

    # Plot fleet info for each vessel class and size
    plot_main_fuel_info_fleet(fleet_results_dict, "ExistingVessels", level="size", ylabel="Existing Vessels", save_label=results_label)
    plot_main_fuel_info_fleet(fleet_results_dict, "Newbuilds", level="size", save_label=results_label)
    plot_main_fuel_info_fleet(fleet_results_dict, "Scrap", level="size", ylabel = "Scrapped Vessels", save_label=results_label)

    # Plot fleet info for each vessel class (aggregated over sizes)
    plot_main_fuel_info_fleet(fleet_results_dict, "ExistingVessels", level="class", ylabel="Existing Vessels", save_label=results_label)
    plot_main_fuel_info_fleet(fleet_results_dict, "Newbuilds", level="class", save_label=results_label)
    plot_main_fuel_info_fleet(fleet_results_dict, "Scrap", level="class", ylabel="Scrapped Vessels", save_label=results_label)

    # Plot fleet info for the full fleet (aggregated over all classes and sizes)
    plot_main_fuel_info_fleet(fleet_results_dict, "ExistingVessels", level="fleet", ylabel="Existing Vessels", save_label=results_label)
    plot_main_fuel_info_fleet(fleet_results_dict, "Newbuilds", level="fleet", save_label=results_label)
    plot_main_fuel_info_fleet(fleet_results_dict, "Scrap", level="fleet", ylabel="Scrapped Vessels", save_label=results_label)
    
    ####################################################################################
            
def main():

    results_file_base = "multi_fuel_full_fleet/all_fuels_base/plots/all_fuels_base_excel_report.xlsx"
    make_all_plots(results_file_base, results_label="base")
    
    #results_file_mod_cap = "multi_fuel_full_fleet/all_fuels_mod_cap/plots/all_fuels_mod_cap_excel_report.xlsx"
    #make_all_plots(results_file_mod_cap, results_label="mod_cap")
    
    ###################################################################################################

main()

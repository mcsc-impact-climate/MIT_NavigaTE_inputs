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

vessel_size_title_with_class = {
    "bulk_carrier_capesize_ice": "Bulk Carrier (Capesize)",
    "bulk_carrier_handy_ice": "Bulk Carrier (Handy)",
    "bulk_carrier_panamax_ice": "Bulk Carrier (Panamax)",
    "container_15000_teu_ice": "Container (15,000 TEU)",
    "container_8000_teu_ice": "Container (8,000 TEU)",
    "container_3500_teu_ice": "Container (3,500 TEU)",
    "tanker_100k_dwt_ice": "Tanker (100k DWT)",
    "tanker_300k_dwt_ice": "Tanker (300k DWT)",
    "tanker_35k_dwt_ice": "Tanker (35k DWT)",
    "gas_carrier_100k_cbm_ice": "Gas Carrier (100k m$^3$)",
}

fuel_colors = {
    "ammonia": "tab:blue",
    "hydrogen": "tab:orange",
    "methanol": "tab:green",
    "diesel": "tab:red",
    "oil": "tab:grey"
}

fuel_names = {
    "ammonia": "ammonia",
    "hydrogen": "liquid hydrogen",
    "methanol": "methanol",
    "diesel": "FT diesel",
    "oil": "LSFO"
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
    
def sort_color_label_columns(columns):
    """
    Returns:
        sorted_cols: list of original column names sorted according to fuel_colors
        color_list: list of colors matching sorted_cols
        label_list: list of display names (from fuel_names) matching sorted_cols
    """
    sorted_cols = []
    colors = []
    labels = []
    remaining_cols = list(columns)

    for fuel_key in fuel_colors.keys():
        matching_cols = [col for col in remaining_cols if fuel_key in col.lower()]
        for col in matching_cols:
            sorted_cols.append(col)
            colors.append(fuel_colors[fuel_key])
            labels.append(fuel_names[fuel_key])
            remaining_cols.remove(col)

    for col in remaining_cols:
        sorted_cols.append(col)
        colors.append("grey")
        labels.append(col)

    return sorted_cols, colors, labels
    
def plot_global_stacked_fuel_info(global_results_dict, column_name, ylabel=None, save_label=None):

    # Access the 'fuel info' dictionary
    fuel_info = global_results_dict['fuel info']
    
    # Initialize an empty DataFrame to collect data for stacking
    stacked_data = pd.DataFrame()

    # Loop through each fuel and extract the column if it exists
    for fuel, df in fuel_info.items():
        if column_name in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

            # Determine label from fuel_names (fallback: original fuel name)
            label = None
            for fuel_key in fuel_names.keys():
                if fuel_key in fuel.lower():
                    label = fuel_names[fuel_key]
                    break
            if label is None:
                label = fuel

            if stacked_data.empty:
                stacked_data['Date'] = df['Date']
                stacked_data[label] = df[column_name]
            else:
                stacked_data = pd.merge(
                    stacked_data,
                    df[['Date', column_name]].rename(columns={column_name: label}),
                    on='Date',
                    how='outer'
                )

    # Ensure all columns except Date are numeric
    stacked_data = stacked_data.sort_values('Date')

    for col in stacked_data.columns:
        if col != 'Date':
            stacked_data[col] = pd.to_numeric(stacked_data[col], errors='coerce')

    # Fill NaN values with 0
    stacked_data = stacked_data.fillna(0)

    stacked_data.set_index('Date', inplace=True)

    # Sort columns and assign colors and labels
    sorted_cols, color_list, label_list = sort_color_label_columns(stacked_data.columns)

    # Plotting the stacked area chart
    plt.figure(figsize=(14, 6))
    plt.stackplot(
        stacked_data.index,
        [stacked_data[col] for col in sorted_cols],
        labels=label_list,
        colors=color_list,
        alpha=0.8
    )

    # Labeling
    plt.xlabel('Date', fontsize=22)
    plt.ylabel(ylabel if ylabel else column_name, fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title='Fuel', fontsize=20, title_fontsize=22, loc='upper left', bbox_to_anchor=(1, 1))

    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot
    if save_label is not None and not save_label.startswith("_"):
        save_label = "_" + save_label
        
    save_path_png = f"plots/multifuel/global_stacked_{column_name}{save_label}.png"
    save_path_pdf = f"plots/multifuel/global_stacked_{column_name}{save_label}.pdf"
        
    print(f"Saving to {save_path_png}")
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    
    print(f"Saving to {save_path_pdf}")
    plt.savefig(save_path_pdf, bbox_inches="tight")
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

    def sort_color_label_columns(columns):
        """
        Returns:
            sorted_cols: list of original column names sorted according to fuel_colors
            color_list: list of colors matching sorted_cols
            label_list: list of display names (from fuel_names) matching sorted_cols
        """
        sorted_cols = []
        colors = []
        labels = []
        remaining_cols = list(columns)

        for fuel_key in fuel_colors.keys():
            matching_cols = [col for col in remaining_cols if fuel_key in col.lower()]
            for col in matching_cols:
                sorted_cols.append(col)
                colors.append(fuel_colors[fuel_key])
                labels.append(fuel_names[fuel_key])
                remaining_cols.remove(col)

        # Any remaining fuels that weren't explicitly matched
        for col in remaining_cols:
            sorted_cols.append(col)
            colors.append("grey")
            labels.append(col)

        return sorted_cols, colors, labels

    fleet_stacked_data = None

    for vessel_class in vessel_classes:
        fleet_results_class_dict = fleet_results_dict[vessel_class]

        class_stacked_data = None

        for vessel_size in fleet_results_class_dict:
            main_fuel_size_dict = fleet_results_class_dict[vessel_size]["main fuel info"]

            stacked_data = None

            for fuel, df in main_fuel_size_dict.items():
                if column in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    
                    df_subset = df[['Date', column]].rename(columns={column: fuel})
                    df_subset.set_index("Date", inplace=True)

                    df_subset[fuel] = pd.to_numeric(df_subset[fuel], errors='coerce')

                    if stacked_data is None:
                        stacked_data = df_subset
                    else:
                        stacked_data = stacked_data.add(df_subset, fill_value=0)

            if stacked_data is not None:
                stacked_data = stacked_data.fillna(0)

                sorted_cols, color_list, label_list = sort_color_label_columns(stacked_data.columns)

                if level == "size":
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.stackplot(
                        stacked_data.index,
                        [stacked_data[col] for col in sorted_cols],
                        labels=label_list,
                        colors=color_list,
                        alpha=0.8
                    )

                    ax.set_xlabel('Date', fontsize=22)
                    ax.set_ylabel(ylabel if ylabel else column, fontsize=22)
                    ax.set_title(f"{vessel_class}: {vessel_size}", fontsize=24)
                    ax.tick_params(axis='both', labelsize=18)
                    ax.grid(True)

                    ax.legend(title='Fuel', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
                    plt.tight_layout(rect=[0, 0, 0.85, 1])

                    if save_label is not None and not save_label.startswith("_"):
                        save_label = "_" + save_label

                    save_path_png = f"plots/multifuel/size_{vessel_size}_{column}{save_label}.png"
                    save_path_pdf = f"plots/multifuel/size_{vessel_size}_{column}{save_label}.pdf"

                    print(f"Saving to {save_path_png}")
                    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
                    print(f"Saving to {save_path_pdf}")
                    plt.savefig(save_path_pdf, bbox_inches="tight")
                    plt.close()

                if level in ["class", "fleet"]:
                    if class_stacked_data is None:
                        class_stacked_data = stacked_data.copy()
                    else:
                        class_stacked_data = class_stacked_data.add(stacked_data, fill_value=0)

        if level == "class" and class_stacked_data is not None:
            class_stacked_data = class_stacked_data.fillna(0)
            sorted_cols, color_list, label_list = sort_color_label_columns(class_stacked_data.columns)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.stackplot(
                class_stacked_data.index,
                [class_stacked_data[col] for col in sorted_cols],
                labels=label_list,
                colors=color_list,
                alpha=0.8
            )

            ax.set_xlabel('Date', fontsize=22)
            ax.set_ylabel(ylabel if ylabel else column, fontsize=22)
            ax.set_title(f"{vessel_class} Fleet", fontsize=24)
            ax.tick_params(axis='both', labelsize=18)
            ax.grid(True)

            ax.legend(title='Fuel', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            if save_label is not None and not save_label.startswith("_"):
                save_label = "_" + save_label

            save_path_png = f"plots/multifuel/class_{vessel_class}_{column}{save_label}.png"
            save_path_pdf = f"plots/multifuel/class_{vessel_class}_{column}{save_label}.pdf"

            print(f"Saving to {save_path_png}")
            plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
            print(f"Saving to {save_path_pdf}")
            plt.savefig(save_path_pdf, bbox_inches="tight")
            plt.close()

        if level == "fleet" and class_stacked_data is not None:
            if fleet_stacked_data is None:
                fleet_stacked_data = class_stacked_data.copy()
            else:
                fleet_stacked_data = fleet_stacked_data.add(class_stacked_data, fill_value=0)

    if level == "fleet" and fleet_stacked_data is not None:
        fleet_stacked_data = fleet_stacked_data.fillna(0)
        sorted_cols, color_list, label_list = sort_color_label_columns(fleet_stacked_data.columns)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.stackplot(
            fleet_stacked_data.index,
            [fleet_stacked_data[col] for col in sorted_cols],
            labels=label_list,
            colors=color_list,
            alpha=0.8
        )

        ax.set_xlabel('Date', fontsize=22)
        ax.set_ylabel(ylabel if ylabel else column, fontsize=22)
        ax.set_title("Full Fleet", fontsize=24)
        ax.tick_params(axis='both', labelsize=18)
        ax.grid(True)

        ax.legend(title='Fuel', fontsize=14, title_fontsize=16, loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        if save_label is not None and not save_label.startswith("_"):
            save_label = "_" + save_label

        save_path_png = f"plots/multifuel/fleet_{column}{save_label}.png"
        save_path_pdf = f"plots/multifuel/fleet_{column}{save_label}.pdf"

        print(f"Saving to {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        print(f"Saving to {save_path_pdf}")
        plt.savefig(save_path_pdf, bbox_inches="tight")
        plt.close()

def read_results_vessel(filename, sheet_name="Vessels"):
    """
    Reads the 'Vessels' sheet from an Excel file and parses data into a nested dictionary:

        {
            <base_vessel_ice>: {
                <fuel>: <DataFrame with Date, Time, and relevant columns for this fuel>
            }
        }

    Assumes the Excel sheet has multi-row headers structured as:
    
    - Row 0: Vessel names, e.g., "container_15000_teu_ice_liquid_hydrogen"
    - Row 1: Measurement types, e.g., "ConsumedEnergy", "TotalEquivalentWTT"
    - Row 2: Fuel types (only present for certain measurements), e.g., "hydrogen", "methanol"
    
    Data starts from row 5 (row index 5).
    """

    import pandas as pd

    # === 1. Read the raw Excel sheet without parsing any row as a header ===
    df_raw = pd.read_excel(filename, sheet_name=sheet_name, header=None)
    
    # === 2. Extract the data portion starting from row 5 ===
    df_data = df_raw.iloc[5:].reset_index(drop=True)  # Skip header rows, reset row index for clean DataFrame

    # === 3. Extract header rows to identify column information ===
    row0 = df_raw.iloc[0]  # Full vessel+fuel names
    row1 = df_raw.iloc[1]  # Primary measurement type (e.g., ConsumedEnergy, TotalEquivalentWTT)
    row2 = df_raw.iloc[2]  # Fuel names, only populated for ConsumedEnergy

    # === 4. Extract the date and time series (assumed common to all vessel data) ===
    date_series = df_data.iloc[:, 0].values
    time_series = df_data.iloc[:, 1].values

    # === 5. Build a column rename map based on row1 and row2 ===
    # This will rename columns based on the measurement type, and append the fuel if applicable.
    col_name_map = {}
    for col_idx in range(df_raw.shape[1]):
        if col_idx < 2:
            continue  # Skip the first two columns (Date and Time)
        
        # Extract labels from rows 1 and 2
        label_from_row1 = str(row1[col_idx]) if pd.notna(row1[col_idx]) else ""
        label_from_row2 = str(row2[col_idx]) if pd.notna(row2[col_idx]) else ""

        # If the measurement is "ConsumedEnergy", include the fuel name in parentheses
        if label_from_row1 == "ConsumedEnergy":
            if label_from_row2:
                final_name = f"{label_from_row1} ({label_from_row2})"
            else:
                # If no fuel name is provided, just use the measurement type
                final_name = label_from_row1
        else:
            # Use the primary measurement type directly for other cases
            final_name = label_from_row1
        
        col_name_map[col_idx] = final_name  # Map column index to new name

    # === 6. Group columns by base vessel name and fuel ===
    # vessel_bases is a nested dictionary:
    # vessel_bases[base_vessel_ice][fuel] = list of column indices associated with that fuel
    vessel_bases = {}
    
    for col_idx in range(df_raw.shape[1]):
        if col_idx < 2:
            continue  # Skip Date and Time columns

        val = str(row0[col_idx]) if pd.notna(row0[col_idx]) else ""
        
        if "ice_" in val:
            # Split the vessel+fuel name at "ice_"
            base_part, fuel_part = val.split("ice_", 1)

            # base_vessel is everything up to and including "ice"
            base_vessel = base_part + "ice"
            fuel = fuel_part  # everything after "ice_"

            # Initialize nested dicts as needed
            if base_vessel not in vessel_bases:
                vessel_bases[base_vessel] = {}
            
            # Append the column index to the appropriate fuel list
            vessel_bases[base_vessel].setdefault(fuel, []).append(col_idx)

    # === 7. Construct the nested output dictionary ===
    # Structure:
    # vessel_results_dict[base_vessel][fuel] = DataFrame with relevant columns for that fuel
    vessel_results_dict = {}

    for base_vessel, fuels_dict in vessel_bases.items():
        # Initialize the dict for this vessel
        vessel_results_dict[base_vessel] = {}

        for fuel, col_indices in fuels_dict.items():
            # Prepare a data dictionary to hold columns for the new DataFrame
            data_dict = {
                "Date": date_series,           # Always include Date
                "Time (days)": time_series     # Always include Time (days)
            }

            # Add each measurement column related to this fuel
            for cidx in col_indices:
                new_col_name = col_name_map.get(cidx, f"Unknown_{cidx}")  # Use the rename map
                col_values = df_data.iloc[:, cidx].values                # Extract column values
                data_dict[new_col_name] = col_values                     # Add to data dict

            # Build the DataFrame for this vessel + fuel
            vessel_df = pd.DataFrame(data_dict)

            # Store the DataFrame in the nested dictionary
            vessel_results_dict[base_vessel][fuel] = vessel_df

    # === 8. Return the complete nested results dictionary ===
    return vessel_results_dict
    
def plot_vessel_fuel_metric(vessel_results_dict, column_name, xlabel=None, save_label=None, relative_to_lsfo=False):
    """
    Plots a given column (e.g., InvestmentMetricExpected) for each vessel (e.g., gas_carrier_100k_cbm_ice)
    over all fuels. Each fuel is shown as a colored marker at the same horizontal level as the vessel label.

    Parameters
    ----------
    vessel_results_dict : dict
        Dictionary of vessel results, structured as:
        { base_vessel_ice : { fuel: DataFrame( columns... ) } }
    column_name : str
        The name of the column to extract from each vessel+fuel DataFrame (e.g., "InvestmentMetricExpected").
    xlabel : str, optional
        Label for the x-axis (the default is column_name).
    save_label : str, optional
        Label to append to filenames for saving the plot.
    """

    # --- Initialize plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # --- Collect vessels and corresponding metrics ---
    vessel_names = list(vessel_results_dict.keys())
    vessel_names.sort()  # Optional: sort for consistent plotting order

    # Create a mapping from fuels to consistent colors
    colors_for_fuels = fuel_colors
    labels_for_fuels = fuel_names

    # For each vessel, plot a horizontal line and scatter points for each fuel
    y_ticks = []
    y_tick_labels = []
    y_pos = 0

    for vessel in vessel_names:
        fuel_dict = vessel_results_dict[vessel]

        # Prepare y-axis label for vessel (with nicer formatting if available)
        vessel_label = vessel_size_title_with_class.get(vessel, vessel)
        y_ticks.append(y_pos)
        y_tick_labels.append(vessel_label)
        
        lsfo_value = fuel_dict["oil"][column_name].mean()

        for fuel, df in fuel_dict.items():
            # Extract the column value (use the last available data point)
            if column_name in df.columns:
                value = df[column_name].mean()  # Average
            else:
                print(f"Column '{column_name}' not found for {vessel} {fuel}")
                continue

            # Get color and label for the fuel
            color = None
            label = None

            for fuel_key in colors_for_fuels.keys():
                if fuel_key in fuel.lower():
                    color = colors_for_fuels[fuel_key]
                    label = labels_for_fuels[fuel_key]
                    break

            if color is None:
                color = 'gray'
                label = fuel

            # Plot the point
            if relative_to_lsfo:
                ax.scatter(value/lsfo_value, y_pos, color=color, label=label, s=150, edgecolors='k')
            else:
                ax.scatter(value, y_pos, color=color, label=label, s=150, edgecolors='k')

        y_pos += 1  # Move to the next row (vessel)

    # --- Customize axes ---
    final_xlabel = xlabel if xlabel else column_name
    if relative_to_lsfo:
        final_xlabel += "\n(relative to LSFO)"
    ax.set_xlabel(final_xlabel if xlabel else column_name, fontsize=22)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # --- Build a custom legend with unique fuel names ---
    handles = []
    labels = []
    for fuel_key, color in colors_for_fuels.items():
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=15, markeredgecolor='k'))
        labels.append(labels_for_fuels[fuel_key])

    # Place legend **outside** plot area to the right
    legend = ax.legend(handles, labels,
                       title='Main Fuel',
                       fontsize=16,
                       title_fontsize=18,
                       loc='center left',
                       bbox_to_anchor=(1.02, 0.5),  # Position it outside, center aligned vertically
                       borderaxespad=0)

    # --- Adjust layout to make room for the legend ---
    plt.subplots_adjust(right=0.75)  # Shrinks the plot area to the left, leaves space on right

    xmin, xmax = ax.get_xlim()
    if xmin < 0:
        ax.set_xlim(0, xmax)

    # --- Save plot ---
    if save_label is not None and not save_label.startswith("_"):
        save_label = "_" + save_label
    else:
        save_label = ""

    if relative_to_lsfo:
        save_path_png = f"plots/multifuel/vessel_fuel_{column_name}{save_label}_rel_to_lsfo.png"
        save_path_pdf = f"plots/multifuel/vessel_fuel_{column_name}{save_label}_rel_to_lsfo.pdf"
    else:
        save_path_png = f"plots/multifuel/vessel_fuel_{column_name}{save_label}.png"
        save_path_pdf = f"plots/multifuel/vessel_fuel_{column_name}{save_label}.pdf"

    print(f"Saving to {save_path_png}")
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")

    print(f"Saving to {save_path_pdf}")
    plt.savefig(save_path_pdf, bbox_inches="tight")

    plt.close()
    
def plot_vessel_fuel_stacked_histograms(vessel_results_dict, column_names, column_labels, xlabel=None, stack_label=None, save_label=None):
    """
    For each vessel, plots a horizontal stacked histogram (bar chart) for the provided list of column names.
    Each fuel has one bar, stacked by the specified column values.

    Parameters
    ----------
    vessel_results_dict : dict
        Dictionary of vessel results, structured as:
        { base_vessel_ice : { fuel: DataFrame( columns... ) } }
    column_names : list of str
        List of column names to include in the stacked histogram.
    xlabel : str, optional
        Label for the x-axis (default is 'Value').
    stack_label : str, optional
        Label to append to filenames for identifying the stack plot content.
    save_label : str, optional
        Label to append to filenames for saving the plots.
    """

    # --- Create a consistent color map for the stack layers ---
    cmap = matplotlib.colormaps.get_cmap('tab20')  # Matplotlib 3.7+ compliant
    column_colors = cmap(np.linspace(0, 1, len(column_names)))
    column_color_map = {col: column_colors[i] for i, col in enumerate(column_names)}

    for vessel, fuel_dict in vessel_results_dict.items():
        vessel_label = vessel_size_title_with_class.get(vessel, vessel)

        # Prepare data for this vessel
        fuels = []
        data_per_fuel = []

        # --- Process each fuel for this vessel ---
        for fuel, df in fuel_dict.items():
            # Get human-readable fuel label by matching substrings from fuel_names
            readable_fuel_label = None
            for fuel_key, fuel_display in fuel_names.items():
                if fuel_key.lower() in fuel.lower():
                    readable_fuel_label = fuel_display
                    break

            # Fallback if no match found
            if readable_fuel_label is None:
                readable_fuel_label = fuel

            fuels.append(readable_fuel_label)
            stacked_values = []

            for col in column_names:
                if col not in df.columns:
                    print(f"Column '{col}' not found for {vessel} {fuel}. Skipping...")
                    stacked_values.append(0)
                    continue

                # Get the mean value for the column
                value = df[col].mean()
                stacked_values.append(value)

            data_per_fuel.append(stacked_values)

        # Convert data into a DataFrame for plotting
        plot_df = pd.DataFrame(data_per_fuel, columns=column_names, index=fuels)

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create horizontal stacked bars
        left = np.zeros(len(plot_df))
        i=0
        for col in column_names:
            ax.barh(plot_df.index,
                    plot_df[col],
                    left=left,
                    label=column_labels[i],
                    color=column_color_map[col])
            left += plot_df[col]
            i+=1

        # Customize axes
        ax.set_title(vessel_label, fontsize=26)
        ax.set_xlabel("Mean " + xlabel if xlabel else 'Value', fontsize=24)
        ax.set_ylabel('Main Fuel', fontsize=24)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)

        # Build and place legend outside the plot area
        ax.legend(fontsize=18, loc='center left', bbox_to_anchor=(1.02, 0.5))
        plt.subplots_adjust(right=0.75)

        # Save plot
        if save_label is not None and not save_label.startswith("_"):
            this_save_label = "_" + save_label
        else:
            this_save_label = ""

        safe_vessel_name = vessel.replace("/", "_")  # Make filename safe
        if stack_label:
            this_stack_label = stack_label
        else:
            this_stack_label = ""

        save_path_png = f"plots/multifuel/stacked_{safe_vessel_name}_{this_stack_label}{this_save_label}.png"
        save_path_pdf = f"plots/multifuel/stacked_{safe_vessel_name}_{this_stack_label}{this_save_label}.pdf"

        print(f"Saving {vessel_label} horizontal stacked plot to {save_path_png}")
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
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
#    plot_main_fuel_info_fleet(fleet_results_dict, "ExistingVessels", level="size", ylabel="Existing Vessels", save_label=results_label)
#    plot_main_fuel_info_fleet(fleet_results_dict, "Newbuilds", level="size", save_label=results_label)
#    plot_main_fuel_info_fleet(fleet_results_dict, "Scrap", level="size", ylabel = "Scrapped Vessels", save_label=results_label)

    # Plot fleet info for each vessel class (aggregated over sizes)
#    plot_main_fuel_info_fleet(fleet_results_dict, "ExistingVessels", level="class", ylabel="Existing Vessels", save_label=results_label)
#    plot_main_fuel_info_fleet(fleet_results_dict, "Newbuilds", level="class", save_label=results_label)
#    plot_main_fuel_info_fleet(fleet_results_dict, "Scrap", level="class", ylabel="Scrapped Vessels", save_label=results_label)
#
#    # Plot fleet info for the full fleet (aggregated over all classes and sizes)
    plot_main_fuel_info_fleet(fleet_results_dict, "ExistingVessels", level="fleet", ylabel="Existing Vessels", save_label=results_label)
    plot_main_fuel_info_fleet(fleet_results_dict, "Newbuilds", level="fleet", save_label=results_label)
    plot_main_fuel_info_fleet(fleet_results_dict, "Scrap", level="fleet", ylabel="Scrapped Vessels", save_label=results_label)
    
    # Plot vessel metrics by fuel
    vessel_results_dict = read_results_vessel(results_file)
    plot_vessel_fuel_metric(vessel_results_dict, "InvestmentMetricExpected", xlabel="Vessel Investment Metric", save_label=results_label, relative_to_lsfo=True)
    plot_vessel_fuel_metric(vessel_results_dict, "CargoMiles", xlabel="Annual Vessel Cargo Miles", save_label=results_label, relative_to_lsfo=True)
    plot_vessel_fuel_metric(vessel_results_dict, "TotalCost", xlabel="Annual Vessel Cost (USD)", save_label=results_label, relative_to_lsfo=True)
#    plot_vessel_fuel_stacked_histograms(vessel_results_dict, ["TotalBaseCost", "TotalPowerSystemCost", "TotalFuelOPEX", "TotalTankCost"], ["Base Vessel", "Power System", "Fuel Purchase", "Tanks"], xlabel="Annual Vessel Cost (USD)", stack_label="TotalCost", save_label=results_label)
    
    ####################################################################################
            
def main():

#    results_file_base = "multi_fuel_full_fleet/all_fuels_base/plots/all_fuels_base_excel_report.xlsx"
#    make_all_plots(results_file_base, results_label="base")
#
#    results_file_base = "multi_fuel_full_fleet/all_fuels_base/plots/all_fuels_base_no_cm_excel_report.xlsx"
#    make_all_plots(results_file_base, results_label="base_no_cm")
#
    results_file_mod_cap = "multi_fuel_full_fleet/all_fuels_mod_cap_orig_tank/plots/all_fuels_mod_cap_orig_tank_excel_report.xlsx"
    make_all_plots(results_file_mod_cap, results_label="mod_cap_orig_tank")

#    results_file_mod_cap = "multi_fuel_full_fleet/all_fuels_mod_cap_orig_tank/plots/all_fuels_mod_cap_orig_tank_no_cm_excel_report.xlsx"
#    make_all_plots(results_file_mod_cap, results_label="mod_cap_orig_tank_no_cm")

#    results_file_mod_cap = "multi_fuel_full_fleet/all_fuels_mod_cap/plots/all_fuels_mod_cap_excel_report.xlsx"
#    make_all_plots(results_file_mod_cap, results_label="mod_cap")
    
    
    ###################################################################################################

main()

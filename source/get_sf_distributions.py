"""
Date: 250318
Author: danikae
Purpose: Estimates stowage factor distributions for different vessel classes
"""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from common_tools import get_fuel_density, get_top_dir

top_dir = get_top_dir()


def get_gas_carrier_sfs():
    """
    Calculate the stowage factor, in m^3/tonne (or L/kg).

    In the case of gas carriers, LNG is primarily carried and has a well-defined density.
    LPG is the second-most carried and has a similar density.
    """

    LNG_density = get_fuel_density("lng")

    LNG_stowage_factor = 1 / LNG_density

    columns = ["Stowage Factor (m^3/tonne)", "Weight"]

    # Initialize empty DataFrame
    df = pd.DataFrame(columns=columns)

    # Create a new row DataFrame
    new_row = pd.DataFrame([[LNG_stowage_factor, 1]], columns=columns)

    # Concatenate the empty df with the new row
    df = pd.concat([df, new_row], ignore_index=True)

    return df


def api_to_density(api):
    """
    Converts the API of crude oil to density (in kg/L) based on the standard conversion presented in the footnote of this page: https://www.eia.gov/dnav/pet/pet_crd_api_adc_mbblpd_m.htm.

    Parameters
    ----------
    api : float
        API gravity of the crude oil

    Returns
    -------
    density : float
        Density of the crude oil, in kg/L
    """

    return 141.5 / (api + 131.5)


def calculate_crude_petrol_sfs():
    """
    Calculates the stowage factor distribution for crude petroleum
    based on 2023 production data.

    Returns
    -------
    result_df : pd.DataFrame
        DataFrame with 'Stowage Factor (m^3/tonne)' and normalized 'Weight' columns
    """

    # Path and loading CSV
    file_path = (
        f"{top_dir}/data/Crude_Oil_and_Lease_Condensate_Production_by_API_Gravity.csv"
    )

    # Skip header rows and set Month as index
    crude_petrol_df = pd.read_csv(file_path, skiprows=6, index_col="Month")

    # Filter for 2023 months
    crude_petrol_df_2023 = crude_petrol_df[crude_petrol_df.index.str.contains("2023")]

    # Compute mean production for each API category
    avg_production = crude_petrol_df_2023.mean()

    # Prepare lists for stowage factors and weights
    central_sf_list = []
    lower_sf_list = []
    upper_sf_list = []
    weight_list = []
    lower_api_list = []
    upper_api_list = []
    prod_rate_list = []

    # Iterate through each column
    for column in avg_production.index:
        # print(f"Processing column: {column}")

        # Extract API gravity values
        match = re.search(
            r"for ([\d\.]+)(?: to ([\d\.]+)| or (Higher|Lower)) Degrees API", column
        )

        if match:
            lower_api = float(match.group(1))
            lower_api_list.append(lower_api)
            upper_api = match.group(2)  # This is None if "or Higher" / "or Lower"
            upper_api_list.append(upper_api)

            # Check if column mentions "Higher" or "Lower"
            if "or Higher" in str(column):
                # Use the inner value of the range (lower bound)
                central_api = lower_api
                upper_api = lower_api
                # print(f"  -> 'or Higher' detected, using lower_api = {central_api}")
            elif "or Lower" in column:
                # Use the inner value of the range (upper bound)
                central_api = lower_api
                upper_api = lower_api
                # print(f"  -> 'or Lower' detected, using upper_api = {central_api}")
            elif upper_api:
                # Midpoint for ranged categories
                central_api = (lower_api + float(upper_api)) / 2
                # print(f"  -> Range detected, using midpoint = {central_api}")
            else:
                # Shouldn't happen, but fallback
                central_api = lower_api
                # print(f"  -> No upper bound, using lower_api = {central_api}")

            # Convert API to density (kg/L)
            upper_api = float(upper_api)
            central_density = api_to_density(central_api)
            upper_density = api_to_density(upper_api)
            lower_density = api_to_density(lower_api)

            # Stowage factor (m³/tonne or L/kg)
            central_stowage_factor = 1 / central_density
            upper_stowage_factor = 1 / upper_density
            lower_stowage_factor = 1 / lower_density

            # Average production as weight
            weight = avg_production[column]

            # Skip if weight is zero or NaN
            if pd.isna(weight) or weight == 0:
                continue

            # Append results
            central_sf_list.append(central_stowage_factor)
            lower_sf_list.append(lower_stowage_factor)
            upper_sf_list.append(upper_stowage_factor)
            weight_list.append(weight)
            prod_rate_list.append(weight)

    # Create the result DataFrame
    central_result_df = pd.DataFrame(
        {"Stowage Factor (m^3/tonne)": central_sf_list, "Weight": weight_list}
    )
    lower_result_df = pd.DataFrame(
        {"Stowage Factor (m^3/tonne)": lower_sf_list, "Weight": weight_list}
    )
    
    upper_result_df = pd.DataFrame(
        {"Stowage Factor (m^3/tonne)": upper_sf_list, "Weight": weight_list}
    )

    # Normalize weights to sum to 1
    central_result_df["Weight"] = central_result_df["Weight"] / central_result_df["Weight"].sum()
    lower_result_df["Weight"] = lower_result_df["Weight"] / lower_result_df["Weight"].sum()
    upper_result_df["Weight"] = upper_result_df["Weight"] / upper_result_df["Weight"].sum()

    # Sort by increasing stowage factor
    central_result_df = central_result_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(
        drop=True
    )
    lower_result_df = lower_result_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(
        drop=True
    )
    upper_result_df = upper_result_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(
        drop=True
    )
    
    result_save_df = pd.DataFrame({"Lower API": lower_api_list, "Upper API": upper_api_list, "Stowage Factor (m^3/tonne)": central_sf_list, "Average US Lower 48 production (thousand barrels / day)": prod_rate_list, "Weight": weight_list})
    result_save_df["Weight"] = result_save_df["Weight"] / result_save_df["Weight"].sum()
    
    result_save_df = result_save_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(
        drop=True
    )
    result_save_df.to_csv(f"{top_dir}/tables/crude_petrol_api_prod_info.csv")
    print(f"Saved crude oil api info to {top_dir}/tables/crude_petrol_api_prod_info.csv")

    return central_result_df, lower_result_df, upper_result_df


def readMeta():
    """
    Reads in the metadata file (functionally keys) for the FAF5 data

    Parameters
    ----------
    None

    Returns
    -------
    dest (pd.DataFrame): A pandas dataframe containing (currently) all domestic regions from the FAF5_metadata
    mode (pd.DataFrame): A pandas dataframe containing (currently) all modes of transit used in the FAF5_metadata
    """

    # Read in Meta Data
    metaPath = f"{top_dir}/data/FAF5.6.1/FAF5_metadata.xlsx"
    meta = pd.ExcelFile(metaPath)

    # Only include truck rail and water
    modes = pd.read_excel(meta, "Mode", index_col="Description")
    trade_types = pd.read_excel(meta, "Trade Type", index_col="Description")
    comms = pd.read_excel(meta, "Commodity (SCTG2)", index_col="Description")
    return modes, comms, trade_types


def plot_weighted_sf_petroleum(sf_type="central"):
    """
    Plots a bar chart of the stowage factors weighted by their normalized production shares.
    """

    # Sort the dataframe by stowage factor (optional if already sorted)
    df_central, df_lower, df_upper = calculate_crude_petrol_sfs()
    if sf_type == "central":
        df = df_central
    elif sf_type == "lower":
        df = df_lower
    else:
        df = df_upper
    df_sorted = df.sort_values(by="Stowage Factor (m^3/tonne)")

    sf_values = df_sorted["Stowage Factor (m^3/tonne)"]
    weights = df_sorted["Weight"]

    plt.figure(figsize=(10, 6))
    plt.bar(
        sf_values, weights, width=0.015, align="center", edgecolor="black", alpha=0.8
    )

    # Labeling and styling
    plt.xlabel("Stowage Factor (m$^3$/tonne)", fontsize=22)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)
    # plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Optional: rotate x labels if there are many bars
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"{top_dir}/plots/sf_distribution_crude_oil.png", dpi=300)
    plt.savefig(f"{top_dir}/plots/sf_distribution_crude_oil.pdf")


def get_commodity_info(filepath=f"{top_dir}/data/faf5_commodity_info.csv"):
    """
    Collect stowage factor info and vessel class association for each commodity.

    Parameters
    ----------
    filepath : str
        Filepath to the csv file to read from

    Returns
    ----------
    commodity_info_df : pd.DataFrame
        Dataframe containing the stowage factor and vessel class info for each commodity
    """

    commodity_info_df = pd.read_csv(
        f"{top_dir}/data/faf5_commodity_info.csv",
        index_col="Description",
        usecols=[
            "Description",
            "Vessel Class",
            "Lower Stowage Factor (m^3/tonne)",
            "Upper Stowage Factor (m^3/tonne)",
        ],
    )
    return commodity_info_df


def plot_commodity_sfs():
    # Read the commodity info data
    commodity_info_df = get_commodity_info()

    # Filter bulk_carrier and container commodities
    bulk_df = commodity_info_df[
        commodity_info_df["Vessel Class"] == "bulk_carrier"
    ].dropna()
    container_df = commodity_info_df[
        commodity_info_df["Vessel Class"] == "container"
    ].dropna()

    # Get tonne-mile sorted commodity lists
    bulk_sorted_commodities = get_sorted_commodities_by_tmiles("bulk_carrier")
    container_sorted_commodities = get_sorted_commodities_by_tmiles("container")

    # Plot bulk_carrier commodities with sorted order
    plot_sf_ranges(
        bulk_df,
        title="Stowage Factor Ranges for Bulk Carrier Commodities",
        color="steelblue",
        save_str="bulk",
        sorted_commodities=bulk_sorted_commodities,
    )

    # Plot container commodities with sorted order
    plot_sf_ranges(
        container_df,
        title="Stowage Factor Ranges for Container Commodities",
        color="darkorange",
        save_str="container",
        sorted_commodities=container_sorted_commodities,
    )


def get_sorted_commodities_by_tmiles(vessel_class):
    """
    Returns a list of commodity names sorted by tonne-miles, descending.

    Parameters
    ----------
    vessel_class : str
        Name of the vessel class

    Returns
    -------
    sorted_commodities : list of str
        Commodity descriptions sorted by tonne-miles
    """
    commodity_tmiles = get_commodity_tms(vessel_class)

    # If no tonne-miles data is available, return empty list
    if not commodity_tmiles:
        print(f"No tonne-mile data found for vessel class: {vessel_class}")
        return []

    sorted_commodities = [
        commodity
        for commodity, tmiles in sorted(
            commodity_tmiles.items(), key=lambda x: x[1], reverse=True
        )
    ]

    return sorted_commodities


def plot_sf_ranges(df, title, color, save_str, sorted_commodities=None):
    """
    Helper function to plot stowage factor ranges for the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with the commodities to plot.
    title : str
        Plot title.
    color : str
        Color of the bars.
    save_str : str
        String to append to the filename when saving plots.
    sorted_commodities : list of str, optional
        List of commodities to sort and plot in order.
    """
    if sorted_commodities:
        # Filter the sorted commodities to ones present in the dataframe
        sorted_commodities_in_df = [c for c in sorted_commodities if c in df.index]

        # Reverse the order for plotting (so highest tonne-miles appears on top)
        sorted_commodities_in_df = list(reversed(sorted_commodities_in_df))

        df_sorted = df.loc[sorted_commodities_in_df]
    else:
        df_sorted = df.sort_values(by="Lower Stowage Factor (m^3/tonne)")

    # Extract necessary values
    commodities = df_sorted.index.tolist()
    lower_sf = df_sorted["Lower Stowage Factor (m^3/tonne)"]
    upper_sf = df_sorted["Upper Stowage Factor (m^3/tonne)"]

    # Calculate ranges
    sf_range = upper_sf - lower_sf

    # Plot horizontal bars
    plt.figure(figsize=(10, 8))
    plt.barh(
        commodities,
        sf_range,
        left=lower_sf,
        height=0.6,
        color=color,
        edgecolor="black",
        alpha=0.7,
    )

    # Labels and styling
    plt.xlabel("Stowage Factor (m³/tonne)", fontsize=22)
    plt.ylabel("Commodity", fontsize=22)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tick_params(axis="both", which="minor", labelsize=18)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.xlim(left=0)

    plt.tight_layout()
    plt.savefig(f"{top_dir}/plots/commodity_sf_range_{save_str}.png", dpi=300)
    plt.savefig(f"{top_dir}/plots/commodity_sf_range_{save_str}.pdf")

    print(f"Saved stowage factor range plot for {save_str} to:")
    print(f"{top_dir}/plots/commodity_sf_range_{save_str}.png")
    print(f"{top_dir}/plots/commodity_sf_range_{save_str}.pdf")


def plot_sf_ranges_tanker(
    title="Stowage Factors for Tanker Commodities", color="navy", save_str="tanker"
):
    """
    Plots stowage factors for tanker commodities: crude petroleum, gasoline, and fuel oils,
    sorted by tonne-miles traveled.

    Parameters
    ----------
    title : str
        Title for the plot.
    color : str
        Color of the bars/markers.
    save_str : str
        String to append to the filename when saving plots.
    """
    # Constants for densities (in kg/L)
    GASOLINE_DENSITY = 0.749  # kg/L, from GREET 2024
    FUEL_OIL_DENSITY = get_fuel_density("lsfo")  # kg/L

    # Calculate stowage factors (m³/tonne = 1 / density in kg/L)
    GASOLINE_SF = 1 / GASOLINE_DENSITY
    FUEL_OIL_SF = 1 / FUEL_OIL_DENSITY

    # Get crude petroleum SF distribution min/max
    central_crude_petrol_df, lower_crude_petrol_df, upper_crude_petrol_df = calculate_crude_petrol_sfs()
    crude_min_sf = central_crude_petrol_df["Stowage Factor (m^3/tonne)"].min()
    crude_max_sf = central_crude_petrol_df["Stowage Factor (m^3/tonne)"].max()

    # Step 1: Get tonne-miles for tanker commodities
    commodity_tmiles = get_commodity_tms("tanker")

    # Step 2: Create a dictionary mapping commodity to its SF range
    tanker_data = {
        "Crude Petroleum": {
            "lower_sf": crude_min_sf,
            "upper_sf": crude_max_sf,
            "tmiles": commodity_tmiles.get("Crude petroleum", 0),
        },
        "Gasoline": {
            "lower_sf": GASOLINE_SF,
            "upper_sf": GASOLINE_SF,
            "tmiles": commodity_tmiles.get("Gasoline", 0),
        },
        "Fuel Oils": {
            "lower_sf": FUEL_OIL_SF,
            "upper_sf": FUEL_OIL_SF,
            "tmiles": commodity_tmiles.get("Fuel oils", 0),
        },
    }

    # Step 3: Sort commodities by tonne-miles (descending)
    sorted_commodities = sorted(
        tanker_data.items(), key=lambda x: x[1]["tmiles"], reverse=True
    )

    # Reverse the order for plotting (largest on top)
    sorted_commodities = list(reversed(sorted_commodities))

    # Step 4: Prepare data for plotting
    commodities = []
    lower_sf = []
    upper_sf = []

    for commodity_name, data in sorted_commodities:
        commodities.append(commodity_name)
        lower_sf.append(data["lower_sf"])
        upper_sf.append(data["upper_sf"])

    # Calculate ranges
    sf_range = [upper - lower for lower, upper in zip(lower_sf, upper_sf)]

    # Step 5: Plot horizontal bars (or markers if range is zero)
    plt.figure(figsize=(10, 8))  # Matching tonne-miles plot dimensions

    for idx, (commodity, lower, upper) in enumerate(
        zip(commodities, lower_sf, upper_sf)
    ):
        if lower == upper:
            # Plot a single point for gasoline and fuel oils
            plt.barh(
                commodity,
                width=0.01,
                left=lower,
                height=0.6,
                color=color,
                edgecolor="black",
                alpha=0.8,
            )
        else:
            # Plot a horizontal bar for crude petroleum range
            plt.barh(
                commodity,
                upper - lower,
                left=lower,
                height=0.6,
                color=color,
                edgecolor="black",
                alpha=0.7,
            )

    # Step 6: Labels and styling
    plt.xlabel("Stowage Factor (m$^3$/tonne)", fontsize=22)
    plt.ylabel("Commodity", fontsize=22)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tick_params(axis="both", which="minor", labelsize=18)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.xlim(left=0)

    plt.tight_layout()

    # Step 7: Save plots
    plt.savefig(f"{top_dir}/plots/commodity_sf_range_{save_str}.png", dpi=300)
    plt.savefig(f"{top_dir}/plots/commodity_sf_range_{save_str}.pdf")

    print(f"Saved tanker SF plot to:")
    print(f"{top_dir}/plots/commodity_sf_range_{save_str}.png")
    print(f"{top_dir}/plots/commodity_sf_range_{save_str}.pdf")


def make_filtered_faf5_csv():
    """
    Reads in the FAF5 data and saves the rows corresponding to imports and exports carried by ship to a csv file
    """

    faf5_data_df = pd.read_csv(
        f"{top_dir}/data/FAF5.6.1/FAF5.6.1.csv",
        usecols=["sctg2", "fr_inmode", "fr_outmode", "trade_type", "tmiles_2025"],
    )

    modes, comms, trade_types = readMeta()
    import_numeric_label = trade_types.loc[
        "Import flows (freight shipments moved from foreign origins to US domestic destinations)",
        "Numeric Label",
    ]
    export_numeric_label = trade_types.loc[
        "Export flows (freight shipments moved from US domestic origins to foreign destinations)",
        "Numeric Label",
    ]
    water_transport_numeric_label = modes.loc["Water", "Numeric Label"]

    # Set up selections
    cImport = faf5_data_df["trade_type"] == import_numeric_label
    cShipImport = faf5_data_df["fr_inmode"] == water_transport_numeric_label
    cExport = faf5_data_df["trade_type"] == export_numeric_label
    cShipExport = faf5_data_df["fr_outmode"] == water_transport_numeric_label

    # Save the basic columns to a csv for subsequent use in this script
    faf5_data_df.to_csv(f"{top_dir}/data/FAF5.6.1/FAF5.6.1_import_export_by_ship.csv")


def get_commodities_class(vessel_class):
    """
    Gets a list of commodities carried by the given vessel class

    Parameters
    ----------
    vessel_class : str
        Name of the vessel class
    """
    commodity_info_df = get_commodity_info()

    # Get commodities carried by the given vessel class
    commodities_class = commodity_info_df[
        commodity_info_df["Vessel Class"] == vessel_class
    ]
    return list(commodities_class.index)


def get_commodity_tms(vessel_class):
    """
    Uses the FAF5 data to get the tonne-miles traveled by each commodity carried by a given vessel class.

    Parameters
    ----------
    vessel_class : str
        Name of the vessel class

    Returns
    ----------
    commodity_info_df : pd.DataFrame
        Dataframe containing the stowage factor and vessel class info for each commodity
    """
    commodities = get_commodities_class(vessel_class)

    faf5_data = pd.read_csv(
        f"{top_dir}/data/FAF5.6.1/FAF5.6.1_import_export_by_ship.csv"
    )
    commodity_tmiles = {}
    modes, comms, trade_types = readMeta()
    for commodity in commodities:
        commodity_numeric_label = comms.loc[commodity, "Numeric Label"]
        commodity_tmiles[commodity] = faf5_data["tmiles_2025"][
            faf5_data["sctg2"] == commodity_numeric_label
        ].sum()

    return commodity_tmiles


def plot_commodity_tms(vessel_class):
    """
    Plots the tonne-miles traveled by each commodity carried by the given vessel class
    as a horizontal bar chart, using the same colors as plot_sf_ranges.

    Parameters
    ----------
    vessel_class : str
        Name of the vessel class
    """
    # Match vessel-class colors from plot_sf_ranges
    color_map = {
        "bulk_carrier": "steelblue",
        "container": "darkorange",
        "tanker": "navy",
        # Extend this dict with other classes as needed
    }

    # Default to 'teal' if vessel_class not in the color map
    chosen_color = color_map.get(vessel_class, "teal")

    # Get the tonne-miles dictionary
    commodity_tmiles = get_commodity_tms(vessel_class)

    if not commodity_tmiles:
        print(f"No data available for vessel class: {vessel_class}")
        return

    # Sort commodities by tonne-miles (optional for better visualization)
    sorted_commodities = sorted(
        commodity_tmiles.items(), key=lambda x: x[1], reverse=True
    )

    # Unzip into two lists
    commodities, tmiles = zip(*sorted_commodities)

    # Create the horizontal bar plot
    plt.figure(figsize=(10, 8))
    plt.barh(commodities, tmiles, color=chosen_color, edgecolor="black", alpha=0.8)

    # Invert y-axis so the largest bar is on top
    plt.gca().invert_yaxis()

    # Labels and styling
    plt.xlabel("Tonne-Miles (2025)", fontsize=22)
    plt.ylabel("Commodity", fontsize=22)
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.tick_params(axis="both", which="minor", labelsize=20)
    # plt.title(f"Tonne-Miles by Commodity for {vessel_class.title()} Vessels", fontsize=16)
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save the figure
    save_path_png = f"{top_dir}/plots/{vessel_class}_commodity_tmiles.png"
    save_path_pdf = f"{top_dir}/plots/{vessel_class}_commodity_tmiles.pdf"
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)

    print(f"Saved tonne-miles bar chart for {vessel_class} to:")
    print(f"{save_path_png}")
    print(f"{save_path_pdf}")


def get_sf_distributions(vessel_class):
    """
    Calculate three distributions of stowage factors carried by a given vessel class,
    weighted by their tonne-miles:
    1. Lower Bound SFs
    2. Upper Bound SFs
    3. Average SFs

    Parameters
    ----------
    vessel_class : str
        Name of the vessel class

    Returns
    -------
    lower_df : pd.DataFrame
        Lower stowage factor distribution (m^3/tonne) and normalized weight.
    upper_df : pd.DataFrame
        Upper stowage factor distribution (m^3/tonne) and normalized weight.
    avg_df : pd.DataFrame
        Average stowage factor distribution (m^3/tonne) and normalized weight.
    """

    # Step 1: Get commodities and their tonne-miles
    commodities = get_commodities_class(vessel_class)
    commodity_tmiles = get_commodity_tms(vessel_class)

    # Step 2: Get commodity SF info
    commodity_info_df = get_commodity_info()

    # Prepare lists for stowage factors and weights
    lower_sf_values = []
    upper_sf_values = []
    avg_sf_values = []
    weights = []

    # Step 3: Iterate through commodities
    for commodity in commodities:
        tmiles = commodity_tmiles.get(commodity, 0)

        # Skip commodities with no tonne-miles
        if tmiles == 0:
            continue

        lower_sf = commodity_info_df.loc[commodity, "Lower Stowage Factor (m^3/tonne)"]
        upper_sf = commodity_info_df.loc[commodity, "Upper Stowage Factor (m^3/tonne)"]

        # Skip if we don't have valid SF data
        if pd.isna(lower_sf) or pd.isna(upper_sf):
            continue

        # Step 4: Append to all three distributions
        lower_sf_values.append(lower_sf)
        upper_sf_values.append(upper_sf)
        avg_sf_values.append((lower_sf + upper_sf) / 2)

        # Append corresponding weight
        weights.append(tmiles)

    # Step 5: Create DataFrames
    lower_df = pd.DataFrame(
        {"Stowage Factor (m^3/tonne)": lower_sf_values, "Weight": weights}
    )

    upper_df = pd.DataFrame(
        {"Stowage Factor (m^3/tonne)": upper_sf_values, "Weight": weights}
    )

    avg_df = pd.DataFrame(
        {"Stowage Factor (m^3/tonne)": avg_sf_values, "Weight": weights}
    )

    # Step 6: Normalize weights for each distribution
    for df in [lower_df, upper_df, avg_df]:
        total_weight = df["Weight"].sum()
        if total_weight > 0:
            df["Weight"] = df["Weight"] / total_weight
        else:
            print(f"No valid tonne-miles found for {vessel_class} commodities.")

    # Step 7: Sort for neatness
    lower_df = lower_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(
        drop=True
    )
    upper_df = upper_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(
        drop=True
    )
    avg_df = avg_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(drop=True)

    sf_data = {"lower": lower_df, "central": avg_df, "upper": upper_df}

    return sf_data


def get_sf_distributions_tanker():
    """
    Calculate stowage factor distribution for tankers, weighted by tonne-miles
    of crude petroleum, fuel oils, and gasoline.

    Returns
    -------
    sf_df : pd.DataFrame
        Stowage factor distribution (m³/tonne) and normalized weight.
    """

    # Step 1: Get commodity tonne-miles for the tanker vessel class
    commodity_tmiles = get_commodity_tms("tanker")

    # Step 2: Extract tonne-miles for target commodities
    crude_tm = commodity_tmiles.get("Crude petroleum", 0)
    fuel_oil_tm = commodity_tmiles.get("Fuel oils", 0)
    gasoline_tm = commodity_tmiles.get("Gasoline", 0)

    # Validate data
    if crude_tm + fuel_oil_tm + gasoline_tm == 0:
        print("No valid tonne-miles found for tanker commodities.")
        return pd.DataFrame()

    # Step 3: Constants for densities (kg/L)
    GASOLINE_DENSITY = 0.749  # kg/L (GREET 2024)
    FUEL_OIL_DENSITY = get_fuel_density("lsfo")  # kg/L

    # Convert densities to stowage factors (m³/tonne)
    gasoline_sf = 1 / GASOLINE_DENSITY
    lsfo_sf = 1 / FUEL_OIL_DENSITY

    # Step 4: Get crude petroleum SF distribution (already normalized within crude)
    central_crude_df, lower_crude_df, upper_crude_df = calculate_crude_petrol_sfs()

    if central_crude_df.empty:
        print("No crude petroleum SF distribution returned.")
        return pd.DataFrame()

    # Scale crude weights by total tonne-miles for crude petroleum
    central_crude_df["Weight"] *= crude_tm
    lower_crude_df["Weight"] *= crude_tm
    upper_crude_df["Weight"] *= crude_tm

    # Step 5: Add Fuel oils and Gasoline as single points
    sf_list = []
    weight_list = []

    # Append fuel oils
    if fuel_oil_tm > 0:
        sf_list.append(lsfo_sf)
        weight_list.append(fuel_oil_tm)

    # Append gasoline
    if gasoline_tm > 0:
        sf_list.append(gasoline_sf)
        weight_list.append(gasoline_tm)

    # Create DataFrame for single point commodities
    single_df = pd.DataFrame(
        {"Stowage Factor (m^3/tonne)": sf_list, "Weight": weight_list}
    )

    # Step 6: Combine crude and other commodities
    central_combined_df = pd.concat([central_crude_df, single_df], ignore_index=True)
    lower_combined_df = pd.concat([lower_crude_df, single_df], ignore_index=True)
    upper_combined_df = pd.concat([upper_crude_df, single_df], ignore_index=True)

    # Step 7: Normalize weights for the entire distribution
    total_weight = central_combined_df["Weight"].sum()
    if total_weight > 0:
        central_combined_df["Weight"] /= total_weight
        lower_combined_df["Weight"] /= total_weight
        upper_combined_df["Weight"] /= total_weight
    else:
        print("No valid weights after combining commodities.")
        return pd.DataFrame()

    # Step 8: Sort for neatness
    central_combined_df = central_combined_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(
        drop=True
    )
    lower_combined_df = lower_combined_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(
        drop=True
    )
    upper_combined_df = upper_combined_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(
        drop=True
    )

    sf_data = {"lower": lower_combined_df, "central": central_combined_df, "upper": upper_combined_df}

    return sf_data


def plot_sf_distributions(vessel_class, bins=5):
    """
    Plots stowage factor distributions for a given vessel_class.

    Parameters
    ----------
    vessel_class : str
        The vessel class to plot distributions for.

    bins : int
        Number of histogram bins (default 5).
    """

    # Select the correct data function based on vessel_class
    if vessel_class.lower() == "tanker":
        sf_data = get_sf_distributions_tanker()  # Single dataframe
    else:
        sf_data = get_sf_distributions(vessel_class)  # Dict with lower/central/upper

    plt.figure(figsize=(10, 6))

    lower_df = sf_data["lower"]
    central_df = sf_data["central"]
    upper_df = sf_data["upper"]

    lower_values = lower_df["Stowage Factor (m^3/tonne)"]
    lower_weights = lower_df["Weight"]

    central_values = central_df["Stowage Factor (m^3/tonne)"]
    central_weights = central_df["Weight"]

    upper_values = upper_df["Stowage Factor (m^3/tonne)"]
    upper_weights = upper_df["Weight"]

    # Lower bound (blue)
    plt.hist(
        lower_values,
        bins=bins,
        weights=lower_weights,
        histtype="stepfilled",
        alpha=0.2,
        color="blue",
    )
    plt.hist(
        lower_values,
        bins=bins,
        weights=lower_weights,
        histtype="step",
        color="blue",
        linewidth=2,
        label="Lower",
    )
    average_lower = (lower_values * lower_weights).sum()
    plt.axvline(
        average_lower,
        ls="--",
        color="darkblue",
        linewidth=2,
        label=f"Average Lower: {average_lower:.2f}",
    )

    # Central (green)
    plt.hist(
        central_values,
        bins=bins,
        weights=central_weights,
        histtype="stepfilled",
        alpha=0.2,
        color="green",
    )
    plt.hist(
        central_values,
        bins=bins,
        weights=central_weights,
        histtype="step",
        color="green",
        linewidth=2,
        label="Central",
    )
    average_central = (central_values * central_weights).sum()
    plt.axvline(
        average_central,
        ls="--",
        color="darkgreen",
        linewidth=2,
        label=f"Average Central: {average_central:.2f}",
    )

    # Upper (red)
    plt.hist(
        upper_values,
        bins=bins,
        weights=upper_weights,
        histtype="stepfilled",
        alpha=0.2,
        color="red",
    )
    plt.hist(
        upper_values,
        bins=bins,
        weights=upper_weights,
        histtype="step",
        color="red",
        linewidth=2,
        label="Upper",
    )
    average_upper = (upper_values * upper_weights).sum()
    plt.axvline(
        average_upper,
        ls="--",
        color="darkred",
        linewidth=2,
        label=f"Average Upper: {average_upper:.2f}",
    )

    plt.legend(fontsize=18)

    # Labels and layout
    plt.xlabel("Stowage Factor (m$^3$/tonne)", fontsize=22)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tick_params(axis="both", which="minor", labelsize=18)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{top_dir}/plots/sf_distribution_{vessel_class}.png", dpi=300)
    plt.savefig(f"{top_dir}/plots/sf_distribution_{vessel_class}.pdf", dpi=300)


def main():
    # Make a filtered version of the FAF5 data for this analysis, if it doesn't already exist
    output_file = f"{top_dir}/data/FAF5.6.1/FAF5.6.1_import_export_by_ship.csv"

    # Check if the filtered CSV already exists
    if not os.path.exists(output_file):
        print(f"{output_file} not found. Creating the filtered CSV...")
        make_filtered_faf5_csv()
    else:
        print(f"{output_file} already exists. Skipping CSV creation.")

    sf_distributions_tanker = get_sf_distributions_tanker()

    plot_sf_distributions("tanker")

    # Plot the distribution of crude petroleum SFs
    plot_weighted_sf_petroleum("central")

    # Plot SF ranges of commodities carried by container and bulk carriers vessels
    plot_commodity_sfs()

    # Plot SF ranges of commodities carried by tankers
    plot_sf_ranges_tanker()

    for vessel_class in ["bulk_carrier", "container", "tanker", "gas_carrier"]:
        if not vessel_class == "gas_carrier":
            plot_commodity_tms(vessel_class)

        if vessel_class == "bulk_carrier" or vessel_class == "container" or vessel_class == "tanker":
            if vessel_class == "bulk_carrier" or vessel_class == "container":
                sf_distributions = get_sf_distributions(vessel_class)
            if vessel_class == "tanker":
                sf_distributions = sf_distributions_tanker
            for dist in ["lower", "central", "upper"]:
                sf_distributions[dist].to_csv(
                    f"{top_dir}/tables/sf_distribution_{vessel_class}_{dist}.csv"
                )

        elif vessel_class == "gas_carrier":
            sf_distribution = get_gas_carrier_sfs()
            sf_distribution.to_csv(f"{top_dir}/tables/sf_distribution_gas_carrier.csv")

        if vessel_class != "gas_carrier":
            plot_sf_distributions(vessel_class)

main()

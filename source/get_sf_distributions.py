"""
Date: 250318
Purpose: Estimates stowage factor distributions for different vessel classes
"""

from common_tools import get_fuel_density, get_top_dir
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

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
        Density of the crude oil
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
    file_path = f"{top_dir}/data/Crude_Oil_and_Lease_Condensate_Production_by_API_Gravity.csv"
    
    # Skip header rows and set Month as index
    crude_petrol_df = pd.read_csv(file_path, skiprows=6, index_col="Month")
    
    # Filter for 2023 months
    crude_petrol_df_2023 = crude_petrol_df[crude_petrol_df.index.str.contains("2023")]
    
    # Compute mean production for each API category
    avg_production = crude_petrol_df_2023.mean()

    # Prepare lists for stowage factors and weights
    sf_list = []
    weight_list = []

    # Iterate through each column
    for column in avg_production.index:
        #print(f"Processing column: {column}")
        
        # Extract API gravity values
        match = re.search(r'for ([\d\.]+)(?: to ([\d\.]+)| or (Higher|Lower)) Degrees API', column)
        
        if match:
            lower_api = float(match.group(1))
            upper_api = match.group(2)  # This is None if "or Higher" / "or Lower"

            # Check if column mentions "Higher" or "Lower"
            if "or Higher" in str(column):
                # Use the inner value of the range (lower bound)
                central_api = lower_api
                #print(f"  -> 'or Higher' detected, using lower_api = {central_api}")
            elif "or Lower" in column:
                # Use the inner value of the range (upper bound)
                central_api = lower_api
                #print(f"  -> 'or Lower' detected, using upper_api = {central_api}")
            elif upper_api:
                # Midpoint for ranged categories
                central_api = (lower_api + float(upper_api)) / 2
                #print(f"  -> Range detected, using midpoint = {central_api}")
            else:
                # Shouldn't happen, but fallback
                central_api = lower_api
                #print(f"  -> No upper bound, using lower_api = {central_api}")

            # Convert API to density (kg/L)
            density = api_to_density(central_api)

            # Stowage factor (m³/tonne or L/kg)
            stowage_factor = 1 / density

            # Average production as weight
            weight = avg_production[column]

            # Skip if weight is zero or NaN
            if pd.isna(weight) or weight == 0:
                continue

            # Append results
            sf_list.append(stowage_factor)
            weight_list.append(weight)

    # Create the result DataFrame
    result_df = pd.DataFrame({
        "Stowage Factor (m^3/tonne)": sf_list,
        "Weight": weight_list
    })

    # Normalize weights to sum to 1
    result_df["Weight"] = result_df["Weight"] / result_df["Weight"].sum()

    # Sort by increasing stowage factor
    result_df = result_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(drop=True)

    return result_df
    
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
    metaPath = (
        f"{top_dir}/data/FAF5.6.1/FAF5_metadata.xlsx"
    )
    meta = pd.ExcelFile(metaPath)

    # Only include truck rail and water
    modes = pd.read_excel(meta, "Mode", index_col="Description")
    trade_types = pd.read_excel(meta, "Trade Type", index_col="Description")
    comms = pd.read_excel(meta, "Commodity (SCTG2)", index_col="Description")
    return modes, comms, trade_types
    
def plot_weighted_sf_petroleum():
    """
    Plots a bar chart of the stowage factors weighted by their normalized production shares.
    """

    # Sort the dataframe by stowage factor (optional if already sorted)
    df = calculate_crude_petrol_sfs()
    df_sorted = df.sort_values(by="Stowage Factor (m^3/tonne)")

    sf_values = df_sorted["Stowage Factor (m^3/tonne)"]
    weights = df_sorted["Weight"]

    plt.figure(figsize=(10, 6))
    plt.bar(sf_values, weights, width=0.015, align='center', edgecolor='black', alpha=0.8)

    # Labeling and styling
    plt.xlabel("Stowage Factor (m$^3$/tonne)")
    plt.ylabel("Normalized Weight (Sum = 1)")
    #plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

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
            "Upper Stowage Factor (m^3/tonne)"
        ]
    )
    return commodity_info_df
    
def plot_commodity_sfs():
    # Read the commodity info data
    commodity_info_df = get_commodity_info()
    
    # Filter bulk_carrier and container commodities
    bulk_df = commodity_info_df[commodity_info_df["Vessel Class"] == "bulk_carrier"].dropna()
    container_df = commodity_info_df[commodity_info_df["Vessel Class"] == "container"].dropna()

    # Plot bulk_carrier commodities
    plot_sf_ranges(
        bulk_df,
        title="Stowage Factor Ranges for Bulk Carrier Commodities",
        color='steelblue',
        save_str="bulk"
    )

    # Plot container commodities
    plot_sf_ranges(
        container_df,
        title="Stowage Factor Ranges for Container Commodities",
        color='darkorange',
        save_str="container"
    )


def plot_sf_ranges(df, title, color, save_str):
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
    """
    # Ensure sorted for better display
    df_sorted = df.sort_values(by="Lower Stowage Factor (m^3/tonne)")

    # Extract necessary values
    commodities = df_sorted.index.tolist()
    lower_sf = df_sorted["Lower Stowage Factor (m^3/tonne)"]
    upper_sf = df_sorted["Upper Stowage Factor (m^3/tonne)"]

    # Calculate ranges
    sf_range = upper_sf - lower_sf

    # Plot horizontal bars
    plt.figure(figsize=(10, 8))
    plt.barh(commodities, sf_range, left=lower_sf, height=0.6, color=color, edgecolor='black', alpha=0.7)

    # Labels and styling
    plt.xlabel("Stowage Factor (m³/tonne)", fontsize=20)
    plt.ylabel("Commodity", fontsize=20)
    plt.title(title)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(f"{top_dir}/plots/commodity_sf_range_{save_str}.png", dpi=300)
    plt.savefig(f"{top_dir}/plots/commodity_sf_range_{save_str}.pdf")
    
def plot_sf_ranges_tanker(title="Stowage Factors for Tanker Commodities", color='navy', save_str="tanker"):
    """
    Plots stowage factors for tanker commodities: crude petroleum, gasoline, and fuel oils.
    
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

    # Calculate stowage factors (m³/tonne = 1000 kg / (density in kg/L))
    GASOLINE_SF = 1 / GASOLINE_DENSITY  # m³/tonne
    FUEL_OIL_SF = 1 / FUEL_OIL_DENSITY  # m³/tonne

    # Crude petroleum SF distribution from existing function
    crude_petrol_df = calculate_crude_petrol_sfs()

    # Prepare the data to plot
    commodities = []
    lower_sf = []
    upper_sf = []

    # Crude petroleum has a distribution -> plot the min and max of the distribution
    crude_min_sf = crude_petrol_df["Stowage Factor (m^3/tonne)"].min()
    crude_max_sf = crude_petrol_df["Stowage Factor (m^3/tonne)"].max()

    commodities.append("Crude Petroleum")
    lower_sf.append(crude_min_sf)
    upper_sf.append(crude_max_sf)

    # Gasoline and Fuel oils are treated as point estimates, so lower and upper are equal
    commodities.append("Gasoline")
    lower_sf.append(GASOLINE_SF)
    upper_sf.append(GASOLINE_SF)

    commodities.append("Fuel Oils")
    lower_sf.append(FUEL_OIL_SF)
    upper_sf.append(FUEL_OIL_SF)

    # Create a DataFrame for consistency (optional but clean)
    tanker_df = pd.DataFrame({
        "Commodity": commodities,
        "Lower Stowage Factor (m^3/tonne)": lower_sf,
        "Upper Stowage Factor (m^3/tonne)": upper_sf
    }).set_index("Commodity")

    # Sort by lower SF if desired
    tanker_df_sorted = tanker_df.sort_values(by="Lower Stowage Factor (m^3/tonne)")

    # Calculate ranges
    sf_range = tanker_df_sorted["Upper Stowage Factor (m^3/tonne)"] - tanker_df_sorted["Lower Stowage Factor (m^3/tonne)"]

    # Plot horizontal bars (or markers if range is zero)
    plt.figure(figsize=(10, 6))

    for idx, row in tanker_df_sorted.iterrows():
        if row["Lower Stowage Factor (m^3/tonne)"] == row["Upper Stowage Factor (m^3/tonne)"]:
            # Plot a single point (marker) for gasoline and fuel oils
            plt.barh(
                idx,
                width=0.01,  # Small width to represent a point
                left=row["Lower Stowage Factor (m^3/tonne)"],
                height=0.6,
                color=color,
                edgecolor='black',
                alpha=0.8
            )
        else:
            # Plot a horizontal bar for crude petroleum range
            plt.barh(
                idx,
                row["Upper Stowage Factor (m^3/tonne)"] - row["Lower Stowage Factor (m^3/tonne)"],
                left=row["Lower Stowage Factor (m^3/tonne)"],
                height=0.6,
                color=color,
                edgecolor='black',
                alpha=0.7
            )

    # Labels and styling
    plt.xlabel("Stowage Factor (m$^3$/tonne)", fontsize=14)
    plt.ylabel("Commodity", fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.xlim(left=0)

    plt.tight_layout()

    # Save plots
    plt.savefig(f"{top_dir}/plots/commodity_sf_range_{save_str}.png", dpi=300)
    plt.savefig(f"{top_dir}/plots/commodity_sf_range_{save_str}.pdf")

    print(f"Saved tanker SF plot to:")
    print(f"{top_dir}/plots/commodity_sf_range_{save_str}.png")
    print(f"{top_dir}/plots/commodity_sf_range_{save_str}.pdf")

    
def make_filtered_faf5_csv():
    """
    Reads in the FAF5 data and saves the rows corresponding to imports and exports carried by ship to a csv file
    """
    
    faf5_data_df = pd.read_csv(f"{top_dir}/data/FAF5.6.1/FAF5.6.1.csv", usecols=["sctg2", "fr_inmode", "fr_outmode", "trade_type", "tmiles_2025"])
    
    modes, comms, trade_types = readMeta()
    import_numeric_label = trade_types.loc["Import flows (freight shipments moved from foreign origins to US domestic destinations)", "Numeric Label"]
    export_numeric_label = trade_types.loc["Export flows (freight shipments moved from US domestic origins to foreign destinations)", "Numeric Label"]
    water_transport_numeric_label = modes.loc["Water", "Numeric Label"]
    
    # Set up selections
    cImport = (faf5_data_df["trade_type"]==import_numeric_label)
    cShipImport = (faf5_data_df["fr_inmode"]==water_transport_numeric_label)
    cExport = (faf5_data_df["trade_type"]==export_numeric_label)
    cShipExport = (faf5_data_df["fr_outmode"]==water_transport_numeric_label)
        
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
    commodities_class = commodity_info_df[commodity_info_df["Vessel Class"]==vessel_class]
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

    faf5_data = pd.read_csv(f"{top_dir}/data/FAF5.6.1/FAF5.6.1_import_export_by_ship.csv")
    commodity_tmiles = {}
    modes, comms, trade_types = readMeta()
    for commodity in commodities:
        commodity_numeric_label = comms.loc[commodity, "Numeric Label"]
        commodity_tmiles[commodity] = faf5_data["tmiles_2025"][faf5_data["sctg2"]==commodity_numeric_label].sum()
    
    return commodity_tmiles
    
def plot_commodity_tms(vessel_class):
    """
    Plots the tonne-miles traveled by each commodity carried by the given vessel class
    as a horizontal bar chart.
    
    Parameters
    ----------
    vessel_class : str
        Name of the vessel class
    """

    # Get the tonne-miles dictionary
    commodity_tmiles = get_commodity_tms(vessel_class)
    
    if not commodity_tmiles:
        print(f"No data available for vessel class: {vessel_class}")
        return

    # Sort commodities by tonne-miles (optional for better visualization)
    sorted_commodities = sorted(commodity_tmiles.items(), key=lambda x: x[1], reverse=True)

    # Unzip into two lists
    commodities, tmiles = zip(*sorted_commodities)

    # Create the horizontal bar plot
    plt.figure(figsize=(10, 8))
    plt.barh(commodities, tmiles, color='teal', edgecolor='black', alpha=0.8)

    # Invert y-axis so the largest bar is on top
    plt.gca().invert_yaxis()

    # Labels and styling
    plt.xlabel("Tonne-Miles (2025)", fontsize=14)
    plt.ylabel("Commodity", fontsize=14)
    plt.title(f"Tonne-Miles by Commodity for {vessel_class.title()} Vessels", fontsize=16)
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # Save the figure
    save_path_png = f"{top_dir}/plots/{vessel_class}_commodity_tmiles.png"
    save_path_pdf = f"{top_dir}/plots/{vessel_class}_commodity_tmiles.pdf"
    plt.savefig(save_path_png)
    plt.savefig(save_path_pdf)
    
    print(f"Saved tonne-miles bar chart for {vessel_class} to:")
    print(f"{save_path_png}")
    print(f"{save_path_pdf}")

    #plt.show()
    
    
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
    lower_df = pd.DataFrame({
        "Stowage Factor (m^3/tonne)": lower_sf_values,
        "Weight": weights
    })

    upper_df = pd.DataFrame({
        "Stowage Factor (m^3/tonne)": upper_sf_values,
        "Weight": weights
    })

    avg_df = pd.DataFrame({
        "Stowage Factor (m^3/tonne)": avg_sf_values,
        "Weight": weights
    })

    # Step 6: Normalize weights for each distribution
    for df in [lower_df, upper_df, avg_df]:
        total_weight = df["Weight"].sum()
        if total_weight > 0:
            df["Weight"] = df["Weight"] / total_weight
        else:
            print(f"No valid tonne-miles found for {vessel_class} commodities.")

    # Step 7: Sort for neatness
    lower_df = lower_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(drop=True)
    upper_df = upper_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(drop=True)
    avg_df = avg_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(drop=True)
    
    sf_data = {
        "lower": lower_df,
        "central": avg_df,
        "upper": upper_df
    }

    return sf_data
    
def get_sf_distribution_tanker():
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
    crude_df = calculate_crude_petrol_sfs()

    if crude_df.empty:
        print("No crude petroleum SF distribution returned.")
        return pd.DataFrame()

    # Scale crude weights by total tonne-miles for crude petroleum
    crude_df["Weight"] *= crude_tm

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
    single_df = pd.DataFrame({
        "Stowage Factor (m^3/tonne)": sf_list,
        "Weight": weight_list
    })

    # Step 6: Combine crude and other commodities
    combined_df = pd.concat([crude_df, single_df], ignore_index=True)

    # Step 7: Normalize weights for the entire distribution
    total_weight = combined_df["Weight"].sum()
    if total_weight > 0:
        combined_df["Weight"] /= total_weight
    else:
        print("No valid weights after combining commodities.")
        return pd.DataFrame()

    # Step 8: Sort for neatness
    combined_df = combined_df.sort_values(by="Stowage Factor (m^3/tonne)").reset_index(drop=True)

    return combined_df

def plot_sf_distributions(vessel_class, bins=5):
    """
    Plots stowage factor distributions for a given vessel_class.
    
    For "tanker", plots a single stowage factor distribution using get_sf_distributions_tanker().
    For all other vessel classes, plots lower, central, and upper distributions using get_sf_distributions().
    
    Parameters
    ----------
    vessel_class : str
        The vessel class to plot distributions for.
    
    bins : int
        Number of histogram bins (default 5).
    """
    
    # Select the correct data function based on vessel_class
    if vessel_class.lower() == "tanker":
        sf_data = get_sf_distribution_tanker()  # Single dataframe
    else:
        sf_data = get_sf_distributions(vessel_class)  # Dict with lower/central/upper
    
    plt.figure(figsize=(10, 6))
    
    # Tanker case: single DataFrame returned
    if isinstance(sf_data, pd.DataFrame):
        sf_df = sf_data
        
        # Extract values and weights
        sf_values = sf_df["Stowage Factor (m^3/tonne)"]
        sf_weights = sf_df["Weight"]

        # Plot histogram (single distribution)
        plt.hist(
            sf_values,
            bins=bins,
            weights=sf_weights,
            histtype='stepfilled',
            alpha=0.2,
            color='blue'
        )
        plt.hist(
            sf_values,
            bins=bins,
            weights=sf_weights,
            histtype='step',
            color='blue',
            linewidth=2,
            #label=vessel_class.capitalize()
        )

    # Non-tanker case: lower, central, upper distributions
    elif isinstance(sf_data, dict):
        lower_df = sf_data['lower']
        central_df = sf_data['central']
        upper_df = sf_data['upper']

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
            histtype='stepfilled',
            alpha=0.2,
            color='blue'
        )
        plt.hist(
            lower_values,
            bins=bins,
            weights=lower_weights,
            histtype='step',
            color='blue',
            linewidth=2,
            label='Lower'
        )

        # Central (green)
        plt.hist(
            central_values,
            bins=bins,
            weights=central_weights,
            histtype='stepfilled',
            alpha=0.2,
            color='green'
        )
        plt.hist(
            central_values,
            bins=bins,
            weights=central_weights,
            histtype='step',
            color='green',
            linewidth=2,
            label='Central'
        )

        # Upper (red)
        plt.hist(
            upper_values,
            bins=bins,
            weights=upper_weights,
            histtype='stepfilled',
            alpha=0.2,
            color='red'
        )
        plt.hist(
            upper_values,
            bins=bins,
            weights=upper_weights,
            histtype='step',
            color='red',
            linewidth=2,
            label='Upper'
        )
        plt.legend(fontsize=18)

    else:
        print(f"Unrecognized data structure returned from get_sf_distributions() for vessel_class: {vessel_class}")
        return

    # Labels and layout
    plt.xlabel("Stowage Factor (m$^3$/tonne)", fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    plt.grid(True, linestyle='--', alpha=0.6)
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

    
    sf_distribution_tanker = get_sf_distribution_tanker()
    
    plot_sf_distributions("tanker")
    
    # Plot the distribution of crude petroleum SFs
    plot_weighted_sf_petroleum()
    
    # Plot SF ranges of commodities carried by container and bulk carriers vessels
    plot_commodity_sfs()
    
    # Plot SF ranges of commodities carried by tankers
    plot_sf_ranges_tanker()
    
    for vessel_class in ["bulk_carrier", "container", "tanker", "gas_carrier"]:
        if not vessel_class == "gas_carrier":
            plot_commodity_tms(vessel_class)
        
        if vessel_class == "bulk_carrier" or vessel_class == "container":
            sf_distributions = get_sf_distributions(vessel_class)
            for dist in ["lower", "central", "upper"]:
                sf_distributions[dist].to_csv(f"{top_dir}/tables/sf_distribution_{vessel_class}_{dist}.csv")
        
        elif vessel_class == "tanker":
            sf_distribution = get_sf_distribution_tanker()
            sf_distribution.to_csv(f"{top_dir}/tables/sf_distribution_tanker.csv")
            
        elif vessel_class == "gas_carrier":
            sf_distribution = get_gas_carrier_sfs()
            sf_distribution.to_csv(f"{top_dir}/tables/sf_distribution_gas_carrier.csv")
        
        if vessel_class != "gas_carrier":
            plot_sf_distributions(vessel_class)
    
main()

"""
Date: Aug 6, 2024
Author: danikam
Purpose: Makes validation plots for csv files produced by make_output_csvs.py
"""

from common_tools import get_top_dir, generate_blue_shades, get_pathway_type, get_pathway_type_color, get_pathway_type_label, get_pathway_label, read_pathway_labels, read_fuel_labels, get_fuel_label, create_directory_if_not_exists
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import re
import os
import geopandas as gpd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# GE - 3/23/25
from matplotlib.colors import LogNorm
from fuzzywuzzy import process
import pycountry
import pycountry_convert as pc
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from parse import parse

matplotlib.rc("xtick", labelsize=18)
matplotlib.rc("ytick", labelsize=18)

RESULTS_DIR = "processed_results"

vessel_types = ["bulk_carrier_ice", "container_ice", "tanker_ice", "gas_carrier_ice"]

vessel_sizes = {
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

region_name_mapping = {
    "United States": "United States of America",
}

region_label_mapping = {
    "SaudiArabia": "Saudi Arabia",
    "WestAustralia": "West Australia",
}

result_components = {
    "TotalCost": ["TotalCAPEX", "TotalFuelOPEX", "TotalExcludingFuelOPEX"],
    "TotalEquivalentWTW": ["TotalEquivalentTTW", "TotalEquivalentWTT"],
    "CostTimesEmissions": [],
    "AverageCostEmissionsRatio": ["HalfCostRatio", "HalfWTWRatio"],
    "CAC": [],
}

# Global string representing the absolute path to the top level of the repo
top_dir = get_top_dir()

# Global dataframe with labels for each fuel production pathway
fuel_labels_df = read_fuel_labels()

# Global dataframe with labels for each fuel production pathway
pathway_labels_df = read_pathway_labels()

def read_quantity_info(top_dir):
    """
    Reads in the long name, units and description for each quantity from an info file.

    Parameters
    ----------
    top_dir: str
        Absolute path to the top level of the repo

    Returns
    -------
    info_df : pandas.DataFrame
        Dataframe containing info about the given quantity.
    """
    info_df = pd.read_csv(f"{top_dir}/info_files/quantity_info.csv").set_index(
        "Quantity"
    )
    return info_df


# Global dataframe containing info (long name, units and description) about each quantity
quantity_info_df = read_quantity_info(top_dir)


def make_region_labels(region_names_camel_case):
    """
    Function to construct region labels with spaces in cases where CamelCase is used to separate different parts of region names.
    Eg. ["SaudiArabia"] --> ["Saudi Arabia"]

    Parameters
    ----------
    region_names_camel_case: list of str
        List of region names in CamelCase format

    Returns
    -------
    region_names_with_spaces : list of str
        List of region names with spaces
    """
    # Insert a space before each uppercase letter (except the first one)
    region_names_with_spaces = [
        re.sub(r"(?<!^)(?<![A-Z])(?=[A-Z])", " ", name)
        for name in region_names_camel_case
    ]
    return region_names_with_spaces


def get_region_label(region):
    """
    Gets the name for the region to be used in plot labels, if the region is included in the region_label_mapping dict.

    Parameters
    ----------
    region : str
        Name of the region as specified in the processed csv file

    Returns
    -------
    region_label : str
        Name of the region to be used for plotting
    """
    if region in region_label_mapping:
        region_label = region_label_mapping[region]
    else:
        region_label = region

    return region_label


def get_filename_info(
    filepath,
    identifier,
    pattern="{fuel}-{pathway_type}-{pathway}-{quantity}-{modifier}.csv",
):
    """
    Parses the filename for a processed csv file to collect relevant info about the file contents

    Parameters
    ----------
    identifier : str
        Identifier to parse the value of from the filename

    filepath : str
        Absolute or relative path to the csv file

    Returns
    -------
    identifier_value : str
        Value of the identifier parsed from the filename
    """

    # Check that the identifier is included in the provided pattern
    if identifier not in pattern:
        raise Exception(
            f"Error: identifier {identifier} not found in provided pattern {pattern}"
        )

    filename = filepath.split("/")[-1]

    result = parse(pattern, filename)

    if result is None:
        raise Exception(
            f"Error: Filename {filename} does not match provided pattern {pattern}"
        )

    identifier_value = result.named[identifier]

    return identifier_value


def find_files_starting_with_substring(directory, substring=""):
    """
    Finds all files within a specified directory (and its subdirectories) that start with a given substring in their filenames.

    Parameters:
    ----------
    directory : str
        The path to the directory to search
    substring : str
        The substring to search for in the filenames.

    Returns:
    -------
    matching_files: list of str
        A list of full file paths for files that contain the given substring in their names.
    """

    matching_files = []
    # Loop through all directories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file contains the given substring
            if file.startswith(substring):
                # Add the full file path to the list
                matching_files.append(os.path.join(root, file))

    return matching_files


def find_unique_identifiers(
    directory,
    identifier,
    substring="",
    pattern="{fuel}-{pathway}-{quantity}-{modifier}.csv",
):
    """
    Finds all unique values of a given identifier in a pattern within filenames in a directory containing the given substring

    Parameters:
    ----------
    directory : str
        The path to the directory to search
    pattern : str
        Pattern containing the given identifier
    identifier : str
        Identifier to find unique instances of
    substring : str
        The substring to search for in the filenames.

    Returns:
    -------
    unique identifiers: list of str
        A list of unique values of the given identifier in strings containing the given substring
    """

    # Check that the identifier is included in the provided pattern
    if identifier not in pattern:
        raise Exception(
            f"Error: identifier {identifier} not found in provided pattern {pattern}"
        )

    filepaths_matching_substring = find_files_starting_with_substring(
        directory, substring
    )
    unique_identifier_values = []
    for filepath in filepaths_matching_substring:
        filename = filepath.split("/")[-1]
        if filename.startswith("."):
            continue
        identifier_value = get_filename_info(filepath, identifier, pattern)
        if identifier_value not in unique_identifier_values:
            unique_identifier_values.append(identifier_value)

    return unique_identifier_values


def add_west_australia(world):
    """
    Adds a custom polygon representing West Australia to the world GeoDataFrame.

    This function manually defines a polygon that approximates the region of West Australia
    and adds it to the given world GeoDataFrame. The polygon is created using the shapely library
    and is added as a new row in the GeoDataFrame with the name 'West Australia'.

    Parameters
    ----------
    world : geopandas.GeoDataFrame
        The GeoDataFrame containing the world map with region boundaries.

    Returns
    -------
    geopandas.GeoDataFrame
        The updated GeoDataFrame that includes the custom region 'West Australia'.
    """

    # Check if 'West Australia' is already in the 'NAME' column
    if "West Australia" in world["NAME"].values:
        return world

    # Load detailed shapefile for Australia that includes state/territory boundaries
    url = "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/georef-australia-state@public/exports/geojson"
    australia_states = gpd.read_file(url)

    # Filter the GeoDataFrame to only include Western Australia
    west_australia = australia_states[
        australia_states["ste_iso3166_code"] == "WA"
    ].copy()

    # Rename 'WA' to 'West Australia' in the 'NAME' column using .loc[]
    west_australia.loc[west_australia["ste_iso3166_code"] == "WA", "NAME"] = (
        "West Australia"
    )

    # Concatenate the West Australia GeoDataFrame with the world GeoDataFrame
    world_with_west_australia = pd.concat([world, west_australia], ignore_index=True)

    return world_with_west_australia

def get_custom_tab20_without_blue():
    # Define the tab20 colormap
    tab20 = plt.get_cmap("tab20")

    # Manually extract colors from tab20, explicitly excluding the blue and cyan shades
    custom_colors = [
        tab20(2),  # Orange
        tab20(4),  # Green
        tab20(6),  # Red
        tab20(8),  # Purple
        tab20(10),  # Brown
        tab20(12),  # Pink
        tab20(14),  # Gray
        tab20(16),  # Olive
        tab20(3),  # Light orange
        tab20(5),  # Light green
        tab20(7),  # Light red
        tab20(9),  # Light purple
        tab20(11),  # Light brown
        tab20(13),  # Light pink
        tab20(15),  # Light gray
        tab20(17),  # Light olive
    ]

    return mcolors.ListedColormap(custom_colors)

def assign_colors_to_strings(strings):
    """
    Assigns a color from a quantized gradient (green to yellow to red) to each string in the input list.

    Parameters
    ----------
    strings : list of str
        List of strings to assign colors to

    Returns
    ----------
    color_dict : Dictionary of tuples
        Dictionary mapping each string to its assigned color (as an RGB tuple).
    """
    # Number of strings (and thus, colors)
    n_colors = len(strings)

    # Generate a wide color range using a colormap from matplotlib
    cmap = get_custom_tab20_without_blue()
    gradient = [cmap(i) for i in range(n_colors)]

    # Assign each string a color
    color_dict = {string: gradient[i][:3] for i, string in enumerate(strings)}

    return color_dict


def get_quantity_label(quantity, quantity_info_df=quantity_info_df):
    """
    Returns the label for the provided quantity for use in plots.

    Parameters
    ----------
    quantity : str
        Quantity to plot.

    quantity_info_df : pandas DataFrame
        Dataframe containing info about each quantity.

    Returns
    -------
    label : str
        Label for the quantity to use for plotting
    """

    return quantity_info_df.loc[quantity, "Long Name"]


def get_units(quantity, modifier, quantity_info_df=quantity_info_df):
    """
    Collects the units for the given quantity and modifier

    Parameters
    ----------
    None

    Returns
    -------
    units : str
        Units for the given quantity and modifier
    """

    base_units = quantity_info_df.loc[quantity, "Units"]
    if pd.isna(base_units):
        return None
    
    else:
        # Modify the denominator if needed based on the modifier
        modifier_denom_dict = {
            "vessel": "year",
            "fleet": "year",
            "per_mile": "nm",
            "per_tonne_mile": "tonne-nm",
            "per_tonne_mile_orig": "tonne-nm",
            "per_cbm_mile": "$m^3-nm$",
            "per_cbm_mile_orig": "$m^3-nm$",
        }

        denom_units = modifier_denom_dict[modifier]

        if "/" not in base_units:
            units = f"{base_units} / {denom_units}"
        else:
            units = base_units

        return units
    

class ProcessedQuantity:
    """
    A class to contain results and functions for NavigaTE results for a given quantity and fuel pathway.
    Each processed pathway result is read from a csv file.

    Attributes
    ----------
    quantity : str
        Quantity evaluated by NavigaTE or derived from its outputs (eg. TotalCAPEX)

    modifier : str
        Modifier to the quantity. Can be one of:
            * vessel: Value of the quantity per vessel
            * fleet: Value of the quantity aggregated over all vessels
            * per_mile: Value of the quantity per nautical mile (nm)
            * per_tonn_mile: Value of the quantity per cargo tonne-nm

    fuel : str
        Fuel produced by the given production pathway (eg. ammonia, hydrogen, etc.)

    pathway_color : str
        Color associated with the given production pathway

    pathway : str
        Name of the production pathway, as it's saved in the name of the input csv file

    result_df : pandas.DataFrame
        Dataframe containing the processed pathway result in the csv file

    label : str
        Label representing the long name of the given quantity

    units : str
        Units of the quantity, accounting for the modifier

    Methods
    -------
    make_file_path(self):
        Constructs the full absolute path to the csv file containing the processed results

    read_result(file_path):
        Reads in a processed csv file and saves it to a pandas dataframe.

    make_hist_by_region(self, vessel_type):
        Plots a stacked histogram of either vessel sizes (if a vessel type is provided as input) or vessel types (if "all" is provided).

    make_all_hists_by_region(self):
        Plots all stacked histograms for the available vessel types
    """

    def __init__(
        self, quantity, modifier, fuel, pathway, results_dir=RESULTS_DIR
    ):
        self.quantity = quantity
        self.modifier = modifier
        self.fuel = fuel
        self.fuel_label = get_fuel_label(self.fuel)
        self.pathway = pathway
        self.pathway_label = get_pathway_label(self.pathway)
        self.pathway_type = get_pathway_type(pathway)
        self.pathway_color = get_pathway_type_color(self.pathway_type)
        self.pathway_type_label = get_pathway_type_label(self.pathway_type)
        self.results_dir = results_dir
        self.result_df = self.read_result()
        self.label = get_quantity_label(self.quantity)
        self.units = get_units(self.quantity, self.modifier)

    def make_file_path(self):
        """
        Constructs the full absolute path to the csv file containing the processed results

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return f"{top_dir}/{self.results_dir}/{self.fuel}-{self.pathway}-{self.quantity}-{self.modifier}.csv"

    def read_result(self):
        """
        Reads in a processed csv file and saves it to a pandas dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        data_df : pandas.DataFrame
            Pandas dataframe containing the data in the processed csv file
        """
        file_path = self.make_file_path()
        return pd.read_csv(file_path, index_col=0)

    def read_custom_quantity_df(self, custom_quantity, modifier="vessel"):
        """
        Reads in a the processed csv file for a custom quantity and saves it to a pandas dataframe.

        Parameters
        ----------
        custom_quantity : str
            Custom quantity to read in

        Returns
        -------
        data_df : pandas.DataFrame
            Pandas dataframe containing the data in the processed csv file
        """
        custom_file_path = f"{top_dir}/{self.results_dir}/{self.fuel}-{self.pathway}-{custom_quantity}-{modifier}.csv"
        return pd.read_csv(custom_file_path, index_col=0)
        
    # GE - 3/19/2025 - read in data from excel sheet
    def read_resource_excel(self, file_path, resource_name):
        """
        Reads the first four columns of data from the sheet with given name in the given Excel file.

        Parameters:
        -----------
        filepath : str
            Path to the Excel file.
            
        resource_name : str
            Name of the sheet to read (e.g., 'Water', 'Carbon Dioxide', 'Natural Gas', 'Electricity').
        

        Returns:
        --------
        DataFrame containing only the first four columns of the excel sheet.
        """
        return pd.read_excel(file_path, sheet_name = resource_name, usecols=["Country", "Average", "Standard Deviation", "Ratio of SD:Average"], index_col="Country")


    def make_hist_by_region(self, vessel_type="all"):
        """
        Plots a stacked histogram of either vessel sizes (if a vessel type is provided as input) or vessel types (if "all" is provided).

        Parameters
        ----------
        vessel_type : str
            Vessel type, can currently be one of:
                * bulk_carrier_ice: bulk carrier vessel (internal combustion engine)
                * container_ice: container vessl (internal combustion engine)
                * tanker_ice: tanker vessel (internal combustion engine)
                * gas_carrier_ice: gas carrier vessel (internal combustion engine)

        Returns
        -------
        None
        """

        # Access the results dataframe for the quantity to use for coloring based on sign
        fig, ax = plt.subplots(figsize=(16, 11))

        # Separate result_df into rows with vs. without '_' (where '_' indicates it's one of several individual estimates for a given region)
        result_df_region_av = self.result_df[~self.result_df.index.str.contains("_")]
        result_df_region_individual = self.result_df[
            self.result_df.index.str.contains("_")
        ]

        # Plot each region with vessel types stacked
        stack_vessel_types = False
        stack_vessel_sizes = False
        if vessel_type == "all":
            legend_title = "Vessel Type"

            # Only stack vessel types if the quantity is expressed for the full fleet with no normalization
            if self.modifier == "fleet":
                stack_by = [
                    f"{vessel_type}" for vessel_type in vessel_types
                ]

                result_df_region_av[stack_by].plot(kind="barh", stacked=True, ax=ax)
                stack_vessel_types = True
            else:
                bar_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

                result_df_region_av[f"fleet"].plot(
                    kind="barh", stacked=False, ax=ax, color=bar_color
                )

            # Add individual region estimates as unfilled circles
            for idx, row in result_df_region_individual.iterrows():
                region = idx.split("_")[0]
                if region in result_df_region_av.index:
                    # Use the same x-location as the corresponding bar
                    x = result_df_region_av.index.get_loc(region)

                    # Plot individual estimates as unfilled circles
                    ax.scatter(
                        [row[f"fleet"]],
                        [x],
                        marker="D",
                        color="black",
                        s=100,
                    )

            # Update legend labels if plotting stacked vessels
            if stack_vessel_types:
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [
                    vessel_type_title[label]
                    for label in labels
                ]

            # Construct the filename to save to
            filename_save = f"{self.fuel}-{self.pathway_type}-{self.pathway}-{self.quantity}-{self.modifier}-hist_by_region_allvessels"

        # Plot each region with vessel sizes stacked
        elif vessel_type in vessel_types:
            legend_title = "Vessel Size"
            vessel_type_label = vessel_type_title[vessel_type]
            ax.text(
                1.05,
                0.55,
                f"Vessel Type: {vessel_type_label}",
                transform=ax.transAxes,
                fontsize=20,
                va="top",
                ha="left",
            )
            if self.modifier == "fleet":
                stack_by = [
                    f"{vessel_size}"
                    for vessel_size in vessel_sizes[vessel_type]
                ]

                result_df_region_av[stack_by].plot(kind="barh", stacked=True, ax=ax)
                stack_vessel_sizes = True

            else:
                bar_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
                result_df_region_av[f"{vessel_type}"].plot(
                    kind="barh", stacked=False, ax=ax, color=bar_color
                )

            # Add individual region estimates as unfilled circles
            for idx, row in result_df_region_individual.iterrows():
                region = idx.split("_")[0]
                if region in result_df_region_av.index:
                    # Use the same x-location as the corresponding bar
                    x = result_df_region_av.index.get_loc(region)

                    # Plot individual estimates as unfilled circles
                    ax.scatter(
                        [row[f"{vessel_type}"]],
                        [x],
                        marker="D",
                        color="black",
                        s=100,
                    )

            # Update legend labels if plotting stacked vessel sizes
            if stack_vessel_sizes:
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [
                    vessel_size_title[label]
                    for label in labels
                ]

            # Construct the filename to save to
            filename_save = f"{self.fuel}-{self.pathway_type}-{self.pathway}-{self.quantity}-{self.modifier}-hist_by_region_{vessel_type}"

        else:
            print(
                f"Vessel type should be one of: {vessel_types}. Returning from hist_by_region without plotting."
            )
            return

        # Add text to indicate the details of what's being plotted
        ax.text(
            1.05,
            0.5,
            f"Fuel: {self.fuel_label}",
            transform=ax.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )
        ax.text(
            1.05,
            0.45,
            f"Fuel Type: {self.pathway_type_label}",
            transform=ax.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )
        ax.text(
            1.05,
            0.4,
            f"Pathway: {self.pathway_label}",
            transform=ax.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )

        # Plot styling common to both cases
        if self.units is None:
            ax.set_xlabel(f"{self.label}", fontsize=22)
        else:
            ax.set_xlabel(f"{self.label} ({self.units})", fontsize=22)
        ax.set_yticks(range(len(result_df_region_av)))
        ax.set_yticklabels(make_region_labels(result_df_region_av.index))
        if stack_vessel_types or stack_vessel_sizes:
            ax.legend(
                handles,
                new_labels,
                title=legend_title,
                fontsize=20,
                title_fontsize=22,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        ax.set_ylabel("")
        plt.tight_layout()
        create_directory_if_not_exists(
            f"{top_dir}/plots/{self.fuel}-{self.pathway_type}-{self.pathway}"
        )

        filepath_save = f"{top_dir}/plots/{self.fuel}-{self.pathway_type}-{self.pathway}/{filename_save}.png"
        print(f"Saving figure to {filepath_save}")
        plt.savefig(filepath_save, dpi=200)

        filepath_save_pdf = f"{top_dir}/plots/{self.fuel}-{self.pathway_type}-{self.pathway}/{filename_save}.pdf"
        print(f"Saving figure to {filepath_save_pdf}")
        plt.savefig(filepath_save_pdf)
        plt.close()

    def make_all_hists_by_region(self):
        """
        Plots all stacked histograms for the available vessel types

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for vessel_type in vessel_types:
            self.make_hist_by_region(vessel_type)

        self.make_hist_by_region("all")

    def add_region_names(self):
        """
        Adds a column called NAME with region names to match naming conventions in the natural-earth-vector file, if that column doesn't already exist

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if "NAME" not in self.result_df.columns:
            # Map the index (which contains region names) to the proper names using the region_name_mapping dictionary
            self.result_df["NAME"] = self.result_df.index.map(
                lambda x: region_name_mapping.get(x, x)
            )

    def map_by_region(self, vessel_type="all", vessel_size="all"):
        """
        Maps the quantity geospatially by region, overlaid on a map of the world

        Parameters
        ----------
        vessel_type : str
            Vessel type, can currently be one of:
                * bulk_carrier_ice: bulk carrier vessel (internal combustion engine)
                * container_ice: container vessel (internal combustion engine)
                * tanker_ice: tanker vessel (internal combustion engine)
                * gas_carrier_ice: gas carrier vessel (internal combustion engine)
                
        vessel_size : str
            Size class for the given vessel

        Returns
        -------
        None
        """
        # If vessel option is provided as "all", plot the quantity for the full fleet
        if vessel_type == "all":
            column = f"fleet"

        # If a vessel_type option other than "all" is provided and vessel_size is set to "all", plot the given quantity for all vessel sizes of the given vessel type
        else:
            # Ensure that a valid vessel type was provided
            vessel_types_list = vessel_sizes.keys()

            if vessel_type not in vessel_types_list:
                raise Exception(
                    f"Error: Vessel type {vessel_type} not recognized. Acceptable types: {vessel_types_list}"
                )

            # If the vessel size is provided as "all", plot the quantity for all sizes of the given vessel type
            if vessel_size == "all":
                column = f"{vessel_type}"

            # If a vessel size other than "all" is provided, plot the quantity for the given vessel type and size
            else:
                # Ensure that a valid vessel size was provided
                vessel_sizes_list = vessel_sizes[vessel_type].keys()
                if vessel_size not in vessel_sizes_list:
                    raise Exception(
                        f"Error: Vessel size {vessel_size} not recognized. Acceptable sizes: {vessel_sizes_list}"
                    )

                column = f"{vessel_size}"

        # Load a base world map from geopandas
        url = "https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(url)

        # Add West Australia to the world geojson
        #world = add_west_australia(world)

        # Add a column "NAME" to self.results_df with region names to match the geojson world file, if needed
        self.add_region_names()

        # Create a figure and axis with appropriate size
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Merge the result_df with the world geodataframe based on the column "NAME" with region names
        merged = world.merge(self.result_df, on="NAME", how="left")

        # Plot the base map
        world.plot(ax=ax, color="white", edgecolor="black")

        # Plot the regions with data, using a colormap to represent the quantity
        merged.plot(
            column=column,
            cmap="coolwarm",
            linewidth=0.8,
            ax=ax,
            edgecolor="0.8",
            legend=False,
        )

        # Create a horizontal colorbar with appropriate formatting
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=1)
        sm = plt.cm.ScalarMappable(
            cmap="coolwarm",
            norm=plt.Normalize(vmin=merged[column].min(), vmax=merged[column].max()),
        )
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        if self.units is None:
            cbar.set_label(f"{self.label}", fontsize=20)
        else:
            cbar.set_label(f"{self.label} ({self.units})", fontsize=20)

        vessel_type_label = "All"
        if not vessel_type == "all":
            vessel_type_label = vessel_type_title[vessel_type]
        
        vessel_size_label = "All"
        if not vessel_size == "all":
            vessel_size_label = vessel_size_title[vessel_size]

        ax.text(
            1.03,
            0.75,
            f"Fuel: {self.fuel_label}",
            transform=ax.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )
        ax.text(
            1.03,
            0.65,
            f"Fuel Type: {self.pathway_type_label}",
            transform=ax.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )
        ax.text(
            1.03,
            0.55,
            f"Pathway: {self.pathway_label}",
            transform=ax.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )
        ax.text(
            1.03,
            0.45,
            f"Vessel Type: {vessel_type_label}",
            transform=ax.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )
        ax.text(
            1.03,
            0.35,
            f"Vessel Size: {vessel_size_label}",
            transform=ax.transAxes,
            fontsize=20,
            va="top",
            ha="left",
        )

        plt.subplots_adjust(left=0.05, right=0.67)
        #plt.tight_layout()

        # Save the plot
        create_directory_if_not_exists(
            f"{top_dir}/plots/{self.fuel}-{self.pathway_type}-{self.pathway}"
        )
        filepath_save = f"{top_dir}/plots/{self.fuel}-{self.pathway_type}-{self.pathway}/{self.fuel}-{self.pathway_type}-{self.pathway}-{self.quantity}-{self.modifier}-map_by_region.png"
        print(f"Saving figure to {filepath_save}")
        plt.savefig(filepath_save, dpi=200)

        filepath_save_pdf = f"{top_dir}/plots/{self.fuel}-{self.pathway_type}-{self.pathway}/{self.fuel}-{self.pathway_type}-{self.pathway}-{self.quantity}-{self.modifier}-map_by_region.pdf"
        print(f"Saving figure to {filepath_save_pdf}")
        plt.savefig(filepath_save_pdf)
        plt.close()
        
    # GE - 3/19/2025 - plot relative resource availability by country on a map
    def plot_resource_availability_map(self, resource_type):
        """
        Plots a world map with a log-scale color gradient representing resource availability by country.

        Parameters:
        -----------
        resource_type : str
            Type of resource (e.g., 'Water', 'Carbon Dioxide', 'Natural Gas', 'Electricity').

        Returns:
        --------
        None
        """
        # Make data frame containing data from resource availability spreadsheet for the desired resource type
        resource_df = self.read_resource_excel(f"{top_dir}/data/NationalResourceAvailabilityData.xlsx", resource_type)
        # smaller data frame only containing the country names and the average resource availabilities
        resource_avg_df = resource_df.iloc[:, [0]]
        
        # Load world map
        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(url)

        # Merge world map with resource data
        merged = world.merge(resource_avg_df, left_on="NAME", right_index=True, how="left")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Filter out NaN and zero values
        valid_values = merged["Average"].replace(0, np.nan).dropna().values

        # Compute vmin and vmax while ignoring NaNs and zeros
        vmin = np.nanmin(valid_values)  # Ignores NaNs and zeros
        vmax = np.nanmax(valid_values)  # Ignores NaNs and zeros
        
        merged.plot(column="Average", cmap="coolwarm", linewidth=0.8, ax=ax, edgecolor="black", legend=True, norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(f"{resource_type} Availability by Country", fontsize=16)
        ax.set_axis_off()
        
        unit_labels = {
            "Natural Gas": "Billion Cubic Meters",
            "Carbon Dioxide": "Megatons",
            "Water": "Cubic Meters",
            "Electricity": "TWh"
        }
        legend_label = unit_labels.get(resource_type)
        

        # Find the color bar and modify its label
        cbar = ax.get_figure().get_axes()[-1]  # Get last axis (which should be the color bar)
        cbar.set_ylabel(f"Log-normal Distribution", fontsize=12)
        
        if resource_type == "Natural Gas":
            resource_type_short = "NG"
        elif resource_type == "Carbon Dioxide":
            resource_type_short = "CO2"
        else:
            resource_type_short = resource_type

        filepath_save = f"{top_dir}/plots/availability_maps/{resource_type_short}-AvailabilityMap.png"
        print(f"Saving figure to {filepath_save}")
        plt.savefig(filepath_save, dpi=200)
        
        
    # GE - 3/23/2025 - plot SD:average ratio by country on a map
    def plot_SD_ratio_map(self, resource_type):
        """
        Plots a world map with a log-scale color gradient representing SD:average ratio by country.

        Parameters:
        -----------
        resource_type : str
            Type of resource (e.g., 'Water', 'Carbon Dioxide', 'Natural Gas', 'Electricity').

        Returns:
        --------
        None
        """
        # Make data frame containing data from resource availability spreadsheet for the desired resource type
        resource_df = self.read_resource_excel(f"{top_dir}/data/NationalResourceAvailabilityData.xlsx", resource_type)
        # smaller data frame only containing the country names and the SD:ratio
        resource_ratio_df = resource_df.iloc[:, [2]]
        
        # Load world map
        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(url)

        # Merge world map with resource data
        merged = world.merge(resource_ratio_df, left_on="NAME", right_index=True, how="left")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Filter out NaN and zero values
        valid_values = merged["Ratio of SD:Average"].replace(0, np.nan).dropna().values

        # Compute vmin and vmax while ignoring NaNs and zeros
        vmin = np.nanmin(valid_values)  # Ignores NaNs and zeros
        vmax = np.nanmax(valid_values)  # Ignores NaNs and zeros
        
        merged.plot(column= "Ratio of SD:Average", cmap="coolwarm", linewidth=0.8, ax=ax, edgecolor="black", legend=True, norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(f"{resource_type} Standard Deviation to Average Availability Ratio by Country", fontsize=16)
        ax.set_axis_off()

        if resource_type == "Natural Gas":
            resource_type_short = "NG"
        elif resource_type == "Carbon Dioxide":
            resource_type_short = "CO2"
        else:
            resource_type_short = resource_type

        filepath_save = f"{top_dir}/plots/availability_maps/{resource_type_short}-SDAvailabilityMap.png"
        print(f"Saving figure to {filepath_save}")
        plt.savefig(filepath_save, dpi=200)
        
        
    # GE - 3/23/25 - Function to get continent based on country using fuzzy matching
    def get_continent(self, country_name, country_list):
        """
        Returns name of continent a given country is part of.

        Parameters:
        -----------
        country_name : str
            Country name as it appears in the data source.
            
        country_list : list of str
            Country name as it appears in the reference, intended to be pycountry.

        Returns:
        --------
        continent_name : str
            Name of continent the country is part of, according to pycountry.
        """
        try:
            # Find the best match for the country name
            best_match = process.extractOne(country_name, country_list)
            
            if best_match:
                matched_country = best_match[0]
                country_alpha2 = pc.country_name_to_country_alpha2(matched_country)
                continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
                continent_name = pc.convert_continent_code_to_continent_name(continent_code)
                return continent_name
            else:
                return None
        except KeyError:
            return None

    # GE - 3/23/25 - stacked plot to compare continential resource availability to global demand
    def add_continent_column(self, resource_type):
        """
        Adds column to dataframe containing the continent in which each country is found.
        
        Parameters:
        -----------
        resource_type : str
            Type of resource (e.g., 'Water', 'Carbon Dioxide', 'Natural Gas', 'Electricity').

        Returns:
        --------
        resource_average_df : dataframe
            Dataframe with column added for continent.
        """
        # List of country names recognized by pycountry
        country_list = [country.name for country in pycountry.countries]
        
        # dataframe from NationalResourceAvailabilityData excel sheet.
        resource_df = self.read_resource_excel(f"{top_dir}/data/NationalResourceAvailabilityData.xlsx", resource_type)
    
        # smaller data frame only containing the country names and the average resource availabilities and standard deviation of each country's data
        resource_avg_df = resource_df.iloc[:, [0,1]]
        
        # Apply the get_continent function to create a new 'Continent' column in the dataframe
        resource_avg_df['Continent'] = resource_avg_df.index.to_series().apply(lambda x: self.get_continent(str(x), country_list) if isinstance(x, str) else None)
        
        # Remove NaN values from the DataFrame (filter out countries with NaN availability)
        resource_avg_df = resource_avg_df.dropna(subset=['Continent'])
        
        return resource_avg_df
    
    
    # GE - 3/23/25 - stacked plot to compare continential resource availability to global demand
    def make_stacked_plot_avail_vs_demand(self, resource_type, fuel, pathway):
        """
        Makes stacked plot to compare global resource availability (sum of continent-level data) to global demand.
        
        Parameters:
        -----------
        

        Returns:
        --------
        None
        """
        resource_avg_cont_df = self.add_continent_column(resource_type)
        
        # Aggregate availability per continent
        continent_avail_df = resource_avg_cont_df.groupby("Continent")["Average"].sum()
        
        # Compute global standard deviation
        global_stdev = np.sqrt(np.sum(resource_avg_cont_df["Standard Deviation"] ** 2))
        
        # Set color palette (one color per continent)
        colors = sns.color_palette("husl", len(continent_avail_df))  # Husl gives distinct colors
        
        if resource_type == "Natural Gas":
            resource_type_short = "NG"
        elif resource_type == "Carbon Dioxide":
            resource_type_short = "CO2"
        else:
            resource_type_short = resource_type
        
        # Get demand data from existing csv for the given pathway and resource
        all_results_df = pd.read_csv(f"{top_dir}/processed_results/{fuel}-{pathway}-Consumed{resource_type_short}_main-fleet.csv")# Load the correct data

        # Extract the global average for the entire fleet
        global_demand = all_results_df.loc[all_results_df["Region"] == "Global Average", "fleet"].values[0]
        
        # Define unit conversion factors (convert demand to match availability)
        unit_conversions = {
            "NG": 1e-6 * 1/35.3,  # Convert from GJ to billion cubic meters
            "CO2": 1e-9,  # Convert from kilograms to megatons
            "Water": 1,  # Convert from cubic meters to cubic meters
            "Electricity": 1e-9,  # Convert from kWh to TWh
        }
        
        # Convert global demand to match availability units
        conversion_factor = unit_conversions.get(resource_type_short)
        global_demand *= conversion_factor
        
        # Set color palette (one color per continent)
        colors = sns.color_palette("tab10", len(continent_avail_df))  # Husl gives distinct colors
        
        # Create stacked bar plot
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Plot availability (stacked by continent)
        bottom = 0
        for continent, value in continent_avail_df.items():
            ax1.bar("Availability", value, bottom=bottom, label=continent, color=colors.pop(0))
            bottom += value
        
        
        # Set the y-axis limits based on availability if order of magnitude or more different
        max_availability = continent_avail_df.sum()
        
        # format pathway label in legend
        parts = pathway.split('_')  # Split by underscores
        if len(parts) > 4:  # Ensure there are at least 3 underscores
            formatted_pathway = '_'.join(parts[:3]) + '_\n' + '_'.join(parts[3:])  # Rejoin with newline
        else:
            formatted_pathway = pathway
        
        if global_demand != 0 and max_availability > 10 * global_demand:
            # Create secondary y-axis for demand
            ax2 = ax1.twinx()
            ax2.bar("Demand", global_demand, color="gray", label=f"Global Demand\n({formatted_pathway})", alpha=1)
        
            # Scale demand axis for visibility vs. availability
            ax1.set_ylim(0, max_availability * 1.2)
            ax2.set_ylim(0, global_demand * 3)
            
            # Merge legends from both axes
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(handles1 + handles2, labels1 + labels2, title="Legend", loc="center left", bbox_to_anchor=(1.3, 0.75))
        else:
            # Create bar for demand
            ax1.bar("Demand", global_demand, color="gray", label=f"Global Demand\n({formatted_pathway})", alpha=1)
            
            # Legend
            handles1, labels1 = ax1.get_legend_handles_labels()
            ax1.legend(handles1, labels1, title="Legend", loc="center left", bbox_to_anchor=(1.3, 0.75))
            
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(right=0.6)  # Moves plot left to create space for the legend
        plt.subplots_adjust(left=0.2)  # Increase left margin space
        
        # add error bars based on standard deviation data
        ax1.errorbar(x=["Availability"], y=[max_availability], yerr=[global_stdev], fmt='o', color='black', capsize=5, markersize=2)
        
        # get y axis units
        # Define unit labels for each resource type
        unit_labels = {
            "NG": "Billion Cubic Meters",
            "CO2": "Megatons",
            "Water": "Cubic Meters",
            "Electricity": "TWh"
        }
        y_axis_label = unit_labels.get(resource_type_short)
        

        # Labels for left-hand axis and title
        plt.title(f"Global {resource_type} Availability vs. Demand")
        ax1.set_ylabel(f"{y_axis_label}")
        ax1.tick_params(axis='y', labelsize=9)  # Adjust label font size
        if 'ax2' in locals():  # If a second axis (demand) exists
            ax2.set_ylabel(f"{y_axis_label}")
            ax2.tick_params(axis='y', labelsize=9)  # Adjust label font size
            
        
        # change y-axis to scientific notation
        ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if 'ax2' in locals():
            ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))


        # Save plots
        filepath_save = f"{top_dir}/plots/{fuel}/{fuel}-{pathway}-{resource_type_short}-StackedAvailDemand.png"
        print(f"Saving figure to {filepath_save}")
        plt.savefig(filepath_save, dpi=200)
        
        
    

class ProcessedPathway:
    """
    A class to contain NavigaTE results for a given fuel production pathway, including all quantities evaluated for the pathway

    Attributes
    ----------

    modifier : str
        Modifier to the quantity. Can be one of:
            * vessel: Value of the quantity per vessel
            * fleet: Value of the quantity aggregated over all vessels
            * per_mile: Value of the quantity per nautical mile (nm)
            * per_tonn_mile: Value of the quantity per cargo tonne-nm

    fuel : str
        Fuel produced by the given production pathway (eg. ammonia, hydrogen, etc.)

    pathway_type : str
        Fuel production pathway type classification

    pathway : str
        Name of the production pathway, as it's saved in the name of the input csv file

    results_dir : str
        Name of the directory containing processed results in csv files

    result_df : pandas.DataFrame
        Dataframe containing the processed pathway result in the csv file

    quantities : list of str
        List of quantities available in the processed data available for the given pathway

    ProcessedQuantities : dict of ProcessedQuantity objects
        Dictionary containing a ProcessedQuantity class object for each pathway

    Methods
    -------
    get_quantities(self):
        Collects the names of all quantities evaluated for the given pathway

    get_processed_quantities(self):
        Collects the names of all quantities evaluated for the given pathway

    make_all_hists_by_region(self, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "fleet"]):
        Executes make_all_hists_by_region() in each ProcessedQuantity class instance contained in the ProcessedQuantities dictionary to produce validation hists for the given pathway, for the selected quantities.

    map_all_by_region(self, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "fleet"])
        Executes map_by_region() in each ProcessedQuantity class instance contained in the ProcessedQuantities dictionary to produce geospatial maps of the given quantities and modifiers.
    """

    def __init__(self, fuel, pathway, results_dir=RESULTS_DIR):
        self.fuel = fuel
        #print(pathway)
        self.pathway_type = get_pathway_type(pathway)
        self.pathway = pathway
        self.results_dir = results_dir
        self.quantities = self.get_quantities()
        self.modifiers = self.get_modifiers()
        self.ProcessedQuantities = self.get_processed_quantities()

    def get_quantities(self):
        """
        Collects the names of all quantities evaluated for the given pathway

        Parameters
        ----------
        None

        Returns
        -------
        quantities : list of str
            List of all quantities evaluated for the pathway
        """
        quantities = find_unique_identifiers(
            self.results_dir,
            "quantity",
            f"{self.fuel}-{self.pathway}",
        )

        return quantities

    def get_modifiers(self):
        """
        Collects unique modifiers available based on a sample quantity.

        Parameters
        ----------
        None

        Returns
        -------
        modifiers : list of str
            List of all available modifiers
        """
        sample_quantity = self.quantities[0]
        modifiers = find_unique_identifiers(
            self.results_dir,
            "modifier",
            f"{self.fuel}-{self.pathway}-{sample_quantity}",
        )
        return modifiers

    def get_processed_quantities(self):
        """
        Collects all instances of the ProcessedQuantity path for the given pathway.

        Parameters
        ----------
        None

        Returns
        -------
        ProcessedQuantities : Dictionary
            Dictionary containing all ProcessedQuantity objects for the given pathway
        """
        ProcessedQuantities = {}
        for quantity in self.quantities:
            ProcessedQuantities[quantity] = {}
            modifiers = find_unique_identifiers(
                self.results_dir,
                "modifier",
                f"{self.fuel}-{self.pathway}-{quantity}",
            )
            for modifier in modifiers:
                ProcessedQuantities[quantity][modifier] = ProcessedQuantity(
                    quantity, modifier, self.fuel, self.pathway
                )

        return ProcessedQuantities

    def apply_to_all_quantities(
        self,
        method_name,
        quantities=["TotalEquivalentWTW", "TotalCost", "CostTimesEmissions", "AverageCostEmissionsRatio"],
        modifiers=["per_tonne_mile", "per_tonne_mile_orig", "per_cbm_mile", "per_cbm_mile_orig", "per_mile", "vessel", "fleet"],
    ):
        """
        Applies the provided method to instances of the ProcessedQuantity class for all provided quantities and modifiers

        Parameters
        ----------
        quantities : str or list of str
            List of quantities to make hists of. If "all" is provided, it will make hists of all quantities in the ProcessedQuantities dictionary

        modifiers : str
            List of modifiers to include in the hists. If "all" is provided, it will make hists with all modifiers in the ProcessedQuantities dictionary

        method : Method of the ProcessedQuantity class

        Returns
        -------
        None
        """

        # Handle the situation where the user wants to apply the method to all quantities and/or all modifiers
        all_available_quantities = self.quantities
        if quantities == "all":
            quantities = all_available_quantities

        for quantity in quantities:
            if quantity not in all_available_quantities:
                raise Exception(
                    f"Error: Provided quantity '{quantity}' is not available in self.ProcessedQuantities. \n\nAvailable quantities: {self.quantities}."
                )

            # Handle the situation where the user wants to apply the method to all available modifiers
            all_available_modifiers = find_unique_identifiers(
                self.results_dir,
                "modifier",
                f"{self.fuel}-{self.pathway}-{quantity}",
            )
            modifiers = all_available_modifiers

            for modifier in modifiers:
                if modifier not in all_available_modifiers:
                    raise Exception(
                        f"Error: Provided modifier '{modifier}' is not available in self.ProcessedQuantities. \n\nAvailable modifiers: {self.modifiers}."
                    )

                # Get the instance of ProcessedQuantity
                processed_quantity_instance = self.ProcessedQuantities[quantity][
                    modifier
                ]

                # Dynamically get the method from the instance and call it
                method_to_call = getattr(processed_quantity_instance, method_name)
                method_to_call()

    def get_region_average_results(self, quantities, modifier):
        """
        Collects results for the given quantities and modifier, averaged over all countries.

        Parameters
        ----------
        quantities : list of str
            List of quantities to make hists of. If "all" is provided, it will make hists of all quantities in the ProcessedQuantities dictionary

        modifier : str
            Modifier to use in evaluating the region average.

        Returns
        -------
        region_av_results_dict : Dictionary of floats
            Dictionary containing the results for the given quantities and modifier
        """
        region_av_results_dict = {}

        column_name = f"fleet"
        for quantity in quantities:
            processed_quantity = self.ProcessedQuantities[quantity][modifier]
            processed_quantity_av = processed_quantity.result_df.loc[
                "Global Average", column_name
            ]
            region_av_results_dict[quantity] = processed_quantity_av

        return region_av_results_dict

    def get_all_region_results(self, quantity, modifier):
        """
        Collects results for the given quantity and modifier for all countries.

        Parameters
        ----------
        quantity : str
            Quantity to use in evaluating the region average.

        modifier : str
            Modifier to use in evaluating the region average.

        Returns
        -------
        individual_region_results_dict : Dictionary of floats
            Dictionary containing the results for each region.
        """

        result_df = self.ProcessedQuantities[quantity][modifier].result_df

        # Separate result_df into rows with vs. without "_" (where "_" indicates it's one of several individual estimates for a given region)
        result_df_region_av = result_df[~result_df.index.str.contains("_")]
        result_df_region_individual = result_df[result_df.index.str.contains("_")]

        countries_av = result_df_region_av.index
        countries_individual = result_df_region_individual.index

        # Get list of countries that have individual entries
        countries_with_multiple_entries = []
        for entry in countries_individual:
            region_name = entry.split("_")[0]
            if region_name not in countries_with_multiple_entries:
                countries_with_multiple_entries.append(region_name)

        individual_region_results_dict = {}
        multiple_region_results_dict = {}
        column_name = f"fleet"
        for region in countries_av:
            if region != "Global Average":
                region_label = get_region_label(region)
                individual_region_results_dict[region_label] = result_df_region_av.loc[
                    region, column_name
                ]
            if region in countries_with_multiple_entries:
                for entry in countries_individual:
                    entry_elements = entry.split("_")
                    entry_region = entry_elements[0]
                    entry_number = entry_elements[1]
                    if entry_region == region:
                        region_label = get_region_label(entry_region)
                        multiple_region_results_dict[
                            f"{region_label} ({entry_number})"
                        ] = result_df_region_individual.loc[entry, column_name]

        return individual_region_results_dict, multiple_region_results_dict

    def make_all_hists_by_region(
        self,
        quantities=["TotalEquivalentWTW", "TotalCost", "CostTimesEmissions", "AverageCostEmissionsRatio"],
        modifiers=["per_tonne_mile", "per_tonne_mile_orig", "per_cbm_mile", "per_cbm_mile_orig", "per_mile", "vessel", "fleet"],
    ):
        """
        Executes make_all_hists_by_region() in each ProcessedQuantity class instance contained in the ProcessedQuantities dictionary to produce validation hists for the given pathway, for the selected quantities.

        Parameters
        ----------
        quantities : str or list of str
            List of quantities to make hists of. If "all" is provided, it will make hists of all quantities in the ProcessedQuantities dictionary

        modifiers : str
            List of modifiers to include in the hists. If "all" is provided, it will make hists with all modifiers in the ProcessedQuantities dictionary

        Returns
        -------
        None
        """

        self.apply_to_all_quantities("make_all_hists_by_region", quantities, modifiers)

    def map_all_by_region(
        self,
        quantities=["TotalEquivalentWTW", "TotalCost", "CostTimesEmissions", "AverageCostEmissionsRatio"],
        modifiers=["per_tonne_mile", "per_mile", "vessel", "fleet"],
    ):
        """
        Executes map_by_region() in each ProcessedQuantity class instance contained in the ProcessedQuantities dictionary to produce geospatial maps of the given quantities and modifiers.

        Parameters
        ----------
        quantities : str or list of str
            List of quantities to make geospatial maps of. If "all" is provided, it will make maps of all quantities in the ProcessedQuantities dictionary

        modifiers : str
            List of modifiers to include in the maps. If "all" is provided, it will make maps with all modifiers in the ProcessedQuantities dictionary

        Returns
        -------
        None
        """
        self.apply_to_all_quantities("map_by_region", quantities, modifiers)
        

class ProcessedFuel:
    """
    A class to contain NavigaTE results for a given fuel, including all its pathways

    Attributes
    ----------
    pathway_names : list of str
        List of pathways with processed data available for the given fuel

    ProcessedPathways : dict of ProcessedPathway objects
        Dictionary containing a ProcessedPathway class object for each pathway
    """

    def __init__(self, fuel, results_dir=RESULTS_DIR):
        self.fuel = fuel
        self.results_dir = results_dir
        self.pathways = self.get_pathways()
        self.ProcessedPathways = self.get_processed_pathways()
        self.type_pathway_dict = self.organize_pathways_by_type()

    def get_pathways(self):
        """
        Collects the names of all pathways contained in processed csv files for the given fuel

        Parameters
        ----------
        None

        Returns
        -------
        pathways : list of Dictionaries
            List of unique pathways available for the given fuel, provided as dictionaries containing the pathway name and its associated type.
        """
        pathways = find_unique_identifiers(
            self.results_dir, "pathway", f"{self.fuel}"
        )
        
        return pathways

    def get_processed_pathways(self):
        """
        Collects all instances of the ProcessedPathway class for the given fuel.

        Parameters
        ----------
        None

        Returns
        -------
        ProcessedPathways : Dictionary
            Dictionary containing all ProcessedPathway objects for the given fuel
        """
        ProcessedPathways = {}
        for pathway in self.pathways:
            ProcessedPathways[pathway] = ProcessedPathway(
                self.fuel, pathway
            )

        return ProcessedPathways

    def organize_pathways_by_type(self):
        """
        Organizes the pathways according to their type (electro_grid, electro_renew, blue, grey, etc.)

        Parameters
        ----------
        None

        Returns
        -------
        type_pathway_dict : dictionary of lists of str
            Dictionary containing a list of pathways corresponding to each type
        """
        type_pathway_dict = {}

        for pathway in self.pathways:
            pathway_type = get_pathway_type(pathway)
            if pathway_type not in type_pathway_dict:
                type_pathway_dict[pathway_type] = [pathway]
            else:
                type_pathway_dict[pathway_type].append(pathway)
        return type_pathway_dict

    def get_all_countries(self):
        """
        Gets a list of all countries for which results are evaluated over all pathways for the given fuel.

        Parameters
        ----------
        None

        Returns
        -------
        all_countries : dictionary of lists of str
            Dictionary containing a list of pathways corresponding to each type
        """
        all_countries = []
        for pathway in self.ProcessedPathways:
            processed_quantities = self.ProcessedPathways[pathway].ProcessedQuantities
            sample_quantity = list(processed_quantities.keys())[0]
            sample_modifier = list(processed_quantities[sample_quantity].keys())[0]

            sample_processed_quantity = processed_quantities[sample_quantity][
                sample_modifier
            ]
            result_df = sample_processed_quantity.result_df

            countries = result_df[~result_df.index.str.contains("_")].index
            for region in countries:
                region_label = get_region_label(region)
                if (
                    region_label not in all_countries
                    and not region_label == "Global Average"
                ):
                    all_countries.append(region_label)

        return all_countries

    def make_stacked_hist(self, quantity, modifier, sub_quantities=[], plot_global=False, exclude_lsfo=False):
        """
        Makes a histogram of the given quantity with respect to the available fuel production pathways.
        Stacks pathways grouped by color and includes yellow bands to indicate total stacked results.

        Parameters
        ----------
        quantity : str
            Quantity to plot the histogram for

        modifier : str
            Modifier to plot the quantity for

        Returns
        -------
        None
        """
        fuel_label = get_fuel_label(self.fuel)
        quantity_label = get_quantity_label(quantity)
        quantity_units = get_units(quantity, modifier)

        if not sub_quantities:
            sub_quantities = [quantity]

        num_pathways = len(self.pathways)
        fig_height = max(6, num_pathways * 0.9)  # Adjust height based on pathways

        fig, ax = plt.subplots(figsize=(20, fig_height))

        # Sort pathways by their associated color
        sorted_pathways = sorted(
            self.pathways,
            key=lambda p: get_pathway_type_color(get_pathway_type(p))
        )

        y_positions = np.arange(len(sorted_pathways))  # Assign y-positions

        cumulative_values = {}
        cumulative_values_negative = {}

        blues = generate_blue_shades(len(sub_quantities))

        # Get all individual countries with processed data for the given fuel
        countries = self.get_all_countries()
        region_colors = assign_colors_to_strings(countries)

        scatter_handles, scatter_labels = [], []
        bar_handles, bar_labels = [], []

        ax.axvline(0, color="black")  # Vertical reference line at 0

        def make_bar(pathway, pathway_name, pathway_label, y_pos):
            """
            Plots a single bar for a given pathway.

            Parameters
            ----------
            pathway : ProcessedPathway
                ProcessedPathway class instance containing the info to plot

            pathway_name : str
                Name of the pathway

            pathway_label : str
                Label for the pathway

            y_pos : float
                Adjusted y position for plotting based on sorting
            """
            if pathway_name not in cumulative_values:
                cumulative_values[pathway_name] = 0
                cumulative_values_negative[pathway_name] = 0

            region_average_results = pathway.get_region_average_results(sub_quantities, modifier)
            all_region_results, _ = pathway.get_all_region_results(quantity, modifier)

            for i, sub_quantity in enumerate(sub_quantities):
                value = region_average_results.get(sub_quantity, 0)

                if value >= 0:
                    bar = ax.barh(
                        y_pos,
                        value,
                        left=cumulative_values[pathway_name],
                        label=get_quantity_label(sub_quantity) if i_pathway == 0 else "",
                        color=blues[i],
                    )
                    cumulative_values[pathway_name] += value
                else:
                    bar = ax.barh(
                        y_pos,
                        value,
                        left=cumulative_values_negative[pathway_name],
                        label=get_quantity_label(sub_quantity) if i_pathway == 0 else "",
                        color=blues[i],
                    )
                    cumulative_values_negative[pathway_name] += value

                if i_pathway == 0:
                    bar_handles.append(bar[0])
                    bar_labels.append(get_quantity_label(sub_quantity))

            if not plot_global and all_region_results:
                for region in all_region_results:
                    if "Global" in region:
                        continue
                    scatter = ax.scatter(
                        all_region_results[region],
                        y_pos,
                        color="black",
                        s=50,
                        marker="o",
                        zorder=100,
                    )

            if not plot_global and i_pathway == 0:
                scatter_handles.append(scatter)
                scatter_labels.append("Individual Countries")

            # **Add yellow band showing total stacked value**
            if cumulative_values_negative[pathway_name] or cumulative_values[pathway_name]:
                total_width = cumulative_values[pathway_name] + cumulative_values_negative[pathway_name]
                bar_height = bar[0].get_height()  # Get the height of the horizontal bar
                y_center = bar[0].get_y() + bar_height / 2  # Calculate the center of the bar

                ax.plot(
                    [total_width, total_width],  # x coordinates (vertical line)
                    [y_center - bar_height * 0.4, y_center + bar_height * 0.4],  # y coordinates
                    color='yellow',
                    linewidth=5,
                    label="Sum" if i_pathway == 0 else ""
                )

            # Set the y-axis label color to match the pathway type
            y_labels = ax.get_yticklabels()
            pathway_type = get_pathway_type(pathway_name)
            if i_pathway < len(y_labels):
                y_labels[i_pathway].set_color(get_pathway_type_color(pathway_type))
                y_labels[i_pathway].set_fontweight("bold")
            cumulative_values[pathway_name] = 0

        # Loop through pathways and sort by color
        i_pathway = 0
        for pathway_name, y_pos in zip(sorted_pathways, y_positions):
            pathway = self.ProcessedPathways[pathway_name]
            pathway_label = get_pathway_label(pathway_name)
            make_bar(pathway, pathway_name, pathway_label, y_pos)
            i_pathway += 1

        # Set y-axis labels **after** all bars are plotted
        ax.set_yticks(y_positions)
        ax.set_yticklabels([get_pathway_label(p) for p in sorted_pathways], fontsize=18)

        # Apply colors to y-axis labels
        y_labels = ax.get_yticklabels()
        for idx, pathway_name in enumerate(sorted_pathways):
            pathway_type = get_pathway_type(pathway_name)
            y_labels[idx].set_color(get_pathway_type_color(pathway_type))
            y_labels[idx].set_fontweight("bold")

        # Add a LSFO bar for comparison (unless excluded)
        if not exclude_lsfo:
            lsfo_pathway = ProcessedPathway("lsfo", "fossil")
            make_bar(lsfo_pathway, "fossil", "LSFO (fossil)", len(y_positions) + 1)
            plt.axvline(
                lsfo_pathway.get_region_average_results([quantity], modifier)[quantity],
                linewidth=3,
                linestyle="--",
                color="black",
            )

        # Add labels and title
        xlabel = f"{quantity_label} ({quantity_units})" if quantity_units else quantity_label
        ax.set_xlabel(xlabel, fontsize=20)
        ax.set_title(f"Fuel: {fuel_label}", fontsize=24)

        # Add legend for stacked bars
        if bar_handles:
            legend1 = ax.legend(
                bar_handles,
                bar_labels,
                fontsize=16,
                title="Components",
                title_fontsize=20,
                bbox_to_anchor=(1.01, 0.8),
                loc="upper left",
                borderaxespad=0.0,
            )

        # Add legend for country scatter points
        if scatter_handles:
            ax.legend(
                scatter_handles,
                scatter_labels,
                fontsize=16,
                bbox_to_anchor=(1.01, 0.35),
                loc="center left",
                borderaxespad=0.0,
            )
            ax.add_artist(legend1)

        plt.subplots_adjust(left=0.25, right=0.8)

        # Construct filename and save figure
        filename_save = f"{self.fuel}-{quantity}-{modifier}-pathway_hist"
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}")
        filepath_save = f"{top_dir}/plots/{self.fuel}/{filename_save}.png"
        print(f"Saving figure to {filepath_save}")
        plt.savefig(filepath_save, dpi=200)
        
        filepath_save_pdf = f"{top_dir}/plots/{self.fuel}/{filename_save}.pdf"
        print(f"Saving figure to {filepath_save_pdf}")
        plt.savefig(filepath_save_pdf)
        
#        ax.set_xscale("log")
#        filepath_save_log = f"{top_dir}/plots/{self.fuel}/{filename_save}_logx.png"
#        print(f"Saving figure to {filepath_save_log}")
#        plt.savefig(filepath_save_log, dpi=200)

        plt.close()



    def make_all_stacked_hists(
        self,
        quantities=["TotalEquivalentWTW", "TotalCost", "CostTimesEmissions", "AverageCostEmissionsRatio"],
        modifiers=["per_tonne_mile", "per_tonne_mile_orig", "per_cbm_mile", "per_cbm_mile_orig", "per_mile", "vessel", "fleet"],
    ):
        """
        Plot a stacked histogram for the given quantities and modifiers with respect to the pathway and region.

        Parameters
        ----------
        quantities : list of str
            List of quantities to make stacked hists for

        modifiers : list of str
            List of modifieers to make stacked hists for

        Returns
        -------
        None
        """

        # Handle the situation where the user wants to plot all quantities and/or all modifiers
        sample_processed_pathway = self.ProcessedPathways[self.pathways[0]]
        all_avalable_quantities = sample_processed_pathway.quantities

        if quantities == "all":
            quantities = all_avalable_quantities

        for quantity in quantities:
            if quantity not in all_avalable_quantities:
                raise Exception(
                    f"Error: Provided quantity '{quantity}' is not available in self.ProcessedQuantities. \n\nAvailable quanities: {all_avalable_quantities}."
                )

            if quantity in result_components:
                sub_quantities = result_components[quantity]
            else:
                sub_quantities = []

            # Handle the situation where the user wants to apply the method to all available modifiers
            all_available_modifiers = find_unique_identifiers(
                self.results_dir,
                "modifier",
                f"{self.fuel}-{sample_processed_pathway.pathway}-{quantity}",
            )
            if modifiers == "all":
                modifiers = all_available_modifiers

            for modifier in modifiers:
                if modifier in all_available_modifiers:
                    self.make_stacked_hist(quantity, modifier, sub_quantities)
                    
    # GE - function to produce histograms of resource demands for each fuel and resource
    # modeled off make_all_stacked_hists()
    def make_all_resource_demands_hists(
        self,
        quantities=["ConsumedCO2_main", "ConsumedNG_main", "ConsumedElectricity_main", "ConsumedWater_main"],
        modifier = "fleet",
    ):
        """
        Plot a stacked histogram for the given quantities at fleet level with respect to the pathway and fuel.

        Parameters
        ----------
        quantities : list of str
            List of quantities to make stacked hists for

        Returns
        -------
        None
        """
        # Handle the situation where the user wants to plot all quantities
        sample_processed_pathway = self.ProcessedPathways[self.pathways[0]]
        all_available_quantities = ["ConsumedCO2_main", "ConsumedNG_main", "ConsumedElectricity_main", "ConsumedWater_main"]

        if quantities == "all":
            quantities = all_available_quantities

        # Handle the case where user enters invalid quantity
        for quantity in quantities:
            if quantity not in all_available_quantities:
                raise Exception(
                    f"Error: Provided quantity '{quantity}' is not available in self.ProcessedQuantities. \n\nAvailable quanities: {all_available_quantities}."
                )

        for quantity in quantities:
            self.make_stacked_hist(quantity, modifier, plot_global = True, exclude_lsfo = True)
    

    def apply_to_all_pathways(self, method_name, *args, **kwargs):
        """
        Applies the provided method to all available pathways.

        Parameters
        ----------
        method_name : str
            The name of the method to apply to each pathway.

        *args : tuple
            Positional arguments to pass to the method.

        **kwargs : dict
            Keyword arguments to pass to the method.

        Returns
        -------
        None
        """

        for pathway in self.ProcessedPathways:
            processed_pathway = self.ProcessedPathways[pathway]
            # Dynamically get the method from the instance and call it
            method_to_call = getattr(processed_pathway, method_name)
            method_to_call(*args, **kwargs)

    def make_all_hists_by_region(
        self,
        quantities=["TotalEquivalentWTW", "TotalCost", "CostTimesEmissions", "AverageCostEmissionsRatio"],
        modifiers=["per_tonne_mile", "per_tonne_mile_orig", "per_cbm_mile", "per_cbm_mile_orig", "per_mile", "vessel", "fleet"],
    ):
        """
        Applies make_all_hists_by_region to all available pathways for the given fuel

        Parameters
        ----------
        quantities : str or list of str
            List of quantities to make hists of. If "all" is provided, it will make hists of all quantities in the ProcessedQuantities dictionary

        modifiers : str
            List of modifiers to include in the hists. If "all" is provided, it will make hists with all modifiers in the ProcessedQuantities dictionary

        Returns
        -------
        None
        """

        self.apply_to_all_pathways("make_all_hists_by_region", quantities, modifiers)

    def map_all_by_region(
        self,
        quantities=["TotalEquivalentWTW", "TotalCost", "CostTimesEmissions", "AverageCostEmissionsRatio"],
        modifiers=["per_tonne_mile", "per_mile", "vessel", "fleet"],
    ):
        """
        Applies map_all_by_region to all available pathways for the given fuel

        Parameters
        ----------
        quantities : str or list of str
            List of quantities to make hists of. If "all" is provided, it will make hists of all quantities in the ProcessedQuantities dictionary

        modifiers : str
            List of modifiers to include in the hists. If "all" is provided, it will make hists with all modifiers in the ProcessedQuantities dictionary

        Returns
        -------
        None
        """

        self.apply_to_all_pathways("map_all_by_region", quantities, modifiers)


def structure_results_fuels_types(
    quantity, modifier, fuels=None
):
    """
    Structures results from all pathways for the given fuels into pathway types.

    Parameters
    ----------
    quantity : str
        Quantity to plot

    modifiers : str
        Modifier for the quantity

    fuels : list of str
        List of fuels to plot. If None provided, gets a list of unique available fuels.

    Returns
    -------
    results_fuels_types : Dictionary
        Dictionary containing the results for all fuel types
    """
    
    # If supplied fuels is None, get unique fuels based on filenames in the results dir
    if fuels is None:
        fuels = find_unique_identifiers(RESULTS_DIR, "fuel", "")

    results_fuels_types = {}
    for fuel in fuels:
        results_fuels_types[fuel] = {}
        processed_fuel = ProcessedFuel(fuel)
        type_pathway_dict = processed_fuel.type_pathway_dict
        for pathway_type in type_pathway_dict:
            results_fuels_types[fuel][pathway_type] = []
            for pathway in type_pathway_dict[pathway_type]:
                processed_quantity = ProcessedQuantity(
                    quantity, modifier, fuel, pathway
                )
                result_df = processed_quantity.result_df

                for region in result_df.index:
                    if "_" not in region and region != "Global Average":
                        results_fuels_types[fuel][pathway_type].append(
                            result_df.loc[region, f"fleet"]
                        )
    return results_fuels_types


def plot_scatter_overlay(structured_results, quantity, modifier, plot_size=(12, 10), overlay_type='bar'):
    """
    Plot the data from a dictionary as horizontal scatterplots overlaid on either bars extending to the mean value or violin plots.

    Parameters
    ----------
    structured_results: dict
        The input data, a dictionary where keys are categories (e.g., 'hydrogen', 'ammonia')
        and values are dictionaries with subcategories (e.g., 'green', 'blue', 'grey') containing lists of float values.
    plot_size: tuple, optional
        Size of the plot (default: (12, 6)).
    overlay_type: str, optional
        Type of plot to overlay ('bar' for bars extending to mean value, 'violin' for violin plots). Default is 'bar'.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=plot_size)
    
    y_base = 0
    y_tick_positions = []  # Y positions for left-side labels
    y_tick_labels = []  # Left-side tick labels
    y_tick_positions_right = []  # Y positions for right-side labels
    y_tick_labels_right = []  # Right-side tick labels
    y_tick_colors_right = []  # Colors for right-side labels

    n_fuels = len(structured_results.keys())
    i_fuel = 0
    fuels_ordered = [f for f in structured_results.keys() if f != "lsfo"]
    fuels_ordered.append("lsfo")       # Ensures that LSFO appears at the top of the plot
    for fuel in fuels_ordered:
        n_pathway_types = 0  # Track number of pathway types per fuel

        for pathway_type in structured_results[fuel]:
            values = structured_results[fuel][pathway_type]
            y_value = y_base + n_pathway_types
            mean_value = np.mean(values) if values else 0

            if overlay_type == 'bar':
                # Plot the bar extending to the mean value
                ax.barh(
                    y_value,
                    mean_value,
                    color=get_pathway_type_color(pathway_type),
                    alpha=0.3,
                    height=0.5
                )
            elif overlay_type == 'violin' and len(values) > 1:
                # Plot the violin plot
                parts = ax.violinplot(
                    values,
                    positions=[y_value],
                    vert=False,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(get_pathway_type_color(pathway_type))
                    pc.set_alpha(0.3)

            # Plot the scatter plot
            ax.scatter(
                values,
                [y_value] * len(values),
                color=get_pathway_type_color(pathway_type),
                edgecolor="black",
                zorder=3
            )

            # Add right-side labels
            y_tick_positions_right.append(y_value)
            y_tick_labels_right.append(get_pathway_type_label(pathway_type))
            y_tick_colors_right.append(get_pathway_type_color(pathway_type))

            n_pathway_types += 1

        # Center the fuel label among its subcategories
        y_tick_positions.append(y_base + (n_pathway_types - 1) / 2)
        y_tick_labels.append(get_fuel_label(fuel))

        # Update y_base for next fuel
        y_base += n_pathway_types + 1  # Space between fuels

        # Add a horizontal line to separate different fuels
        i_fuel += 1
        if i_fuel != n_fuels:
            ax.axhline(y=y_base - 1, color="black", linewidth=2)

    # Add a vertical line for LSFO reference
    ax.axvline(structured_results["lsfo"]["grey"], ls="--", color="grey")

    # Set y-ticks with labels on the left side
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)

    # Set the labels and colors for the right-side y-axis
    ax_right = ax.twinx()
    ax_right.set_yticks(y_tick_positions_right)
    ax_right.set_yticklabels(y_tick_labels_right)
    ax_right.set_ylim(ax.get_ylim())  # Match left axis limits

    # Color right-side labels
    for tick_label, color in zip(ax_right.get_yticklabels(), y_tick_colors_right):
        tick_label.set_color(color)

    # Set axis labels
    quantity_label = get_quantity_label(quantity)
    units = get_units(quantity, modifier)
    xlabel = f"{quantity_label} ({units})" if units else f"{quantity_label}"
    ax.set_xlabel(xlabel, fontsize=22)
    
    plt.subplots_adjust(left=0.2, right=0.75)
    plt.tight_layout()
    
    # Save plots
    create_directory_if_not_exists(f"{top_dir}/plots/scatter_overlay")
    filepath_save = f"{top_dir}/plots/scatter_overlay/{quantity}-{modifier}.png"
    print(f"Saving figure to {filepath_save}")
    plt.savefig(filepath_save, dpi=200)
    
    filepath_save_pdf = f"{top_dir}/plots/scatter_overlay/{quantity}-{modifier}.pdf"
    print(f"Saving figure to {filepath_save_pdf}")
    plt.savefig(filepath_save_pdf)
    
#    ax.set_xscale("log")
#    filepath_save_log = f"{top_dir}/plots/scatter_overlay/{quantity}-{modifier}_logx.png"
#    print(f"Saving figure to {filepath_save_log}")
#    plt.savefig(filepath_save_log, dpi=200)
    plt.close()
    
    
def collect_cargo_mile_results(fuels=None):
    """
    Collect CargoMiles for each fuel and vessel type. Note that CargoMiles is independent of the fuel production pathway and region.
    
    Parameters
    ----------
    fuels : list of str
        List of fuels to plot. If None provided, gets a list of unique available fuels.

    Returns
    -------
    results_fuels_types : Dictionary
        Dictionary containing the results for all fuel types
    """
    
    # If supplied fuels is None, get unique fuels based on filenames in the results dir
    if fuels is None:
        fuels = find_unique_identifiers(RESULTS_DIR, "fuel", "")
    
    cargo_mile_results = {}
    for fuel in fuels:
        cargo_mile_results[fuel] = {}
        
        pathways = find_unique_identifiers(
            RESULTS_DIR, "pathway", fuel
        )
        
        processed_cargo_miles = ProcessedQuantity("CargoMiles", "fleet", fuel, pathways[0])
        result_df = processed_cargo_miles.result_df
        for vessel_type in vessel_types:
            cargo_mile_results[fuel][vessel_type] = result_df.loc["Global Average", vessel_type]
    
    return cargo_mile_results
    
def plot_cargo_miles():
    """
    Plot a stacked horizontal bar chart for cargo miles by fuel and vessel type.
    
    Parameters
    ----------
    cargo_mile_results : dict
        Dictionary where keys are fuel types and values are dictionaries.
        Each inner dictionary contains cargo miles for different vessel types (e.g., 'bulk_carrier_ice', 'container_ice', etc.).
    
    Returns
    -------
    None
        This function generates and displays a horizontal stacked bar plot, showing cargo miles stacked by vessel type for each fuel.
    """
    
    cargo_mile_results = collect_cargo_mile_results()
    
    # Get the list of fuels (keys) and vessel types (sub-keys) from the dictionary
    fuels = list(cargo_mile_results.keys())
    fuel_labels = [get_fuel_label(fuel) for fuel in fuels]
    vessel_types = list(next(iter(cargo_mile_results.values())).keys())

    # Create a figure and axes object for the plot
    fig, ax = plt.subplots(figsize=(12, len(fuels) * 0.7))  # Adjust the figure size based on the number of fuels
    
    # Initialize the bottom position for the stacked bars (all zeros at the start)
    bottom = [0] * len(fuels)

    # Loop over vessel types to plot the stacking
    for vessel_type in vessel_types:
        # Get the values of cargo miles for each fuel for the current vessel type
        values = [cargo_mile_results[fuel][vessel_type] for fuel in fuels]
        
        # Plot a horizontal bar for this vessel type on top of the previous bars (stacking)
        ax.barh(fuels, values, left=bottom, label=vessel_type_title[vessel_type])

        # Update the bottom position for the next stack
        bottom = [b + v for b, v in zip(bottom, values)]

    # Update the y-axis labels to use the fuel_labels
    ax.set_yticks(range(len(fuels)))  # Set y-ticks based on the number of fuels
    ax.set_yticklabels(fuel_labels, fontsize=18)  # Use fuel_labels for the y-axis tick labels
    
    # Add a vertical dashed line at the LSFO Cargo Miles
    lsfo_cargo_miles = cargo_mile_results.get("lsfo", {})
    total_cargo_miles_lsfo = sum(lsfo_cargo_miles.values())
    ax.axvline(total_cargo_miles_lsfo, color="black", linestyle="--")

    # Add labels and title
    ax.set_xlabel("Cargo Miles (tonne-miles)", fontsize=20)
    ax.legend(title="Vessel Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, title_fontsize=20)

    # Show the plot
    plt.tight_layout()
    plt.savefig("plots/cargo_miles_comparison.png", dpi=300)
    plt.savefig("plots/cargo_miles_comparison.pdf")
    

def main():

# ------- Sample execution of class methods for testing and development -------#
    processed_quantity = ProcessedQuantity("TotalCost", "fleet", "ammonia", "LTE_H_grid_E")
    processed_quantity.map_by_region()

    processed_quantity = ProcessedQuantity("TotalEquivalentWTW", "fleet", "ammonia", "LTE_H_grid_E")
    processed_quantity.map_by_region()
    
    processed_quantity = ProcessedQuantity("AverageCostEmissionsRatio", "vessel", "ammonia", "LTE_H_grid_E")
    processed_quantity.map_by_region()
    
#    processed_quantity = ProcessedQuantity("AverageCostEmissionsRatio", "vessel", "ammonia", "ATRCCS_H_grid_E")
#    processed_quantity.map_by_region()
    
#    processed_quantity.make_hist_by_region("bulk_carrier_ice")
#    processed_quantity = ProcessedQuantity("TotalCost", "vessel", "lng", "fossil")
#    processed_quantity.map_by_region()
#    processed_quantity.make_hist_by_region()
#    processed_quantity = ProcessedQuantity("TotalCost", "per_tonne_mile", "liquid_hydrogen", "LTE_H_grid_E")
#    processed_quantity.make_hist_by_region()
#    processed_quantity = ProcessedQuantity("TotalCost", "per_tonne_mile_orig", "liquid_hydrogen", "LTE_H_grid_E")
#    processed_quantity.map_by_region()
#
#    processed_pathway = ProcessedPathway("liquid_hydrogen", "LTE_H_grid_E")
#    processed_pathway.make_all_hists_by_region()
#    processed_pathway.map_all_by_region()
    
    # GE - test make_stacked_hist() with two new boolean arguments so it applies to global scale
#    processed_fuel_GE = ProcessedFuel("methanol")
#    processed_fuel_GE.make_stacked_hist("ConsumedWater_main", "fleet", plot_global = True, exclude_lsfo = True)
    
    # GE - test make_all_resource_demands_hist() - NOT WOKRING FOR "all" CASE
#    processed_fuel_GE_test = ProcessedFuel("ammonia")
#    processed_fuel_GE_test.make_all_resource_demands_hists()
    
    processed_fuel = ProcessedFuel("lng")
#    processed_fuel.make_stacked_hist("ConsumedElectricity_main", "vessel", ["TotalCAPEX", "TotalFuelOPEX", "TotalExcludingFuelOPEX"])

#    processed_fuel.make_stacked_hist("TotalCost", "fleet", ["TotalCAPEX", "TotalFuelOPEX", "TotalExcludingFuelOPEX"])
#    processed_fuel.make_stacked_hist("TotalCost", "per_tonne_mile_orig", ["TotalCAPEX", "TotalFuelOPEX", "TotalExcludingFuelOPEX"])
#    processed_fuel.make_stacked_hist("TotalCost", "per_cbm_mile", ["TotalCAPEX", "TotalFuelOPEX", "TotalExcludingFuelOPEX"])
#    processed_fuel.make_stacked_hist("TotalCost", "fleet", ["TotalCAPEX", "TotalFuelOPEX", "TotalExcludingFuelOPEX"])
#    processed_fuel.make_stacked_hist("TotalEquivalentWTW", "fleet", ["TotalEquivalentTTW", "TotalEquivalentWTT"])
#    processed_fuel.make_stacked_hist("CostTimesEmissions", "vessel", [])
#    processed_fuel.make_stacked_hist("AverageCostEmissionsRatio", "vessel", ["HalfCostRatio", "HalfWTWRatio"])
#    processed_fuel.make_stacked_hist("TotalCost", "vessel", [])
#    processed_fuel.make_stacked_hist("TotalEquivalentWTW", "vessel", [])
#    processed_fuel.make_stacked_hist("CAC", "vessel", [])
#    processed_fuel.make_stacked_hist("CostTimesEmissions", "vessel", [])
#    processed_fuel.make_stacked_hist("AverageCostEmissionsRatio", "vessel", [])


# -----------------------------------------------------------------------------#
    
#    # Loop through all fuels of interest
#    for fuel in ["liquid_hydrogen", "compressed_hydrogen", "ammonia", "methanol", "FTdiesel"]: #["compressed_hydrogen", "liquid_hydrogen", "ammonia", "methanol", "FTdiesel", "lsfo"]:
#        processed_fuel = ProcessedFuel(fuel)

#         Make validation plots for each fuel, pathway and quantity
#        processed_fuel.make_all_hists_by_region()
#        processed_fuel.map_all_by_region()
#        processed_fuel.make_all_stacked_hists()
    
#        processed_fuel.make_stacked_hist("TotalCost", "fleet", ["TotalCAPEX", "TotalFuelOPEX", "TotalExcludingFuelOPEX"])
#        processed_fuel.make_stacked_hist("TotalEquivalentWTW", "fleet", ["TotalEquivalentTTW", "TotalEquivalentWTT"])
#        processed_fuel.make_stacked_hist("CostTimesEmissions", "vessel", [])
#        processed_fuel.make_stacked_hist("AverageCostEmissionsRatio", "vessel", [])
#        processed_fuel.make_stacked_hist("CAC", "vessel", [])
    """
    
#    structured_results = structure_results_fuels_types("ConsumedElectricity_main", "fleet")
#    plot_scatter_overlay(structured_results, "ConsumedElectricity_main", "fleet", overlay_type="bar")
#    structured_results = structure_results_fuels_types("ConsumedCO2_main", "fleet")
#    plot_scatter_overlay(structured_results, "ConsumedCO2_main", "fleet", overlay_type="bar")
#    structured_results = structure_results_fuels_types("ConsumedWater_main", "fleet")
#    plot_scatter_overlay(structured_results, "ConsumedWater_main", "fleet", overlay_type="bar")
#    structured_results = structure_results_fuels_types("ConsumedLCB_main", "fleet")
#    plot_scatter_overlay(structured_results, "ConsumedLCB_main", "fleet", overlay_type="bar")
#    structured_results = structure_results_fuels_types("ConsumedNG_main", "fleet")
#    plot_scatter_overlay(structured_results, "ConsumedNG_main", "fleet", overlay_type="bar")
    
#    structured_results = structure_results_fuels_types("TotalCost", "fleet")
#    plot_scatter_overlay(structured_results, "TotalCost", "fleet", overlay_type="violin")
#    structured_results = structure_results_fuels_types("TotalEquivalentWTW", "fleet")
#    plot_scatter_overlay(structured_results, "TotalEquivalentWTW", "fleet", overlay_type="violin")
#    structured_results = structure_results_fuels_types("CostTimesEmissions", "vessel")
#    plot_scatter_overlay(structured_results, "CostTimesEmissions", "vessel", overlay_type="violin")
#    structured_results = structure_results_fuels_types("AverageCostEmissionsRatio", "vessel")
#    plot_scatter_overlay(structured_results, "AverageCostEmissionsRatio", "vessel", overlay_type="violin")
#    structured_results = structure_results_fuels_types("CAC", "vessel")
#    plot_scatter_overlay(structured_results, "CAC", "vessel", overlay_type="violin")
    
    """
#    for quantity in ["ConsumedWater_main", "ConsumedNG_main", "ConsumedElectricity_main", "ConsumedCO2_main", "CAC", "TotalCost", "TotalEquivalentWTW", "CostTimesEmissions", "AverageCostEmissionsRatio"]:

 #       for modifier in ["fleet", "vessel", "per_mile", "per_tonne_mile", "per_cbm_mile"]: #["vessel", "fleet", "per_mile", "per_tonne_mile", "per_tonne_mile_orig", "per_cbm_mile", "per_cbm_mile_orig"]:
 #           if (quantity == "AverageCostEmissionsRatio" or quantity == "CAC" or quantity == "CostTimesEmissions") and modifier != "vessel":
 #               continue
 #           structured_results = structure_results_fuels_types(quantity, modifier)
 #           if "_main" in quantity:
 #               plot_scatter_overlay(structured_results, quantity, modifier, overlay_type="bar")
 #           else:
 #               plot_scatter_overlay(structured_results, quantity, modifier, overlay_type="violin")

#    for quantity in ["ConsumedWater_main", "ConsumedNG_main", "ConsumedElectricity_main", "ConsumedCO2_main", "ConsumedLCB_main", "CAC", "TotalCost", "TotalEquivalentWTW", "CostTimesEmissions", "AverageCostEmissionsRatio"]:
#        for modifier in ["fleet", "vessel", "per_mile", "per_tonne_mile", "per_cbm_mile"]: #["vessel", "fleet", "per_mile", "per_tonne_mile", "per_tonne_mile_orig", "per_cbm_mile", "per_cbm_mile_orig"]:
#            if (quantity == "AverageCostEmissionsRatio" or quantity == "CAC" or quantity == "CostTimesEmissions") and modifier != "vessel":
#                continue
#            structured_results = structure_results_fuels_types(quantity, modifier)
#            if "_main" in quantity:
#                plot_scatter_overlay(structured_results, quantity, modifier, overlay_type="bar")
#            else:
#                plot_scatter_overlay(structured_results, quantity, modifier, overlay_type="violin")

# GE - 3/20/2025 - testing out resource mapping functions and stacked plot of availability vs. demand functions
    for resource in ["Water", "Electricity", "Carbon Dioxide", "Natural Gas"]:
        processed_quantity.plot_resource_availability_map(resource)
        processed_quantity.plot_SD_ratio_map(resource)
    
#    for fuel in ["ammonia", "liquid_hydrogen", "compressed_hydrogen", "methanol", "FTdiesel"]:
#        for resource in ["Water", "Electricity", "Carbon Dioxide", "Natural Gas"]:
#            for h_path in ["SMR_H", "SMRCCS_H", "LTE_H"]:
#                for e_path in ["grid_E", "renewable_E", "nuke_E"]:
#                    if fuel == "methanol" or fuel == "FTdiesel":
#                        for c_path in ["DAC_C", "BEC_C", "C"]: # "C" for if SMRCCS used for both H and C production
#                            if h_path != "SMRCCS_H" and c_path == "C":
#                                continue
#                            else:
#                                pathway = f"{h_path}_{c_path}_{e_path}"
#                    else:
#                        pathway = f"{h_path}_{e_path}"
#
#                    processed_quantity.make_stacked_plot_avail_vs_demand(resource, fuel, pathway)
    
        
#    processed_quantity.make_stacked_plot_avail_vs_demand("Water", "compressed_hydrogen", "LTE_H_grid_E")
#    processed_quantity.make_stacked_plot_avail_vs_demand("Electricity", "FTdiesel", "SMRCCS_H_BEC_C_grid_E")

main()

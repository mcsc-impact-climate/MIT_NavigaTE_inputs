"""
Date: Aug 6, 2024
Author: danikam
Purpose: Makes validation plots for csv files produced by make_output_csvs.py
"""

from common_tools import get_top_dir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import os
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon
import matplotlib.colors as mcolors

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
    "bulk_carrier_ice": "Bulk Carrier (ICE)",
    "container_ice": "Container (ICE)",
    "tanker_ice": "Tanker (ICE)",
    "gas_carrier_ice": "Gas Carrier (ICE)"
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
    "SaudiArabia": "Saudi Arabia",
    "UAE": "United Arab Emirates",
    "USA": "United States of America",
    "WestAustralia": "West Australia",
}

region_label_mapping = {
    "SaudiArabia": "Saudi Arabia",
    "WestAustralia": "West Australia",
}

result_components = {
    "TotalCost": ["TotalCAPEX", "TotalFuelOPEX", "TotalExcludingFuelOPEX"],
    "TotalEquivalentWTW": ["TotalEquivalentTTW", "TotalEquivalentWTT"]
}

pathway_type_colors = {
    "electro_renew": "limegreen",
    "electro_grid": "darkgreen",
    "blue": "blue",
    "grey": "grey",
}

pathway_type_labels = {
    "electro_renew": "Electro (renewables)",
    "electro_grid": "Electro (grid)",
    "blue": "Blue",
    "grey": "Grey",
}

# Global string representing the absolute path to the top level of the repo
top_dir = get_top_dir()

def read_pathway_labels(top_dir):
    """
    Reads in the label and description of each fuel production pathway from an info file.

    Parameters
    ----------
    top_dir: str
        Absolute path to the top level of the repo

    Returns
    -------
    pathway_labels_df : pandas.DataFrame
        Dataframe containing the label and description of each fuel production pathway.
    """
    pathway_labels_df = pd.read_csv(f"{top_dir}/info_files/pathway_info.csv").set_index("Pathway Name")
    return pathway_labels_df

# Global dataframe with labels for each fuel production pathway
pathway_labels_df = read_pathway_labels(top_dir)

def get_pathway_label(pathway, pathway_labels_df = pathway_labels_df):
    """
    Returns the label in the row of the pathway_labels_df corresponding to the given pathway.
    
    Parameters
    ----------
    pathway_labels_df: pandas DataFrame
        Dataframe containing labels and descriptions for each fuel production pathway
        
    pathway : str
        String identifier for the pathway of interest

    Returns
    -------
    pathway_label : str
        User-friendly label corresponding to the given pathway ID.
    """
    return pathway_labels_df.loc[pathway, 'Label']

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
    info_df = pd.read_csv(f"{top_dir}/info_files/quantity_info.csv").set_index("Quantity")
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
    region_names_with_spaces = [re.sub(r'(?<!^)(?<![A-Z])(?=[A-Z])', ' ', name) for name in region_names_camel_case]
    return region_names_with_spaces
    
def get_country_label(country):
    """
    Gets the name for the country to be used in plot labels, if the country is included in the region_label_mapping dict.
    
    Parameters
    ----------
    country : str
        Name of the country as specified in the processed csv file

    Returns
    -------
    country_label : str
        Name of the country to be used for plotting
    """
    if country in region_label_mapping:
        country_label = region_label_mapping[country]
    else:
        country_label = country
        
    return country_label
    
def get_filename_info(filepath, identifier, pattern = "{fuel}-{pathway_type}-{pathway}-{quantity}-{modifier}.csv"):
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
    if not identifier in pattern:
        raise Exception(f"Error: identifier {identifier} not found in provided pattern {pattern}")
    
    filename = filepath.split("/")[-1]
    
    result = parse(pattern, filename)
    
    if result is None:
        raise Exception(f"Error: Filename {filename} does not match provided pattern {pattern}")
    
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
    
def find_unique_identifiers(directory, identifier, substring = "", pattern = "{fuel}-{pathway_type}-{pathway}-{quantity}-{modifier}.csv"):
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
    if not identifier in pattern:
        raise Exception(f"Error: identifier {identifier} not found in provided pattern {pattern}")
    
    filenames_matching_substring = find_files_starting_with_substring(directory, substring)
    unique_identifier_values = []
    for filename in filenames_matching_substring:
        identifier_value = get_filename_info(filename, identifier, pattern)
        
        if not identifier_value in unique_identifier_values:
            unique_identifier_values.append(identifier_value)
            
    return unique_identifier_values
    
def find_unique_identifier_pairs(directory, identifier1, identifier2, substring = "", pattern = "{fuel}-{pathway_type}-{pathway}-{quantity}-{modifier}.csv"):
    """
    Finds all unique pairs of values for a given identifier in a pattern within filenames in a directory containing the given substring
    
    Parameters:
    ----------
    directory : str
        The path to the directory to search
    pattern : str
        Pattern containing the given identifier
    identifier1 : str
        First identifier to find in a pair
    identifier2 : str
        Second identifier to find in a pair
    substring : str
        The substring to search for in the filenames.

    Returns:
    -------
    unique_identifier_dicts: list of dict
        A list of unique dictionaries containing pairs of values for the given identifiers in strings containing the given substring.
    """
    
    # Check that the identifiers are included in the provided pattern
    if identifier1 not in pattern:
        raise Exception(f"Error: identifier {identifier1} not found in provided pattern {pattern}")
        
    if identifier2 not in pattern:
        raise Exception(f"Error: identifier {identifier2} not found in provided pattern {pattern}")
    
    filenames_matching_substring = find_files_starting_with_substring(directory, substring)
    unique_identifier_dicts = []
    
    for filename in filenames_matching_substring:
        identifier1_value = get_filename_info(filename, identifier1, pattern)
        identifier2_value = get_filename_info(filename, identifier2, pattern)
        
        # Create a dictionary of the two identifier values
        identifier_dict = {identifier1: identifier1_value, identifier2: identifier2_value}
        
        # Add the dictionary to the list if it is unique
        if identifier_dict not in unique_identifier_dicts:
            unique_identifier_dicts.append(identifier_dict)
            
    return unique_identifier_dicts
    
def create_directory_if_not_exists(directory_path):
    """
    Creates a directory if it doesn't already exist.

    Parameters:
    ----------
    directory_path : str
        The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
        
def add_west_australia(world):
    """
    Adds a custom polygon representing West Australia to the world GeoDataFrame.
    
    This function manually defines a polygon that approximates the region of West Australia
    and adds it to the given world GeoDataFrame. The polygon is created using the shapely library
    and is added as a new row in the GeoDataFrame with the name 'West Australia'.
    
    Parameters
    ----------
    world : geopandas.GeoDataFrame
        The GeoDataFrame containing the world map with country boundaries.
    
    Returns
    -------
    geopandas.GeoDataFrame
        The updated GeoDataFrame that includes the custom region 'West Australia'.
    """
    
    # Check if 'West Australia' is already in the 'NAME' column
    if 'West Australia' in world['NAME'].values:
        return world
    
    # Load detailed shapefile for Australia that includes state/territory boundaries
    url = "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets/georef-australia-state@public/exports/geojson"
    australia_states = gpd.read_file(url)
    
    # Filter the GeoDataFrame to only include Western Australia
    west_australia = australia_states[australia_states['ste_iso3166_code'] == 'WA'].copy()

    # Rename 'WA' to 'West Australia' in the 'NAME' column using .loc[]
    west_australia.loc[west_australia['ste_iso3166_code'] == 'WA', 'NAME'] = 'West Australia'

    # Concatenate the West Australia GeoDataFrame with the world GeoDataFrame
    world_with_west_australia = pd.concat([world, west_australia], ignore_index=True)
    
    return world_with_west_australia
    
def get_custom_tab20_without_blue():
    # Define the tab20 colormap
    tab20 = plt.get_cmap('tab20')

    # Manually extract colors from tab20, explicitly excluding the blue and cyan shades
    custom_colors = [
        tab20(2),  # Orange
        tab20(4),  # Green
        tab20(6),  # Red
        tab20(8),  # Purple
        tab20(10), # Brown
        tab20(12), # Pink
        tab20(14), # Gray
        tab20(16), # Olive
        tab20(3),  # Light orange
        tab20(5),  # Light green
        tab20(7),  # Light red
        tab20(9),  # Light purple
        tab20(11), # Light brown
        tab20(13), # Light pink
        tab20(15), # Light gray
        tab20(17)  # Light olive
    ]
    
    return mcolors.ListedColormap(custom_colors)
    
def generate_blue_shades(num_shades):
    """
    Generates a list of blue shades ranging from light to dark.

    Parameters
    ----------
    num_shades : int
        The number of blue shades to generate.

    Returns
    -------
    blue_shades : list of str
        A list of blue shades in hex format, ranging from light to dark.
    """
    # Define the start and end colors (light blue to dark blue)
    light_blue = mcolors.to_rgba('#add8e6')  # Light blue
    dark_blue = mcolors.to_rgba('#00008b')   # Dark blue

    # Create a list of colors by interpolating between light blue and dark blue
    blue_shades = [
        mcolors.to_hex((light_blue[0] * (1 - i/(num_shades-1)) + dark_blue[0] * (i/(num_shades-1)),
                        light_blue[1] * (1 - i/(num_shades-1)) + dark_blue[1] * (i/(num_shades-1)),
                        light_blue[2] * (1 - i/(num_shades-1)) + dark_blue[2] * (i/(num_shades-1)),
                        1.0))
        for i in range(num_shades)
    ]

    return blue_shades
    
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
    
def get_quantity_label(quantity, quantity_info_df = quantity_info_df):
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
    
def get_units(quantity, modifier, quantity_info_df = quantity_info_df):
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
    
    # Modify the denominator if needed based on the modifier
    modifier_denom_dict = {
        "vessel": "year",
        "fleet": "year",
        "per_mile": "nm",
        "per_tonne_mile": "tonne-nm"
    }
    
    denom_units = modifier_denom_dict[modifier]
    
    return f"{base_units} / {denom_units}"
    
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
            
    pathway_type : str
        Color associated with the given production pathway
            * grey: From fossil sources
            * blue: From fossil sources coupled with carbon capture and storage (CCS)
            * electro_renew: From electrolytic hydrogen powered by renewables
            * electro_grid: From electrolytic hydrogen powered by the grid
    
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
        
    get_units(self):
        Collects the units for the given quantity and modifier
        
    make_hist_by_region(self, vessel_type):
        Plots a stacked histogram of either vessel sizes (if a vessel type is provided as input) or vessel types (if "all" is provided).
        
    make_all_hists_by_region(self):
        Plots all stacked histograms for the available vessel types
    """
    
    def __init__(self, quantity, modifier, fuel, pathway_type, pathway, results_dir = RESULTS_DIR):
        self.quantity = quantity
        self.modifier = modifier
        self.fuel = fuel
        self.pathway_type = pathway_type
        self.pathway = pathway
        self.pathway_label = get_pathway_label(pathway, pathway_labels_df)
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
        return f"{top_dir}/{self.results_dir}/{self.fuel}-{self.pathway_type}-{self.pathway}-{self.quantity}-{self.modifier}.csv"
        
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
        
    def make_hist_by_region(self, vessel_type):
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
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Separate result_df into rows with vs. without '_' (where '_' indicates it's one of several individual estimates for a given region)
        result_df_region_av = self.result_df[~self.result_df.index.str.contains('_')]
        result_df_region_individual = self.result_df[self.result_df.index.str.contains('_')]
        
        # Plot each region with vessel types stacked
        stack_vessel_types = False
        stack_vessel_sizes = False
        if vessel_type == "all":
            legend_title = "Vessel Type"
            
            # Don't stack vessel types if quantities are normalized per mile or per tonne-mile
            if self.modifier == "per_mile" or self.modifier == "per_tonne_mile":
                result_df_region_av[f"fleet_{self.fuel}"].plot(kind='barh', stacked=False, ax=ax)
            else:
                stack_by = [f"{vessel_type}_{self.fuel}" for vessel_type in vessel_types]
                result_df_region_av[stack_by].plot(kind='barh', stacked=True, ax=ax)
                stack_vessel_types = True
            
            # Add individual region estimates as unfilled circles
            for idx, row in result_df_region_individual.iterrows():
                region = idx.split('_')[0]
                if region in result_df_region_av.index:
                    # Use the same x-location as the corresponding bar
                    x = result_df_region_av.index.get_loc(region)

                    # Plot individual estimates as unfilled circles
                    ax.scatter([row[f"fleet_{self.fuel}"]], [x], marker="D", color='black', s=100)
            
            # Update legend labels if plotting stacked vessels
            if stack_vessel_types:
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [vessel_type_title[label.replace(f"_{self.fuel}", "")] for label in labels]
            
            # Construct the filename to save to
            filename_save = f"{self.fuel}-{self.pathway_type}-{self.pathway}-{self.quantity}-{self.modifier}-hist_by_region_allvessels"

        # Plot each region with vessel sizes stacked
        elif vessel_type in vessel_types:
            legend_title = "Vessel Size"
            vessel_type_label = vessel_type_title[vessel_type]
            ax.text(1.05, 0.55, f"Vessel Type: {vessel_type_label}", transform=ax.transAxes, fontsize=20, va='top', ha='left')
            
            if self.modifier == "per_mile" or self.modifier == "per_tonne_mile":
                result_df_region_av[f"{vessel_type}_{self.fuel}"].plot(kind='barh', stacked=False, ax=ax)
                
            else:
                stack_by = [f"{vessel_size}_{self.fuel}" for vessel_size in vessel_sizes[vessel_type]]
                result_df_region_av[stack_by].plot(kind='barh', stacked=True, ax=ax)
                stack_vessel_sizes = True
            
            # Add individual region estimates as unfilled circles
            for idx, row in result_df_region_individual.iterrows():
                region = idx.split('_')[0]
                if region in result_df_region_av.index:
                    # Use the same x-location as the corresponding bar
                    x = result_df_region_av.index.get_loc(region)

                    # Plot individual estimates as unfilled circles
                    ax.scatter([row[f"{vessel_type}_{self.fuel}"]], [x], marker="D", color='black', s=100)
            
            # Update legend labels if plotting stacked vessel sizes
            if stack_vessel_sizes:
                handles, labels = ax.get_legend_handles_labels()
                new_labels = [vessel_size_title[label.replace(f"_{self.fuel}", "")] for label in labels]
            
            # Construct the filename to save to
            filename_save = f"{self.fuel}-{self.pathway_type}-{self.pathway}-{self.quantity}-{self.modifier}-hist_by_region_{vessel_type}"
            
        else:
            print(f"Vessel type should be one of: {vessel_types}. Returning from hist_by_region without plotting.")
            return
        
        # Add text to indicate the details of what's being plotted
        ax.text(1.05, 0.5, f"Fuel: {self.fuel}", transform=ax.transAxes, fontsize=20, va='top', ha='left')
        ax.text(1.05, 0.45, f"Color: {self.pathway_type}", transform=ax.transAxes, fontsize=20, va='top', ha='left')
        ax.text(1.05, 0.4, f"Pathway: {self.pathway_label}", transform=ax.transAxes, fontsize=20, va='top', ha='left')
        
        # Plot styling common to both cases
        ax.set_xlabel(f"{self.label} ({self.units})", fontsize=22)
        ax.set_yticks(range(len(result_df_region_av)))
        #plt.xticks(rotation=45)
        ax.set_yticklabels(make_region_labels(result_df_region_av.index))
        if stack_vessel_types or stack_vessel_sizes:
            ax.legend(handles, new_labels, title=legend_title, fontsize=20, title_fontsize=22, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}_{self.pathway_type}_{self.pathway}")
        plt.savefig(f"{top_dir}/plots/{self.fuel}_{self.pathway_type}_{self.pathway}/{filename_save}.png", dpi=200)
        #plt.savefig(f"{top_dir}/plots/{self.fuel}_{self.pathway_type}_{self.pathway}_hists_by_region/{filename_save}.pdf")
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
            self.result_df["NAME"] = self.result_df.index.map(lambda x: region_name_mapping.get(x, x))
            
        
    def map_by_region(self, vessel_type="all", vessel_size="all"):
        """
        Maps the quantity geospatially by region, overlaid on a map of the world
        
        Parameters
        ----------
        column : str
            Name of the column in the processed csv file to map by region (defaults to "fleet_{self.fuel}").

        Returns
        -------
        None
        """
        
        # If vessel option is provided as "all", plot the quantity for the full fleet
        if vessel_type == "all":
            column = f"fleet_{self.fuel}"
        
        # If a vessel_type option other than "all" is provided and vessel_size is set to "all", plot the given quantity for all vessel sizes of the given vessel type
        else:
            # Ensure that a valid vessel type was provided
            vessel_types_list = vessel_sizes.keys()
            
            if not vessel_type in vessel_sizes_list:
                Exception(f"Error: Vessel type {vessel_type} not recognized. Acceptable types: {vessel_types_list}")
            
            # If the vessel size is provided as "all", plot the quantity for all sizes of the given vessel type
            if vessel_size == "all":
                column = f"{vessel_type}_{self.fuel}"
            
            # If a vessel size other than "all" is provided, plot the quantity for the given vessel type and size
            else:
                # Ensure that a valid vessel size was provided
                vessel_sizes_list = vessel_sizes[vessel_types_list].keys()
                if not vessel_size in vessel_sizes_list:
                    Exception(f"Error: Vessel size {vessel_size} not recognized. Acceptable sizes: {vessel_sizes_list}")
                
                column = f"{vessel_size}_{self.fuel}"

        # Load a base world map from geopandas
        url = "https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/ne_110m_admin_0_countries.geojson"
        world = gpd.read_file(url)
        
        # Add West Australia to the world geojson
        world = add_west_australia(world)
        
        # Add a column "NAME" to self.results_df with region names to match the geojson world file, if needed
        self.add_region_names()
                
        # Create a figure and axis with appropriate size
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Merge the result_df with the world geodataframe based on the column "NAME" with region names
        merged = world.merge(self.result_df, on='NAME', how='left')
                
        # Plot the base map
        world.plot(ax=ax, color='lightgrey', edgecolor='black')

        # Plot the regions with data, using a colormap to represent the quantity
        merged.plot(column=column, cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8', legend=False)

        # Set the title and labels
        ax.set_xticks([])
        ax.set_yticks([])

        # Create a horizontal colorbar with appropriate formatting
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.2)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=merged[column].min(), vmax=merged[column].max()))
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label(f"{self.label} ({self.units})", fontsize=20)
        
        # Save the plot
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}_{self.pathway_type}_{self.pathway}")
        plt.savefig(f"{top_dir}/plots/{self.fuel}_{self.pathway_type}_{self.pathway}/{self.quantity}_{self.modifier}_map_by_region.png", dpi=200)
        plt.close()
        
    
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
        Color associated with the given production pathway
            * grey: From fossil sources
            * blue: From fossil sources coupled with carbon capture and storage (CCS)
            * electro_grid: From electrolytic hydrogen powered by the grid
            * electro_renew: From electrolytic hydrogen powered by renewables
    
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
    
    def __init__(self, fuel, pathway_type, pathway, results_dir = RESULTS_DIR):
        self.fuel = fuel
        self.pathway_type = pathway_type
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
        quantities = find_unique_identifiers(self.results_dir, "quantity", f"{self.fuel}-{self.pathway_type}-{self.pathway}")
        
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
        modifiers = find_unique_identifiers(self.results_dir, "modifier", f"{self.fuel}-{self.pathway_type}-{self.pathway}-{sample_quantity}")
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
            modifiers = find_unique_identifiers(self.results_dir, "modifier", f"{self.fuel}-{self.pathway_type}-{self.pathway}-{quantity}")
            for modifier in modifiers:
                ProcessedQuantities[quantity][modifier] = ProcessedQuantity(quantity, modifier, self.fuel, self.pathway_type, self.pathway)
        
        return ProcessedQuantities
        
    def apply_to_all_quantities(self, method_name, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "per_mile", "vessel", "fleet"]):
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
                raise Exception(f"Error: Provided quantity '{quantity}' is not available in self.ProcessedQuantities. \n\nAvailable quantities: {self.quantities}.")
            
            # Handle the situation where the user wants to apply the method to all available modifiers
            all_available_modifiers = find_unique_identifiers(self.results_dir, "modifier", f"{self.fuel}-{self.pathway_type}-{self.pathway}-{quantity}")
            modifiers = all_available_modifiers
                
            for modifier in modifiers:
                if modifier not in all_available_modifiers:
                    raise Exception(f"Error: Provided modifier '{modifier}' is not available in self.ProcessedQuantities. \n\nAvailable modifiers: {self.modifiers}.")
                
                # Get the instance of ProcessedQuantity
                processed_quantity_instance = self.ProcessedQuantities[quantity][modifier]
                
                # Dynamically get the method from the instance and call it
                method_to_call = getattr(processed_quantity_instance, method_name)
                method_to_call()
                
    def get_country_average_results(self, quantities, modifier):
        """
        Collects results for the given quantities and modifier, averaged over all countries.
        
        Parameters
        ----------
        quantities : list of str
            List of quantities to make hists of. If "all" is provided, it will make hists of all quantities in the ProcessedQuantities dictionary
            
        modifier : str
            Modifier to use in evaluating the country average.
        
        Returns
        -------
        country_av_results_dict : Dictionary of floats
            Dictionary containing the results for the given quantities and modifier
        """
        country_av_results_dict = {}
        
        column_name = f"fleet_{self.fuel}"
        for quantity in quantities:
            processed_quantity = self.ProcessedQuantities[quantity][modifier]
            processed_quantity_av = processed_quantity.result_df.loc["Global Average", column_name]
            country_av_results_dict[quantity] = processed_quantity_av
            
        return country_av_results_dict
        
    def get_all_country_results(self, quantity, modifier):
        """
        Collects results for the given quantity and modifier for all countries.
        
        Parameters
        ----------
        quantity : str
            Quantity to use in evaluating the country average.
            
        modifier : str
            Modifier to use in evaluating the country average.
        
        Returns
        -------
        individual_country_results_dict : Dictionary of floats
            Dictionary containing the results for each country.
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
            country_name = entry.split("_")[0]
            if country_name not in countries_with_multiple_entries:
                countries_with_multiple_entries.append(country_name)
        
        individual_country_results_dict = {}
        multiple_country_results_dict = {}
        column_name = f"fleet_{self.fuel}"
        for country in countries_av:
            if country != "Global Average":
                country_label = get_country_label(country)
                individual_country_results_dict[country_label] = result_df_region_av.loc[country, column_name]
            if country in countries_with_multiple_entries:
                for entry in countries_individual:
                    entry_elements = entry.split("_")
                    entry_country = entry_elements[0]
                    entry_number = entry_elements[1]
                    if entry_country == country:
                        country_label = get_country_label(entry_country)
                        multiple_country_results_dict[f"{country_label} ({entry_number})"] = result_df_region_individual.loc[entry, column_name]
                        
        return individual_country_results_dict, multiple_country_results_dict
                        
        
    def make_all_hists_by_region(self, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "per_mile", "vessel", "fleet"]):
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
                
    def map_all_by_region(self, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "per_mile", "vessel", "fleet"]):
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

    def __init__(self, fuel, results_dir = RESULTS_DIR):
        self.fuel = fuel
        self.results_dir = results_dir
        self.pathways_with_type = self.get_pathways_with_type()
        self.pathways = self.get_pathways_no_type()
        self.ProcessedPathways = self.get_processed_pathways()
        self.type_pathway_dict = self.organize_pathways_by_type()

    def get_pathways_with_type(self):
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
        pathways = find_unique_identifier_pairs(self.results_dir, "pathway", "pathway_type", f"{self.fuel}")
    
        return pathways
        
    def get_pathways_no_type(self):
        """
        Collects the names of all pathways contained in processed csv files for the given fuel
        
        Parameters
        ----------
        None

        Returns
        -------
        pathways : list of str
            List of all pathways evaluated for the fuel
        """
        pathways = []
        for pathway_dict in self.pathways_with_type:
            pathways.append(pathway_dict["pathway"])
            
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
        for pathway_dict in self.pathways_with_type:
            pathway = pathway_dict["pathway"]
            pathway_type = pathway_dict["pathway_type"]
            ProcessedPathways[pathway] = ProcessedPathway(self.fuel, pathway_type, pathway)
                    
        return ProcessedPathways
        
    def organize_pathways_by_type(self):
        """
        Organizes the pathways according to their type (electro_grid, electro_renew, blue, grey)
        
        Parameters
        ----------
        None

        Returns
        -------
        type_pathway_dict : dictionary of lists of str
            Dictionary containing a list of pathways corresponding to each type
        """
        type_pathway_dict = {
            "electro_grid": [],
            "electro_renew": [],
            "blue": [],
            "grey": [],
        }
        
        for pathway_dict in self.pathways_with_type:
            pathway_name = pathway_dict["pathway"]
            pathway_type = pathway_dict["pathway_type"]
            type_pathway_dict[pathway_type].append(pathway_name)
            
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
            
            sample_processed_quantity = processed_quantities[sample_quantity][sample_modifier]
            result_df = sample_processed_quantity.result_df
            
            countries = result_df[~result_df.index.str.contains("_")].index
            for country in countries:
                country_label = get_country_label(country)
                if country_label not in all_countries and not country_label == "Global Average":
                    all_countries.append(country_label)
                    
        return all_countries
            
        
    def make_stacked_hist(self, quantity, modifier, sub_quantities=[]):
        """
        Makes a histogram of the given quantity with respect to the available fuel production pathways.

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
        quantity_label = get_quantity_label(quantity)
        quantity_units = get_units(quantity, modifier)

        if sub_quantities == []:
            sub_quantities = [quantity]

        num_pathways = len(self.pathways)
        fig_height = max(6, num_pathways * 0.9)  # Adjust this factor as needed

        fig, ax = plt.subplots(figsize=(15, fig_height))

        # Create an empty dictionary to hold the cumulative values for stacking
        cumulative_values = {}

        blues = generate_blue_shades(len(sub_quantities))
        
        # Get all individual countries with processed data for the given fuel
        countries = self.get_all_countries()
        country_colors = assign_colors_to_strings(countries)
        
        countries_labelled = []
        scatter_handles = []  # To collect scatter plot legend handles
        scatter_labels = []  # To collect scatter plot legend labels
        bar_handles = []  # To collect bar plot legend handles
        bar_labels = []  # To collect bar plot legend labels
        
        def make_bar(pathway, pathway_type, pathway_name, pathway_label):
            """
            Plots a single bar for a given pathway
            
            Parameters
            ----------
            pathway : ProcessedPathway
                ProcessedPathway class instance containing the info to plot

            pathway_type : str
                Pathway type (electro_grid, electro_renew, blue, grey)

            pathway_name : str
                Name of the pathway
                
            pathway_label : str
                Name of the pathway to use for labels when plotting

            Returns
            -------
            None
            """
            # Initialize cumulative value for this pathway
            if pathway_name not in cumulative_values:
                cumulative_values[pathway_name] = 0

            # Collect the country average results
            country_average_results = pathway.get_country_average_results(sub_quantities, modifier)
            
            # Collect the individual country results
            all_country_results, multiple_country_results = pathway.get_all_country_results(quantity, modifier)

            # Get the values for each sub_quantity and stack them
            for i, sub_quantity in enumerate(sub_quantities):
                value = country_average_results.get(sub_quantity, 0)
                bar = ax.barh(pathway_label, value, left=cumulative_values[pathway_name], label=get_quantity_label(sub_quantity) if i_pathway == 0 else "", color=blues[i])
                cumulative_values[pathway_name] += value
                
                # Add the bar handles and labels only once
                if i_pathway == 0:
                    bar_handles.append(bar[0])
                    bar_labels.append(get_quantity_label(sub_quantity))
                
                # Plot the individual country results as a scatter plot
                if all_country_results:
                    for country in all_country_results:
                        if "Global" in country:
                            continue
                        scatter = ax.scatter(all_country_results[country], pathway_label, color=country_colors[country], s=100, marker="D", label=get_country_label(country) if country not in countries_labelled else "")
                        
                        # Add the country to the list of countries that have been labeled so it only appears in the legend once
                        if country not in countries_labelled:
                            countries_labelled.append(country)
                            scatter_handles.append(scatter)
                            scatter_labels.append(get_country_label(country))

            # Set the y-axis label color to match the pathway type
            y_labels = ax.get_yticklabels()
            if i_pathway < len(y_labels):
                y_labels[i_pathway].set_color(pathway_type_colors[pathway_type])
                y_labels[i_pathway].set_fontweight('bold')

        # Loop through each color and pathway
        i_pathway = 0
        for pathway_type in self.type_pathway_dict:
            for pathway_name in self.type_pathway_dict[pathway_type]:
                pathway = self.ProcessedPathways[pathway_name]
                pathway_label = get_pathway_label(pathway_name, pathway_labels_df)

                make_bar(pathway, pathway_type, pathway_name, pathway_label)
        
                i_pathway += 1
                    
        # Add a bar for LSFO fossil for comparison
        
        lsfo_pathway = ProcessedPathway("lsfo", "grey", "fossil")
        make_bar(lsfo_pathway, "grey", "lsfo", "LSFO (fossil)")
        plt.axvline(lsfo_pathway.get_country_average_results([quantity], modifier)[quantity], linewidth=3, linestyle="--", color="black")

        # Add labels and title
        ax.set_xlabel(f"{quantity_label} ({quantity_units})", fontsize=20)
        ax.set_title(f"Fuel: {self.fuel}", fontsize=24)
        
        # Add a legend for the stacked bar components (sub-quantities)
        if bar_handles:
            legend1 = ax.legend(bar_handles, bar_labels, fontsize=16, title="Components", title_fontsize=20, bbox_to_anchor=(1.01, 1.05), loc='upper left', borderaxespad=0.)

        # Add a separate legend for countries
        if scatter_handles:
            legend2 = ax.legend(scatter_handles, scatter_labels, fontsize=16, title="Countries", title_fontsize=20, bbox_to_anchor=(1.01, 0.35), loc='center left', borderaxespad=0.)
        
        # Add the bar component legend back after the country legend if both legends are present
        if bar_handles and scatter_handles:
            ax.add_artist(legend1)

        plt.subplots_adjust(left=0.2, right=0.75)
        #plt.tight_layout()

        # Construct the filename to save to
        filename_save = f"{self.fuel}-{quantity}-{modifier}-pathway_hist"

        # Save the figure
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}")
        plt.savefig(f"{top_dir}/plots/{self.fuel}/{filename_save}.png", dpi=200)
        
        
    def make_all_stacked_hists(self, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "per_mile", "vessel", "fleet"]):
        """
        Plot a stacked histogram for the given quantities and modifiers with respect to the pathway and country.

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
            if not quantity in all_avalable_quantities:
                raise Exception(f"Error: Provided quantity '{quantity}' is not available in self.ProcessedQuantities. \n\nAvailable quanities: {all_avalable_quantities}.")
                
            if quantity in result_components:
                sub_quantities = result_components[quantity]
            else:
                sub_quantities = []

            # Handle the situation where the user wants to apply the method to all available modifiers
            all_available_modifiers = find_unique_identifiers(self.results_dir, "modifier", f"{self.fuel}-{sample_processed_pathway.pathway_type}-{sample_processed_pathway.pathway}-{quantity}")
            if modifiers == "all":
                modifiers = all_available_modifiers
                
            for modifier in modifiers:
                if modifier not in all_available_modifiers:
                    raise Exception(f"Error: Provided modifier '{modifier}' is not available in self.ProcessedQuantities. \n\nAvailable modifiers: {self.modifiers}.")
                self.make_stacked_hist(quantity, modifier, sub_quantities)
                
    def apply_to_all_pathways(self, method_name, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "per_mile", "vessel", "fleet"]):
        """
        Applies the provided method to all available pathways
        
        Parameters
        ----------
        quantities : str or list of str
            List of quantities to make hists of. If "all" is provided, it will make hists of all quantities in the ProcessedQuantities dictionary
            
        modifiers : str
            List of modifiers to include in the hists. If "all" is provided, it will make hists with all modifiers in the ProcessedQuantities dictionary
            
        method : Method of the ProcessedPathway class
        
        Returns
        -------
        None
        """
        
        for pathway in self.ProcessedPathways:
            processed_pathway = self.ProcessedPathways[pathway]
            # Dynamically get the method from the instance and call it
            method_to_call = getattr(processed_pathway, method_name)
            method_to_call(quantities=quantities, modifiers=modifiers)
                
    def make_all_hists_by_region(self, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "per_mile", "vessel", "fleet"]):
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
        
    def map_all_by_region(self, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "per_mile", "vessel", "fleet"]):
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
        
def structure_results_fuels_types(quantity, modifier, fuels=["ammonia", "hydrogen", "lsfo"]):
    """
    Structures results from all pathways for the given fuels into pathway colors.
    
    Parameters
    ----------
    quantity : str
        Quantity to plot
        
    modifiers : str
        Modifier for the quantity
        
    fuels : list of str
        List of fuels to plot

    Returns
    -------
    None
    """
    
    results_fuels_types = {}
    for fuel in fuels:
        results_fuels_types[fuel] = {}
        processed_fuel = ProcessedFuel(fuel)
        type_pathway_dict = processed_fuel.type_pathway_dict
        for pathway_type in type_pathway_dict:
            results_fuels_types[fuel][pathway_type] = []
            for pathway in type_pathway_dict[pathway_type]:
                processed_quantity = ProcessedQuantity(quantity, modifier, fuel, pathway_type, pathway)
                result_df = processed_quantity.result_df
                
                for country in result_df.index:
                    if not "_" in country and not country != "Global Average":
                        results_fuels_types[fuel][pathway_type].append(result_df.loc[country, f"fleet_{fuel}"])
    return results_fuels_types

def plot_scatter_violin(data, quantity, modifier, plot_size=(12, 6)):
    """
    Plot the data from a dictionary as horizontal scatterplots with optional overlayed violin plots.

    Parameters
    ----------
    data: dict
        The input data, a dictionary where keys are categories (e.g., 'hydrogen', 'ammonia')
        and values are dictionaries with subcategories (e.g., 'green', 'blue', 'grey') containing lists of float values.
    plot_size: tuple, optional
        Size of the plot (default: (12, 6)).
        
    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=plot_size)

    y_base = 0
    y_tick_positions = []  # This will store the y positions for the tick labels on the left side
    y_tick_labels = []     # This will store the corresponding labels for the ticks on the left side
    
    y_tick_positions_right = []  # This will store the y positions for the tick labels on the right side
    y_tick_labels_right = []     # This will store the corresponding labels for the ticks on the right side
    y_tick_colors_right = []     # This will store the colors for the right-side tick labels

    n_fuels = len(data.keys())
    i_fuel = 0
    for fuel in data.keys():
        n_fuel_types = 0  # Keep track of how many fuel types are present for this fuel

        for i, fuel_type in enumerate(pathway_type_colors.keys()):
            if fuel_type in data[fuel]:
                values = data[fuel][fuel_type]
                y_value = y_base + n_fuel_types

                # Plot the violin plot using Matplotlib's violinplot directly
                if len(values) > 1:
                    parts = ax.violinplot(values, positions=[y_value], vert=False, showmeans=False, showmedians=False, showextrema=False)
                    for pc in parts['bodies']:
                        pc.set_facecolor(pathway_type_colors[fuel_type])
                        pc.set_alpha(0.3)

                # Plot the scatter plot
                ax.scatter(values, [y_value] * len(values), color=pathway_type_colors[fuel_type], edgecolor='black')

                # Add right-side labels
                y_tick_positions_right.append(y_value)
                y_tick_labels_right.append(pathway_type_labels[fuel_type])
                y_tick_colors_right.append(pathway_type_colors[fuel_type])

                n_fuel_types += 1

        # Store the position of the tick for this fuel, centered among its subcategories on the left side
        y_tick_positions.append(y_base + (n_fuel_types - 1) / 2)
        y_tick_labels.append(fuel)

        # Update y_base to the next starting position for the next fuel
        y_base += n_fuel_types + 1  # Add 1 for spacing between different fuels
        
        # Add a horizontal line to separate different fuels
        i_fuel += 1
        if not i_fuel == n_fuels:
            ax.axhline(y=y_base-1, color='black', linewidth=2)
            
    # Add a vertical line for LSFO
    ax.axvline(data["lsfo"]["grey"], ls="--", color="grey")

    # Set y-ticks with calculated positions and corresponding labels on the left side
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels)
    
    # Set the labels and colors for the right-side y-axis
    ax_right = ax.twinx()
    ax_right.set_yticks(y_tick_positions_right)
    ax_right.set_yticklabels(y_tick_labels_right)

    # Set identical limits for both y-axes
    ax.set_ylim(ax.get_ylim())  # Lock the left axis limits
    ax_right.set_ylim(ax.get_ylim())  # Set the right axis limits to match the left

    # Color the right-side y-axis labels
    for tick_label, color in zip(ax_right.get_yticklabels(), y_tick_colors_right):
        tick_label.set_color(color)
    
    quantity_label = get_quantity_label(quantity)
    units = get_units(quantity, modifier)
    ax.set_xlabel(f"{quantity_label} ({units})", fontsize=22)
    plt.subplots_adjust(left=0.2, right=0.75)
    plt.tight_layout()
    create_directory_if_not_exists(f"{top_dir}/plots/scatter_violin")
    plt.savefig(f"{top_dir}/plots/scatter_violin/{quantity}-{modifier}.png", dpi=200)

def main():
    # Loop through all fuels of interest
    for fuel in ["hydrogen", "ammonia", "lsfo"]:
        processed_fuel = ProcessedFuel(fuel)

        # Make validation plots for each fuel, pathway and quantity
        processed_fuel.make_all_hists_by_region()
        processed_fuel.map_all_by_region()
        processed_fuel.make_all_stacked_hists()
    
    for quantity in ["TotalCost", "TotalEquivalentWTW"]:
        for modifier in ["vessel", "fleet", "per_mile", "per_tonne_mile"]:
            structured_results = structure_results_fuels_types(quantity, modifier)
            plot_scatter_violin(structured_results, quantity, modifier)

main()

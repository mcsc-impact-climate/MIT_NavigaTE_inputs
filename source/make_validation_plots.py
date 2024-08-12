"""
Date: Aug 6, 2024
Author: danikam
Purpose: Makes validation plots for csv files produced by make_output_csvs.py
"""

from common_tools import get_top_dir
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
import os
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon

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

def get_pathway_label(pathway, pathway_labels_df):
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
    
def get_filename_info(filepath, identifier, pattern = "{fuel}-{pathway_color}-{pathway}-{quantity}-{modifier}.csv"):
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
    
def find_files_with_substring(directory, substring=""):
    """
    Finds all files within a specified directory (and its subdirectories) that contain a given substring in their filenames.

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
            if substring in file:
                # Add the full file path to the list
                matching_files.append(os.path.join(root, file))

    return matching_files
    
def find_unique_identifiers(directory, identifier, substring = "", pattern = "{fuel}-{pathway_color}-{pathway}-{quantity}-{modifier}.csv"):
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
    
    filenames_matching_substring = find_files_with_substring(directory, substring)
    unique_identifier_values = []
    for filename in filenames_matching_substring:
        identifier_value = get_filename_info(filename, identifier, pattern)
        
        if not identifier_value in unique_identifier_values:
            unique_identifier_values.append(identifier_value)
            
    return unique_identifier_values
    
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
            * grey: From fossil sources
            * blue: From fossil sources coupled with carbon capture and storage (CCS)
            * green: From electrolytic hydrogen
    
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
    
    def __init__(self, quantity, modifier, fuel, pathway_color, pathway, results_dir = RESULTS_DIR):
        self.quantity = quantity
        self.modifier = modifier
        self.fuel = fuel
        self.pathway_color = pathway_color
        self.pathway = pathway
        self.pathway_label = get_pathway_label(pathway, pathway_labels_df)
        self.results_dir = results_dir
        self.result_df = self.read_result()
        self.label = quantity_info_df.loc[self.quantity, "Long Name"]
        self.units = self.get_units()
        
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
        return f"{top_dir}/{self.results_dir}/{self.fuel}-{self.pathway_color}-{self.pathway}-{self.quantity}-{self.modifier}.csv"
        
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
        
    def get_units(self):
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
        
        base_units = quantity_info_df.loc[self.quantity, "Units"]
        
        # Modify the denominator if needed based on the modifier
        modifier_denom_dict = {
            "vessel": "year",
            "fleet": "year",
            "per_mile": "nm",
            "per_tonne_mile": "tonne-nm"
        }
        
        denom_units = modifier_denom_dict[self.modifier]
        
        return f"{base_units} / {denom_units}"
        
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
            filename_save = f"{self.fuel}-{self.pathway_color}-{self.pathway}-{self.quantity}-hist_by_region_allvessels"

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
            filename_save = f"{self.fuel}-{self.pathway_color}-{self.pathway}-{self.quantity}-{self.modifier}-hist_by_region_{vessel_type}"
            
        else:
            print(f"Vessel type should be one of: {vessel_types}. Returning from hist_by_region without plotting.")
            return
        
        # Add text to indicate the details of what's being plotted
        ax.text(1.05, 0.5, f"Fuel: {self.fuel}", transform=ax.transAxes, fontsize=20, va='top', ha='left')
        ax.text(1.05, 0.45, f"Color: {self.pathway_color}", transform=ax.transAxes, fontsize=20, va='top', ha='left')
        ax.text(1.05, 0.4, f"Pathway: {self.pathway_label}", transform=ax.transAxes, fontsize=20, va='top', ha='left')
        
        # Plot styling common to both cases
        ax.set_xlabel(f"{self.label} ({self.units})", fontsize=22)
        ax.set_yticks(range(len(result_df_region_av)))
        #plt.xticks(rotation=45)
        ax.set_yticklabels(make_region_labels(result_df_region_av.index))
        if stack_vessel_types or stack_vessel_sizes:
            ax.legend(handles, new_labels, title=legend_title, fontsize=20, title_fontsize=22, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}_{self.pathway_color}_{self.pathway}")
        plt.savefig(f"{top_dir}/plots/{self.fuel}_{self.pathway_color}_{self.pathway}/{filename_save}.png", dpi=200)
        #plt.savefig(f"{top_dir}/plots/{self.fuel}_{self.pathway_color}_{self.pathway}_hists_by_region/{filename_save}.pdf")
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
            
        
    def map_by_region(self, column=None):
        """
        Maps the quantity geospatially by region, overlaid on a map of the world
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        # If the column to plot isn't specified, set it to the value of the given quantity for the entire fleet
        if column is None:
            column = f"fleet_{self.fuel}"

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
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}_{self.pathway_color}_{self.pathway}")
        plt.savefig(f"{top_dir}/plots/{self.fuel}_{self.pathway_color}_{self.pathway}/{self.quantity}_map_by_region.png", dpi=200)
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
            
    pathway_color : str
        Color associated with the given production pathway
            * grey: From fossil sources
            * blue: From fossil sources coupled with carbon capture and storage (CCS)
            * green: From electrolytic hydrogen
    
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
    """
    
    def __init__(self, fuel, pathway_color, pathway, results_dir = RESULTS_DIR):
        self.fuel = fuel
        self.pathway_color = pathway_color
        self.pathway = pathway
        self.results_dir = results_dir
        self.quantities = self.get_quantities()
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
        quantities = find_unique_identifiers(self.results_dir, "quantity", f"{self.fuel}-{self.pathway_color}-{self.pathway}")
        
        return quantities
        
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
            modifiers = find_unique_identifiers(self.results_dir, "modifier", f"{self.fuel}-{self.pathway_color}-{self.pathway}-{quantity}")
            for modifier in modifiers:
                ProcessedQuantities[quantity][modifier] = ProcessedQuantity(quantity, modifier, self.fuel, self.pathway_color, self.pathway)
        
        return ProcessedQuantities
        
    def make_all_hists_by_region(self, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "fleet"]):
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
        
        # Handle the situation where the user wants to plot all quantities and/or all modifiers
        if quantities == "all":
            quantities = self.ProcessedQuantities
        
        if modifiers == "all":
            modifiers = self.ProcessedQuantities[quantity]
        
        for quantity in quantities:
            if not quantity in self.ProcessedQuantities:
                raise Exception(f"Error: Provided quantity '{quantity}' is not available in self.ProcessedQuantities. \n\nAvailable quanities: {self.quantities}.")
            for modifier in modifiers:
                self.ProcessedQuantities[quantity][modifier].make_all_hists_by_region()
                
    def map_all_by_region(self, quantities=["TotalEquivalentWTW", "TotalCost"], modifiers=["per_tonne_mile", "fleet"]):
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
        
        # Handle the situation where the user wants to plot all quantities and/or all modifiers
        if quantities == "all":
            quantities = self.ProcessedQuantities
        
        if modifiers == "all":
            modifiers = self.ProcessedQuantities[quantity]
        
        for quantity in quantities:
            if not quantity in self.ProcessedQuantities:
                raise Exception(f"Error: Provided quantity '{quantity}' is not available in self.ProcessedQuantities. \n\nAvailable quanities: {self.quantities}.")
            for modifier in modifiers:
                self.ProcessedQuantities[quantity][modifier].map_by_region()
        
#class ProcessedFuel:
#    """
#    A class to contain NavigaTE results for a given fuel, including all its pathways
#
#    Attributes
#    ----------
#    pathway_names : list of str
#        List of pathways with processed data available for the given fuel
#
#    ProcessedPathways : dict of ProcessedPathway objects
#        Dictionary containing a ProcessedPathway class object for each pathway
#    """
#
#    def __init__(self, fuel, results_dir = RESULTS_DIR):
#        self.fuel = fuel
#        self.pathway_names = self.get_pathway_names()
#        self.ProcessedPathways = self.get_ProcessedPathways()
#
#    def get_pathway_names(self):
#        """
#        Collects the names of all pathways contained in processed csv files for the given fuel
#        """
        

def main():
    
    lsfo_pathway_TotalCost_fleet = ProcessedQuantity("TotalCost", "fleet", "ammonia", "electro", "LTE_grid")
    lsfo_pathway_WTW_fleet = ProcessedQuantity("TotalEquivalentWTW", "fleet", "ammonia", "electro", "LTE_grid")
    
    #lsfo_pathway_WTW_fleet.make_hist_by_region("all")
    #lsfo_pathway_WTW_fleet.make_hist_by_region("bulk_carrier_ice")
    lsfo_pathway_WTW_fleet.map_by_region()
    
    
    #print(lsfo_pathway_TotalCost_fleet.result_df)
    #lsfo_pathway_TotalCost_fleet.make_all_hists_by_region()
    #lsfo_pathway_WTW_fleet.make_all_hists_by_region()
    #get_filename_info("/Users/danikamacdonell/Git/MIT_NavigaTE_inputs/processed_results/lsfo-grey-fossil-TotalFuelOPEX-per_mile.csv")
    
#    pathway = ProcessedPathway("ammonia", "blue", "ATR_CCS")
#    pathway.make_all_hists_by_region()
#    pathway.map_all_by_region()

main()

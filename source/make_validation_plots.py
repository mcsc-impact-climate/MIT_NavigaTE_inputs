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
    "gas_carrier_100k_cbm_ice": "100k m^3",
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

def make_country_labels(country_names_camel_case):
    """
    Function to construct country labels with spaces in cases where CamelCase is used to separate different parts of country names.
    Eg. ["SaudiArabia"] --> ["Saudi Arabia"]
    
    Parameters
    ----------
    country_names_camel_case: list of str
        List of country names in CamelCase format

    Returns
    -------
    country_names_with_spaces : list of str
        List of country names with spaces
    """
    # Insert a space before each uppercase letter (except the first one)
    country_names_with_spaces = [re.sub(r'(?<!^)(?<![A-Z])(?=[A-Z])', ' ', name) for name in country_names_camel_case]
    return country_names_with_spaces
    
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
        
    make_hist_by_country(self, vessel_type):
        Plots a stacked histogram of either vessel sizes (if a vessel type is provided as input) or vessel types (if "all" is provided).
        
    make_all_hists_by_country(self):
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
        
    def make_hist_by_country(self, vessel_type):
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
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Separate result_df into rows with vs. without '_' (where '_' indicates it's one of several individual estimates for a given country)
        result_df_country_av = self.result_df[~self.result_df.index.str.contains('_')]
        result_df_country_individual = self.result_df[self.result_df.index.str.contains('_')]
        
        # Plot each country with vessel types stacked
        if vessel_type == "all":
            legend_title = "Vessel Type"
            stack_by = [f"{vessel_type}_{self.fuel}" for vessel_type in vessel_types]
            
            # Only plot stacked histograms for country averages
            result_df_country_av[stack_by].plot(kind='barh', stacked=True, ax=ax)
            
            # Add individual country estimates as unfilled circles
            for idx, row in result_df_country_individual.iterrows():
                country = idx.split('_')[0]
                if country in result_df_country_av.index:
                    # Use the same x-location as the corresponding bar
                    x = result_df_country_av.index.get_loc(country)

                    # Plot individual estimates as unfilled circles
                    ax.scatter([x], [row[f"fleet_{self.fuel}"]], marker="D", color='black', s=100)
            
            # Update legend labels
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [vessel_type_title[label.replace(f"_{self.fuel}", "")] for label in labels]
            
            # Construct the filename to save to
            filename_save = f"{self.fuel}-{self.pathway_color}-{self.pathway}-{self.quantity}-hist_by_country_allvessels"

        # Plot each country with vessel sizes stacked
        elif vessel_type in vessel_types:
            legend_title = "Vessel Size"
            ax.text(1.05, 0.55, f"Vessel Type: {vessel_type}", transform=ax.transAxes, fontsize=18, va='top', ha='left')
            
            stack_by = [f"{vessel_size}_{self.fuel}" for vessel_size in vessel_sizes[vessel_type]]
            
            # Only plot stacked histograms for country averages
            result_df_country_av[stack_by].plot(kind='barh', stacked=True, ax=ax)
            
            # Add individual country estimates as unfilled circles
            for idx, row in result_df_country_individual.iterrows():
                country = idx.split('_')[0]
                if country in result_df_country_av.index:
                    # Use the same x-location as the corresponding bar
                    x = result_df_country_av.index.get_loc(country)

                    # Plot individual estimates as unfilled circles
                    ax.scatter([x], [row[f"{vessel_type}_{self.fuel}"]], marker="D", color='black', s=100)
            
            # Update legend labels
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [vessel_size_title[label.replace(f"_{self.fuel}", "")] for label in labels]
            
            # Construct the filename to save to
            filename_save = f"{self.fuel}-{self.pathway_color}-{self.pathway}-{self.quantity}-{self.modifier}-hist_by_country_{vessel_type}"
            
        else:
            print(f"Vessel type should be one of: {vessel_types}. Returning from hist_by_country without plotting.")
            return
        
        # Add text to indicate the details of what's being plotted
        ax.text(1.05, 0.5, f"Fuel: {self.fuel}", transform=ax.transAxes, fontsize=18, va='top', ha='left')
        ax.text(1.05, 0.45, f"Color: {self.pathway_color}", transform=ax.transAxes, fontsize=18, va='top', ha='left')
        ax.text(1.05, 0.4, f"Pathway: {self.pathway_label}", transform=ax.transAxes, fontsize=18, va='top', ha='left')
        
        # Plot styling common to both cases
        ax.set_xlabel(f"{self.label} ({self.units})", fontsize=22)
        ax.set_yticks(range(len(result_df_country_av)))
        #plt.xticks(rotation=45)
        ax.set_yticklabels(make_country_labels(result_df_country_av.index))
        ax.legend(handles, new_labels, title=legend_title, fontsize=18, title_fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}_{self.pathway_color}_{self.pathway}_hists_by_country")
        plt.savefig(f"{top_dir}/plots/{self.fuel}_{self.pathway_color}_{self.pathway}_hists_by_country/{filename_save}.png", dpi=200)
        #plt.savefig(f"{top_dir}/plots/{self.fuel}_{self.pathway_color}_{self.pathway}_hists_by_country/{filename_save}.pdf")
        plt.close()
        
    def make_all_hists_by_country(self):
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
            self.make_hist_by_country(vessel_type)
            
        self.make_hist_by_country("all")
    
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
        
    def make_all_hists_by_country(self, quantities=["TotalEquivalentWTW", "TotalCost"]):
        """
        Executes make_all_hists_by_country() in each ProcessedQuantity class instance contained in the ProcessedQuantities dictionary to produce validation hists for the given pathway, for the selected quantities.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        for quantity in quantities:
            if not quantity in self.ProcessedQuantities:
                raise Exception(f"Error: Provided quantity '{quantity}' is not available in self.ProcessedQuantities. \n\nAvailable quanities: {self.quantities}.")
            for modifier in self.ProcessedQuantities[quantity]:
                self.ProcessedQuantities[quantity][modifier].make_all_hists_by_country()
        
        
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
    
    lsfo_pathway_WTW_fleet.make_hist_by_country("all")
    lsfo_pathway_WTW_fleet.make_hist_by_country("bulk_carrier_ice")
    
    #print(lsfo_pathway_TotalCost_fleet.result_df)
    #lsfo_pathway_TotalCost_fleet.make_all_hists_by_country()
    #lsfo_pathway_WTW_fleet.make_all_hists_by_country()
    #get_filename_info("/Users/danikamacdonell/Git/MIT_NavigaTE_inputs/processed_results/lsfo-grey-fossil-TotalFuelOPEX-per_mile.csv")
    
    #pathway = ProcessedPathway("ammonia", "blue", "ATR_CCS")
    #pathway.make_all_hists_by_country()

main()

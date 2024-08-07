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

matplotlib.rc("xtick", labelsize=18)
matplotlib.rc("ytick", labelsize=18)

RESULTS_DIR = "processed_results"

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
        Dataframe containing the label and description of each pathway.
    """
    pathway_labels_df = pd.read_csv(f"{top_dir}/info_files/pathway_info.csv").set_index("Pathway Name")
    return pathway_labels_df

# Global dataframe with labels for each fuel production pathway
pathway_labels_df = read_pathway_labels(top_dir)

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
    
class ProcessedPathway:
    """
    A class to contain results and functions for NavigaTE results for a given quantity and fuel pathway.
    Each processed pathway result is read from a csv file.
    
    Attributes
    ----------
    quantity : str
        Quantity evaluated by NavigaTE or derived from its outputs (eg. TotalCAPEX)
        
    modifier : str
        Modifier to the quantity. Can be one of:
            * None: Value of the quantity per vessel
            * fleet: Value of the quantity aggregated over all vessels
            * per_mile: Value of the quantity per nautical mile (nm)
            * per_tonn_mile: Value of the quantity per cargo tonne-nm
            
    pathway_color : str
        Color associated with the given production pathway
            * grey: From fossil sources
            * blue: From fossil sources coupled with carbon capture and storage (CCS)
            * green: From electrolytic hydrogen
    
    pathway : str
        Name of the production pathway, as it's saved in the name of the input csv file
        
    result_df : pandas.DataFrame
        Dataframe containing the processed pathway result in the csv file

    Methods
    -------
    read_result(file_path):
        Reads in a processed csv file and saves it to a pandas dataframe.
    """
    
    def __init__(self, quantity, modifier, fuel, pathway_color, pathway, results_dir = RESULTS_DIR):
        self.quantity = quantity
        self.modifier = modifier
        self.fuel = fuel
        self.pathway_color = pathway_color
        self.pathway = pathway
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
        if self.modifier is None:
            modifier_str = ""
        else:
            modifier_str = f"_{self.modifier}"
        return f"{top_dir}/{self.results_dir}/{self.fuel}-{self.pathway_color}-{self.pathway}-{self.quantity}{modifier_str}.csv"
        
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
            None: "year",
            "fleet": "year",
            "per_mile": "nm",
            "per_tonne_mile": "tonne-nm"
        }
        
        denom_units = modifier_denom_dict[self.modifier]
        
        return f"{base_units} / {denom_units}"
        
    def hist_by_country(self, vessel_type):
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
        vessel_types = list(vessels.keys())
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_ylabel(f"{self.label} ({self.units})", fontsize=22)
        ax.set_title(f"{self.fuel}: {self.pathway_color} {pathway_labels_df.loc[self.pathway, 'Label']}", fontsize=24)
        
        # Separate result_df into rows with vs. without '_' (where '_' indicates it's one of several individual estimates for a given country)
        result_df_country_av = self.result_df[~self.result_df.index.str.contains('_')]
        result_df_country_individual = self.result_df[self.result_df.index.str.contains('_')]
        
        if vessel_type == "all":
            stack_by = [f"{vessel_type}_{self.fuel}" for vessel_type in vessel_types]
            
            # Only plot stacked histograms for country averages
            result_df_country_av[stack_by].plot(kind='bar', stacked=True, ax=ax)
            
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
            filename_save = f"{self.fuel}-{self.pathway_color}-{self.pathway}-hist_by_country_allvessels"

        elif vessel_type in vessel_types:
            stack_by = list(vessels[vessel_type])
        else:
            print(f"Vessel type should be one of: {vessel_types}. Returning from hist_by_country without plotting.")
            return
        
        # Plot styling common to both cases
        ax.set_xticks(range(len(result_df_country_av)))
        plt.xticks(rotation=45)
        ax.set_xticklabels(make_country_labels(result_df_country_av.index))
        ax.legend(handles, new_labels, title="Vessel Type", fontsize=18, title_fontsize=20, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{top_dir}/plots/{filename_save}.png", dpi=300)
        plt.savefig(f"{top_dir}/plots/{filename_save}.pdf")
        

def main():
    
    lsfo_pathway_TotalCost_fleet = ProcessedPathway("TotalCost", "fleet", "ammonia", "blue", "ATR_CCS")
    
    #print(lsfo_pathway_TotalCost_fleet.result_df)
    lsfo_pathway_TotalCost_fleet.hist_by_country("all")
    

main()

"""
Date: Sept 3, 2024
Author: danikam
Purpose: Plots WTT cost and emission breakdowns for each fuel
"""

from common_tools import get_top_dir, get_pathway_type, get_pathway_type_color, get_pathway_type_label, get_pathway_label, get_fuel_label, create_directory_if_not_exists
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import numpy as np

matplotlib.rc("xtick", labelsize=18)
matplotlib.rc("ytick", labelsize=18)

# Python dictionary containing paths to files to read in for production and processing data for each fuel
# Note: File paths are provided relative to the top level of the git repo
WTT_input_files = {
    "ammonia": {
        "Production": {
            "Hydrogen Production": "input_fuel_pathway_data/production/hydrogen_costs_emissions.csv",
        },
        "Process": {
            "H-NH3 Conversion": "input_fuel_pathway_data/process/hydrogen_to_ammonia_conversion_costs_emissions.csv",
        },
    },
    "compressed_hydrogen": {
        "Production": {
            "Hydrogen Production": "input_fuel_pathway_data/production/hydrogen_costs_emissions.csv",
        },
        "Process": {
            "Hydrogen Compression": "input_fuel_pathway_data/process/hydrogen_compression_costs_emissions.csv",
        },
    },
    "liquid_hydrogen": {
        "Production": {
            "Hydrogen Production": "input_fuel_pathway_data/production/hydrogen_costs_emissions.csv",
        },
        "Process": {
            "Hydrogen Liquefaction": "input_fuel_pathway_data/process/hydrogen_liquefaction_costs_emissions.csv",
        },
    },
    "FTdiesel": {
        "Production": {
            "F-T Diesel Production": "input_fuel_pathway_data/production/FTdiesel_costs_emissions.csv"
        },
        "Process": {},
    },
    "methanol": {
        "Production": {
            "Methanol Production": "input_fuel_pathway_data/production/methanol_costs_emissions.csv"
        },
        "Process": {},
    },
}

stage_colors = {
    "Hydrogen Production": "gold",
    "F-T Diesel Production": "gold",
    "Methanol Production": "gold",
    "H-NH3 Conversion": "turquoise",
    "Hydrogen Compression": "orangered",
    "Hydrogen Liquefaction": "orchid",
}

    
class PathwayWTT:
    """
    A class to contain results and functions for WTT costs and emissions for a given fuel and production pathway. Each fuel pathway result is read in from csv files produced by calculate_fuel_costs_emissions.py.
    
    Attributes
    ----------
    fuel : str
        Fuel produced by the given production pathway (eg. ammonia, hydrogen, etc.)

    pathway_type : str
        Fuel production pathway type classification

    pathway : str
        Name of the production pathway, as it's saved in the name of the input csv file
    
    pathway_data_df : pandas.DataFrame
        Dataframe containing the WTT costs and emissions for the given pathway
        
    cost_bar_dict : Dictionary
        Dictionary with the contents of a cost bar to be plotted for the given pathway
    
    emissions_bar_dict : Dictionary
        Dictionary with the contents of an emissions bar to be plotted for the given pathway
    """
    
    def __init__(self, fuel, pathway, data_df):
        self.fuel = fuel
        self.pathway = pathway
        self.pathway_label = get_pathway_label(self.pathway)
        self.pathway_type = get_pathway_type(pathway)
        self.pathway_color = get_pathway_type_color(self.pathway_type)
        self.pathway_type_label = get_pathway_type_label(self.pathway_type)
        self.pathway_data_df = self.get_pathway_data(data_df)
        self.stages = self.get_stages()
        
    def get_pathway_data(self, data_df):
        """
        Filters the input data in data_df for rows pertaining to the given pathway. For the given fuel, data_df contains cost and emissions data for each fuel production pathway and region.
        
        Parameters
        ----------
        data_df : pandas.DataFrame
            Dataframe containing the WTT costs and emissions for all production pathways and regions for the given fuel

        Returns
        -------
        pathway_data_df : pandas.DataFrame
            Dataframe containing the WTT costs and emissions for the given pathway
        """
        return data_df[data_df["Pathway Name"] == self.pathway]
        
    def get_stages(self):
        """
        Parses the available stages from the dataframe for the pathway
        
        Parameters
        ----------
        None

        Returns
        -------
        stages : list
            List of distinct stages for which costs and emissions are quantified
        """
        
        return list(WTT_input_files[self.fuel]["Production"].keys()) + list(WTT_input_files[self.fuel]["Process"].keys())
        
    def make_bar(self, quantity):
        """
        Makes a dictionary containing info to plot a cost bar
        
        Parameters
        ----------
        quantity : str
            Quantity to make a bar for. Currently can be either cost or emissions

        Returns
        -------
        cost_bar_dict : Dictionary
            Dictionary containing data, colors, hatching, and labels for a cost bar
        """
        if not (quantity == "cost" or quantity == "emissions"):
            raise Exception("Error: supplied quantity must be either cost or emissions")
        
        bar_dict = {
            "data": [],
            "label": [],
            "color": [],
            "hatch": [],     # No hatch is used for CapEx and crosshatch for OpEx
        }
                
        for stage in self.stages:
        
            if quantity == "emissions":
                bar_dict["data"].append(self.pathway_data_df[f"{stage}: Emissions"].mean())
                bar_dict["label"].append(stage)
                bar_dict["color"].append(stage_colors[stage])
                bar_dict["hatch"].append("")
        
            if quantity == "cost":
                # OpEx cost data
                bar_dict["data"].append(self.pathway_data_df[f"{stage}: OpEx"].mean())
                bar_dict["label"].append(stage)
                bar_dict["color"].append(stage_colors[stage])
                bar_dict["hatch"].append("")
            
                # CapEx cost data
                bar_dict["data"].append(self.pathway_data_df[f"{stage}: CapEx"].mean())
                bar_dict["label"].append('_nolegend_')   # Omit the stage label for CapEx since it was already included for OpEx
                bar_dict["color"].append(stage_colors[stage])
                bar_dict["hatch"].append("xxx")
            
        return bar_dict
        
    def get_all_summed_results(self, quantity):
        """
        Collects results for the given quantity (currently cost or emissions) summed over all stages for all fuel production regions
        
        Parameters
        ----------
        quantity : str
            Quantity to collect results for. Currently can be either cost or emissions

        Returns
        -------
        summed_results_df : pandas.DataFrame
            Pandas dataframe containing summed results for all fuel production regions
        """
        i_stage = 0
        for stage in self.stages:
            
            if quantity == "cost":
                result_arr = np.asarray(self.pathway_data_df[f"{stage}: OpEx"] + self.pathway_data_df[f"{stage}: CapEx"])
                
            if quantity == "emissions":
                result_arr = np.asarray(self.pathway_data_df[f"{stage}: Emissions"])
                
            summed_result_arr = result_arr if i_stage == 0 else summed_result_arr + result_arr
            
            i_stage += 1
        return summed_result_arr
            
class FuelWTT:
    """
    A class to contain results and functions for WTT costs and emissions for a given fuel.
    
    Attributes
    ----------
    fuel : str
        Fuel produced by the given production pathway (eg. ammonia, hydrogen, etc.)
        
    pathways : list of str
        List of all pathways for the given fuel
    
    cost_emissions_df : pandas.DataFrame
        Pandas dataframe containing the breakdown of costs and emissions for all WTT stages for the given fuel
    """
    
    def __init__(self, fuel):
        self.fuel = fuel
        self.fuel_label = get_fuel_label(fuel)
        self.cost_emissions_df = self.collect_costs_emissions()
        self.pathways = self.collect_pathways()
        self.cost_emissions_df = self.collect_costs_emissions()
        
    def collect_pathways(self):
        """
        Collects a list of all production pathways for the given fuel

        Parameters
        ----------
        None

        Returns
        -------
        pathways : list
            List of pathways for the given fuel
        """
        return self.cost_emissions_df["Pathway Name"].unique()
    
    def collect_costs_emissions(self):
        """
        Collects costs and emissions for all production pathways of a given fuel, for each stage of fuel production and processing.

        Parameters
        ----------
        None

        Returns
        -------
        costs_emissions_df : pandas DataFrame
            Pandas dataframe containing the breakdown of WTT costs (CapEx and OpEx) and emissions by fuel production and process stage
        """
        
        i_stage=0
        top_dir = get_top_dir()
        def collect_stage_data(filepath_stage, stage, stage_type="Production"):
            """
            Collects the cost and emissions data for all fuel pathways, for a given fuel production or process stage

            Parameters
            ----------
            filepath_stage : str
                Path to the file containing the costs and emissions data by fuel production pathway and region for the given file and production/processing stage
                
            stage : str
                Name of the stage
                
            stage_type : str
                Type of stage (either Production or Process)

            Returns
            -------
            
            stage_data_df : pandas DataFrame
                Pandas dataframe containing the breakdown of costs (CapEx and OpEx) and emissions for the given production or process stage
            """
        
            stage_data_df = pd.read_csv(f"{top_dir}/{filepath_stage}")
            if stage_type == "Production":
                stage_data_df = stage_data_df[["Electricity Source", "Pathway Name", "Region", "Number", "CapEx [$/tonne]", "OpEx [$/tonne]", "Emissions [kg CO2e / kg fuel]"]]
            if stage_type == "Process":     # Exclude the 'Pathway Name' from the Process stage since it's identical to the 'Electricity Source' in this case
                stage_data_df = stage_data_df[["Electricity Source", "Region", "Number", "CapEx [$/tonne]", "OpEx [$/tonne]", "Emissions [kg CO2e / kg fuel]"]]
            stage_data_df.rename(columns={"CapEx [$/tonne]": f"{stage}: CapEx"}, inplace=True)
            stage_data_df.rename(columns={"OpEx [$/tonne]": f"{stage}: OpEx"}, inplace=True)
            stage_data_df.rename(columns={"Emissions [kg CO2e / kg fuel]": f"{stage}: Emissions"}, inplace=True)
        
            return stage_data_df
        
        # First, collect the costs and emissions associated with production (these can be distinct for each production pathway)
        for production_stage in WTT_input_files[self.fuel]["Production"]:
            filepath_stage = WTT_input_files[self.fuel]["Production"][production_stage]
            stage_data_df = collect_stage_data(filepath_stage, production_stage, "Production")
            
            # Either initialize or merge the production process dataframes, depending on whether we've already read one in
            if i_stage == 0:
                costs_emissions_df = stage_data_df
            else:
                costs_emissions_df = pd.merge(costs_emissions_df, stage_data_df, on=["Electricity Source", "Pathway Name", "Region", "Number"])
            
            i_stage += 1
        
        # Next, collect the costs and emissions associated with fuel processing (these can be distinct for each electricity source)
        for process_stage in WTT_input_files[self.fuel]["Process"]:
            filepath_stage = WTT_input_files[self.fuel]["Process"][process_stage]
            stage_data_df = collect_stage_data(filepath_stage, process_stage, "Process")
            
            # Add columns with the costs and emissions for this process stage to the existing dataframe
            # Results are merged based on their electricity source, region and region number
            costs_emissions_df = pd.merge(costs_emissions_df, stage_data_df, on=["Electricity Source", "Region", "Number"])
            
        return costs_emissions_df

    def make_stacked_hist(self, quantity="cost"):
        """
        Makes a histogram of emissions or cost stages respect to the available fuel production pathways.

        Parameters
        ----------
        quantity : str
            Indicates which quantity we're considering (currently either cost or emissions)

        Returns
        -------
        None
        """
        
        if not (quantity == "cost" or quantity == "emissions"):
            raise Exception("Error: supplied quantity must be either cost or emissions")

        num_pathways = len(self.pathways)
        fig_height = max(6, num_pathways * 0.9)  # Adjust this factor as needed

        fig, ax = plt.subplots(figsize=(20, fig_height))

        # Create an empty dictionary to hold the cumulative values for stacking
        cumulative_values = {}
        cumulative_values_negative = {}

        scatter_handles = []  # To collect scatter plot legend handles
        scatter_labels = []  # To collect scatter plot legend labels
        bar_handles = []  # To collect bar plot legend handles
        bar_labels = []  # To collect bar plot legend labels
        
        # Add a vertical line at 0
        ax.axvline(0, color="black")
        def plot_bar(pathway_wtt, pathway_name, pathway_label):
            """
            Plots a single bar for a given pathway

            Parameters
            ----------
            pathway_wtt : PathwayWTT
                PathwayWTT class instance containing the info to plot

            pathway_name : str
                Name of the pathway

            pathway_label : str
                Name of the pathway to use for labels when plotting

            Returns
            -------
            None
            """
            # Initialize cumulative positive and negative values for this pathway
            if pathway_name not in cumulative_values:
                cumulative_values[pathway_name] = 0
                cumulative_values_negative[pathway_name] = 0

            # Collect the info for the bar and all cumulative results for the given pathway
            bar_info = pathway_wtt.make_bar(quantity)

            # Collect all results for the given pathway, summed over all stages
            all_summed_results = pathway_wtt.get_all_summed_results(quantity)
            
            # Get the values for each sub_quantity and stack them
            for i in range(len(bar_info["data"])):
                value = bar_info["data"][i]
                
                if value >= 0:
                    bar = ax.barh(
                        pathway_label,
                        value,
                        left=cumulative_values[pathway_name],
                        label=bar_info["label"][i],
                        color=bar_info["color"][i],
                        hatch=bar_info["hatch"][i],
                        edgecolor='black'
                    )
                    cumulative_values[pathway_name] += value
                else:
                    bar = ax.barh(
                        pathway_label,
                        value,
                        left=cumulative_values_negative[pathway_name],
                        label=bar_info["label"][i],
                        color=bar_info["color"][i],
                        hatch=bar_info["hatch"][i],
                        edgecolor='black'
                    )
                    cumulative_values_negative[pathway_name] += value

                # Add the bar handles and labels only once
                if i_pathway == 0:
                    bar_handles.append(bar[0])
                    bar_labels.append(bar_info["label"][i])
                    
                # Get the current y-tick labels and positions (categories)
                yticks = ax.get_yticks()
                yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]

                # Find the numeric y-position for the current pathway label
                y_pos = yticks[yticklabels.index(pathway_label)]

                # Plot the individual region results as a scatter plot
                scatter = ax.scatter(
                    all_summed_results,
                    y_pos*np.ones(len(all_summed_results)),
                    color="black",  # region_colors[region],
                    s=50,
                    marker="o",
                    zorder=100,
                )
                            
            if i_pathway == 0:
                scatter_handles.append(scatter)
                scatter_labels.append("Individual Countries")

            # If there are negative values, draw a gray vertical bar at the cumulative sum position
            if cumulative_values_negative[pathway_name] and cumulative_values[pathway_name]:
                bar_width = cumulative_values[pathway_name] + cumulative_values_negative[pathway_name]  # Get the total width of the bar
                bar_height = bar[0].get_height()  # Get the height of the horizontal bar
                y_center = bar[0].get_y() + bar_height / 2  # Calculate the center of the bar

                ax.plot(
                    [bar_width, bar_width],  # x coordinates (vertical line)
                    [y_center - bar_height * 0.4, y_center + bar_height * 0.4],  # y coordinates
                    color='gray',
                    linewidth=5,
                    label="Sum" if i_pathway==0 else ""
                )
            # Set the y-axis label color to match the pathway type
            y_labels = ax.get_yticklabels()
            pathway_type = get_pathway_type(pathway_name)
            if i_pathway < len(y_labels):
                y_labels[i_pathway].set_color(get_pathway_type_color(pathway_type))
                y_labels[i_pathway].set_fontweight("bold")
                

        # Loop through each color and pathway
        i_pathway = 0
        for pathway_name in self.pathways:
            pathway_wtt = PathwayWTT(self.fuel, pathway_name, self.cost_emissions_df)
            pathway_label = pathway_wtt.pathway_label

            plot_bar(pathway_wtt, pathway_name, pathway_label)

            i_pathway += 1

        # Add labels and title
        if quantity == "cost":
            quantity_label = "WTT Cost"
            quantity_units = "USD/tonne"
        if quantity == "emissions":
            quantity_label = "WTT Emissions"
            quantity_units = "kg CO2e / kg fuel"
        ax.set_xlabel(f"{quantity_label} ({quantity_units})", fontsize=20)
        ax.set_title(f"Fuel: {self.fuel_label}", fontsize=24)

        # Add a legend for the stacked bar components (sub-quantities)
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

        # Add a separate legend for countries
        if scatter_handles:
            ax.legend(
                scatter_handles,
                scatter_labels,
                fontsize=16,
                bbox_to_anchor=(1.01, 0.35),
                loc="center left",
                borderaxespad=0.0,
            )

        # Add the bar component legend back after the region legend if both legends are present
        if bar_handles and scatter_handles:
            ax.add_artist(legend1)

        # If quantity is cost, add the legend for OpEx (no hatch) and CapEx (hatch)
        if quantity == "cost":
            # Create custom patches for OpEx and CapEx
            op_ex_patch = Patch(facecolor='white', edgecolor='black', label='OpEx')
            cap_ex_patch = Patch(facecolor='white', edgecolor='black', hatch='xxx', label='CapEx')

            # Add the OpEx and CapEx legend
            ax.legend(
                handles=[op_ex_patch, cap_ex_patch],
                fontsize=16,
                title="Cost Types",
                title_fontsize=20,
                bbox_to_anchor=(1.01, 0.35),
                loc="center left",
                borderaxespad=0.0,
            )

        plt.subplots_adjust(left=0.25, right=0.8)

        # Construct the filename to save to
        filename_save = f"{self.fuel}-{quantity}-WTT_hist"

        # Save the figure
        top_dir = get_top_dir()
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}")
        filepath_save = f"{top_dir}/plots/{self.fuel}/{filename_save}.png"
        print(f"Saving figure to {filepath_save}")
        plt.savefig(filepath_save, dpi=200)
        plt.close()

def main():

    for fuel in ["compressed_hydrogen", "liquid_hydrogen", "ammonia", "methanol", "FTdiesel"]:
        fuel_wtt = FuelWTT(fuel)
        fuel_wtt.make_stacked_hist("emissions")
        fuel_wtt.make_stacked_hist("cost")
    

main()

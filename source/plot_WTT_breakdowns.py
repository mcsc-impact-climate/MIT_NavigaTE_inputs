"""
Date: Sept 3, 2024
Author: danikam
Purpose: Plots WTG cost and emission breakdowns for each fuel
"""

from common_tools import get_top_dir, get_pathway_type, get_pathway_type_color, get_pathway_type_label, get_pathway_label, get_fuel_label, create_directory_if_not_exists
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import numpy as np
from collections import defaultdict

H2_PER_NH3 = 3.02352/17.03022  # kg H2 required to produce 1 kg of NH3

matplotlib.rc("xtick", labelsize=18)
matplotlib.rc("ytick", labelsize=18)

# Python dictionary containing paths to files to read in for production and processing data for each fuel
# Note: File paths are provided relative to the top level of the git repo
WTG_input_files = {
    "ammonia": {
        "Production": {
            "H Production": "input_fuel_pathway_data/production/hydrogen_costs_emissions.csv",
        },
        "Process": {
            "H-NH3 Conversion": "input_fuel_pathway_data/process/hydrogen_to_ammonia_conversion_costs_emissions.csv",
        },
    },
    "compressed_hydrogen": {
        "Production": {
            "H Production": "input_fuel_pathway_data/production/hydrogen_costs_emissions.csv",
        },
        "Process": {
            "H Compression": "input_fuel_pathway_data/process/hydrogen_compression_costs_emissions.csv",
        },
    },
    "liquid_hydrogen": {
        "Production": {
            "H Production": "input_fuel_pathway_data/production/hydrogen_costs_emissions.csv",
        },
        "Process": {
            "H Liquefaction": "input_fuel_pathway_data/process/hydrogen_liquefaction_costs_emissions.csv",
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
            "MeOH Production": "input_fuel_pathway_data/production/methanol_costs_emissions.csv"
        },
        "Process": {},
    },
    "lng": {
        "Production": {
            "NG Production": "input_fuel_pathway_data/production/ng_costs_emissions.csv"
            },
        "Process": {
            "NG Liquefaction": "input_fuel_pathway_data/process/ng_liquefaction_costs_emissions.csv"
            },
    },
}

stage_colors = {
    "H Production": "gold",
    "F-T Diesel Production": "gold",
    "MeOH Production": "gold",
    "H-NH3 Conversion": "turquoise",
    "H Compression": "orangered",
    "H Liquefaction": "orchid",
    "NG Production": "gold",
    "NG Liquefaction": "orchid"
}

continent_regions = {
    "Africa": ["South Africa"],
    "Americas": ["Brazil", "Canada", "Mexico", "United States"],
    "Asia": ["Australia", "China", "India", "Indonesia", "Japan", "Malaysia", "Philippines", "Singapore", "South Korea", "Taipei", "Thailand"],
    "Europe": ["Austria", "Belgium", "Czech Republic", "Finland", "France", "Germany", "Greece", "Hungary", "Italy", "Netherlands", "Norway", "Poland", "Portugal", "Russia", "Spain", "Sweden", "Switzerland", "Turkey", "United Kingdom"],
    "Middle East": ["Oman", "Saudi Arabia", "South Arabia", "United Arab Emirates"]
}

fuel_pathways = {
    "e-hydrogen (liquefied)": ["LTE_H_grid_E", "LTE_H_solar_E", "LTE_H_wind_E"],
    "e-hydrogen (compressed)": ["LTE_H_grid_E", "LTE_H_solar_E", "LTE_H_wind_E"],
    "e-ammonia": ["LTE_H_grid_E", "LTE_H_solar_E", "LTE_H_wind_E"],
    "Blue ammonia": ["SMRCCS_H_solar_E", "SMRCCS_H_wind_E" "SMRCCS_H_grid_E"],
    "e-methanol": ["LTE_H_SMRCCS_C_grid_E", "LTE_H_SMRCCS_C_solar_E", "LTE_H_SMRCCS_C_wind_E", "LTE_H_BEC_C_grid_E", "LTE_H_BEC_C_solar_E", "LTE_H_BEC_C_wind_E", "LTE_H_DAC_C_grid_E", "LTE_H_DAC_C_solar_E", "LTE_H_DAC_C_wind_E"],
    "e-diesel": ["LTE_H_SMRCCS_C_grid_E", "LTE_H_SMRCCS_C_solar_E", "LTE_H_SMRCCS_C_wind_E", "LTE_H_BEC_C_grid_E", "LTE_H_BEC_C_solar_E", "LTE_H_BEC_C_wind_E", "LTE_H_DAC_C_grid_E", "LTE_H_DAC_C_solar_E", "LTE_H_DAC_C_wind_E"],
}

fuel_fuels = {
    "e-hydrogen (liquefied)": "liquid_hydrogen",
    "e-hydrogen (compressed)": "compressed_hydrogen",
    "e-ammonia": "ammonia",
    "Blue ammonia": "ammonia",
    "e-methanol": "methanol",
    "e-diesel": "FTdiesel",
}
    
class PathwayWTG:
    """
    A class to contain results and functions for WTG costs and emissions for a given fuel and production pathway. Each fuel pathway result is read in from csv files produced by calculate_fuel_costs_emissions.py.
    
    Attributes
    ----------
    fuel : str
        Fuel produced by the given production pathway (eg. ammonia, hydrogen, etc.)

    pathway_type : str
        Fuel production pathway type classification

    pathway : str
        Name of the production pathway, as it's saved in the name of the input csv file
    
    pathway_data_df : pandas.DataFrame
        Dataframe containing the WTG costs and emissions for the given pathway
        
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
            Dataframe containing the WTG costs and emissions for all production pathways and regions for the given fuel

        Returns
        -------
        pathway_data_df : pandas.DataFrame
            Dataframe containing the WTG costs and emissions for the given pathway
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
        
        return list(WTG_input_files[self.fuel]["Production"].keys()) + list(WTG_input_files[self.fuel]["Process"].keys())
        
    def make_bar(self, quantity, continent="all"):
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
        
        if continent == "all":
            pathway_data_df = self.pathway_data_df.copy()
        else:
            regions = continent_regions[continent]
            pathway_data_df = self.pathway_data_df[self.pathway_data_df["Region"].isin(regions)]
                
        for stage in self.stages:
        
            if quantity == "emissions":
                bar_dict["data"].append(pathway_data_df[f"{stage}: Emissions"].mean())
                bar_dict["label"].append(stage)
                bar_dict["color"].append(stage_colors[stage])
                bar_dict["hatch"].append("")
        
            if quantity == "cost":
                # OpEx cost data
                bar_dict["data"].append(pathway_data_df[f"{stage}: OpEx"].mean())
                bar_dict["label"].append(stage)
                bar_dict["color"].append(stage_colors[stage])
                bar_dict["hatch"].append("")
            
                # CapEx cost data
                bar_dict["data"].append(pathway_data_df[f"{stage}: CapEx"].mean())
                bar_dict["label"].append('_nolegend_')   # Omit the stage label for CapEx since it was already included for OpEx
                bar_dict["color"].append(stage_colors[stage])
                bar_dict["hatch"].append("xxx")
            
        return bar_dict
        
    def get_all_summed_results(self, quantity, continent="all"):
        """
        Collects results for the given quantity (currently cost or emissions) summed over all stages for all fuel production regions
        
        Parameters
        ----------
        quantity : str
            Quantity to collect results for. Currently can be either cost or emissions
            
        continent : str
            Indicates which continent the regions need to belong to. If "all", allow for all world regions

        Returns
        -------
        summed_results_df : pandas.DataFrame
            Pandas dataframe containing summed results for all fuel production regions
        """
        i_stage = 0
        summed_result_arr = pd.DataFrame()
        for stage in self.stages:
            result_arr = pd.DataFrame()
            if quantity == "cost":
                result_arr["cost"] = self.pathway_data_df[f"{stage}: OpEx"] + self.pathway_data_df[f"{stage}: CapEx"]
                
            if quantity == "emissions":
                result_arr["emissions"] = self.pathway_data_df[f"{stage}: Emissions"]
                
            summed_result_arr[quantity] = result_arr if i_stage == 0 else summed_result_arr + result_arr
            
            i_stage += 1
        
        summed_result_arr["Region"] = self.pathway_data_df["Region"]
        if not continent == "all":
            regions = continent_regions[continent]
            summed_result_arr = summed_result_arr[summed_result_arr["Region"].isin(regions)]
        return summed_result_arr
            
class FuelWTG:
    """
    A class to contain results and functions for WTG costs and emissions for a given fuel.
    
    Attributes
    ----------
    fuel : str
        Fuel produced by the given production pathway (eg. ammonia, hydrogen, etc.)
        
    pathways : list of str
        List of all pathways for the given fuel
    
    cost_emissions_df : pandas.DataFrame
        Pandas dataframe containing the breakdown of costs and emissions for all WTG stages for the given fuel
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
            Pandas dataframe containing the breakdown of WTG costs (CapEx and OpEx) and emissions by fuel production and process stage
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
            
            # Account values from per tonne of H2 to per tonne of ammonia in the case of hydrogen production for ammonia
            if stage == "Hydrogen Production" and self.fuel == "ammonia":
                stage_data_df["Hydrogen Production: CapEx"] = stage_data_df["Hydrogen Production: CapEx"] * H2_PER_NH3
                stage_data_df["Hydrogen Production: OpEx"] = stage_data_df["Hydrogen Production: OpEx"] * H2_PER_NH3
                stage_data_df["Hydrogen Production: Emissions"] = stage_data_df["Hydrogen Production: Emissions"] * H2_PER_NH3
        
            return stage_data_df
        
        # First, collect the costs and emissions associated with production (these can be distinct for each production pathway)
        for production_stage in WTG_input_files[self.fuel]["Production"]:
            filepath_stage = WTG_input_files[self.fuel]["Production"][production_stage]
            stage_data_df = collect_stage_data(filepath_stage, production_stage, "Production")
            
            # Either initialize or merge the production process dataframes, depending on whether we've already read one in
            if i_stage == 0:
                costs_emissions_df = stage_data_df
            else:
                costs_emissions_df = pd.merge(costs_emissions_df, stage_data_df, on=["Electricity Source", "Pathway Name", "Region", "Number"])
            
            i_stage += 1
        
        # Next, collect the costs and emissions associated with fuel processing (these can be distinct for each electricity source)
        for process_stage in WTG_input_files[self.fuel]["Process"]:
            filepath_stage = WTG_input_files[self.fuel]["Process"][process_stage]
            stage_data_df = collect_stage_data(filepath_stage, process_stage, "Process")
            
            # Add columns with the costs and emissions for this process stage to the existing dataframe
            # Results are merged based on their electricity source, region and region number
            costs_emissions_df = pd.merge(costs_emissions_df, stage_data_df, on=["Electricity Source", "Region", "Number"])
            
        return costs_emissions_df

    def make_stacked_hist(self, quantity="cost"):
        """
        Makes a histogram of emissions or cost stages with fuel pathways grouped by color.

        Parameters
        ----------
        quantity : str
            Indicates which quantity we're considering (either cost or emissions)

        Returns
        -------
        None
        """

        if quantity not in {"cost", "emissions"}:
            raise ValueError("Error: supplied quantity must be either 'cost' or 'emissions'")

        num_pathways = len(self.pathways)
        fig_height = max(6, num_pathways * 0.9)  # Adjust this factor as needed

        fig, ax = plt.subplots(figsize=(20, fig_height))

        # Sort pathways by their associated color
        sorted_pathways = sorted(
            self.pathways,
            key=lambda p: get_pathway_type_color(get_pathway_type(p))
        )

        y_positions = np.arange(len(sorted_pathways))  # Assign new y-positions

        cumulative_values = {}
        cumulative_values_negative = {}

        scatter_handles = []  # To collect scatter plot legend handles
        scatter_labels = []  # To collect scatter plot legend labels
        bar_handles = []  # To collect bar plot legend handles
        bar_labels = []  # To collect bar plot legend labels

        ax.axvline(0, color="black")

        def plot_bar(pathway_wtt, pathway_name, pathway_label, y_pos):
            """
            Plots a single bar for a given pathway.

            Parameters
            ----------
            pathway_wtt : PathwayWTG
                PathwayWTG class instance containing the info to plot

            pathway_name : str
                Name of the pathway

            pathway_label : str
                Label for the pathway

            y_pos : float
                Adjusted y position for plotting based on sorting

            Returns
            -------
            None
            """
            if pathway_name not in cumulative_values:
                cumulative_values[pathway_name] = 0
                cumulative_values_negative[pathway_name] = 0

            bar_info = pathway_wtt.make_bar(quantity)

            all_summed_results = pathway_wtt.get_all_summed_results(quantity)

            for i in range(len(bar_info["data"])):
                value = bar_info["data"][i]

                if value >= 0:
                    bar = ax.barh(
                        y_pos,
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
                        y_pos,
                        value,
                        left=cumulative_values_negative[pathway_name],
                        label=bar_info["label"][i],
                        color=bar_info["color"][i],
                        hatch=bar_info["hatch"][i],
                        edgecolor='black'
                    )
                    cumulative_values_negative[pathway_name] += value

                if i_pathway == 0:
                    bar_handles.append(bar[0])
                    bar_labels.append(bar_info["label"][i])

            scatter = ax.scatter(
                all_summed_results[quantity],
                y_pos * np.ones(len(all_summed_results[quantity])),
                color="black",
                s=50,
                marker="o",
                zorder=100,
            )

            if i_pathway == 0:
                scatter_handles.append(scatter)
                scatter_labels.append("Individual Countries")

        # Loop through each sorted pathway and assign colors to labels
        i_pathway = 0
        for pathway_name, y_pos in zip(sorted_pathways, y_positions):
            pathway_wtt = PathwayWTG(self.fuel, pathway_name, self.cost_emissions_df)
            pathway_label = pathway_wtt.pathway_label
            plot_bar(pathway_wtt, pathway_name, pathway_label, y_pos)
            i_pathway += 1

        # Set y-axis labels **after all bars have been plotted**
        ax.set_yticks(y_positions)
        ax.set_yticklabels([get_pathway_label(p) for p in sorted_pathways], fontsize=18)

        # Apply colors to y-axis labels
        y_labels = ax.get_yticklabels()
        for idx, pathway_name in enumerate(sorted_pathways):
            pathway_type = get_pathway_type(pathway_name)
            y_labels[idx].set_color(get_pathway_type_color(pathway_type))
            y_labels[idx].set_fontweight("bold")

        quantity_label = "WTG Cost" if quantity == "cost" else "WTG Emissions"
        quantity_units = "USD/tonne" if quantity == "cost" else "kg CO2e / kg fuel"
        ax.set_xlabel(f"{quantity_label} ({quantity_units})", fontsize=24)
        ax.set_title(f"Fuel: {self.fuel_label}", fontsize=28)

        if bar_handles:
            legend1 = ax.legend(
                bar_handles,
                bar_labels,
                fontsize=20,
                title="Components",
                title_fontsize=22,
                bbox_to_anchor=(1.01, 0.8),
                loc="upper left",
                borderaxespad=0.0,
            )

            ax.add_artist(legend1)

        if scatter_handles:
            legend2 = ax.legend(
                scatter_handles,
                scatter_labels,
                fontsize=20,
                title="Individual Countries",
                title_fontsize=22,
                bbox_to_anchor=(1.01, 0.5),
                loc="center left",
                borderaxespad=0.0,
            )

            ax.add_artist(legend2)

        if quantity == "cost":
            op_ex_patch = Patch(facecolor='white', edgecolor='black', label='OpEx')
            cap_ex_patch = Patch(facecolor='white', edgecolor='black', hatch='xxx', label='CapEx')

            legend3 = ax.legend(
                handles=[op_ex_patch, cap_ex_patch],
                fontsize=20,
                title="Cost Types",
                title_fontsize=22,
                bbox_to_anchor=(1.01, 0.25),
                loc="center left",
                borderaxespad=0.0,
            )
            ax.add_artist(legend3)

        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.subplots_adjust(left=0.27, right=0.8)

        filename_save = f"{self.fuel}-{quantity}-WTG_hist"
        top_dir = get_top_dir()
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}")
        filepath_save = f"{top_dir}/plots/{self.fuel}/{filename_save}.png"
        filepath_save_pdf = f"{top_dir}/plots/{self.fuel}/{filename_save}.pdf"
        print(f"Saving figure to {filepath_save}")
        plt.savefig(filepath_save, dpi=200)
        plt.savefig(filepath_save_pdf)
        plt.close()
        
def get_MMMCZCS_fuel_cost(MMMCZCS_fuel, year, continent):
    """
    Collect the fuel cost for the given pathway, year and continent from data downloaded in csv format from the MMMCZCS's fuel cost calculator tool (https://www.zerocarbonshipping.com/cost-calculator)
    
    Parameters
    ----------
    MMMCZCS_fuel : str
        Name of the fuel as denoted in the MMMCZCS's fuel cost calculator tool

    continent : str
        Name of the continent to plot the stacked hist for

    quantity : str
        Name of the quantity to plot (can currently be either cost or emissions per tonne of fuel)

    Returns
    -------
    None
    """
    
    top_dir = get_top_dir()
    
    continent_nospace = continent.replace(" ", "")
    highE_df = pd.read_csv(f"{top_dir}/data/MMMCZCS_fuel_cost_calculator_results/{continent_nospace}_highEcost.csv", index_col=0)
    lowE_df = pd.read_csv(f"{top_dir}/data/MMMCZCS_fuel_cost_calculator_results/{continent_nospace}_lowEcost.csv", index_col=0)
    
    fuel_cost = {}
    fuel_cost["High E Cost"] = highE_df[str(year)][highE_df.index==MMMCZCS_fuel].iloc[0]
    fuel_cost["Low E Cost"] = lowE_df[str(year)][lowE_df.index==MMMCZCS_fuel].iloc[0]
    return fuel_cost
    
        
def make_fuel_continent_stacked_hist(MMMCZCS_fuel, continent, quantity="cost"):
    """
    For a given fuel defined in the MMMCZCS fuel cost calculator tool (https://www.zerocarbonshipping.com/cost-calculator), make a stacked horizontal histogram of our internally-calculated WTG costs, with one bar per region and pathway
    
    Parameters
    ----------
    MMMCZCS_fuel : str
        Name of the fuel as denoted in the MMMCZCS's fuel cost calculator tool

    continent : str
        Name of the continent to plot the stacked hist for

    quantity : str
        Name of the quantity to plot (can currently be either cost or emissions per tonne of fuel)

    Returns
    -------
    None
    """
    
    fuel_wtt = FuelWTG(fuel_fuels[MMMCZCS_fuel])
    regions = continent_regions[continent]
    
    # Create a mapping of pathway to color
    pathway_color_mapping = {
        pathway_name: get_pathway_type_color(get_pathway_type(pathway_name))
        for pathway_name in fuel_wtt.pathways
    }
    
    #### Sort the pathways by their associated color to group them ####
    pathways = fuel_pathways[MMMCZCS_fuel]
    
    # Group pathways by color
    pathways_by_color = defaultdict(list)
    for p in self.pathways:
        color = get_pathway_type_color(get_pathway_type(p))
        pathways_by_color[color].append(p)

    # Sort each color group alphabetically by pathway label
    for color in pathways_by_color:
        pathways_by_color[color].sort(key=lambda p: get_pathway_label(p).lower())

    # Sort color groups by the label of their first pathway
    sorted_color_groups = sorted(
        pathways_by_color.items(),
        key=lambda item: get_pathway_label(item[1][0]).lower()
    )

    # Flatten to get final sorted pathway list
    sorted_pathways = [p for _, group in sorted_color_groups for p in group]
    #sorted_pathways = sorted(pathways, key=lambda p: pathway_color_mapping[p])
    
    num_pathways = len(pathways)
    fig_height = max(6, num_pathways * 0.9)  # Adjust this factor as needed

    fig, ax = plt.subplots(figsize=(20, fig_height))

    # Create an empty dictionary to hold the cumulative values for stacking
    cumulative_values = {}
    cumulative_values_negative = {}

    bar_handles = []  # To collect bar plot legend handles
    bar_labels = []  # To collect bar plot legend labels
    scatter_handles = []  # To collect scatter plot legend handles
    scatter_labels = []  # To collect scatter plot legend labels
    
    # Add a vertical line at 0
    ax.axvline(0, color="black")
    
    def plot_bar(pathway_wtt, pathway_name, pathway_label):
        """
        Plots a single bar for a given pathway

        Parameters
        ----------
        pathway_wtt : PathwayWTG
            PathwayWTG class instance containing the info to plot

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
        bar_info = pathway_wtt.make_bar(quantity, continent)
        
        # Collect all results for the given pathway, summed over all stages
        all_summed_results = pathway_wtt.get_all_summed_results(quantity, continent)
        
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
        cmap = plt.colormaps.get_cmap('tab20')
        colors = [cmap(i) for i in range(len(all_summed_results))]
        # Scatter plot with different colors for each region
        for idx, region in enumerate(all_summed_results["Region"]):
            scatter = ax.scatter(
                all_summed_results["cost"].iloc[idx],
                y_pos,  # y_pos is the y-coordinate for the current pathway
                color=colors[idx],  # Use colormap indexing to get the color
                label=region,
                s=100,
                marker="o",
                zorder=100,
            )
            if i_pathway == 0:
                scatter_handles.append(scatter)
                scatter_labels.append(region)

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
            y_labels[i_pathway].set_color(pathway_wtt.pathway_color)
            y_labels[i_pathway].set_fontweight("bold")

    # Loop through each pathway and region
    i_pathway = 0
    for pathway_name in sorted_pathways:
        pathway_wtt = PathwayWTG(fuel_wtt.fuel, pathway_name, fuel_wtt.cost_emissions_df)
        pathway_label = pathway_wtt.pathway_label

        plot_bar(pathway_wtt, pathway_name, pathway_label)

        i_pathway += 1
        
    # Add the fuel costs from the MMMCZCS fuel cost calculator
    MMMCZCS_costs = get_MMMCZCS_fuel_cost(MMMCZCS_fuel, 2024, continent)
    
    # Add bars for the MMMCZCS fuel cost with high and low E
    ax.barh(f"MMMCZCS (high E cost)", MMMCZCS_costs["High E Cost"], color="grey", edgecolor="black")
    ax.barh(f"MMMCZCS (low E cost)", MMMCZCS_costs["Low E Cost"], color="grey", edgecolor="black")
    y_labels = ax.get_yticklabels()
    y_labels[i_pathway].set_color("black")
    y_labels[i_pathway+1].set_color("black")
        
    # Add labels and title
    if quantity == "cost":
        quantity_label = "WTG Cost"
        quantity_units = "USD/tonne"
    if quantity == "emissions":
        quantity_label = "WTG Emissions"
        quantity_units = "kg CO2e / kg fuel"
    ax.set_xlabel(f"{quantity_label} ({quantity_units})", fontsize=20)
    ax.set_title(f"{MMMCZCS_fuel} ({continent})", fontsize=24)

    # Add a legend for the stacked bar components (sub-quantities)
    if bar_handles:
        legend1 = ax.legend(
            bar_handles,
            bar_labels,
            fontsize=20,
            title="Components",
            title_fontsize=22,
            bbox_to_anchor=(1.01, 0.8),
            loc="upper left",
            borderaxespad=0.0,
        )

    # Add a separate legend for countries
    if scatter_handles:
        scatter_legend = ax.legend(
            scatter_handles,
            scatter_labels,
            fontsize=16,
            title="Individual Countries",
            title_fontsize=20,
            bbox_to_anchor=(1.5, 0.5),
            loc="center left",
            borderaxespad=0.0,
        )
        ax.add_artist(scatter_legend)

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

    plt.subplots_adjust(left=0.25, right=0.6)

    # Construct the filename to save to
    MMMCZCS_fuel_save = MMMCZCS_fuel.replace(" ", "_").replace("(", "").replace(")", "")
    continent_save = continent.replace(" ", "")
    filename_save = f"{MMMCZCS_fuel_save}-{continent_save}-{quantity}-WTG_hist"

    # Save the figure
    top_dir = get_top_dir()
    create_directory_if_not_exists(f"{top_dir}/plots/mmmczcs_fuel_cost_comparison")
    filepath_save = f"{top_dir}/plots/mmmczcs_fuel_cost_comparison/{filename_save}.png"
    print(f"Saving figure to {filepath_save}")
    plt.savefig(filepath_save, dpi=200)
    plt.close()

def main():
    
    for fuel in ["compressed_hydrogen", "liquid_hydrogen", "ammonia", "methanol", "FTdiesel", "lng"]:
        fuel_wtt = FuelWTG(fuel)
        fuel_wtt.make_stacked_hist("emissions")
        fuel_wtt.make_stacked_hist("cost")

#    for MMMCZCS_fuel in fuel_pathways.keys():
#        for continent in continent_regions.keys():
#            make_fuel_continent_stacked_hist(MMMCZCS_fuel, continent, "cost")

main()

    

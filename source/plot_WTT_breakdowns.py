"""
Date: Sept 3, 2024
Author: danikam
Purpose: Plots WTG cost and emission breakdowns for each fuel
"""

from common_tools import get_top_dir, get_pathway_type, get_pathway_type_color, get_pathway_type_label, get_pathway_label, get_fuel_label, create_directory_if_not_exists, get_fuel_LHV
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
import numpy as np
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common_tools import (
    create_directory_if_not_exists,
    get_fuel_label,
    get_pathway_label,
    get_pathway_type,
    get_pathway_type_color,
    get_pathway_type_label,
    get_top_dir,
)
from matplotlib.patches import Patch

H2_PER_NH3 = 3.02352 / 17.03022  # kg H2 required to produce 1 kg of NH3

matplotlib.rc("xtick", labelsize=18)
matplotlib.rc("ytick", labelsize=18)

lsfo_wtg_emissions = 0.56        # kg CO2e / kg fuel
lsfo_wtg_cost_2025 = 634.942     # USD / tonne
lsfo_wtg_cost_2030 = 553.707     # USD / tonne

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
    "lsng": {
        "Production": {
            "SNG Production": "input_fuel_pathway_data/production/sng_costs_emissions.csv"
        },
        "Process": {
            "SNG Liquefaction": "input_fuel_pathway_data/process/sng_liquefaction_costs_emissions.csv"
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
    "SNG Production": "gold",
    "NG Liquefaction": "orchid",
    "SNG Liquefaction": "orchid",
}

continent_regions = {
    "Africa": ["South Africa"],
    "Americas": ["Brazil", "Canada", "Mexico", "United States"],
    "Asia": [
        "Australia",
        "China",
        "India",
        "Indonesia",
        "Japan",
        "Malaysia",
        "Philippines",
        "Singapore",
        "South Korea",
        "Taipei",
        "Thailand",
    ],
    "Europe": [
        "Austria",
        "Belgium",
        "Czech Republic",
        "Finland",
        "France",
        "Germany",
        "Greece",
        "Hungary",
        "Italy",
        "Netherlands",
        "Norway",
        "Poland",
        "Portugal",
        "Russia",
        "Spain",
        "Sweden",
        "Switzerland",
        "Turkey",
        "United Kingdom",
    ],
    "Middle East": ["Oman", "Saudi Arabia", "South Arabia", "United Arab Emirates"],
}

fuel_pathways = {
    "e-hydrogen (liquefied)": ["LTE_H_grid_E", "LTE_H_solar_E", "LTE_H_wind_E"],
    "e-hydrogen (compressed)": ["LTE_H_grid_E", "LTE_H_solar_E", "LTE_H_wind_E"],
    "e-ammonia": ["LTE_H_grid_E", "LTE_H_solar_E", "LTE_H_wind_E"],
    "Blue ammonia": ["SMRCCS_H_solar_E", "SMRCCS_H_wind_E" "SMRCCS_H_grid_E"],
    "e-methanol": [
        "LTE_H_SMRCCS_C_grid_E",
        "LTE_H_SMRCCS_C_solar_E",
        "LTE_H_SMRCCS_C_wind_E",
        "LTE_H_BEC_C_grid_E",
        "LTE_H_BEC_C_solar_E",
        "LTE_H_BEC_C_wind_E",
        "LTE_H_DAC_C_grid_E",
        "LTE_H_DAC_C_solar_E",
        "LTE_H_DAC_C_wind_E",
    ],
    "e-diesel": [
        "LTE_H_SMRCCS_C_grid_E",
        "LTE_H_SMRCCS_C_solar_E",
        "LTE_H_SMRCCS_C_wind_E",
        "LTE_H_BEC_C_grid_E",
        "LTE_H_BEC_C_solar_E",
        "LTE_H_BEC_C_wind_E",
        "LTE_H_DAC_C_grid_E",
        "LTE_H_DAC_C_solar_E",
        "LTE_H_DAC_C_wind_E",
    ],
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

        return list(WTG_input_files[self.fuel]["Production"].keys()) + list(
            WTG_input_files[self.fuel]["Process"].keys()
        )

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
            "hatch": [],  # No hatch is used for CapEx and crosshatch for OpEx
        }

        if continent == "all":
            pathway_data_df = self.pathway_data_df.copy()
        else:
            regions = continent_regions[continent]
            pathway_data_df = self.pathway_data_df[
                self.pathway_data_df["Region"].isin(regions)
            ]

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
                bar_dict["label"].append(
                    "_nolegend_"
                )  # Omit the stage label for CapEx since it was already included for OpEx
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
                result_arr["cost"] = (
                    self.pathway_data_df[f"{stage}: OpEx"]
                    + self.pathway_data_df[f"{stage}: CapEx"]
                )

            if quantity == "emissions":
                result_arr["emissions"] = self.pathway_data_df[f"{stage}: Emissions"]

            summed_result_arr[quantity] = (
                result_arr if i_stage == 0 else summed_result_arr + result_arr
            )

            i_stage += 1

        summed_result_arr["Region"] = self.pathway_data_df["Region"]
        if not continent == "all":
            regions = continent_regions[continent]
            summed_result_arr = summed_result_arr[
                summed_result_arr["Region"].isin(regions)
            ]
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

        i_stage = 0
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
                stage_data_df = stage_data_df[
                    [
                        "Electricity Source",
                        "Pathway Name",
                        "Region",
                        "Number",
                        "CapEx [$/tonne]",
                        "OpEx [$/tonne]",
                        "Emissions [kg CO2e / kg fuel]",
                    ]
                ]
            if (
                stage_type == "Process"
            ):  # Exclude the 'Pathway Name' from the Process stage since it's identical to the 'Electricity Source' in this case
                stage_data_df = stage_data_df[
                    [
                        "Electricity Source",
                        "Region",
                        "Number",
                        "CapEx [$/tonne]",
                        "OpEx [$/tonne]",
                        "Emissions [kg CO2e / kg fuel]",
                    ]
                ]
            stage_data_df.rename(
                columns={"CapEx [$/tonne]": f"{stage}: CapEx"}, inplace=True
            )
            stage_data_df.rename(
                columns={"OpEx [$/tonne]": f"{stage}: OpEx"}, inplace=True
            )
            stage_data_df.rename(
                columns={"Emissions [kg CO2e / kg fuel]": f"{stage}: Emissions"},
                inplace=True,
            )

            # Account values from per tonne of H2 to per tonne of ammonia in the case of hydrogen production for ammonia
            if stage == "H Production" and self.fuel == "ammonia":
                stage_data_df["H Production: CapEx"] = (
                    stage_data_df["H Production: CapEx"] * H2_PER_NH3
                )
                stage_data_df["H Production: OpEx"] = (
                    stage_data_df["H Production: OpEx"] * H2_PER_NH3
                )
                stage_data_df["H Production: Emissions"] = (
                    stage_data_df["H Production: Emissions"] * H2_PER_NH3
                )

            return stage_data_df

        # First, collect the costs and emissions associated with production (these can be distinct for each production pathway)
        for production_stage in WTG_input_files[self.fuel]["Production"]:
            filepath_stage = WTG_input_files[self.fuel]["Production"][production_stage]
            stage_data_df = collect_stage_data(
                filepath_stage, production_stage, "Production"
            )

            # Either initialize or merge the production process dataframes, depending on whether we've already read one in
            if i_stage == 0:
                costs_emissions_df = stage_data_df
            else:
                costs_emissions_df = pd.merge(
                    costs_emissions_df,
                    stage_data_df,
                    on=["Electricity Source", "Pathway Name", "Region", "Number"],
                )

            i_stage += 1

        # Next, collect the costs and emissions associated with fuel processing (these can be distinct for each electricity source)
        for process_stage in WTG_input_files[self.fuel]["Process"]:
            filepath_stage = WTG_input_files[self.fuel]["Process"][process_stage]
            stage_data_df = collect_stage_data(filepath_stage, process_stage, "Process")

            # Add columns with the costs and emissions for this process stage to the existing dataframe
            # Results are merged based on their electricity source, region and region number
            costs_emissions_df = pd.merge(
                costs_emissions_df,
                stage_data_df,
                on=["Electricity Source", "Region", "Number"],
            )

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
            raise ValueError(
                "Error: supplied quantity must be either 'cost' or 'emissions'"
            )

        num_pathways = len(self.pathways)
        fig_height = max(6, num_pathways * 0.9)  # Adjust this factor as needed

        fig, ax = plt.subplots(figsize=(20, fig_height))

        # Sort pathways by their associated color
        #        sorted_pathways = sorted(
        #            self.pathways,
        #            key=lambda p: get_pathway_type_color(get_pathway_type(p))
        #        )

        #### Sort the pathways by their associated color to group them ####

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
            key=lambda item: get_pathway_label(item[1][0]).lower(),
        )

        # Flatten to get final sorted pathway list
        sorted_pathways = [p for _, group in sorted_color_groups for p in group]

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
                        edgecolor="black",
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
                        edgecolor="black",
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
            op_ex_patch = Patch(facecolor="white", edgecolor="black", label="OpEx")
            cap_ex_patch = Patch(
                facecolor="white", edgecolor="black", hatch="xxx", label="CapEx"
            )

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
        
    def plot_country_bars(self, quantity="cost", countries=[], per_GJ=False):
        """
        Makes a horizontal bar plot showing WTG cost or emissions for each pathway,
        with side-by-side bars for each selected country. Values can be normalized per GJ or per kg.

        Parameters
        ----------
        quantity : str
            Either 'cost' or 'emissions'

        countries : list of str
            List of country names to show as individual bars per pathway

        per_GJ : bool, optional (default=False)
            If True, normalize values by fuel LHV to show per GJ fuel. Otherwise, show per kg.

        Returns
        -------
        None
        """

        if quantity not in {"cost", "emissions"}:
            raise ValueError("Quantity must be either 'cost' or 'emissions'")
        if not countries:
            raise ValueError("Must provide at least one country")

        if per_GJ:
            LHV_GJ_per_tonne = get_fuel_LHV(self.fuel)
        else:
            LHV_GJ_per_tonne = 1.0

        # Sort pathways by color and label
        pathways_by_color = defaultdict(list)
        for p in self.pathways:
            color = get_pathway_type_color(get_pathway_type(p))
            pathways_by_color[color].append(p)

        for color in pathways_by_color:
            pathways_by_color[color].sort(key=lambda p: get_pathway_label(p).lower())

        sorted_color_groups = sorted(pathways_by_color.items(), key=lambda item: get_pathway_label(item[1][0]).lower())
        sorted_pathways = [p for _, group in sorted_color_groups for p in group]

        num_pathways = len(sorted_pathways)
        num_countries = len(countries)
        fig_height = max(6, (num_pathways + 1) * 0.9)

        fig, ax = plt.subplots(figsize=(20, fig_height))
        bar_width = 0.8 / num_countries
        cmap = plt.get_cmap("tab10")
        country_colors = {country: cmap(i) for i, country in enumerate(countries)}

        # Plot country bars for each pathway (y = 0 to num_pathways-1)
        y_positions = np.arange(num_pathways)
        for i_pathway, pathway_name in enumerate(sorted_pathways):
            pathway_wtt = PathwayWTG(self.fuel, pathway_name, self.cost_emissions_df)
            data_df = pathway_wtt.get_all_summed_results(quantity)

            for j, country in enumerate(countries):
                match = data_df[data_df["Region"] == country]
                if not match.empty:
                    value_per_kg = match.iloc[0][quantity]
                    value_converted = value_per_kg / LHV_GJ_per_tonne
                else:
                    value_converted = 0

                ax.barh(
                    y=i_pathway + j * bar_width - (bar_width * (num_countries - 1) / 2),
                    width=value_converted,
                    height=bar_width * 0.9,
                    color=country_colors[country],
                    edgecolor="black",
                    label=country if i_pathway == 0 else None,
                )

        # LSFO Reference bar at the top (after all pathways)
        lsfo_y = num_pathways
        lsfo_label = "LSFO"
        lsfo_color = "gray"
        lsfo_LHV_GJ_per_tonne = get_fuel_LHV("lsfo")

        if quantity == "emissions":
            lsfo_value = 0.56
            if per_GJ:
                lsfo_value /= lsfo_LHV_GJ_per_tonne
            ax.barh(
                y=lsfo_y,
                width=lsfo_value,
                height=bar_width * 0.9,
                color=lsfo_color,
                edgecolor="black",
                label="LSFO"
            )
            ax.axvline(x=lsfo_value, color='black', ls='--')

        elif quantity == "cost":
            lsfo_vals = [634.942, 553.707]
            avg_cost = sum(lsfo_vals) / 2
            error = abs(lsfo_vals[0] - lsfo_vals[1]) / 2
            if per_GJ:
                avg_cost /= lsfo_LHV_GJ_per_tonne
                error /= lsfo_LHV_GJ_per_tonne
            ax.barh(
                y=lsfo_y,
                width=avg_cost,
                height=bar_width * 0.9,
                color=lsfo_color,
                edgecolor="black",
                label="LSFO"
            )
            ax.errorbar(
                x=avg_cost,
                y=lsfo_y,
                xerr=error,
                fmt='none',
                ecolor='black',
                capsize=5,
                linewidth=1.5
            )
            ax.axvline(x=avg_cost, color='black', ls='--')

        # Set y-ticks and labels
        full_y_positions = np.append(y_positions, lsfo_y)
        full_labels = [get_pathway_label(p) for p in sorted_pathways] + [lsfo_label]
        ax.set_yticks(full_y_positions)
        ax.set_yticklabels(full_labels, fontsize=22)

        # Correctly map tick labels to y-positions and apply styles
        tick_labels = ax.get_yticklabels()
        tick_positions = ax.get_yticks()
        y_to_label = dict(zip(tick_positions, tick_labels))

        # Style LSFO
        y_to_label[lsfo_y].set_color(lsfo_color)
        y_to_label[lsfo_y].set_fontweight("bold")

        # Style pathways
        for i, p in enumerate(sorted_pathways):
            y_val = y_positions[i]
            label = y_to_label[y_val]
            label.set_color(get_pathway_type_color(get_pathway_type(p)))
            label.set_fontweight("bold")

        ax.axvline(0, color="black", linewidth=1)

        units = {
            ("cost", False): "USD / tonne fuel",
            ("cost", True): "USD / GJ fuel",
            ("emissions", False): "kg CO2e / kg fuel",
            ("emissions", True): "kg CO2e / GJ fuel"
        }[(quantity, per_GJ)]

        ax.set_xlabel(f"{quantity.title()} ({units})", fontsize=24)
        ax.set_title(f"{self.fuel_label}", fontsize=26)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        handles = [Patch(color=country_colors[c], label=c) for c in countries]
        ax.legend(handles=handles, fontsize=20, title="Country", title_fontsize=24, bbox_to_anchor=(1.01, 0.5), loc="center left")

        plt.tight_layout()
        plt.subplots_adjust(left=0.27, right=0.8)

        countries_str = "_".join(c.replace(" ", "") for c in countries)
        suffix = "perGJ" if per_GJ else "perkg"
        filename = f"{self.fuel}-{quantity}-WTG_country_bars-{suffix}-{countries_str}"
        top_dir = get_top_dir()
        create_directory_if_not_exists(f"{top_dir}/plots/{self.fuel}")
        plt.savefig(f"{top_dir}/plots/{self.fuel}/{filename}.png", dpi=200)
        plt.savefig(f"{top_dir}/plots/{self.fuel}/{filename}.pdf")
        plt.close()
        print(f"Saved to: {top_dir}/plots/{self.fuel}/{filename}.png")


def main():

    for fuel in [
        "compressed_hydrogen",
        "liquid_hydrogen",
        "ammonia",
        "methanol",
        "FTdiesel",
        "lng",
        "lsng",
    ]:
        fuel_wtt = FuelWTG(fuel)
        fuel_wtt.make_stacked_hist("emissions")
        fuel_wtt.make_stacked_hist("cost")
#        fuel_wtt.plot_country_bars("emissions", ["Singapore", "Netherlands"], per_GJ=True)
#        fuel_wtt.plot_country_bars("cost", ["Singapore", "Netherlands"], per_GJ=True)

main()

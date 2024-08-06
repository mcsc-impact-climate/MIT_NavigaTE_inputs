"""
Date: June 4, 2024
Author: danikam
Purpose: Reads and plots vessel-level output data from NavigaTE runs to compare lifecycle costs, emissions and energy requirements, for full fleet
"""

from common_tools import get_top_dir
import numpy as np
import pandas as pd
from parse import parse
import os
import matplotlib.pyplot as plt
import matplotlib

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap

matplotlib.rc("xtick", labelsize=18)
matplotlib.rc("ytick", labelsize=18)

TONNES_PER_TEU = 14  # Average tons of cargo in one TEU
LB_PER_GAL_LNG = 3.49  # Source: https://www.dco.uscg.mil/Portals/9/DCO%20Documents/5p/LGCNCOE/docs/LNGF-DesignConsiderations.pdf?ver=2019-09-11-142443-480

# Convert LNG density from lb/gal to tonnes per m^3
GAL_PER_M3 = 264.172
LB_PER_TONNE = 2204.62
TONNES_PER_M3_LNG = LB_PER_GAL_LNG * GAL_PER_M3 / LB_PER_TONNE

vessels = {
    "bulk": [
        "bulk_carrier_capesize_ice",
        "bulk_carrier_handy_ice",
        "bulk_carrier_panamax_ice",
    ],
    "container": [
        "container_15000_teu_ice",
        "container_8000_teu_ice",
        "container_3500_teu_ice",
    ],
    "tanker": ["tanker_100k_dwt_ice", "tanker_300k_dwt_ice", "tanker_35k_dwt_ice"],
    "gas_carrier": ["gas_carrier_100k_cbm_ice"],
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

vessel_size_number = {
    "bulk_carrier_capesize_ice": 2013,
    "bulk_carrier_handy_ice": 4186,
    "bulk_carrier_panamax_ice": 6384,
    "container_15000_teu_ice": 464,
    "container_8000_teu_ice": 1205,
    "container_3500_teu_ice": 3896,
    "tanker_100k_dwt_ice": 3673,
    "tanker_300k_dwt_ice": 866,
    "tanker_35k_dwt_ice": 8464,
    "gas_carrier_100k_cbm_ice": "2156",
}

fuel_components = {
    "hydrogen": ["hydrogen (main)", "lsfo (pilot)"],
    "ammonia": ["ammonia (main)", "lsfo (pilot)"],
    "lsfo": ["lsfo (main)"],
}


def read_results(fuel, pathway, country, number, filename, all_results_df):
    if fuel == "lsfo":
        results_df_columns = [
            "Date",
            "Time (days)",
            "TotalEquivalentWTT",
            "TotalEquivalentTTW",
            "TotalEquivalentWTW",
            "Miles",
            "CargoMiles",
            "SpendEnergy",
            "TotalCAPEX",
            "TotalExcludingFuelOPEX",
            "TotalFuelOPEX",
            "ConsumedEnergy_lsfo",
        ]
    else:
        results_df_columns = [
            "Date",
            "Time (days)",
            "TotalEquivalentWTT",
            "TotalEquivalentTTW",
            "TotalEquivalentWTW",
            "Miles",
            "CargoMiles",
            "SpendEnergy",
            "TotalCAPEX",
            "TotalExcludingFuelOPEX",
            "TotalFuelOPEX",
            f"ConsumedEnergy_{fuel}",
            "ConsumedEnergy_lsfo",
        ]

    results = pd.ExcelFile(filename)
    results_df = pd.read_excel(results, "Vessels")

    results_dict = {}
    for vessel_type in vessels:
        for vessel in vessels[vessel_type]:
            results_df_vessel = results_df.filter(regex=f"Date|Time|{vessel}").drop(
                [0, 1, 2]
            )
            results_df_vessel.columns = results_df_columns
            results_df_vessel = results_df_vessel.set_index("Date")

            results_dict["Vessel"] = f"{vessel}_{fuel}"
            results_dict["Fuel"] = fuel
            results_dict["Pathway"] = pathway
            results_dict["Country"] = country
            results_dict["Number"] = number
            results_dict["WTT Emissions (tonnes CO2 / year)"] = float(
                results_df_vessel["TotalEquivalentWTT"].loc["2024-01-01"]
            )
            results_dict["TTW Emissions (tonnes CO2 / year)"] = float(
                results_df_vessel["TotalEquivalentTTW"].loc["2024-01-01"]
            )
            results_dict["WTW Emissions (tonnes CO2 / year)"] = float(
                results_df_vessel["TotalEquivalentWTW"].loc["2024-01-01"]
            )
            results_dict["CAPEX (USD / year)"] = float(
                results_df_vessel["TotalCAPEX"].loc["2024-01-01"]
            )
            results_dict["Fuel Cost (USD / year)"] = float(
                results_df_vessel["TotalFuelOPEX"].loc["2024-01-01"]
            )
            results_dict["Other OPEX (USD / year)"] = float(
                results_df_vessel["TotalExcludingFuelOPEX"].loc["2024-01-01"]
            )
            results_dict["Total Cost (USD / year)"] = (
                float(results_df_vessel["TotalCAPEX"].loc["2024-01-01"])
                + float(results_df_vessel["TotalFuelOPEX"].loc["2024-01-01"])
                + float(results_df_vessel["TotalExcludingFuelOPEX"].loc["2024-01-01"])
            )
            results_dict["Energy Spend (GJ / year)"] = float(
                results_df_vessel["SpendEnergy"].loc["2024-01-01"]
            )
            results_dict["Miles / year"] = float(
                results_df_vessel["Miles"].loc["2024-01-01"]
            )
            if vessel_type == "container":
                results_dict["Cargo tonne-miles / year"] = (
                    float(results_df_vessel["CargoMiles"].loc["2024-01-01"])
                    * TONNES_PER_TEU
                )
            elif vessel_type == "gas_carrier":
                results_dict["Cargo tonne-miles / year"] = (
                    float(results_df_vessel["CargoMiles"].loc["2024-01-01"])
                    * TONNES_PER_M3_LNG
                )
            else:
                results_dict["Cargo tonne-miles / year"] = float(
                    results_df_vessel["CargoMiles"].loc["2024-01-01"]
                )

            if fuel == "lsfo":
                results_dict["Energy Consumed (GJ / year) [main]"] = float(
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"]
                )
                results_dict["Energy Consumed (GJ / year) [pilot]"] = (
                    float(results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"])
                    * 0
                )
            else:
                results_dict["Energy Consumed (GJ / year) [main]"] = float(
                    results_df_vessel[f"ConsumedEnergy_{fuel}"].loc["2024-01-01"]
                )
                results_dict["Energy Consumed (GJ / year) [pilot]"] = float(
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"]
                )

            results_row_df = pd.DataFrame([results_dict])
            all_results_df = pd.concat(
                [all_results_df, results_row_df], ignore_index=True
            )

    return all_results_df


# def collect_all_results_old(top_dir):
#
#    # Collect all data of interest in a dataframe
#    columns = ['Vessel', 'Fuel', 'Pathway', 'Country', 'WTT Emissions (tonnes CO2 / year)', 'TTW Emissions (tonnes CO2 / year)', 'WTW Emissions (tonnes CO2 / year)', 'CAPEX (USD / year)', 'Fuel Cost (USD / year)', 'Other OPEX (USD / year)', 'Total Cost (USD / year)', 'Energy Consumed (GJ / year) [main]', 'Energy Consumed (GJ / year) [pilot]', 'Energy Spend (GJ / year)', 'Miles / year', 'Cargo tonne-miles / year']
#
#    all_results_df = pd.DataFrame(columns=columns)
#
#    results_filename = f'{top_dir}/all_outputs_full_fleet/report_lsfo.xlsx'
#    all_results_df = read_results('lsfo', 'fossil', 'Global', results_filename, all_results_df)
#
#    for fuel in fuels:
#        for blue_pathway in blue_pathways:
#            for blue_country in blue_countries:
#                # Read in the excel file with output results and add them to the dataframe
#                results_filename = f'{top_dir}/all_outputs_full_fleet/report_{fuel}_blue_{blue_pathway}_{blue_country}.xlsx'
#                all_results_df = read_results(fuel, blue_pathway, blue_country, results_filename, all_results_df)
#
#        for grey_pathway in grey_pathways:
#            for grey_country in grey_countries:
#                # Read in the excel file with output results and add them to the dataframe
#                results_filename = f'{top_dir}/all_outputs_full_fleet/report_{fuel}_grey_{grey_pathway}_{grey_country}.xlsx'
#                all_results_df = read_results(fuel, grey_pathway, grey_country, results_filename, all_results_df)
#
#        for electro_pathway in electro_pathways:
#            for electro_country in electro_countries:
#                # Read in the excel file with output results and add them to the dataframe
#                results_filename = f'{top_dir}/all_outputs_full_fleet/report_{fuel}_electro_{electro_pathway}_{electro_country}.xlsx'
#                all_results_df = read_results(fuel, electro_pathway, electro_country, results_filename, all_results_df)
#
#    print(all_results_df)
#
#    return all_results_df


def extract_info_from_filename(filename):
    pattern = "report_{fuel}-{fuel_type}-{pathway}-{country}-{number}.xlsx"
    result = parse(pattern, filename)
    if result:
        return result.named
    return None


def collect_all_results(top_dir):
    files = os.listdir(f"{top_dir}/all_outputs_full_fleet/")
    fuel_pathway_country_tuples = [
        extract_info_from_filename(file)
        for file in files
        if extract_info_from_filename(file)
    ]

    # Collect all data of interest in a dataframe
    columns = [
        "Vessel",
        "Fuel",
        "Pathway",
        "Country",
        "Number",
        "WTT Emissions (tonnes CO2 / year)",
        "TTW Emissions (tonnes CO2 / year)",
        "WTW Emissions (tonnes CO2 / year)",
        "CAPEX (USD / year)",
        "Fuel Cost (USD / year)",
        "Other OPEX (USD / year)",
        "Total Cost (USD / year)",
        "Energy Consumed (GJ / year) [main]",
        "Energy Consumed (GJ / year) [pilot]",
        "Energy Spend (GJ / year)",
        "Miles / year",
        "Cargo tonne-miles / year",
    ]

    all_results_df = pd.DataFrame(columns=columns)

    results_filename = f"{top_dir}/all_outputs_full_fleet/report_lsfo-1.xlsx"
    all_results_df = read_results(
        "lsfo", "fossil", "Global", 1, results_filename, all_results_df
    )

    for fuel_pathway_country in fuel_pathway_country_tuples:
        fuel = fuel_pathway_country["fuel"]
        fuel_type = fuel_pathway_country["fuel_type"]
        pathway = fuel_pathway_country["pathway"]
        country = fuel_pathway_country["country"]
        number = fuel_pathway_country["number"]
        results_filename = f"{top_dir}/all_outputs_full_fleet/report_{fuel}-{fuel_type}-{pathway}-{country}-{number}.xlsx"
        all_results_df = read_results(
            fuel, pathway, country, number, results_filename, all_results_df
        )

    return all_results_df


def add_averages(all_results_df):
    # Convert CAPEX and Other OPEX columns to float
    all_results_df["CAPEX (USD / year)"] = pd.to_numeric(
        all_results_df["CAPEX (USD / year)"], errors="coerce"
    )
    all_results_df["Other OPEX (USD / year)"] = pd.to_numeric(
        all_results_df["Other OPEX (USD / year)"], errors="coerce"
    )

    # Identify numeric columns for mean calculation
    numeric_cols = all_results_df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate the weight for each row based on the number of occurrences of the country for each vessel, fuel, and pathway
    all_results_df["Country Count"] = all_results_df.groupby(
        ["Vessel", "Fuel", "Pathway", "Country"]
    )["Country"].transform("count")
    all_results_df["Weight"] = 1 / all_results_df["Country Count"]

    # Multiply the numeric columns by their respective weights
    weighted_cols = {col: f"{col}_weighted" for col in numeric_cols}
    for col, weighted_col in weighted_cols.items():
        all_results_df[weighted_col] = all_results_df[col] * all_results_df["Weight"]

    # Calculate the weighted sum for each group
    weighted_sum = (
        all_results_df.groupby(["Vessel", "Fuel", "Pathway"])[
            list(weighted_cols.values())
        ]
        .sum()
        .reset_index()
    )

    # Calculate the sum of weights for each group
    weight_sum = (
        all_results_df.groupby(["Vessel", "Fuel", "Pathway"])["Weight"]
        .sum()
        .reset_index()
    )

    # Merge the weighted sums and weight sums
    weighted_avg = weighted_sum.merge(weight_sum, on=["Vessel", "Fuel", "Pathway"])

    # Calculate the weighted average
    for col, weighted_col in weighted_cols.items():
        weighted_avg[col] = weighted_avg[weighted_col] / weighted_avg["Weight"]

    # Add a new column to indicate these are averages
    weighted_avg["Country"] = "Country Average"

    # Select only the required columns
    weighted_avg = weighted_avg[["Vessel", "Fuel", "Pathway", "Country"] + numeric_cols]

    # Append the weighted_avg back to the original DataFrame
    all_results_df = pd.concat([all_results_df, weighted_avg], ignore_index=True)
    all_results_df.loc[all_results_df["Country"] == "Country Average", "Number"] = "1"

    return all_results_df


def add_number_of_vessels(all_results_df):
    # Create a function to extract the base vessel name
    def extract_base_vessel_name(vessel_name):
        # Splits the vessel name at '_' and keeps up to the third component
        return "_".join(vessel_name.split("_")[:-1])

    # Apply the function to the 'Vessel' column to create a new column for mapping
    all_results_df["base_vessel_name"] = all_results_df["Vessel"].apply(
        extract_base_vessel_name
    )

    # Map the number of vessels from the dictionary to the new column
    all_results_df["Number of Vessels"] = (
        all_results_df["base_vessel_name"].map(vessel_size_number).astype(float)
    )

    # Optionally, you can drop the 'base_vessel_name' column if it's not needed
    all_results_df.drop("base_vessel_name", axis=1, inplace=True)

    return all_results_df


def sum_to_fleet(all_results_df, column_sum, filter, column_divide=None):
    if column_divide is None:
        return (
            all_results_df[column_sum][filter]
            * all_results_df["Number of Vessels"][filter]
        ).sum()
    else:
        return (
            all_results_df[column_sum][filter]
            * all_results_df["Number of Vessels"][filter]
            / all_results_df[column_divide][filter]
        ).sum()


def process_emissions_pathway(
    all_results_df,
    fuel,
    pathway,
    countries,
    emissions_average,
    emissions_WTW,
    fuel_pathways,
    per_mile,
    per_cargo_mile,
    unique_country_labels,
):
    emissions_WTW[f"{fuel} ({pathway})"] = {}

    fuel_pathways.append(f"{fuel} ({pathway})")

    if per_mile:
        column_divide = "Miles / year"
    elif per_cargo_mile:
        column_divide = "Cargo tonne-miles / year"
    else:
        column_divide = None

    filter = (
        (all_results_df["Fuel"] == fuel)
        & (all_results_df["Pathway"] == pathway)
        & (all_results_df["Country"] == "Country Average")
    )

    emissions_average["Tank-to-wake"].append(
        sum_to_fleet(
            all_results_df, "TTW Emissions (tonnes CO2 / year)", filter, column_divide
        )
    )

    emissions_average["Well-to-tank"].append(
        sum_to_fleet(
            all_results_df, "WTT Emissions (tonnes CO2 / year)", filter, column_divide
        )
    )

    # Determine countries with multiple numbers across all pathways
    countries_with_multiple_numbers = all_results_df.groupby("Country")[
        "Number"
    ].nunique()
    countries_with_multiple_numbers = countries_with_multiple_numbers[
        countries_with_multiple_numbers > 1
    ].index.tolist()

    for country in countries:
        filter = (
            (all_results_df["Fuel"] == fuel)
            & (all_results_df["Pathway"] == pathway)
            & (all_results_df["Country"] == country)
        )
        numbers = all_results_df[filter]["Number"].unique()

        if len(numbers) == 1:
            number = numbers[0]
            if number == "1" and country not in countries_with_multiple_numbers:
                emissions_WTW[f"{fuel} ({pathway})"][country] = sum_to_fleet(
                    all_results_df,
                    "WTW Emissions (tonnes CO2 / year)",
                    filter,
                    column_divide,
                )
                if country not in unique_country_labels:
                    unique_country_labels.append(country)
            else:
                emissions_WTW[f"{fuel} ({pathway})"][f"{country} (1)"] = sum_to_fleet(
                    all_results_df,
                    "WTW Emissions (tonnes CO2 / year)",
                    filter,
                    column_divide,
                )
                if f"{country} (1)" not in unique_country_labels:
                    unique_country_labels.append(f"{country} (1)")
        elif len(numbers) > 1:
            for number in numbers:
                filter_number = filter & (all_results_df["Number"] == number)
                emissions_WTW[f"{fuel} ({pathway})"][f"{country} ({number})"] = (
                    sum_to_fleet(
                        all_results_df,
                        "WTW Emissions (tonnes CO2 / year)",
                        filter_number,
                        column_divide,
                    )
                )
                if f"{country} ({number})" not in unique_country_labels:
                    unique_country_labels.append(f"{country} ({number})")


def process_costs_pathway(
    all_results_df,
    fuel,
    pathway,
    countries,
    costs_average,
    costs_total,
    fuel_pathways,
    per_mile,
    per_cargo_mile,
    unique_country_labels,
):
    costs_total[f"{fuel} ({pathway})"] = {}

    fuel_pathways.append(f"{fuel} ({pathway})")

    if per_mile:
        column_divide = "Miles / year"
    elif per_cargo_mile:
        column_divide = "Cargo tonne-miles / year"
    else:
        column_divide = None

    filter = (
        (all_results_df["Fuel"] == fuel)
        & (all_results_df["Pathway"] == pathway)
        & (all_results_df["Country"] == "Country Average")
    )

    costs_average["CAPEX"].append(
        sum_to_fleet(all_results_df, "CAPEX (USD / year)", filter, column_divide)
    )
    costs_average["Other OPEX"].append(
        sum_to_fleet(all_results_df, "Other OPEX (USD / year)", filter, column_divide)
    )
    costs_average["Fuel Cost"].append(
        sum_to_fleet(all_results_df, "Fuel Cost (USD / year)", filter, column_divide)
    )

    # Determine countries with multiple numbers across all pathways
    countries_with_multiple_numbers = all_results_df.groupby("Country")[
        "Number"
    ].nunique()
    countries_with_multiple_numbers = countries_with_multiple_numbers[
        countries_with_multiple_numbers > 1
    ].index.tolist()

    for country in countries:
        filter = (
            (all_results_df["Fuel"] == fuel)
            & (all_results_df["Pathway"] == pathway)
            & (all_results_df["Country"] == country)
        )
        numbers = all_results_df[filter]["Number"].unique()

        if len(numbers) == 1:
            number = numbers[0]
            if number == "1" and country not in countries_with_multiple_numbers:
                costs_total[f"{fuel} ({pathway})"][country] = sum_to_fleet(
                    all_results_df, "Total Cost (USD / year)", filter, column_divide
                )
                if country not in unique_country_labels:
                    unique_country_labels.append(country)
            else:
                costs_total[f"{fuel} ({pathway})"][f"{country} (1)"] = sum_to_fleet(
                    all_results_df, "Total Cost (USD / year)", filter, column_divide
                )
                if f"{country} (1)" not in unique_country_labels:
                    unique_country_labels.append(f"{country} (1)")
        elif len(numbers) > 1:
            for number in numbers:
                filter_number = filter & (all_results_df["Number"] == number)
                costs_total[f"{fuel} ({pathway})"][f"{country} ({number})"] = (
                    sum_to_fleet(
                        all_results_df,
                        "Total Cost (USD / year)",
                        filter_number,
                        column_divide,
                    )
                )
                if f"{country} ({number})" not in unique_country_labels:
                    unique_country_labels.append(f"{country} ({number})")


def process_fuel_energy_consumed(
    all_results_df, fuel, countries, fuel_energy_consumed, per_mile, per_cargo_mile
):
    fuel_energy_consumed[fuel] = {}

    if per_mile:
        column_divide = "Miles / year"
    elif per_cargo_mile:
        column_divide = "Cargo tonne-miles / year"
    else:
        column_divide = None

    fuel_energy_consumed[fuel]["main"] = sum_to_fleet(
        all_results_df, "Energy Consumed (GJ / year) [main]", filter, column_divide
    )

    fuel_energy_consumed[fuel]["pilot"] = sum_to_fleet(
        all_results_df, "Energy Consumed (GJ / year) [pilot]", filter, column_divide
    )


def plot_bar_stacked_stages(
    property_average, property_total, fuel_pathways, unique_country_labels
):
    num_pathways = len(fuel_pathways)
    fig_height = max(6, num_pathways * 0.5)  # Adjust this factor as needed

    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.axvline(
        property_total["lsfo (fossil)"]["Country Average"], color="black", ls="--"
    )
    bottom = np.zeros(len(fuel_pathways))
    width = 0.5

    fuel_pathways_label = [
        fuel_pathway.replace("_", "-") for fuel_pathway in fuel_pathways
    ]

    for property_type, property in property_average.items():
        ax.barh(
            fuel_pathways_label,
            property,
            width,
            label=property_type + " (Avg)",
            left=bottom,
            alpha=0.4,
        )
        bottom += property

    # Create a color map with distinct colors and map each country to a unique color
    base_cmap = plt.get_cmap("tab20")
    cmap = ListedColormap(base_cmap.colors[: len(unique_country_labels)])
    colours = [cmap(i) for i in range(len(unique_country_labels))]
    country_color_map = {
        country: colour for country, colour in zip(unique_country_labels, colours)
    }

    # Overlay WTW for individual countries if there's significant spread between countries
    countries_labelled = []
    for pathway, countries in property_total.items():
        index = fuel_pathways.index(pathway)
        totals = np.asarray(list(property_total[pathway].values()))
        if np.std(totals) / np.mean(totals) > 0.01:
            for country in countries:
                # Check if the country's value is not zero or NaN
                if property_total[pathway][country] > 0:
                    if country == "Country Average":
                        continue
                    elif country in countries_labelled:
                        ax.scatter(
                            property_total[pathway][country],
                            index,
                            s=50,
                            color=country_color_map[country],
                            marker="D",
                        )
                    else:
                        ax.scatter(
                            property_total[pathway][country],
                            index,
                            s=50,
                            color=country_color_map[country],
                            marker="D",
                            label=f"{country}",
                        )
                        countries_labelled.append(country)

    # Place the legend outside the plot
    ax.legend(fontsize=16, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    return fig, ax


def plot_emissions(all_results_df, per_mile=False, per_cargo_mile=False):
    unique_countries = all_results_df["Country"].unique()
    fuel_pathways = []
    emissions_average = {"Tank-to-wake": [], "Well-to-tank": []}
    emissions_WTW = {}
    unique_country_labels = []

    # Sum emissions over all vessel types
    process_emissions_pathway(
        all_results_df,
        "lsfo",
        "fossil",
        ["Country Average"],
        emissions_average,
        emissions_WTW,
        fuel_pathways,
        per_mile,
        per_cargo_mile,
        unique_country_labels,
    )

    unique_fuels = all_results_df["Fuel"].unique()

    for fuel in unique_fuels:
        applicable_pathways = all_results_df[all_results_df["Fuel"] == fuel][
            "Pathway"
        ].unique()
        for pathway in applicable_pathways:
            if pathway != "fossil":
                process_emissions_pathway(
                    all_results_df,
                    fuel,
                    pathway,
                    unique_countries,
                    emissions_average,
                    emissions_WTW,
                    fuel_pathways,
                    per_mile,
                    per_cargo_mile,
                    unique_country_labels,
                )

    fig, ax = plot_bar_stacked_stages(
        emissions_average, emissions_WTW, fuel_pathways, unique_country_labels
    )

    if per_mile:
        ax.set_xlabel("CO$_2$e Emissions (tonnes / mile)", fontsize=22)
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)  # Adjust this to make space for the legend
        plt.savefig("plots/all_emissions_full_fleet_per_mile.png", dpi=300)
    elif per_cargo_mile:
        ax.set_xlabel("CO$_2$e Emissions (tonnes / cargo tonne-mile)", fontsize=22)
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)  # Adjust this to make space for the legend
        plt.savefig("plots/all_emissions_full_fleet_per_ton_mile.png", dpi=300)
    else:
        ax.set_xlabel("CO$_2$e Emissions (tonnes / year)", fontsize=22)
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)  # Adjust this to make space for the legend
        plt.savefig("plots/all_emissions_full_fleet.png", dpi=300)
    plt.close()


def plot_emissions_vessel_type(all_results_df, per_mile=False, per_cargo_mile=False):
    fig, ax = plt.subplots(figsize=(18, 10))

    # Define unique fuel pathways
    unique_pathways = all_results_df["Fuel"] + " (" + all_results_df["Pathway"] + ")"
    unique_vessels = all_results_df["Vessel"].unique()

    # Simplify vessel names by using only the first word
    simplified_vessel_names = {v: v.split("_")[0] for v in unique_vessels}
    all_results_df["Simplified Vessel"] = all_results_df["Vessel"].map(
        simplified_vessel_names
    )

    # Initialize data storage for plot
    data_for_plot = {
        pathway: {"TTW": [], "WTT": [], "Labels": []}
        for pathway in unique_pathways.unique()
    }

    if per_mile:
        column_divide = "Miles / year"
    elif per_cargo_mile:
        column_divide = "Cargo tonne-miles / year"
    else:
        column_divide = None

    # Loop over each unique pathway
    for pathway in unique_pathways.unique():
        for vessel_type in set(simplified_vessel_names.values()):
            filter = (
                (
                    all_results_df["Fuel"] + " (" + all_results_df["Pathway"] + ")"
                    == pathway
                )
                & (all_results_df["Simplified Vessel"] == vessel_type)
                & (all_results_df["Country"] == "Country Average")
            )

            if not all_results_df[filter].empty:
                ttw_sum = sum_to_fleet(
                    all_results_df,
                    "TTW Emissions (tonnes CO2 / year)",
                    filter,
                    column_divide,
                )
                wtt_sum = sum_to_fleet(
                    all_results_df,
                    "WTT Emissions (tonnes CO2 / year)",
                    filter,
                    column_divide,
                )
                data_for_plot[pathway]["TTW"].append(ttw_sum)
                data_for_plot[pathway]["WTT"].append(wtt_sum)
                data_for_plot[pathway]["Labels"].append(vessel_type)

    # Define colors and bar width
    num_colors = len(set(simplified_vessel_names.values()))
    colors = [cm.Accent(x) for x in range(num_colors)]
    bar_width = 0.6

    # Plotting
    index = np.arange(len(data_for_plot)) * 2
    for i, (pathway, data) in enumerate(data_for_plot.items()):
        ttw_left = np.zeros(len(index))
        wtt_left = np.zeros(len(index))

        for ttw, wtt, color, label in zip(
            data["TTW"], data["WTT"], colors, data["Labels"]
        ):
            ax.barh(i * 2 + bar_width, wtt, bar_width, left=wtt_left[i], color=color)
            ax.barh(i * 2, ttw, bar_width, left=ttw_left[i], color=color, hatch="///")
            ttw_left[i] += ttw
            wtt_left[i] += wtt

    # Set labels and legends
    ax.set_ylabel("Fuel Pathways", fontsize=22)
    ax.set_xlabel("CO$_2$ Emissions (tonnes/year)", fontsize=22)
    ax.set_title(
        "Stacked TTW and WTT Emissions by Vessel Type and Fuel Pathway", fontsize=24
    )
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels([path.replace(" (", "(") for path in data_for_plot.keys()])

    # Create a custom legend for the vessel types
    from matplotlib.patches import Patch

    legend_elements_vessel = [
        Patch(facecolor=col, label=label)
        for col, label in zip(colors, set(simplified_vessel_names.values()))
    ]
    legend_vessel = ax.legend(
        handles=legend_elements_vessel,
        title="Vessel Types",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=18,
        title_fontsize=22,
    )

    # Add the first legend manually to the axes
    ax.add_artist(legend_vessel)

    # Create a custom legend for the emission stages
    legend_elements_stage = [
        Patch(facecolor="white", edgecolor="black", label="Well-to-Tank"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Tank-to-Wake"),
    ]
    ax.legend(
        handles=legend_elements_stage,
        title="Emission Phases",
        bbox_to_anchor=(1.05, 0.7),
        loc="upper left",
        fontsize=18,
        title_fontsize=22,
    )

    if per_mile:
        ax.set_xlabel("CO$_2$e Emissions (tonnes / mile)", fontsize=22)
        plt.tight_layout()
        plt.savefig(
            "plots/all_emissions_full_fleet_vessels_stacked_per_mile.png", dpi=300
        )
        plt.close()
    elif per_cargo_mile:
        ax.set_xlabel("CO$_2$e Emissions (tonnes / cargo tonne-mile)", fontsize=22)
        plt.tight_layout()
        plt.savefig(
            "plots/all_emissions_full_fleet_vessels_stacked_per_ton_mile.png", dpi=300
        )
        plt.close()
    else:
        ax.set_xlabel("CO$_2$e Emissions (tonnes / year)", fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/all_emissions_full_fleet_vessels_stacked.png", dpi=300)
        plt.close()


def plot_costs(all_results_df, per_mile=False, per_cargo_mile=False):
    unique_countries = all_results_df["Country"].unique()
    fuel_pathways = []
    costs_average = {
        "CAPEX": [],
        "Other OPEX": [],
        "Fuel Cost": [],
    }
    costs_total = {}
    unique_country_labels = []

    # Sum emissions over all vessel types
    process_costs_pathway(
        all_results_df,
        "lsfo",
        "fossil",
        ["Country Average"],
        costs_average,
        costs_total,
        fuel_pathways,
        per_mile,
        per_cargo_mile,
        unique_country_labels,
    )

    unique_fuels = all_results_df["Fuel"].unique()

    for fuel in unique_fuels:
        applicable_pathways = all_results_df[all_results_df["Fuel"] == fuel][
            "Pathway"
        ].unique()
        for pathway in applicable_pathways:
            if pathway != "fossil":
                process_costs_pathway(
                    all_results_df,
                    fuel,
                    pathway,
                    unique_countries,
                    costs_average,
                    costs_total,
                    fuel_pathways,
                    per_mile,
                    per_cargo_mile,
                    unique_country_labels,
                )

    fig, ax = plot_bar_stacked_stages(
        costs_average, costs_total, fuel_pathways, unique_country_labels
    )

    if per_mile:
        ax.set_xlabel("Cost (USD / mile)", fontsize=22)
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)  # Adjust this to make space for the legend
        plt.savefig("plots/all_costs_full_fleet_per_mile.png", dpi=300)
    elif per_cargo_mile:
        ax.set_xlabel("Cost (USD / cargo tonne-mile)", fontsize=22)
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)  # Adjust this to make space for the legend
        plt.savefig("plots/all_costs_full_fleet_per_ton_mile.png", dpi=300)
    else:
        ax.set_xlabel("Cost (USD / year)", fontsize=22)
        plt.tight_layout()
        plt.subplots_adjust(right=0.8)  # Adjust this to make space for the legend
        plt.savefig("plots/all_costs_full_fleet.png", dpi=300)
    plt.close()


def plot_fuel_energy_consumed(all_results_df, per_mile=False, per_cargo_mile=False):
    # Drop fuel pathway info from the results dataframe since it's not relevant for the fuel energy consumed
    all_results_df_no_pathway_info = all_results_df.drop_duplicates(
        subset=["Vessel", "Fuel", "Country"]
    ).drop(columns=["Pathway"])

    fuel_energy_consumed = {}

    fuels = all_results_df["Fuel"].unique()  # DMM: Not sure about this, may need to fix
    for fuel in fuels + ["lsfo"]:
        process_fuel_energy_consumed(
            all_results_df_no_pathway_info,
            fuel,
            ["Country Average"],
            fuel_energy_consumed,
            per_mile,
            per_cargo_mile,
        )

    # Plot the energy consumed for each fuel, stacking the pilot and main fuel components
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axhline(fuel_energy_consumed["lsfo"]["main"], color="black", ls="--")

    # Define the x-axis locations for the groups
    fuel_pathways = list(fuel_energy_consumed.keys())
    x = np.arange(len(fuel_pathways))

    # Width of the bars
    width = 0.5

    # Plotting the 'pilot' section
    pilot_values = [fuel_energy_consumed[fuel]["pilot"] for fuel in fuel_pathways]
    ax.bar(x, pilot_values, width, label="Pilot", color="lightblue")

    # Plotting the 'main' section on top of 'pilot'
    main_values = [fuel_energy_consumed[fuel]["main"] for fuel in fuel_pathways]
    ax.bar(x, main_values, width, bottom=pilot_values, label="Main", color="blue")
    ax.set_xticks(x)
    ax.set_xticklabels(fuel_pathways)
    ymin, ymax = ax.get_ylim()
    ymax = ymax * 1.2
    ax.set_ylim(ymin, ymax)

    ax.legend(fontsize=20)

    if per_mile:
        ax.set_ylabel("Fuel Energy Consumed (GJ / year)", fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/energy_consumed_full_fleet_per_mile.png", dpi=300)
        plt.close()
    elif per_cargo_mile:
        ax.set_ylabel("Fuel Energy Consumed (GJ / cargo tonne-mile)", fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/energy_consumed_full_fleet_per_ton_mile.png", dpi=300)
        plt.close()
    else:
        ax.set_ylabel("Fuel Energy Consumed (GJ / year)", fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/energy_consumed_full_fleet.png", dpi=300)
        plt.close()


def plot_costs_vessel_type(all_results_df, per_mile=False, per_cargo_mile=False):
    fig, ax = plt.subplots(figsize=(18, 10))

    # Define unique fuel pathways
    unique_pathways = all_results_df["Fuel"] + " (" + all_results_df["Pathway"] + ")"
    unique_vessels = all_results_df["Vessel"].unique()

    # Simplify vessel names by using only the first word
    simplified_vessel_names = {v: v.split("_")[0] for v in unique_vessels}
    all_results_df["Simplified Vessel"] = all_results_df["Vessel"].map(
        simplified_vessel_names
    )

    # Initialize data storage for plot
    data_for_plot = {
        pathway: {"CAPEX": [], "Other OPEX": [], "Fuel Cost": [], "Labels": []}
        for pathway in unique_pathways.unique()
    }

    # Determine the divisor to put on the bottom of the summed costs (if any)
    if per_mile:
        column_divide = "Miles / year"
    elif per_cargo_mile:
        column_divide = "Cargo tonne-miles / year"
    else:
        column_divide = None

    # Loop over each unique pathway
    for pathway in unique_pathways.unique():
        for vessel_type in set(simplified_vessel_names.values()):
            filter = (
                (
                    all_results_df["Fuel"] + " (" + all_results_df["Pathway"] + ")"
                    == pathway
                )
                & (all_results_df["Simplified Vessel"] == vessel_type)
                & (all_results_df["Country"] == "Country Average")
            )

            if not all_results_df[filter].empty:
                capex_sum = sum_to_fleet(
                    all_results_df, "CAPEX (USD / year)", filter, column_divide
                )
                other_opex_sum = sum_to_fleet(
                    all_results_df, "Other OPEX (USD / year)", filter, column_divide
                )
                fuel_cost_sum = sum_to_fleet(
                    all_results_df, "Fuel Cost (USD / year)", filter, column_divide
                )
                data_for_plot[pathway]["CAPEX"].append(capex_sum)
                data_for_plot[pathway]["Other OPEX"].append(other_opex_sum)
                data_for_plot[pathway]["Fuel Cost"].append(fuel_cost_sum)
                data_for_plot[pathway]["Labels"].append(vessel_type)

    # Define colors and bar width
    num_colors = len(set(simplified_vessel_names.values()))
    colors = [cm.Accent(x) for x in range(num_colors)]
    bar_width = 0.5

    # Plotting
    index = np.arange(len(data_for_plot)) * 2
    for i, (pathway, data) in enumerate(data_for_plot.items()):
        capex_left = np.zeros(len(index))
        other_opex_left = np.zeros(len(index))
        fuel_cost_left = np.zeros(len(index))

        for capex, other_opex, fuel_cost, color, label in zip(
            data["CAPEX"], data["Other OPEX"], data["Fuel Cost"], colors, data["Labels"]
        ):
            ax.barh(
                i * 2 + bar_width * 1.5,
                capex,
                bar_width,
                left=capex_left[i],
                color=color,
            )
            ax.barh(
                i * 2 + bar_width / 2,
                other_opex,
                bar_width,
                left=other_opex_left[i],
                color=color,
                hatch="///",
            )
            ax.barh(
                i * 2 - bar_width / 2,
                fuel_cost,
                bar_width,
                left=fuel_cost_left[i],
                color=color,
                hatch="xxx",
            )
            capex_left[i] += capex
            other_opex_left[i] += other_opex
            fuel_cost_left[i] += fuel_cost

    # Set labels and legends
    ax.set_ylabel("Fuel Pathways", fontsize=22)
    ax.set_title("Stacked Costs by Vessel Type and Fuel Pathway", fontsize=24)
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels([path.replace(" (", "(") for path in data_for_plot.keys()])

    # Create a custom legend for the vessel types
    from matplotlib.patches import Patch

    legend_elements_vessel = [
        Patch(facecolor=col, label=label)
        for col, label in zip(colors, set(simplified_vessel_names.values()))
    ]
    legend_vessel = ax.legend(
        handles=legend_elements_vessel,
        title="Vessel Types",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=18,
        title_fontsize=22,
    )

    # Add the first legend manually to the axes
    ax.add_artist(legend_vessel)

    # Create a custom legend for the emission stages
    legend_elements_stage = [
        Patch(facecolor="white", edgecolor="black", label="CAPEX"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="Other OPEX"),
        Patch(facecolor="white", edgecolor="black", hatch="xxx", label="Fuel Cost"),
    ]
    ax.legend(
        handles=legend_elements_stage,
        title="Cost Components",
        bbox_to_anchor=(1.05, 0.7),
        loc="upper left",
        fontsize=18,
        title_fontsize=22,
    )

    if per_mile:
        ax.set_xlabel("Costs (USD / mile)", fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/all_costs_full_fleet_vessels_stacked_per_mile.png", dpi=300)
        plt.close()
    elif per_cargo_mile:
        ax.set_xlabel("Costs (USD / cargo tonne-mile)", fontsize=22)
        plt.tight_layout()
        plt.savefig(
            "plots/all_costs_full_fleet_vessels_stacked_per_ton_mile.png", dpi=300
        )
        plt.close()
    else:
        ax.set_xlabel("Costs (USD / year)", fontsize=22)
        plt.tight_layout()
        plt.savefig("plots/all_costs_full_fleet_vessels_stacked.png", dpi=300)
        plt.close()


def compare_vessel_property(filepath, property, property_label, property_unit):
    results = pd.ExcelFile(filepath)
    results_df = pd.read_excel(results, "Vessels")
    results_df_columns = [
        "Date",
        "Time (days)",
        "TotalEquivalentWTT",
        "TotalEquivalentTTW",
        "TotalEquivalentWTW",
        "Miles",
        "CargoMiles",
        "SpendEnergy",
        "TotalCAPEX",
        "TotalExcludingFuelOPEX",
        "TotalFuelOPEX",
        "ConsumedEnergy_lsfo",
    ]
    all_results_df = pd.DataFrame(columns=["Vessel", property])

    results_dict = {}
    for vessel_type in vessels:
        for vessel in vessels[vessel_type]:
            results_df_vessel = results_df.filter(regex=f"Date|Time|{vessel}").drop(
                [0, 1, 2]
            )
            results_df_vessel = results_df_vessel
            results_df_vessel.columns = results_df_columns
            results_df_vessel = results_df_vessel.set_index("Date")

            results_dict["Vessel"] = f"{vessel}"
            if property == "CargoMiles" and vessel_type == "container":
                results_dict[property] = (
                    results_df_vessel[property].loc["2024-01-01"] * TONNES_PER_TEU
                )
            else:
                results_dict[property] = results_df_vessel[property].loc["2024-01-01"]
            results_row_df = pd.DataFrame([results_dict])
            all_results_df = pd.concat(
                [all_results_df, results_row_df], ignore_index=True
            )

    # Categorize vessels by the first part of their name
    all_results_df["Category"] = all_results_df["Vessel"].apply(
        lambda x: x.split("_")[0].title()
    )

    # Aggregate miles by category
    category_miles = all_results_df.groupby("Category")[property].sum()

    # Make a bar plot of miles traveled for each vessel
    color = ["blue", "purple", "red"]
    blues = cm.Blues(np.linspace(0.3, 0.9, len(color)))
    purples = cm.Purples(np.linspace(0.3, 0.9, len(color)))
    reds = cm.Reds(np.linspace(0.3, 0.9, len(color)))
    color_gradient = [blues, purples, reds]

    fig, ax = plt.subplots(figsize=(10, 6))
    category_miles.plot(
        kind="bar", ax=ax, color=["blue", "purple", "red"]
    )  # Colors for bulk_carrier, container, tanker
    ax.set_title(f"{property_label} by Vessel Class", fontsize=24)
    ax.set_xlabel("Vessel Class", fontsize=20)
    ax.set_ylabel(f"{property_label} ({property_unit})", fontsize=20)
    ax.set_xticklabels(category_miles.index, rotation=0)

    plt.tight_layout()
    plt.savefig(f"plots/vessel_{property}_split.png")
    plt.close()

    # Iterate over each vessel class and create a separate pie chart
    i = 0
    for category, vessel_list in vessels.items():
        fig, ax = plt.subplots(figsize=(10, 7))  # Create a new figure for each category
        category_data = all_results_df[all_results_df["Vessel"].isin(vessel_list)]
        ax.pie(
            category_data[property],
            labels=category_data["Vessel"].apply(lambda x: vessel_size_title[x]),
            autopct="%1.1f%%",
            startangle=90,
            colors=color_gradient[i],
            textprops={"fontsize": 20},
        )
        ax.set_title(
            f"{category.capitalize()} {property_label} by Size Class", fontsize=24
        )
        plt.savefig(f"plots/{category}_{property}_split.png")
        plt.close()
        i += 1


def main():
    # Get the path to the top level of the Git repo
    top_dir = get_top_dir()

    all_results_df = collect_all_results(top_dir)

    all_results_df = add_averages(all_results_df)

    all_results_df = add_number_of_vessels(all_results_df)

    plot_emissions(all_results_df)
    plot_emissions(all_results_df, per_mile=True)
    plot_emissions(all_results_df, per_cargo_mile=True)

    plot_costs(all_results_df)
    plot_costs(all_results_df, per_mile=True)
    plot_costs(all_results_df, per_cargo_mile=True)


#    plot_fuel_energy_consumed(all_results_df)
#
#    plot_emissions_vessel_type(all_results_df)
#    plot_emissions_vessel_type(all_results_df, per_mile=True)
#    plot_emissions_vessel_type(all_results_df, per_cargo_mile=True)
#
#    plot_costs_vessel_type(all_results_df)
#    plot_costs_vessel_type(all_results_df, per_mile=True)
#    plot_costs_vessel_type(all_results_df, per_cargo_mile=True)
#
#    compare_vessel_property(f'{top_dir}/all_outputs_full_fleet/report_lsfo.xlsx', 'Miles', 'Annual Miles', 'miles')
#    compare_vessel_property(f'{top_dir}/all_outputs_full_fleet/report_lsfo.xlsx', 'CargoMiles', 'Annual Cargo Miles', 'ton-miles')
#    compare_vessel_property(f'{top_dir}/all_outputs_full_fleet/report_lsfo.xlsx', 'SpendEnergy', 'Annual Energy Demand', 'GJ')

main()

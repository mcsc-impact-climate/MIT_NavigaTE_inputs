"""
Date: 241118
Author: danikam
Purpose: Compare cost per mile and per m^3 of cargo after accounting for boil-off and tank size correction.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from common_tools import get_top_dir, create_directory_if_not_exists, generate_blue_shades, get_pathway_label, read_pathway_labels, read_fuel_labels, get_fuel_label
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from parse import parse

def generate_orange_shades(num_shades):
    """
    Generates a list of orange shades ranging from light to dark.

    Parameters
    ----------
    num_shades : int
        The number of orange shades to generate.

    Returns
    -------
    orange_shades : list of str
        A list of orange shades in hex format, ranging from light to dark.
    """
    light_orange = mcolors.to_rgba("#FFDAB9")  # Light orange
    dark_orange = mcolors.to_rgba("#FF8C00")  # Dark orange

    if num_shades > 1:
        orange_shades = [
            mcolors.to_hex(
                (
                    dark_orange[0] * (1 - i / (num_shades - 1))
                    + light_orange[0] * (i / (num_shades - 1)),
                    dark_orange[1] * (1 - i / (num_shades - 1))
                    + light_orange[1] * (i / (num_shades - 1)),
                    dark_orange[2] * (1 - i / (num_shades - 1))
                    + light_orange[2] * (i / (num_shades - 1)),
                    1.0,
                )
            )
            for i in range(num_shades)
        ]
    else:
        orange_shades = [light_orange]

    return orange_shades

def generate_purple_shades(num_shades):
    """
    Generates a list of purple shades ranging from light to dark.

    Parameters
    ----------
    num_shades : int
        The number of purple shades to generate.

    Returns
    -------
    purple_shades : list of str
        A list of purple shades in hex format, ranging from light to dark.
    """
    light_purple = mcolors.to_rgba("#E6E6FA")  # Light purple
    dark_purple = mcolors.to_rgba("#800080")  # Dark purple

    if num_shades > 1:
        purple_shades = [
            mcolors.to_hex(
                (
                    dark_purple[0] * (1 - i / (num_shades - 1))
                    + light_purple[0] * (i / (num_shades - 1)),
                    dark_purple[1] * (1 - i / (num_shades - 1))
                    + light_purple[1] * (i / (num_shades - 1)),
                    dark_purple[2] * (1 - i / (num_shades - 1))
                    + light_purple[2] * (i / (num_shades - 1)),
                    1.0,
                )
            )
            for i in range(num_shades)
        ]
    else:
        purple_shades = [light_purple]

    return purple_shades

def generate_grey_shades(num_shades):
    """
    Generates a list of grey shades ranging from light to dark.

    Parameters
    ----------
    num_shades : int
        The number of grey shades to generate.

    Returns
    -------
    grey_shades : list of str
        A list of grey shades in hex format, ranging from light to dark.
    """
    light_grey = mcolors.to_rgba("#D3D3D3")  # Light grey
    dark_grey = mcolors.to_rgba("#696969")  # Dark grey

    if num_shades > 1:
        grey_shades = [
            mcolors.to_hex(
                (
                    dark_grey[0] * (1 - i / (num_shades - 1))
                    + light_grey[0] * (i / (num_shades - 1)),
                    dark_grey[1] * (1 - i / (num_shades - 1))
                    + light_grey[1] * (i / (num_shades - 1)),
                    dark_grey[2] * (1 - i / (num_shades - 1))
                    + light_grey[2] * (i / (num_shades - 1)),
                    1.0,
                )
            )
            for i in range(num_shades)
        ]
    else:
        grey_shades = [light_grey]

    return grey_shades


# Constants
RESULTS_DIR_NO_BOILOFF = "processed_results_no_boiloff"
RESULTS_DIR_WITH_BOILOFF = "processed_results_with_boiloff"
VESSEL_TYPES = [
    "bulk_carrier_ice",
    "container_ice",
    "tanker_ice",
    "gas_carrier_ice",
]
VESSEL_TYPE_TITLES = {
    "bulk_carrier_ice": "Bulk Carrier",
    "container_ice": "Container",
    "tanker_ice": "Tanker",
    "gas_carrier_ice": "Gas Carrier",
}
COLORS = {
    "no_boiloff": generate_blue_shades(3),
    "with_boiloff": generate_orange_shades(3),
    "tank_corrected": generate_purple_shades(3),
}
LABELS = [
    "No boiloff or tank size correction",
    "With boiloff, no tank size correction",
    "With boiloff, with tank size correction",
]

def read_processed_data(results_dir, fuel, pathway, quantity):
    """
    Reads the processed CSV file for the given fuel, pathway, and quantity.
    
    Parameters:
    ----------
    results_dir : str
        Directory containing the processed results
    fuel : str
        Fuel name
    pathway : str
        Production pathway
    quantity : str
        Quantity to read

    Returns:
    -------
    pd.DataFrame
        Processed data as a pandas DataFrame
    """
    file_path = f"{get_top_dir()}/{results_dir}/{fuel}-{pathway}-{quantity}.csv"
    return pd.read_csv(file_path, index_col=0)

def plot_histogram_for_vessel_types(fuel, pathway, quantity="TotalCost", modifier="per_tonne_mile"):
    """
    Plots a histogram comparing TotalCost per tonne mile between all vessel types for a given fuel and pathway.
    
    Parameters:
    ----------
    fuel : str
        Fuel name
    pathway : str
        Production pathway
    quantity : str
        Quantity to plot (default: "TotalCost")
    modifier : str
        Modifier for the quantity (default: "per_tonne_mile")
    """
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Main histogram plot
    ax_hist = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_hist)

    x_labels = []
    x_positions = []
    width = 0.2
    index = 0
    
    # Store ratios for the secondary axis
    ratios_no_boiloff = []
    ratios_with_boiloff = []
    ratios_tank_corrected = []

    for vessel_type in VESSEL_TYPES:
        x_labels.append(VESSEL_TYPE_TITLES[vessel_type])
        x_positions.append(index)

        # Read data for TotalCAPEX, TotalFuelOPEX, and TotalExcludingFuelOPEX from the three cases
        no_boiloff_capex_df = read_processed_data(
            RESULTS_DIR_NO_BOILOFF, fuel, pathway, f"TotalCAPEX-{modifier}_orig"
        )
        no_boiloff_opex_df = read_processed_data(
            RESULTS_DIR_NO_BOILOFF, fuel, pathway, f"TotalFuelOPEX-{modifier}_orig"
        )
        no_boiloff_excl_df = read_processed_data(
            RESULTS_DIR_NO_BOILOFF, fuel, pathway, f"TotalExcludingFuelOPEX-{modifier}_orig"
        )

        with_boiloff_capex_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalCAPEX-{modifier}_orig"
        )
        with_boiloff_opex_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalFuelOPEX-{modifier}_orig"
        )
        with_boiloff_excl_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalExcludingFuelOPEX-{modifier}_orig"
        )

        tank_corrected_capex_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalCAPEX-{modifier}"
        )
        tank_corrected_opex_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalFuelOPEX-{modifier}"
        )
        tank_corrected_excl_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalExcludingFuelOPEX-{modifier}"
        )
        
        lsfo_totalcost_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, "lsfo", "fossil", f"TotalCost-{modifier}"
        )

        # Extract relevant values
        no_boiloff_values = {
            "TotalCAPEX": no_boiloff_capex_df.loc["Global Average", vessel_type],
            "TotalFuelOPEX": no_boiloff_opex_df.loc["Global Average", vessel_type],
            "TotalExcludingFuelOPEX": no_boiloff_excl_df.loc["Global Average", vessel_type],
        }

        with_boiloff_values = {
            "TotalCAPEX": with_boiloff_capex_df.loc["Global Average", vessel_type],
            "TotalFuelOPEX": with_boiloff_opex_df.loc["Global Average", vessel_type],
            "TotalExcludingFuelOPEX": with_boiloff_excl_df.loc["Global Average", vessel_type],
        }

        tank_corrected_values = {
            "TotalCAPEX": tank_corrected_capex_df.loc["Global Average", vessel_type],
            "TotalFuelOPEX": tank_corrected_opex_df.loc["Global Average", vessel_type],
            "TotalExcludingFuelOPEX": tank_corrected_excl_df.loc["Global Average", vessel_type],
        }
        
        lsfo_totalcost_value = lsfo_totalcost_df.loc["Global Average", vessel_type]
        
        # Aggregate costs for each case
        no_boiloff_total = (
            no_boiloff_capex_df.loc["Global Average", vessel_type]
            + no_boiloff_opex_df.loc["Global Average", vessel_type]
            + no_boiloff_excl_df.loc["Global Average", vessel_type]
        )

        with_boiloff_total = (
            with_boiloff_capex_df.loc["Global Average", vessel_type]
            + with_boiloff_opex_df.loc["Global Average", vessel_type]
            + with_boiloff_excl_df.loc["Global Average", vessel_type]
        )
        
        tank_corrected_total = (
            tank_corrected_capex_df.loc["Global Average", vessel_type]
            + tank_corrected_opex_df.loc["Global Average", vessel_type]
            + tank_corrected_excl_df.loc["Global Average", vessel_type]
        )
        
        # Compute ratios
        ratios_no_boiloff.append(no_boiloff_total / lsfo_totalcost_value)
        ratios_with_boiloff.append(with_boiloff_total / lsfo_totalcost_value)
        ratios_tank_corrected.append(tank_corrected_total / lsfo_totalcost_value)

        # Plot stacked bars for each case
        ax_hist.bar(
            index - width,
            no_boiloff_values["TotalCAPEX"],
            width,
            color=COLORS["no_boiloff"][0],
        )
        ax_hist.bar(
            index - width,
            no_boiloff_values["TotalFuelOPEX"],
            width,
            bottom=no_boiloff_values["TotalCAPEX"],
            color=COLORS["no_boiloff"][1],
        )
        ax_hist.bar(
            index - width,
            no_boiloff_values["TotalExcludingFuelOPEX"],
            width,
            bottom=no_boiloff_values["TotalCAPEX"]
            + no_boiloff_values["TotalFuelOPEX"],
            color=COLORS["no_boiloff"][2],
        )
        ax_hist.plot([index-width*2, index+width*2], [lsfo_totalcost_value, lsfo_totalcost_value], ls="--", color="red", linewidth=2)

        ax_hist.bar(
            index,
            with_boiloff_values["TotalCAPEX"],
            width,
            color=COLORS["with_boiloff"][0],
        )
        ax_hist.bar(
            index,
            with_boiloff_values["TotalFuelOPEX"],
            width,
            bottom=with_boiloff_values["TotalCAPEX"],
            color=COLORS["with_boiloff"][1],
        )
        ax_hist.bar(
            index,
            with_boiloff_values["TotalExcludingFuelOPEX"],
            width,
            bottom=with_boiloff_values["TotalCAPEX"]
            + with_boiloff_values["TotalFuelOPEX"],
            color=COLORS["with_boiloff"][2],
        )

        ax_hist.bar(
            index + width,
            tank_corrected_values["TotalCAPEX"],
            width,
            color=COLORS["tank_corrected"][0],
        )
        ax_hist.bar(
            index + width,
            tank_corrected_values["TotalFuelOPEX"],
            width,
            bottom=tank_corrected_values["TotalCAPEX"],
            color=COLORS["tank_corrected"][1],
        )
        ax_hist.bar(
            index + width,
            tank_corrected_values["TotalExcludingFuelOPEX"],
            width,
            bottom=tank_corrected_values["TotalCAPEX"]
            + tank_corrected_values["TotalFuelOPEX"],
            color=COLORS["tank_corrected"][2],
        )

        index += 1

    # Customize the plot
    fuel_label = get_fuel_label(fuel)
    pathway_label = get_pathway_label(pathway)
    ax_hist.set_title(f"{fuel_label}: {pathway_label}", fontsize=22)
    ax_hist.set_xticks(x_positions)
    ax_hist.set_xticklabels(x_labels, fontsize=18, rotation=0)
    if modifier == "per_tonne_mile":
        ax_hist.set_ylabel("Total Cost (USD per tonne-mile)", fontsize=20)
    else:
        ax_hist.set_ylabel("Total Cost (USD per m$^3$-mile)", fontsize=20)
    ymin, ymax = ax_hist.get_ylim()
    ax_hist.set_ylim(ymin, ymax*1.5)
    ax_hist.tick_params(axis='both', labelsize=18)
    ax_ratio.tick_params(axis='both', labelsize=18)
    
    # Plot ratios as markers
    ax_ratio.plot(
        [x - width for x in x_positions],
        ratios_no_boiloff,
        "o",
        color="blue",
        label="No Boiloff / LSFO",
    )
    ax_ratio.plot(
        x_positions,
        ratios_with_boiloff,
        "o",
        color="orange",
        label="With Boiloff / LSFO",
    )
    ax_ratio.plot(
        [x + width for x in x_positions],
        ratios_tank_corrected,
        "o",
        color="purple",
        label="Tank Corrected / LSFO",
    )
    
    # Customize secondary y-axis
    ax_ratio.set_ylabel("Ratio to LSFO Total Cost", fontsize=20)
    ax_ratio.tick_params(axis="y", labelsize=16)

    # Add legends
    greys = generate_grey_shades(3)
    legend_handles_1 = [
        plt.Rectangle((0, 0), 1, 1, color=greys[0], label="CAPEX"),
        plt.Rectangle((0, 0), 1, 1, color=greys[1], label="Fuel OPEX"),
        plt.Rectangle((0, 0), 1, 1, color=greys[2], label="Other OPEX"),
    ]
    legend_handles_2 = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["no_boiloff"][0], label=LABELS[0]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["with_boiloff"][0], label=LABELS[1]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["tank_corrected"][0], label=LABELS[2]),
    ]
    legend_handles_3 = [
        Line2D([0], [0], color="red", linestyle="--", label="LSFO", linewidth=2)
    ]
    
    ax_hist.add_artist(
        ax_hist.legend(
            handles=legend_handles_1,
            title="Cost Components",
            loc="upper left",
            fontsize=16,
            title_fontsize=18,
        )
    )
    
    ax_hist.add_artist(
        ax_hist.legend(
            handles=legend_handles_2,
            title="Boiloff and Tank Correction",
            loc="upper right",
            fontsize=16,
            title_fontsize=18,
        )
    )
    
    ax_hist.add_artist(
        ax_hist.legend(
            handles=legend_handles_3,
            loc="upper center",
            fontsize=16,
            title_fontsize=18,
            bbox_to_anchor=(0.36, 0.93)  # Replace x, y with desired coordinates
        )
    )
    
    ax_ratio.axhline(1, ls="--", color="black")

    # Save and display the plot
    create_directory_if_not_exists(f"{get_top_dir()}/plots/{fuel}-{pathway}")
    output_path_png = f"{get_top_dir()}/plots/{fuel}-{pathway}/total_cost_comparison_{fuel}_{pathway}_{modifier}.png"
    output_path_pdf = f"{get_top_dir()}/plots/{fuel}-{pathway}/total_cost_comparison_{fuel}_{pathway}_{modifier}.pdf"
    plt.savefig(output_path_png, dpi=300)
    plt.savefig(output_path_pdf)
    plt.close()
    print(f"Plot saved at {output_path_png}")
    print(f"Plot saved at {output_path_pdf}")
    
def plot_histogram_for_vessel_classes(vessel_type, fuel, pathway, quantity="TotalCost", modifier="per_tonne_mile"):
    """
    Plots a histogram comparing TotalCost per tonne mile between all vessel types for a given fuel and pathway.
    
    Parameters:
    ----------
    vessel_type : str
        The type of vessel (e.g., "bulk_carrier_ice").
    fuel : str
        Fuel name
    pathway : str
        Production pathway
    quantity : str
        Quantity to plot (default: "TotalCost")
    modifier : str
        Modifier for the quantity (default: "per_tonne_mile")
    """
    if vessel_type not in VESSEL_TYPES:
        raise ValueError(f"Invalid vessel type. Choose from: {VESSEL_TYPES}")

    # Define the classes for the given vessel type
    vessel_classes = {
        "bulk_carrier_ice": ["bulk_carrier_capesize_ice", "bulk_carrier_handy_ice", "bulk_carrier_panamax_ice"],
        "container_ice": ["container_15000_teu_ice", "container_8000_teu_ice", "container_3500_teu_ice"],
        "tanker_ice": ["tanker_100k_dwt_ice", "tanker_300k_dwt_ice", "tanker_35k_dwt_ice"],
        "gas_carrier_ice": ["gas_carrier_100k_cbm_ice"],
    }

    vessel_class_titles = {
        "bulk_carrier_capesize_ice": "Capesize",
        "bulk_carrier_handy_ice": "Handy",
        "bulk_carrier_panamax_ice": "Panamax",
        "container_15000_teu_ice": "15,000 TEU",
        "container_8000_teu_ice": "8,000 TEU",
        "container_3500_teu_ice": "3,500 TEU",
        "tanker_100k_dwt_ice": "100k DWT",
        "tanker_300k_dwt_ice": "300k DWT",
        "tanker_35k_dwt_ice": "35k DWT",
        "gas_carrier_100k_cbm_ice": "100k m³",
    }
    
    vessel_type_title = {
        "bulk_carrier_ice": "Bulk Carrier",
        "container_ice": "Container",
        "tanker_ice": "Tanker",
        "gas_carrier_ice": "Gas Carrier",
    }

    classes = vessel_classes[vessel_type]
    
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    # Main histogram plot
    ax_hist = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_hist)

    x_labels = []
    x_positions = []
    width = 0.2
    index = 0
    
    # Store ratios for the secondary axis
    ratios_no_boiloff = []
    ratios_with_boiloff = []
    ratios_tank_corrected = []

    for vessel_class in classes:
        x_labels.append(vessel_class_titles[vessel_class])
        x_positions.append(index)

        # Read data for TotalCAPEX, TotalFuelOPEX, and TotalExcludingFuelOPEX from the three cases
        no_boiloff_capex_df = read_processed_data(
            RESULTS_DIR_NO_BOILOFF, fuel, pathway, f"TotalCAPEX-{modifier}_orig"
        )
        no_boiloff_opex_df = read_processed_data(
            RESULTS_DIR_NO_BOILOFF, fuel, pathway, f"TotalFuelOPEX-{modifier}_orig"
        )
        no_boiloff_excl_df = read_processed_data(
            RESULTS_DIR_NO_BOILOFF, fuel, pathway, f"TotalExcludingFuelOPEX-{modifier}_orig"
        )

        with_boiloff_capex_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalCAPEX-{modifier}_orig"
        )
        with_boiloff_opex_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalFuelOPEX-{modifier}_orig"
        )
        with_boiloff_excl_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalExcludingFuelOPEX-{modifier}_orig"
        )

        tank_corrected_capex_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalCAPEX-{modifier}"
        )
        tank_corrected_opex_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalFuelOPEX-{modifier}"
        )
        tank_corrected_excl_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, fuel, pathway, f"TotalExcludingFuelOPEX-{modifier}"
        )
        
        lsfo_totalcost_df = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, "lsfo", "fossil", f"TotalCost-{modifier}"
        )

        # Extract relevant values
        no_boiloff_values = {
            "TotalCAPEX": no_boiloff_capex_df.loc["Global Average", vessel_class],
            "TotalFuelOPEX": no_boiloff_opex_df.loc["Global Average", vessel_class],
            "TotalExcludingFuelOPEX": no_boiloff_excl_df.loc["Global Average", vessel_class],
        }

        with_boiloff_values = {
            "TotalCAPEX": with_boiloff_capex_df.loc["Global Average", vessel_class],
            "TotalFuelOPEX": with_boiloff_opex_df.loc["Global Average", vessel_class],
            "TotalExcludingFuelOPEX": with_boiloff_excl_df.loc["Global Average", vessel_class],
        }

        tank_corrected_values = {
            "TotalCAPEX": tank_corrected_capex_df.loc["Global Average", vessel_class],
            "TotalFuelOPEX": tank_corrected_opex_df.loc["Global Average", vessel_class],
            "TotalExcludingFuelOPEX": tank_corrected_excl_df.loc["Global Average", vessel_class],
        }
        
        lsfo_totalcost_value = lsfo_totalcost_df.loc["Global Average", vessel_class]
        
        # Aggregate costs for each case
        no_boiloff_total = (
            no_boiloff_capex_df.loc["Global Average", vessel_class]
            + no_boiloff_opex_df.loc["Global Average", vessel_class]
            + no_boiloff_excl_df.loc["Global Average", vessel_class]
        )

        with_boiloff_total = (
            with_boiloff_capex_df.loc["Global Average", vessel_class]
            + with_boiloff_opex_df.loc["Global Average", vessel_class]
            + with_boiloff_excl_df.loc["Global Average", vessel_class]
        )
        
        tank_corrected_total = (
            tank_corrected_capex_df.loc["Global Average", vessel_class]
            + tank_corrected_opex_df.loc["Global Average", vessel_class]
            + tank_corrected_excl_df.loc["Global Average", vessel_class]
        )
        
        # Compute ratios
        ratios_no_boiloff.append(no_boiloff_total / lsfo_totalcost_value)
        ratios_with_boiloff.append(with_boiloff_total / lsfo_totalcost_value)
        ratios_tank_corrected.append(tank_corrected_total / lsfo_totalcost_value)

        # Plot stacked bars for each case
        ax_hist.bar(
            index - width,
            no_boiloff_values["TotalCAPEX"],
            width,
            color=COLORS["no_boiloff"][0],
        )
        ax_hist.bar(
            index - width,
            no_boiloff_values["TotalFuelOPEX"],
            width,
            bottom=no_boiloff_values["TotalCAPEX"],
            color=COLORS["no_boiloff"][1],
        )
        ax_hist.bar(
            index - width,
            no_boiloff_values["TotalExcludingFuelOPEX"],
            width,
            bottom=no_boiloff_values["TotalCAPEX"]
            + no_boiloff_values["TotalFuelOPEX"],
            color=COLORS["no_boiloff"][2],
        )
        ax_hist.plot([index-width*2, index+width*2], [lsfo_totalcost_value, lsfo_totalcost_value], ls="--", color="red", linewidth=2)

        ax_hist.bar(
            index,
            with_boiloff_values["TotalCAPEX"],
            width,
            color=COLORS["with_boiloff"][0],
        )
        ax_hist.bar(
            index,
            with_boiloff_values["TotalFuelOPEX"],
            width,
            bottom=with_boiloff_values["TotalCAPEX"],
            color=COLORS["with_boiloff"][1],
        )
        ax_hist.bar(
            index,
            with_boiloff_values["TotalExcludingFuelOPEX"],
            width,
            bottom=with_boiloff_values["TotalCAPEX"]
            + with_boiloff_values["TotalFuelOPEX"],
            color=COLORS["with_boiloff"][2],
        )

        ax_hist.bar(
            index + width,
            tank_corrected_values["TotalCAPEX"],
            width,
            color=COLORS["tank_corrected"][0],
        )
        ax_hist.bar(
            index + width,
            tank_corrected_values["TotalFuelOPEX"],
            width,
            bottom=tank_corrected_values["TotalCAPEX"],
            color=COLORS["tank_corrected"][1],
        )
        ax_hist.bar(
            index + width,
            tank_corrected_values["TotalExcludingFuelOPEX"],
            width,
            bottom=tank_corrected_values["TotalCAPEX"]
            + tank_corrected_values["TotalFuelOPEX"],
            color=COLORS["tank_corrected"][2],
        )

        index += 1

    # Customize the plot
    fuel_label = get_fuel_label(fuel)
    pathway_label = get_pathway_label(pathway)
    vessel_type_label = vessel_type_title[vessel_type]
    ax_hist.set_title(f"{vessel_type_label} with {fuel_label}: {pathway_label}", fontsize=22)
    ax_hist.set_xticks(x_positions)
    ax_hist.set_xticklabels(x_labels, fontsize=18, rotation=0)
    if modifier == "per_tonne_mile":
        ax_hist.set_ylabel("Total Cost (USD per tonne-mile)", fontsize=20)
    else:
        ax_hist.set_ylabel("Total Cost (USD per m$^3$-mile)", fontsize=20)
    ymin, ymax = ax_hist.get_ylim()
    ax_hist.set_ylim(ymin, ymax*1.5)
    ax_hist.tick_params(axis='both', labelsize=18)
    ax_ratio.tick_params(axis='both', labelsize=18)
    
    # Plot ratios as markers
    ax_ratio.plot(
        [x - width for x in x_positions],
        ratios_no_boiloff,
        "o",
        color="blue",
        label="No Boiloff / LSFO",
    )
    ax_ratio.plot(
        x_positions,
        ratios_with_boiloff,
        "o",
        color="orange",
        label="With Boiloff / LSFO",
    )
    ax_ratio.plot(
        [x + width for x in x_positions],
        ratios_tank_corrected,
        "o",
        color="purple",
        label="Tank Corrected / LSFO",
    )
    
    # Customize secondary y-axis
    ax_ratio.set_ylabel("Ratio to LSFO Total Cost", fontsize=20)
    ax_ratio.tick_params(axis="y", labelsize=16)

    # Add legends
    greys = generate_grey_shades(3)
    legend_handles_1 = [
        plt.Rectangle((0, 0), 1, 1, color=greys[0], label="CAPEX"),
        plt.Rectangle((0, 0), 1, 1, color=greys[1], label="Fuel OPEX"),
        plt.Rectangle((0, 0), 1, 1, color=greys[2], label="Other OPEX"),
    ]
    legend_handles_2 = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS["no_boiloff"][0], label=LABELS[0]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["with_boiloff"][0], label=LABELS[1]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS["tank_corrected"][0], label=LABELS[2]),
    ]
    legend_handles_3 = [
        Line2D([0], [0], color="red", linestyle="--", label="LSFO", linewidth=2)
    ]
    
    ax_hist.add_artist(
        ax_hist.legend(
            handles=legend_handles_1,
            title="Cost Components",
            loc="upper left",
            fontsize=16,
            title_fontsize=18,
        )
    )
    
    ax_hist.add_artist(
        ax_hist.legend(
            handles=legend_handles_2,
            title="Boiloff and Tank Correction",
            loc="upper right",
            fontsize=16,
            title_fontsize=18,
        )
    )
    
    ax_hist.add_artist(
        ax_hist.legend(
            handles=legend_handles_3,
            loc="upper center",
            fontsize=16,
            title_fontsize=18,
            bbox_to_anchor=(0.36, 0.93)  # Replace x, y with desired coordinates
        )
    )
    
    ax_ratio.axhline(1, ls="--", color="black")

    # Save and display the plot
    create_directory_if_not_exists(f"{get_top_dir()}/plots/{fuel}-{pathway}")
    output_path_png = f"{get_top_dir()}/plots/{fuel}-{pathway}/{vessel_type}_total_cost_comparison_{fuel}_{pathway}_{modifier}.png"
    output_path_pdf = f"{get_top_dir()}/plots/{fuel}-{pathway}/{vessel_type}_total_cost_comparison_{fuel}_{pathway}_{modifier}.pdf"
    plt.savefig(output_path_png, dpi=300)
    plt.savefig(output_path_pdf, dpi=300)
    plt.close()
    print(f"Plot saved at {output_path_png}")
    print(f"Plot saved at {output_path_pdf}")
    
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
    
def get_pathways(fuel, results_dir=RESULTS_DIR_NO_BOILOFF):
    """
    Collects the names of all pathways contained in processed csv files for the given fuel

    Parameters
    ----------
    fuel : str
        Name øf the fuel to collect pathways for
        
    results_dir : str
        Path to the director to read filenames from and parse pathway names

    Returns
    -------
    pathways : list of str
        List of unique pathways available for the given fuel
    """
    pathways = find_unique_identifiers(
        results_dir, "pathway", fuel
    )
    
    return pathways


# Example Usage
if __name__ == "__main__":
#    plot_histogram_for_vessel_types("liquid_hydrogen", "LTE_H_grid_E", modifier="per_tonne_mile")
#    plot_histogram_for_vessel_types("liquid_hydrogen", "LTE_H_grid_E", modifier="per_cbm_mile")
#    plot_histogram_for_vessel_classes("bulk_carrier_ice", "liquid_hydrogen", "LTE_H_grid_E", modifier="per_tonne_mile")
    
    for fuel in ["compressed_hydrogen", "liquid_hydrogen", "ammonia"]:
        print(fuel)
        pathways = get_pathways(fuel)

        for pathway in ["BG_H_grid_E"]:#pathways:
            #plot_histogram_for_vessel_types(fuel, pathway, modifier="per_tonne_mile")
            #plot_histogram_for_vessel_types(fuel, pathway, modifier="per_cbm_mile")

            for vessel_type in VESSEL_TYPES:
                print(vessel_type)
                plot_histogram_for_vessel_classes(vessel_type, fuel, pathway, modifier="per_tonne_mile")
                plot_histogram_for_vessel_classes(vessel_type, fuel, pathway, modifier="per_cbm_mile")


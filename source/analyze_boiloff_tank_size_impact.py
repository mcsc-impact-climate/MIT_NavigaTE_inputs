"""
Date: 241118
Author: danikam
Purpose: Compare cost per mile and per m^3 of cargo after accounting for boil-off and tank size correction.
"""

import os
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from common_tools import (
    create_directory_if_not_exists,
    generate_blue_shades,
    get_fuel_label,
    get_pathway_label,
    get_top_dir,
    read_fuel_labels,
    read_pathway_labels,
)
from matplotlib.lines import Line2D
from parse import parse
import seaborn as sns
import numpy as np

top_dir = get_top_dir()

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
RESULTS_DIR_BASE = "processed_results_orig_caps_no_boiloff"
RESULTS_DIR_MOD_CAPS = "processed_results_mod_caps_no_boiloff"
RESULTS_DIR_WITH_BOILOFF = "processed_results_mod_caps_with_boiloff"
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
    "base": generate_blue_shades(3),
    "mod_caps": generate_orange_shades(3),
    "with_boiloff": generate_purple_shades(3),
}
LABELS = [
    "No tank size corr or boil-off loss",
    "With tank size corr, no boil-off loss",
    "With tank size corr and boil-off loss",
]

LABELS_WITH_NEWLINES = [
    "No tank size corr\n or boil-off loss",
    "With tank size corr,\n no boil-off loss",
    "With tank size corr\n and boil-off loss",
]

fuel_colors = {
    "ammonia": "blue",
    "methanol": "green",
    "FTdiesel": "orange",
    "liquid_hydrogen": "red",
    "compressed_hydrogen": "purple",
    "lng": "teal",
}

# Define the classes for the given vessel type
vessel_classes = {
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
    "tanker_ice": [
        "tanker_100k_dwt_ice",
        "tanker_300k_dwt_ice",
        "tanker_35k_dwt_ice",
    ],
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
    file_path = f"{top_dir}/{results_dir}/{fuel}-{pathway}-{quantity}.csv"
    # print(file_path)
    processed_data = pd.read_csv(file_path, index_col=0)
    return processed_data


def plot_histogram_for_vessel_types(
    fuel,
    pathway,
    quantity="TotalCost",
    modifier="per_tonne_mile",
    include_stowage=False,
):
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax_hist = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_hist)

    x_labels = []
    x_positions = []
    width = 0.2
    index = 0

    ratios_base = []
    ratios_mod_caps = []
    ratios_with_boiloff = []

    stowage_str = "_final" if include_stowage else ""

    for vessel_type in VESSEL_TYPES:
        x_labels.append(VESSEL_TYPE_TITLES[vessel_type])
        x_positions.append(index)

        def read_set(directory, suffix):
            return {
                "CAPEX": read_processed_data(directory, fuel, pathway, f"CAPEX-{modifier}{suffix}").loc["Global Average", vessel_type],
                "FuelOPEX": read_processed_data(directory, fuel, pathway, f"FuelOPEX-{modifier}{suffix}").loc["Global Average", vessel_type],
                "OPEX": read_processed_data(directory, fuel, pathway, f"OPEX-{modifier}{suffix}").loc["Global Average", vessel_type],
            }

        base = read_set(RESULTS_DIR_BASE, f"_lsfo{stowage_str}")
        mod = read_set(RESULTS_DIR_MOD_CAPS, stowage_str)
        boil = read_set(RESULTS_DIR_WITH_BOILOFF, stowage_str)

        lsfo_totalcost = read_processed_data(
            RESULTS_DIR_WITH_BOILOFF, "lsfo", "fossil", f"TotalCost-{modifier}{stowage_str}"
        ).loc["Global Average", vessel_type]

        base_total = sum(base.values())
        mod_total = sum(mod.values())
        boil_total = sum(boil.values())

        ratios_base.append(base_total / lsfo_totalcost)
        ratios_mod_caps.append(mod_total / lsfo_totalcost)
        ratios_with_boiloff.append(boil_total / lsfo_totalcost)

        ax_hist.bar(index - width, 1000*base["CAPEX"], width, color=COLORS["base"][0])
        ax_hist.bar(index - width, 1000*base["OPEX"], width, bottom=1000*base["CAPEX"], color=COLORS["base"][1])
        ax_hist.bar(index - width, 1000*base["FuelOPEX"], width, bottom=1000*base["CAPEX"] + 1000*base["OPEX"], color=COLORS["base"][2])
        ax_hist.plot([index - width * 2, index + width * 2], [1000*lsfo_totalcost] * 2, ls="--", color="red", linewidth=2)

        ax_hist.bar(index, 1000*mod["CAPEX"], width, color=COLORS["mod_caps"][0])
        ax_hist.bar(index, 1000*mod["OPEX"], width, bottom=1000*mod["CAPEX"], color=COLORS["mod_caps"][1])
        ax_hist.bar(index, 1000*mod["FuelOPEX"], width, bottom=1000*mod["CAPEX"] + 1000*mod["OPEX"], color=COLORS["mod_caps"][2])

        ax_hist.bar(index + width, 1000*boil["CAPEX"], width, color=COLORS["with_boiloff"][0])
        ax_hist.bar(index + width, 1000*boil["OPEX"], width, bottom=1000*boil["CAPEX"], color=COLORS["with_boiloff"][1])
        ax_hist.bar(index + width, 1000*boil["FuelOPEX"], width, bottom=1000*boil["CAPEX"] + 1000*boil["OPEX"], color=COLORS["with_boiloff"][2])

        index += 1

    ax_hist.set_xticks(x_positions)
    ax_hist.set_ylabel("Total Cost (USD / kilotonne-mile)" if modifier == "per_tonne_mile" else "Total Cost (USD / m$^3$-mile)", fontsize=30)
    ax_hist.tick_params(axis="both", labelsize=28)
    ax_hist.set_xticklabels(x_labels, fontsize=26)

    ax_ratio.set_ylabel("Ratio to Fuel Oil", fontsize=30)
    ax_ratio.tick_params(axis="both", labelsize=28)
    ax_ratio.set_ylim(0, max(max(ratios_base + ratios_mod_caps + ratios_with_boiloff), 1.1) * 1.2)

    for i, ratio_list, color, shift in zip(
        range(len(x_positions)), [ratios_base, ratios_mod_caps, ratios_with_boiloff],
        ["blue", "orange", "purple"], [-width, 0, width]
    ):
        for x, y in zip([x + shift for x in x_positions], ratio_list):
            ax_ratio.plot(x, y, "o", color=color)
            ax_ratio.hlines(y, x - width / 2, x + width / 2, color=color, linewidth=1)

    ax_ratio.axhline(1, ls="--", color="red", linewidth=2)
    ax_ratio.tick_params(bottom=False, labelbottom=False)

    plt.subplots_adjust(bottom=0.05, top=0.95)

    create_directory_if_not_exists(f"{top_dir}/plots/{fuel}-{pathway}")
    suffix = f"{fuel}_{pathway}_{modifier}{stowage_str}"
    output_dir = f"{top_dir}/plots/{fuel}-{pathway}"

    # Save no-legend version first
    plt.tight_layout()
    no_legend_png = f"{output_dir}/total_cost_comparison_{suffix}_nolegend.png"
    no_legend_pdf = f"{output_dir}/total_cost_comparison_{suffix}_nolegend.pdf"
    plt.savefig(no_legend_png, dpi=300)
    plt.savefig(no_legend_pdf)
    print(f"No-legend plot saved: {no_legend_png}")

    # Add legends now
    greys = generate_grey_shades(3)
    handles1 = [plt.Rectangle((0, 0), 1, 1, color=greys[i], label=l) for i, l in enumerate(["CAPEX", "Fuel OPEX", "Other OPEX"])]
    handles2 = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k][0], label=LABELS[i]) for i, k in enumerate(["base", "mod_caps", "with_boiloff"])]
    handles3 = [Line2D([0], [0], color="red", linestyle="--", label="Conventional Fuel Oil", linewidth=2)]

    legend1 = ax_hist.legend(
        handles=handles1,
        title="Cost Components",
        loc="upper left",
        fontsize=20,
        title_fontsize=22,
    )
    ax_hist.add_artist(legend1)

    legend2 = ax_hist.legend(
        handles=handles2,
        title="Correction(s) Applied",
        loc="upper right",
        fontsize=20,
        title_fontsize=22,
    )
    ax_hist.add_artist(legend2)

    legend3 = ax_hist.legend(
        handles=handles3,
        loc="upper center",
        bbox_to_anchor=(0.227, 0.65),
        fontsize=20
    )
    ax_hist.add_artist(legend3)

    # Save full version with legends
    ymin, ymax = ax_hist.get_ylim()
    ax_hist.set_ylim(ymin, ymax * 1.5)
    plt.tight_layout()
    full_png = f"{output_dir}/total_cost_comparison_{suffix}.png"
    full_pdf = f"{output_dir}/total_cost_comparison_{suffix}.pdf"
    plt.savefig(full_png, dpi=300)
    plt.savefig(full_pdf)
    print(f"Full plot saved: {full_png}")

    plt.close()

    # Save legend-only version
    plot_legend_only(fuel, pathway, modifier, include_stowage)


def plot_legend_only(fuel, pathway, modifier, include_stowage):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(15, 1.7))
    ax.axis("off")

    # Legend handles
    greys = generate_grey_shades(3)
    handles_components = [
        plt.Rectangle((0, 0), 1, 1, color=greys[i], label=label)
        for i, label in enumerate(["CAPEX", "Fuel OPEX", "Other OPEX"])
    ]

    handles_corrections = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[key][0], label=LABELS[i])
        for i, key in enumerate(["base", "mod_caps", "with_boiloff"])
    ]

    handles_conventional = [
        Line2D([0], [0], color="red", linestyle="--", label="Conventional Fuel Oil", linewidth=2)
    ]

    # Add top legend: Correction(s) Applied
    legend_corrections = ax.legend(
        handles=handles_corrections,
        title="Correction(s) Applied",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=3,
        fontsize=16,
        title_fontsize=18,
    )
    ax.add_artist(legend_corrections)

    # Add bottom-left legend: Cost Components
    legend_components = ax.legend(
        handles=handles_components,
        title="Cost Components",
        loc="center left",
        bbox_to_anchor=(0.1, 0.13),  # shifted slightly left and lower
        fontsize=15,
        title_fontsize=17,
        ncol=3
    )
    ax.add_artist(legend_components)

    # Add bottom-right legend: Conventional Fuel Oil
    legend_conventional = ax.legend(
        handles=handles_conventional,
        loc="center right",
        bbox_to_anchor=(0.9, 0.13),  # shifted slightly right and lower
        fontsize=15,
    )
    ax.add_artist(legend_conventional)

    # Save
    stowage_str = "_final" if include_stowage else ""
    suffix = f"{fuel}_{pathway}_{modifier}{stowage_str}"
    output_dir = f"{top_dir}/plots/{fuel}-{pathway}"
    create_directory_if_not_exists(output_dir)

    fig.savefig(f"{output_dir}/legend_only_{suffix}.png", dpi=300)
    fig.savefig(f"{output_dir}/legend_only_{suffix}.pdf")
    plt.close()
    print(f"Legend-only plot saved: {output_dir}/legend_only_{suffix}.png")


    
def make_summary_plot(fuels, pathways):
    """
    Constructs a color gradient of average increase in price relative to LSFO with each correction.

    Parameters:
    ----------
    fuels : str
        List of fuels for which to obtain cost summaries
        
    pathways : dict
        Dictionary mapping each fuel (str) to its production pathway (str)

    Returns:
    -------
    pd.DataFrame
        Merged DataFrame containing total cost and percentage increase due
        to corrections, for all fuels, indexed by vessel class and size.
    """
    
    cost_ratios = pd.DataFrame()
    avg_cost_ratios_no_corr = {}
    avg_cost_ratios_mod_caps = {}
    avg_cost_ratios_with_boiloff = {}
    
    totalcost_df_no_corr_lsfo = read_processed_data(RESULTS_DIR_BASE, "lsfo", pathways["lsfo"], "TotalCost-per_tonne_mile_lsfo_final")
    totalcost_df_mod_caps_lsfo = read_processed_data(RESULTS_DIR_MOD_CAPS, "lsfo", pathways["lsfo"], "TotalCost-per_tonne_mile_final")
    totalcost_df_with_boiloff_lsfo = read_processed_data(RESULTS_DIR_WITH_BOILOFF, "lsfo", pathways["lsfo"], "TotalCost-per_tonne_mile_final")
    
    for fuel in fuels:
        totalcost_df_no_corr = read_processed_data(RESULTS_DIR_BASE, fuel, pathways[fuel], "TotalCost-per_tonne_mile_lsfo_final")
        totalcost_df_mod_caps = read_processed_data(RESULTS_DIR_MOD_CAPS, fuel, pathways[fuel], "TotalCost-per_tonne_mile_final")
        totalcost_df_with_boiloff = read_processed_data(RESULTS_DIR_WITH_BOILOFF, fuel, pathways[fuel], "TotalCost-per_tonne_mile_final")
        
        cost_ratios_no_corr = []
        cost_ratios_mod_caps = []
        cost_ratios_with_boiloff = []
        for vessel_class in vessel_classes:
            for vessel_size in vessel_classes[vessel_class]:
                cost_ratios_no_corr.append(totalcost_df_no_corr.loc["Global Average", vessel_size] / totalcost_df_no_corr_lsfo.loc["Global Average", vessel_size])
                cost_ratios_mod_caps.append(totalcost_df_mod_caps.loc["Global Average", vessel_size] / totalcost_df_mod_caps_lsfo.loc["Global Average", vessel_size])
                cost_ratios_with_boiloff.append(totalcost_df_with_boiloff.loc["Global Average", vessel_size] / totalcost_df_with_boiloff_lsfo.loc["Global Average", vessel_size])
        cost_ratios_no_corr = np.asarray(cost_ratios_no_corr)
        cost_ratios_mod_caps = np.asarray(cost_ratios_mod_caps)
        cost_ratios_with_boiloff = np.asarray(cost_ratios_with_boiloff)
        
        avg_cost_ratios_no_corr[f"{fuel} (avg)"] = np.mean(cost_ratios_no_corr)
        avg_cost_ratios_no_corr[f"{fuel} (min)"] = np.min(cost_ratios_no_corr)
        avg_cost_ratios_no_corr[f"{fuel} (max)"] = np.max(cost_ratios_no_corr)

        avg_cost_ratios_mod_caps[f"{fuel} (avg)"] = np.mean(cost_ratios_mod_caps)
        avg_cost_ratios_mod_caps[f"{fuel} (min)"] = np.min(cost_ratios_mod_caps)
        avg_cost_ratios_mod_caps[f"{fuel} (max)"] = np.max(cost_ratios_mod_caps)
        
        avg_cost_ratios_with_boiloff[f"{fuel} (avg)"] = np.mean(cost_ratios_with_boiloff)
        avg_cost_ratios_with_boiloff[f"{fuel} (min)"] = np.min(cost_ratios_with_boiloff)
        avg_cost_ratios_with_boiloff[f"{fuel} (max)"] = np.max(cost_ratios_with_boiloff)

    all_ratios_dict = {
        LABELS_WITH_NEWLINES[0]: avg_cost_ratios_no_corr,
        LABELS_WITH_NEWLINES[1]: avg_cost_ratios_mod_caps,
        LABELS_WITH_NEWLINES[2]: avg_cost_ratios_with_boiloff,
    }
    
    cost_ratios = pd.DataFrame.from_dict(all_ratios_dict, orient="index")
    
    # Extract and sort fuel names by avg values in the "no corrections" row
    avg_values = {k.split(" ")[0]: v for k, v in avg_cost_ratios_no_corr.items() if "(avg)" in k}
    sorted_fuels = sorted(avg_values, key=avg_values.get)

    # Build the ordered column list: interleave avg, min, max per fuel
    ordered_columns = []
    for fuel in sorted_fuels:
        ordered_columns.extend([
            f"{fuel} (avg)",
            f"{fuel} (min)",
            f"{fuel} (max)"
        ])

    # Reorder columns
    cost_ratios = cost_ratios[ordered_columns]
    
    # Extract average values only
    avg_df = cost_ratios.loc[:, cost_ratios.columns.str.contains(r"\(avg\)")]
    avg_values = avg_df.values

    # Extract min/max values for annotations
    min_df = cost_ratios.loc[:, cost_ratios.columns.str.contains(r"\(min\)")]
    max_df = cost_ratios.loc[:, cost_ratios.columns.str.contains(r"\(max\)")]
    annotations = np.empty_like(avg_values, dtype=object)
    for i in range(avg_values.shape[0]):
        for j in range(avg_values.shape[1]):
            avg = avg_values[i, j]
            max_ = max_df.iloc[i, j]
            min_ = min_df.iloc[i, j]
            annotations[i, j] = f"${avg:.2f}^{{+{(max_-avg):.2f}}}_{{-{(avg-min_):.2f}}}$"

    # Prepare the plot
    plt.figure(figsize=(len(avg_df.columns)*1.5, 3))
    ax = sns.heatmap(
        avg_df,
        cmap="RdYlGn_r",  # red for high, green for low
        annot=annotations,
        fmt="",
        linewidths=0.5,
        cbar_kws={"label": "Cost Ratio to Fuel Oil"},
        xticklabels=[get_fuel_label(col.replace(" (avg)", "")).replace(" ", "\n") for col in avg_df.columns],
        yticklabels=LABELS_WITH_NEWLINES
    )

    # Style the plot
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), ha='center', rotation=0)
    #plt.title("Cost Ratio Relative to LSFO with Successive Corrections")
    plt.tight_layout()
    plt.savefig(f"{top_dir}/plots/cargo_loss_cost_summary.png", dpi=300)
    plt.savefig(f"{top_dir}/plots/cargo_loss_cost_summary.pdf")
    print(f"Plot saved to {top_dir}/plots/cargo_loss_cost_summary.png and .pdf")
    
    
def make_cost_summary(fuels, pathways):
    """
    Constructs and merges cost summary DataFrames for multiple fuels,
    comparing base and boil-off corrected total costs per tonne-mile
    across vessel classes and sizes.

    Parameters:
    ----------
    fuels : str
        List of fuels for which to obtain cost summaries
        
    pathways : dict
        Dictionary mapping each fuel (str) to its production pathway (str)

    Returns:
    -------
    pd.DataFrame
        Merged DataFrame containing total cost and percentage increase due
        to corrections, for all fuels, indexed by vessel class and size.
    """
    cost_summaries = {}
    for fuel in fuels+["lsfo"]:
        all_vessel_classes = []
        all_vessel_sizes = []
        totalcost_no_corrs = []
        totalcost_with_corrs = []
        corr_perc_cost_increases = []
        cost_summary_df = pd.DataFrame()
        
        totalcost_df_no_corr = read_processed_data(RESULTS_DIR_BASE, fuel, pathways[fuel], "TotalCost-per_tonne_mile_lsfo_final")
        totalcost_df_with_corr = read_processed_data(RESULTS_DIR_WITH_BOILOFF, fuel, pathways[fuel], "TotalCost-per_tonne_mile_final")
        
        for vessel_class in vessel_classes:
            for vessel_size in vessel_classes[vessel_class]:
                all_vessel_classes.append(vessel_class)
                all_vessel_sizes.append(vessel_size)
                totalcost_no_corr = totalcost_df_no_corr.loc["Global Average", vessel_size]
                totalcost_with_corr = totalcost_df_with_corr.loc["Global Average", vessel_size]
                corr_perc_cost_increase = 100 * (totalcost_with_corr - totalcost_no_corr) / totalcost_no_corr
                totalcost_no_corrs.append(totalcost_no_corr*1000)    # Convert to USD / 1000 tonne-miles
                totalcost_with_corrs.append(totalcost_with_corr*1000)
                corr_perc_cost_increases.append(corr_perc_cost_increase)
        
        cost_summary_df["Vessel Class"] = all_vessel_classes
        cost_summary_df["Vessel Size"] = all_vessel_sizes
        cost_summary_df[f"{fuel} Total Cost Before Corrs"] = totalcost_no_corrs
        cost_summary_df[f"{fuel} Total Cost"] = totalcost_with_corrs
        cost_summary_df[f"{fuel} % Increase"] = corr_perc_cost_increases
        
        cost_summaries[fuel] = cost_summary_df

    # Merge all fuel-specific cost summaries into one
    merged_df = None
    for fuel_df in cost_summaries.values():
        if merged_df is None:
            merged_df = fuel_df
        else:
            merged_df = pd.merge(merged_df, fuel_df, on=["Vessel Class", "Vessel Size"], how="outer")

    merged_df.to_csv(f"{top_dir}/tables/cargo_loss_cost_summary.csv")
    print(f"Saved cost summary to {top_dir}/tables/cargo_loss_cost_summary.csv")
    return merged_df


#def plot_histogram_for_vessel_classes(
#    vessel_type,
#    fuel,
#    pathway,
#    quantity="TotalCost",
#    modifier="per_tonne_mile",
#    include_stowage=False,
#):
#    """
#    Plots a histogram comparing TotalCost per tonne mile between all vessel types for a given fuel and pathway.
#
#    Parameters:
#    ----------
#    vessel_type : str
#        The type of vessel (e.g., "bulk_carrier_ice").
#    fuel : str
#        Fuel name
#    pathway : str
#        Production pathway
#    quantity : str
#        Quantity to plot (default: "TotalCost")
#    modifier : str
#        Modifier for the quantity (default: "per_tonne_mile")
#    """
#    if vessel_type not in VESSEL_TYPES:
#        raise ValueError(f"Invalid vessel type. Choose from: {VESSEL_TYPES}")
#
#    classes = vessel_classes[vessel_type]
#
#    fig = plt.figure(figsize=(12, 12))
#    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
#
#    # Main histogram plot
#    ax_hist = fig.add_subplot(gs[0])
#    ax_ratio = fig.add_subplot(gs[1], sharex=ax_hist)
#
#    x_labels = []
#    x_positions = []
#    width = 0.2
#    index = 0
#
#    # Store ratios for the secondary axis
#    ratios_no_boiloff = []
#    ratios_with_boiloff = []
#    ratios_tank_corrected = []
#
#    stowage_str = ""
#    if include_stowage:
#        stowage_str = "_final"
#
#    for vessel_class in classes:
#        x_labels.append(vessel_class_titles[vessel_class])
#        x_positions.append(index)
#
#        # Read data for TotalCAPEX, TotalFuelOPEX, and TotalExcludingFuelOPEX from the three cases
#        no_boiloff_capex_df = read_processed_data(
#            RESULTS_DIR_NO_BOILOFF,
#            fuel,
#            pathway,
#            f"TotalCAPEX-{modifier}{stowage_str}",
#        )
#        no_boiloff_opex_df = read_processed_data(
#            RESULTS_DIR_NO_BOILOFF,
#            fuel,
#            pathway,
#            f"TotalFuelOPEX-{modifier}{stowage_str}",
#        )
#        no_boiloff_excl_df = read_processed_data(
#            RESULTS_DIR_NO_BOILOFF,
#            fuel,
#            pathway,
#            f"TotalExcludingFuelOPEX-{modifier}{stowage_str}",
#        )
#
#        with_boiloff_capex_df = read_processed_data(
#            RESULTS_DIR_WITH_BOILOFF,
#            fuel,
#            pathway,
#            f"TotalCAPEX-{modifier}{stowage_str}",
#        )
#        with_boiloff_opex_df = read_processed_data(
#            RESULTS_DIR_WITH_BOILOFF,
#            fuel,
#            pathway,
#            f"TotalFuelOPEX-{modifier}{stowage_str}",
#        )
#        with_boiloff_excl_df = read_processed_data(
#            RESULTS_DIR_WITH_BOILOFF,
#            fuel,
#            pathway,
#            f"TotalExcludingFuelOPEX-{modifier}{stowage_str}",
#        )
#
#        tank_corrected_capex_df = read_processed_data(
#            RESULTS_DIR_WITH_BOILOFF,
#            fuel,
#            pathway,
#            f"TotalCAPEX-{modifier}{stowage_str}",
#        )
#        tank_corrected_opex_df = read_processed_data(
#            RESULTS_DIR_WITH_BOILOFF,
#            fuel,
#            pathway,
#            f"TotalFuelOPEX-{modifier}{stowage_str}",
#        )
#        tank_corrected_excl_df = read_processed_data(
#            RESULTS_DIR_WITH_BOILOFF,
#            fuel,
#            pathway,
#            f"TotalExcludingFuelOPEX-{modifier}{stowage_str}",
#        )
#
#        lsfo_totalcost_df = read_processed_data(
#            RESULTS_DIR_WITH_BOILOFF, "lsfo", "fossil", f"TotalCost-{modifier}{stowage_str}"
#        )
#
#        # Extract relevant values
#        no_boiloff_values = {
#            "TotalCAPEX": no_boiloff_capex_df.loc["Global Average", vessel_class],
#            "TotalFuelOPEX": no_boiloff_opex_df.loc["Global Average", vessel_class],
#            "TotalExcludingFuelOPEX": no_boiloff_excl_df.loc[
#                "Global Average", vessel_class
#            ],
#        }
#
#        with_boiloff_values = {
#            "TotalCAPEX": with_boiloff_capex_df.loc["Global Average", vessel_class],
#            "TotalFuelOPEX": with_boiloff_opex_df.loc["Global Average", vessel_class],
#            "TotalExcludingFuelOPEX": with_boiloff_excl_df.loc[
#                "Global Average", vessel_class
#            ],
#        }
#
#        tank_corrected_values = {
#            "TotalCAPEX": tank_corrected_capex_df.loc["Global Average", vessel_class],
#            "TotalFuelOPEX": tank_corrected_opex_df.loc["Global Average", vessel_class],
#            "TotalExcludingFuelOPEX": tank_corrected_excl_df.loc[
#                "Global Average", vessel_class
#            ],
#        }
#
#        lsfo_totalcost_value = lsfo_totalcost_df.loc["Global Average", vessel_class]
#
#        # Aggregate costs for each case
#        no_boiloff_total = (
#            no_boiloff_capex_df.loc["Global Average", vessel_class]
#            + no_boiloff_opex_df.loc["Global Average", vessel_class]
#            + no_boiloff_excl_df.loc["Global Average", vessel_class]
#        )
#
#        with_boiloff_total = (
#            with_boiloff_capex_df.loc["Global Average", vessel_class]
#            + with_boiloff_opex_df.loc["Global Average", vessel_class]
#            + with_boiloff_excl_df.loc["Global Average", vessel_class]
#        )
#
#        tank_corrected_total = (
#            tank_corrected_capex_df.loc["Global Average", vessel_class]
#            + tank_corrected_opex_df.loc["Global Average", vessel_class]
#            + tank_corrected_excl_df.loc["Global Average", vessel_class]
#        )
#
#        # Compute ratios
#        ratios_no_boiloff.append(no_boiloff_total / lsfo_totalcost_value)
#        ratios_with_boiloff.append(with_boiloff_total / lsfo_totalcost_value)
#        ratios_tank_corrected.append(tank_corrected_total / lsfo_totalcost_value)
#
#        # Plot stacked bars for each case
#        ax_hist.bar(
#            index - width,
#            no_boiloff_values["TotalCAPEX"],
#            width,
#            color=COLORS["no_boiloff"][0],
#        )
#        ax_hist.bar(
#            index - width,
#            no_boiloff_values["TotalFuelOPEX"],
#            width,
#            bottom=no_boiloff_values["TotalCAPEX"],
#            color=COLORS["no_boiloff"][1],
#        )
#        ax_hist.bar(
#            index - width,
#            no_boiloff_values["TotalExcludingFuelOPEX"],
#            width,
#            bottom=no_boiloff_values["TotalCAPEX"] + no_boiloff_values["TotalFuelOPEX"],
#            color=COLORS["no_boiloff"][2],
#        )
#        ax_hist.plot(
#            [index - width * 2, index + width * 2],
#            [lsfo_totalcost_value, lsfo_totalcost_value],
#            ls="--",
#            color="red",
#            linewidth=2,
#        )
#
#        ax_hist.bar(
#            index,
#            with_boiloff_values["TotalCAPEX"],
#            width,
#            color=COLORS["with_boiloff"][0],
#        )
#        ax_hist.bar(
#            index,
#            with_boiloff_values["TotalFuelOPEX"],
#            width,
#            bottom=with_boiloff_values["TotalCAPEX"],
#            color=COLORS["with_boiloff"][1],
#        )
#        ax_hist.bar(
#            index,
#            with_boiloff_values["TotalExcludingFuelOPEX"],
#            width,
#            bottom=with_boiloff_values["TotalCAPEX"]
#            + with_boiloff_values["TotalFuelOPEX"],
#            color=COLORS["with_boiloff"][2],
#        )
#
#        ax_hist.bar(
#            index + width,
#            tank_corrected_values["TotalCAPEX"],
#            width,
#            color=COLORS["tank_corrected"][0],
#        )
#        ax_hist.bar(
#            index + width,
#            tank_corrected_values["TotalFuelOPEX"],
#            width,
#            bottom=tank_corrected_values["TotalCAPEX"],
#            color=COLORS["tank_corrected"][1],
#        )
#        ax_hist.bar(
#            index + width,
#            tank_corrected_values["TotalExcludingFuelOPEX"],
#            width,
#            bottom=tank_corrected_values["TotalCAPEX"]
#            + tank_corrected_values["TotalFuelOPEX"],
#            color=COLORS["tank_corrected"][2],
#        )
#
#        index += 1
#
#    # Customize the plot
#    fuel_label = get_fuel_label(fuel)
#    pathway_label = get_pathway_label(pathway)
#    vessel_type_label = vessel_type_title[vessel_type]
#    ax_hist.set_title(
#        f"{vessel_type_label} with {fuel_label}: {pathway_label}", fontsize=22
#    )
#    ax_hist.set_xticks(x_positions)
#    ax_hist.set_xticklabels(x_labels, fontsize=18, rotation=0)
#    if modifier == "per_tonne_mile":
#        ax_hist.set_ylabel("Total Cost (USD per tonne-mile)", fontsize=20)
#    else:
#        ax_hist.set_ylabel("Total Cost (USD per m$^3$-mile)", fontsize=20)
#    ymin, ymax = ax_hist.get_ylim()
#    ax_hist.set_ylim(ymin, ymax * 1.5)
#    ax_hist.tick_params(axis="both", labelsize=18)
#    ax_ratio.tick_params(axis="both", labelsize=18)
#
#    # Plot ratios as markers
#    ax_ratio.plot(
#        [x - width for x in x_positions],
#        ratios_no_boiloff,
#        "o",
#        color="blue",
#        label="No Boiloff / LSFO",
#    )
#    ax_ratio.plot(
#        x_positions,
#        ratios_with_boiloff,
#        "o",
#        color="orange",
#        label="With Boiloff / LSFO",
#    )
#    ax_ratio.plot(
#        [x + width for x in x_positions],
#        ratios_tank_corrected,
#        "o",
#        color="purple",
#        label="Tank Corrected / LSFO",
#    )
#
#    # Customize secondary y-axis
#    ax_ratio.set_ylabel("Ratio to LSFO Total Cost", fontsize=20)
#    ax_ratio.tick_params(axis="y", labelsize=16)
#
#    # Add legends
#    greys = generate_grey_shades(3)
#    legend_handles_1 = [
#        plt.Rectangle((0, 0), 1, 1, color=greys[0], label="CAPEX"),
#        plt.Rectangle((0, 0), 1, 1, color=greys[1], label="Fuel OPEX"),
#        plt.Rectangle((0, 0), 1, 1, color=greys[2], label="Other OPEX"),
#    ]
#    legend_handles_2 = [
#        plt.Rectangle((0, 0), 1, 1, color=COLORS["no_boiloff"][0], label=LABELS[0]),
#        plt.Rectangle((0, 0), 1, 1, color=COLORS["with_boiloff"][0], label=LABELS[1]),
#        plt.Rectangle((0, 0), 1, 1, color=COLORS["tank_corrected"][0], label=LABELS[2]),
#    ]
#    legend_handles_3 = [
#        Line2D([0], [0], color="red", linestyle="--", label="LSFO", linewidth=2)
#    ]
#
#    ax_hist.add_artist(
#        ax_hist.legend(
#            handles=legend_handles_1,
#            title="Cost Components",
#            loc="upper left",
#            fontsize=16,
#            title_fontsize=18,
#        )
#    )
#
#    ax_hist.add_artist(
#        ax_hist.legend(
#            handles=legend_handles_2,
#            title="Boiloff and Tank Correction",
#            loc="upper right",
#            fontsize=16,
#            title_fontsize=18,
#        )
#    )
#
#    ax_hist.add_artist(
#        ax_hist.legend(
#            handles=legend_handles_3,
#            loc="upper center",
#            fontsize=16,
#            title_fontsize=18,
#            bbox_to_anchor=(0.36, 0.93),  # Replace x, y with desired coordinates
#        )
#    )
#
#    ax_ratio.axhline(1, ls="--", color="black")
#
#    # Save and display the plot
#    create_directory_if_not_exists(f"{top_dir}/plots/{fuel}-{pathway}")
#    output_path_png = f"{top_dir}/plots/{fuel}-{pathway}/{vessel_type}_total_cost_comparison_{fuel}_{pathway}_{modifier}{stowage_str}.png"
#    output_path_pdf = f"{top_dir}/plots/{fuel}-{pathway}/{vessel_type}_total_cost_comparison_{fuel}_{pathway}_{modifier}{stowage_str}.pdf"
#    plt.savefig(output_path_png, dpi=300)
#    plt.savefig(output_path_pdf, dpi=300)
#    plt.close()
#    print(f"Plot saved at {output_path_png}")
#    print(f"Plot saved at {output_path_pdf}")


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


def get_pathways(fuel, results_dir):
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
    pathways = find_unique_identifiers(results_dir, "pathway", fuel)

    return pathways

pathways = {
    "ammonia": "LTE_H_solar_E",
    "liquid_hydrogen": "LTE_H_solar_E",
    "compressed_hydrogen": "LTE_H_solar_E",
    "methanol": "LTE_H_DAC_C_solar_E",
    "FTdiesel": "LTE_H_DAC_C_solar_E",
    "lng": "fossil",
    "lsfo": "fossil"
}

if __name__ == "__main__":
    #    plot_histogram_for_vessel_types("liquid_hydrogen", "LTE_H_grid_E", modifier="per_tonne_mile")
    #    plot_histogram_for_vessel_types("liquid_hydrogen", "LTE_H_grid_E", modifier="per_cbm_mile")
    #    plot_histogram_for_vessel_classes("bulk_carrier_ice", "liquid_hydrogen", "LTE_H_grid_E", modifier="per_tonne_mile")
    
    fuels = ["liquid_hydrogen", "compressed_hydrogen", "ammonia", "lng", "methanol"]
    cost_summary_df = make_cost_summary(fuels, pathways)
    make_summary_plot(fuels, pathways)

    
    for fuel in ["liquid_hydrogen", "compressed_hydrogen", "ammonia", "lng", "methanol"]:  # , "ammonia"]:  # "compressed_hydrogen"
        pathway = pathways[fuel]

        #plot_histogram_for_vessel_types(fuel, pathway, modifier="per_tonne_mile")
        #plot_histogram_for_vessel_types(fuel, pathway, modifier="per_cbm_mile")
        plot_histogram_for_vessel_types(
            fuel, pathway, modifier="per_tonne_mile", include_stowage=True)
        plot_histogram_for_vessel_types(
            fuel, pathway, modifier="per_cbm_mile", include_stowage=True
        )

#
#        for vessel_type in VESSEL_TYPES:
#            plot_histogram_for_vessel_classes(
#                vessel_type, fuel, pathway, modifier="per_tonne_mile"
#            )
#            plot_histogram_for_vessel_classes(
#                vessel_type, fuel, pathway, modifier="per_cbm_mile"
#            )
#            plot_histogram_for_vessel_classes(
#                vessel_type,
#                fuel,
#                pathway,
#                modifier="per_tonne_mile",
#                include_stowage=True,
#            )
#            plot_histogram_for_vessel_classes(
#                vessel_type,
#                fuel,
#                pathway,
#                modifier="per_cbm_mile",
#                include_stowage=True,
#            )
    

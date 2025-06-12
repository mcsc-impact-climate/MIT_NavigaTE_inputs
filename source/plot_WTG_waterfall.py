"""
Date: June 11, 2025
Author: danikae
Purpose: Makes waterfall plots of WTG cost and emissions components.
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from common_tools import get_top_dir, ensure_directory_exists

top_dir = get_top_dir()


def plot_waterfall(ax, components, title, ylabel, total_label='Total'):
    labels = list(components.keys())
    values = list(components.values())

    cum_values = [0]
    for v in values[:-1]:
        cum_values.append(cum_values[-1] + v)

    fig_values = values + [sum(values)]
    fig_labels = labels + [total_label]
    fig_cum = cum_values + [0]  # last bar starts at 0

    colors = ['green' if v >= 0 else 'red' for v in values] + ['blue']

    for i, (label, val, base, color) in enumerate(zip(fig_labels, fig_values, fig_cum, colors)):
        ax.bar(label, val, bottom=base if i < len(values) else 0, color=color)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color='black', linewidth=0.8)
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')
        label.set_rotation_mode('anchor')


def plot_fuel_waterfalls(country, fuel, pathway):
    filename = f"{fuel}_{pathway}_{country}_components.json"
    filepath = os.path.join(
        top_dir, "input_fuel_pathway_data", "production", "cost_emissions_components", filename
    )
    if not os.path.exists(filepath):
        print(f"Skipping missing file: {filename}")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)

    cost_data = {**data.get("CapEx (2024$/tonne)", {}), **data.get("OpEx (2024$/tonne)", {})}
    cost_data = {k: v for k, v in cost_data.items() if v != 0}

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plot_waterfall(
        ax1, cost_data,
        title=f"Cost Breakdown for {fuel} via {pathway} in {country.replace('_', ' ').title()}",
        ylabel="2024 $/tonne"
    )
    plt.tight_layout()
    plt.savefig(f"plots/WTG_waterfall/{fuel}_{pathway}_{country}_costs.png", dpi=300)
    plt.savefig(f"plots/WTG_waterfall/{fuel}_{pathway}_{country}_costs.pdf")
    plt.close(fig1)

    emissions_data = data.get("emissions (kg CO2e / kg fuel)", {})
    emissions_data = {k: v for k, v in emissions_data.items() if v != 0}

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plot_waterfall(
        ax2, emissions_data,
        title=f"Emissions Breakdown for {fuel} via {pathway} in {country.replace('_', ' ').title()}",
        ylabel="kg CO2e / kg fuel"
    )
    plt.tight_layout()
    plt.savefig(f"plots/WTG_waterfall/{fuel}_{pathway}_{country}_emissions.png", dpi=300)
    plt.savefig(f"plots/WTG_waterfall/{fuel}_{pathway}_{country}_emissions.pdf")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Generate waterfall plots for all fuels/pathways for a given country.")
    parser.add_argument("--country", required=True, help="Country name in lowercase with underscores (e.g., singapore)")

    args = parser.parse_args()
    country = args.country
    folder = os.path.join(top_dir, "input_fuel_pathway_data", "production", "cost_emissions_components")

    ensure_directory_exists("plots/WTG_waterfall")

    for filename in os.listdir(folder):
        if filename.endswith(".json") and filename.endswith(f"{country}_components.json"):
            try:
                parts = filename.replace(f"_{country}_components.json", "").split("_", 1)
                fuel, pathway = parts[0], parts[1]
                plot_fuel_waterfalls(country, fuel, pathway)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()

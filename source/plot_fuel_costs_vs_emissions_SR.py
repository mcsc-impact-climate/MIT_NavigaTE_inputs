"""
Date: 251227
Author: danikae
Purpose: Plot the total costs and emissions of fuel production + storage + transport for each fuel and production pathway
"""

import matplotlib.pyplot as plt
import pandas as pd
from common_tools import get_top_dir


def plot_fuel_costs_vs_emissions():
    """Plot the total costs and emissions of fuel production + storage + transport for each fuel and production pathway."""
    ports = ["Singapore", "Rotterdam"]
    
    fuels = [
        "FTdiesel",
        "lng",
        "ammonia",
        "compressed_hydrogen",
        "liquid_hydrogen",
        "methanol",
        "bio_cfp",
        #        "bio_leo"
    ]
    
    top_dir = get_top_dir()
    
    for port in ports:
        for fuel in fuels:
            costs_emissions_path = f"{top_dir}/input_fuel_pathway_data/final_fuel/{port}/{fuel}_{port}_costs_emissions.csv"
            fuel_costs_emissions_df = pd.read_csv(costs_emissions_path)
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get unique pathway names for coloring
            pathways = fuel_costs_emissions_df["Pathway Name"].unique()
            colors = plt.cm.tab20(range(len(pathways)))
            pathway_colors = {pathway: colors[i] for i, pathway in enumerate(pathways)}
            
            # Plot each pathway as a separate series
            for pathway in pathways:
                pathway_data = fuel_costs_emissions_df[fuel_costs_emissions_df["Pathway Name"] == pathway]
                ax.scatter(
                    pathway_data["Final LCOF [$/tonne]"],
                    pathway_data["Final Emissions [kg CO2e / kg fuel]"],
                    label=pathway,
                    color=pathway_colors[pathway],
                    s=100,
                    alpha=0.7,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            # Labels and title
            ax.set_xlabel("LCOF [$/tonne]", fontsize=12, fontweight='bold')
            ax.set_ylabel("Emissions [kg CO2e / kg fuel]", fontsize=12, fontweight='bold')
            ax.set_title(f"{fuel.upper()} - Cost vs Emissions ({port})", fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Save figure
            output_dir = f"{top_dir}/plots"
            output_path = f"{output_dir}/{fuel}_{port}_costs_vs_emissions.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {output_path}")
            plt.close()


def main():
    """Main routine."""
    plot_fuel_costs_vs_emissions()


if __name__ == "__main__":
    main()
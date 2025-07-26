"""
Date: Jan. 21, 2025
Author: danikam
Purpose: Reads in vessel ranges and plots them as a histogram
"""

import matplotlib.pyplot as plt
import pandas as pd
from common_tools import get_top_dir

top_dir = get_top_dir()

# Read the CSV file
file_path = f"{top_dir}/tables/vessel_info.csv"
df = pd.read_csv(file_path)

# Extract the relevant columns
vessels = df["Vessel"]
ranges = df["Nominal Range (nautical miles)"]

# Plot horizontal bars
plt.figure(figsize=(10, 6))
plt.barh(vessels, ranges, color="skyblue", edgecolor="black")

# Add labels and title
plt.xlabel("$R$ (nautical miles)", fontsize=24)
plt.tick_params(axis="y", labelsize=22)
plt.tick_params(axis="x", labelsize=22)
plt.tight_layout()

# Show the plot
plt.savefig(f"{top_dir}/plots/vessel_ranges.png", dpi=300)
plt.savefig(f"{top_dir}/plots/vessel_ranges.pdf")
print(f"Plot saved to {top_dir}/plots/vessel_ranges.pdf")
